"""Stage channels: ``send_hidden`` / ``recv_hidden`` over three backends.

Every backend moves the same thing, a small JSON header plus one tensor, and is
one-way (A pushes to B). Pipelines and ping-pong benchmarks are built from pairs
of one-way links, so the three backends are compared on identical traffic.

* ``tcp``  - length-prefixed frames on a raw socket (TCP_NODELAY), zero protobuf.
* ``grpc`` - a client-streaming gRPC call carrying ``google.protobuf.BytesValue``
             messages, i.e. a real protobuf encode/decode per hop.
* ``gloo`` - a pairwise ``torch.distributed`` Gloo process group per link,
             bootstrapped over a tiny TCP rendezvous so links can be created at
             runtime (nodes join dynamically; no global world size).

``DelayedSender`` wraps any sender with emulated one-way delay/jitter/rate, our
``tc netem`` substitute on Windows/macOS hosts and in single-host runs.
"""
from __future__ import annotations

import json
import queue
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from .timing import precise_sleep

OnMessage = Callable[[dict, Optional[torch.Tensor]], None]

_DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
       "int64": torch.int64, "int32": torch.int32, "uint8": torch.uint8}


# --------------------------------------------------------------------------- framing
def pack(header: dict, tensor: Optional[torch.Tensor]) -> Tuple[bytes, memoryview]:
    """Serialise header + tensor. Returns (header_bytes, payload_view)."""
    if tensor is not None:
        t = tensor.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous()
        header = dict(header)
        header["_dt"] = str(t.dtype).replace("torch.", "")
        header["_sh"] = list(t.shape)
        payload = memoryview(t.view(torch.uint8).numpy()).cast("B") if t.numel() else memoryview(b"")
    else:
        payload = memoryview(b"")
    return json.dumps(header, separators=(",", ":")).encode(), payload


def unpack_tensor(header: dict, buf) -> Optional[torch.Tensor]:
    if "_dt" not in header:
        return None
    dt = _DT[header["_dt"]]
    shape = header["_sh"]
    if len(buf) == 0:
        return torch.empty(shape, dtype=dt)
    return torch.frombuffer(buf, dtype=torch.uint8).view(dt).view(shape)


def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("peer closed")
        got += r
    return buf


@dataclass
class LinkStats:
    msgs: int = 0
    bytes: int = 0
    ser_ms: float = 0.0
    send_ms: float = 0.0


class Sender:
    stats: LinkStats

    def send(self, header: dict, tensor: Optional[torch.Tensor] = None) -> float:
        """Send one message; returns serialisation time in ms."""
        raise NotImplementedError

    def close(self):
        pass


class Server:
    address: str

    def close(self):
        pass


# --------------------------------------------------------------------------- raw TCP
_FRAME = struct.Struct("!IQ")


class TcpSender(Sender):
    def __init__(self, address: str):
        host, port = address.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lock = threading.Lock()
        self.stats = LinkStats()

    def send(self, header, tensor=None):
        t0 = time.perf_counter()
        hb, payload = pack(header, tensor)
        t1 = time.perf_counter()
        with self.lock:
            self.sock.sendall(_FRAME.pack(len(hb), len(payload)) + hb)
            if len(payload):
                self.sock.sendall(payload)
        t2 = time.perf_counter()
        s = self.stats
        s.msgs += 1
        s.bytes += len(hb) + len(payload)
        s.ser_ms += (t1 - t0) * 1e3
        s.send_ms += (t2 - t1) * 1e3
        return (t1 - t0) * 1e3

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class TcpServer(Server):
    def __init__(self, host: str, port: int, on_message: OnMessage):
        self.on_message = on_message
        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((host, port))
        self.lsock.listen(64)
        self.port = self.lsock.getsockname()[1]
        self.address = f"{_advertise(host)}:{self.port}"
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._closed:
            try:
                c, _ = self.lsock.accept()
            except OSError:
                return
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c: socket.socket):
        try:
            while True:
                hl, pl = _FRAME.unpack(_recv_exact(c, _FRAME.size))
                t_arrive = time.perf_counter()
                header = json.loads(_recv_exact(c, hl))
                buf = _recv_exact(c, pl) if pl else b""
                t0 = time.perf_counter()
                tensor = unpack_tensor(header, buf)
                header["_deser_ms"] = (time.perf_counter() - t0) * 1e3
                header["_recv_ms"] = (t0 - t_arrive) * 1e3
                self.on_message(header, tensor)
        except (ConnectionError, OSError) as e:
            import logging

            logging.getLogger("lanshard.transport").info("tcp link closed: %r", e)
        except Exception:
            import logging

            logging.getLogger("lanshard.transport").exception("tcp receive failed")
        finally:
            c.close()

    def close(self):
        self._closed = True
        try:
            self.lsock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- gRPC
_GRPC_METHOD = "/lanshard.Stage/Push"
_GRPC_OPTS = [("grpc.max_send_message_length", 512 << 20), ("grpc.max_receive_message_length", 512 << 20)]


def _grpc_encode(hb: bytes, payload) -> bytes:
    from google.protobuf.wrappers_pb2 import BytesValue

    return BytesValue(value=struct.pack("!I", len(hb)) + hb + bytes(payload)).SerializeToString()


class GrpcSender(Sender):
    def __init__(self, address: str):
        import grpc

        self.channel = grpc.insecure_channel(address, options=_GRPC_OPTS)
        grpc.channel_ready_future(self.channel).result(timeout=30)
        call = self.channel.stream_unary(_GRPC_METHOD, request_serializer=lambda b: b,
                                         response_deserializer=lambda b: b)
        self.q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self.future = call.future(iter(self.q.get, None))
        self.stats = LinkStats()

    def send(self, header, tensor=None):
        t0 = time.perf_counter()
        hb, payload = pack(header, tensor)
        msg = _grpc_encode(hb, payload)  # protobuf encode (+ copy of the payload)
        t1 = time.perf_counter()
        self.q.put(msg)
        s = self.stats
        s.msgs += 1
        s.bytes += len(msg)
        s.ser_ms += (t1 - t0) * 1e3
        return (t1 - t0) * 1e3

    def close(self):
        self.q.put(None)
        try:
            self.channel.close()
        except Exception:
            pass


class GrpcServer(Server):
    def __init__(self, host: str, port: int, on_message: OnMessage):
        import grpc
        from concurrent import futures
        from google.protobuf.wrappers_pb2 import BytesValue

        def push(request_iterator, context):
            for raw in request_iterator:
                t0 = time.perf_counter()
                val = BytesValue.FromString(raw).value  # protobuf decode
                hl = struct.unpack_from("!I", val)[0]
                header = json.loads(val[4:4 + hl])
                tensor = unpack_tensor(header, bytearray(val[4 + hl:]))
                header["_deser_ms"] = (time.perf_counter() - t0) * 1e3
                on_message(header, tensor)
            return b""

        handler = grpc.method_handlers_generic_handler("lanshard.Stage", {
            "Push": grpc.stream_unary_rpc_method_handler(push, request_deserializer=lambda b: b,
                                                         response_serializer=lambda b: b)})
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=32), options=_GRPC_OPTS)
        self.server.add_generic_rpc_handlers((handler,))
        self.port = self.server.add_insecure_port(f"{host}:{port}")
        self.server.start()
        self.address = f"{_advertise(host)}:{self.port}"

    def close(self):
        self.server.stop(0)


# --------------------------------------------------------------------------- Gloo
def _gloo_pg(host: str, port: int, rank: int, local_ip: Optional[str] = None):
    """Two-rank Gloo group. ``local_ip`` pins Gloo's network device to the address this link
    actually uses; by default Gloo resolves the machine's hostname, which on some hosts (CI
    runners, multi-homed laptops) points at an unreachable interface and hangs the handshake."""
    import torch.distributed as dist
    from datetime import timedelta

    store = dist.TCPStore(host, port, 2, rank == 0, timeout=timedelta(seconds=60), use_libuv=False)
    timeout = timedelta(seconds=600)
    if local_ip:
        try:
            opts = dist.ProcessGroupGloo._Options()
            opts._timeout = timeout
            opts._devices = [dist.ProcessGroupGloo.create_device(hostname=local_ip)]
            return store, dist.ProcessGroupGloo(store, rank, 2, opts)
        except (AttributeError, TypeError, RuntimeError):
            pass  # older torch: fall back to the default device
    return store, dist.ProcessGroupGloo(store, rank, 2, timeout)


class GlooSender(Sender):
    """Rank 1 of a two-rank Gloo group; the receiving server is rank 0."""

    def __init__(self, address: str):
        host, port = address.rsplit(":", 1)
        s = socket.create_connection((host, int(port)))
        store_port = struct.unpack("!I", _recv_exact(s, 4))[0]
        local_ip = s.getsockname()[0]
        s.close()
        self.store, self.pg = _gloo_pg(host, store_port, 1, local_ip)
        self.lock = threading.Lock()
        self.stats = LinkStats()

    def send(self, header, tensor=None):
        t0 = time.perf_counter()
        hb, payload = pack(header, tensor)
        body = torch.empty(len(hb) + len(payload), dtype=torch.uint8)
        body[:len(hb)] = torch.frombuffer(bytearray(hb), dtype=torch.uint8)
        if len(payload):
            body[len(hb):] = torch.frombuffer(payload, dtype=torch.uint8)
        lens = torch.tensor([len(hb), len(payload)], dtype=torch.int64)
        t1 = time.perf_counter()
        with self.lock:
            self.pg.send([lens], 0, 0).wait()
            self.pg.send([body], 0, 1).wait()
        s = self.stats
        s.msgs += 1
        s.bytes += body.numel()
        s.ser_ms += (t1 - t0) * 1e3
        s.send_ms += (time.perf_counter() - t1) * 1e3
        return (t1 - t0) * 1e3


class GlooServer(Server):
    def __init__(self, host: str, port: int, on_message: OnMessage):
        self.on_message = on_message
        self.bind_host = host
        self.lsock = socket.socket()
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((host, port))
        self.lsock.listen(64)
        self.port = self.lsock.getsockname()[1]
        self.address = f"{_advertise(host)}:{self.port}"
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                c, _ = self.lsock.accept()
            except OSError:
                return
            sp = free_port()
            local_ip = c.getsockname()[0]
            # the store master must exist before the peer connects to it
            threading.Thread(target=self._pair, args=(sp, local_ip), daemon=True).start()
            time.sleep(0.05)
            c.sendall(struct.pack("!I", sp))
            c.close()

    def _pair(self, store_port: int, local_ip: Optional[str] = None):
        store, pg = _gloo_pg("0.0.0.0" if self.bind_host in ("", "0.0.0.0") else self.bind_host, store_port, 0,
                             local_ip)
        lens = torch.empty(2, dtype=torch.int64)
        try:
            while True:
                pg.recv([lens], 1, 0).wait()
                t_arrive = time.perf_counter()
                hl, pl = int(lens[0]), int(lens[1])
                body = torch.empty(hl + pl, dtype=torch.uint8)
                pg.recv([body], 1, 1).wait()
                t0 = time.perf_counter()
                raw = body.numpy()
                header = json.loads(raw[:hl].tobytes())
                tensor = unpack_tensor(header, bytearray(raw[hl:].tobytes())) if pl or "_dt" in header else None
                header["_deser_ms"] = (time.perf_counter() - t0) * 1e3
                header["_recv_ms"] = (t0 - t_arrive) * 1e3
                self.on_message(header, tensor)
        except Exception:
            return

    def close(self):
        try:
            self.lsock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- netem
@dataclass
class NetemProfile:
    """One-way delay model: base + |N(0, jitter)| + occasional retransmission spike + size/rate.

    ``samples_ms`` (if given) replaces the parametric part with an empirical
    distribution measured on the real network (see ``bench/measure_rtt.py``).
    """

    base_ms: float = 0.0
    jitter_ms: float = 0.0
    spike_prob: float = 0.0
    spike_ms: float = 0.0
    rate_mbps: float = 0.0
    samples_ms: List[float] = field(default_factory=list)
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    @property
    def active(self) -> bool:
        return bool(self.base_ms or self.jitter_ms or self.spike_prob or self.rate_mbps or self.samples_ms)

    def latency_s(self) -> float:
        """Propagation part of the delay: base + jitter + occasional spike."""
        r = self._rng
        if self.samples_ms:
            d = r.choice(self.samples_ms)
        else:
            d = self.base_ms + abs(r.gauss(0.0, self.jitter_ms)) if self.jitter_ms else self.base_ms
            if self.spike_prob and r.random() < self.spike_prob:
                d += self.spike_ms
        return d / 1e3

    def transfer_s(self, nbytes: int) -> float:
        """Time the link is busy sending ``nbytes`` at ``rate_mbps``."""
        return nbytes * 8 / (self.rate_mbps * 1e6) if self.rate_mbps else 0.0

    def delay_s(self, nbytes: int) -> float:
        """One-way delay of a single message on an idle link."""
        return self.latency_s() + self.transfer_s(nbytes)

    @classmethod
    def from_spec(cls, spec: Optional[str]) -> "NetemProfile":
        """'wifi' | 'none' | path/to/profile.json | 'base=3,jitter=4,spike_prob=0.01,spike=40,rate=400'."""
        if not spec or spec == "none":
            return cls()
        if spec == "wifi":  # defaults from the proposal: 2-20 ms RTT with spikes, ~400 Mbps
            return cls(base_ms=1.5, jitter_ms=2.5, spike_prob=0.01, spike_ms=30.0, rate_mbps=400.0)
        if spec.endswith(".json"):
            with open(spec) as f:
                d = json.load(f)
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        kw = {}
        alias = {"base": "base_ms", "jitter": "jitter_ms", "spike": "spike_ms", "rate": "rate_mbps"}
        for part in spec.split(","):
            k, v = part.split("=")
            kw[alias.get(k, k)] = float(v)
        return cls(**kw)


class DelayedSender(Sender):
    """Applies a NetemProfile to an inner sender while preserving FIFO order.

    The link sends one message at a time, as ``tc netem rate`` does: a message starts when the
    previous one has left, takes size / rate to send, then the latency. A large prefill chunk
    therefore holds back the decode tokens queued behind it on the same link.
    """

    def __init__(self, inner: Sender, profile: NetemProfile):
        self.inner, self.profile = inner, profile
        self.stats = inner.stats
        self.q: "queue.Queue" = queue.Queue()
        self._last_due = 0.0
        self._link_free = 0.0  # when the previous message finishes sending
        self.enqueued = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def send(self, header, tensor=None):
        self.enqueued += 1
        nbytes = (tensor.numel() * tensor.element_size()) if tensor is not None else 0
        self._link_free = max(time.perf_counter(), self._link_free) + self.profile.transfer_s(nbytes)
        due = max(self._link_free + self.profile.latency_s(), self._last_due)
        self._last_due = due
        if tensor is not None and tensor.device.type != "cpu":
            tensor = tensor.cpu()
        self.q.put((due, header, tensor))
        return 0.0

    def _pump(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            due, header, tensor = item
            precise_sleep(due - time.perf_counter())
            try:
                self.inner.send(header, tensor)
            except Exception:
                import logging

                logging.getLogger("lanshard.transport").exception("delayed send failed (mb %s)", header.get("mb"))

    def close(self):
        self.q.put(None)
        self.inner.close()


class AsyncSender(Sender):
    """Moves serialisation + socket writes off the compute thread."""

    def __init__(self, inner: Sender):
        self.inner = inner
        self.stats = inner.stats
        self.q: "queue.Queue" = queue.Queue()
        self.enqueued = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def send(self, header, tensor=None):
        self.enqueued += 1
        self.q.put((header, tensor))
        return 0.0

    def _pump(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                self.inner.send(*item)
            except Exception:
                import logging

                logging.getLogger("lanshard.transport").exception("async send failed (mb %s)", item[0].get("mb"))

    def close(self):
        self.q.put(None)
        self.inner.close()


# --------------------------------------------------------------------------- factory
BACKENDS = ("tcp", "grpc", "gloo")


def _advertise(host: str) -> str:
    if host in ("", "0.0.0.0"):
        return local_ip()
    return host


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def serve(backend: str, host: str, port: int, on_message: OnMessage) -> Server:
    return {"tcp": TcpServer, "grpc": GrpcServer, "gloo": GlooServer}[backend](host, port, on_message)


def connect(backend: str, address: str, netem: Optional[NetemProfile] = None, asynchronous: bool = True) -> Sender:
    s: Sender = {"tcp": TcpSender, "grpc": GrpcSender, "gloo": GlooSender}[backend](address)
    if netem is not None and netem.active:
        return DelayedSender(s, netem)
    return AsyncSender(s) if asynchronous else s


# --------------------------------------------------------------------------- control plane
class JsonConn:
    """Newline-delimited JSON over TCP for the control plane (join, plans, metrics)."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.rfile = sock.makefile("rb")
        self.lock = threading.Lock()

    @classmethod
    def connect(cls, address: str, retries: int = 60) -> "JsonConn":
        host, port = address.rsplit(":", 1)
        for i in range(retries):
            try:
                return cls(socket.create_connection((host, int(port))))
            except OSError:
                if i == retries - 1:
                    raise
                time.sleep(0.5)
        raise RuntimeError("unreachable")

    def send(self, obj: dict):
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
        with self.lock:
            self.sock.sendall(data)

    def recv(self) -> Optional[dict]:
        line = self.rfile.readline()
        if not line:
            return None
        return json.loads(line)

    def shutdown(self):
        """Unblock a reader waiting in ``recv`` (closing the socket alone does not)."""
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
