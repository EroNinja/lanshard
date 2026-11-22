"""LANShard worker: owns a contiguous layer range, its KV cache, and one data endpoint.

Lifecycle: join (send device info) -> profile one layer on request -> measure
RTT to peers on request -> serve micro-batches for whatever plans the
coordinator installs. Plans are versioned; a worker may hold layers for two
plans at once during a drain-then-switch transition (I3), and a message always
names the plan version it was admitted under.
"""
from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import queue
import socket
import statistics
import struct
import sys
import threading
import time
import uuid
from collections import deque
from typing import Dict, Optional

import psutil
import torch

from . import transport as T
from .model_utils import LayerLoader, Segment, StageRunner, default_device, default_dtype, parse_dtype
from .partitioner import COORD, Plan
from .timing import precise_sleep
from .probe import Slowdown, device_info, expected_layer_ms, profile_layer, sync

log = logging.getLogger("lanshard.worker")


# --------------------------------------------------------------------------- echo (RTT probe)
class EchoServer:
    """Length-prefixed echo used to measure per-hop RTT with a decode-sized payload."""

    def __init__(self, host: str, port: int = 0):
        self.lsock = socket.socket()
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((host, port))
        self.lsock.listen(16)
        self.port = self.lsock.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                c, _ = self.lsock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    @staticmethod
    def _serve(c):
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                n = struct.unpack("!I", T._recv_exact(c, 4))[0]
                data = T._recv_exact(c, n)
                c.sendall(struct.pack("!I", n) + data)
        except (ConnectionError, OSError):
            c.close()


def measure_rtt(address: str, payload: int = 8192, n: int = 20, netem: Optional[T.NetemProfile] = None) -> Dict:
    host, port = address.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=10)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    data = os.urandom(payload)
    samples = []
    for i in range(n + 3):
        t0 = time.perf_counter()
        if netem is not None and netem.active:  # emulate the link both ways
            precise_sleep(netem.delay_s(payload) + netem.delay_s(payload))
        s.sendall(struct.pack("!I", payload) + data)
        T._recv_exact(s, 4 + payload)
        if i >= 3:
            samples.append((time.perf_counter() - t0) * 1e3)
    s.close()
    samples.sort()
    return {"median": samples[len(samples) // 2], "p10": samples[len(samples) // 10],
            "p90": samples[int(len(samples) * 0.9)], "samples": samples}


# --------------------------------------------------------------------------- worker
class Worker:
    def __init__(self, coordinator: str, node_id: Optional[str] = None, device: Optional[str] = None,
                 dtype: Optional[str] = None, transport: str = "tcp", host: str = "0.0.0.0",
                 advertise: Optional[str] = None, slowdown: float = 1.0, mem_limit_gb: Optional[float] = None,
                 netem: Optional[str] = None, profile_budget_s: float = 15.0, wire_dtype: str = "float16"):
        self.coordinator = coordinator
        self.node_id = node_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:4]}"
        self.device = device or default_device()
        if self.device.startswith("cpu") and os.environ.get("LANSHARD_THREADS") is None:
            # leave cores for co-located processes (the coordinator, GPU workers' launch threads)
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        elif os.environ.get("LANSHARD_THREADS"):
            torch.set_num_threads(int(os.environ["LANSHARD_THREADS"]))
        self.dtype = parse_dtype(dtype) if dtype else default_dtype(self.device)
        self.wire_dtype = parse_dtype(wire_dtype)
        self.transport = transport
        self.host = host
        self.advertise = advertise
        self.slowdown = Slowdown(slowdown)
        self.mem_limit_gb = mem_limit_gb
        self.netem = T.NetemProfile.from_spec(netem)
        self.profile_budget_s = profile_budget_s
        self.overhead = 0

        self.loader: Optional[LayerLoader] = None
        self.runner: Optional[StageRunner] = None
        self.profile: Optional[Dict] = None
        self.plans: Dict[int, Plan] = {}
        self.addrs: Dict[str, str] = {}
        self.senders: Dict[str, T.Sender] = {}
        self.inbox: "queue.Queue" = queue.Queue()
        self.layer_lock = threading.Lock()

        # stats
        self.busy_s = 0.0
        self.stats_t0 = time.perf_counter()
        self.ratios: deque = deque()   # (t, observed/expected)
        self.n_msgs = 0
        self.n_recv = 0
        self.last_mb = None
        self.stop = threading.Event()

    # ---------------------------------------------------------------- setup
    def _addr(self, port: int) -> str:
        host = self.advertise or (T.local_ip() if self.host in ("", "0.0.0.0") else self.host)
        return f"{host}:{port}"

    def run(self):
        self.data_server = T.serve(self.transport, self.host, 0, self._on_data)
        self.echo = EchoServer(self.host)
        self.ctrl = T.JsonConn.connect(self.coordinator)
        info = device_info(self.device, self.mem_limit_gb)
        info["slowdown"] = self.slowdown.factor
        self.ctrl.send({"type": "join", "node_id": self.node_id, "info": info,
                        "data_addr": self._addr(self.data_server.port), "echo_addr": self._addr(self.echo.port)})
        log.info("joined %s as %s (%s, %s)", self.coordinator, self.node_id, info["accel"], self.device)
        threading.Thread(target=self._compute_loop, daemon=True).start()
        threading.Thread(target=self._stats_loop, daemon=True).start()
        while not self.stop.is_set():
            msg = self.ctrl.recv()
            if msg is None:
                log.warning("coordinator went away")
                break
            try:
                self._handle(msg)
            except Exception as e:  # report and keep serving
                log.exception("control message failed")
                self.ctrl.send({"type": "error", "node_id": self.node_id, "error": repr(e), "for": msg.get("type")})
        self.stop.set()

    def _handle(self, msg: Dict):
        t = msg["type"]
        if t == "profile":
            threading.Thread(target=self._do_profile, args=(msg,), daemon=True).start()
        elif t == "ping_peers":
            threading.Thread(target=self._do_ping, args=(msg,), daemon=True).start()
        elif t == "prepare":
            threading.Thread(target=self._do_prepare, args=(msg,), daemon=True).start()
        elif t == "release":
            self._release(int(msg["version"]))
        elif t == "free":
            self.inbox.put(("free", msg["rids"]))
        elif t == "set_slowdown":
            self.slowdown.factor = float(msg["factor"])
            log.info("slowdown set to %.2f", self.slowdown.factor)
        elif t == "dump":
            import faulthandler
            import sys

            log.error("thread dump requested by coordinator (inbox=%d, last_mb=%s)", self.inbox.qsize(), self.last_mb)
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
            sys.stdout.flush()
        elif t == "reset_stats":
            self.busy_s, self.stats_t0, self.n_msgs = 0.0, time.perf_counter(), 0
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
        elif t == "shutdown":
            self.stop.set()
            os._exit(0)

    def _do_profile(self, msg):
        model = msg["model"]
        log.info("profiling one layer of %s for %.0fs", model, msg.get("budget_s", self.profile_budget_s))
        self.loader = LayerLoader(model)
        self.runner = StageRunner(self.loader.cfg, self.device, self.dtype)
        self.profile = profile_layer(model, self.device, self.dtype, budget_s=msg.get("budget_s", self.profile_budget_s),
                                     slowdown=self.slowdown)
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        info = device_info(self.device, self.mem_limit_gb)
        self.ctrl.send({"type": "profile_result", "node_id": self.node_id, "profile": self.profile,
                        "mem_free": info["mem_free"]})

    def _do_ping(self, msg):
        payload = int(msg.get("payload", 8192))
        out = {}
        for nid, addr in msg["peers"].items():
            if nid == self.node_id:
                continue
            try:
                out[nid] = measure_rtt(addr, payload, int(msg.get("n", 20)), self.netem)["median"]
            except OSError as e:
                out[nid] = None
                log.warning("ping %s failed: %s", nid, e)
        self.ctrl.send({"type": "rtt", "node_id": self.node_id, "rtt": out})

    def _do_prepare(self, msg):
        plan = Plan.from_dict(msg["plan"])
        self.addrs.update(msg["addrs"])
        self.overhead = int(msg.get("overhead", 0))
        need = plan.layers_of(self.node_id)
        try:
            with self.layer_lock:
                missing = [i for i in need if i not in self.runner.layers]
                # honour the advertised capacity: old + new layers must fit together while
                # the old plan drains, otherwise ask for the pause-drain-load fallback
                cap = self._capacity_layers(int(msg.get("kv_tokens", 0)))
                if cap is not None and len(self.runner.layers) + len(missing) > cap:
                    self.ctrl.send({"type": "prepare_failed", "node_id": self.node_id, "version": plan.version,
                                    "error": f"capacity: {len(self.runner.layers)}+{len(missing)} > {cap} layers"})
                    return
                t0 = time.perf_counter()
                if missing:
                    self.runner.layers.update(self.loader.load_layers(missing, self.device, self.dtype))
                load_s = time.perf_counter() - t0
                # registered under the lock: a release of an older plan that arrives while this
                # one is being staged must count these layers as in use
                self.plans[plan.version] = plan
            # pre-connect to the successor so the first token does not pay connection setup
            for i, st in enumerate(plan.stages):
                if st.node_id == self.node_id:
                    nxt = plan.stages[i + 1].node_id if i + 1 < len(plan.stages) else COORD
                    self._sender(nxt)
            log.info("plan v%d ready: layers %s (loaded %d in %.1fs)", plan.version,
                     f"{need[0]}-{need[-1]}" if need else "none", len(missing), load_s)
            self.ctrl.send({"type": "ready", "node_id": self.node_id, "version": plan.version,
                            "loaded": len(missing), "load_s": load_s, "mem": self._mem()})
        except torch.OutOfMemoryError as e:
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            self.ctrl.send({"type": "prepare_failed", "node_id": self.node_id, "version": plan.version,
                            "error": "oom"})

    def _capacity_layers(self, kv_tokens: int) -> Optional[int]:
        if self.mem_limit_gb is None or not self.profile:
            return None
        per = self.profile["mem_layer"] + self.profile["kv_per_tok_layer"] * kv_tokens
        budget = self.mem_limit_gb * (1 << 30) - self.overhead
        return max(0, int(budget // per))

    def _release(self, version: int):
        with self.layer_lock:
            self.plans.pop(version, None)
            keep = set()
            for p in self.plans.values():
                keep.update(p.layers_of(self.node_id))
            for li in [li for li in self.runner.layers if li not in keep]:
                del self.runner.layers[li]
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        log.info("released v%d; holding %d layers", version, len(self.runner.layers))

    def _sender(self, node_id: str) -> T.Sender:
        s = self.senders.get(node_id)
        if s is None:
            s = T.connect(self.transport, self.addrs[node_id], self.netem)
            self.senders[node_id] = s
        return s

    # ---------------------------------------------------------------- data path
    def _on_data(self, header: Dict, tensor):
        self.n_recv += 1
        header["_arrive"] = time.perf_counter()
        self.inbox.put(("mb", header, tensor))

    def _compute_loop(self):
        while True:
            item = self.inbox.get()
            if item[0] == "free":
                for rid in item[1]:
                    self.runner.kv.free(rid)
                continue
            _, h, x = item
            # native watchdog: dumps all stacks even if the interpreter is wedged in C code
            faulthandler.dump_traceback_later(60, repeat=False, file=sys.stdout)
            try:
                self._process(h, x)
            except Exception:
                log.exception("micro-batch %s failed", h.get("mb"))
            finally:
                faulthandler.cancel_dump_traceback_later()

    def _process(self, h: Dict, x: torch.Tensor):
        self.last_mb = h.get("mb")
        plan = self.plans[h["v"]]
        st_idx = h["st"]
        stage = plan.stages[st_idx]
        layer_ids = list(range(stage.start, stage.end))
        segs = [Segment.from_list(s) for s in h["seg"]]
        q_ms = (time.perf_counter() - h.pop("_arrive")) * 1e3
        sync(self.device)
        t0 = time.perf_counter()
        y = self.runner.forward(x, segs, layer_ids)
        y = y.to(self.wire_dtype)
        sync(self.device)
        dt = time.perf_counter() - t0
        self.slowdown.apply(dt)
        compute = time.perf_counter() - t0
        self.busy_s += compute
        self.n_msgs += 1
        n_dec = sum(1 for s in segs if s.n == 1)
        n_pre = sum(s.n for s in segs if s.n > 1)
        if self.profile and n_pre == 0:  # drift is tracked on decode micro-batches
            exp = expected_layer_ms(self.profile, n_dec, 0) * len(layer_ids)
            if exp > 0:
                self.ratios.append((time.time(), compute * 1e3 / exp))
        trace = h.get("tr", [])
        trace.append([self.node_id, round(compute * 1e3, 3), round(h.pop("_deser_ms", 0.0), 3), round(q_ms, 3)])
        h["tr"] = trace
        h.pop("_recv_ms", None)
        h.pop("_dt", None)
        h.pop("_sh", None)
        nxt = plan.stages[st_idx + 1].node_id if st_idx + 1 < len(plan.stages) else COORD
        h["st"] = st_idx + 1
        self._sender(nxt).send(h, y.cpu())

    # ---------------------------------------------------------------- stats
    def _mem(self) -> Dict:
        d = {"rss": psutil.Process().memory_info().rss}
        if self.device.startswith("cuda"):
            d["peak_alloc"] = torch.cuda.max_memory_allocated()
            d["alloc"] = torch.cuda.memory_allocated()
        if self.runner is not None:
            d["kv"] = self.runner.kv.bytes_used()
        return d

    def _stats_loop(self, period: float = 1.0, window_s: float = 10.0):
        while not self.stop.is_set():
            time.sleep(period)
            now = time.time()
            while self.ratios and self.ratios[0][0] < now - window_s:
                self.ratios.popleft()
            med = statistics.median(r for _, r in self.ratios) if len(self.ratios) >= 5 else None
            wall = time.perf_counter() - self.stats_t0
            try:
                self.ctrl.send({"type": "stats", "node_id": self.node_id, "t": now, "ratio": med,
                                "n_ratio": len(self.ratios), "busy_s": self.busy_s, "wall_s": wall,
                                "msgs": self.n_msgs, "recv": self.n_recv, "inbox": self.inbox.qsize(),
                                "last_mb": self.last_mb,
                                "links": {k: {"enq": getattr(v, "enqueued", None), "sent": v.stats.msgs,
                                              "pending": v.q.qsize() if hasattr(v, "q") else 0}
                                          for k, v in list(self.senders.items())},
                                "slowdown": self.slowdown.factor,
                                "layers": sorted(self.runner.layers) if self.runner else [],
                                "kv_reqs": len(self.runner.kv) if self.runner else 0, "mem": self._mem()})
            except OSError:
                return
            except Exception:  # never let a stats hiccup kill reporting
                log.exception("stats loop")


def main(argv=None):
    ap = argparse.ArgumentParser(description="LANShard worker")
    ap.add_argument("--coordinator", required=True, help="host:port of the coordinator control endpoint")
    ap.add_argument("--node-id")
    ap.add_argument("--device", help="cuda | cuda:1 | mps | cpu (default: best available)")
    ap.add_argument("--dtype", help="float16 | bfloat16 | float32")
    ap.add_argument("--transport", default="tcp", choices=T.BACKENDS)
    ap.add_argument("--host", default="0.0.0.0", help="bind address")
    ap.add_argument("--advertise", help="address peers should use to reach this worker")
    ap.add_argument("--slowdown", type=float, default=1.0, help="emulate a slower device (x times)")
    ap.add_argument("--mem-limit-gb", type=float, help="advertise at most this much free memory")
    ap.add_argument("--netem", help="outbound delay emulation: wifi | none | profile.json | base=..,jitter=..")
    ap.add_argument("--profile-budget", type=float, default=15.0)
    ap.add_argument("--wire-dtype", default="float16", help="dtype of hidden states sent to the next stage")
    ap.add_argument("--log", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log, format=f"%(asctime)s [{a.node_id or 'worker'}] %(levelname)s %(message)s")
    Worker(a.coordinator, a.node_id, a.device, a.dtype, a.transport, a.host, a.advertise, a.slowdown,
           a.mem_limit_gb, a.netem, a.profile_budget, a.wire_dtype).run()


if __name__ == "__main__":
    main()
