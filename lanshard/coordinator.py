"""LANShard coordinator: queue, scheduler, planner, drift monitor; embedding and sampling.

Data path for one micro-batch::

    coordinator --embed--> stage 1 --> ... --> stage k --hidden--> coordinator --sample

Scheduling (I4): requests are admitted against a KV-token budget and bound to
the plan that is active at admission. Decode micro-batches are sized as the
smallest batch that keeps the pipeline full, ``ceil(R / D)`` for R decoding
requests and pipeline depth D (stages + hop latency expressed in stage times,
measured online). Prefill goes in separate large chunks with its own slots, so
a long prompt never inflates decode TPOT by riding in a decode batch.

Repartitioning (I3): workers report a rolling median of observed/expected
stage time. When a node drifts past ``threshold`` for ``sustain_s`` (or leaves),
the planner re-runs with that node's layer time scaled; new layers are staged
in the background, then admission switches to the new plan while in-flight
requests drain on the old one. Nothing is migrated or recomputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import queue
import socket
import statistics
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from . import transport as T
from .model_utils import HeadModules, Segment, default_device, default_dtype, load_config, parse_dtype, resolve_model_dir
from .partitioner import COORD, InfeasibleError, Plan, make_plan, node_specs_from_profiles, plan_cost
from .worker import EchoServer, measure_link

log = logging.getLogger("lanshard.coordinator")
REQUEUE = -1  # Request.migrate_to sentinel: return to the queue instead of moving to a plan


# --------------------------------------------------------------------------- config
@dataclass
class SchedulerConfig:
    max_decode_mb: int = 16          # cap on decode micro-batch size
    prefill_chunk: int = 512         # tokens per prefill micro-batch
    max_prefill_inflight: int = 1    # prefill micro-batches in flight at once
    depth: Optional[int] = None      # fixed pipeline depth (None = adaptive from RTT)
    max_depth: int = 12
    kv_token_budget: int = 16384     # sum of (prompt + max_new) over admitted requests
    max_active: int = 64
    planning_batch: int = 4          # decode micro-batch size the planner optimises for


@dataclass
class DriftConfig:
    enabled: bool = True
    threshold: float = 0.25          # 25% over the calibrated stage time
    sustain_s: float = 30.0
    calib_s: float = 15.0            # window that establishes each node's baseline ratio
    cooldown_s: float = 30.0
    settle_s: float = 20.0           # ignore drift right after a switch (stage shapes just changed)
    min_gain: float = 0.10           # predicted bottleneck improvement required to switch
    # On a switch, in-flight requests with at least this many tokens left are moved to the
    # new plan by re-prefilling prompt+generated text ("recompute"); the rest drain in place.
    # "drain" = never move anything (pure drain-then-switch).
    migrate: str = "recompute"
    migrate_min_remaining: int = 32
    horizon_s: float = 300.0         # how far ahead a switch must pay for itself
    pause_estimate_s: float = 10.0   # assumed pause of a non-stageable switch until one is measured


@dataclass
class Request:
    rid: int
    prompt: List[int]
    max_new: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None
    ignore_eos: bool = False
    stop_ids: Optional[List[int]] = None
    on_token: Optional[Callable[[int], None]] = None
    # runtime
    tokens: List[int] = field(default_factory=list)
    kv: int = 0  # id of this request's KV cache on the workers; a new one for every (re)fill
    cached: int = 0
    prefill_target: int = 0
    version: Optional[int] = None
    inflight: bool = False
    generated: int = 0
    t_arrival: float = 0.0
    t_admit: float = 0.0
    t_first: float = 0.0
    t_done: float = 0.0
    token_times: List[float] = field(default_factory=list)
    finish_reason: str = ""
    restarts: int = 0
    migrate_to: Optional[int] = None
    migrations: int = 0
    future: Future = field(default_factory=Future)
    gen: Optional[torch.Generator] = None

    @property
    def max_len(self) -> int:
        return len(self.prompt) + self.max_new + 1

    @property
    def decoding(self) -> bool:
        return self.cached >= self.prefill_target and self.generated > 0


@dataclass
class Node:
    node_id: str
    conn: T.JsonConn
    info: Dict
    data_addr: str
    echo_addr: str
    profile: Optional[Dict] = None
    rtt: Dict[str, float] = field(default_factory=dict)
    mbps: Dict[str, float] = field(default_factory=dict)  # measured bandwidth to each peer
    alive: bool = True
    stats: Optional[Dict] = None
    baseline: Optional[float] = None
    calib: List[float] = field(default_factory=list)
    calib_t0: Optional[float] = None
    over_since: Optional[float] = None
    under_since: Optional[float] = None
    ready_versions: set = field(default_factory=set)
    failed_versions: set = field(default_factory=set)
    last_seen: float = field(default_factory=time.time)


class Coordinator:
    def __init__(self, model: str, device: Optional[str] = None, dtype: Optional[str] = None,
                 transport: str = "tcp", host: str = "0.0.0.0", port: int = 29500,
                 advertise: Optional[str] = None, policy: str = "latency",
                 sched: Optional[SchedulerConfig] = None, drift: Optional[DriftConfig] = None,
                 netem: Optional[str] = None, profile_budget_s: float = 15.0, overhead_gb: float = 0.6,
                 rtt_payload: Optional[int] = None, wire_dtype: str = "float16",
                 heartbeat_timeout_s: float = 15.0):
        self.model = model
        self.device = device or default_device()
        self.dtype = parse_dtype(dtype) if dtype else default_dtype(self.device)
        self.wire_dtype = parse_dtype(wire_dtype)  # what stage 1 receives; workers choose their own
        self.transport = transport
        self.host, self.port, self.advertise = host, port, advertise
        self.policy = policy
        self.sched = sched or SchedulerConfig()
        self.drift = drift or DriftConfig()
        self.netem = T.NetemProfile.from_spec(netem)
        self.profile_budget_s = profile_budget_s
        self.overhead = int(overhead_gb * (1 << 30))
        self.heartbeat_timeout_s = heartbeat_timeout_s  # workers report every second
        self.unservable: Optional[str] = None  # set when the live nodes cannot hold the model

        self.cfg = load_config(model)
        self.num_layers = self.cfg.num_hidden_layers
        self.rtt_payload = rtt_payload or self.cfg.hidden_size * 2
        self.head: Optional[HeadModules] = None
        self.tokenizer = None
        self.eos_ids: set = set()

        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.nodes: Dict[str, Node] = {}
        self.plans: Dict[int, Plan] = {}
        self.active_version: Optional[int] = None
        self.next_version = 1
        self.scale: Dict[str, float] = {}
        self.admission_paused = False
        self.replanning = False
        self.last_replan = 0.0
        self.last_switch = 0.0
        self.last_pause_s: Optional[float] = None  # measured cost of the last paused switch

        self.waiting: deque = deque()
        self.active: Dict[int, Request] = {}
        self.next_rid = 1
        self.next_kv = 1
        self.inflight_mbs: Dict[int, Dict] = {}
        self.next_mb = 1
        self.pending_free: Dict[int, List[int]] = {}
        self.results_q: "queue.Queue" = queue.Queue()
        self.ctrl_q: "queue.Queue" = queue.Queue()
        self.senders: Dict[str, T.Sender] = {}

        # metrics
        self.loop_ms_ewma: Optional[float] = None
        self.stage_ms_ewma: Optional[float] = None
        self.mb_records: List[Dict] = []
        self.req_records: List[Dict] = []
        self.token_log: List[float] = []
        self.drift_log: List[Dict] = []
        self.events: List[Dict] = []
        self.node_stats_log: List[Dict] = []
        self.t_start = time.time()
        self.running = False

    # ================================================================ lifecycle
    def _addr(self, port: int) -> str:
        host = self.advertise or (T.local_ip() if self.host in ("", "0.0.0.0") else self.host)
        return f"{host}:{port}"

    def start(self, load_head: bool = True):
        if load_head:
            self.head = HeadModules(self.model, self.device, self.dtype)
            from transformers import AutoTokenizer, GenerationConfig

            self.tokenizer = AutoTokenizer.from_pretrained(resolve_model_dir(self.model))
            eos = []
            try:
                gc = GenerationConfig.from_pretrained(resolve_model_dir(self.model))
                eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
            except Exception:
                pass
            if self.tokenizer.eos_token_id is not None:
                eos.append(self.tokenizer.eos_token_id)
            self.eos_ids = {e for e in eos if e is not None}
        self.data_server = T.serve(self.transport, self.host, 0, self._on_result)
        self.echo = EchoServer(self.host)
        self.lsock = socket.socket()
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((self.host, self.port))
        self.lsock.listen(32)
        self.port = self.lsock.getsockname()[1]
        self.running = True
        for target in (self._accept_loop, self._scheduler_loop, self._sampler_loop, self._drift_loop,
                       self._ctrl_sender_loop, self._watchdog_loop):
            threading.Thread(target=target, daemon=True).start()
        log.info("coordinator up: control %s, data %s (%s), model %s on %s", self._addr(self.port),
                 self._addr(self.data_server.port), self.transport, self.model, self.device)
        return self

    @property
    def control_address(self) -> str:
        return self._addr(self.port)

    def shutdown(self, workers: bool = True):
        self.running = False
        with self.lock:
            nodes = list(self.nodes.values())
        if workers:
            for n in nodes:
                try:
                    n.conn.send({"type": "shutdown"})
                except OSError:
                    pass

    def event(self, kind: str, **kw):
        e = {"t": time.time() - self.t_start, "event": kind, **kw}
        self.events.append(e)
        log.info("event %s %s", kind, kw if kw else "")

    # ================================================================ control plane
    def _accept_loop(self):
        while self.running:
            try:
                c, _ = self.lsock.accept()
            except OSError:
                return
            threading.Thread(target=self._control_conn, args=(T.JsonConn(c),), daemon=True).start()

    def _control_conn(self, conn: T.JsonConn):
        node: Optional[Node] = None
        while True:
            try:
                msg = conn.recv()
            except (OSError, ValueError):
                msg = None
            if msg is None:
                break
            t = msg["type"]
            with self.cv:
                if t == "join":
                    node = Node(msg["node_id"], conn, msg["info"], msg["data_addr"], msg["echo_addr"])
                    self.nodes[node.node_id] = node
                    self.event("join", node=node.node_id, accel=node.info.get("accel"),
                               mem_free_gb=round(node.info["mem_free"] / 2 ** 30, 2))
                elif node is None:
                    continue
                node.last_seen = time.time()
                if t == "profile_result":
                    node.profile = msg["profile"]
                    node.info["mem_free"] = min(node.info["mem_free"], msg.get("mem_free", node.info["mem_free"]))
                elif t == "rtt":
                    node.rtt = {k: v for k, v in msg["rtt"].items() if v is not None}
                    node.mbps = {k: v for k, v in msg.get("mbps", {}).items() if v}
                elif t == "ready":
                    node.ready_versions.add(msg["version"])
                    self.event("node_ready", node=node.node_id, version=msg["version"],
                               loaded=msg.get("loaded"), load_s=round(msg.get("load_s", 0), 2))
                elif t == "prepare_failed":
                    node.failed_versions.add(msg["version"])
                elif t == "stats":
                    node.stats = msg
                    self.node_stats_log.append({"t": msg["t"] - self.t_start, "node": node.node_id,
                                                "ratio": msg["ratio"], "busy_s": msg["busy_s"],
                                                "wall_s": msg["wall_s"], "msgs": msg["msgs"],
                                                "slowdown": msg["slowdown"], "n_layers": len(msg["layers"]),
                                                "kv_reqs": msg["kv_reqs"], **{f"mem_{k}": v for k, v in msg["mem"].items()}})
                elif t == "error":
                    log.error("worker %s error on %s: %s", node.node_id, msg.get("for"), msg.get("error"))
                self.cv.notify_all()
        if node is not None:
            with self.cv:
                node.alive = False
                self.cv.notify_all()
            self.event("node_lost", node=node.node_id)
            if self.running:
                threading.Thread(target=self._handle_node_loss, args=(node.node_id,), daemon=True).start()

    def _wait(self, pred, timeout: float, what: str):
        deadline = time.time() + timeout
        with self.cv:
            while not pred():
                left = deadline - time.time()
                if left <= 0:
                    raise TimeoutError(f"timed out waiting for {what}")
                self.cv.wait(min(left, 0.5))

    def alive_nodes(self) -> List[Node]:
        with self.lock:
            return [n for n in self.nodes.values() if n.alive]

    # ================================================================ bootstrap
    def bootstrap(self, n_workers: int, timeout: float = 600.0, policy: Optional[str] = None) -> Plan:
        """Wait for workers, profile them (I1), measure RTTs, plan (I2) and install the plan."""
        self._wait(lambda: len(self.alive_nodes()) >= n_workers, timeout, f"{n_workers} workers to join")
        self.profile_all(timeout)
        self.measure_rtts(timeout)
        plan = self.compute_plan(policy or self.policy)
        self.install_plan(plan, timeout)
        return plan

    def profile_all(self, timeout: float = 600.0):
        nodes = self.alive_nodes()
        # profile one node at a time so co-located workers do not disturb each other's timings
        for n in nodes:
            n.profile = None
            n.conn.send({"type": "profile", "model": self.model, "budget_s": self.profile_budget_s})
            self._wait(lambda: n.profile is not None or not n.alive, timeout, f"profile of {n.node_id}")
            if n.profile:
                p = n.profile
                self.event("profiled", node=n.node_id, t_decode_ms=p["t_decode_ms"],
                           t_prefill_ms_per_tok=round(p["t_prefill_ms_per_tok"], 4),
                           mem_layer_mb=round(p["mem_layer"] / 2 ** 20, 1))

    def measure_rtts(self, timeout: float = 120.0):
        nodes = self.alive_nodes()
        peers = {n.node_id: n.echo_addr for n in nodes}
        peers[COORD] = self._addr(self.echo.port)
        for n in nodes:
            n.rtt = {}
            n.conn.send({"type": "ping_peers", "peers": peers, "payload": self.rtt_payload})
        coord_rtt, coord_mbps = {}, {}
        for n in nodes:
            try:
                m = measure_link(n.echo_addr, self.rtt_payload, 20, self.netem)
                coord_rtt[n.node_id] = m["rtt"]
                if m["mbps"]:
                    coord_mbps[n.node_id] = m["mbps"]
            except OSError:
                pass
        self._wait(lambda: all(n.rtt or not n.alive for n in nodes), timeout, "rtt reports")
        self.coord_rtt, self.coord_mbps = coord_rtt, coord_mbps
        self.event("rtt", matrix=self.rtt_matrix(), mbps=self.bandwidth_matrix())

    def rtt_matrix(self) -> Dict[str, Dict[str, float]]:
        m = {COORD: dict(getattr(self, "coord_rtt", {}))}
        for n in self.alive_nodes():
            m[n.node_id] = {k: round(v, 3) for k, v in n.rtt.items()}
        return m

    def bandwidth_matrix(self) -> Dict[str, Dict[str, float]]:
        m = {COORD: dict(getattr(self, "coord_mbps", {}))}
        for n in self.alive_nodes():
            m[n.node_id] = {k: round(v, 1) for k, v in n.mbps.items()}
        return m

    def _link_kw(self) -> Dict:
        """What the planner needs to price link occupancy: one decode micro-batch per hop."""
        payload = self.sched.planning_batch * self.cfg.hidden_size * torch.empty(0, dtype=self.wire_dtype).element_size()
        return {"payload_bytes": payload, "bandwidth_mbps": self.bandwidth_matrix()}

    # ================================================================ planning
    def _specs(self, policy: str, exclude: set):
        recs = {n.node_id: {"info": n.info, "profile": n.profile} for n in self.alive_nodes()
                if n.profile and n.node_id not in exclude}
        return node_specs_from_profiles(recs, self.sched.planning_batch, self._kv_tokens_per_node(len(recs)),
                                        self.overhead, self.scale if policy.startswith("latency") else None)

    def compute_plan(self, policy: Optional[str] = None, exclude: Optional[set] = None,
                     current: Optional[Plan] = None) -> Plan:
        """Plan with ``policy``. With ``current``, prefer a plan whose new layers can be staged
        next to the running ones (zero-pause switch); fall back to an unconstrained plan."""
        policy = policy or self.policy
        exclude = exclude or set()
        specs = self._specs(policy, exclude)
        with self.lock:
            version = self.next_version
            self.next_version += 1
        kw = {}
        if current is not None and policy.startswith("latency"):
            kw["current"] = {st.node_id: (st.start, st.end) for st in current.stages if st.node_id not in exclude}
        link = self._link_kw()
        try:
            plan = make_plan(policy, specs, self.num_layers, self.rtt_matrix(), version, **link, **kw)
        except InfeasibleError:
            if not kw:
                raise
            plan = make_plan(policy, specs, self.num_layers, self.rtt_matrix(), version, **link)
        plan.meta.update(scale=dict(self.scale), specs=[asdict(s) for s in specs])
        log.info("planned %s (%.2f ms to plan)", plan.summary(), plan.plan_time_ms)
        return plan

    def _kv_tokens_per_node(self, n_nodes: int) -> int:
        # every node holds KV for all admitted requests, but only for its own layers,
        # so the per-layer reserve is the whole token budget
        return self.sched.kv_token_budget

    def install_plan(self, plan: Plan, timeout: float = 600.0) -> bool:
        """Stage the plan on its nodes and switch admission to it (drain-then-switch)."""
        t0 = time.time()
        addrs = {n.node_id: n.data_addr for n in self.alive_nodes()}
        addrs[COORD] = self._addr(self.data_server.port)
        members = [self.nodes[s.node_id] for s in plan.stages]
        with self.lock:
            self.plans[plan.version] = plan
        for n in members:
            n.conn.send({"type": "prepare", "plan": plan.to_dict(), "addrs": addrs, **self._prep_extra()})
        self._wait(lambda: all(plan.version in n.ready_versions or plan.version in n.failed_versions
                               or not n.alive for n in members), timeout, f"plan v{plan.version}")
        if any(not n.alive for n in members):
            return self._abort_plan(plan, "node lost while staging")
        if any(plan.version in n.failed_versions for n in members):
            # not enough memory to hold old and new layers at once: stop admitting,
            # drain the old plan, release it, then stage again
            self.event("staging_oom_fallback", version=plan.version)
            t_pause = time.time()
            with self.cv:
                self.admission_paused = True
            old = self.active_version
            if self.drift.migrate == "recompute":
                # don't wait for long requests: put them back in the queue with their text so far;
                # they are re-prefilled on the new plan once it is active
                with self.cv:
                    n_req = 0
                    for r in list(self.active.values()):
                        if r.version == old and r.max_new - r.generated >= self.drift.migrate_min_remaining:
                            if r.inflight:
                                r.migrate_to = REQUEUE
                            else:
                                self._requeue(r)
                            n_req += 1
                self.event("requeue_for_switch", requests=n_req)
            self._wait(lambda: not any(r.version == old for r in self.active.values()), timeout, "drain")
            if old is not None:
                self._release_version(old)
            for n in members:
                n.failed_versions.discard(plan.version)
                n.conn.send({"type": "prepare", "plan": plan.to_dict(), "addrs": addrs, **self._prep_extra()})
            self._wait(lambda: all(plan.version in n.ready_versions or plan.version in n.failed_versions
                                   or not n.alive for n in members), timeout, "restage")
            if not all(plan.version in n.ready_versions and n.alive for n in members):
                # admission stays paused and the old plan is gone: the caller must recover
                return self._abort_plan(plan, "restage failed")
            self.last_pause_s = time.time() - t_pause
        stage_s = time.time() - t0
        self._sender(plan.stages[0].node_id)  # connect before switching
        with self.cv:
            old = self.active_version
            self.active_version = plan.version
            self.admission_paused = False
            self.unservable = None
            self.cv.notify_all()
        self.last_switch = time.time()
        self.event("switch", version=plan.version, old=old, stage_s=round(stage_s, 2), plan=plan.summary())
        if old is not None and self.drift.migrate == "recompute":
            self._migrate_stragglers(old, plan.version)
        self._maybe_release()
        return True

    def _abort_plan(self, plan: Plan, reason: str) -> bool:
        self.event("plan_aborted", version=plan.version, reason=reason)
        self._release_version(plan.version)  # nodes that did stage it must not keep its layers
        return False

    def _migrate_stragglers(self, old: int, new: int):
        """Move long-running old-plan requests onto the new plan instead of waiting for them.

        A request is re-prefilled on the new plan with its prompt plus everything generated so
        far, so output continuity is preserved (the next token is sampled from the same context).
        Requests close to finishing stay on the old plan.
        """
        moved = waiting = 0
        with self.cv:
            for r in list(self.active.values()):
                if r.version != old or r.max_new - r.generated < self.drift.migrate_min_remaining:
                    continue
                if r.inflight:
                    r.migrate_to = new  # applied when its in-flight micro-batch returns
                    waiting += 1
                else:
                    self._migrate(r, new)
                    moved += 1
            self.cv.notify_all()
        self.event("migrate", old=old, new=new, moved_now=moved, moved_on_return=waiting,
                   left_to_drain=sum(1 for r in self.active.values() if r.version == old) - moved - waiting)

    def _requeue(self, r: Request):
        """Return an admitted request to the front of the queue, keeping its generated text."""
        self.pending_free.setdefault(r.version, []).append(r.kv)
        self.active.pop(r.rid, None)
        r.cached, r.inflight, r.version, r.migrate_to = 0, False, None, None
        r.restarts += 1
        r.migrations += 1
        self.waiting.appendleft(r)

    def _new_kv(self, r: Request):
        """Give ``r`` a fresh KV cache id. Frees go over the control socket and micro-batches over
        the data sockets, so a free can reach a worker after the request's next prefill. With a
        new id per fill, a late free can only ever drop the old cache."""
        r.kv = self.next_kv
        self.next_kv += 1

    def _migrate(self, r: Request, new: int):
        self.pending_free.setdefault(r.version, []).append(r.kv)
        self._new_kv(r)
        r.version = new
        r.cached = 0
        r.prefill_target = len(r.tokens)
        r.migrate_to = None
        r.migrations += 1

    def _prep_extra(self) -> Dict:
        return {"kv_tokens": self._kv_tokens_per_node(0), "overhead": self.overhead}

    def _release_version(self, version: int):
        plan = self.plans.get(version)
        if plan is None:
            return
        for n in self.alive_nodes():
            n.conn.send({"type": "release", "version": version})
        with self.lock:
            self.plans.pop(version, None)
        self.event("drained", version=version)

    def _maybe_release(self):
        with self.lock:
            stale = [v for v in self.plans if v != self.active_version
                     and not any(r.version == v for r in self.active.values())
                     and v < (self.active_version or 0)]
        for v in stale:
            self._release_version(v)

    def _begin_replan(self, wait: bool) -> bool:
        with self.cv:
            while self.replanning:
                if not wait:
                    return False
                self.cv.wait(0.5)
            self.replanning = True
            return True

    def _end_replan(self):
        with self.cv:
            self.replanning = False
            self.last_replan = time.time()
            self.cv.notify_all()

    def replan(self, reason: str):
        """Drift-triggered replan: switch only if ``_choose_plan`` finds a plan worth it."""
        if not self._begin_replan(wait=False):
            return
        try:
            self.event("replan_trigger", reason=reason, scale=dict(self.scale))
            cur = self.plans.get(self.active_version)
            plan = self._choose_plan(cur) if cur else None
            if plan is not None:
                self.install_plan(plan)
        except InfeasibleError as e:
            self.event("replan_infeasible", error=str(e))
        except Exception as e:  # a node died mid-switch, a timeout: recover below
            log.exception("replan failed")
            self.event("replan_failed", error=repr(e))
        finally:
            self._end_replan()
        self._ensure_servable(reason)

    def _servable(self) -> bool:
        plan = self.plans.get(self.active_version)
        return plan is not None and all(self.nodes[s.node_id].alive for s in plan.stages)

    def _ensure_servable(self, reason: str, attempts: int = 3):
        """Make sure the active plan can run; if not, install one over the live nodes.

        Called after a node loss and after any replan, because a switch that fails half way
        (a node dies while staging, a paused switch cannot restage) leaves admission paused.
        It waits for a replan in progress instead of being dropped by it. If the live nodes
        cannot hold the model, queued requests fail with an error instead of waiting forever.
        """
        if self.active_version is None:
            return  # nothing has been installed yet: a node lost while joining is bootstrap's business
        self._begin_replan(wait=True)
        try:
            error: Exception = RuntimeError("could not install a plan")
            for _ in range(attempts):
                with self.cv:
                    if self._servable():
                        self.admission_paused = False
                        self.cv.notify_all()
                        return
                    self.admission_paused = True
                self.event("replan_trigger", reason=reason, scale=dict(self.scale))
                try:
                    self.install_plan(self.compute_plan(self.policy))
                except InfeasibleError as e:
                    error = e
                    break
                except Exception as e:
                    log.exception("recovery replan failed")
                    error = e
            else:
                if self._servable():
                    return
            self.event("replan_infeasible", error=str(error))
            self._fail_all(f"the live nodes cannot hold the model: {error}")
        finally:
            self._end_replan()

    def _fail_all(self, msg: str):
        with self.cv:
            reqs = list(self.waiting) + list(self.active.values())
            for r in self.active.values():
                self.pending_free.setdefault(r.version, []).append(r.kv)
            self.waiting.clear()
            self.active.clear()
            self.inflight_mbs.clear()
            self.unservable = msg
            self.cv.notify_all()
        for r in reqs:
            r.finish_reason = "error"
            r.future.set_exception(RuntimeError(msg))

    def _choose_plan(self, cur: Plan) -> Optional[Plan]:
        """Choose between keeping the current plan, a zero-pause staged plan and an unconstrained
        plan that may need a pause, by expected tokens over the next ``horizon_s`` seconds.

        Throughput is taken as proportional to 1 / bottleneck. A plan that needs a pause earns
        nothing for the estimated pause (the last measured one, else ``pause_estimate_s``).
        """
        H = self.drift.horizon_s
        specs = self._specs(self.policy, set())
        rtt = self.rtt_matrix()
        staged = self.compute_plan(self.policy, current=cur)
        free = self.compute_plan(self.policy)
        hop_mode = staged.meta.get("hop_mode", "overlap")
        now_ms = plan_cost(cur, specs, rtt, hop_mode, **self._link_kw())
        pause = self.last_pause_s if self.last_pause_s is not None else self.drift.pause_estimate_s

        def layout(p):
            return [(st.node_id, st.start, st.end) for st in p.stages]

        cands = [("keep", None, H / now_ms)]
        if layout(staged) != layout(cur):
            cands.append(("staged", staged, H / staged.bottleneck_ms))
        if layout(free) != layout(cur) and layout(free) != layout(staged):
            needs_pause = self._needs_pause(cur, free)
            cands.append(("paused" if needs_pause else "staged", free,
                          (H - (pause if needs_pause else 0.0)) / free.bottleneck_ms))
        kind, plan, value = max(cands, key=lambda c: c[2])
        gain = value / cands[0][2] - 1
        equiv = {k: round(H / v, 1) for k, _, v in cands}
        if plan is None or gain < self.drift.min_gain:
            self.event("replan_skipped", current_ms=round(now_ms, 1), best=kind, predicted_gain=round(gain, 3),
                       options_equiv_bottleneck_ms=equiv, pause_est_s=round(pause, 1))
            return None
        self.event("replan_accepted", version=plan.version, option=kind, predicted_gain=round(gain, 3),
                   current_ms=round(now_ms, 1), new_ms=round(plan.bottleneck_ms, 1),
                   options_equiv_bottleneck_ms=equiv, pause_est_s=round(pause, 1))
        return plan

    def _needs_pause(self, cur: Plan, new: Plan) -> bool:
        """Would some node have to hold more than its capacity while both plans are loaded?"""
        caps = {sp.node_id: sp.max_layers() for sp in self._specs(self.policy, set())}
        old = {st.node_id: set(range(st.start, st.end)) for st in cur.stages}
        for st in new.stages:
            union = old.get(st.node_id, set()) | set(range(st.start, st.end))
            if len(union) > caps.get(st.node_id, 10 ** 9):
                return True
        return False

    def _handle_node_loss(self, node_id: str):
        with self.cv:
            hit = [v for v, p in self.plans.items() if any(s.node_id == node_id for s in p.stages)]
            # requests on a plan that lost a stage cannot finish there: requeue them with
            # everything generated so far as the new prompt (the one case we recompute)
            for r in list(self.active.values()):
                if r.version in hit:
                    self.pending_free.setdefault(r.version, []).append(r.kv)
                    del self.active[r.rid]
                    r.cached, r.inflight, r.version, r.migrate_to = 0, False, None, None
                    r.restarts += 1
                    self.waiting.appendleft(r)
            for mb, rec in list(self.inflight_mbs.items()):
                if rec["v"] in hit:
                    del self.inflight_mbs[mb]
            if self.active_version in hit:
                self.admission_paused = True
            dead = self.senders.pop(node_id, None)
        if dead is not None:
            dead.close()
        self._ensure_servable(f"node {node_id} lost")
        for v in hit:
            if v != self.active_version:
                self._release_version(v)

    # ================================================================ drift monitor (I3)
    def _check_heartbeats(self):
        """A suspended laptop or a dropped Wi-Fi link leaves the control socket open, so the
        read never ends. Treat a node that has been silent for too long as lost."""
        now = time.time()
        for n in self.alive_nodes():
            if now - n.last_seen > self.heartbeat_timeout_s:
                self.event("heartbeat_timeout", node=n.node_id, silent_s=round(now - n.last_seen, 1))
                n.last_seen = now  # report once
                n.conn.shutdown()  # wakes the reader, which runs the node-loss path

    def _drift_loop(self):
        while self.running:
            time.sleep(1.0)
            try:
                self._check_heartbeats()
                self._drift_step()
            except Exception:  # must not kill the monitor thread
                log.exception("drift monitor")

    def _drift_step(self):
        if not self.drift.enabled or self.active_version is None or self.replanning:
            return
        now = time.time()
        if now - self.last_switch < self.drift.settle_s:
            for nd in list(self.nodes.values()):
                nd.over_since = nd.under_since = None
            return
        plan = self.plans.get(self.active_version)
        if plan is None:
            return
        trigger = None
        for s in plan.stages:
            n = self.nodes.get(s.node_id)
            if n is None or not n.alive or not n.stats or n.stats.get("ratio") is None:
                continue
            ratio = n.stats["ratio"]
            if n.baseline is None:
                if n.calib_t0 is None:
                    n.calib_t0 = now
                n.calib.append(ratio)
                if now - n.calib_t0 >= self.drift.calib_s:
                    n.baseline = statistics.median(n.calib)
                    self.event("drift_baseline", node=n.node_id, baseline=round(n.baseline, 3))
                continue
            drift = ratio / n.baseline
            cur = self.scale.get(n.node_id, 1.0)
            rel = drift / cur
            self.drift_log.append({"t": now - self.t_start, "node": n.node_id, "ratio": ratio,
                                   "drift": drift, "scale": cur})
            if rel > 1 + self.drift.threshold:
                n.over_since = n.over_since or now
                n.under_since = None
            elif cur > 1.0 and drift < 1 + self.drift.threshold:  # back near its baseline: recovered
                n.under_since = n.under_since or now
                n.over_since = None
            else:
                n.over_since = n.under_since = None
            since = n.over_since or n.under_since
            if since and now - since >= self.drift.sustain_s and now - self.last_replan >= self.drift.cooldown_s:
                trigger = (n, drift)
        if trigger:
            n, drift = trigger
            self.scale[n.node_id] = round(max(1.0, drift), 3)
            n.over_since = n.under_since = None
            threading.Thread(target=self.replan, args=(f"drift {n.node_id} x{drift:.2f}",), daemon=True).start()

    # ================================================================ requests
    def submit(self, prompt: List[int], max_new: int = 128, temperature: float = 0.0, top_p: float = 1.0,
               top_k: int = 0, seed: Optional[int] = None, ignore_eos: bool = False,
               on_token: Optional[Callable[[int], None]] = None, stop_ids: Optional[List[int]] = None) -> Request:
        if len(prompt) == 0:
            raise ValueError("empty prompt")
        if self.unservable:
            raise RuntimeError(self.unservable)
        with self.cv:
            r = Request(self.next_rid, list(prompt), max_new, temperature, top_p, top_k, seed, ignore_eos,
                        stop_ids, on_token)
            self.next_rid += 1
            r.tokens = list(prompt)
            r.prefill_target = len(prompt)
            r.t_arrival = time.time()
            if temperature > 0:
                r.gen = torch.Generator(device="cpu")
                r.gen.manual_seed(seed if seed is not None else r.rid)
            self.waiting.append(r)
            self.cv.notify_all()
        return r

    def generate(self, prompt: List[int], **kw) -> List[int]:
        r = self.submit(prompt, **kw)
        return r.future.result()

    def chat(self, messages: List[Dict], **kw) -> str:
        ids = self.encode_chat(messages)
        out = self.generate(ids, **kw)
        return self.tokenizer.decode(out, skip_special_tokens=True)

    def encode_chat(self, messages: List[Dict]) -> List[int]:
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return list(ids)

    # ================================================================ scheduler
    def _depth(self, plan: Plan) -> int:
        if self.sched.depth:
            return self.sched.depth
        k = len(plan.stages)
        if self.loop_ms_ewma and self.stage_ms_ewma:
            d = math.ceil(self.loop_ms_ewma / max(self.stage_ms_ewma, 0.05))
            return int(min(max(d, k), self.sched.max_depth))
        return k + 1

    def _admit(self):
        if self.admission_paused or self.active_version is None:
            return
        used = sum(r.max_len for r in self.active.values())
        while self.waiting and len(self.active) < self.sched.max_active:
            r = self.waiting[0]
            if used + r.max_len > self.sched.kv_token_budget and self.active:
                break
            self.waiting.popleft()
            r.version = self.active_version
            self._new_kv(r)
            r.t_admit = time.time()
            if r.restarts:  # resume from what has been generated so far
                r.prefill_target = len(r.tokens)
            self.active[r.rid] = r
            used += r.max_len

    def _scheduler_loop(self):
        while self.running:
            batches = []
            with self.cv:
                self._admit()
                batches = self._form_batches()
                frees = self.pending_free
                self.pending_free = {}
                if not batches and not frees:
                    self.cv.wait(0.01)
            for v, rids in frees.items():
                self._send_free(v, rids)
            for b in batches:
                self._dispatch(*b)
            if frees:
                self._maybe_release()

    def _form_batches(self):
        out = []
        by_version: Dict[int, List[Request]] = {}
        for r in self.active.values():
            by_version.setdefault(r.version, []).append(r)
        infl_dec = sum(1 for m in self.inflight_mbs.values() if m["kind"] == "decode")
        infl_pre = sum(1 for m in self.inflight_mbs.values() if m["kind"] == "prefill")
        for v, reqs in sorted(by_version.items()):
            plan = self.plans.get(v)
            if plan is None:
                continue
            depth = self._depth(plan)
            decoding = [r for r in reqs if r.decoding]
            ready_dec = sorted((r for r in decoding if not r.inflight), key=lambda r: r.token_times[-1] if r.token_times else 0)
            mb_size = max(1, min(self.sched.max_decode_mb, math.ceil(len(decoding) / depth))) if decoding else 0
            while ready_dec and infl_dec < depth:
                take, ready_dec = ready_dec[:mb_size], ready_dec[mb_size:]
                segs = [Segment(r.kv, r.cached, 1, r.max_len) for r in take]
                for r in take:
                    r.inflight = True
                out.append((v, "decode", take, segs))
                infl_dec += 1
            ready_pre = [r for r in reqs if not r.decoding and not r.inflight]
            while ready_pre and infl_pre < self.sched.max_prefill_inflight:
                budget, take, segs = self.sched.prefill_chunk, [], []
                for r in list(ready_pre):
                    if budget <= 0:
                        break
                    n = min(budget, r.prefill_target - r.cached)
                    segs.append(Segment(r.kv, r.cached, n, r.max_len))
                    take.append(r)
                    ready_pre.remove(r)
                    r.inflight = True
                    budget -= n
                out.append((v, "prefill", take, segs))
                infl_pre += 1
        return out

    def _sender(self, node_id: str) -> T.Sender:
        s = self.senders.get(node_id)
        if s is None:
            s = T.connect(self.transport, self.nodes[node_id].data_addr, self.netem)
            self.senders[node_id] = s
        return s

    def _dispatch(self, version: int, kind: str, reqs: List[Request], segs: List[Segment]):
        plan = self.plans[version]
        t0 = time.perf_counter()
        ids = []
        for r, s in zip(reqs, segs):
            ids.extend(r.tokens[s.start:s.start + s.n])
        x = self.head.embed_tokens(torch.tensor(ids, dtype=torch.long)).to(self.wire_dtype)
        embed_ms = (time.perf_counter() - t0) * 1e3
        with self.lock:
            mb = self.next_mb
            self.next_mb += 1
            self.inflight_mbs[mb] = {"v": version, "kind": kind, "rids": [r.rid for r in reqs],
                                     "segs": segs, "t_send": time.perf_counter(), "embed_ms": embed_ms,
                                     "n_tok": len(ids)}
        header = {"mb": mb, "v": version, "st": 0, "seg": [s.to_list() for s in segs], "tr": []}
        try:
            self._sender(plan.stages[0].node_id).send(header, x.cpu())
        except OSError as e:
            log.error("send to %s failed: %s", plan.stages[0].node_id, e)

    def _send_free(self, version: int, rids: List[int]):
        plan = self.plans.get(version)
        nodes = [s.node_id for s in plan.stages] if plan else [n.node_id for n in self.alive_nodes()]
        for nid in nodes:
            n = self.nodes.get(nid)
            if n and n.alive:
                self.ctrl_q.put((n, {"type": "free", "rids": rids}))

    def _ctrl_sender_loop(self):
        # control-plane writes happen here so the scheduler never blocks on a socket
        while self.running:
            try:
                n, msg = self.ctrl_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                n.conn.send(msg)
            except OSError:
                pass

    def _retransmit_stale(self):
        """Re-send micro-batches that have been in flight far longer than normal.

        Safe because a micro-batch is idempotent: re-running the same segments writes
        identical K/V at the same positions, and a late original result is ignored.
        """
        limit = max(30.0, 20 * (self.loop_ms_ewma or 0) / 1e3)
        now = time.perf_counter()
        with self.lock:
            stale = [(mb, rec) for mb, rec in self.inflight_mbs.items() if now - rec["t_send"] > limit]
            for mb, _ in stale:
                del self.inflight_mbs[mb]
        for mb, rec in stale:
            reqs = [self.active.get(rid) for rid in rec["rids"]]
            if any(r is None for r in reqs) or rec["v"] not in self.plans:
                continue
            self.event("retransmit", mb=mb, age_s=round(now - rec["t_send"], 1), mb_kind=rec["kind"])
            self._dispatch(rec["v"], rec["kind"], reqs, rec["segs"])

    def _watchdog_loop(self, stall_s: float = 60.0):
        import faulthandler
        import sys

        last_n, last_change = 0, time.time()
        while self.running:
            time.sleep(5)
            self._retransmit_stale()
            n = self.next_mb
            busy = bool(self.active or self.waiting)
            if n != last_n or not busy:
                last_n, last_change = n, time.time()
                continue
            if time.time() - last_change > stall_s:
                log.error("STALL: %d active, %d waiting, %d in flight, no dispatch for %.0fs; thread dump follows",
                          len(self.active), len(self.waiting), len(self.inflight_mbs), time.time() - last_change)
                for nd in self.alive_nodes():
                    st = nd.stats or {}
                    log.error("  %s: recv=%s processed=%s inbox=%s last_mb=%s links=%s", nd.node_id, st.get("recv"),
                              st.get("msgs"), st.get("inbox"), st.get("last_mb"), st.get("links"))
                    self.ctrl_q.put((nd, {"type": "dump"}))
                log.error("  in flight: %s", sorted(self.inflight_mbs))
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                self.event("stall", active=len(self.active), waiting=len(self.waiting),
                           inflight=len(self.inflight_mbs))
                last_change = time.time()

    # ================================================================ results + sampling
    def _on_result(self, header: Dict, tensor: torch.Tensor):
        header["_t_recv"] = time.perf_counter()
        self.results_q.put((header, tensor))

    def _sampler_loop(self):
        while self.running:
            try:
                header, hidden = self.results_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_result(header, hidden)
            except Exception:
                log.exception("result processing failed")

    def _process_result(self, header: Dict, hidden: torch.Tensor):
        with self.lock:
            rec = self.inflight_mbs.pop(header["mb"], None)
        if rec is None:
            return  # belonged to a plan that lost a node
        t_recv = header["_t_recv"]
        t0 = time.perf_counter()
        rows, reqs = [], []
        off = 0

        def current(rid, s):
            """The request this segment was computed for, unless it has been requeued or migrated
            since the dispatch (node loss): then the result belongs to a cache that no longer exists."""
            r = self.active.get(rid)
            return r if r is not None and r.kv == s.rid else None

        for rid, s in zip(rec["rids"], rec["segs"]):
            r = current(rid, s)
            if r is not None and s.start + s.n >= r.prefill_target:
                rows.append(off + s.n - 1)
                reqs.append((r, s))
            off += s.n
        new_tokens = []
        if rows:
            logits = self.head.logits(hidden[rows])
            for i, (r, _) in enumerate(reqs):
                new_tokens.append(self._sample(r, logits[i]))
        sample_ms = (time.perf_counter() - t0) * 1e3
        now = time.time()
        finished = []
        with self.cv:
            for rid, s in zip(rec["rids"], rec["segs"]):
                r = current(rid, s)
                if r is None:
                    continue
                r.cached = s.start + s.n
                r.inflight = False
            for (r, s), tok in zip(reqs, new_tokens):
                if current(r.rid, s) is not r:
                    continue
                r.tokens.append(tok)
                r.generated += 1
                r.token_times.append(now)
                if r.generated == 1 and not r.t_first:
                    r.t_first = now
                self.token_log.append(now)
                if r.on_token:
                    try:
                        r.on_token(tok)
                    except Exception:
                        pass
                reason = None
                if not r.ignore_eos and tok in self.eos_ids:
                    reason = "stop"
                elif r.stop_ids and tok in r.stop_ids:
                    reason = "stop"
                elif r.generated >= r.max_new:
                    reason = "length"
                if reason:
                    r.finish_reason = reason
                    finished.append(r)
            for r in finished:
                self.active.pop(r.rid, None)
                self.pending_free.setdefault(r.version, []).append(r.kv)
                r.t_done = now
            for rid in rec["rids"]:  # deferred migrations, after this step's token is appended
                r = self.active.get(rid)
                if r is not None and r.migrate_to is not None:
                    if r.migrate_to == REQUEUE:
                        self._requeue(r)
                    else:
                        self._migrate(r, r.migrate_to)
            self.cv.notify_all()
        loop_ms = (t_recv - rec["t_send"]) * 1e3
        tr = header.get("tr", [])
        stage_max = max((c + d for _, c, d, _ in tr), default=0.0)
        a = 0.2
        if rec["kind"] == "decode":
            # Pipeline depth = how many stage-times one trip takes *without* queueing.
            # Using the raw loop time would feed queueing back into the depth estimate,
            # shrink micro-batches, and cause more queueing.
            free_loop = max(stage_max, loop_ms - sum(q for _, _, _, q in tr))
            self.loop_ms_ewma = free_loop if self.loop_ms_ewma is None else (1 - a) * self.loop_ms_ewma + a * free_loop
            self.stage_ms_ewma = stage_max if self.stage_ms_ewma is None else (1 - a) * self.stage_ms_ewma + a * stage_max
        compute = sum(c for _, c, _, _ in tr)
        deser = sum(d for _, _, d, _ in tr) + header.get("_deser_ms", 0.0)
        queue_ms = sum(q for _, _, _, q in tr)
        self.mb_records.append({
            "t": now - self.t_start, "mb": header["mb"], "v": rec["v"], "kind": rec["kind"],
            "n_seg": len(rec["segs"]), "n_tok": rec["n_tok"], "loop_ms": loop_ms, "compute_ms": compute,
            "deser_ms": deser, "queue_ms": queue_ms,
            "network_ms": max(0.0, loop_ms - compute - deser - queue_ms),
            "embed_ms": rec["embed_ms"], "sample_ms": sample_ms,
            "stages": ";".join(f"{n}:{c:.2f}" for n, c, _, _ in tr)})
        for r in finished:
            self.req_records.append(self._req_record(r))
            r.future.set_result(r.tokens[len(r.prompt):])

    def _sample(self, r: Request, logits: torch.Tensor) -> int:
        if r.temperature <= 0:
            return int(torch.argmax(logits))
        logits = logits.cpu() / r.temperature
        if r.top_k:
            v, _ = torch.topk(logits, r.top_k)
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, -1)
        if r.top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, -1) - sp < r.top_p
            sp = sp * keep
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1, generator=r.gen))

    def _req_record(self, r: Request) -> Dict:
        itl = [b - a for a, b in zip(r.token_times, r.token_times[1:])]
        return {"rid": r.rid, "prompt_len": len(r.prompt), "gen_len": r.generated, "version": r.version,
                "t_arrival": r.t_arrival - self.t_start, "t_admit": r.t_admit - self.t_start,
                "ttft_ms": (r.t_first - r.t_arrival) * 1e3 if r.t_first else None,
                "tpot_ms": statistics.mean(itl) * 1e3 if itl else None,
                "itl_max_ms": max(itl) * 1e3 if itl else None,
                "e2e_ms": (r.t_done - r.t_arrival) * 1e3, "finish": r.finish_reason, "restarts": r.restarts,
                "migrations": r.migrations,
                "itl_ms": [round(x * 1e3, 3) for x in itl]}

    # ================================================================ control helpers
    def set_slowdown(self, node_id: str, factor: float):
        self.nodes[node_id].conn.send({"type": "set_slowdown", "factor": factor})
        self.event("set_slowdown", node=node_id, factor=factor)

    def reset_metrics(self):
        with self.lock:
            self.mb_records, self.req_records, self.token_log = [], [], []
            self.drift_log, self.node_stats_log = [], []
        for n in self.alive_nodes():
            n.conn.send({"type": "reset_stats"})

    def status(self) -> Dict:
        with self.lock:
            plan = self.plans.get(self.active_version)
            return {
                "model": self.model, "transport": self.transport, "policy": self.policy,
                "active_version": self.active_version, "plan": plan.to_dict() if plan else None,
                "waiting": len(self.waiting), "active": len(self.active), "inflight_mbs": len(self.inflight_mbs),
                "nodes": {n.node_id: {"alive": n.alive, "accel": n.info.get("accel"), "os": n.info.get("os"),
                                      "mem_free_gb": round(n.info["mem_free"] / 2 ** 30, 2),
                                      "profile": n.profile, "rtt": n.rtt, "baseline": n.baseline,
                                      "scale": self.scale.get(n.node_id, 1.0),
                                      "stats": {k: v for k, v in (n.stats or {}).items() if k != "type"}}
                          for n in self.nodes.values()},
                "loop_ms_ewma": self.loop_ms_ewma, "stage_ms_ewma": self.stage_ms_ewma,
            }

    def device_table(self) -> List[Dict]:
        rows = []
        for n in self.nodes.values():
            i, p = n.info, n.profile or {}
            rows.append({"node": n.node_id, "host": i.get("hostname"), "os": i.get("os"), "accel": i.get("accel"),
                         "device": i.get("device"), "mem_free_gb": round(i["mem_free"] / 2 ** 30, 2),
                         "ram_total_gb": round(i["ram_total"] / 2 ** 30, 1), "slowdown": i.get("slowdown", 1.0),
                         "t_layer_b1_ms": p.get("t_decode_ms", {}).get("1"),
                         "t_layer_b4_ms": p.get("t_decode_ms", {}).get("4"),
                         "t_prefill_ms_per_tok": p.get("t_prefill_ms_per_tok"),
                         "mem_layer_mb": round(p.get("mem_layer", 0) / 2 ** 20, 1)})
        return rows

    def export(self, out_dir: str, extra: Optional[Dict] = None):
        os.makedirs(out_dir, exist_ok=True)

        def wcsv(name, rows):
            if not rows:
                return
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})

        wcsv("requests.csv", self.req_records)
        wcsv("microbatches.csv", self.mb_records)
        wcsv("drift.csv", self.drift_log)
        wcsv("node_stats.csv", self.node_stats_log)
        wcsv("devices.csv", self.device_table())
        with open(os.path.join(out_dir, "tokens.json"), "w") as f:
            json.dump([t - self.t_start for t in self.token_log], f)
        with open(os.path.join(out_dir, "run.json"), "w") as f:
            json.dump({"status": self.status(), "events": self.events, "rtt": self.rtt_matrix(),
                       "mbps": self.bandwidth_matrix(),
                       "plans": {v: p.to_dict() for v, p in self.plans.items()},
                       "sched": asdict(self.sched), "drift": asdict(self.drift), **(extra or {})},
                      f, indent=1, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser(description="LANShard coordinator")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--workers", type=int, required=True, help="number of workers to wait for")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--advertise")
    ap.add_argument("--device")
    ap.add_argument("--dtype")
    ap.add_argument("--transport", default="tcp", choices=T.BACKENDS)
    ap.add_argument("--policy", default="latency", choices=["latency", "latency-additive", "latency-fixed", "memory", "even"])
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--netem")
    ap.add_argument("--profile-budget", type=float, default=15.0)
    ap.add_argument("--kv-budget", type=int, default=16384)
    ap.add_argument("--http", type=int, default=8000, help="port for the OpenAI-compatible API (0 = off)")
    ap.add_argument("--log", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log, format="%(asctime)s [coord] %(levelname)s %(message)s")
    c = Coordinator(a.model, a.device, a.dtype, a.transport, a.host, a.port, a.advertise, a.policy,
                    SchedulerConfig(kv_token_budget=a.kv_budget), DriftConfig(enabled=not a.no_drift),
                    a.netem, a.profile_budget).start()
    print(f"workers: python -m lanshard.worker --coordinator {c.control_address} --transport {a.transport}", flush=True)
    plan = c.bootstrap(a.workers)
    print(plan.summary(), flush=True)
    if a.http:
        from .api import serve_api

        serve_api(c, a.host, a.http)
    else:
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
