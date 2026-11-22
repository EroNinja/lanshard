"""I2: latency-balanced partitioning (and the memory-proportional baseline B2).

Pipeline shape: coordinator -> stage 1 -> ... -> stage k -> coordinator.
Stage cost for node n holding ``m`` layers with successor ``s``::

    cost = m * t_layer(n) + hop(n -> s)

where ``t_layer`` is the profiled per-layer time for the planning micro-batch
and ``hop`` is the measured one-way latency (RTT / 2) plus payload/bandwidth.
Decode throughput of a full pipeline is set by its slowest stage, so we
minimise the maximum stage cost (ties broken by total latency), subject to each
node's memory: ``m * mem_layer + m * kv_bytes_per_layer + overhead <= mem_free``.

For a fixed node order the exact DP is O(L^2 N). With N <= 5 we also search
every ordered subset of nodes (<= 325 orders), so a node on a bad link is moved
to the end or dropped entirely if that lowers the bottleneck.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .probe import expected_layer_ms

COORD = "coordinator"


@dataclass
class NodeSpec:
    node_id: str
    t_layer_ms: float          # per-layer time for the planning micro-batch
    mem_free: int              # bytes available for weights + KV
    mem_layer: int             # bytes of weights per layer
    kv_bytes_per_layer: int = 0  # KV reserve per layer (kv_per_tok_layer * kv_token_budget)
    overhead: int = 0          # fixed reserve (runtime context, activations)

    def max_layers(self) -> int:
        per = self.mem_layer + self.kv_bytes_per_layer
        return max(0, int((self.mem_free - self.overhead) // per)) if per else 10 ** 6


@dataclass
class Stage:
    node_id: str
    start: int
    end: int  # exclusive
    cost_ms: float = 0.0

    @property
    def n(self) -> int:
        return self.end - self.start


@dataclass
class Plan:
    version: int
    policy: str
    stages: List[Stage]
    bottleneck_ms: float
    latency_ms: float
    num_layers: int
    plan_time_ms: float = 0.0
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Plan":
        d = dict(d)
        d["stages"] = [Stage(**s) for s in d["stages"]]
        return cls(**d)

    def layers_of(self, node_id: str) -> List[int]:
        for s in self.stages:
            if s.node_id == node_id:
                return list(range(s.start, s.end))
        return []

    def summary(self) -> str:
        parts = [f"{s.node_id}[{s.start}:{s.end}]={s.cost_ms:.1f}ms" for s in self.stages]
        return f"v{self.version} {self.policy}: " + " -> ".join(parts) + f" | bottleneck {self.bottleneck_ms:.1f}ms"


class InfeasibleError(RuntimeError):
    pass


def xfer_ms(bandwidth_mbps, a: str, b: str, payload_bytes: int) -> float:
    """Link occupancy: how long the link a -> b is busy sending one planning payload.

    ``bandwidth_mbps`` is one number for every link or a measured ``{a: {b: mbps}}`` matrix;
    a link that is missing from the matrix counts as unlimited.
    """
    mbps = bandwidth_mbps.get(a, {}).get(b) if isinstance(bandwidth_mbps, dict) else bandwidth_mbps
    return payload_bytes * 8 / (mbps * 1e6) * 1e3 if mbps else 0.0


def hop_ms(rtt: Dict[str, Dict[str, float]], a: str, b: str, payload_bytes: int = 0,
           bandwidth_mbps=0.0) -> float:
    r = rtt.get(a, {}).get(b)
    if r is None:
        r = rtt.get(b, {}).get(a, 0.0)
    return r / 2.0 + xfer_ms(bandwidth_mbps, a, b, payload_bytes)


def _dp_fixed_order(order: Sequence[NodeSpec], L: int, rtts: Sequence[float], xfers: Sequence[float],
                    hop_mode: str = "overlap", current: Optional[Dict[str, tuple]] = None,
                    limit: Optional[float] = None):
    """Exact min-max contiguous split for a fixed node order.

    With ``limit`` it instead returns the split with the lowest total latency among those whose
    every stage costs at most ``limit``. Running it once without and once with the optimal
    bottleneck as the limit gives the exact lexicographic optimum (bottleneck, then latency);
    breaking ties inside the min-max pass alone does not, because the best prefix by bottleneck
    is not always the best prefix by latency.

    rtts[i]/xfers[i]: one-way latency and link-occupancy time of node i's outbound hop.
    hop_mode "overlap": with enough micro-batches in flight a hop's latency overlaps other
    stages' compute, so a stage limits throughput by max(compute, link occupancy) and the
    latency only counts toward single-stream latency (the tie-breaker).
    hop_mode "additive": the proposal's original model, compute + one-way latency.
    current: {node: (start, end)} of the running plan; if given, a node may only take a
    range whose union with its current one fits its memory, so the new layers can be
    staged while the old plan drains (no admission pause).
    """
    k = len(order)
    INF = math.inf
    f = np.full((k + 1, L + 1), INF)
    lat = np.full((k + 1, L + 1), INF)
    arg = np.zeros((k + 1, L + 1), dtype=int)
    f[0][0] = 0.0
    lat[0][0] = 0.0
    for j in range(1, k + 1):
        n = order[j - 1]
        cap = min(n.max_layers(), L)
        old = (current or {}).get(n.node_id)
        for e in range(j, L + 1):
            s_arr = np.arange(e)
            cnt = (e - s_arr).astype(float)
            comp = cnt * n.t_layer_ms
            if hop_mode == "additive":
                cost = comp + rtts[j - 1]
            else:
                cost = np.maximum(comp, xfers[j - 1])
            latc_all = comp + rtts[j - 1] + xfers[j - 1]
            bad = cnt > cap
            if old is not None and current is not None:
                os_, oe = old
                overlap = np.clip(np.minimum(e, oe) - np.maximum(s_arr, os_), 0, None)
                bad |= (cnt + (oe - os_) - overlap) > cap
            elif current is not None:
                pass  # a node joining the pipeline holds nothing yet
            cost = np.where(bad, INF, cost)
            if limit is not None:
                tl = np.where(cost <= limit + 1e-9, lat[j - 1][:e] + latc_all, INF)
                pick = int(np.argmin(tl))
                if np.isfinite(tl[pick]):
                    f[j][e], lat[j][e], arg[j][e] = max(f[j - 1][pick], cost[pick]), tl[pick], pick
                continue
            prev = f[j - 1][:e]
            cand = np.maximum(prev, cost)
            best = cand.min()
            if not np.isfinite(best):
                continue
            ties = np.flatnonzero(cand <= best + 1e-9)
            tl = lat[j - 1][ties] + latc_all[ties]
            pick = ties[int(np.argmin(tl))]
            f[j][e], lat[j][e], arg[j][e] = best, tl.min(), pick
    if not np.isfinite(f[k][L]):
        return None
    cuts, e = [], L
    for j in range(k, 0, -1):
        s = arg[j][e]
        cuts.append((int(s), int(e)))
        e = s
    cuts.reverse()
    return float(f[k][L]), float(lat[k][L]), cuts


def latency_balanced(nodes: Sequence[NodeSpec], num_layers: int, rtt: Dict[str, Dict[str, float]],
                     version: int = 1, search_order: bool = True, payload_bytes: int = 0,
                     bandwidth_mbps=0.0, coord_ms: float = 0.0, hop_mode: str = "overlap",
                     current: Optional[Dict[str, tuple]] = None) -> Plan:
    t0 = time.perf_counter()
    best = None
    if search_order and len(nodes) <= 6:
        orders = (p for k in range(1, len(nodes) + 1) for p in itertools.permutations(nodes, k))
    else:
        orders = iter([tuple(nodes)])

    def one_way(a, b):
        return hop_ms(rtt, a, b)

    def xfer(a, b):
        return xfer_ms(bandwidth_mbps, a, b, payload_bytes)

    feasible = []  # (bottleneck, order, rtts, xfers, first hop)
    for order in orders:
        if sum(min(n.max_layers(), num_layers) for n in order) < num_layers:
            continue
        nxt = [order[i + 1].node_id if i + 1 < len(order) else COORD for i in range(len(order))]
        rtts = [one_way(order[i].node_id, nxt[i]) for i in range(len(order))]
        xfers = [xfer(order[i].node_id, nxt[i]) for i in range(len(order))]
        res = _dp_fixed_order(order, num_layers, rtts, xfers, hop_mode, current)
        if res is None:
            continue
        first = one_way(COORD, order[0].node_id)
        bott = max(res[0], coord_ms + first) if hop_mode == "additive" else res[0]
        feasible.append((bott, order, rtts, xfers, first))
    # second pass: among the orders that reach the best bottleneck, the lowest total latency
    b_star = min((c[0] for c in feasible), default=math.inf)
    for bott, order, rtts, xfers, first in feasible:
        if bott > b_star + 1e-9:
            continue
        bott, lat, cuts = _dp_fixed_order(order, num_layers, rtts, xfers, hop_mode, current, limit=b_star)
        if hop_mode == "additive":
            bott = max(bott, coord_ms + first)
        lat += coord_ms + first
        key = (round(bott, 6), round(lat, 6))
        if best is None or key < best[0]:
            best = (key, order, cuts, rtts, xfers)
    if best is None:
        raise InfeasibleError("cluster memory cannot hold the model" if current is None
                              else "no plan can be staged next to the current one")
    (bott, lat), order, cuts, rtts, xfers = best
    stages = []
    for n, (s, e), r, x in zip(order, cuts, rtts, xfers):
        comp = (e - s) * n.t_layer_ms
        stages.append(Stage(n.node_id, s, e, comp + r if hop_mode == "additive" else max(comp, x)))
    return Plan(version, "latency" if hop_mode == "overlap" else "latency-additive", stages, bott, lat,
                num_layers, (time.perf_counter() - t0) * 1e3,
                meta={"orders_searched": search_order, "hop_mode": hop_mode, "staged": current is not None})


def plan_cost(plan: "Plan", specs: Sequence[NodeSpec], rtt: Dict, hop_mode: str = "overlap",
              payload_bytes: int = 0, bandwidth_mbps=0.0) -> float:
    """Bottleneck of an existing plan under (possibly updated) node specs, same model as the DP."""
    by = {s.node_id: s for s in specs}
    worst = 0.0
    for i, st in enumerate(plan.stages):
        n = by.get(st.node_id)
        if n is None:
            return math.inf
        comp = st.n * n.t_layer_ms
        nxt = plan.stages[i + 1].node_id if i + 1 < len(plan.stages) else COORD
        worst = max(worst, comp + hop_ms(rtt, st.node_id, nxt) if hop_mode == "additive"
                    else max(comp, xfer_ms(bandwidth_mbps, st.node_id, nxt, payload_bytes)))
    return worst


def memory_proportional(nodes: Sequence[NodeSpec], num_layers: int, rtt: Optional[Dict] = None,
                        version: int = 1, payload_bytes: int = 0, bandwidth_mbps=0.0,
                        coord_ms: float = 0.0) -> Plan:
    """B2: split by free memory in join order (what exo / llama.cpp RPC do)."""
    t0 = time.perf_counter()
    rtt = rtt or {}
    mems = np.array([max(n.mem_free - n.overhead, 0) for n in nodes], dtype=float)
    if mems.sum() <= 0:
        raise InfeasibleError("no memory")
    raw = mems / mems.sum() * num_layers
    alloc = np.floor(raw).astype(int)
    alloc = np.maximum(alloc, 1)
    while alloc.sum() > num_layers:
        alloc[int(np.argmax(alloc))] -= 1
    rem = raw - np.floor(raw)
    for i in np.argsort(-rem):
        if alloc.sum() >= num_layers:
            break
        alloc[i] += 1
    caps = [n.max_layers() for n in nodes]
    # spill overflow to whoever still has room, in join order
    over = sum(max(0, a - c) for a, c in zip(alloc, caps))
    alloc = [min(a, c) for a, c in zip(alloc, caps)]
    for i in range(len(alloc)):
        take = min(over, caps[i] - alloc[i])
        alloc[i] += take
        over -= take
    if over or sum(alloc) != num_layers:
        raise InfeasibleError("cluster memory cannot hold the model")
    stages, s = [], 0
    kept = [(n, int(a)) for n, a in zip(nodes, alloc) if a > 0]
    for i, (n, a) in enumerate(kept):
        nxt = kept[i + 1][0].node_id if i + 1 < len(kept) else COORD
        h = hop_ms(rtt, n.node_id, nxt, payload_bytes, bandwidth_mbps)
        stages.append(Stage(n.node_id, s, s + a, a * n.t_layer_ms + h))
        s += a
    first = hop_ms(rtt, COORD, kept[0][0].node_id, payload_bytes, bandwidth_mbps)
    bott = max([st.cost_ms for st in stages] + [coord_ms + first])
    lat = sum(st.cost_ms for st in stages) + coord_ms + first
    return Plan(version, "memory", stages, float(bott), float(lat), num_layers, (time.perf_counter() - t0) * 1e3)


def node_specs_from_profiles(nodes: Dict[str, Dict], planning_batch: int, kv_token_budget: int,
                             overhead_bytes: int, scale: Optional[Dict[str, float]] = None) -> List[NodeSpec]:
    """Build planner inputs from join records {node_id: {"info":..., "profile":...}}.

    ``scale`` multiplies a node's profiled layer time (used by drift replanning).
    """
    out = []
    scale = scale or {}
    for nid, rec in nodes.items():
        prof, info = rec["profile"], rec["info"]
        t = expected_layer_ms(prof, planning_batch, 0) * scale.get(nid, 1.0)
        out.append(NodeSpec(nid, t, int(info["mem_free"]), int(prof["mem_layer"]),
                            int(prof["kv_per_tok_layer"]) * kv_token_budget, overhead_bytes))
    return out


def make_plan(policy: str, specs: Sequence[NodeSpec], num_layers: int, rtt: Dict, version: int,
              **kw) -> Plan:
    if policy == "latency":
        return latency_balanced(specs, num_layers, rtt, version=version, **kw)
    if policy == "latency-additive":  # the proposal's original cost model (hop RTT added to stage time)
        return latency_balanced(specs, num_layers, rtt, version=version, hop_mode="additive", **kw)
    if policy == "latency-fixed":
        p = latency_balanced(specs, num_layers, rtt, version=version, search_order=False, **kw)
        p.policy = "latency-fixed"
        return p
    kw.pop("current", None)
    kw.pop("hop_mode", None)
    if policy == "memory":
        return memory_proportional(specs, num_layers, rtt, version=version, **kw)
    if policy == "even":
        fake = [NodeSpec(s.node_id, s.t_layer_ms, 1, s.mem_layer, s.kv_bytes_per_layer, 0) for s in specs]
        for f in fake:
            f.mem_free = 10 ** 15
        p = memory_proportional(fake, num_layers, rtt, version=version, **kw)
        p.policy = "even"
        return p
    raise ValueError(policy)
