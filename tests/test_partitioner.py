import itertools
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lanshard.partitioner import (COORD, InfeasibleError, NodeSpec, hop_ms, latency_balanced,  # noqa: E402
                                  memory_proportional)

GB = 1 << 30


def brute_force(nodes, L, rtt, mode="additive", with_latency=False):
    """Minimum bottleneck (then minimum total latency) over every ordered subset and contiguous split."""
    best = (float("inf"), float("inf"))
    for k in range(1, len(nodes) + 1):
        for order in itertools.permutations(nodes, k):
            for cuts in itertools.combinations(range(1, L), k - 1):
                bounds = (0,) + cuts + (L,)
                costs, ok, lat = [], True, hop_ms(rtt, COORD, order[0].node_id)
                for i, n in enumerate(order):
                    m = bounds[i + 1] - bounds[i]
                    if m > n.max_layers():
                        ok = False
                        break
                    nxt = order[i + 1].node_id if i + 1 < k else COORD
                    comp = m * n.t_layer_ms
                    costs.append(comp + hop_ms(rtt, n.node_id, nxt) if mode == "additive" else comp)
                    lat += comp + hop_ms(rtt, n.node_id, nxt)
                if ok:
                    if mode == "additive":
                        costs.append(hop_ms(rtt, COORD, order[0].node_id))
                    best = min(best, (round(max(costs), 6), round(lat, 6)))
    return best if with_latency else best[0]


def test_dp_is_lexicographically_optimal_on_random_clusters():
    """Lowest bottleneck first, then lowest single-stream latency, against brute force."""
    import random

    rng = random.Random(0)
    for _ in range(60):
        nodes = [NodeSpec(f"n{i}", rng.choice([1, 1.5, 2, 3, 5]), rng.randint(2, 9) * GB, GB)
                 for i in range(rng.randint(2, 4))]
        ids = [n.node_id for n in nodes] + [COORD]
        rtt = {a: {b: rng.choice([2, 4, 8, 20, 60]) for b in ids if b != a} for a in ids}
        L = rng.randint(6, 10)
        for mode in ("additive", "overlap"):
            want = brute_force(nodes, L, rtt, mode, with_latency=True)
            if want[0] == float("inf"):
                with pytest.raises(InfeasibleError):
                    latency_balanced(nodes, L, rtt, hop_mode=mode)
                continue
            p = latency_balanced(nodes, L, rtt, hop_mode=mode)
            assert (p.bottleneck_ms, p.latency_ms) == pytest.approx(want)


def test_dp_matches_brute_force():
    nodes = [NodeSpec("a", 1.0, 10 * GB, GB // 4), NodeSpec("b", 2.0, 10 * GB, GB // 4),
             NodeSpec("c", 3.5, 1 * GB, GB // 4)]
    rtt = {"a": {"b": 4, "c": 10, COORD: 2}, "b": {"a": 4, "c": 6, COORD: 3},
           "c": {"a": 10, "b": 6, COORD: 20}, COORD: {"a": 2, "b": 3, "c": 20}}
    for L in (6, 9, 12):
        for mode in ("additive", "overlap"):
            p = latency_balanced(nodes, L, rtt, hop_mode=mode)
            assert p.bottleneck_ms == pytest.approx(brute_force(nodes, L, rtt, mode))
        assert sum(s.n for s in p.stages) == L
        assert p.stages[0].start == 0 and p.stages[-1].end == L


def test_faster_node_gets_more_layers():
    nodes = [NodeSpec("fast", 1.0, 8 * GB, 100 << 20), NodeSpec("slow", 2.0, 8 * GB, 100 << 20)]
    p = latency_balanced(nodes, 30, {})
    layers = {s.node_id: s.n for s in p.stages}
    assert layers["fast"] == 20 and layers["slow"] == 10


def test_memory_policy_ignores_speed():
    nodes = [NodeSpec("fast", 1.0, 4 * GB, 100 << 20), NodeSpec("slow", 2.0, 4 * GB, 100 << 20)]
    p = memory_proportional(nodes, 30)
    assert [s.n for s in p.stages] == [15, 15]


BADLINK_RTT = {"a": {"b": 2, "c": 2, COORD: 2}, "b": {"a": 2, "c": 2, COORD: 2},
               "c": {"a": 60, "b": 60, COORD: 60}, COORD: {"a": 2, "b": 2, "c": 2}}


def test_additive_model_starves_bad_link():
    nodes = [NodeSpec(n, 1.0, 8 * GB, 100 << 20) for n in ("a", "b", "c")]
    p = latency_balanced(nodes, 24, BADLINK_RTT, hop_mode="additive")
    layers = {s.node_id: s.n for s in p.stages}
    assert layers.get("c", 0) < layers["a"]


def test_overlap_model_balances_compute_and_uses_rtt_as_tiebreak():
    nodes = [NodeSpec(n, 1.0, 8 * GB, 100 << 20) for n in ("a", "b", "c")]
    p = latency_balanced(nodes, 24, BADLINK_RTT)
    assert sorted(s.n for s in p.stages) == [8, 8, 8]
    assert p.bottleneck_ms == pytest.approx(8.0)


def test_staged_replan_fits_next_to_current_plan():
    # a has room for 12 layers and currently holds [0:10); it may grow by at most 2
    nodes = [NodeSpec("a", 1.0, 12 * GB, GB), NodeSpec("b", 3.0, 30 * GB, GB)]
    cur = {"a": (0, 10), "b": (10, 20)}
    p = latency_balanced(nodes, 20, {}, current=cur)
    for st in p.stages:
        n = next(x for x in nodes if x.node_id == st.node_id)
        os_, oe = cur[st.node_id]
        union = len(set(range(os_, oe)) | set(range(st.start, st.end)))
        assert union <= n.max_layers()
    free = latency_balanced(nodes, 20, {})
    assert p.bottleneck_ms >= free.bottleneck_ms


def test_plan_cost_reflects_new_speeds():
    from lanshard.partitioner import plan_cost

    nodes = [NodeSpec("a", 1.0, 8 * GB, 100 << 20), NodeSpec("b", 1.0, 8 * GB, 100 << 20)]
    p = latency_balanced(nodes, 20, {})
    slow = [NodeSpec("a", 3.0, 8 * GB, 100 << 20), nodes[1]]
    assert plan_cost(p, slow, {}) == pytest.approx(30.0)


def test_memory_constraint_respected():
    nodes = [NodeSpec("big", 0.5, 10 * GB, GB), NodeSpec("small", 3.0, 3 * GB, GB)]
    p = latency_balanced(nodes, 12, {})
    for s in p.stages:
        n = next(x for x in nodes if x.node_id == s.node_id)
        assert s.n <= n.max_layers()


def test_infeasible():
    with pytest.raises(InfeasibleError):
        latency_balanced([NodeSpec("a", 1.0, 1 * GB, GB)], 4, {})


def test_kv_reserve_reduces_capacity():
    n = NodeSpec("a", 1.0, 4 * GB, GB // 2, kv_bytes_per_layer=GB // 2)
    assert n.max_layers() == 4


def test_plans_are_json_serialisable():
    import json

    nodes = [NodeSpec("a", 1.0, 4 * GB, 100 << 20), NodeSpec("b", 2.0, 3 * GB, 100 << 20)]
    for p in (memory_proportional(nodes, 28), latency_balanced(nodes, 28, {})):
        json.dumps(p.to_dict())


def test_expected_layer_ms_interpolates():
    from lanshard.probe import expected_layer_ms

    prof = {"t_decode_ms": {"1": 1.0, "2": 2.0, "4": 3.0, "8": 5.0}, "t_prefill_ms_per_tok": 0.1}
    assert expected_layer_ms(prof, 3, 0) == pytest.approx(2.5)
    assert expected_layer_ms(prof, 16, 0) == pytest.approx(9.0)
    assert expected_layer_ms(prof, 0, 10) == pytest.approx(1.0)
