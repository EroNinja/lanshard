"""End-to-end on a 2-worker local cluster (slow, ~2 min; needs the 0.5B weights cached).

Forced mid-generation replans must not change the output:
* a stageable replan migrates in-flight requests to the new plan (re-prefill);
* a non-stageable replan (old + new layers do not fit) pauses admission, requeues long
  requests with their text so far, swaps the layers, and resumes them on the new plan.
Both must be token-identical to the run with no replan. One request is used so that micro-batch
composition (which changes FP16 rounding) is the same in every run; with several concurrent
requests even two replan-free runs can differ after a few dozen tokens.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_model import MODEL, _have_model  # noqa: E402

pytestmark = pytest.mark.skipif(not _have_model() or os.environ.get("LANSHARD_SKIP_SLOW"),
                                reason="model not cached or slow tests disabled")

QS = ["Explain pipeline parallelism in detail, with an example."]
# 0.5B: ~28 MB/layer + 8 MB KV reserve/layer; 0.3 GB overhead -> each worker holds at most 16 layers
WORKERS = [{"node_id": "w1", "mem_limit_gb": 0.88}, {"node_id": "w2", "mem_limit_gb": 0.88}]


def _run(mode: str):
    from lanshard.cluster import local_cluster
    from lanshard.coordinator import DriftConfig
    from lanshard.partitioner import Plan, Stage

    with local_cluster(MODEL, WORKERS, drift=DriftConfig(enabled=False), coord_netem=None, profile_budget=2,
                       overhead_gb=0.3) as c:
        reqs = [c.submit(c.encode_chat([{"role": "user", "content": q}]), max_new=200, ignore_eos=True) for q in QS]
        events = []
        if mode != "none":
            time.sleep(3.0)
            cur = c.plans[c.active_version]
            if mode == "staged":
                c.scale["w1"] = 3.0
                plan = c.compute_plan(current=cur)
            else:  # swap the halves: neither worker can hold old + new layers at once
                with c.lock:
                    v = c.next_version
                    c.next_version += 1
                plan = Plan(v, "manual", [Stage(s.node_id, t.start, t.end) for s, t in
                                          zip(reversed(cur.stages), cur.stages)], 0.0, 0.0, cur.num_layers)
            c.install_plan(plan)
            events = [e["event"] for e in c.events]
        outs = [r.future.result(timeout=600) for r in reqs]
        return outs, [r.migrations for r in reqs], events


@pytest.fixture(scope="module")
def baseline():
    return _run("none")[0]


def test_staged_replan_with_migration_is_token_identical(baseline):
    outs, migrations, events = _run("staged")
    assert "migrate" in events and sum(migrations) > 0
    assert outs == baseline


def test_paused_replan_with_requeue_is_token_identical(baseline):
    outs, migrations, events = _run("paused")
    assert "staging_oom_fallback" in events and "requeue_for_switch" in events
    assert sum(migrations) > 0
    assert outs == baseline
