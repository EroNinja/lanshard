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


def _workers():
    """Two workers that hold at most 16 of the 24 layers each, whatever dtype this machine computes in
    (FP16 on a GPU, FP32 on CPU; a fixed cap in GB only fits one of them)."""
    from lanshard.model_utils import LayerLoader, default_device, default_dtype

    loader, dtype = LayerLoader(MODEL), default_dtype(default_device())
    per = loader.layer_param_bytes(0, dtype) + loader.geom.kv_bytes_per_token_per_layer(dtype) * 16384
    limit = 0.3 + 16.5 * per / (1 << 30)
    wire = str(dtype).replace("torch.", "")  # lossless hops, so the cut positions cannot change the output
    return [{"node_id": n, "mem_limit_gb": limit, "wire_dtype": wire} for n in ("w1", "w2")], wire


def _run(mode: str):
    from lanshard.cluster import local_cluster
    from lanshard.coordinator import DriftConfig
    from lanshard.partitioner import Plan, Stage

    workers, wire = _workers()
    with local_cluster(MODEL, workers, drift=DriftConfig(enabled=False), coord_netem=None, profile_budget=2,
                       overhead_gb=0.3, wire_dtype=wire) as c:
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
