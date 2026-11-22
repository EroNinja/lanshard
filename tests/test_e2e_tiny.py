"""End-to-end tests on a tiny random-weight model: real worker processes, real transports, no download.

Everything runs in FP32 on CPU with one request, so every plan must give exactly the tokens that
stock ``transformers`` gives, including across mid-generation re-splits.
"""

import signal
import threading

import pytest
import torch

from lanshard.model_utils import HeadModules, LayerLoader, Segment, StageRunner
from tiny_model import LAYERS

PROMPT = list(range(5, 45))
MAX_NEW = 400
KV_BUDGET = 4096
OVERHEAD_GB = 0.01


def _workers(model, cap_layers, n=2):
    """Worker specs whose advertised memory holds exactly ``cap_layers`` layers."""
    g = LayerLoader(model)
    per = g.layer_param_bytes(0, torch.float32) + g.geom.kv_bytes_per_token_per_layer(torch.float32) * KV_BUDGET
    limit = OVERHEAD_GB + (cap_layers + 0.5) * per / (1 << 30)
    # FP32 on the wire: the default FP16 wire rounds FP32 activations at every hop, so the output
    # would depend on where the cuts are
    return [{"node_id": f"w{i + 1}", "device": "cpu", "mem_limit_gb": limit, "wire_dtype": "float32"} for i in range(n)]


def _cluster(model, workers, **kw):
    from lanshard.cluster import local_cluster
    from lanshard.coordinator import DriftConfig, SchedulerConfig

    kw.setdefault("drift", DriftConfig(enabled=False))
    return local_cluster(
        model,
        workers,
        sched=SchedulerConfig(kv_token_budget=KV_BUDGET),
        coord_netem=None,
        profile_budget=1,
        overhead_gb=OVERHEAD_GB,
        device="cpu",
        wire_dtype="float32",
        **kw,
    )


def _submit_and_wait_for(c, n_tokens, max_new=MAX_NEW):
    """Submit the prompt and return once ``n_tokens`` have been generated (generation continues)."""
    seen, reached = [], threading.Event()

    def on_token(tok):
        seen.append(tok)
        if len(seen) >= n_tokens:
            reached.set()

    r = c.submit(PROMPT, max_new=max_new, ignore_eos=True, on_token=on_token)
    assert reached.wait(120)
    return r


@pytest.fixture(scope="module")
def reference(tiny_model):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=torch.float32).eval()
    out = hf.generate(
        torch.tensor([PROMPT]),
        max_new_tokens=MAX_NEW,
        min_new_tokens=MAX_NEW,
        do_sample=False,
        attention_mask=torch.ones(1, len(PROMPT), dtype=torch.long),
    )
    return out[0, len(PROMPT) :].tolist()


def test_two_stage_chunked_prefill_matches_hf(tiny_model, reference):
    L = LayerLoader(tiny_model)
    head = HeadModules(tiny_model, "cpu", torch.float32)
    r1, r2 = StageRunner(L.cfg, "cpu", torch.float32), StageRunner(L.cfg, "cpu", torch.float32)
    a, b = range(0, 3), range(3, LAYERS)
    r1.layers, r2.layers = L.load_layers(a, "cpu", torch.float32), L.load_layers(b, "cpu", torch.float32)
    P, cut = len(PROMPT), 17
    for start, chunk in ((0, PROMPT[:cut]), (cut, PROMPT[cut:])):
        seg = [Segment(7, start, len(chunk), P + 64)]
        x = r2.forward(r1.forward(head.embed_tokens(torch.tensor(chunk)), seg, a), seg, b)
    out = [int(head.logits(x[-1:]).argmax())]
    while len(out) < 48:
        seg = [Segment(7, P + len(out) - 1, 1, P + 64)]
        x = r2.forward(r1.forward(head.embed_tokens(torch.tensor([out[-1]])), seg, a), seg, b)
        out.append(int(head.logits(x).argmax()))
    assert out == reference[:48]


@pytest.mark.parametrize("transport", ["tcp", "grpc"])
def test_cluster_matches_hf(tiny_model, reference, transport):
    with _cluster(tiny_model, _workers(tiny_model, 6), transport=transport) as c:
        assert len(c.plans[c.active_version].stages) == 2  # the model fits on no single worker
        assert c.generate(PROMPT, max_new=MAX_NEW, ignore_eos=True) == reference


def test_concurrent_requests_all_finish(tiny_model, reference):
    with _cluster(tiny_model, _workers(tiny_model, 6)) as c:
        reqs = [c.submit(PROMPT[: 10 + 3 * i], max_new=40 + i, ignore_eos=True) for i in range(8)]
        outs = [r.future.result(timeout=120) for r in reqs]
        assert [len(o) for o in outs] == [40 + i for i in range(8)]
        assert any(m["n_seg"] > 1 for m in c.mb_records)  # packed micro-batches were used


def test_staged_replan_with_migration_is_token_identical(tiny_model, reference):
    with _cluster(tiny_model, _workers(tiny_model, 6)) as c:
        r = _submit_and_wait_for(c, 30)
        cur = c.plans[c.active_version]
        c.scale[cur.stages[0].node_id] = 3.0
        plan = c.compute_plan(current=cur)
        assert [(s.node_id, s.n) for s in plan.stages] != [(s.node_id, s.n) for s in cur.stages]
        assert c.install_plan(plan)
        assert r.future.result(timeout=120) == reference
        events = [e["event"] for e in c.events]
        assert "migrate" in events and "staging_oom_fallback" not in events and r.migrations > 0


def test_paused_replan_with_requeue_is_token_identical(tiny_model, reference):
    from lanshard.partitioner import Plan, Stage

    with _cluster(tiny_model, _workers(tiny_model, 6)) as c:
        r = _submit_and_wait_for(c, 30)
        cur = c.plans[c.active_version]
        with c.lock:
            v = c.next_version
            c.next_version += 1
        # swap the two halves: neither worker can hold old + new layers at once
        a, b = cur.stages
        plan = Plan(v, "manual", [Stage(b.node_id, 0, b.n), Stage(a.node_id, b.n, LAYERS)], 0.0, 0.0, LAYERS)
        assert c.install_plan(plan)
        assert r.future.result(timeout=120) == reference
        events = [e["event"] for e in c.events]
        assert "staging_oom_fallback" in events and "requeue_for_switch" in events and r.migrations > 0


def test_migration_tolerates_a_late_kv_free(tiny_model, reference):
    """Control messages (free) and micro-batches travel on different sockets, so a free for a
    migrated request can reach a worker after that request's re-prefill on the new plan."""
    with _cluster(tiny_model, _workers(tiny_model, 6)) as c:
        send_free = c._send_free
        c._send_free = lambda v, ids: threading.Timer(0.3, send_free, (v, ids)).start()
        r = _submit_and_wait_for(c, 30)
        cur = c.plans[c.active_version]
        c.scale[cur.stages[0].node_id] = 3.0
        assert c.install_plan(c.compute_plan(current=cur))
        assert r.future.result(timeout=120) == reference
        assert r.migrations > 0


# ---------------------------------------------------------------------------- failures
def _first_stage_node(c):
    return c.plans[c.active_version].stages[0].node_id


def test_killed_worker_is_replaced_and_output_is_identical(tiny_model, reference):
    with _cluster(tiny_model, _workers(tiny_model, 6, n=3)) as c:
        r = _submit_and_wait_for(c, 30)
        victim = _first_stage_node(c)
        c.worker_procs[victim].kill()
        assert r.future.result(timeout=120) == reference
        assert r.restarts > 0 and victim not in [s.node_id for s in c.plans[c.active_version].stages]


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="needs SIGSTOP")
def test_frozen_worker_is_detected_by_heartbeat(tiny_model, reference):
    """A suspended laptop keeps its sockets open, so only missing heartbeats reveal the loss."""
    with _cluster(tiny_model, _workers(tiny_model, 6, n=3), heartbeat_timeout_s=3.0) as c:
        r = _submit_and_wait_for(c, 30)
        victim = _first_stage_node(c)
        c.worker_procs[victim].send_signal(signal.SIGSTOP)
        assert r.future.result(timeout=120) == reference
        assert "heartbeat_timeout" in [e["event"] for e in c.events]


def test_node_loss_during_a_replan_is_not_dropped(tiny_model, reference):
    with _cluster(tiny_model, _workers(tiny_model, 6, n=3)) as c:
        r = _submit_and_wait_for(c, 30)
        with c.cv:
            c.replanning = True  # as if a drift replan were in progress
        c.worker_procs[_first_stage_node(c)].kill()
        c._wait(lambda: "node_lost" in [e["event"] for e in c.events], 30, "node loss")
        with c.cv:
            c.replanning = False
            c.cv.notify_all()
        assert r.future.result(timeout=120) == reference


def test_requests_fail_when_the_survivors_cannot_hold_the_model(tiny_model):
    with _cluster(tiny_model, _workers(tiny_model, 6, n=2)) as c:
        r = _submit_and_wait_for(c, 30)
        c.worker_procs[_first_stage_node(c)].kill()
        with pytest.raises(RuntimeError, match="cannot hold the model"):
            r.future.result(timeout=60)
        with pytest.raises(RuntimeError, match="cannot hold the model"):
            c.submit(PROMPT, max_new=4)


def test_worker_lost_before_the_first_plan_does_not_poison_the_cluster(tiny_model, reference):
    with _cluster(tiny_model, _workers(tiny_model, 6, n=3), bootstrap=False) as c:
        c._wait(lambda: len(c.alive_nodes()) == 3, 60, "workers to join")
        c.worker_procs["w3"].kill()
        c._wait(lambda: len(c.alive_nodes()) == 2, 30, "node loss")
        c.bootstrap(2)
        assert c.generate(PROMPT, max_new=MAX_NEW, ignore_eos=True) == reference
