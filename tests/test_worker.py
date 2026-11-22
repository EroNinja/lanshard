"""Worker plan bookkeeping, with the loader and the network stubbed out."""
import threading
import time
import types

from lanshard.partitioner import Plan, Stage
from lanshard.worker import Worker


def _plan(version, end):
    return Plan(version, "manual", [Stage("w", 0, end)], 0.0, 0.0, end).to_dict()


def _worker(load_layers):
    w = Worker("unused:0", node_id="w", device="cpu")
    w.runner = types.SimpleNamespace(layers={})
    w.loader = types.SimpleNamespace(load_layers=load_layers)
    w.ctrl = types.SimpleNamespace(sent=[], send=lambda m: w.ctrl.sent.append(m))
    w._sender = lambda node_id: None
    w._mem = lambda: {}
    return w


def test_release_during_prepare_keeps_the_new_plans_layers():
    """An old plan can finish draining while a newer plan is still being staged. Releasing it must
    not drop the layers the newer plan just loaded, or its first micro-batch fails with a KeyError."""
    loading, proceed = threading.Event(), threading.Event()

    def slow_load(ids, device, dtype):
        loading.set()
        assert proceed.wait(10)
        return {i: object() for i in ids}

    w = _worker(lambda ids, device, dtype: {i: object() for i in ids})
    for v in (1, 2):
        w._do_prepare({"plan": _plan(v, 4), "addrs": {}})
    w.loader.load_layers = slow_load

    prep = threading.Thread(target=w._do_prepare, args=({"plan": _plan(3, 6), "addrs": {}},))
    prep.start()
    assert loading.wait(10)
    rel = threading.Thread(target=w._release, args=(1,))
    rel.start()
    time.sleep(0.2)  # let the release reach the point where it waits for the staging to finish
    proceed.set()
    prep.join(10)
    rel.join(10)

    assert sorted(w.runner.layers) == [0, 1, 2, 3, 4, 5]
    assert sorted(w.plans) == [2, 3]
    assert [m["type"] for m in w.ctrl.sent] == ["ready"] * 3
