import threading
import time

import pytest
import torch

from lanshard import transport as T


@pytest.mark.parametrize("backend", T.BACKENDS)
def test_roundtrip(backend):
    got, ev = [], threading.Event()

    def on(h, t):
        got.append((h, t))
        if len(got) == 3:
            ev.set()

    srv = T.serve(backend, "127.0.0.1", 0, on)
    s = T.connect(backend, f"127.0.0.1:{srv.port}")
    a = torch.randn(5, 4096).half()
    b = torch.randn(2, 3).to(torch.bfloat16)
    try:
        s.send({"i": 1}, a)
        s.send({"i": 2}, b)
        s.send({"i": 3})
        assert ev.wait(30)
    finally:  # an open gRPC stream keeps the interpreter alive after the tests finish
        s.close()
        srv.close()
    assert [h["i"] for h, _ in got] == [1, 2, 3]  # FIFO
    assert torch.equal(got[0][1], a)
    assert torch.equal(got[1][1], b)
    assert got[2][1] is None
    assert "_deser_ms" in got[0][0]


def test_netem_delay_and_order():
    got, ev = [], threading.Event()

    def on(h, t):
        got.append((h["i"], time.perf_counter()))
        if len(got) == 20:
            ev.set()

    srv = T.serve("tcp", "127.0.0.1", 0, on)
    prof = T.NetemProfile(base_ms=20, jitter_ms=10, seed=1)
    s = T.connect("tcp", f"127.0.0.1:{srv.port}", prof)
    t0 = time.perf_counter()
    for i in range(20):
        s.send({"i": i}, torch.zeros(4))
    assert ev.wait(30)
    assert [i for i, _ in got] == list(range(20))
    assert got[0][1] - t0 >= 0.019


def test_netem_spec_parsing():
    p = T.NetemProfile.from_spec("base=3,jitter=2,spike_prob=0.1,spike=30,rate=400")
    assert p.base_ms == 3 and p.jitter_ms == 2 and p.spike_ms == 30 and p.rate_mbps == 400
    assert T.NetemProfile.from_spec("wifi").active
    assert not T.NetemProfile.from_spec("none").active


def test_measure_link_recovers_the_bandwidth():
    from lanshard.worker import EchoServer, measure_link

    echo = EchoServer("127.0.0.1")
    addr = f"127.0.0.1:{echo.port}"
    m = measure_link(addr, 8192, 10, T.NetemProfile(base_ms=2, rate_mbps=20, seed=0))
    assert m["mbps"] == pytest.approx(20, rel=0.25)
    assert m["rtt"] > 4
    fast = measure_link(addr, 8192, 10)["mbps"]
    assert fast is None or fast > 1000  # loopback: no limit worth modelling
    echo.lsock.close()
