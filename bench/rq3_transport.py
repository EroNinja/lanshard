"""RQ3: per-hop latency of gRPC vs raw TCP vs torch.distributed/Gloo.

Ping-pong between two processes (or two hosts) over the exact stage channel
used by the pipeline. For each backend and payload size we record RTT and the
local serialisation / deserialisation time, and derive the per-hop split::

    one_way = RTT / 2,  transfer = one_way - ser - deser

Local:      python bench/rq3_transport.py
Two hosts:  host B: python bench/rq3_transport.py --role echo --bind 0.0.0.0 --port 29600
            host A: python bench/rq3_transport.py --peer <B-ip>:29600
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from common import ROOT, pct, results_dir, save_json

from lanshard import transport as T

SIZES = [8 << 10, 16 << 10, 64 << 10, 1 << 20, 4 << 20, 16 << 20]


def echo_role(backend: str, bind: str, port: int, netem: str):
    """Serve on `port`; first message names the address to answer to."""
    prof = T.NetemProfile.from_spec(netem)
    state = {"back": None}

    def on(h, t):
        if h.get("hello"):
            state["back"] = T.connect(backend, h["hello"], prof, asynchronous=False)
            state["back"].send({"ack": 1})
            return
        if h.get("bye"):
            os._exit(0)
        h = {k: v for k, v in h.items() if not k.startswith("_")}
        state["back"].send(h, t)

    srv = T.serve(backend, bind, port, on)
    print(f"READY {srv.port}", flush=True)
    while True:
        time.sleep(3600)


def client(backend: str, peer: str, sizes, iters_small: int, iters_large: int, netem: str, bind: str):
    prof = T.NetemProfile.from_spec(netem)
    got = threading.Event()
    box = {}

    def on(h, t):
        box["h"], box["t"], box["t_recv"] = h, t, time.perf_counter()
        got.set()

    srv = T.serve(backend, bind, 0, on)
    host = bind if bind not in ("0.0.0.0", "") else T.local_ip()
    out = T.connect(backend, peer, prof, asynchronous=False)
    out.send({"hello": f"{host}:{srv.port}"})
    if not got.wait(60):
        raise TimeoutError("no ack from echo peer")
    rows = []
    for size in sizes:
        n_el = size // 2
        x = torch.randn(n_el).half()
        iters = iters_small if size <= (64 << 10) else iters_large
        for i in range(iters + 5):
            got.clear()
            t0 = time.perf_counter()
            ser_ms = out.send({"seq": i}, x)
            if not got.wait(60):
                raise TimeoutError("lost ping")
            rtt = (box["t_recv"] - t0) * 1e3
            deser = box["h"].get("_deser_ms", 0.0)
            if i >= 5:
                rows.append({"backend": backend, "bytes": size, "iter": i - 5, "rtt_ms": rtt,
                             "ser_ms": ser_ms, "deser_ms": deser})
    try:
        out.send({"bye": 1})
    except Exception:
        pass
    return rows


def summarise(rows):
    out = []
    keys = sorted({(r["backend"], r["bytes"]) for r in rows}, key=lambda k: (k[1], k[0]))
    for b, s in keys:
        rs = [r for r in rows if r["backend"] == b and r["bytes"] == s]
        rtt = [r["rtt_ms"] for r in rs]
        ser = statistics.median(r["ser_ms"] for r in rs)
        de = statistics.median(r["deser_ms"] for r in rs)
        ow = statistics.median(rtt) / 2
        out.append({"backend": b, "bytes": s, "n": len(rs), "rtt_p50": pct(rtt, 50), "rtt_p95": pct(rtt, 95),
                    "rtt_p99": pct(rtt, 99), "one_way_p50": ow, "ser_p50": ser, "deser_p50": de,
                    "transfer_p50": max(0.0, ow - ser - de)})
    # overhead of each backend relative to raw TCP at the same size
    base = {r["bytes"]: r["one_way_p50"] for r in out if r["backend"] == "tcp"}
    for r in out:
        if r["bytes"] in base:
            r["added_vs_tcp_ms"] = r["one_way_p50"] - base[r["bytes"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="client", choices=["client", "echo"])
    ap.add_argument("--backends", default="tcp,grpc,gloo")
    ap.add_argument("--peer", help="host:port of a remote echo (default: spawn locally)")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--iters-large", type=int, default=30)
    ap.add_argument("--netem", default="none")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.role == "echo":
        backend = a.backends.split(",")[0]
        return echo_role(backend, a.bind, a.port, a.netem)
    sizes = [int(s) for s in a.sizes.split(",")]
    rows = []
    for b in a.backends.split(","):
        proc = None
        peer = a.peer
        if not peer:
            env = dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1")
            proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--role", "echo", "--backends", b,
                                     "--bind", "127.0.0.1", "--netem", a.netem], stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, env=env, text=True)
            line = proc.stdout.readline()
            while line and not line.startswith("READY"):
                line = proc.stdout.readline()
            peer = f"127.0.0.1:{line.split()[1]}"
        print(f"[{b}] pinging {peer}", flush=True)
        rows += client(b, peer, sizes, a.iters, a.iters_large, a.netem, a.bind)
        if proc:
            time.sleep(0.3)
            proc.kill()
    summ = summarise(rows)
    out = results_dir("rq3-transport" + (f"-{a.tag}" if a.tag else "") + (f"-{a.netem}" if a.netem != "none" else ""))
    with open(os.path.join(out, "pings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    save_json(os.path.join(out, "summary.json"), {"config": vars(a), "rows": summ})
    print(f"{'backend':6} {'bytes':>9} {'rtt p50':>8} {'p99':>8} {'ser':>7} {'deser':>7} {'xfer':>7} {'+vs tcp':>8}")
    for r in summ:
        print(f"{r['backend']:6} {r['bytes']:>9} {r['rtt_p50']:8.3f} {r['rtt_p99']:8.3f} {r['ser_p50']:7.3f} "
              f"{r['deser_p50']:7.3f} {r['transfer_p50']:7.3f} {r.get('added_vs_tcp_ms', 0):8.3f}")
    print("results:", out)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
