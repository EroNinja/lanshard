"""Measure real Wi-Fi RTT between two laptops and emit a netem profile.

host B: python bench/measure_rtt.py --serve --port 29700
host A: python bench/measure_rtt.py --peer <B-ip>:29700 --n 2000 --out wifi_profile.json

The JSON holds one-way delay samples (RTT/2) and fits; pass it to workers with
``--netem wifi_profile.json`` to reproduce the network in single-host runs, or
convert to a ``tc qdisc ... netem delay Xms Yms`` line on Linux nodes (printed).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from common import pct

from lanshard.worker import EchoServer, measure_rtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=29700)
    ap.add_argument("--peer")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--payload", type=int, default=8192)
    ap.add_argument("--out", default="wifi_profile.json")
    a = ap.parse_args()
    if a.serve:
        EchoServer(a.bind, a.port)
        print(f"echo on {a.bind}:{a.port}", flush=True)
        while True:
            time.sleep(3600)
    r = measure_rtt(a.peer, a.payload, a.n)
    s = r["samples"]
    one_way = [x / 2 for x in s]
    prof = {"samples_ms": one_way, "rtt_p50": pct(s, 50), "rtt_p95": pct(s, 95), "rtt_p99": pct(s, 99),
            "rtt_max": max(s), "payload": a.payload, "n": len(s),
            "base_ms": min(one_way), "jitter_ms": statistics.pstdev(one_way)}
    with open(a.out, "w") as f:
        json.dump(prof, f)
    print(json.dumps({k: v for k, v in prof.items() if k != "samples_ms"}, indent=1))
    print(f"tc qdisc add dev wlan0 root netem delay {pct(one_way, 50):.1f}ms {statistics.pstdev(one_way):.1f}ms distribution normal")


if __name__ == "__main__":
    main()
