"""Run the full experiment suite used in the report, then plot.

    python bench/run_all.py --quick      # ~1 h on one laptop (what produced the committed results)
    python bench/run_all.py              # proposal-scale: 5 repeats, 1000 MMLU, 500 GSM8K, 30-min RQ2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--skip", default="", help="comma list of: rq3,rq1,rq2,quality,b1")
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    q = a.quick
    if "rq3" not in skip:
        sh("rq3_transport.py", "--iters", "300" if q else "1000")
        sh("rq3_transport.py", "--netem", "wifi", "--iters", "200" if q else "1000", "--iters-large", "15")
    if "b1" not in skip:
        sh("b1_single.py", "--model", a.model, "--n", "10" if q else "50", "--max-prompt", "1000", "--max-out", "300",
           "--out-median", "120", "--prompt-median", "300")
    if "rq1" not in skip:
        repeats = 3 if q else 5
        for preset in (["hetero3", "badlink3"] if q else ["hetero3", "badlink3", "hetero4"]):
            for r in range(repeats):
                for pol in ("memory", "latency"):
                    args = ["serve.py", "--model", a.model, "--preset", preset, "--policy", pol, "--repeat", str(r),
                            "--seed", str(r), "--n", "40" if q else "200", "--rate", "1.5",
                            "--max-prompt", "1000", "--max-out", "300", "--out-median", "120", "--prompt-median", "300"]
                    # RQ1 isolates partitioning (I1+I2 vs B2); drift handling is evaluated in RQ2
                    args.append("--no-drift")
                    sh(*args)
    if "rq2" not in skip:
        sh("sustained.py", "--model", a.model, "--duration", "420" if q else "1800", "--throttle-at", "120" if q else "300")
    if "quality" not in skip:
        sh("quality.py", "--model", a.model, "--exact", "50", "--mmlu", "300" if q else "1000",
           "--gsm8k", "60" if q else "500")
    sh("plot.py")


if __name__ == "__main__":
    main()
