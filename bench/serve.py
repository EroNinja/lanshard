"""RQ1 / B2: replay a ShareGPT-like workload through an emulated cluster.

    python bench/serve.py --preset hetero3 --policy latency --rate 2 --n 60
    python bench/serve.py --preset hetero3 --policy memory  --rate 2 --n 60 --no-drift

Writes results/<stamp>-serve-<preset>-<policy>/ with requests.csv,
microbatches.csv, node_stats.csv, devices.csv, run.json and summary.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

from common import ROOT, build_workload, replay, results_dir, save_json, summarize

from lanshard.cluster import local_cluster, parse_worker_spec
from lanshard.coordinator import DriftConfig, SchedulerConfig
from clusters import PRESETS


def run(args) -> dict:
    workers = PRESETS[args.preset] if args.preset else [parse_worker_spec(w) for w in args.worker]
    name = f"serve-{args.preset or 'custom'}-{args.policy}-{args.transport}" + (f"-r{args.repeat}" if args.repeat is not None else "")
    out = args.out or results_dir(name)
    sched = SchedulerConfig(kv_token_budget=args.kv_budget, prefill_chunk=args.prefill_chunk,
                            depth=args.depth, max_decode_mb=args.max_decode_mb)
    drift = DriftConfig(enabled=not args.no_drift)
    with local_cluster(args.model, workers, args.transport, args.policy, sched=sched, drift=drift,
                       coord_netem=args.coord_netem, log_dir=os.path.join(out, "logs"),
                       profile_budget=args.profile_budget, overhead_gb=args.overhead_gb) as coord:
        wl = build_workload(coord.tokenizer, args.n, args.seed, args.sharegpt,
                            prompt_range=(args.min_prompt, args.max_prompt), out_range=(args.min_out, args.max_out),
                            prompt_median=args.prompt_median, out_median=args.out_median)
        if args.warmup:
            replay(coord, wl[: args.warmup], float("inf"), args.seed)
        coord.reset_metrics()
        import time

        time.sleep(1.2)
        wall = replay(coord, wl, args.rate, args.seed)
        time.sleep(1.5)  # let final node stats arrive
        summary = summarize(coord, wall)
        summary.update(config=vars(args), workers=workers)
        coord.export(out, {"summary": summary})
        save_json(os.path.join(out, "summary.json"), summary)
    print(json.dumps({k: summary[k] for k in ("throughput_tok_s", "tpot_ms", "ttft_ms", "plan", "mean_bubble_ratio")},
                     indent=1))
    print("results:", out)
    return summary


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--worker", action="append", default=[])
    ap.add_argument("--policy", default="latency", choices=["latency", "latency-additive", "latency-fixed", "memory", "even"])
    ap.add_argument("--transport", default="tcp")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--coord-netem", default="wifi")
    ap.add_argument("--rate", type=float, default=2.0, help="request arrival rate (req/s); inf = burst")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sharegpt", help="path to a ShareGPT JSON dump (default: ShareGPT-like synthetic lengths)")
    ap.add_argument("--min-prompt", type=int, default=100)
    ap.add_argument("--max-prompt", type=int, default=2000)
    ap.add_argument("--prompt-median", type=float, default=350)
    ap.add_argument("--min-out", type=int, default=50)
    ap.add_argument("--max-out", type=int, default=500)
    ap.add_argument("--out-median", type=float, default=180)
    ap.add_argument("--kv-budget", type=int, default=16384)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--max-decode-mb", type=int, default=16)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--profile-budget", type=float, default=8.0)
    ap.add_argument("--overhead-gb", type=float, default=0.3, help="per-node fixed memory reserve")
    ap.add_argument("--repeat", type=int)
    ap.add_argument("--out")
    return ap


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [coord] %(levelname)s %(message)s")
    run(parser().parse_args())
