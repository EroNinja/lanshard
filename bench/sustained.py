"""RQ2: does drift-triggered repartitioning hold throughput when a laptop throttles?

Closed-loop load (``--concurrency`` clients each resubmitting when done) for
``--duration`` seconds. At ``--throttle-at`` the chosen node is slowed by
``--factor`` (wall-clock, as a thermally throttled GPU would be). Run once with
drift handling on and once off; compare throughput timelines and measure the
transition cost (seconds of throughput lost between trigger and new steady state).

    python bench/sustained.py --duration 1800 --throttle-at 300     # proposal's 30-minute run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import threading
import time

from common import build_workload, results_dir, save_json, summarize, throughput_series

from lanshard.cluster import local_cluster
from lanshard.coordinator import DriftConfig, SchedulerConfig
from clusters import PRESETS


def closed_loop(coord, workload, concurrency, stop_at, seed=0):
    rng = random.Random(seed)
    done = threading.Event()

    def client(i):
        while time.time() < stop_at:
            w = workload[rng.randrange(len(workload))]
            r = coord.submit(w["prompt_ids"], max_new=w["max_new"], ignore_eos=True)
            r.future.result()

    ts = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=max(0, stop_at - time.time()) + 600)


def transition_cost(times, tput, t_throttle, events, bin_s):
    """Throughput-seconds lost after the throttle, relative to the final steady state."""
    post = [v for t, v in zip(times, tput) if t >= times[-1] * 0.8 and t < times[-1] - bin_s]
    steady = statistics.mean(post) if post else 0.0
    switch = next((e["t"] for e in events if e["event"] == "switch" and e.get("old")), None)
    end = (switch + 30) if switch else times[-1]
    lost = sum(max(0.0, steady - v) * bin_s for t, v in zip(times, tput) if t_throttle <= t < end)
    return {"steady_after_tok_s": steady, "switch_t": switch,
            "lost_token_s": lost, "degraded_s_equiv": lost / steady if steady else None}


def run_once(args, drift_on: bool, out_root: str):
    tag = ("drift-on" + ("" if args.migrate == "recompute" else "-drain")) if drift_on else "drift-off"
    out = os.path.join(out_root, tag)
    sched = SchedulerConfig(kv_token_budget=args.kv_budget)
    drift = DriftConfig(enabled=drift_on, threshold=args.threshold, sustain_s=args.sustain, calib_s=args.calib,
                        cooldown_s=args.sustain, migrate=args.migrate)
    with local_cluster(args.model, PRESETS[args.preset], args.transport, "latency", sched=sched, drift=drift,
                       coord_netem=args.coord_netem, log_dir=os.path.join(out, "logs"),
                       profile_budget=args.profile_budget, overhead_gb=0.3) as c:
        wl = build_workload(c.tokenizer, 200, args.seed, prompt_range=(100, args.max_prompt),
                            out_range=(50, args.max_out), prompt_median=args.prompt_median, out_median=args.out_median)
        c.reset_metrics()
        t0 = time.time()
        c.t_start = t0
        stop_at = t0 + args.duration

        def throttle():
            time.sleep(args.throttle_at)
            c.set_slowdown(args.node, args.factor)
            if args.recover_at:
                time.sleep(args.recover_at - args.throttle_at)
                c.set_slowdown(args.node, 1.0)

        threading.Thread(target=throttle, daemon=True).start()
        closed_loop(c, wl, args.concurrency, stop_at, args.seed)
        wall = time.time() - t0
        time.sleep(1.5)
        c.export(out)
        summ = summarize(c, wall)
        toks = [t - t0 for t in c.token_log]
        times, tput = throughput_series(toks, args.bin, args.duration)
        pre = [v for t, v in zip(times, tput) if 0.4 * args.throttle_at <= t < args.throttle_at]
        post = [v for t, v in zip(times, tput) if args.throttle_at + args.sustain + 30 <= t < args.duration - args.bin]
        summ.update(tag=tag, throughput_pre=statistics.mean(pre) if pre else None,
                    throughput_post=statistics.mean(post) if post else None,
                    transition=transition_cost(times, tput, args.throttle_at, c.events, args.bin),
                    series={"t": times, "tok_s": tput}, events=c.events, config=vars(args))
        c.export(out, {"summary": {k: v for k, v in summ.items() if k != "series"}})
        save_json(os.path.join(out, "summary.json"), summ)
    print(tag, json.dumps({k: summ[k] for k in ("throughput_pre", "throughput_post", "transition", "plan")}, indent=1),
          flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--preset", default="hetero3")
    ap.add_argument("--transport", default="tcp")
    ap.add_argument("--coord-netem", default="wifi")
    ap.add_argument("--duration", type=float, default=1800)
    ap.add_argument("--throttle-at", type=float, default=300)
    ap.add_argument("--recover-at", type=float, default=0)
    ap.add_argument("--node", default="gpu-fast")
    ap.add_argument("--factor", type=float, default=2.5)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--sustain", type=float, default=30)
    ap.add_argument("--calib", type=float, default=15)
    ap.add_argument("--bin", type=float, default=5)
    ap.add_argument("--kv-budget", type=int, default=16384)
    ap.add_argument("--max-prompt", type=int, default=1000)
    ap.add_argument("--prompt-median", type=float, default=300)
    ap.add_argument("--max-out", type=int, default=300)
    ap.add_argument("--out-median", type=float, default=120)
    ap.add_argument("--profile-budget", type=float, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="on,off")
    ap.add_argument("--migrate", default="recompute", choices=["recompute", "drain"])
    a = ap.parse_args()
    out = results_dir(f"rq2-sustained-{int(a.duration)}s")
    res = {}
    for m in a.modes.split(","):
        res[m] = run_once(a, m == "on", out)
    save_json(os.path.join(out, "comparison.json"), {m: {k: v for k, v in r.items() if k not in ("series",)}
                                                      for m, r in res.items()})
    print("results:", out)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [coord] %(levelname)s %(message)s")
    main()
