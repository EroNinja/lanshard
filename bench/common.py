"""Shared helpers for experiments: workloads, replay, summaries, result folders."""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

_CORPUS = (
    "Pipeline parallelism splits a neural network into consecutive stages that run on different devices. "
    "Each stage receives activations from its predecessor, applies its layers, and forwards the result. "
    "Consumer laptops differ widely in compute even when their memory looks the same, and they throttle "
    "under sustained load. Wireless networks add latency that varies from packet to packet. "
    "A scheduler that overlaps many requests can keep every stage busy despite these delays. "
    "The quick brown fox jumps over the lazy dog while the committee reviews the quarterly budget. "
    "Historians still debate the causes of the war, but most agree that trade disputes played a role. "
    "To bake bread, combine flour, water, salt and yeast, knead the dough, let it rise, then bake it. "
    "The derivative of a product is the first function times the derivative of the second plus the reverse. "
)


def results_dir(name: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = os.path.join(ROOT, "results", f"{stamp}-{name}")
    os.makedirs(d, exist_ok=True)
    return d


def sharegpt_like_lengths(n: int, seed: int = 0, prompt_range=(100, 2000), out_range=(50, 500),
                          prompt_median: float = 350, out_median: float = 180) -> List[tuple]:
    """Log-normal lengths clipped to the proposal's ranges (ShareGPT is roughly log-normal)."""
    rng = np.random.default_rng(seed)
    p = np.clip(rng.lognormal(math.log(prompt_median), 0.8, n), *prompt_range).astype(int)
    o = np.clip(rng.lognormal(math.log(out_median), 0.6, n), *out_range).astype(int)
    return list(zip(p.tolist(), o.tolist()))


def load_sharegpt(path: str, n: int, seed: int = 0) -> List[tuple]:
    """(prompt_text, output_len_in_chars) pairs from a ShareGPT JSON dump."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(data)
    out = []
    for conv in data:
        turns = conv.get("conversations", [])
        if len(turns) >= 2 and turns[0].get("from") == "human" and turns[1].get("from") == "gpt":
            out.append((turns[0]["value"], turns[1]["value"]))
        if len(out) >= n:
            break
    return out


def build_workload(tokenizer, n: int, seed: int = 0, sharegpt: Optional[str] = None,
                   prompt_range=(100, 2000), out_range=(50, 500), prompt_median=350, out_median=180) -> List[Dict]:
    """List of {prompt_ids, max_new} with real token ids and controlled lengths."""
    corpus_ids = tokenizer(_CORPUS * 40)["input_ids"]
    items = []
    if sharegpt:
        for text, answer in load_sharegpt(sharegpt, n, seed):
            ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                tokenize=True)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)[: prompt_range[1]]
            out = int(np.clip(len(tokenizer(answer)["input_ids"]), *out_range))
            items.append({"prompt_ids": ids, "max_new": out})
        return items
    rng = random.Random(seed)
    for plen, olen in sharegpt_like_lengths(n, seed, prompt_range, out_range, prompt_median, out_median):
        start = rng.randrange(0, len(corpus_ids) - plen - 1)
        items.append({"prompt_ids": corpus_ids[start:start + plen], "max_new": olen})
    return items


def replay(coord, workload: List[Dict], rate: float, seed: int = 0, ignore_eos: bool = True,
           timeout: float = 3600.0) -> float:
    """Submit requests with Poisson arrivals at ``rate`` req/s (inf = all at once); wait for all."""
    rng = random.Random(seed)
    reqs = []
    t0 = time.time()
    next_t = t0
    for item in workload:
        if rate != float("inf"):
            next_t += rng.expovariate(rate)
            d = next_t - time.time()
            if d > 0:
                time.sleep(d)
        reqs.append(coord.submit(item["prompt_ids"], max_new=item["max_new"], ignore_eos=ignore_eos))
    for r in reqs:
        r.future.result(timeout=max(1.0, timeout - (time.time() - t0)))
    return time.time() - t0


def pct(xs: Sequence[float], q: float) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def summarize(coord, wall_s: float) -> Dict:
    reqs = coord.req_records
    gen = sum(r["gen_len"] for r in reqs)
    tpots = [r["tpot_ms"] for r in reqs if r["tpot_ms"] is not None]
    ttfts = [r["ttft_ms"] for r in reqs if r["ttft_ms"] is not None]
    itls = [x for r in reqs for x in r["itl_ms"]]
    mbs = coord.mb_records
    dec = [m for m in mbs if m["kind"] == "decode"]
    nodes = {}
    for n in coord.nodes.values():
        st = n.stats or {}
        if st.get("wall_s"):
            nodes[n.node_id] = {"busy_frac": st["busy_s"] / st["wall_s"],
                                "bubble_ratio": 1 - st["busy_s"] / st["wall_s"],
                                "peak_alloc_gb": (st["mem"].get("peak_alloc") or 0) / 2 ** 30,
                                "rss_gb": st["mem"]["rss"] / 2 ** 30, "n_layers": len(st["layers"])}
    plan = coord.plans.get(coord.active_version)
    return {
        "n_requests": len(reqs), "wall_s": wall_s, "gen_tokens": gen,
        "throughput_tok_s": gen / wall_s if wall_s else None,
        "per_request_tok_s": statistics.mean(r["gen_len"] / (r["e2e_ms"] / 1e3) for r in reqs) if reqs else None,
        "ttft_ms": {"p50": pct(ttfts, 50), "p95": pct(ttfts, 95), "p99": pct(ttfts, 99)},
        "tpot_ms": {"p50": pct(tpots, 50), "p95": pct(tpots, 95), "p99": pct(tpots, 99)},
        "itl_ms": {"p50": pct(itls, 50), "p95": pct(itls, 95), "p99": pct(itls, 99)},
        "decode_mb": {"n": len(dec), "mean_size": statistics.mean(m["n_seg"] for m in dec) if dec else None,
                      "loop_ms_p50": pct([m["loop_ms"] for m in dec], 50),
                      "compute_ms_p50": pct([m["compute_ms"] for m in dec], 50),
                      "network_ms_p50": pct([m["network_ms"] for m in dec], 50),
                      "deser_ms_p50": pct([m["deser_ms"] for m in dec], 50),
                      "queue_ms_p50": pct([m["queue_ms"] for m in dec], 50)},
        "nodes": nodes,
        "plan": plan.summary() if plan else None,
        "mean_bubble_ratio": statistics.mean(v["bubble_ratio"] for v in nodes.values()) if nodes else None,
    }


def throughput_series(token_times: Sequence[float], bin_s: float = 5.0, t_end: Optional[float] = None):
    if not token_times:
        return [], []
    t_end = t_end or max(token_times)
    bins = int(math.ceil(t_end / bin_s))
    counts = [0] * bins
    for t in token_times:
        if 0 <= t < bins * bin_s:
            counts[int(t // bin_s)] += 1
    return [i * bin_s for i in range(bins)], [c / bin_s for c in counts]


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
