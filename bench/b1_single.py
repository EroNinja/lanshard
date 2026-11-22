"""B1: single-machine Hugging Face Transformers on the best laptop.

Gives the quality ground truth (greedy outputs) and an honest latency ceiling
(single-stream TPOT with no network hops). Runs the same workload as
``serve.py`` one request at a time, the way a single-user local setup would.

    python bench/b1_single.py --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
from common import build_workload, pct, results_dir, save_json

from lanshard.model_utils import resolve_model_dir


def load_hf(model: str, device: str = "cuda", dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = resolve_model_dir(model)
    tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForCausalLM.from_pretrained(d, dtype=dtype).to(device).eval()
    return tok, m


@torch.no_grad()
def greedy(m, ids, max_new: int, ignore_eos: bool = False):
    """Plain greedy decoding with per-token timestamps (no repetition penalty or sampling)."""
    dev = next(m.parameters()).device
    x = torch.tensor([ids], device=dev)
    kw = dict(max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0, top_k=None, top_p=None,
              temperature=None)
    if ignore_eos:
        kw.update(min_new_tokens=max_new)
    times = []

    from transformers import StoppingCriteria, StoppingCriteriaList

    class Clock(StoppingCriteria):
        def __call__(self, input_ids, scores, **k):
            torch.cuda.synchronize() if dev.type == "cuda" else None
            times.append(time.time())
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    t0 = time.time()
    out = m.generate(x, stopping_criteria=StoppingCriteriaList([Clock()]), **kw)
    return out[0, len(ids):].tolist(), t0, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prompt", type=int, default=2000)
    ap.add_argument("--prompt-median", type=float, default=350)
    ap.add_argument("--max-out", type=int, default=500)
    ap.add_argument("--out-median", type=float, default=180)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    tok, m = load_hf(a.model, a.device)
    wl = build_workload(tok, a.n, a.seed, prompt_range=(100, a.max_prompt), out_range=(50, a.max_out),
                        prompt_median=a.prompt_median, out_median=a.out_median)
    greedy(m, wl[0]["prompt_ids"][:50], 5)  # warm-up
    rows = []
    t_start = time.time()
    for i, w in enumerate(wl):
        out, t0, times = greedy(m, w["prompt_ids"], w["max_new"], ignore_eos=True)
        itl = [b - a_ for a_, b in zip(times, times[1:])]
        rows.append({"rid": i, "prompt_len": len(w["prompt_ids"]), "gen_len": len(out),
                     "ttft_ms": (times[0] - t0) * 1e3, "tpot_ms": statistics.mean(itl) * 1e3 if itl else None,
                     "e2e_ms": (times[-1] - t0) * 1e3})
    wall = time.time() - t_start
    gen = sum(r["gen_len"] for r in rows)
    tp = [r["tpot_ms"] for r in rows]
    summary = {"system": "B1-hf-single", "model": a.model, "n": a.n, "wall_s": wall, "gen_tokens": gen,
               "throughput_tok_s": gen / wall,
               "tpot_ms": {"p50": pct(tp, 50), "p95": pct(tp, 95), "p99": pct(tp, 99)},
               "ttft_ms": {"p50": pct([r["ttft_ms"] for r in rows], 50), "p99": pct([r["ttft_ms"] for r in rows], 99)},
               "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if a.device == "cuda" else None,
               "config": vars(a)}
    out = results_dir("b1-single")
    save_json(os.path.join(out, "summary.json"), summary)
    save_json(os.path.join(out, "requests.json"), rows)
    print(json.dumps(summary, indent=1))
    print("results:", out)


if __name__ == "__main__":
    main()
