"""Quality: is the distributed output the same as single-machine output?

1. Greedy exact match of LANShard vs B1 (HF Transformers) on chat prompts.
2. MMLU accuracy (single-letter answer) on both systems + per-question agreement.
3. GSM8K accuracy (greedy chain of thought, last number) on both systems.

LANShard requests are submitted concurrently, so the comparison covers the
batched, chunked-prefill, multi-stage path, not just batch size 1.

    python bench/quality.py --mmlu 1000 --gsm8k 500 --exact 50
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re

import torch
from common import results_dir, save_json

from lanshard.cluster import local_cluster
from lanshard.coordinator import DriftConfig, SchedulerConfig
from clusters import PRESETS

EXACT_PROMPTS = [
    "Explain pipeline parallelism in two sentences.", "Write a haiku about Wi-Fi.",
    "What is 17*23? Show your work.", "List three uses of a laptop GPU.", "Who wrote Hamlet?",
    "Summarise the causes of the French Revolution.", "Give me a Python function that reverses a string.",
    "What is the capital of Australia and why is it not Sydney?", "Describe how a KV cache speeds up decoding.",
    "Translate 'good morning, how are you?' into Spanish and French.",
]


def mmlu_items(n):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(n))
    out = []
    for r in ds:
        opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
        q = (f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n"
             f"{r['question']}\n{opts}\n\nAnswer with the letter of the correct option only.")
        out.append({"prompt": q, "answer": "ABCD"[r["answer"]]})
    return out


def gsm8k_items(n):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    return [{"prompt": r["question"] + "\nSolve step by step, then give the final answer after '####'.",
             "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for r in ds]


def parse_letter(text):
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else text.strip()[:1]


def parse_number(text):
    tail = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def encode(tok, prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def run_b1(model, tasks):
    from b1_single import greedy, load_hf

    tok, m = load_hf(model)
    outs = {}
    for name, items, max_new in tasks:
        res = []
        for i, it in enumerate(items):
            o, _, _ = greedy(m, encode(tok, it["prompt"]), max_new)
            res.append(o)
            if i % 100 == 0:
                print(f"B1 {name} {i}/{len(items)}", flush=True)
        outs[name] = res
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return tok, outs


def run_lanshard(model, tasks, preset, transport):
    outs = {}
    sched = SchedulerConfig(kv_token_budget=16384, max_active=48)
    with local_cluster(model, PRESETS[preset], transport, "latency", sched=sched, drift=DriftConfig(enabled=False),
                       coord_netem=None, profile_budget=4, overhead_gb=0.3) as c:
        plan = c.plans[c.active_version].summary()
        for name, items, max_new in tasks:
            reqs = [c.submit(encode(c.tokenizer, it["prompt"]), max_new=max_new) for it in items]
            outs[name] = [r.future.result() for r in reqs]
            print(f"LANShard {name} done", flush=True)
    return outs, plan


def strip_eos(toks, eos):
    out = []
    for t in toks:
        if t in eos:
            break
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--exact", type=int, default=10)
    ap.add_argument("--exact-tokens", type=int, default=128)
    ap.add_argument("--mmlu", type=int, default=200)
    ap.add_argument("--gsm8k", type=int, default=50)
    ap.add_argument("--gsm8k-tokens", type=int, default=320)
    ap.add_argument("--preset", default="hetero3-lan")
    ap.add_argument("--transport", default="tcp")
    a = ap.parse_args()
    exact = [{"prompt": EXACT_PROMPTS[i % len(EXACT_PROMPTS)] + ("" if i < len(EXACT_PROMPTS) else f" (variant {i})")}
             for i in range(a.exact)]
    tasks = [("exact", exact, a.exact_tokens)]
    if a.mmlu:
        tasks.append(("mmlu", mmlu_items(a.mmlu), 4))
    if a.gsm8k:
        tasks.append(("gsm8k", gsm8k_items(a.gsm8k), a.gsm8k_tokens))
    tok, b1 = run_b1(a.model, tasks)
    ls, plan = run_lanshard(a.model, tasks, a.preset, a.transport)
    eos = {tok.eos_token_id, *tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])}
    report = {"model": a.model, "plan": plan, "config": vars(a)}
    for name, items, _ in tasks:
        B = [strip_eos(x, eos) for x in b1[name]]
        L = [strip_eos(x, eos) for x in ls[name]]
        match = sum(x == y for x, y in zip(B, L))
        div = [next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
               for x, y in zip(B, L) if x != y]
        r = {"n": len(items), "exact_match": match / len(items), "first_divergence_tokens": div}
        if name in ("mmlu", "gsm8k"):
            parse = parse_letter if name == "mmlu" else parse_number
            pb = [parse(tok.decode(x, skip_special_tokens=True)) for x in B]
            pl = [parse(tok.decode(x, skip_special_tokens=True)) for x in L]
            r["acc_b1"] = sum(p == it["answer"] for p, it in zip(pb, items)) / len(items)
            r["acc_lanshard"] = sum(p == it["answer"] for p, it in zip(pl, items)) / len(items)
            r["answer_agreement"] = sum(p == q for p, q in zip(pb, pl)) / len(items)
        report[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "first_divergence_tokens"}), flush=True)
    out = results_dir("quality")
    save_json(os.path.join(out, "summary.json"), report)
    save_json(os.path.join(out, "outputs.json"), {"b1": b1, "lanshard": ls})
    print("results:", out)


if __name__ == "__main__":
    main()
