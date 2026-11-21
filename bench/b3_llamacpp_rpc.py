"""B3: llama.cpp ``--rpc`` on the same machines (third-party sanity baseline).

Requires llama.cpp built with ``-DGGML_RPC=ON`` (official releases ship
``rpc-server`` and ``llama-server``) and a GGUF of the same model, e.g.
``Qwen2.5-1.5B-Instruct-f16.gguf`` or ``Llama-3.1-8B-Instruct-Q4_K_M.gguf``.

On every worker laptop:
    rpc-server --host 0.0.0.0 --port 50052            # (add -c to cache tensors)

On the coordinator laptop:
    python bench/b3_llamacpp_rpc.py --bin-dir C:/llama.cpp/bin --gguf model.gguf \
        --rpc 192.168.1.11:50052,192.168.1.12:50052 --n 60 --rate 2

This script starts ``llama-server --rpc ...`` (which splits layers by free
memory, i.e. the B2 policy) and replays the same ShareGPT-like workload
against its OpenAI endpoint, writing the same summary.json metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request

from common import build_workload, pct, results_dir, save_json


def stream_one(url, prompt_ids, max_new, rec):
    body = json.dumps({"prompt": prompt_ids, "n_predict": max_new, "ignore_eos": True, "temperature": 0,
                       "stream": True, "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"})
    t0 = time.time()
    times = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data: "):
                d = json.loads(line[6:])
                if d.get("content") is not None and not d.get("stop"):
                    times.append(time.time())
    itl = [b - a for a, b in zip(times, times[1:])]
    rec.append({"ttft_ms": (times[0] - t0) * 1e3 if times else None, "gen_len": len(times),
                "tpot_ms": sum(itl) / len(itl) * 1e3 if itl else None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--rpc", required=True, help="comma-separated host:port of rpc-server instances")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    exe = os.path.join(a.bin_dir, "llama-server" + (".exe" if os.name == "nt" else ""))
    srv = subprocess.Popen([exe, "-m", a.gguf, "--rpc", a.rpc, "-ngl", "999", "--port", str(a.port),
                            "-np", str(a.parallel), "-c", str(4096 * a.parallel), "-cb"])
    url = f"http://127.0.0.1:{a.port}"
    for _ in range(600):
        try:
            urllib.request.urlopen(url + "/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    from transformers import AutoTokenizer

    from lanshard.model_utils import resolve_model_dir

    tok = AutoTokenizer.from_pretrained(resolve_model_dir(a.tokenizer))
    wl = build_workload(tok, a.n)
    recs, ths = [], []
    t0 = time.time()
    import random

    rng = random.Random(0)
    for w in wl:
        time.sleep(rng.expovariate(a.rate))
        th = threading.Thread(target=stream_one, args=(url, w["prompt_ids"], w["max_new"], recs))
        th.start()
        ths.append(th)
    for th in ths:
        th.join()
    wall = time.time() - t0
    srv.terminate()
    tp = [r["tpot_ms"] for r in recs]
    summary = {"system": "B3-llama.cpp-rpc", "n": a.n, "wall_s": wall,
               "throughput_tok_s": sum(r["gen_len"] for r in recs) / wall,
               "tpot_ms": {"p50": pct(tp, 50), "p95": pct(tp, 95), "p99": pct(tp, 99)},
               "ttft_ms": {"p50": pct([r["ttft_ms"] for r in recs], 50)}, "config": vars(a)}
    out = results_dir("b3-llamacpp-rpc")
    save_json(os.path.join(out, "summary.json"), summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
