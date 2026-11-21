# LANShard: device-aware LLM layer partitioning over Wi-Fi

CSE 598 (Fall 2026), Team 20: Omer Kuyucu, Ashwanth Balakrishnan, Faisal Barkat Khorasi.

LANShard serves one LLM across 2-5 heterogeneous laptops on a shared Wi-Fi network. It is a
coordinator plus workers built on Hugging Face Transformers:

| Idea | Where | What it does |
|---|---|---|
| **I1** discovery + profiling | `lanshard/probe.py` | A joining node reports OS / accelerator / free memory, then times one real decoder layer at decode batch sizes 1-8 and a 256-token prefill chunk. Output: `(t_layer, mem_layer)`. |
| **I2** latency-balanced partitioner | `lanshard/partitioner.py` | Exact DP over contiguous layer ranges that minimises the slowest stage, subject to per-node memory (weights + KV reserve). It searches every ordered node subset (N<=5). Stage cost is compute; hop latency overlaps with pipelining, so it only breaks ties (`--policy latency`). The proposal's original "compute + RTT" model is kept as `--policy latency-additive`. |
| **I3** drift-triggered repartitioning | `lanshard/coordinator.py` (`_drift_loop`, `_choose_plan`, `install_plan`) | Workers report a rolling median of observed/expected stage time. When a node drifts >25% for 30 s (or leaves), the coordinator prices three options over a 5-minute horizon: keep the plan, a **zero-pause staged plan** (new layers fit next to the old ones, enforced inside the DP), or an **unconstrained plan** that needs a short pause. It switches only if the best option gains >=10%. On a switch, long in-flight requests are **migrated** (re-prefilled on the new plan); short ones finish in place. |
| **I4** phase-aware continuous batching | `lanshard/coordinator.py` (`_form_batches`) | Packed micro-batches mixing many requests' segments. Decode batches are `ceil(R / D)` (the smallest that keeps D = measured loop/stage pipeline depth busy). Prefill runs in separate 512-token chunks with their own slot. |
| Transport study | `lanshard/transport.py` | `send_hidden`/`recv_hidden` with three interchangeable backends: raw length-prefixed **TCP**, **gRPC** (protobuf `BytesValue`), **torch.distributed Gloo** (pairwise process groups created at runtime). Plus `NetemProfile` delay/jitter/spike/rate emulation for Windows/macOS hosts. |

The coordinator owns the queue, scheduler, planner, embedding and sampling. Each worker owns one contiguous
layer range, its KV cache and one data endpoint. Workers load **only their own layers** straight from the
safetensors shards (the model skeleton is built on the `meta` device). Attention is a packed, per-segment
implementation, so prefill chunks and decode tokens from different requests share one forward pass. Greedy
output is token-identical to stock `transformers` for a single request, including across a mid-generation
repartition (`tests/test_model.py`, `tests/test_cluster.py`). With many concurrent requests, FP16 batching can
flip near-tie tokens; MMLU/GSM8K accuracy is unchanged (see the report).

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x. Works with CUDA, Apple MPS and CPU-only workers, on Windows, macOS and Linux mixed.

## Run on real laptops

Coordinator (any machine; it also needs the embedding/LM head, ~0.5 GB for 1.5B):

```bash
python -m lanshard.coordinator --model Qwen/Qwen2.5-1.5B-Instruct --workers 3 --policy latency --http 8000
```

Each worker laptop:

```bash
python -m lanshard.worker --coordinator <coordinator-ip>:29500
```

Workers are discovered at join time (nothing about the device list is configured). After profiling and
planning, the coordinator prints the plan and serves an OpenAI-compatible API:

```bash
curl http://<coordinator-ip>:8000/v1/chat/completions -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":64}"
```

`GET /status` shows the plan, per-node profile, RTT, drift ratio, busy time and memory.

Useful flags: `--transport tcp|grpc|gloo` (must match on all nodes), `--policy latency|latency-additive|memory|even`,
`--no-drift`, `--netem wifi|profile.json|base=..,jitter=..`, worker `--device cuda|mps|cpu`,
`--slowdown 2` and `--mem-limit-gb 3` (emulate a slower/smaller device).

## Run an emulated cluster on one machine

A worker is a process, not a machine. This starts a coordinator plus 3 workers: a fast GPU, a 2x slower GPU and a CPU-only node:

```bash
python -m lanshard.cluster --model Qwen/Qwen2.5-0.5B-Instruct --worker id=gpu-a --worker id=gpu-b,slowdown=2 --worker id=cpu,device=cpu --prompt "Explain pipeline parallelism."
```

## Experiments (`bench/`)

| Script | Question | Output |
|---|---|---|
| `rq3_transport.py` | RQ3: per-hop cost of gRPC vs TCP vs Gloo at 8 KB to 16 MB, with ser/deser/transfer split; `--netem wifi`; `--peer` for two real hosts | `results/*-rq3-transport*/` |
| `serve.py` | RQ1/B2: ShareGPT-like replay (Poisson arrivals, 100-2000 prompt, 50-500 output, or `--sharegpt file.json`) through a preset cluster, `--policy latency` vs `--policy memory --no-drift` | throughput, TTFT, TPOT p50/95/99, per-hop split, bubble ratio, peak memory |
| `sustained.py` | RQ2: closed-loop load, throttle one node mid-run, drift on vs off; transition cost | throughput timeline + events |
| `quality.py` | Greedy exact match vs B1, MMLU and GSM8K accuracy on both systems | `results/*-quality/` |
| `b1_single.py` | B1: single-machine HF Transformers | latency ceiling |
| `b3_llamacpp_rpc.py` | B3: `llama.cpp --rpc` on the same machines (needs llama.cpp binaries + GGUF) | same metrics |
| `measure_rtt.py` | Measure real Wi-Fi RTT between two laptops and emit a netem profile JSON | profile for `--netem` |
| `run_all.py` | Everything above, then `plot.py` | `results/figures/` |

Cluster presets live in `bench/clusters.py`. Each run writes a dated folder under `results/` with config, logs and CSVs.

## Tests

```bash
python -m pytest tests -q
```

These check the DP against brute force (both cost models) over all orders and splits, the memory and staging
constraints, transport FIFO and round-trip on all three backends, and netem ordering. With the 0.5B weights
cached they also check exact greedy agreement of a two-stage split with HF, and token-identical output across
staged and paused mid-generation replans (~2 min). `LANSHARD_SKIP_SLOW=1` skips the cluster tests.

## Report

See [`REPORT.md`](REPORT.md) for the design, the results from this repository's runs and their limitations.
