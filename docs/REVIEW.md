# LANShard: code and method review

Reviewer: Ashwanth Balakrishnan. Date: 2026-11-21. Reviewed commit: `0a38faa` (`main`).
Branch with the changes: `review/ashwanth-improvements`.

File and line references point at `main` (`0a38faa`) unless they say otherwise, so they stay valid
while the branch moves. "Fixed" means fixed on the review branch with a test. "Open" means not
changed here.

## How this review was done

* Read all of `lanshard/`, `bench/`, `tests/`, the CI workflow, README, REPORT and the review PR #1
  (which has no comments yet, so nothing here duplicates it).
* Ran the test suite on `main` and on the branch, on a 4-core ARM Linux machine with no GPU, with
  transformers 4.57 and 5.24.
* Wrote a failing test for every runtime bug before fixing it. Where a bug is a race, the test forces
  the bad ordering (delays a message, blocks a loader, kills or suspends a worker process).
* Recomputed the RQ1 and RQ2 claims from the committed `results/` folders.
* Checked the prior-work comparison against the papers and repositories (sources in section 4).

Nothing here re-measures an RQ number. This machine has no GPU and is not the laptop the results
came from. Section 7 lists what that leaves unverified.

## 1. What the system does and how it is built

LANShard serves one decoder-only LLM that fits on no single laptop by cutting its layers into
contiguous ranges, one range per laptop, and passing hidden states down the chain over Wi-Fi.

* **Coordinator** (`lanshard/coordinator.py`): owns the embedding, final norm and LM head, the request
  queue, the scheduler, the planner calls, the drift monitor and the metrics. It embeds tokens, sends
  a micro-batch to stage 1, receives the last stage's hidden states, and samples.
* **Workers** (`lanshard/worker.py`): each loads only its own layers from the safetensors shards
  (`lanshard/model_utils.py`), keeps the KV cache for those layers, runs micro-batches from an inbox on
  one compute thread, and forwards to the next stage. Attention is a custom packed implementation, so
  one micro-batch can mix prefill chunks and single decode tokens from many requests.
* **Profiling** (`lanshard/probe.py`): a joining node times one real layer at decode batch sizes 1 to 8
  and one 256-token prefill, and measures RTT to every peer.
* **Planner** (`lanshard/partitioner.py`): exact DP over contiguous splits that minimises the slowest
  stage under per-node memory, searching every ordered subset of nodes. A staging constraint keeps
  replans loadable next to the running plan. `memory` is the baseline that splits by free memory.
* **Repartitioning** (`_drift_loop`, `_choose_plan`, `install_plan`): workers report observed over
  profiled stage time. A sustained 25% drift triggers a replan, which is taken only for a predicted
  gain of 10% or more. Plans are versioned. New layers are staged next to old ones, admission switches,
  and long in-flight requests are moved by re-prefilling their text on the new plan.
* **Transports** (`lanshard/transport.py`): one send/receive interface over raw TCP, gRPC and Gloo,
  plus a netem-style delay emulator. The control plane is newline-delimited JSON over TCP.
* **Front end** (`lanshard/api.py`): OpenAI-compatible HTTP. `lanshard/cluster.py` runs a whole cluster
  on one host with worker subprocesses.

The design is sound and the code is compact for what it does. The numerics are careful (stock HF
modules, token-identical to `transformers`). The weak side is everything that happens when a
message, a node or a link does not behave: most findings below are there.

## 2. Findings by severity

### Critical

**C1. A late KV free silently corrupts a migrated request. Fixed.**
`coordinator.py:486-501` (`_requeue`, `_migrate`), `coordinator.py:812-818` (`_send_free`),
`worker.py:287-290`, `model_utils.py:302-303`.
A migrated or requeued request keeps its request id as the KV key. The free for its old cache goes
through the control socket. Its re-prefill on the new plan goes through the data sockets. Nothing
orders the two, and every worker is in both plans during a staged switch. If the free arrives second,
`KVStore.free` drops the cache the new plan just filled, the next step allocates `torch.empty` and
attends over uninitialised memory, and the request returns wrong tokens with no error anywhere.
The free usually wins the race by about a millisecond, which is why the existing tests pass.
Reproduced by delaying the free by 0.3 s: the output diverges from the reference at token 67
(`tests/test_e2e_tiny.py::test_migration_tolerates_a_late_kv_free`).
Fix: a request gets a fresh cache id at every admission and migration, so a late free can only drop
the old cache.

**C2. The system does not run on current transformers. Fixed.**
`model_utils.py:243`. `Qwen2RotaryEmbedding` lost its `device` argument in transformers 5, so
`make_rotary` raises `TypeError` and no worker can profile or serve. Dependencies are unpinned
(`requirements.txt`, `pyproject.toml:10`), CI installs the latest release
(`.github/workflows/tests.yml:15`), and every test that touches the model is skipped in CI, so
nothing noticed. On `main` with transformers 5.24: `tests/test_model.py` errors, and a cluster start
hangs for the full bootstrap timeout because the worker thread dies silently (see M5).
Fix: construct the rotary embedding from the config when `device` is rejected. CI now runs both
transformers major lines and runs model code (section 5).

**C3. After a failure the cluster can stop serving forever with no error. Fixed.**
Four separate paths end with admission paused and every client blocked on a future:
* `coordinator.py:524-528`: `replan()` returns immediately if a replan is in progress. A node loss that
  arrives during a drift replan is dropped. Reproduced.
* `coordinator.py:537-539`: when the surviving nodes cannot hold the model, the planner's
  `InfeasibleError` becomes a log line and nothing else. Reproduced.
* `coordinator.py:448`: the paused-switch path waits only for `ready`. A node that dies or fails to
  restage blocks for 600 s, then raises in a daemon thread. Found by reading.
* `coordinator.py:265-273`: a node is "lost" only when its control socket returns EOF. A suspended
  laptop or a dropped Wi-Fi link keeps the socket open, so the loss is never noticed and micro-batches
  are retransmitted to it every 30 s. Reproduced with `SIGSTOP` (section 6).
Fix: every node loss and every replan ends in `_ensure_servable`, which waits for a running replan,
installs a plan over the live nodes if the active one cannot run, and otherwise fails the queued
requests with an error. Nodes silent for 15 s are treated as lost (workers already report every
second).

### Major

**M1. Releasing an old plan can delete the layers a newer plan just staged. Fixed.**
`worker.py:259-266` computes the set of layers to keep before taking the layer lock.
`worker.py:241` registers a plan only after its layers are loaded. If plan v1 finishes draining while
v3 is loading, the release deletes v3's new layers, the worker still reports `ready`, and every v3
micro-batch fails with `KeyError`. Reproduced deterministically in
`tests/test_worker.py::test_release_during_prepare_keeps_the_new_plans_layers`.
This is not the cause of the two stalls recorded in `results/`: both had a single plan (M6).

**M2. The link-occupancy term of the cost model was never active. Fixed.**
REPORT section 5 says the stage cost is `max(compute, link occupancy)` and section 4.2 says a node on a
slow enough link "will still be given fewer layers or dropped". In the code the coordinator never
passes a payload size or a bandwidth (`coordinator.py:392`), so `xfer` is always 0
(`partitioner.py:181-182`), `plan_cost` ignores it (`partitioner.py:214-225`), and bandwidth is one
number for all links. Fix: each RTT probe also measures the link's bandwidth, and the DP and
`plan_cost` price occupancy per link.

**M3. The emulated bandwidth was a delay, not a capacity. Fixed, and it affects recorded results.**
`transport.py:404-414`, `transport.py:449`. The rate term is added to each message's delay, so any
number of messages can be in the air at once. Five 25 KB messages on a 2 Mbps link all arrived after
0.1 s (a real link, and `tc netem rate`, takes 0.5 s). With this model an emulated link cannot limit
throughput and a 1.5 MB prefill chunk never holds back the decode tokens behind it. This matters for
the `badlink3` conclusion "hop latency does not limit throughput": the emulator could not have shown
otherwise. The serving runs in `results/` (RQ1, RQ2) used the old model and were not re-measured.
RQ3 is not affected: its ping-pong has one message in flight per link, and a before/after run gives
the same numbers (section 6).

**M4. A broken data link never recovers. Partly fixed.**
`transport.py:116-138`: `TcpSender` holds one socket for life, and senders are cached
(`worker.py:271-276`, `coordinator.py:785-790`). One TCP reset breaks that hop for good while the
control connection stays up, so the node is never declared lost. `transport.py:219-232`: `GrpcSender`
never inspects the stream's future, so after a failure it queues messages forever. Fix: TCP reconnects
once and resends, gRPC raises on the next send. Open: gRPC and Gloo do not reconnect.

**M5. Worker control requests fail silently. Fixed.**
`worker.py:164-168`: profile, ping and prepare run on bare threads. Only CUDA OOM is handled
(`worker.py:246`). Any other exception kills the thread and the coordinator waits 600 to 900 s for a
reply. This is what turns C2 into a 15-minute hang.

**M6. One lost micro-batch holds a pipeline slot for 30 s, and the stall root cause is still open.**
`coordinator.py:762` caps in-flight decode micro-batches at the pipeline depth, and a micro-batch that
never returns keeps its slot until `_retransmit_stale` fires (`coordinator.py:832-849`, 30 s minimum).
Lose `depth` micro-batches and nothing is dispatched. Both recorded stalls match this exactly:
"12 active, 0 waiting, 3 in flight" on a 3-stage plan with every coordinator thread idle
(`results/run_all.log:556`). So the stall is three micro-batches lost between workers. Why they were
lost is not answered by this review: the worker logs of those runs contain nothing, and the
pre-hardening code is not in the history. Workers now report a failed micro-batch to the coordinator,
so a recurrence will name the worker and the exception. Open: an explicit per-micro-batch failure
reply with immediate retransmit, and a shorter timeout derived from the measured loop time.

**M7. `pytest` never exits outside CI. Fixed.**
`tests/test_transport.py:14-35` never closes its sender or server, and an open gRPC client stream
keeps the interpreter alive. `tests/conftest.py:11-18` hides this in CI with `os._exit`. Locally,
`python -m pytest tests -q` (the README command) prints "17 passed" and then hangs.

**M8. The end-to-end tests only pass on a GPU. Fixed.**
`tests/test_cluster.py:25-26` hard-codes memory caps sized for FP16 layers. On a CPU-only machine the
layers are FP32, the two workers cannot hold the model, and both tests error in setup with
`InfeasibleError`.

**M9. The RQ1 headline compares runs with different workloads. Docs corrected.** See section 3.

**M10. The RQ2 "+10%" is the size of the baseline's run-to-run spread. Documented.** See section 3.

**M11. No authentication anywhere. Open.**
The control port, every data port and the HTTP API bind `0.0.0.0` by default
(`coordinator.py:1082`, `worker.py:377`) and accept anyone. On a shared Wi-Fi network any host can
join as a worker and receive the hidden states of every prompt, send tensors to a worker's data port
(the header's `max_len` sizes an allocation), or use the API. A shared join token plus binding to
the LAN interface is the minimum before a real multi-laptop run on a campus network.

**M12. Nodes that join after bootstrap are never used. Open.**
`coordinator.py:326-333` profiles and plans once. A later join is recorded and ignored, so after
losing capacity the only recovery is a restart. README says workers "are discovered when they join".

**M13. The drift signal is confounded by context length. Open.**
The profile times decode over a 512-token context (`probe.py:75`), and the worker compares every
decode micro-batch to it (`worker.py:321-324`) whatever the real contexts are. A shift toward longer
conversations looks like a slowdown. The 15 s baseline hides this only while the mix is stationary.

**M14. Prefill is not pipelined. Open.**
`coordinator.py:54` and `coordinator.py:769-782`: one prefill chunk is in flight at a time and a
request waits for each chunk to come back before sending the next. Chunks of one request could
follow each other down the pipeline (each stage is FIFO), which would cut TTFT for long prompts by up
to the number of stages. See section 4.

### Minor

* **Planner tie-break is not exact. Fixed.** `partitioner.py:149-155` breaks ties inside the min-max
  DP. The bottleneck is optimal, but on 600 random clusters the single-stream latency was above the
  optimum in 10 (up to 7%).
* **Planning time.** REPORT.md:78 says 3 to 50 ms. That holds for three nodes. Five nodes take about
  1 s on this machine (325 orders), and `partitioner.py:173` allows six (1,956 orders). A replan
  decision plans two or three times. Open.
* **FP16 wire is hard-coded. Fixed (option).** `worker.py:105`, `coordinator.py:798`. FP32 CPU stages
  are rounded at every hop, so the output depends on where the cuts are. A tiny FP32 cluster diverged
  from `transformers` at token 90 for this reason alone.
* **Stale result after a requeue. Fixed.** `coordinator.py:904-923` applies a result to whatever
  request has that id now, including one requeued by a node loss and re-admitted since.
* **Wrong count in the `migrate` event. Fixed.** `coordinator.py:483-484` subtracts the requests it
  already moved a second time, so `left_to_drain` showed -1 in a live run.
* **Aborted plans leak layers. Fixed.** `coordinator.py:418-420` returns without releasing the plan on
  the nodes that had staged it.
* **Drift monitor can die. Fixed.** `coordinator.py:620-667` has no exception guard and iterates
  `self.nodes` without the lock (`coordinator.py:627`).
* **Oversized requests.** `coordinator.py:720` admits a request larger than the whole KV budget when
  nothing else is active, which breaks the memory reserve the planner assumed. Open.
* **Unbounded metrics.** `mb_records`, `token_log`, `node_stats_log`, `events` grow forever in a
  long-running server (`coordinator.py:192-197`). Open.
* **Server close.** `TcpServer.close` left accepted connections open and, on Linux, the port bound
  (`transport.py:191-196`). Fixed. `GlooServer` swallowed every error (`transport.py:370`). Fixed.
  `free_port` is a bind-then-close race (`transport.py:66-71`, used at `transport.py:344`). Open.
* **API.** No input validation and a dropped connection on any error (`api.py:38-48`). Fixed. A client
  that disconnects does not cancel its request. Open.
* **Callback under the scheduler lock.** `coordinator.py:931-935` runs `on_token` while holding the
  lock the scheduler needs. Open.
* **Stale documentation.** `coordinator.py:18` ("Nothing is migrated or recomputed"),
  `transport.py:1` (`send_hidden` / `recv_hidden` do not exist), `partitioner.py:6` (additive cost),
  REPORT.md:40 ("about 2,000 lines", it is 3,138), `bench/rerun_after_fixes.sh:2` (wrong section).
  Fixed.
* **Dead or unused code.** `_kv_tokens_per_node` ignores its argument (`coordinator.py:401`).
  `coord_ms` is never passed (`partitioner.py:169`). The `latency-fixed` and `even` policies and the
  `homog2` and `hetero4` presets appear in no committed result. `worker.py:177-178` re-imports modules
  already imported. `coordinator.py:809` catches an error the asynchronous sender never raises. Open,
  listed so nobody assumes they are exercised.
* **Benchmarks.** `bench/run_all.py:18` ignores failed steps and still plots. The rerun script has no
  `set -e`. `bench/sustained.py:30` shares one RNG across 12 client threads, so the workload differs
  between variants with the same seed. B1 is sequential batch size 1, so it is a latency reference
  and not a throughput baseline. Open.
* **Packaging.** Dependencies are listed twice and unpinned. Tests rely on `sys.path` edits.
* **Structure.** `coordinator.py` is one 1,113-line class holding membership, scheduling, planning
  glue, drift, migration and metrics. Splitting membership and metrics out would make the locking
  rules easier to state. Open.

## 3. Measurement validity

**RQ1 pairs different workloads.** REPORT.md:204 computes "+12% to +21%" from 30.6 vs 27.2,
29.7 vs 24.5 and 27.6 vs 23.7 tok/s. The `summary.json` files show that the first two pairs used
different seeds, and therefore different requests:

| start | policy | seed | generated tokens | tok/s |
|---|---|---|---|---|
| 13:14 | latency-additive | 0 | 6635 | 30.6 |
| 13:19 | memory | 1 | 4818 | 27.2 |
| 13:26 | latency-additive | 1 | 4818 | 29.7 |
| 13:30 | memory | 2 | 5848 | 24.5 |
| 15:38 | memory | 0 | 6635 | 23.7 |
| 15:45 | latency | 0 | 6635 | 27.6 |
| 15:51 | latency | 1 | 4818 | 22.1 |
| 15:57 | latency | 2 | 5848 | 23.7 |

The seed alone moves throughput by several percent (the three `badlink3` memory runs give 72.6,
68.8 and 65.0). Only two pairs have the same workload and were taken minutes apart:

* seed 1: latency-additive 29.7 vs memory 27.2, **+9%** (7 minutes apart);
* seed 0: latency 27.6 vs memory 23.7, **+16%** (7 minutes apart).

In both, the treatment ran after the baseline on a heating GPU, so thermal drift works against it.
The honest statement is "+9% and +16% in the two valid pairs", and for the final `latency` policy
there is one valid pair. The other two `latency` runs (22.1, 23.7) are below or equal to the
same-seed memory runs taken two hours earlier, which the report attributes to heat. README and REPORT
now state this.

**The load point is far past saturation.** 40 requests arrive in about 27 s and the runs last 160
to 280 s. The offered load is 180 to 250 tok/s against a capacity of 22 to 31. Median TTFT is 9 to
20 s, which is queueing. TPOT is active requests divided by throughput, so "p99 TPOT is equal or
better" restates the throughput result. There is no point below saturation, where the partitioning
policy affects latency in a different way (section 4).

**RQ2 has one run per variant and the effect is inside the baseline's spread.** The claim is 15.9
(final) vs 14.5 (static) tok/s after the throttle. A second static run from the same day
(`results/_rq2-repro-drift-off`) gives 13.2 after the throttle and 25.4 before it, against 14.5 and
21.6. The baseline alone varies by 1.3 tok/s, about 10%. The variants also ran one after another in
one process on the heating laptop, in a fixed order.

**The emulation favours the baseline, and one part could not show the effect it was used to test.**
The slowdown waits after the forward pass and leaves the shared GPU free (the report says this).
The emulated bandwidth could not limit throughput (M3).

**The cost model over-predicts.** The planner predicts a 40% lower bottleneck for `hetero3`. The
measured gain is 9 to 16%. `_choose_plan` accepts a replan for a predicted 10% gain, which on this
evidence is 3 to 4% in practice.

**What a convincing version looks like.** Same seeds for both arms, interleaved or randomised order,
at least five repeats with confidence intervals, a load sweep that includes points below
saturation, and separate machines. `bench/run_all.py` already loops over repeats; it needs the
interleaving and a rate sweep.

## 4. The core idea: critique, prior work, proposed improvements

### Is splitting by measured speed over Wi-Fi the right approach?

Partly. Three things hold up, three do not.

Holds up:
* Measured per-layer time is the right planner input. llama.cpp RPC splits by free memory in device
  order with no latency or bandwidth term, and exo splits by memory share with a "TODO: profile
  actual connection speeds". Neither uses measured speed.
* Min-max over stages with `max(compute, communication)` per stage is the established throughput
  objective. EdgeShard and PipeEdge both use exactly this form and also select the device subset and
  order. The team reached the same model after the additive one failed, which is a good sign, but
  the report should cite them rather than present it as a fix.
* Online repartitioning with a staging constraint and token-identical migration is the part that
  the checked sources do not describe: EdgeShard, PipeEdge and Galaxy plan once offline, and llama.cpp
  and exo are static at load. Petals rebalances at runtime with a 20% hysteresis, which matches the
  10% threshold idea here. This is where the project's contribution is.

Does not hold up:
1. **The objective optimises one regime.** It minimises the slowest decode stage at a fixed
   micro-batch of 4. It ignores prefill, the coordinator's own work, and any fixed cost per stage,
   and it over-predicts gains by about 3x. On a GPU a prefill token costs about 80 times less than a
   decode token; on a CPU the ratio is much smaller, so the right split depends on the prompt to
   output ratio of the workload.
2. **At low load the objective is the wrong one.** With fewer requests than pipeline stages, token
   latency is the sum of all stages and hops, not the maximum. EdgeShard has a separate latency DP for
   this case. LANShard always minimises the maximum and uses latency only to break ties.
3. **The binding constraint is memory, not the split.** With 33 layers of capacity for 28 layers,
   a throttled node can shed only about 5 layers, which is why RQ2 tops out near 10%. Anything that
   widens the feasible set (per-device quantization) moves the result more than a better split.

And Wi-Fi itself: for decode, 8 to 32 KB per hop is nothing next to a 50 ms stage. Wi-Fi hurts in two
places the current design does not address: every token pays every hop's latency and jitter (TPOT is
10 to 20 times a single GPU's), and prefill moves megabytes per chunk per hop, which is where TTFT
goes.

### Prior work used for the comparison

| System | Split by | Network model | Runtime adaptation | Source |
|---|---|---|---|---|
| EdgeShard | profiled per-layer time, memory, bandwidth; DP for latency (sum) and for throughput (max of compute and communication per stage); selects device subset and order | activation size / bandwidth | not described | arxiv.org/abs/2405.14371 |
| PipeEdge | same max(compute, communication) form; DP over layers, device subsets and order | payload / bandwidth | not described (QuantPipe adapts bit width to bandwidth) | par.nsf.gov/servlets/purl/10345258 |
| Petals | each server picks the block span that most raises swarm throughput; clients route by measured latency | measured ping and throughput | servers rebalance only for a gain of 20% or more; clients replay cached inputs to rebuild a replacement server's KV | arxiv.org/abs/2209.01188, arxiv.org/abs/2312.08361 |
| Galaxy | tensor and sequence parallelism inside each layer, split by profiled capacity | overlaps communication with compute | none | arxiv.org/abs/2405.17245 |
| exo | memory share (ring memory weighted) | ignored | topology only | github.com/exo-explore/exo |
| llama.cpp RPC | free memory, device order | ignored | none | github.com/ggml-org/llama.cpp tools/rpc |

Petals also quantizes weights to 8 or 4 bits and compresses hidden states on the wire. The PipeEdge,
exo and llama.cpp rows were read directly from the paper and source. The EdgeShard, Petals and Galaxy
rows were checked through the papers' HTML with less care, so verify them before citing in the report.

### Proposed improvements to the method

Implemented in this PR:

| Improvement | What it changes |
|---|---|
| Measured per-link bandwidth, per-link occupancy in the DP and in `plan_cost` | The planner can now shrink or drop a stage whose outbound link is the bottleneck (M2). |
| Exact lexicographic objective (bottleneck, then latency) | Among plans with the best throughput, the planner returns the one with the lowest single-stream latency. |
| Emulated bandwidth as link occupancy | The emulator can show link-bound behaviour and head-of-line blocking behind prefill chunks (M3). |
| Fresh KV cache id per fill | Migration stays correct whatever the order of control and data messages (C1). |
| Failure handling: heartbeat detection, recovery after any failed switch, fail-fast when the model no longer fits | Covers the "node leaves" half of I3 for a suspended laptop or a dropped link, not only a closed socket (C3). |
| Lossless wire option | FP32 clusters stay token-identical across re-splits. |

Proposed, not implemented (ordered by expected value):

1. **Move KV instead of recomputing it.** A layer's K and V depend only on that layer's input, not on
   which node computed them. On a switch, only layers that change owner need their KV on the new
   owner. Send it directly: about 1 KB per token per moved layer for the 1.5B model, so a 1,000-token
   request moving 3 layers is 3 MB. That replaces the 15 s re-prefill dip with a transfer of under a
   second and keeps the unchanged layers' KV in place. It needs a worker-to-worker KV message, an ack
   to the coordinator, and a dtype cast when a layer moves between FP32 and FP16 nodes.
2. **Per-device quantization as a planner dimension.** Profile each node at FP16 and at 8-bit (and
   4-bit where supported), giving `(t_layer, mem_layer)` per precision, and let the DP choose the
   precision per node. This attacks the memory constraint that caps RQ2. It needs a quality
   evaluation per mix, which `bench/quality.py` can already run.
3. **A workload-aware objective.** Cost per generated token for a stage with `m` layers:
   `m * (t_decode(b) / b + rho * t_prefill_per_token)`, and for its link:
   `(1 + rho) * hidden * bytes_per_value * 8 / bandwidth`, where `rho` is prompt tokens per generated
   token, measured online. Minimise the maximum over stages. This brings prefill into both terms and
   makes the link term bind where it really does.
4. **Choose the objective by load.** Below about one request per stage, minimise the sum (the DP
   already computes it); above, minimise the maximum.
5. **Pipelined prefill.** Send a prompt's chunks back to back and track sent and confirmed tokens
   separately. Per-stage FIFO order keeps this correct.
6. **Calibrate the model from the pipeline.** Workers already report observed over predicted stage
   time. Fit a fixed cost per stage and a per-layer cost from those reports and plan with the fitted
   values, instead of scaling one node's profile.
7. **Hidden-state compression on slow links**, with the bit width chosen from the measured bandwidth
   (Petals, QuantPipe). Mostly a prefill and TTFT improvement.
8. **Membership.** Profile and use nodes that join later, re-measure RTT and bandwidth periodically,
   and add a join token.

## 5. What changed on the review branch

In commit order (a few items span two commits):

1. `model_utils.py`: rotary embedding works on transformers 4 and 5 (C2).
2. `--wire-dtype` for workers and coordinator (default FP16, unchanged).
3. Tests: a tiny random-weight Qwen2 checkpoint generated at test time, and end-to-end tests on it
   with real worker processes. Transport tests close what they open, and the CI-only `os._exit` hook
   is gone (M7).
4. Worker: plan registration and release under one lock (M1).
5. Coordinator: fresh KV cache id per fill, stale results ignored (C1).
6. Coordinator: `_ensure_servable`, heartbeats, fail-fast, aborted plans released, guarded drift
   thread (C3). A node lost before the first plan is left to bootstrap, and a heartbeat round is
   skipped right after the coordinator itself was stalled.
7. Partitioner: exact tie-break.
8. Bandwidth measurement and per-link occupancy (M2).
9. Netem link occupancy (M3).
10. Transport: TCP reconnect, gRPC failure reporting, full server close (M4).
11. Worker: failed control requests and failed micro-batches are reported (M5, part of M6).
12. API: 400 and 503 responses.
13. Tests: the 0.5B cluster test derives its memory caps from the compute dtype (M8).
14. CI: transformers 4 and 5 matrix, end-to-end tests run on every push.
15. Docs: this file, README and REPORT corrections, stale docstrings.

Behaviour that changes for existing users: netem runs with a rate are slower for large payloads (M3),
plans can differ where a link is slow or where several plans tie on the bottleneck, a silent worker
is dropped after 15 s, and requests fail with an error instead of waiting when the cluster cannot
serve.

## 6. Test results

Machine: 4-core ARM Linux, CPU only, Python 3.11, torch 2.15.

Before (`main`):

| Environment | Result |
|---|---|
| transformers 5.24, no weights cached | 17 passed, 4 skipped, then pytest does not exit (killed after 10 minutes) |
| transformers 5.24, 0.5B weights cached | `test_model.py`: 2 errors (`TypeError` in `make_rotary`); `test_cluster.py` hangs in bootstrap |
| transformers 4.57, 0.5B weights cached | 19 passed, 2 errors (`test_cluster.py`: `InfeasibleError` on CPU) |

After (review branch), 0.5B weights cached:

| Environment | Result |
|---|---|
| transformers 5.24 | 42 passed, pytest exits on its own |
| transformers 4.57 | 42 passed, pytest exits on its own |

Without the weights the four 0.5B tests skip and the other 38 run (this is what CI does). The
timing-sensitive files (`test_e2e_tiny.py`, `test_transport.py`, `test_worker.py`) were also run three
more times, twice as two copies in parallel on the 4 cores: 24 passed each time.

Benchmarks that run on CPU, before (`main`) and after (branch). These are functional checks on a
different machine, not replacements for the numbers in REPORT.

RQ3 ping-pong (`bench/rq3_transport.py --iters 150 --iters-large 10`), RTT p50 in ms:

| payload | backend | loopback, main | loopback, branch | emulated Wi-Fi, main | emulated Wi-Fi, branch |
|---|---|---|---|---|---|
| 8 KB | tcp | 0.32 | 0.27 | 7.4 | 7.7 |
| 8 KB | grpc | 0.72 | 0.72 | 8.0 | 8.6 |
| 8 KB | gloo | 0.84 | 0.84 | 8.5 | 8.7 |
| 16 MB | tcp | 9.5 | 9.6 | 687 | 687 |
| 16 MB | grpc | 56.3 | 55.1 | 745 | 742 |
| 16 MB | gloo | 17.6 | 18.3 | 702 | 715 |

No change from the branch, as expected for one message in flight. The ordering matches REPORT 4.1
(TCP fastest, gRPC slowest at 16 MB).

RQ2 path (`bench/sustained.py`, 0.5B model, `hetero3` preset as three CPU workers, 150 s, throttle
2.5x at 45 s, 4 clients, one run each): both `main` and the branch detect the throttle, accept a
staged replan, switch in under 1.1 s and migrate the in-flight requests. Throughput before and after
the throttle: 9.0 and 6.1 tok/s on `main`, 8.7 and 5.8 on the branch, which is within the noise of
single 150 s runs on a shared 4-core machine. This is the only place the drift monitor, `replan`
and `_choose_plan` run end to end after the refactor; it logged no error.

Bugs reproduced by a test that fails on `main`'s logic and passes on the branch:

| Bug | Test | Failure before the fix |
|---|---|---|
| C1 late free | `test_migration_tolerates_a_late_kv_free` | wrong tokens from index 67 |
| C3 loss during replan | `test_node_loss_during_a_replan_is_not_dropped` | request never finishes |
| C3 survivors too small | `test_requests_fail_when_the_survivors_cannot_hold_the_model` | request never finishes |
| C3 silent node | `test_frozen_worker_is_detected_by_heartbeat` | request never finishes (checked with the timeout disabled) |
| M1 release race | `test_release_during_prepare_keeps_the_new_plans_layers` | layers 4 and 5 deleted |
| M3 netem rate | `test_netem_rate_is_link_occupancy` | last message at 0.1 s, not 0.5 s |
| M4 TCP | `test_tcp_sender_reconnects_after_the_link_breaks` | link stays dead, port stays bound |
| M4 gRPC | `test_grpc_sender_reports_a_broken_stream` | send never fails |
| tie-break | `test_dp_is_lexicographically_optimal_on_random_clusters` | latency above optimum |

## 7. Not verified, and why

* **Every RQ number.** No GPU here and not the original laptop. The numbers in README and REPORT are
  the team's measurements. The only edits to them are the RQ1 pairing correction and the RQ2 caveat,
  both computed from the committed `results/`.
* **Effect of the netem and planner changes on RQ1 and RQ2.** The serving runs should be repeated,
  `badlink3` first. Expect little change for decode-only traffic at 100 to 400 Mbps and a visible
  change in TTFT and tail TPOT where prefill chunks share a slow link. RQ3 is not affected.
* **GPU, FP16, MPS and Windows paths.** All tests here ran FP32 on Linux CPU. The changes are
  dtype-neutral by construction, but nobody has run them on CUDA yet. The team's laptops are Windows:
  the suspended-worker test is skipped there (no `SIGSTOP`), the rest should run.
* **Gloo inside a cluster.** Only the round-trip test covers Gloo. The end-to-end tests cover TCP and
  gRPC.
* **The bandwidth estimate on real Wi-Fi.** Tested against the emulator (within 25% at 20 Mbps). Real
  links have burstier loss, and one probe at join can be unlucky.
* **The stall's root cause** (M6).
* **Heartbeat timeout on slow real hardware.** 15 s without a report from a worker that reports every
  second. A laptop that swaps heavily while loading an 8B model could be dropped by mistake. The
  timeout is a `Coordinator` argument.
* **B3, real laptops, proposal-scale repeats.** Unchanged from the README's TODO list.
