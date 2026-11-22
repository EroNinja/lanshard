"""I1: discovery and profiling.

A joining node describes itself (OS, CPU, RAM, accelerator, free memory) and
then times one real decoder layer of the target model at a few decode batch
sizes and one prefill chunk. The planner works from these measurements,
``(t_layer, mem_layer)``, rather than from free memory alone.

``slowdown`` and ``mem_limit_gb`` let one physical machine stand in for a
slower or smaller one (e.g. several workers sharing one GPU): the slowdown is
applied as real wall-clock time after each forward, so the profiler and the
runtime both see the slower device, exactly as they would see a real one.
"""
from __future__ import annotations

import os
import platform
import socket
import time
from typing import Dict, Optional

import psutil
import torch

from .timing import precise_sleep
from .model_utils import LayerLoader, Segment, StageRunner, default_dtype


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def device_info(device: str, mem_limit_gb: Optional[float] = None) -> Dict:
    vm = psutil.virtual_memory()
    info = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_total": vm.total,
        "ram_free": vm.available,
        "device": device,
    }
    if device.startswith("cuda"):
        idx = torch.device(device).index or 0
        free, total = torch.cuda.mem_get_info(idx)
        info.update(accel=torch.cuda.get_device_name(idx), mem_total=total, mem_free=free, kind="cuda")
    elif device.startswith("mps"):
        # unified memory: the GPU can use roughly 2/3 of RAM by default
        info.update(accel="Apple GPU (MPS)", mem_total=vm.total, mem_free=int(vm.available * 0.66), kind="mps")
    else:
        info.update(accel="CPU", mem_total=vm.total, mem_free=int(vm.available * 0.8), kind="cpu")
    if mem_limit_gb is not None:
        info["mem_free"] = min(info["mem_free"], int(mem_limit_gb * (1 << 30)))
        info["mem_limited"] = True
    return info


class Slowdown:
    """Wall-clock throttle: after a forward taking t, wait (factor-1)*t."""

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def apply(self, elapsed_s: float):
        if self.factor > 1.0:
            precise_sleep(elapsed_s * (self.factor - 1.0))


def profile_layer(model_id: str, device: str, dtype: Optional[torch.dtype] = None,
                  budget_s: float = 15.0, context: int = 512, batch_sizes=(1, 2, 4, 8),
                  prefill_tokens: int = 256, slowdown: Optional[Slowdown] = None) -> Dict:
    dtype = dtype or default_dtype(device)
    slowdown = slowdown or Slowdown(1.0)
    loader = LayerLoader(model_id)
    runner = StageRunner(loader.cfg, device, dtype)
    runner.layers = loader.load_layers([0], device, dtype)
    g = loader.geom
    max_b = max(batch_sizes)
    max_len = context + 64

    def timed(fn) -> float:
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        dt = time.perf_counter() - t0
        slowdown.apply(dt)
        return (time.perf_counter() - t0) * 1e3

    # fill a context for each fake request so decode attends over `context` tokens
    for r in range(max_b):
        x = torch.randn(context, g.hidden, device=device, dtype=dtype) * 0.02
        runner.forward(x, [Segment(r, 0, context, max_len)], [0])

    def decode(b):
        x = torch.randn(b, g.hidden, device=device, dtype=dtype) * 0.02
        return lambda: runner.forward(x, [Segment(r, context, 1, max_len) for r in range(b)], [0])

    def prefill():
        x = torch.randn(prefill_tokens, g.hidden, device=device, dtype=dtype) * 0.02
        return lambda: runner.forward(x, [Segment(10_000, 0, prefill_tokens, prefill_tokens)], [0])

    for _ in range(3):  # warm-up (kernel selection, allocator)
        timed(decode(1))
        timed(prefill())

    deadline = time.perf_counter() + budget_s
    per_case = budget_s / (len(batch_sizes) + 1)
    t_decode = {}
    for b in batch_sizes:
        fn, samples, end = decode(b), [], time.perf_counter() + per_case
        while time.perf_counter() < end or len(samples) < 5:
            samples.append(timed(fn))
        t_decode[b] = sorted(samples)[len(samples) // 2]
    fn, samples = prefill(), []
    while time.perf_counter() < deadline or len(samples) < 3:
        samples.append(timed(fn))
    t_prefill = sorted(samples)[len(samples) // 2]

    return {
        "model": model_id,
        "dtype": str(dtype).replace("torch.", ""),
        "context": context,
        "t_decode_ms": {str(b): v for b, v in t_decode.items()},   # ms per layer per micro-batch
        "t_prefill_ms_per_tok": t_prefill / prefill_tokens,
        "mem_layer": loader.layer_param_bytes(0, dtype),
        "kv_per_tok_layer": g.kv_bytes_per_token_per_layer(dtype),
        "num_layers": g.num_layers,
        "slowdown": slowdown.factor,
    }


def expected_layer_ms(profile: Dict, decode_tokens: int, prefill_tokens: int) -> float:
    """Interpolate the profile to estimate one layer's time for a given micro-batch."""
    pts = sorted((int(b), v) for b, v in profile["t_decode_ms"].items())
    t = 0.0
    if decode_tokens:
        b = decode_tokens
        if b <= pts[0][0]:
            t += pts[0][1]
        elif b >= pts[-1][0]:
            (b0, v0), (b1, v1) = pts[-2], pts[-1]
            t += v1 + (b - b1) * (v1 - v0) / max(1, b1 - b0)
        else:
            for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
                if b0 <= b <= b1:
                    t += v0 + (b - b0) * (v1 - v0) / (b1 - b0)
                    break
    if prefill_tokens:
        t += prefill_tokens * profile["t_prefill_ms_per_tok"]
    return t
