"""Named emulated cluster presets for single-host experiments.

Each preset mirrors a heterogeneous laptop group from the proposal. On one
machine the differences are produced by real mechanisms: a CPU-only worker,
a wall-clock slowdown (the device genuinely takes longer per forward), an
advertised memory cap, and netem-style delay/jitter on every outbound link.

``hetero3``: one fast GPU laptop (8 GB), one GPU laptop that is ~2x slower
with 6 GB (the RTX 4060 vs GTX 1660 Ti case of C1), one CPU-only laptop.
Memory caps are scaled to the model so the split is memory-constrained in the
same proportions as the 8B FP16 configuration.
"""

PRESETS = {
    "hetero3": [
        {"node_id": "gpu-fast", "mem_limit_gb": 2.0, "netem": "wifi"},
        {"node_id": "gpu-slow", "slowdown": 2.0, "mem_limit_gb": 1.5, "netem": "wifi"},
        {"node_id": "cpu-only", "device": "cpu", "slowdown": 1.5, "mem_limit_gb": 1.6, "netem": "wifi"},
    ],
    "hetero3-lan": [  # same devices, no emulated Wi-Fi (isolates compute heterogeneity)
        {"node_id": "gpu-fast", "mem_limit_gb": 2.0},
        {"node_id": "gpu-slow", "slowdown": 2.0, "mem_limit_gb": 1.5},
        {"node_id": "cpu-only", "device": "cpu", "slowdown": 1.5, "mem_limit_gb": 1.6},
    ],
    "badlink3": [  # equal devices, one on a bad link (C3)
        {"node_id": "gpu-a", "mem_limit_gb": 2.0, "netem": "wifi"},
        {"node_id": "gpu-b", "mem_limit_gb": 2.0, "netem": "wifi"},
        {"node_id": "gpu-badlink", "mem_limit_gb": 2.0, "netem": "base=8,jitter=6,spike_prob=0.03,spike=40,rate=100"},
    ],
    "homog2": [
        {"node_id": "gpu-a", "netem": "wifi"},
        {"node_id": "gpu-b", "netem": "wifi"},
    ],
    "hetero4": [
        {"node_id": "gpu-fast", "mem_limit_gb": 2.0, "netem": "wifi"},
        {"node_id": "gpu-slow", "slowdown": 2.0, "mem_limit_gb": 1.5, "netem": "wifi"},
        {"node_id": "gpu-mid", "slowdown": 1.4, "mem_limit_gb": 1.5, "netem": "wifi"},
        {"node_id": "cpu-only", "device": "cpu", "slowdown": 1.5, "mem_limit_gb": 1.6, "netem": "wifi"},
    ],
}
