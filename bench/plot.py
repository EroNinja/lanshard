"""Figures and tables for the report from results/*.

    python bench/plot.py            # writes results/figures/*.png and results/figures/tables.md
"""
from __future__ import annotations

import glob
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
COL = {"latency": "#2a6fdb", "latency-additive": "#9bb7e8", "memory": "#d9822b", "even": "#8a8a8a", "latency-fixed": "#5aa469",
       "tcp": "#2a6fdb", "grpc": "#d9822b", "gloo": "#5aa469", "on": "#2a6fdb", "off": "#d9822b"}


def load(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        try:
            with open(p) as f:
                out.append((p, json.load(f)))
        except (OSError, ValueError):
            pass
    return out


def med_iqr(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None
    return float(np.median(xs)), float(np.percentile(xs, 25)), float(np.percentile(xs, 75))


def fig_rq1(lines):
    runs = load("*-serve-*/summary.json")
    if not runs:
        return
    groups = {}
    for p, s in runs:
        c = s["config"]
        key = (c.get("preset"), c["policy"])
        groups.setdefault(key, []).append(s)
    presets = sorted({k[0] for k in groups})
    policies = [p for p in ("memory", "even", "latency-fixed", "latency-additive", "latency") if any(k[1] == p for k in groups)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    w = 0.8 / max(1, len(policies))
    lines.append("\n### RQ1: partitioning policy (median [IQR] over repeats)\n")
    lines.append("| preset | policy | runs | throughput tok/s | TPOT p50 ms | TPOT p99 ms | TTFT p50 ms | bubble | plan |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for j, pol in enumerate(policies):
        tp, p99 = [], []
        for i, pre in enumerate(presets):
            ss = groups.get((pre, pol), [])
            m, lo, hi = med_iqr([s["throughput_tok_s"] for s in ss])
            q, qlo, qhi = med_iqr([s["tpot_ms"]["p99"] for s in ss])
            if m is not None:
                axes[0].bar(i + j * w, m, w, yerr=[[m - lo], [hi - m]], color=COL.get(pol), label=pol if i == 0 else None, capsize=3)
                axes[1].bar(i + j * w, q, w, yerr=[[q - qlo], [qhi - q]], color=COL.get(pol), capsize=3)
                p50 = med_iqr([s["tpot_ms"]["p50"] for s in ss])[0]
                tt = med_iqr([s["ttft_ms"]["p50"] for s in ss])[0]
                bub = med_iqr([s.get("mean_bubble_ratio") for s in ss])[0]
                lines.append(f"| {pre} | {pol} | {len(ss)} | {m:.1f} [{lo:.1f}-{hi:.1f}] | {p50:.0f} | {q:.0f} [{qlo:.0f}-{qhi:.0f}] | "
                             f"{tt:.0f} | {bub:.2f} | {ss[-1]['plan'].split(': ', 1)[-1]} |")
    for ax, t in zip(axes, ("aggregate throughput (tok/s)", "p99 TPOT (ms)")):
        ax.set_xticks([i + w * (len(policies) - 1) / 2 for i in range(len(presets))])
        ax.set_xticklabels(presets)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rq1_policy.png"), dpi=150)
    plt.close(fig)


def fig_rq2(lines):
    runs = load("*-rq2-sustained-*/drift-*/summary.json")
    if not runs:
        return
    latest = {}
    for p, s in runs:
        latest[s["tag"]] = s  # newest wins (sorted by stamp)
    colors = {"drift-off": "#d9822b", "drift-on-drain": "#9bb7e8", "drift-on-staged-only": "#7a9a5a",
              "drift-on": "#2a6fdb"}
    labels = {"drift-off": "drift off (B2-style static plan)", "drift-on-drain": "drift on, pure drain-then-switch",
              "drift-on-staged-only": "drift on + migration, staged-only (declined)",
              "drift-on": "drift on + migration + cost-aware choice (final)"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    lines.append("\n### RQ2: sustained run, gpu-fast throttled 2.5x at t=120 s\n")
    lines.append("| variant | tok/s before | tok/s steady after | retained | tokens in 300 s after throttle | switch pause | TPOT p99 ms |")
    lines.append("|---|---|---|---|---|---|---|")
    thr = None
    for tag in ("drift-off", "drift-on-drain", "drift-on-staged-only", "drift-on"):
        s = latest.get(tag)
        if not s:
            continue
        t, x = s["series"]["t"], s["series"]["tok_s"]
        ax.plot(t, x, color=colors[tag], label=labels[tag], lw=1.5)
        thr = thr or next((e["t"] for e in s["events"] if e["event"] == "set_slowdown"), None)
        sw = [e for e in s["events"] if e["event"] == "switch" and e.get("old")]
        for e in sw:
            ax.axvline(e["t"], color=colors[tag], ls=":", lw=1)
        after = sum(v * (t[1] - t[0]) for tt, v in zip(t, x) if thr and thr <= tt < thr + 300)
        pre, post = s["throughput_pre"], s["throughput_post"]
        pause = ", ".join(f"{e['stage_s']:.1f}s" for e in sw) or "-"
        lines.append(f"| {labels[tag]} | {pre:.1f} | {post:.1f} | {post / pre:.0%} | {after:.0f} | {pause} | "
                     f"{s['tpot_ms']['p99']:.0f} |")
    if thr:
        ax.axvline(thr, color="red", lw=1, label="throttle injected")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tokens/s (5 s bins)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rq2_sustained.png"), dpi=150)
    plt.close(fig)


def fig_rq3(lines):
    runs = load("*-rq3-transport*/summary.json")
    if not runs:
        return
    for p, s in runs:
        tag = os.path.basename(os.path.dirname(p)).split("rq3-transport")[-1].strip("-") or "loopback"
        rows = s["rows"]
        backends = [b for b in ("tcp", "grpc", "gloo") if any(r["backend"] == b for r in rows)]
        sizes = sorted({r["bytes"] for r in rows})
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        small = [z for z in sizes if z <= 16384]
        x = np.arange(len(small))
        w = 0.8 / len(backends)
        for j, b in enumerate(backends):
            rs = {r["bytes"]: r for r in rows if r["backend"] == b}
            ser = [rs[z]["ser_p50"] for z in small]
            de = [rs[z]["deser_p50"] for z in small]
            tx = [rs[z]["transfer_p50"] for z in small]
            xs = x + j * w
            axes[0].bar(xs, ser, w, color=COL[b], alpha=1.0, label=f"{b}")
            axes[0].bar(xs, de, w, bottom=ser, color=COL[b], alpha=0.6)
            axes[0].bar(xs, tx, w, bottom=np.add(ser, de), color=COL[b], alpha=0.3)
            axes[1].plot(sizes, [rs[z]["one_way_p50"] for z in sizes], "o-", color=COL[b], label=b)
        axes[0].set_xticks(x + w * (len(backends) - 1) / 2)
        axes[0].set_xticklabels([f"{z // 1024} KB" for z in small])
        axes[0].set_ylabel("one-way ms (ser | deser | transfer)")
        axes[0].set_title(f"decode payloads ({tag})")
        axes[0].legend(frameon=False)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("payload bytes")
        axes[1].set_ylabel("one-way p50 ms")
        axes[1].set_title("payload sweep (decode -> prefill)")
        axes[1].legend(frameon=False)
        for a in axes:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"rq3_transport_{tag}.png"), dpi=150)
        plt.close(fig)
        lines.append(f"\n### RQ3: per-hop latency ({tag})\n")
        lines.append("| payload | backend | RTT p50 ms | RTT p99 ms | ser ms | deser ms | transfer ms | added vs TCP (one-way) ms |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: (r["bytes"], r["backend"])):
            lines.append(f"| {r['bytes'] // 1024} KB | {r['backend']} | {r['rtt_p50']:.3f} | {r['rtt_p99']:.3f} | "
                         f"{r['ser_p50']:.3f} | {r['deser_p50']:.3f} | {r['transfer_p50']:.3f} | {r.get('added_vs_tcp_ms', 0):+.3f} |")


def tables_misc(lines):
    q = load("*-quality/summary.json")
    if q:
        p, s = q[-1]
        lines.append("\n### Quality vs B1 (single-machine HF)\n")
        lines.append(f"Plan: `{s['plan']}`\n")
        lines.append("| task | n | greedy exact match | B1 accuracy | LANShard accuracy | answer agreement |")
        lines.append("|---|---|---|---|---|---|")
        for k in ("exact", "mmlu", "gsm8k"):
            if k in s:
                r = s[k]
                acc = lambda v: f"{v:.3f}" if v is not None else "-"  # noqa: E731
                lines.append(f"| {k} | {r['n']} | {r['exact_match']:.3f} | {acc(r.get('acc_b1'))} | "
                             f"{acc(r.get('acc_lanshard'))} | {acc(r.get('answer_agreement'))} |")
    b1 = load("*-b1-single/summary.json")
    if b1:
        p, s = b1[-1]
        lines.append("\n### B1 single-machine reference\n")
        lines.append(f"Throughput {s['throughput_tok_s']:.1f} tok/s (one request at a time), TPOT p50 "
                     f"{s['tpot_ms']['p50']:.0f} ms / p99 {s['tpot_ms']['p99']:.0f} ms, peak VRAM {s['peak_alloc_gb']:.2f} GB.")
    serve = load("*-serve-*/run.json")
    if serve:
        p, s = serve[-1]
        lines.append("\n### Device table (from the discovery probe, latest run)\n")
        lines.append("| node | OS | accelerator | device | free mem GB (advertised) | emulated slowdown | t_layer b=1 ms | t_layer b=4 ms | prefill ms/tok | layer MB |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for nid, n in s["status"]["nodes"].items():
            pr = n.get("profile") or {}
            td = pr.get("t_decode_ms", {})
            lines.append(f"| {nid} | {n.get('os')} | {n.get('accel')} | {pr.get('dtype', '')} | {n['mem_free_gb']} | "
                         f"{(n.get('stats') or {}).get('slowdown', '')} | {td.get('1', 0):.2f} | {td.get('4', 0):.2f} | "
                         f"{pr.get('t_prefill_ms_per_tok', 0):.4f} | {pr.get('mem_layer', 0) / 2 ** 20:.0f} |")


def main():
    os.makedirs(FIG, exist_ok=True)
    lines = ["# LANShard result tables (generated by bench/plot.py)"]
    for f in (fig_rq3, fig_rq1, fig_rq2, tables_misc):
        try:
            f(lines)
        except Exception as e:  # keep going so one broken run does not hide the rest
            lines.append(f"\n> {f.__name__} failed: {e!r}\n")
    with open(os.path.join(FIG, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("figures in", FIG)


if __name__ == "__main__":
    main()
