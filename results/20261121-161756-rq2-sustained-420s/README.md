# RQ2 runs (all 420 s, gpu-fast throttled 2.5x at t=120 s)

| folder | variant |
|---|---|
| `drift-off/` | no drift handling (static plan) |
| `drift-on-drain/` | drift on, pure drain-then-switch (first fixed version) |
| `drift-on-staged-only/` | + straggler migration, only zero-pause staged plans considered |
| `drift-on/` | final: + cost-aware choice between keep / staged / paused plans |

Earlier attempts kept for reference: `../_rq2-first-attempt` (before the fixes in REPORT.md section 5,
including the stalled drift-off run), `../_rq2-repro-drift-off` (stall reproduction attempt, ran clean),
`../_rq2-crashed-kind-bug` (crashed at the first replan due to a logging bug, fixed).
