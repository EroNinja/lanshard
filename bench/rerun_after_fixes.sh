#!/usr/bin/env bash
# Re-runs the experiments affected by the planner/drift fixes (see REPORT.md section 6).
cd "$(dirname "$0")"
COMMON="--model Qwen/Qwen2.5-1.5B-Instruct --n 40 --rate 1.5 --max-prompt 1000 --max-out 300 --out-median 120 --prompt-median 300 --no-drift"
echo ">> serve hetero3 memory r0"; python serve.py --preset hetero3 --policy memory --repeat 0 --seed 0 $COMMON
for r in 0 1 2; do
  echo ">> serve hetero3 latency r$r"; python serve.py --preset hetero3 --policy latency --repeat $r --seed $r $COMMON
done
for r in 0 1 2; do
  echo ">> serve badlink3 latency r$r"; python serve.py --preset badlink3 --policy latency --repeat $r --seed $r $COMMON
done
echo ">> sustained"; python sustained.py --model Qwen/Qwen2.5-1.5B-Instruct --duration 420 --throttle-at 120
echo ">> plot"; python plot.py
echo "RERUN DONE"
