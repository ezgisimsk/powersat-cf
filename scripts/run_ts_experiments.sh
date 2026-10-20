#!/usr/bin/env bash
# Frame-level time-series experiments (thermal plants). Finished runs are skipped.
# usage: bash scripts/run_ts_experiments.sh "0 1 2"
cd "$(dirname "$0")/.."
mkdir -p results/ts_runs
for seed in ${1:-0}; do
  for split in random geo; do
    for v in frame_noimg frame_img; do
      out="results/ts_runs/${split}__${v}__s${seed}"
      [ -f "$out/run.json" ] && continue
      echo "=== $(date +%H:%M) $out"
      for attempt in 1 2; do
        python -I scripts/17_train_ts.py --variant "$v" --split "$split" --seed "$seed" --out "$out" > "$out.log" 2>&1 \
          && break || { echo "FAILED (attempt $attempt) $out"; cp "$out.log" "$out.attempt$attempt.log"; }
      done
    done
  done
done
echo "ts queue finished $(date +%H:%M)"
