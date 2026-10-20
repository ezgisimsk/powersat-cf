#!/usr/bin/env bash
# Sequential experiment queue; finished runs (with preds_test.csv) are skipped, so it can be re-launched.
# usage: bash scripts/run_experiments.sh "0 1 2" [base|residual]
cd "$(dirname "$0")/.."
mkdir -p results/runs
SEEDS=${1:-"0"}
SET=${2:-base}
# fields: variant aux [res]
if [ "$SET" = residual ]; then
  CONFIGS=(
    "tab 0 res"
    "img 0.3 res"
    "fusion_concat 0.3 res"
    "fusion_xattn 0.3 res"
  )
else
  CONFIGS=(
    "tab 0"
    "img 0"
    "img 0.3"
    "fusion_concat 0.3"
    "fusion_xattn 0"
    "fusion_xattn 0.3"
  )
fi
for seed in $SEEDS; do
  for split in random geo; do
    for cfg in "${CONFIGS[@]}"; do
      set -- $cfg
      name=$1; flags=""
      [ "$3" = res ] && { name="$1-res"; flags="--residual"; }
      out="results/runs/${split}__${name}__aux${2}__s${seed}"
      [ -f "$out/preds_test.csv" ] && continue
      echo "=== $(date +%H:%M) $out"
      for attempt in 1 2; do  # one retry: sporadic cuDNN failures under memory pressure
        python -I scripts/09_train.py --variant "$1" --aux "$2" $flags --split "$split" --seed "$seed" --out "$out" > "$out.log" 2>&1 \
          && break || { echo "FAILED (attempt $attempt) $out (see $out.attempt$attempt.log)"; cp "$out.log" "$out.attempt$attempt.log"; }
      done
    done
  done
done
echo "queue finished $(date +%H:%M)"
