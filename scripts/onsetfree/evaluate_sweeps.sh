#!/usr/bin/env bash
set -euo pipefail
cd /DATA/teaching/new-approach/repo
export PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo
export OMP_NUM_THREADS=4
for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  output="results/onsetfree/l0_sweep/${setting}"
  python3 -m scripts.onsetfree.evaluate_l0 \
    --model-dir "$output" \
    --attack-config "$source_run/config.json" \
    --output-json "$output/evaluation.json"
done
