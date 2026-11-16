#!/usr/bin/env bash
set -euo pipefail
cd /DATA/teaching/new-approach/repo
export PYTHONUNBUFFERED=1
export PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo
export OMP_NUM_THREADS=8

for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  cost="results/onsetfree/${setting}_cost.json"
  for variant in default half_scale; do
    if [[ "$variant" == default ]]; then
      input="results/onsetfree/l0/${setting}/l0.pt"
    else
      input="results/onsetfree/l0_sweep/${setting}/l0_k32_s0.5.pt"
    fi
    output="results/onsetfree/l3/${setting}/${variant}"
    python3 -m scripts.onsetfree.run_repair \
      --source-run "$source_run" --input-model "$input" \
      --output-dir "$output" --repair-rounds 3 --cost-json "$cost"
  done
done
echo "L3 clean repair completed for both registered L0 starting points and partitions."
