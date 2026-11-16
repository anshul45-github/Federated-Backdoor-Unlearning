#!/usr/bin/env bash
set -euo pipefail
cd /DATA/teaching/new-approach/repo
export PYTHONUNBUFFERED=1
export PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo
export OMP_NUM_THREADS=8

for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  while [[ ! -f "results/onsetfree/l1/${setting}/metadata.json" ]]; do
    if ! kill -0 "$(cat logs/onsetfree_pipeline.pid)" 2>/dev/null; then
      echo "Default L1 unavailable for ${setting}" >&2
      exit 1
    fi
    sleep 30
  done
  for seed in 43 44; do
    output="results/onsetfree/l1_seeds/${setting}/seed${seed}"
    if [[ ! -f "$output/metadata.json" ]]; then
      python3 -m scripts.onsetfree.run_l1 \
        --source-run "$source_run" \
        --counts-json "$source_run/client_class_histograms.json" \
        --output-dir "$output" --rank 50 --curvature-samples 2048 --seed "$seed"
    fi
    if [[ ! -f "$output/evaluation.json" ]]; then
      python3 -m scripts.onsetfree.evaluate_l0 \
        --model-dir "$output" --attack-config "$source_run/config.json" \
        --output-json "$output/evaluation.json"
    fi
  done
done
echo "Three total default L1 curvature/Lanczos seeds evaluated for both partitions."
