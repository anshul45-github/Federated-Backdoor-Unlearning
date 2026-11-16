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
  output="results/onsetfree/l1_hessian/${setting}"
  if [[ ! -f "$output/metadata.json" ]]; then
    python3 -m scripts.onsetfree.run_l1 \
      --source-run "$source_run" \
      --counts-json "$source_run/client_class_histograms.json" \
      --output-dir "$output" --rank 20 --curvature-samples 512 \
      --seed 42 --curvature hessian
  fi
  if [[ ! -f "$output/evaluation.json" ]]; then
    python3 -m scripts.onsetfree.evaluate_l0 \
      --model-dir "$output" --attack-config "$source_run/config.json" \
      --output-json "$output/evaluation.json"
  fi
done
echo "Exact-Hessian spectral ablation complete for both partitions."
