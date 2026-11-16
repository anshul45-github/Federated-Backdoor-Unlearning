#!/usr/bin/env bash
set -euo pipefail
cd /DATA/teaching/new-approach/repo
export PYTHONUNBUFFERED=1
export PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo
export OMP_NUM_THREADS=8

mkdir -p results/onsetfree/diagnostics
for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  l1="results/onsetfree/l1/${setting}"
  while [[ ! -f "$l1/metadata.json" ]]; do
    if ! kill -0 "$(cat logs/onsetfree_pipeline.pid)" 2>/dev/null; then
      echo "Default L1 unavailable for ${setting}" >&2
      exit 1
    fi
    sleep 30
  done
  python3 -m scripts.onsetfree.diagnose_cosine \
    --attack-model "$source_run/final_model.pt" \
    --counterfactual-model "results/onsetfree/counterfactual_${setting}/final_model.pt" \
    --method-dir "$l1" \
    --output-json "results/onsetfree/diagnostics/${setting}_l1_cosine.json"
  python3 -m scripts.onsetfree.benign_validation \
    --source-run "$source_run" \
    --counts-json "$source_run/client_class_histograms.json" \
    --attack-model "$source_run/final_model.pt" \
    --method "c1=results/onsetfree/l0/${setting}/c1.pt" \
    --method "l0=results/onsetfree/l0/${setting}/l0.pt" \
    --method "l1=$l1/l1.pt" \
    --l1-metadata "$l1/metadata.json" \
    --samples 512 \
    --output-json "results/onsetfree/diagnostics/${setting}_benign_validation.json"
done
echo "Cosine and benign-validation diagnostics complete."
