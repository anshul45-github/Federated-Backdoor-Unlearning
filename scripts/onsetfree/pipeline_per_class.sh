#!/usr/bin/env bash
set -euo pipefail
cd /DATA/teaching/new-approach/repo
export PYTHONUNBUFFERED=1
export PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo
export OMP_NUM_THREADS=8

for setting in iid noniid; do
  source="results/onsetfree/phase2_${setting}_attack"
  while [[ ! -f "results/onsetfree/stage1.DONE" || ! -f "results/onsetfree/l1/${setting}/metadata.json" ]]; do
    if ! kill -0 "$(cat logs/onsetfree_pipeline.pid)" 2>/dev/null; then
      echo "Baseline/L1 pipeline ended before per-class evaluation was ready" >&2
      exit 1
    fi
    sleep 30
  done
  python3 -m scripts.onsetfree.evaluate_per_class \
    --attack-config "$source/config.json" \
    --model "attack=$source/final_model.pt" \
    --model "counterfactual=results/onsetfree/counterfactual_${setting}/final_model.pt" \
    --model "federaser_ecal1=results/onsetfree/federaser_${setting}_ecal1/final_model.pt" \
    --model "federaser_ecal2=results/onsetfree/federaser_${setting}_ecal2/final_model.pt" \
    --model "c1=results/onsetfree/l0/${setting}/c1.pt" \
    --model "l0_default=results/onsetfree/l0/${setting}/l0.pt" \
    --model "l0_half_scale=results/onsetfree/l0_sweep/${setting}/l0_k32_s0.5.pt" \
    --model "l1_default=results/onsetfree/l1/${setting}/l1.pt" \
    --model "l3_default=results/onsetfree/l3/${setting}/default/final_model.pt" \
    --output-json "results/onsetfree/diagnostics/${setting}_per_class_asr.json"
done
echo "Per-source-class ASR completed for both partitions."
