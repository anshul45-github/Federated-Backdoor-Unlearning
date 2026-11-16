#!/usr/bin/env bash
set -euo pipefail
cd /DATA/teaching/new-approach/repo
export PYTHONUNBUFFERED=1
export PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo
export OMP_NUM_THREADS=8

for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  for samples in 512 2048; do
    output="results/onsetfree/l2_heavy_ball/${setting}/bc${samples}_w5"
    if [[ ! -f "$output/metadata.json" ]]; then
      python3 -m scripts.onsetfree.run_l2 \
        --source-run "$source_run" \
        --counts-json "$source_run/client_class_histograms.json" \
        --l1-spectrum "results/onsetfree/l1/${setting}/spectrum.pt" \
        --output-dir "$output" --window 5 \
        --curvature-samples "$samples" --seed 42 \
        --transfer heavy_ball
    fi
    if [[ ! -f "$output/evaluation.json" ]]; then
      python3 -m scripts.onsetfree.evaluate_l0 \
        --model-dir "$output" --attack-config "$source_run/config.json" \
        --output-json "$output/evaluation.json"
    fi
    if [[ ! -f "$output/cost.json" ]]; then
      python3 -m scripts.onsetfree.measure_ggn_flops \
        --metadata "$output/metadata.json" --output-json "$output/cost.json"
    fi
  done
done
echo "Momentum-exact aggregate-K L2 ablation complete for both partitions."
