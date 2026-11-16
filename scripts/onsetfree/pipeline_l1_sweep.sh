#!/usr/bin/env bash
set -euo pipefail
cd /DATA/teaching/new-approach/repo
export PYTHONUNBUFFERED=1
export PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo
export OMP_NUM_THREADS=8

while [[ ! -f results/onsetfree/stage1.DONE ]]; do
  if ! kill -0 "$(cat logs/onsetfree_pipeline.pid)" 2>/dev/null; then
    echo "Baseline pipeline exited before stage1.DONE" >&2
    exit 1
  fi
  sleep 30
done

for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  for rank in 20 50 100; do
    for samples in 512 2048; do
      if [[ "$rank" == 50 && "$samples" == 2048 ]]; then
        output="results/onsetfree/l1/${setting}"
      else
        output="results/onsetfree/l1_sweep/${setting}/k${rank}_bc${samples}"
      fi
      if [[ ! -f "$output/metadata.json" ]]; then
        python3 -m scripts.onsetfree.run_l1 \
          --source-run "$source_run" \
          --counts-json "$source_run/client_class_histograms.json" \
          --output-dir "$output" --rank "$rank" \
          --curvature-samples "$samples" --seed 42
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
done
echo "L1 registered rank/sample-size sweep complete; curvature and BN ablations remain."
