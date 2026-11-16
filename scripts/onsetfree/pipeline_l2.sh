#!/usr/bin/env bash
set -euo pipefail
cd /DATA/teaching/new-approach/repo
export PYTHONUNBUFFERED=1
export PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo
export OMP_NUM_THREADS=8

for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  l1_output="results/onsetfree/l1/${setting}"
  while [[ ! -f "$l1_output/metadata.json" ]]; do
    if ! kill -0 "$(cat logs/onsetfree_pipeline.pid)" 2>/dev/null; then
      echo "L1 pipeline exited before ${setting} spectrum was ready" >&2
      exit 1
    fi
    sleep 30
  done
  for samples in 512 2048; do
    for window in 1 2 5 10; do
      output="results/onsetfree/l2/${setting}/bc${samples}_w${window}"
      if [[ ! -f "$output/metadata.json" ]]; then
        python3 -m scripts.onsetfree.run_l2 \
          --source-run "$source_run" \
          --counts-json "$source_run/client_class_histograms.json" \
          --l1-spectrum "$l1_output/spectrum.pt" \
          --output-dir "$output" \
          --window "$window" --curvature-samples "$samples" --seed 42
      fi
      if [[ ! -f "$output/evaluation.json" ]]; then
        python3 -m scripts.onsetfree.evaluate_l0 \
          --model-dir "$output" --attack-config "$source_run/config.json" \
          --output-json "$output/evaluation.json"
      fi
    done
  done
done
echo "L2 GGN window and sample-size sweep complete; Hessian/BN ablations remain."
