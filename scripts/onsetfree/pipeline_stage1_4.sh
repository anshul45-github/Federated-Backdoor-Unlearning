#!/usr/bin/env bash
set -euo pipefail

cd /DATA/teaching/new-approach/repo
export PYTHONUNBUFFERED=1
export PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo
export OMP_NUM_THREADS=8

for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  while [[ ! -f "$source_run/final_results.json" ]]; do
    if ! kill -0 "$(cat "logs/${setting}_trajectory.pid")" 2>/dev/null; then
      echo "${setting} trajectory process exited without final_results.json" >&2
      exit 1
    fi
    sleep 30
  done
  python3 -m scripts.onsetfree.verify_trajectory \
    --run-dir "$source_run" \
    --counts-json "$source_run/client_class_histograms.json" \
    --output-json "results/onsetfree/${setting}_trajectory_verification.json"
  if [[ ! -f "results/onsetfree/${setting}_cost.json" ]]; then
    python3 -m scripts.onsetfree.measure_flops \
      --counts-json "$source_run/client_class_histograms.json" \
      --output-json "results/onsetfree/${setting}_cost.json"
  fi
done

# The zero-client-work estimator can run before calibration baselines.
for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  output="results/onsetfree/l0/${setting}"
  if [[ ! -f "$output/metadata.json" ]]; then
    python3 -m scripts.onsetfree.run_l0 \
      --run-dir "$source_run" \
      --counts-json "$source_run/client_class_histograms.json" \
      --output-dir "$output" --rank 32 --scale 1 --seed 42
  fi
  if [[ ! -f "$output/evaluation.json" ]]; then
    python3 -m scripts.onsetfree.evaluate_l0 \
      --model-dir "$output" --attack-config "$source_run/config.json" \
      --output-json "$output/evaluation.json"
  fi
done

for setting in iid noniid; do
  python3 -m scripts.onsetfree.run_counterfactual \
    --config "configs/v2_${setting}.yaml" \
    --output-dir "results/onsetfree/counterfactual_${setting}"
done

for setting in iid noniid; do
  for ecal in 1 2; do
    python3 -m scripts.onsetfree.run_federaser \
      --source-run "results/onsetfree/phase2_${setting}_attack" \
      --output-dir "results/onsetfree/federaser_${setting}_ecal${ecal}" \
      --ecal "$ecal"
  done
done
python3 -m scripts.onsetfree.collect_baselines

for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  output="results/onsetfree/l1/${setting}"
  if [[ ! -f "$output/metadata.json" ]]; then
    python3 -m scripts.onsetfree.run_l1 \
      --source-run "$source_run" \
      --counts-json "$source_run/client_class_histograms.json" \
      --output-dir "$output" --rank 50 --curvature-samples 2048 --seed 42
  fi
  if [[ ! -f "$output/evaluation.json" ]]; then
    python3 -m scripts.onsetfree.evaluate_l0 \
      --model-dir "$output" --attack-config "$source_run/config.json" \
      --output-json "$output/evaluation.json"
  fi
done
echo "Default Stage 1, L0 and L1 outputs completed. Registered sweeps, L2 and report remain." 
