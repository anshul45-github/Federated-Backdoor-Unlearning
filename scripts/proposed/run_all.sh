#!/usr/bin/env bash
set -euo pipefail
cd /home/teaching/isp
export PYTHONPATH=/home/teaching/isp PYTHONUNBUFFERED=1
PY=/home/teaching/miniconda3/bin/python
mkdir -p results/jacobian logs
for setting in iid noniid; do
  source="results/jacobian/${setting}_attack_logged"
  count=$(find "$source/trajectory" -name "delta_client_*.pt" | wc -l)
  [[ "$count" -eq 500 && -f "$source/trajectory/round_050/global_model.pt" ]] || { echo "Incomplete trajectory: $source ($count)"; exit 1; }
done
"$PY" -m pytest -q tests/test_jvp_cal.py
touch results/jacobian/corrected_tests.DONE
run_stage() {
  local name="$1"; shift
  local out="results/jacobian/$name"
  if [[ -f "results/jacobian/$name.DONE" && -s "$out/metrics.json" && -f "$out/COMPLETE" ]]; then echo "Skipping $name"; return; fi
  "$PY" scripts/proposed/run_jvp_cal.py --out-dir "$out" "$@"
  [[ -s "$out/metrics.json" && -f "$out/COMPLETE" ]]
  touch "results/jacobian/$name.DONE"
}
# Short, fixed configurations are exploratory; test results are not used to select them.
for setting in iid noniid; do
  for seed in 123 124 125; do
    run_stage "corrected_${setting}_ggn_s${seed}" --trajectory "results/jacobian/${setting}_attack_logged" --rescale --seed "$seed"
  done
  run_stage "corrected_${setting}_ggn_steps158" --trajectory "results/jacobian/${setting}_attack_logged" --rescale --steps 158
  run_stage "corrected_${setting}_ggn_midpoint" --trajectory "results/jacobian/${setting}_attack_logged" --rescale --point midpoint
done
for setting in iid noniid; do
  for method in fed_e1 fed_e2; do
    run_stage "corrected_${setting}_${method}" --trajectory "results/jacobian/${setting}_attack_logged" --method "$method"
  done
done
touch results/jacobian/corrected_pipeline.DONE
