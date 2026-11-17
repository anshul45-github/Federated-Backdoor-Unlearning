#!/usr/bin/env bash
set -euo pipefail
cd /home/teaching/isp
export PYTHONPATH=/home/teaching/isp PYTHONUNBUFFERED=1
PY=/home/teaching/miniconda3/bin/python
for setting in iid noniid; do
  out="results/jacobian/followup_${setting}_anchor5_w2"
  marker="results/jacobian/followup_${setting}_anchor5_w2.DONE"
  if [[ ! -s "$out/metrics.json" || ! -f "$marker" ]]; then
    "$PY" scripts/proposed/run_jvp_cal.py --trajectory "results/jacobian/${setting}_attack_logged" --out-dir "$out" --rescale --anchor-every 5 --warmup 2
    test -s "$out/metrics.json"
    touch "$marker"
  fi
done
touch results/jacobian/followup_anchors.DONE
