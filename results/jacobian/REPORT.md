# JVP-Cal experiment report

## Verdict

The tested unanchored GGN configuration fails L0 on both settings: its clean accuracy is more than five percentage points below same-trajectory FedEraser E_cal=1. The single-seed anchored configuration meets the L0 thresholds in both settings. L1 and L2 are not demonstrated; a faithful FedRecover comparison and full-run FLOP measurements are unavailable. These are exploratory test-set results, not validation-selected results. Low ASR alone is not proof of unlearning.

## Same-trajectory results

| Setting | Method | CA % | ASR % | Round processing seconds |
|---|---|---:|---:|---:|
| iid | attack | 74.90 | 50.61 | 1394.13 (full attack run) |
| iid | stale | 67.08 | 0.47 | 1.45 |
| iid | FedEraser Ecal=1 | 71.55 | 2.46 | 467.11 |
| iid | FedEraser Ecal=2 | 75.14 | 2.91 | 916.45 |
| iid | GGN seed mean +/- sample SD | 64.97 +/- 0.19 | 1.65 +/- 0.03 | 11.29 +/- 0.03 |
| iid | GGN midpoint, seed 123 | 64.63 | 2.29 | 11.47 |
| iid | GGN steps=158, seed 123 | 10.17 | 0.00 | 11.27 |
| iid | GGN anchor every 5, warmup 2, seed 123 | 73.38 | 1.88 | 114.74 |
| iid | stale anchor every 5, warmup 2, seed 123 | 73.10 | 2.12 | 110.12 |
| noniid | attack | 70.80 | 30.02 | 1370.89 (full attack run) |
| noniid | stale | 31.44 | 2.66 | 1.46 |
| noniid | FedEraser Ecal=1 | 68.23 | 3.53 | 485.26 |
| noniid | FedEraser Ecal=2 | 70.01 | 2.93 | 945.29 |
| noniid | GGN seed mean +/- sample SD | 36.38 +/- 0.63 | 16.63 +/- 0.82 | 11.19 +/- 0.24 |
| noniid | GGN midpoint, seed 123 | 42.77 | 14.52 | 11.36 |
| noniid | GGN steps=158, seed 123 | 10.09 | 0.00 | 11.39 |
| noniid | GGN anchor every 5, warmup 2, seed 123 | 67.29 | 3.34 | 119.81 |
| noniid | stale anchor every 5, warmup 2, seed 123 | 66.00 | 3.32 | 120.03 |

Processing time is a sum of per-round wall times and excludes setup and evaluation; attack time is from its complete run and is not directly comparable. Values come from per-run JSON. FedEraser uses this repo's calibration logic in a new resumable runner, with different random seeds than the original baseline script. M_CF, M_sub and FedRecover B2 are not available as paired measurements; historical summaries are not mixed into this table.

## Oracle diagnostic

For 9 selected IID round/client pairs, stale vs unscaled GGN mean cosine was 0.13408 vs 0.13423; relative error was 1.29237 vs 1.29232. This small average improvement is not evidence of a robust benefit. See corrected_iid_oracle_sample.json and corrected_iid_oracle.png. The oracle uses eight examples, two full local epochs with fixed client order, and no norm rescaling; it does not exactly reproduce the 64-example rescaled estimator, and is not an exhaustive per-client diagnostic.

## Method, credibility and limitations

The estimator computes softmax generalized Gauss-Newton products using torch.func JVP/VJP and can compute exact Hessian products with double backward. The corrected replay uses the difference between replayed and stored global parameters, a fixed client subsample, eval-mode BatchNorm, historical buffer deltas and global norm rescaling. The default one-step coefficient is not an approximation validated for the true momentum SGD trajectory. A 158-step coefficient is an exploratory stress test, not a momentum correction. Anchor runs use actual one-epoch calibration. Four unit tests passed in logs/corrected_pipeline.log.

FedRecover already estimates updates from historical information using L-BFGS and correction; general historical Taylor/Hessian estimation is not novel. See docs/jacobian/NOVELTY_CHECK.md. Only one attack seed per partition, CIFAR-10 and SmallCNN are available. Test results have been observed; no held-out validation-based tuning was done. Train-mode curvature, BN recomputation, a faithful B2 implementation, per-class ASR, full FLOP accounting and server storage costs are incomplete. Resume checkpoints restore model weights but not random generator state. Anchor results and earlier seed runs should be treated as exploratory. The matched anchor schedules use the same nominal seed, but JVP data access consumes random numbers, so client augmentation is not strictly paired. The anchor comparison uses only one seed.

An earlier runner mistakenly used the historical client update as the Taylor displacement, and named a two-epoch run iid_fed_e1. Its JVP results and old plots are invalid; use corrected_ or followup_ outputs. Earlier report claims have been withdrawn.

## Reproduction

Run from /home/teaching/isp with PYTHONPATH set to the repository root. The trajectory files must already exist. Use nohup setsid env PYTHONUNBUFFERED=1 bash scripts/proposed/run_all.sh > logs/corrected_pipeline.log 2>&1 < /dev/null &. Existing progress checkpoints and stage markers control resumption. For anchors use scripts/proposed/run_followup.sh. For the sampled oracle use scripts/proposed/oracle_diagnostic.py with --replay results/jacobian/corrected_iid_ggn_s123 and an unused --out path. Tests: /home/teaching/miniconda3/bin/python -m pytest -q tests/test_jvp_cal.py. Long jobs should be detached and logged.

## FLOP probe

A single batch operator-level probe is recorded in followup_flop_probe.json. FlopCounterMode omits some operators and the probe is not an end-to-end client FLOP measurement; no 3x FLOP claim is made.
