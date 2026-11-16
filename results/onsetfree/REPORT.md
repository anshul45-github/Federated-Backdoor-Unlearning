# Onset-free unlearning: interim measured report

## Verdict

On the one regenerated CIFAR-10 white-patch trajectory per partition, the **registered IID L0 half-scale ablation** meets S0 and S1 against matched FedEraser Ecal 1: 73.54% clean accuracy (CA), 1.66% attack success rate (ASR), and zero new client-training FLOPs, versus 71.99%/2.33% and 77.18 trillion client FLOPs. This is an ablation, not the prespecified default or a validation-selected method. Default L0 meets S0 only for IID and loses substantial non-IID CA. Three benign L3 repair rounds rescue CA in both partitions, but add about 9.45-9.75 trillion client FLOPs and three communication rounds; their saving over Ecal 1 is only about 8.17x, so they do not meet S1's 10x requirement. The registered L2 Taylor recurrence is unstable on IID. The final S2 and cross-seed verdicts remain unverified while matched non-IID Ecal 2 and L1 sweeps run.

## Matched endpoints (test percentages)

All methods in a partition use its newly regenerated 50-round, 500-delta V2 white-patch attack trajectory and the same 10,000-image test set. ASR is over 9,000 triggered non-target images. `—` means a result is still pending or the metric is not applicable. The numbers below come from JSON outputs under the remote `results/onsetfree/` directory, not the older README endpoints.

| Method | IID CA | IID ASR | Non-IID CA | Non-IID ASR | Additional client FLOPs / communication |
|---|---:|---:|---:|---:|---|
| Attacked model | 74.70 | 51.93 | 71.59 | 29.32 | 0 / 0 |
| Counterfactual retraining | 74.92 | 3.08 | 70.32 | 3.41 | approximately 157.52 / 162.43 trillion; 50 rounds |
| C1 parameter subtraction | 64.31 | 0.11 | 27.15 | 1.06 | 0 / 0 |
| FedEraser Ecal 1 | 71.99 | 2.33 | 67.69 | 2.51 | 77.18 / 79.59 trillion; 49 calibration rounds |
| FedEraser Ecal 2 | 75.15 | 2.80 | — | — | 154.37 / 159.18 trillion; 49 calibration rounds |
| L0 default, rank 32, scale 1 | 67.63 | 0.24 | 34.30 | 0.69 | 0 / 0 |
| L0 registered half-scale | 73.54 | 1.66 | 56.48 | 1.78 | 0 / 0 |
| L1 default, rank 50, Bc 2048 | 70.27 | 1.34 | — | — | IID GGN compute 14.70 trillion ATen FLOPs; 0 rounds |
| L3: default L0 + 3 benign rounds | 74.67 | 1.57 | 69.16 | 2.24 | 9.45 / 9.75 trillion; 3 rounds |
| L3: half-scale L0 + 3 benign rounds | 74.93 | 3.19 | 69.67 | 3.74 | 9.45 / 9.75 trillion; 3 rounds |

FLOPs count forward/backward ATen operations via `torch.utils.flop_counter.FlopCounterMode` and batch-count scaling. They exclude optimizer arithmetic, data augmentation, server aggregation, SVD/Lanczos control overhead, communication bytes and storage. Counterfactual/FedEraser costs are derived from the matched retained-client epoch count; L3 cost uses the same count and its actual three rounds. Zero client FLOPs for L0 still entails server work. The full accounting method and exact integer counts are in `iid_cost.json` and `noniid_cost.json` on the remote host. The approximate non-IID costs above should be checked against `noniid_cost.json` before treating rounded figures as final.

## Prespecified success levels

| Level | Current observation |
|---|---|
| S0: ASR <10% and CA within 5 points of FedEraser Ecal 1 | Met by IID default L0, IID default L1, IID registered half-scale L0, and the measured L3 repairs in both partitions. Non-IID default L0 fails CA. |
| S1: CA within 2 points, ASR <=5%, and at least 10x fewer client FLOPs than Ecal 1 | Met by **registered IID half-scale L0** on the single trajectory (CA difference +1.55 points; zero client FLOPs). Default IID L1 misses the 10x cost test. L3 repair misses the 10x test; its savings are about 8.17x. No measured non-IID method meets S1 yet. |
| S2: match or beat Ecal 2 at lower cost | No measured IID method matches Ecal 2's 75.15% CA while maintaining its low ASR. Non-IID Ecal 2 is pending. |

## Predictions and diagnostics

| Prediction | Current outcome |
|---|---|
| P1: C1 benign loss/quadratic much larger than L1 | Supported on one IID 512-example benign sample: actual loss change 0.2733 vs 0.0909; half-delta-H-delta 0.3986 vs 0.1567. Non-IID pending. This sample is disjoint from the curvature sample but comes from the original training clients. |
| P2: L1 per-layer cosine to counterfactual beats C1 and L0 | Refuted for IID: whole-vector cosine L1 0.0834, C1 0.2353, L0 0.1803; L1 beat both in only 2 of 10 layers with all cosines defined. Non-IID pending. |
| P3: onset within about 5 rounds on IID | Refuted: registered kappa values alarm at rounds 15 and 12 versus true onset 30; non-IID has no alarm. |
| P4: ASR removal versus W saturates near memory horizon | Inconclusive: all 50 IID L1 stiff modes had contraction <0.5 (maximum 0.0002802), suggesting a short memory under the fixed-curvature model, but IID L2 first-order Taylor outputs collapse to roughly 10% CA and erratic ASR (0-100%) across W=1,2,5,10 and Bc=512/2048. |

The default L0 is better than C1 in CA (+3.32 IID and +7.15 non-IID points) but does not dominate C1 in both CA and ASR. Random-subspace seeds, rank/scale sweeps, C2/C4 and BatchNorm recalibration are in `l0_sweep/` and `l0/`; see `docs/onsetfree/L0_SUMMARY.md`. IID L2 instability is detailed in `docs/onsetfree/L1_L2_SUMMARY.md`. IID gates with a detected onset at round 15 improve CA when the early flat correction is suppressed, but detector P3 fails and the non-IID detector has no alarm. The +20 delayed gate reached 72.96% CA / 1.16% ASR with zero client FLOPs; it was added after inspecting the onset error and is **evaluation-informed post-hoc**, so it is excluded from the success verdict. These gates are ablations, not a selected deployable method.

## Reproduce inside the authorized remote folder

The remote project clone is `/DATA/teaching/new-approach/repo`. Dependencies and CIFAR-10 are also under `/DATA/teaching/new-approach`. From the clone, export `PYTHONPATH=/DATA/teaching/new-approach/deps:/DATA/teaching/new-approach/repo`. The orchestrated commands are:

```bash
bash scripts/onsetfree/pipeline_stage1_4.sh
for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  python3 -m scripts.onsetfree.run_l0_sweep \
    --source-run "$source_run" \
    --counts-json "$source_run/client_class_histograms.json" \
    --output-dir "results/onsetfree/l0_sweep/${setting}"
done
bash scripts/onsetfree/evaluate_sweeps.sh
bash scripts/onsetfree/pipeline_l1_sweep.sh
bash scripts/onsetfree/pipeline_l1_seeds.sh
bash scripts/onsetfree/pipeline_l2.sh
bash scripts/onsetfree/pipeline_l3.sh
bash scripts/onsetfree/pipeline_diagnostics.sh
```

Each long command has been run detached with a PID and log inside `logs/`, and resumable jobs keep their own checkpoints. Source and model paths, sample indices, seed, eigenvalues, wall time and reconstruction checks are in the relevant metadata JSON. The remote host used Python 3.10.12, PyTorch 2.6.0+cu124 and an RTX A5000; these differ from the proposal's claimed pins. Stage markers are withheld until their full deliverables are verified. This report is an interim measured snapshot and will be updated when pending pipelines finish.

## Limits

Only CIFAR-10, SmallCNN, one malicious client and one attacked trajectory seed per partition have been run. Clients are simulated. The L2 `eta_eff=eta/(1-momentum)` Taylor step is a heuristic under many local momentum steps; its collapse is evidence against using this implementation. BatchNorm buffers are retained by default; recalibration is an ablation. Adaptive attackers, variable onset, repeated attack seeds and per-class ASR remain untested. Rank/sample sweeps and second/third L1 seeds are still running or queued. No claim of publication novelty follows from these results; see `docs/onsetfree/NOVELTY_CHECK.md` for the abstract-level literature review.
