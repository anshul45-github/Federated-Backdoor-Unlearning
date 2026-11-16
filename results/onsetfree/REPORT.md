# Onset-free unlearning: measured report

## Verdict

On the one regenerated CIFAR-10 white-patch trajectory per partition, the **registered IID L0 half-scale ablation** meets S0 and S1 against matched FedEraser Ecal 1: 73.54% clean accuracy (CA), 1.66% attack success rate (ASR), and zero new client-training FLOPs, versus 71.99%/2.33% and 77.18 trillion client FLOPs. This is an ablation, not the prespecified default or a validation-selected method. Default L0 meets S0 only for IID and loses substantial non-IID CA. Three benign L3 repair rounds rescue CA in both partitions, but add 9.45/9.75 trillion client FLOPs and three communication rounds; their saving over Ecal 1 is about 8.17x, so they do not meet S1's 10x requirement. No measured valid method reaches S2 in either partition. The registered L2 Taylor recurrence collapses, while a momentum-aware transfer is stable but computationally expensive. All registered measurements are complete on one attacked trajectory per partition; attack-seed generalization and truly held-out benign selection remain unverified.

## Matched endpoints (test percentages)

All methods in a partition use its newly regenerated 50-round, 500-delta V2 white-patch attack trajectory and the same 10,000-image test set. ASR is over 9,000 triggered non-target images. `—` means the method is not applicable. The numbers below come from JSON outputs under the remote `results/onsetfree/` directory and the committed `summary_metrics.json`, not the older README endpoints.

| Method | IID CA | IID ASR | Non-IID CA | Non-IID ASR | Additional client FLOPs / communication |
|---|---:|---:|---:|---:|---|
| Attacked model | 74.70 | 51.93 | 71.59 | 29.32 | 0 / 0 |
| Counterfactual retraining | 74.92 | 3.08 | 70.32 | 3.41 | approximately 157.52 / 162.43 trillion; 50 rounds |
| C1 parameter subtraction | 64.31 | 0.11 | 27.15 | 1.06 | 0 / 0 |
| FedEraser Ecal 1 | 71.99 | 2.33 | 67.69 | 2.51 | 77.18 / 79.59 trillion; 49 calibration rounds |
| FedEraser Ecal 2 | 75.15 | 2.80 | 69.95 | 2.59 | 154.37 / 159.18 trillion; 49 calibration rounds |
| L0 default, rank 32, scale 1 | 67.63 | 0.24 | 34.30 | 0.69 | 0 / 0 |
| L0 registered half-scale | 73.54 | 1.66 | 56.48 | 1.78 | 0 / 0 |
| L1 default, rank 50, Bc 2048 | 70.27 | 1.34 | 43.07 | 5.84 | 14.70 trillion GGN FLOPs per partition; 0 rounds |
| L2 Taylor default, W=5, Bc=2048 | 10.00 | 100.00 | 10.00 | 100.00 | L1 spectrum + 0.735 trillion GGN FLOPs; failed approximation |
| L2 momentum transfer, W=5, Bc=2048 | 74.42 | 2.17 | 69.83 | 2.88 | 130.83 / 152.14 trillion GGN FLOPs including L1 spectrum; 0 rounds |
| L3: default L0 + 3 benign rounds | 74.67 | 1.57 | 69.16 | 2.24 | 9.45 / 9.75 trillion; 3 rounds |
| L3: half-scale L0 + 3 benign rounds | 74.93 | 3.19 | 69.67 | 3.74 | 9.45 / 9.75 trillion; 3 rounds |
| Oracle-tuned onset gate (+20; post-hoc), alpha 0 | 72.96 | 1.16 | — | — | 0 client FLOPs; **evaluation-informed, excluded from valid success claims** |

Clean-accuracy gap to the matched counterfactual (method minus counterfactual, percentage points):

| Method | IID gap | Non-IID gap |
|---|---:|---:|
| Attacked | -0.22 | +1.27 |
| C1 | -10.61 | -43.17 |
| FedEraser Ecal 1 | -2.93 | -2.63 |
| FedEraser Ecal 2 | +0.23 | -0.37 |
| L0 default | -7.29 | -36.02 |
| L0 half-scale | -1.38 | -13.84 |
| L1 default | -4.65 | -27.25 |
| L2 Taylor default | -64.92 | -60.32 |
| L2 momentum W=5/Bc=2048 | -0.50 | -0.49 |
| L3 default | -0.25 | -1.16 |
| L3 half-scale | +0.01 | -0.65 |
| Post-hoc onset gate | -1.96 | — |

FLOPs count forward/backward ATen operations via `torch.utils.flop_counter.FlopCounterMode` and batch-count scaling. They exclude optimizer arithmetic, data augmentation, server aggregation, SVD/Lanczos control overhead, communication bytes and storage. Counterfactual/FedEraser costs are derived from the matched retained-client epoch count; L3 cost uses the same count and its actual three rounds. Zero client FLOPs for L0 still entails server work. The exact integer baseline costs and endpoint hashes are in `baselines.json`; the per-batch counting method is in `iid_cost.json` and `noniid_cost.json` on the remote host.

## Prespecified success levels

| Level | Current observation |
|---|---|
| S0: ASR <10% and CA within 5 points of FedEraser Ecal 1 | Met by IID default L0, IID default L1, IID registered half-scale L0, the stable W=5/Bc=2048 L2 transfer in both partitions, and the L3 repairs in both partitions. Non-IID default L0/L1 fail CA. |
| S1: CA within 2 points, ASR <=5%, and at least 10x fewer client FLOPs than Ecal 1 | Met by **registered IID half-scale L0** on the single trajectory (CA difference +1.55 points; zero client FLOPs). Default IID L1 misses the 10x cost test. L3 repair misses the 10x test; its savings are about 8.17x. No measured non-IID method meets S1. |
| S2: match or beat Ecal 2 at lower cost | Not met by any measured valid method. IID repair reaches at most 74.93% CA versus 75.15% for Ecal 2. Stable L2 reaches 74.42%/2.17% IID versus 75.15%/2.80%, and 69.83%/2.88% non-IID versus 69.95%/2.59%; it improves one metric in IID but misses CA, and misses both metrics in non-IID. |

## Predictions and diagnostics

| Prediction | Current outcome |
|---|---|
| P1: C1 benign loss/quadratic much larger than L1 | Supported on fixed 512-example benign samples in both partitions. IID actual loss change 0.2733 vs 0.0909 and quadratic 0.3986 vs 0.1567; non-IID 6.7303 vs 0.9720 and 7.6166 vs 0.4680. Samples are disjoint from curvature estimation but come from original training clients. |
| P2: L1 per-layer cosine to counterfactual beats C1 and L0 | Refuted in both partitions. IID whole-vector cosine L1 0.0834, C1 0.2353, L0 0.1803; non-IID 0.0125, 0.1211, 0.1498. L1 beat both in only 2 of 10 comparable layers in each partition. |
| P3: onset within about 5 rounds on IID | Refuted: registered kappa values alarm at rounds 15 and 12 versus true onset 30; non-IID has no alarm. |
| P4: ASR removal versus W saturates near memory horizon | Inconclusive. All 50 L1 stiff modes had contraction <0.5 in both partitions. The stable Bc=512 transfer gave IID ASR 1.83, 1.96, 2.00, 2.07% and non-IID 3.84, 3.57, 3.47, 3.10% for W=1,2,5,10; non-IID did not clearly saturate at the predicted short horizon. The registered Taylor sweep collapsed. |

The default L0 is better than C1 in CA (+3.32 IID and +7.15 non-IID points) but does not dominate C1 in both CA and ASR. Random-subspace seeds, rank/scale sweeps, C2/C4 and BatchNorm recalibration are in `l0_sweep/` and `l0/`; see `docs/onsetfree/L0_SUMMARY.md`. IID L2 instability is detailed in `docs/onsetfree/L1_L2_SUMMARY.md`. IID gates with a detected onset at round 15 improve CA when the early flat correction is suppressed, but detector P3 fails and the non-IID detector has no alarm. The +20 delayed gate reached 72.96% CA / 1.16% ASR with zero client FLOPs; it was added after inspecting the onset error and is **evaluation-informed post-hoc**, so it is excluded from the success verdict. These gates are ablations, not a selected deployable method.

Per-source-class ASR was also evaluated for nine non-target classes. Overall rates match the original evaluations. The largest source-class ASR for default L1 was 4.1% IID and 12.3% non-IID; the latter shows that its 5.84% aggregate hides a substantially worse class. Full class tables are in remote `diagnostics/{iid,noniid}_per_class_asr.json`.

The three default L1 curvature/Lanczos seeds produced IID CA 69.82–70.27% / ASR 1.14–1.34% and non-IID CA 43.07–47.71% / ASR 5.84–8.58%. The best-CA registered rank/Bc L1 setting was rank 100, Bc 2048: 71.52%/1.76% IID and 51.71%/7.17% non-IID. At rank 20/Bc 512, exact Hessian versus GGN gave 68.74%/1.32% versus 68.84%/1.33% IID and 37.65%/18.12% versus 41.09%/12.48% non-IID. Train-stat BatchNorm GGN at the same rank/sample reached 68.01%/0.72% IID and 46.08%/4.76% non-IID. See `docs/onsetfree/L1_L2_SUMMARY.md` for the ablation interpretation. The stable L2 Bc=512 CA improved from 73.00 to 74.00% IID and 63.74 to 68.34% non-IID between W=1 and W=10, while its GGN cost rose proportionally with W.

### Compute, wall time, storage and counterfactual gap

The default L0 server construction took 8.89 s IID and 8.43 s non-IID, with no additional client training. Default L1 took 17.99 s and 17.75 s and 100 GGN-vector products per partition; the measured GGN work was 14.70 trillion FLOPs. The failed Taylor L2 W=5 added 5 GGN-vector products and 0.735 trillion FLOPs, while the stable momentum transfer W=5 added 790/935 GGN-vector products and 116.13/137.44 trillion FLOPs. These timings are remote compute spans, excluding orchestration and evaluation. Default L0 output directories occupy about 13.16 MB each; default L1 about 111.28 MB each because the rank-50 eigenbasis is saved; L2 sweep about 17.67 MB per partition. Each archived trajectory is about 1.204 GB. The stable L2 W=5/Bc=2048 CA is 0.50 points below IID counterfactual and 0.49 below non-IID counterfactual; default L3 is 0.25 and 1.16 points below respectively. Communication is 0 rounds for L0/L1/L2, 3 for L3 repair and 49 calibration rounds for FedEraser. GGN FLOPs are counted from one SmallCNN vector product times batch and call counts; they omit uncounted operations and memory transfer.

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
bash scripts/onsetfree/pipeline_l1_hessian.sh
bash scripts/onsetfree/pipeline_l1_bn.sh
bash scripts/onsetfree/pipeline_l2.sh
bash scripts/onsetfree/pipeline_l2_heavy_ball.sh
bash scripts/onsetfree/pipeline_l3.sh
bash scripts/onsetfree/pipeline_diagnostics.sh
bash scripts/onsetfree/pipeline_per_class.sh
for setting in iid noniid; do
  source_run="results/onsetfree/phase2_${setting}_attack"
  python3 -m scripts.onsetfree.run_onset \
    --source-run "$source_run" \
    --counts-json "$source_run/client_class_histograms.json" \
    --output-dir "results/onsetfree/onset/${setting}" \
    --true-onset 30  # evaluation annotation only
done
python3 -m scripts.onsetfree.run_gating \
  --source-run results/onsetfree/phase2_iid_attack \
  --counts-json results/onsetfree/phase2_iid_attack/client_class_histograms.json \
  --detector-json results/onsetfree/onset/iid/detector.json \
  --output-dir results/onsetfree/gating/iid --shifts 0 10 20
python3 -m scripts.onsetfree.evaluate_l0 \
  --model-dir results/onsetfree/gating/iid \
  --attack-config results/onsetfree/phase2_iid_attack/config.json \
  --output-json results/onsetfree/gating/iid/evaluation.json
python3 -m scripts.onsetfree.plot_results --setting iid
python3 -m scripts.onsetfree.plot_results --setting noniid
python3 -m scripts.onsetfree.collect_summary
```

Long commands were run detached with a PID and log inside `logs/`, and the long training jobs keep resumable checkpoints. Source and model paths, sample indices, seed, eigenvalues, wall time and reconstruction checks are in the relevant metadata JSON. The remote host used Python 3.10.12, PyTorch 2.6.0+cu124 and an RTX A5000; these differ from the proposal's claimed pins. The branch contains `baselines.json`, `summary_metrics.json`, 16 figures and stage markers for completed stages. The archived trajectories and model checkpoints remain under the authorized remote folder because they occupy several gigabytes.

## Limits

Only CIFAR-10, SmallCNN, one malicious client and one attacked trajectory seed per partition were run. Three seeds cover random subspaces and L1 curvature/Lanczos only, not independent attack training. Clients are simulated. The original L2 `eta_eff=eta/(1-momentum)` Taylor step is a heuristic under many local momentum steps; its collapse is evidence against using that approximation. The stable transfer uses the weighted mean local K rounded to an integer, so it is not exact per-client replay for heterogeneous non-IID clients. BatchNorm buffers are retained by default; recalibration and train-stat curvature are ablations. Adaptive attackers, variable onset, independently held-out benign selection, attack-seed replication and midpoint curvature remain untested. The post-hoc +20 gate used the observed onset error and is excluded from valid success claims. No claim of publication novelty follows from these results; see `docs/onsetfree/NOVELTY_CHECK.md` for the abstract-level literature review.
