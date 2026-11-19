# Federated backdoor unlearning — project experiment review

**As of 18 November 2026.** Covers the historical 20-client prototype, the 50-round white-patch and checkerboard benchmarks, the TCUG experiments, and the regenerated onset-free study. The [experiment appendix](PROJECT_EXPERIMENT_APPENDIX.md) has every saved onset-free sweep cell, the complete TCUG run inventory, and links to all component patch candidates and round-level results.

## What the project found

- **Subtraction was fast and often erased the trigger response at a large cost to utility.** On the original non-IID white-patch run, CA fell from 69.80% to 28.04% as ASR fell from 32.21% to 0.87%. On the extreme-skew checkerboard run, subtraction and KD both collapsed to 100% target prediction. Two-epoch FedEraser was the strongest conventional utility-preserving baseline across these saved settings.
- **The strongest saved checkerboard repair trade-off came from training-free TCUG-2 followed by three clean rounds.** At round 53 it reached **75.57% CA / 0.16% ASR IID** and **70.70% / 0.24% non-IID**, improving both endpoint metrics over the saved round-50 FedEraser checkpoint in each partition. It used the known pre-attack donor window and additional labeled client training. The checkerboard study has no independently retrained client-excluded reference.
- **Counterfactual fidelity is a separate outcome.** In the historical factorial, FedEraser's clean agreement with retraining fell from 89.63% to 64.13% when high label skew and size imbalance co-occurred, even though CA stayed near 75%. The effect is descriptive on one partition per cell.
- **The onset-free study found an inexpensive IID result and a persistent non-IID utility problem.** Registered half-scale L0 at rank 32 reached **73.54% / 1.66% IID** with zero new client training. Its non-IID counterpart reached **56.48% / 1.78%**. Stable curvature transfer and three clean rounds recovered much of the utility at higher cost. The prespecified Taylor recurrence and onset detector failed.

## Protocols and metrics

CA means clean accuracy. ASR is the share of triggered non-target inputs predicted as the target class. UTR is the target prediction rate on the same non-target inputs without a trigger. CF agreement is prediction agreement with an independently retrained client-excluded model, where available. Values are percentages unless marked as percentage points (pp), seconds, or FLOPs. Tables with paired values use **CA / ASR** unless specified.

The historical benchmark used CIFAR-10, SmallCNN, 20 clients with 10 selected per round, and 100 rounds. Its frozen evaluation uses 7,500 clean images, with target examples excluded for ASR and 2,500 separate CIFAR-10 examples reserved for historical KD. Some preliminary training and evaluation reports use other populations. The later protocol uses CIFAR-10 and SmallCNN with 10 clients, full participation during base training, sample-weighted FedAvg, 50 rounds, two local epochs, and seed 42. IID and the saved Dirichlet α = 0.5 draw are the main partitions. CA uses 10,000 clean images, ASR and UTR use 9,000 non-target images, and the target is class 0 (airplane). Checkerboard component selection uses two half-test splits. [Historical benchmark](baseline/canonical_benchmark.md) · [50-round protocol](baseline/EXPERIMENT_PROTOCOL.md) · [Metrics](baseline/METHODS_AND_METRICS.md)

## 1. Historical prototype and unlearning baselines

The first milestone was centralized CIFAR-10 training. The archive records the runner and default 10-epoch configuration. It also names the subsequent clean IID FedAvg run, an initial BadNets run, Stage 4 tuning attempts, and the finalized Stage 3B attack. The centralized, initial FedAvg, initial BadNets, and intermediate tuning endpoints could not be recovered from the inspected files. The frozen Stage 3B attack and the main Stage 4–6C baseline evaluations survive in the reports. [Original project README](historical/SOURCE_PROJECT_README.md) · [Frozen benchmark](baseline/canonical_benchmark.md) · [Archive inventory](historical/storage/storage_archive_verification.md)

| Historical endpoint | CA | ASR | Clean CF agreement | Wall time |
|---|---:|---:|---:|---:|
| Stage 3B attacked | 74.31 | 87.15 | 85.25 | — |
| Stage 4 retraining reference | 74.48 | 3.02 | Reference | ~1,800 s |
| Stage 5A subtraction | 10.53 | 0.00 | 10.00 | ~4.85 s |
| Stage 5B subtraction + KD | 56.00 | 3.09 | 63.89 | ~15 s |
| Stage 6 FedEraser | 75.00 | 2.24 | 87.40 | ~1,680 s |

Stage 6C measured relative parameter distances to Stage 4 of **82.77%** for subtraction, **101.76%** for KD, and **60.60%** for FedEraser. The report evaluated the original patch, a shifted patch, and a larger patch. The original and shifted evaluations report identical rates for all five endpoints, so the shifted result is best read as an archived observation. The larger-patch ASRs were **49.42** attacked, **2.81** reference, **0.00** subtraction, **3.49** KD, and **2.03** FedEraser. Stage 6D later found that Stage 4 filled the excluded client's slot by resampling, yielding a different benign participation history. [Stage 6C analysis](historical/6c/stage_6c_analysis.md) · [Stage 6D reference audit](historical/6d/stage_6d_phase_a_report.md)

### Stage 6D — timing and natural forgetting

The first timing proposal was deferred for storage. The executed pilot used equal exposure, with 15 malicious client participations in each timing arm. Early ASR peaked at 100.00% and fell to 5.36% before repair. Late ASR remained 85.83% at the end. Phase A recorded **74.89% CA / 0.00% ASR** for the no-attack control and **75.07% CA** for the matched client-excluded control, which agreed with the earlier Stage 4 reference on 83.22% of predictions. The later post-B1 review gives **74.76% CA / 2.32% ASR** for its evaluation reference. Phase B's own pre-unlearning readings are preserved because the reports use different measurements. [Phase A](historical/6d/stage_6d_phase_a_report.md) · [Phase B design](historical/6d/stage_6d_phase_b_audit.md) · [Post-B1 review](historical/post_b1_research_review.md)

| Timing | Phase A final CA / ASR | Peak ASR | Phase B pre-repair CA / ASR | Subtraction CA / ASR / agreement | KD CA / ASR / agreement | FedEraser CA / ASR / agreement |
|---|---:|---:|---:|---:|---:|---:|
| Early | 73.91 / 5.36 | 100.00 | 73.63 / 5.33 | 33.35 / 0.01 / 35.27 | 67.73 / 2.59 / 78.31 | 74.03 / 2.37 / 89.00 |
| Middle | 74.93 / 16.67 | 84.59 | 74.68 / 16.61 | 38.76 / 0.13 / 40.83 | 67.20 / 2.74 / 77.76 | 74.68 / 2.59 / 88.48 |
| Late | 73.78 / 85.83 | 85.83 | 73.57 / 85.82 | 37.72 / 0.34 / 41.31 | 65.08 / 2.29 / 74.01 | 75.07 / 3.05 / 89.04 |

Across the nine repairs, clean CF agreement spanned 6.04 pp for subtraction, 4.30 pp for KD and 0.56 pp for FedEraser. Recorded times for Early/Middle/Late were 7.4/6.7/6.7 s for subtraction, 5.8/5.8/5.8 s for KD's timer, and 1,614.5/1,663.0/1,617.1 s for FedEraser. All three attacks still changed 16–21% of predictions relative to the no-attack model at the endpoint. The final preservation record says the raw Stage 6D checkpoints and deltas were deleted to regain about 11.5 GB. [Nine-run report](historical/6d/stage_6d_phase_b_report.md) · [Preservation record](historical/6d/stage6d_regeneration_record_final.md)

### Stages 7A–7B — heterogeneity and scaling

7A trained four regimes. The 7B audit found that 7A's counterfactual exclusion argument was ineffective, evaluation overlapped the KD proxy, best-test checkpoint selection leaked evaluation data, and integer BatchNorm counters inflated geometry statistics. Use 7B's corrected recovery rows for interpretation. The appendix preserves the superseded 7A rows with the original labels. [7A preflight](historical/7a/stage_7a_preflight_report.md) · [7A report](historical/7a/stage_7a_final_report.md) · [7B corrections](historical/7b/stage_7b_corrected_report.md)

| Regime | 7A attacked CA / ASR | 7A clean-control CA | 7A peak ASR | 7B true CF CA |
|---|---:|---:|---:|---:|
| C1, IID | 74.23 / 87.26 | 74.89 | 87.26 | 74.76 |
| C2, α = 0.5 | 72.40 / 82.44 | 72.29 | 88.82 | 71.97 |
| C3, α = 0.1 | 61.75 / 67.34 | 60.25 | 100.00 | 59.24 |
| C4, α = 0.1 with stronger scaling | 60.51 / 81.58 | 60.25 | 100.00 | 59.24 |

| Corrected regime | Subtraction CA / ASR / agreement | KD CA / ASR / agreement | FedEraser CA / ASR / agreement |
|---|---:|---:|---:|
| C1 | 10.53 / 0.00 / 10.88 | 56.36 / 4.71 / 63.45 | 75.01 / 2.38 / 89.55 |
| C2 | 11.48 / 0.00 / 13.17 | 48.53 / 5.51 / 59.97 | 74.24 / 2.04 / 82.80 |
| C3 | 11.23 / 0.00 / 19.80 | 30.45 / 0.77 / 50.11 | 74.25 / 2.40 / 64.44 |
| C4 | 10.01 / 0.00 / 14.79 | 21.71 / 0.00 / 36.11 | 74.19 / 2.26 / 61.91 |

In strong non-IID, FedEraser had about 15 pp higher CA than true retraining while agreeing on only 62–64% of predictions. The corrected benign norm coefficient of variation was 0.10 IID versus 0.32 under strong heterogeneity. The correction retained the dispersion pattern with much smaller absolute norms. Stronger scaling lowered FedEraser agreement another 2.53 pp in this partition. [7B results and geometry](historical/7b/stage_7b_corrected_report.md)

### Stage 7C — factorial and recovery ablations

The canonical factorial varied label heterogeneity and client-size imbalance independently. D1 balanced all 20 clients at 2,500 examples while skewing labels. D2 kept labels close to uniform with client sizes from 116 to 12,266 and size CV 1.11. D0 and D3 use the recomputed factorial evaluations, which differ slightly from 7B. [Canonical synthesis](historical/7c/stage7c_factorial_synthesis_v2.md) · [Attack-strength audit](historical/7c/stage7c_final_factorial_robustness_audit.md)

| Condition | Label skew / size imbalance | Attacked CA / ASR | CF CA / ASR | FedEraser CA / ASR / agreement |
|---|---|---:|---:|---:|
| D0 | Low / low | 74.31 / 87.15 | 74.72 / 2.38 | 74.77 / 2.52 / 89.63 |
| D1 | High / low | 74.23 / 36.84 | 74.84 / 2.78 | 74.85 / 1.60 / 87.09 |
| D2 | Low / high | 77.61 / 72.48 | 77.40 / 2.66 | 72.95 / 1.89 / 81.83 |
| D3 | High / high | 61.76 / 67.35 | 60.29 / 1.35 | 74.65 / 2.24 / 64.13 |

- **D1 controls:** Subtraction **50.73 / 0.86 / 54.47** and KD **72.41 / 1.29 / 82.96**. The exploratory D1 stronger-attack run reached **74.19 / 51.87** before repair. Its subtraction **54.69 / 0.34 / 58.67**, KD **71.84 / 1.49 / 82.05**, and FedEraser **74.71 / 2.90 / 88.20** use a different attack configuration and sit outside the canonical factorial. An earlier D1 attempt was interrupted. [Canonical D1](historical/7c/stage7c_d1_canonical_analysis.md) · [Exploratory D1](historical/7c/stage7c_d1_analysis.md) · [Storage audit](historical/storage/storage_archive_verification.md)
- **D2 controls:** Subtraction **10.77 / 0.00 / 9.68** and KD **58.24 / 4.06 / 65.20**. The first D2 run stopped at round 29 without state for an exact resume. A restarted run used the wrong scaling and was invalidated. The completed restart added round checkpoints, RNG snapshots and completion markers. The metrics audit corrected a utility-gap/CF-gap mix-up and double-counted update scaling. [Recovery audit](historical/7c/stage_7c_recovery_audit.md) · [Restart report](historical/7c/stage_7c_d2_recovery_report.md) · [Corrected D2](historical/7c/stage7c_d2_analysis_corrected.md)
- **Factorial contrast:** Agreement changed −2.54 pp for label skew alone and −7.80 pp for size imbalance alone. The additive prediction was 79.29%; measured D3 was 64.13%, an interaction of **−15.16 pp**. Realized attacked ASR varied from 36.84% to 87.15% across cells. The final audit found the same ordering after scaling CF gap by initial prediction disagreement. This supports an interaction in the observed data and leaves replication and causal mechanism open. [Final robustness audit](historical/7c/stage7c_final_factorial_robustness_audit.md)

## 2. Main 50-round white-patch benchmark

Phase 1 trained clean and attacked models in both partitions. Phase 2 regenerated attacked trajectories with logging, trained independent C1–C9 references, and tested weighted subtraction, KD from the subtracted checkpoint with the attacked teacher, and FedEraser with one or two calibration epochs. The Phase 2 logged attack endpoints in the canonical comparison match the Phase 1 metrics shown here. The three-round IID smoke run gave **42.29 / 5.52** for clean training and **39.73 / 12.32** for attacked training. [Phase 1 results](../results/phase1/phase1_summary.csv) · [Phase 2 audit](../results/phase2/audit_stage1/stage1_audit_report.md) · [Smoke output](../results/smoke/smoke_test_dir/final_results.json)

Historical subtraction removes the sum of C0's sample-weighted updates across all logged rounds, including its clean rounds. It edits the floating-point BN statistics as well as the parameters. KD then trains the subtracted model for five epochs on 5,000 balanced CIFAR-100 proxy images, using temperature-4 KL distillation and frozen student BN statistics. FedEraser starts from the saved initialization, reuses the first retained-client deltas, and then recalibrates retained-client updates on the evolving reconstructed model. It scales each trainable calibration update to the L2 norm of its stored counterpart. The corrected reconstruction bypasses norm scaling for BN running statistics and preserves BN counters. [Implementation walkthrough](PROJECT_WALKTHROUGH.md) · [FedEraser run registry](../results/phase2/federaser/RUN_REGISTRY.md)

| Endpoint | IID CA / ASR | Non-IID CA / ASR | IID / non-IID clean CF agreement |
|---|---:|---:|---:|
| Clean training | 75.11 / 2.73 | 70.10 / 3.30 | 93.01 / 90.21 |
| Attacked training | 75.01 / 56.07 | 69.80 / 32.21 | 92.77 / 90.28 |
| Historical subtraction | 66.88 / 0.24 | 28.04 / 0.87 | 76.25 / 35.31 |
| Subtraction + KD | 72.82 / 2.40 | 60.36 / 3.69 | 88.54 / 78.65 |
| FedEraser Ecal 1 | 72.11 / 2.67 | 65.75 / 3.53 | 84.95 / 85.34 |
| FedEraser Ecal 2 | 74.99 / 2.34 | 68.89 / 3.09 | 89.29 / 88.18 |
| Independent retraining | 74.87 / 3.01 | 69.64 / 3.40 | Reference |

- **Subtraction and KD:** Non-IID subtraction's macro F1 was 17.10%, airplane recall was 14.20%, and 41.48% of clean predictions went to class 8. KD recovered 32.32 pp of non-IID CA, and its ASR was 0.29 pp above the retraining reference. Low ASR alone would have missed the subtraction failure. [Canonical metrics](../results/phase2/federaser/corrected_evaluation_v2/federaser_comparison.csv) · [Class distribution](../results/phase2/audit_stage1/noniid_m_sub_predicted_distribution.csv)
- **Calibration epochs and corrected runs:** Ecal 2 gained 2.88 pp IID and 3.14 pp non-IID CA over Ecal 1, with lower ASR in both. The initial two FedEraser runs inherited weight decay and used incorrect UTR evaluation. Four subsequent Ecal runs were superseded because calibration loaders were unseeded and integer BN counters drifted. The appendix has all ten saved FedEraser endpoints. [Run registry](../results/phase2/federaser/RUN_REGISTRY.md) · [Final comparison](../results/phase2/federaser/corrected_evaluation_v2/federaser_comparison.csv)
- **Timing:** Subtraction was 0.18/3.77 s, subtraction + KD 7.82/11.29 s, Ecal 1 reconstruction 630/636 s, Ecal 2 reconstruction 1,211/1,258 s, and historical retraining 2,514/4,219 s (IID/non-IID). Timer boundaries vary. A controlled three-round local-work microbenchmark measured 75.08 s for retraining-style work and 75.82 s for FedEraser with reads, including 0.21 s of I/O. [Timing ledger](../results/runtime_ledger.csv) · [Controlled timing](../results/phase2/federaser/controlled_timing.json)

## 3. Checkerboard benchmark, component patching, and extreme skew

The checkerboard reached **92.18% IID** and **80.89% non-IID ASR**, versus **56.07%** and **32.21%** in the original white-patch records, at comparable CA. These are separate full runs. The first IID checkerboard folder has configuration only, and the completed rerun supplies the IID results and trajectory. The saved KD repairs used a balanced 5,000-image CIFAR-100 training proxy. [Checkerboard results and protocol](checkerboard/CHECKERBOARD_UNLEARNING.md) · [IID final result](../results/checkerboard/iid_20261024_015516_rerun_001/final_results.json) · [Non-IID final result](../results/checkerboard/noniid_20261024_150813_189/final_results.json)

| Endpoint | IID CA / ASR | Non-IID α = 0.5 CA / ASR | α = 0.01 CA / ASR / UTR / macro F1 |
|---|---:|---:|---:|
| Attacked | 75.26 / 92.18 | 71.14 / 80.89 | 33.48 / 99.97 / 1.19 / 26.30 |
| Subtraction | 68.21 / 0.02 | 31.25 / 0.13 | 10.00 / 100.00 / 100.00 / 1.82 |
| Subtraction + KD | 73.09 / 0.41 | 59.35 / 0.79 | 10.00 / 100.00 / 100.00 / 1.82 |
| FedEraser Ecal 2 | 75.19 / 1.18 | 70.18 / 0.94 | 34.86 / 6.57 / 13.92 / 28.41 |

**Extreme skew.** The α = 0.01 series includes a partition-only preflight and a completed seed-42 run. C0 held 9,999 images across two classes, about 20% of the training set. At this skew, subtraction and KD labeled every clean non-target image as the target. FedEraser reduced ASR by 93.40 pp and raised CA by 1.38 pp over the attacked model. Its UTR was still 13.92% and CA remained 34.86%. [Stress-test result](../results/label_skew/alpha001_20261029_214828_723364/seed_42/report.md) · [Stress-test protocol](label_skew/EXTREME_LABEL_SKEW.md)

**Component patching (IID).** A two-stage search blended historical Conv2 channels and their BN components into G50. Stage 1 evaluated all 64 channels at 12 donor-round/blend combinations. Stage 2 advanced G29 at blend 0.25, tested 64 single channels, and evaluated 780 prefixes across 12 orders including random rankings. The selected 42-channel prefix met the one-point selection CA-loss budget. Selection CA/ASR moved **75.14 / 91.82 → 75.20 / 88.58**. Held-out CA/ASR moved **75.38 / 92.53 → 75.12 / 88.58**, a 3.95 pp reduction in ASR with 0.26 pp CA loss. The manifest records zero optional calibration steps. The residual ASR remained high. All candidate tables and order winners are linked in the appendix. [Patch report](../results/checkerboard/iid_20261024_015516_rerun_001/component_patching/report.md) · [Protocol](checkerboard/COMPONENT_PATCHING.md)

## 4. Temporal Clean Update Grafting (TCUG)

### Rollback and continuation

G29 precedes the attack and contains C0's earlier clean updates. Each method trains C1–C9 for rounds 30–50. TCUG-0 aggregates the retained clients. TCUG-1 also grafts a static mean of C0's five clean deltas from rounds 25–29. TCUG-2 reranks the five donors each round by cosine to the current retained-client update and averages the top three. Grafting uses C0's original FedAvg weight (10.00% IID, 7.194% non-IID) for trainable parameters, while retained clients supply BN buffers. The default cap never clipped a graft in the completed TCUG-1/2 runs. [TCUG results and settings](../README.md) · [Run-by-run endpoints](PROJECT_EXPERIMENT_APPENDIX.md#tcug-continuations-and-repairs)

| Endpoint | IID CA / ASR | Non-IID CA / ASR |
|---|---:|---:|
| G29 anchor | 70.42 / 1.77 | 66.17 / 1.68 |
| FedEraser Ecal 2, reevaluated | 75.19 / 1.18 | 70.18 / 0.94 |
| TCUG-0, CPU | 75.17 / 1.34 | 69.97 / 2.09 |
| TCUG-0, earlier GPU batch | 75.14 / 1.53 | 70.01 / 1.83 |
| TCUG-1, earlier GPU batch, round 46 | 74.57 / 1.37 | — |
| TCUG-0, matched main GPU batch | 75.43 / 1.44 | 70.27 / 1.89 |
| TCUG-1, matched main GPU batch | 74.99 / 1.62 | 70.93 / 1.87 |
| TCUG-2, completed GPU batch | 75.09 / 1.49 | 70.59 / 1.77 |

Grafting reduced IID CA relative to TCUG-0 in the completed GPU batch comparison. In non-IID, TCUG-1 gained 0.66 pp CA over TCUG-0, and TCUG-2 gave the lowest continuation ASR. FedEraser had lower ASR than all three continuations in each partition. Across the CPU and GPU TCUG-0 runs, CA spanned 0.29 pp IID and 0.30 pp non-IID. The median donor-norm cap was 2.1177 IID and 1.8051 non-IID. TCUG-1 graft norms were 1.1497 and 1.2229, and the maximum TCUG-2 norms were 1.3742 and 1.3450, respectively. Every graft kept clip scale 1.0. [GPU batches](../results/tcug_gpu/) · [CPU rollback](../results/rollback/) · [Clipping results](../README.md)

### Direct checkpoint edits and three-round healing

| Endpoint | IID CA / ASR | Non-IID CA / ASR |
|---|---:|---:|
| Training-free TCUG, fixed donor mean | 72.86 / 0.09 | 69.67 / 0.12 |
| Training-free TCUG-2, top-three donors, CPU | 71.92 / 0.04 | 68.52 / 0.11 |
| Training-free TCUG-2, GPU reevaluation | 71.91 / 0.04 | 68.52 / 0.11 |
| Training-free TCUG-2 + three clean rounds | **75.57 / 0.16** | **70.70 / 0.24** |
| Rolling training-free TCUG-2 | 50.68 / 0.03 | 40.86 / 0.24 |

- **Fixed mean and top-three donors:** Both replace the weighted C0 contribution at the endpoint for rounds 30–50 with estimates from rounds 20–29. Fixed-mean TCUG adjusts all floating tensors, including BN buffers, and uses no cap. TCUG-2 selects by retained-update cosine and keeps attacked BN buffers. It had no clipping in either partition. The fixed mean retained 0.94 pp more CA IID and 1.15 pp more non-IID. Donor selection and BN policy both change in that comparison. [Fixed mean](../results/training_free_tcug/) · [Top-three](../results/training_free_tcug_2/)
- **Healing:** Compared with its GPU starting checkpoint, training-free TCUG-2 plus three rounds recovered **3.66 / 2.18 pp** CA (IID/non-IID), with ASR rising **0.11 / 0.13 pp** using unrounded values. Versus the FedEraser endpoints, it gained **0.38 / 0.52 pp** CA and reduced ASR by **1.02 / 0.70 pp**. It finished at round 53 after nine-client FedAvg for three additional rounds. [GPU follow-up](../results/tcug_followup_gpu/20261115_144254/)
- **Rolling window:** This variant retained rounds 1–5, then recursively reconstructed rounds 6–50 using the preceding five C0 updates. Clipping occurred in 14/45 IID and 20/45 non-IID rounds. It had much lower CA than fixed-window TCUG-2 in both partitions. Donor recursion and the number of replaced rounds changed together. [Rolling-window results](../results/training_free_rolling_tcug_2/)

### Direction-preserving TCUG and selective healing

The direction-preserving edit subtracts all 50 recorded C0 parameter contributions, then adds back a fraction λ of C0's positive projection onto the sample-weighted retained-client direction. It requires the client identity and historical updates. It uses no onset information and retains attacked BN buffers. The norm cap activated in 0/50 rounds per partition. Positive alignments occurred in 50 IID and 34 non-IID rounds. [Projection results and histories](../results/direction_preserving_tcug/)

| Endpoint | IID CA / ASR | Non-IID CA / ASR |
|---|---:|---:|
| λ = 0, parameter-only control | 65.21 / 0.00 | 27.54 / 0.00 |
| λ = 0.25 | 65.28 / 0.00 | 27.53 / 0.00 |
| λ = 0.5 | 65.43 / 0.00 | 27.52 / 0.01 |
| λ = 1 | 65.57 / 0.00 | 27.52 / 0.01 |
| λ = 1 + all-nine healing | 75.19 / 0.18 | 67.58 / 0.50 |
| λ = 0 + all-nine healing | 75.33 / 0.16 | 67.20 / 0.46 |
| λ = 0 + recall-based five-client healing (CBSH) | 74.41 / 0.27 | 63.88 / 0.22 |
| λ = 0 + random five-client healing, seed 42 | 75.21 / 0.20 | 64.94 / 0.37 |

The projection recovered only 0.36 pp IID CA at λ = 1 before healing. After healing the λ = 1 utility change versus λ = 0 was −0.14 pp IID and +0.38 pp non-IID. Class-balanced selective healing (CBSH) scores clients using validation recall deficits and per-client class counts. Both five-client arms hold out 100 retained training images per class for validation. The random control exceeded CBSH by 0.80 pp IID and 1.06 pp non-IID CA on the saved selections. CBSH's ASR was 0.07 pp higher IID and 0.14 pp lower non-IID than random. The all-nine healing comparator uses the full retained training partitions, so the two five-client arms give the more direct selection-rule comparison. Recorded λ = 0 follow-up times were 89.00/92.38 s for all-nine healing, 59.57/70.83 s for CBSH, and 59.68/61.58 s for random healing (IID/non-IID). [λ = 1 healing](../results/direction_preserving_followup_gpu/) · [λ = 0 healing](../results/direction_preserving_lambda0_followup_gpu/) · [CBSH](../results/cbsh_gpu/) · [Random healing](../results/cbsh_random_gpu/)

## 5. Matched onset-free study

This branch regenerated two fresh 50-round white-patch trajectories and their independent retraining and FedEraser references. Each source has 500 client deltas and verified round reconstructions. The attacked checkpoints, partitions, and metrics belong to this matched study. The methods take the identity of the client under repair and use stored deltas and, where specified, retained data. Test ASR, the true onset, and the reference model enter evaluation. The oracle-informed gate is identified below. [Measured branch report](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fd/results/onsetfree/REPORT.md) · [Saved sweep and diagnostic tables](PROJECT_EXPERIMENT_APPENDIX.md#onset-free-sweeps)

| Endpoint | IID CA / ASR | Non-IID CA / ASR | Extra communication |
|---|---:|---:|---:|
| Regenerated attack | 74.70 / 51.93 | 71.59 / 29.32 | — |
| Retraining reference | 74.92 / 3.08 | 70.32 / 3.41 | 50 rounds |
| FedEraser Ecal 1 | 71.99 / 2.33 | 67.69 / 2.51 | 49 rounds |
| FedEraser Ecal 2 | 75.15 / 2.80 | 69.95 / 2.59 | 49 rounds |
| C1 parameter-only subtraction | 64.31 / 0.11 | 27.15 / 1.06 | 0 |
| C2 contrast-only | 64.46 / 0.08 | 26.69 / 1.06 | 0 |
| C3 random basis, seed 42 | 65.37 / 0.09 | 29.58 / 2.23 | 0 |
| C4 C0-update basis | 74.97 / 45.20 | 70.13 / 29.68 | 0 |
| L0 default, rank 32, scale 1 | 67.63 / 0.24 | 34.30 / 0.69 | 0 |
| L0 default + BN recalibration | 67.53 / 0.41 | 43.23 / 0.74 | 0 client-training rounds |
| L0 rank 32, scale 0.5 | 73.54 / 1.66 | 56.48 / 1.78 | 0 |
| L1 rank 50, 2,048 curvature examples | 70.27 / 1.34 | 43.07 / 5.84 | 0 |
| L2 Taylor, W = 5, 2,048 examples | 10.00 / 100.00 | 10.00 / 100.00 | 0 |
| L2 stable transfer, W = 5, 2,048 examples | 74.42 / 2.17 | 69.83 / 2.88 | 0 |
| L3 default L0 + 3 benign rounds | 74.67 / 1.57 | 69.16 / 2.24 | 3 rounds |
| L3 half-scale L0 + 3 benign rounds | 74.93 / 3.19 | 69.67 / 3.74 | 3 rounds |

### L0 — benign update subspace and scale

L0 forms a weighted sum of the round-wise differences between C0's update and the retained-client average. For each trainable tensor, it estimates a low-rank basis from stacked retained-client updates and subtracts the component of the contrast outside that basis. The sweep evaluated ranks 4/8/16/32/64, scales 0.5/1, controls C1–C4, and random subspaces with seeds 42/43/44. There are **54 evaluations per partition**. Default L0 gained 3.32 pp IID and 7.15 pp non-IID CA versus C1. BN recalibration added 8.93 pp CA in non-IID. At scale 0.5 rank 64 gave **73.72 / 2.02 IID** and **59.02 / 2.26 non-IID**. In IID, half-scale C1 (**72.63 / 1.29**) and C2 (**72.44 / 1.12**) clear the S1 numerical thresholds, as do all 15 half-scale random-basis cells and all five half-scale L0 ranks. This is a descriptive check of the saved grid on one trajectory. The default was rank 32 at full scale. [Branch L0 analysis](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fd/docs/onsetfree/L0_SUMMARY.md) · [Full L0 grid](PROJECT_EXPERIMENT_APPENDIX.md#l0-rank-scale-and-random-seeds)

### L1 and L2 — curvature, BatchNorm, and time-varying transfer

- **L1 stationary spectrum:** A low-rank generalized Gauss–Newton (GGN) spectrum on retained examples transports perturbations through a fixed-curvature momentum model. The rank/retained-sample grid used 20/50/100 and 512/2,048 examples. The highest CA cell was rank 100 with 2,048 examples, **71.52 / 1.76 IID** and **51.71 / 7.17 non-IID**. Three stochastic curvature seeds at the default produced **69.82–70.27 CA / 1.14–1.34 ASR IID** and **43.07–47.71 CA / 5.84–8.58 ASR non-IID**, all on the same attack trajectory per partition.
- **Curvature controls:** At rank 20 and 512 examples, eval-mode GGN scored **68.84 / 1.33 IID**, **41.09 / 12.48 non-IID**. Exact Hessian gave **68.74 / 1.32** and **37.65 / 18.12**. GGN with BN using batch statistics and dropout disabled gave **68.01 / 0.72** and **46.08 / 4.76**. Train-stat BN improved both metrics in non-IID at the tested setting.
- **L2 Taylor:** All 16 runs across both partitions, W = 1/2/5/10 and sample sizes 512/2,048, finished at approximately 10% CA. ASR fluctuated between 0% and 100%, tracking degenerate predictions. The single Taylor effective step was unstable for the recorded series of local momentum updates.
- **Momentum-aware transfer:** Replacing the Taylor step with repeated heavy-ball transfer stabilized the endpoint. With 512 samples, W = 1/2/5/10 delivered IID **73.00/1.83, 73.50/1.96, 73.79/2.00, 74.00/2.07** and non-IID **63.74/3.84, 65.89/3.57, 67.77/3.47, 68.34/3.10**. The larger W = 5, 2,048-sample run got within 0.50 pp IID and 0.49 pp non-IID of retraining CA. It used about **130.83/152.14 trillion** server GGN FLOPs including the L1 spectrum and approximated heterogeneous local step counts with rounded aggregate counts.

[Curvature and window report](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fd/docs/onsetfree/L1_L2_SUMMARY.md) · [Every L1 and L2 result](PROJECT_EXPERIMENT_APPENDIX.md#l1-and-l2-curvature-sweeps)

### L3, detector, and registered predictions

L3 continues for three benign FedAvg rounds from the default or half-scale L0. Default L3 improved CA by **7.04 pp IID** and **34.86 pp non-IID** over its input. The half-scale input led to slightly higher final CA and higher ASR in both partitions. The repairs use **9.45/9.75 trillion** client FLOPs (IID/non-IID), about **8.17× fewer** than matched Ecal 1 (77.18/79.59 trillion). Ecal 2 uses 154.37/159.18 trillion and retraining about 157.52/162.43 trillion. Default L1 uses 14.70 trillion server GGN FLOPs per partition. These FLOP estimates exclude optimizer arithmetic, server orchestration, and data/communication overhead. [Measured cost and endpoints](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fd/results/onsetfree/REPORT.md)

| Prediction or check | Measured result |
|---|---|
| P1, L1 reduces benign loss increase and curvature penalty versus C1 | Supported on the fixed 512-example samples. Loss increases C1→L1 were 0.2733→0.0909 IID and 6.7303→0.9720 non-IID. Half-Hessian quadratics were 0.3986→0.1567 and 7.6166→0.4680. The benign samples were disjoint from curvature estimation and drawn from the original training clients. |
| P2, L1 is more aligned with counterfactual displacement than C1 and L0 | Refuted. Whole-vector cosines C1/L0/L1 were 0.2353/0.1803/0.0834 IID and 0.1211/0.1498/0.0125 non-IID. L1 beat both in 2 of 10 comparable layers per partition. |
| P3, detect onset within roughly five rounds in IID | Refuted. Leave-one-out benign residuals plus CUSUM gave IID alarms at rounds 15 and 12 for κ = 0.5 and 1.0, versus actual round 30. Non-IID had no alarm. Zero alarms on nine benign pseudo-target traces is a small empirical check. |
| P4, the useful L2 window saturates near the estimated stiff-mode memory | Inconclusive. All 50 stiff modes contracted below 0.5 in the fixed-curvature model. Under stable transfer, non-IID ASR kept improving through W = 10 and IID ASR rose slightly with W. |
| Per-class ASR | L1's aggregate ASR of 5.84% non-IID hid a worst source-class ASR of 12.3%. IID L1's highest class ASR was 4.1%. The appendix includes all nine classes for every evaluated diagnostic endpoint. |

**Gate ablation.** Twenty-seven IID gates crossed early-correction multiplier α ∈ {0, 0.5, 1}, margin ∈ {0, 2, 5}, and shift ∈ {0, 10, 20}. At no margin and α = 0, the detected gate gave **69.82 / 0.44**, +10 gave **71.04 / 0.46**, and the post-hoc +20 gave **72.96 / 1.16**. The +20 shift used the observed onset error and is excluded from the prespecified verdict. Non-IID had no gate following its missed alarm. [Onset results](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fd/docs/onsetfree/ONSET_SUMMARY.md) · [All gates](PROJECT_EXPERIMENT_APPENDIX.md#iid-onset-gates)

**Success thresholds.** S0 required ASR below 10% and CA within 5 pp of matched Ecal 1. Default L0/L1 qualified on IID, while stable W = 5 with 2,048 examples and both L3 repairs qualified in both partitions. S1 required CA within 2 pp, ASR at most 5%, and at least 10× fewer FLOPs under the study's cost comparison. The registered IID half-scale L0 ablation met it, alongside the simple and random controls noted above. The L3 repairs missed the cost target. S2 required matching or beating Ecal 2 at lower cost, with no valid measured method qualifying. The study used one attacked trajectory per partition. Random-subspace and curvature seeds probe method randomness on those trajectories. [Prespecified verdict](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fd/results/onsetfree/REPORT.md)

## Implementation, audits, and reporting

- **Training and artifact infrastructure.** The project includes seeded IID and Dirichlet partitions, sample-weighted FedAvg, SmallCNN, round metrics, saved class histograms and client-update trajectories. The historical D2 recovery added RNG snapshots, round-level checkpoints and completion markers. The organized checkout restored CIFAR modules, removed a global TLS-verification bypass, and documented which files are available from Git and which require local archives. [Project walkthrough](PROJECT_WALKTHROUGH.md) · [Import and verification record](../manifests/source_import_manifest.md) · [Artifact guide](guides/REPRODUCIBILITY_AND_ARTIFACTS.md)
- **Repair and evaluation tooling.** The codebase includes trigger-aware baseline evaluation, exact-partition counterfactual retraining, FedEraser reconstruction logs and runtime accounting, configurable TCUG donors and repair, and grouped class/round metrics. The onset-free branch added verified replay, matrix-free GGN and Hessian operators, stochastic spectral diagnostics, resumable counterfactual training, FLOP accounting, six stage records and 16 saved figures. [Experiment entry points](../scripts/README.md) · [Onset-free status](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fd/docs/onsetfree/STATUS.md) · [Presentation material](../presentation/README.md)
- **Recorded verification.** The baseline handoff reported 15 passing active tests and six failures in the historical suite tied to unavailable old paths/configs. The onset-free closeout reported 14 passing focused tests on the GPU host. Its two regenerated attack trajectories each reproduced 50 rounds and 500 client deltas with zero reconstruction error in the saved checks. The branch also completed a 15-abstract related-work review. Its scope was abstract-level. [Baseline verification](../manifests/source_import_manifest.md) · [Onset-free audit](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fd/docs/onsetfree/AUDIT.md) · [Related-work record](https://github.com/anshul45-github/Federated-Backdoor-Unlearning/blob/f81d1fd/docs/onsetfree/NOVELTY_CHECK.md)

## Coverage and limits

The review uses the checked-out TCUG results, the main branch's historical documentation, the onset-free branch's measured report and compact data, and the saved onset-free run inventory on the GPU host. It records recoverable metrics, repeat batches, incomplete attempts and superseded results. Checkpoints were not retrained or reevaluated for this review. Older centralized and early FL endpoints and some exploratory attempt metrics are absent from the inspected project records. Phase 7D, adaptive-attack experiments, independently held-out benign model selection, attack-seed replication and midpoint curvature appear as proposed or untested work in the reviewed records. [Project walkthrough](PROJECT_WALKTHROUGH.md) · [Source import manifest](../manifests/source_import_manifest.md) · [Experiment appendix](PROJECT_EXPERIMENT_APPENDIX.md)
