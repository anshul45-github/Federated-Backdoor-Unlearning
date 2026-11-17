# ISP Project: Client-Level Backdoor Unlearning in Federated Learning

This project studies how to suppress a backdoor introduced by Client 0 while preserving clean accuracy in federated learning. The reported 50-round experiments use CIFAR-10, SmallCNN, sample-weighted FedAvg, 10 clients, and seed 42. The main non-IID partition uses Dirichlet α = 0.5. Clean accuracy (CA) uses 10,000 test images, and attack success rate (ASR) is the fraction of 9,000 triggered non-target test images predicted as the target class. Tables report percentages unless otherwise stated.

## 1. Original study — 3×3 white-patch attack

Phase 1 trained clean and attacked models. Phase 2 compared subtraction, knowledge distillation (KD), and FedEraser against a counterfactual reference (M_CF) independently trained without Client 0.

| Endpoint | IID CA | IID ASR | Non-IID CA | Non-IID ASR |
|---|---:|---:|---:|---:|
| Clean training | 75.11 | 2.73 | 70.10 | 3.30 |
| Attacked model | 75.01 | 56.07 | 69.80 | 32.21 |
| Subtraction | 66.88 | 0.24 | 28.04 | 0.87 |
| Subtraction + KD | 72.82 | 2.40 | 60.36 | 3.69 |
| FedEraser, 1 calibration epoch | 72.11 | 2.67 | 65.75 | 3.53 |
| FedEraser, 2 calibration epochs | 74.99 | 2.34 | 68.89 | 3.09 |
| M_CF retraining reference | 74.87 | 3.01 | 69.64 | 3.40 |

The attack sharply increased ASR with little change in clean accuracy. Subtraction caused the largest utility loss, especially in non-IID. KD recovered some of that loss, and two-epoch FedEraser came closest to M_CF in clean accuracy. Its clean prediction agreement with M_CF was 89.29% IID and 88.18% non-IID. [Protocol](docs/baseline/EXPERIMENT_PROTOCOL.md) · [Phase 1 CSV](results/phase1/phase1_summary.csv) · [Corrected Phase 2 CSV](results/phase2/federaser/corrected_evaluation_v2/federaser_comparison.csv)

## 2. Checkerboard baselines — subtraction, KD, and FedEraser

The 4×4 checkerboard experiments use the respective attacked round-50 models. Subtraction removes C0's weighted historical deltas. KD initializes from the subtracted model and distills on a balanced CIFAR-100 proxy. FedEraser reconstructs using the retained clients with two calibration epochs.

| Endpoint | IID CA | IID ASR | Non-IID CA | Non-IID ASR |
|---|---:|---:|---:|---:|
| Attacked model | 75.26 | 92.18 | 71.14 | 80.89 |
| Subtraction | 68.21 | 0.02 | 31.25 | 0.13 |
| Subtraction + KD | 73.09 | 0.41 | 59.35 | 0.79 |
| FedEraser, 2 calibration epochs | 75.19 | 1.18 | 70.18 | 0.94 |

Subtraction gave the lowest ASR and the lowest clean accuracy among the three methods in both partitions. FedEraser preserved the most clean accuracy. [IID run](results/checkerboard/iid_20261024_015516_rerun_001/) · [Non-IID run](results/checkerboard/noniid_20261024_150813_189/) · [Unlearning protocol](docs/checkerboard/CHECKERBOARD_UNLEARNING.md)

### Extreme label-skew stress test

The separate α = 0.01 checkerboard run uses one seed and has no matched counterfactual retraining reference. UTR is the target prediction rate on untriggered non-target images.

| Endpoint | CA | ASR | UTR | Macro F1 |
|---|---:|---:|---:|---:|
| Attacked model | 33.48 | 99.97 | 1.19 | 26.30 |
| Subtraction | 10.00 | 100.00 | 100.00 | 1.82 |
| Subtraction + KD | 10.00 | 100.00 | 100.00 | 1.82 |
| FedEraser, 2 calibration epochs | 34.86 | 6.57 | 13.92 | 28.41 |

Subtraction and KD collapsed to target-class prediction. FedEraser reduced ASR substantially, while clean accuracy remained 34.86%. [Saved stress-test report](results/label_skew/alpha001_20261029_214828_723364/seed_42/report.md) · [Stress-test protocol](docs/label_skew/EXTREME_LABEL_SKEW.md)

## 3. Checkerboard component patching

The patch search blends historical Conv2 filters and their BatchNorm components into the attacked round-50 model. It screens donors from rounds 10, 20, and 29 on a selection split. The saved IID patch used **42 channels from round 29** with a blend weight of **0.25** and a selection CA-loss budget of **1.00 percentage point**.

| IID split | Original CA | Patched CA | Original ASR | Patched ASR |
|---|---:|---:|---:|---:|
| Selection | 75.14 | 75.20 | 91.82 | 88.58 |
| Held-out report | 75.38 | 75.12 | 92.53 | 88.58 |

On the held-out report split, ASR fell **3.95 percentage points** and clean accuracy fell **0.26 points**. Each split uses half of the CIFAR-10 test set, so these figures have different evaluation populations from the full-test baseline and TCUG tables. A saved patching result is available for IID only. [Patch report](results/checkerboard/iid_20261024_015516_rerun_001/component_patching/report.md) · [Patch search protocol](docs/checkerboard/COMPONENT_PATCHING.md)

## 4. Temporal Clean Update Grafting (TCUG)

Each variant loads the clean round-29 checkpoint (G29) and trains C1–C9 on their original partitions for rounds **30–50**. G29 includes C0's clean contributions from rounds 1–29. C0 performs no fresh local training during continuation. The experiments measure backdoor repair and endpoint utility.

- **TCUG-0** aggregates the fresh C1–C9 updates with sample weights normalized over those clients.
- **TCUG-1** adds the same mean of C0's clean parameter deltas from rounds 25–29 in each continuation round, with a global L2 cap.
- **TCUG-2** ranks those five donors each round by cosine similarity to the current sample-weighted C1–C9 parameter update, then averages and L2-caps the top three. Ties favor newer donors.

Both grafting variants use C0's original FedAvg weight (10.00% IID and 7.194% non-IID). BatchNorm's trainable parameters can be grafted. Its running statistics and counters follow the C1–C9 aggregate. [Experiment runner](scripts/run_checkerboard_rollback.py) · [Graft implementation](src/unlearning/temporal_grafting.py)

### GPU endpoint comparison

TCUG-0 and TCUG-1 come from the same complete GPU batch (`20261115_120004`). TCUG-2 comes from `tcug2_20261115_125159`. Each partition shares the same G29 and FedEraser checkpoint hashes across those batches. All continuation endpoints are from round 50. FedEraser is the saved two-epoch checkpoint reevaluated by the runner.

| Endpoint | IID CA | IID ASR | IID UTR | IID macro F1 | Non-IID CA | Non-IID ASR | Non-IID UTR | Non-IID macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G29 | 70.42 | 1.77 | 3.59 | 70.20 | 66.17 | 1.68 | 4.60 | 64.60 |
| FedEraser | 75.19 | **1.18** | 2.52 | 75.04 | 70.18 | **0.94** | 4.00 | 69.01 |
| TCUG-0 | **75.43** | 1.44 | 3.51 | 75.26 | 70.27 | 1.89 | 4.82 | 69.06 |
| TCUG-1 | 74.99 | 1.62 | 3.89 | 74.79 | **70.93** | 1.87 | 4.70 | 69.89 |
| TCUG-2 | 75.09 | 1.49 | 3.69 | 74.86 | 70.59 | 1.77 | 4.78 | 69.54 |

- **IID:** TCUG-0 led the TCUG variants on clean accuracy and ASR. It exceeded FedEraser's clean accuracy by **0.24 percentage points**, with ASR **0.27 points higher**.
- **Non-IID:** TCUG-1 had the highest clean accuracy, **0.75 points above FedEraser**. TCUG-2 had the lowest ASR among TCUG variants. Compared with TCUG-1, it traded **0.34 points** of clean accuracy for **0.10 points** lower ASR. Its ASR remained **0.82 points above FedEraser**, calculated from the unrounded values.

[TCUG-0 IID](results/tcug_gpu/20261115_120004/tcug-0/iid/comparison.csv) · [TCUG-0 non-IID](results/tcug_gpu/20261115_120004/tcug-0/noniid/comparison.csv) · [TCUG-1 IID](results/tcug_gpu/20261115_120004/tcug-1/iid/comparison.csv) · [TCUG-1 non-IID](results/tcug_gpu/20261115_120004/tcug-1/noniid/comparison.csv) · [TCUG-2 IID](results/tcug_gpu/tcug2_20261115_125159/tcug-2/iid/comparison.csv) · [TCUG-2 non-IID](results/tcug_gpu/tcug2_20261115_125159/tcug-2/noniid/comparison.csv)

The earlier GPU batch (`20261115_015724`) completed TCUG-0 at **75.14% CA / 1.53% ASR** IID and **70.01% / 1.83%** non-IID. Its TCUG-1 IID log ends at round 46. The earlier CPU TCUG-0 runs reached **75.17% / 1.34%** IID and **69.97% / 2.09%** non-IID. These runs are kept separately from the main comparison. Small differences should be interpreted in light of the variation between runs and the single seed per partition. [Earlier GPU batch](results/tcug_gpu/20261115_015724/) · [CPU results](results/rollback/)

### Did norm clipping activate?

The default cap is the median L2 norm of the donor updates. **Every applied graft had a clip scale of 1.0.** TCUG-1's static mean fell below the cap in both partitions. In TCUG-2, every top-three mean also fell below the cap. These runs therefore measure grafting with the default cap in place and no norm reduction.

| Variant | Partition | Raw graft L2 norm | Cap | Rounds clipped |
|---|---|---:|---:|---:|
| TCUG-1 | IID | 1.1497 | 2.1177 | 0/21 |
| TCUG-1 | Non-IID | 1.2229 | 1.8051 | 0/21 |
| TCUG-2 | IID | 1.3459–1.3742 | 2.1177 | 0/21 |
| TCUG-2 | Non-IID | 1.3303–1.3450 | 1.8051 | 0/21 |

[TCUG-1 IID configuration](results/tcug_gpu/20261115_120004/tcug-1/iid/run_config.json) · [TCUG-1 non-IID configuration](results/tcug_gpu/20261115_120004/tcug-1/noniid/run_config.json) · [TCUG-2 IID graft history](results/tcug_gpu/tcug2_20261115_125159/tcug-2/iid/graft_history.json) · [TCUG-2 non-IID graft history](results/tcug_gpu/tcug2_20261115_125159/tcug-2/noniid/graft_history.json)

### Training-free TCUG-2

This variant edits the attacked round-50 checkpoint directly. For each round 30–50 it subtracts C0's recorded parameter delta at C0's original FedAvg weight and adds a clean graft at the same weight. The graft averages the top three C0 deltas from rounds 20–29, ranked by cosine similarity to the sample-weighted C1–C9 deltas recorded for that round. The method keeps the attacked checkpoint's BatchNorm running statistics and counters.

| Endpoint | IID CA | IID ASR | Non-IID CA | Non-IID ASR |
|---|---:|---:|---:|---:|
| FedEraser, 2 calibration epochs | 75.19 | 1.18 | 70.18 | 0.94 |
| Original TCUG-2, 21 continuation rounds | 75.09 | 1.49 | 70.59 | 1.77 |
| Training-free TCUG-2 | 71.92 | 0.04 | 68.52 | 0.11 |

Compared with original TCUG-2, training-free TCUG-2 lowered ASR by **1.44 percentage points IID** and **1.66 points non-IID**, while clean accuracy fell by **3.17** and **2.07 points**, respectively. Compared with FedEraser, it lowered ASR by **1.13 points IID** and **0.83 points non-IID**, while clean accuracy fell by **3.27** and **1.66 points**. Differences use unrounded ASR values. The training-free endpoints were evaluated on CPU, and the comparison uses the previously reported GPU evaluations for original TCUG-2 and FedEraser. [IID repair result](results/training_free_tcug_2/iid/results.csv) · [Non-IID repair result](results/training_free_tcug_2/noniid/results.csv)

**Clipping activated in 0 of 21 rounds in each partition.** The largest graft norms were **1.3546 IID** and **1.3513 non-IID**, below their default caps of **2.0896** and **1.7948**. Original TCUG-2 also recorded zero clipped rounds per partition. [IID graft history](results/training_free_tcug_2/iid/graft_history.json) · [Non-IID graft history](results/training_free_tcug_2/noniid/graft_history.json)

### Training-free TCUG-2 + three clean FedAvg rounds

The follow-up initializes from the training-free TCUG-2 repaired checkpoint and runs rounds **51–53** with C1–C9 on their original labeled CIFAR-10 partitions. The repaired and continued endpoints were evaluated on the same GPU. The IID repaired checkpoint measured 71.91% CA on GPU and 71.92% in the earlier CPU evaluation above. The table uses the GPU starting values for the before/after comparison. [Direct repair runner](scripts/run_training_free_tcug.py) · [Three-round follow-up runner](scripts/run_training_free_tcug_followup.py)

| Endpoint | IID CA | IID ASR | Non-IID CA | Non-IID ASR |
|---|---:|---:|---:|---:|
| Subtraction | 68.21 | **0.02** | 31.25 | 0.13 |
| Subtraction + KD | 73.09 | 0.41 | 59.35 | 0.79 |
| FedEraser, 2 calibration epochs | 75.19 | 1.18 | 70.18 | 0.94 |
| Training-free TCUG-2 (GPU reevaluation) | 71.91 | 0.04 | 68.52 | **0.11** |
| Training-free TCUG-2 + three clean rounds | **75.57** | 0.16 | **70.70** | 0.24 |

- **Versus direct repair:** The three rounds raised clean accuracy by **3.66 percentage points IID** and **2.18 points non-IID**. ASR rose by **0.11** and **0.13 points**, respectively.
- **Versus subtraction:** The follow-up gained **7.36 points** of clean accuracy IID and **39.45 points** non-IID, with ASR **0.13** and **0.11 points** higher.
- **Versus KD and FedEraser:** The follow-up reached higher clean accuracy and lower ASR than both in each saved partition. Against FedEraser, clean accuracy rose by **0.38 points IID** and **0.52 points non-IID**, with ASR lower by **1.02** and **0.70 points**, respectively.

Differences use unrounded ASR values. These results cover one seed per partition and use round 53 as the endpoint. The continuation uses standard FedAvg, and the training-free repair supplying its starting weights recorded zero clipped grafts. [IID GPU comparison](results/tcug_followup_gpu/20261115_144254/iid/comparison.csv) · [Non-IID GPU comparison](results/tcug_followup_gpu/20261115_144254/noniid/comparison.csv)

### Direction-preserving TCUG

This training-free sweep uses all 50 saved rounds and assumes C0 is the client to repair. The repair does not use the attack onset. Each round projects C0's parameter delta onto the positive direction of the sample-weighted C1–C9 delta. The default cap limits the projection norm to the norm of that round's C1–C9 delta. The endpoint subtracts C0's original weighted contribution and adds back λ times the weighted projection. BatchNorm running statistics and counters stay at their attacked values.

| Endpoint | IID CA | IID ASR | Non-IID CA | Non-IID ASR |
|---|---:|---:|---:|---:|
| Attacked | 75.26 | 92.18 | 71.14 | 80.89 |
| λ = 0 | 65.21 | 0.00 | 27.54 | 0.00 |
| λ = 0.25 | 65.28 | 0.00 | 27.53 | 0.00 |
| λ = 0.5 | 65.43 | 0.00 | 27.52 | 0.01 |
| λ = 1 | 65.57 | 0.00 | 27.52 | 0.01 |

The cap activated in **0 of 50 rounds** in each partition. C0 had positive alignment in 50 IID rounds and 34 non-IID rounds. Relative to λ = 0, retaining the full positive projection recovered 0.36 percentage points of clean accuracy in IID and reduced it by 0.02 points in non-IID. λ = 0 is the parameter-only subtraction control for this sweep. The earlier subtraction baseline also adjusts floating-point BatchNorm buffers. [IID sweep](results/direction_preserving_tcug/iid/comparison.csv) · [Non-IID sweep](results/direction_preserving_tcug/noniid/comparison.csv) · [Runner](scripts/run_direction_preserving_tcug.py)

### Selective healing after λ = 0

Each follow-up starts from the parameter-only subtraction checkpoint and runs clean FedAvg for rounds 51–53. Regular healing uses all nine retained clients and their original training partitions. CBSH holds out 100 images per class from the retained training partitions, then selects five clients each round using the current model's per-class validation recall. Its score for each client is the sum of its healing class counts weighted by the corresponding recall deficits. The random-five control uses the same holdout, training settings, and sample-weighted FedAvg as CBSH. It draws five clients uniformly without replacement each round using selection seed 42. The two partitions use the same sampled client schedule.

| Endpoint | IID CA | IID ASR | Non-IID CA | Non-IID ASR |
|---|---:|---:|---:|---:|
| λ = 0 starting checkpoint, GPU evaluation | 65.20 | 0.00 | 27.54 | 0.00 |
| Regular healing, all nine clients | 75.33 | 0.16 | 67.20 | 0.46 |
| CBSH, five selected clients | 74.41 | 0.27 | 63.88 | 0.22 |
| Random selection, five clients | 75.21 | 0.20 | 64.94 | 0.37 |

CBSH finished **0.92 percentage points below** regular healing in IID clean accuracy and **3.32 points below** in non-IID. Its ASR was **0.11 points higher** in IID and **0.23 points lower** in non-IID, using unrounded values.

In the matched five-client comparison, random selection gained **0.80 points** of clean accuracy and lowered ASR by **0.07 points** in IID. In non-IID, it gained **1.06 points** of clean accuracy and raised ASR by **0.14 points**. These results use one partition seed and one random client schedule.

| Round | CBSH IID clients | CBSH non-IID clients | Random-five clients, both partitions |
|---:|---|---|---|
| 51 | C3, C6, C2, C4, C1 | C6, C9, C1, C4, C5 | C2, C1, C6, C3, C9 |
| 52 | C5, C7, C4, C6, C3 | C9, C1, C7, C6, C5 | C4, C3, C6, C1, C5 |
| 53 | C5, C7, C4, C6, C3 | C1, C9, C6, C5, C4 | C2, C7, C1, C8, C6 |

[Regular IID](results/direction_preserving_lambda0_followup_gpu/20261116_184856/iid/comparison.csv) · [Regular non-IID](results/direction_preserving_lambda0_followup_gpu/20261116_184856/noniid/comparison.csv) · [CBSH IID metrics](results/cbsh_gpu/20261116_193407/iid/comparison.csv) · [CBSH non-IID metrics](results/cbsh_gpu/20261116_193407/noniid/comparison.csv) · [IID selection](results/cbsh_gpu/20261116_193407/iid/selection_history.json) · [Non-IID selection](results/cbsh_gpu/20261116_193407/noniid/selection_history.json)

[Random-five IID metrics](results/cbsh_random_gpu/seed42/iid/comparison.csv) · [Random-five non-IID metrics](results/cbsh_random_gpu/seed42/noniid/comparison.csv) · [Random-five IID selection](results/cbsh_random_gpu/seed42/iid/selection_history.json) · [Random-five non-IID selection](results/cbsh_random_gpu/seed42/noniid/selection_history.json)

## Setup and reproducibility

Create and activate a Python environment, then install dependencies with `python -m pip install -r requirements.txt`. For GPU runs, install the PyTorch and Torchvision builds compatible with the host's NVIDIA driver and check `torch.cuda.is_available()` before launching. The reported GPU logs record PyTorch `2.8.0+cu128`. Run the active tests with `python -m pytest`. Tests in `tests/historical/` require historical artifacts.

Git tracks the result summaries, configurations, and code. Model checkpoints, client partitions, and trajectories are local artifacts. A fresh clone needs those files to reproduce checkpoint-based results. A full 50-round trajectory includes 500 client deltas and 50 round checkpoints per partition. For TCUG use the saved G29 and partition files, the matching attacked and FedEraser endpoints and metadata, and the C0 deltas for the chosen donor window.

From the repository root, see `python -m scripts.run_checkerboard_rollback --help` for the TCUG command. It accepts `--variant tcug-0`, `--variant tcug-1`, or `--variant tcug-2`, plus `--graft_window` and `--graft_max_norm`. Each completed run writes `comparison.csv`, `evaluation.json`, `run_config.json`, round metrics, and endpoint weights. TCUG-2 also writes `graft_history.json`.

[Run index](results/README.md) · [Experiment commands](scripts/README.md) · [Artifact and reproducibility guide](docs/guides/REPRODUCIBILITY_AND_ARTIFACTS.md) · [Project walkthrough](docs/PROJECT_WALKTHROUGH.md) · [Historical research](docs/historical/) · [Presentation assets](presentation/README.md)
