# Complete project walkthrough

Prepared from this checkout on 23 October 2026. This is an explanation of the existing work, not a record of newly executed training. Existing source code and experimental artifacts were left unchanged.

## 1. What this project studies

The research question is: **after a malicious federated client has helped train a model, can we remove that client's influence, suppress its backdoor, and preserve useful classification behavior?**

Federated learning trains a shared model using separate client datasets. In this repository, those clients are simulated sequentially in one Python process; there is no deployed network of client devices. Each simulated client receives the current model, trains locally, and returns a parameter update. The server combines these updates with sample-count-weighted FedAvg.

The malicious client is known in advance: Client 0. Identifying an unknown attacker is outside the implemented experiment. Client 0 learns a shortcut: images containing a small white patch should be classified as Airplane. The other clients train on clean data.

The project evaluates four responses:

- **Historical subtraction, M_sub:** subtract Client 0's recorded contributions from the final attacked model.
- **Knowledge distillation, M_KD:** start with M_sub and recover useful predictions by matching the attacked model's soft outputs on clean proxy images.
- **FedEraser:** reconstruct training without Client 0, using retained clients to estimate new update directions and historical updates to supply magnitudes.
- **Counterfactual retraining, M_CF:** train a separate reference from the same initial weights and client partitions, excluding Client 0 throughout.

M_CF is a reference model. It is not used as the KD teacher and is not itself an inexpensive unlearning method. Removing Client 0 means removing its clean contributions as well as its malicious contributions.

## 2. Two experiment histories coexist

| Property | Current Phase 1 / Phase 2 snapshot | Historical staged research |
|---|---|---|
| Client pool | 10 | 20 |
| Participation | All 10 each round | Usually 10 selected per round |
| Federated rounds | 50 | 100 |
| Main partitions | IID; Dirichlet alpha=0.5 | IID; heterogeneity and size-imbalance variants |
| Attack | Client 0 data poisoning, no explicit update amplification in the current runner | Data poisoning plus update scaling, commonly gamma=5, with stress variants |
| KD proxy | 5,000 CIFAR-100 training images | Earlier experiments use 2,500 CIFAR-10 test images |
| Clean evaluation | Full 10,000-image CIFAR-10 test set | Frozen 7,500-image evaluation subset |
| Main retained artifacts | `results/`, `results_phase2/` | Mainly reports and archived scripts; many referenced raw run folders are absent |

The root README, `configs/v2_*.yaml`, current source, saved Phase 1/2 evaluations, and corrected FedEraser v2 folders describe the current baseline. `docs/canonical_benchmark.md` describes the historical benchmark despite its authoritative title. Do not combine the two protocols' numbers.

## 3. Work completed in the current baseline

### 3.1 Data and model

CIFAR-10 contains ten image classes. The experiment uses 50,000 training images and 10,000 test images, each 32 by 32 RGB. Training applies random cropping and horizontal flipping; testing uses deterministic tensor conversion and normalization.

The IID partition randomly assigns 5,000 images to each of ten clients. The Non-IID partition uses a per-class Dirichlet draw with alpha=0.5, producing different class mixtures and different client sizes. The saved Non-IID Client 0 has 3,597 images, so its aggregation weight is 3,597/50,000 = 0.07194. After removing Client 0, the retained clients contain 46,403 images and their weights are renormalized over that total.

SmallCNN has two convolutional blocks and two fully connected layers:

```text
Input: 3 x 32 x 32
Conv(3 -> 32, 3x3, padding=1) -> BatchNorm -> ReLU -> MaxPool
Conv(32 -> 64, 3x3, padding=1) -> BatchNorm -> ReLU -> MaxPool
Flatten: 64 x 8 x 8 = 4096
Linear(4096 -> 128) -> ReLU -> Dropout(0.5)
Linear(128 -> 10 logits)
```

It contains 545,290 trainable parameters by architecture arithmetic. BatchNorm also has running statistics and integer tracking buffers, which require separate handling during update operations.

### 3.2 Clean and attacked federated training

The main settings are seed 42, 50 rounds, two local epochs, batch size 64, SGD learning rate 0.01, momentum 0.9, and implicit optimizer-default weight decay 0 in the current runner. Every client participates every round; the current loop is hardcoded to that full-participation protocol.

Each client starts from the same current server state. Its update is:

`delta_client = local_state - global_state`

The server then computes:

`next_global = global + sum((client_samples / participating_samples) * delta_client)`

The attack runs during rounds 30 through 50, inclusive. A fixed set of non-Airplane samples, numbering floor(10% of Client 0's total samples), receives a 3x3 white bottom-right patch and target label 0. This is 500 IID images or 359 images in the saved Non-IID split. The patch uses normalized white pixel values, not the literal value 1 after normalization.

Four Phase 1 endpoints were saved: IID clean, IID attack, Non-IID clean, and Non-IID attack. Attacked training was also repeated with trajectory logging to support subsequent unlearning.

### 3.3 Historical subtraction

The implementation computes approximately:

`M_sub = M_attack - sum_over_all_50_rounds(weight_of_C0 * delta_C0)`

IID uses weight 0.1; Non-IID derives the actual weight from saved partitions. Subtraction affects floating-point state, including BatchNorm running means and variances; integer counters remain unchanged. It removes all recorded Client 0 contributions, including pre-attack rounds.

This arithmetic cannot undo how earlier contributions changed later clients' training starting points. The retained clients' later historical updates were themselves computed along the attacked trajectory. This explains why subtraction is an approximate baseline rather than exact counterfactual retraining.

### 3.4 Knowledge distillation recovery

The attacked endpoint is the frozen teacher. The student starts from M_sub. The proxy has 50 images from each CIFAR-100 class, totaling 5,000 training images. CIFAR-100 labels select a balanced subset; the student still outputs the ten CIFAR-10 classes, and those proxy labels are not used in the distillation loss.

The student trains for five epochs with SGD, learning rate 0.001, momentum 0.9, batch size 64, and temperature T=4. The loss is T-squared-scaled KL divergence between teacher and student probabilities. Teacher evaluation mode is fixed. Student Dropout stays active, while BatchNorm running statistics are frozen and checked for equality before and after KD.

IID KD is in `scripts/run_kd.py`; Non-IID KD is in `scripts/run_noniid_stage_abc.py`. The latter also performs prerequisite audits before training.

### 3.5 FedEraser reconstruction and corrections

The corrected runner is `scripts/run_federaser_v5.py`. It starts from the saved initial model, excludes Client 0, and reconstructs all 50 rounds using Clients 1 through 9. Round 1 reuses retained clients' historical updates directly. Later rounds train retained clients for Ecal=1 or Ecal=2 epochs from the reconstructed state.

For eligible floating-point weights and biases, calibration computes:

`calibrated_update = norm(historical_update) * calibration_update / norm(calibration_update)`

The norm is global across eligible tensors, not separately per layer. BatchNorm running means and variances use the unscaled calibration deltas. Integer `num_batches_tracked` counters are preserved by the corrected runner's modified aggregation. The reusable source helper handles calibration; the runner duplicates some of this mathematics and adds its own aggregation policy.

The registry preserves earlier mistakes rather than hiding them:

- `federaser/iid/` and `noniid/` are superseded: inherited weight decay and incorrect evaluation handling.
- `iid_Ecal1/`, `iid_Ecal2/`, `noniid_Ecal1/`, `noniid_Ecal2/` are superseded: calibration seeding and BatchNorm counter problems.
- The four folders ending `_v2` are the corrected runs used in the current comparison.

### 3.6 Counterfactual reference

The IID and Non-IID reference scripts load the attacked run's exact initial weights and saved partitions, train only Clients 1 through 9 for 50 rounds, and use sample-count-renormalized FedAvg. They also evaluate during training. Although they do not require historical deltas, they **do require the excluded `client_partitions.pt` files** in their current implementation.

## 4. What the saved results show

These numbers come from retained CSV/JSON records. Model inference was not rerun during this walkthrough.

| Partition | Clean model accuracy | Attacked model accuracy | Clean model ASR | Attacked model ASR |
|---|---:|---:|---:|---:|
| IID | 75.11% | 75.01% | 2.73% | 56.07% |
| Non-IID | 70.10% | 69.80% | 3.30% | 32.21% |

Clean accuracy barely changes while trigger success increases sharply. Consequently, normal test accuracy alone would miss this attack.

| Endpoint | IID accuracy | IID ASR | Non-IID accuracy | Non-IID ASR |
|---|---:|---:|---:|---:|
| Attacked | 75.01% | 56.07% | 69.80% | 32.21% |
| Subtraction | 66.88% | 0.24% | 28.04% | 0.87% |
| Subtraction + KD | 72.82% | 2.40% | 60.36% | 3.69% |
| FedEraser Ecal=1, corrected | 72.11% | 2.67% | 65.75% | 3.53% |
| FedEraser Ecal=2, corrected | 74.99% | 2.34% | 68.89% | 3.09% |
| Counterfactual reference | 74.87% | 3.01% | 69.64% | 3.40% |

Subtraction suppresses target predictions partly at the expense of broad classification capability. In the saved Non-IID subtraction run, Cat, Deer, Horse, and Truck have zero correct predictions and receive no predicted labels at all. KD repairs some utility. FedEraser Ecal=2 has the strongest observed utility/security balance among the tested unlearning settings.

Corrected Ecal=2 clean prediction agreement with M_CF is 89.29% IID and 88.18% Non-IID. Similar test accuracy does not mean identical predictions or complete removal of training influence.

### Metric meanings

| Metric | Population and meaning |
|---|---|
| Clean accuracy | Correct classifications over all 10,000 clean test images |
| ASR / triggered target rate | Fraction of 9,000 triggered non-Airplane images predicted as Airplane |
| UTR | Fraction of the same 9,000 non-Airplane images, without triggers, predicted as Airplane |
| Airplane recall | Correct classifications of the 1,000 genuine Airplane images |
| Macro-F1 | Mean class-specific F1 across ten classes; captures broader class failures |
| CF agreement | Fraction of candidate predictions matching M_CF on the specified population |

The triggered and untriggered populations are two views of the same images. They are not disjoint samples. Some JSON files store Macro-F1 as a fraction from 0 to 1 and others as a percentage; the consolidated table normalizes units.

Measured subtraction plus KD times are about 7.82 seconds IID and 11.29 seconds Non-IID. Corrected Ecal=2 reconstruction loops record about 1,211 and 1,258 seconds. Reference wall times are about 2,514 and 4,219 seconds. These timer boundaries differ: reference runs include repeated evaluation, while reconstruction-loop timers omit setup/final evaluation and include historical-update reads. These values do not establish a fair end-to-end speedup.

## 5. Historical research stages and what they contributed

### Initial stages: baseline, attack, reference, subtraction, KD, FedEraser

The archived centralized runner establishes a conventional single-machine baseline. The archived federated runner supports the older 20-client, partially participating protocol, attack scaling, historical participation schedules, and reference runs. Stage 5A subtracts historical contributions; Stage 5B repairs with KD. The earlier FedEraser report compares reconstruction with full retraining. Most associated raw checkpoint folders are not included in this checkout.

### Stage 6C: counterfactual and trajectory analysis

This stage compares parameter distances, prediction agreement, layerwise differences, trigger variants, and cost/fidelity tradeoffs. It motivates studying training path dependence: removing the sum of earlier updates does not reconstruct the optimization trajectory. Claims in this historical report are interpretations of that experiment, not proofs of a unique causal explanation.

### Stage 6D: attack timing

The controlled pilot moves 15 malicious Client 0 participation events between early, middle, and late windows, with no-attack and matched client-excluded controls. Phase A validates attacks and controls; Phase B compares three unlearning methods across the three windows.

The historical report records FedEraser agreement of 89.00%, 88.48%, and 89.04% across early, middle, and late. KD achieves 78.31%, 77.76%, and 74.01%; subtraction achieves 35.27%, 40.83%, and 41.31%. This is single-seed evidence. It does not establish timing invariance generally. Preservation documents explain storage removal and regeneration dependencies; the earlier and final preservation notes contain differing timing descriptions, so use the final record together with the archived runner for exact historical interpretation.

### Stage 7A: heterogeneity and attack-strength stress tests

Four conditions combine IID/moderate/strong Non-IID partitions with standard or stronger attack scaling. The initial report suggests that FedEraser preserves accuracy while losing counterfactual agreement under strong heterogeneity.

**The subsequent Stage 7A audit invalidates the original counterfactual analysis.** Client 0 was not actually excluded from purported reference models. Additional problems include evaluation/proxy overlap, test-selected checkpoints, and incorrect update norms that include BatchNorm integer trackers. Read the audit before quoting the original final report.

### Stage 7B: methodological corrections

This stage fixes reference exclusion, evaluation split use, final-checkpoint selection, and geometry calculations. Its corrected report still observes lower agreement under strong heterogeneity: roughly 89.55% IID, 82.80% moderate Non-IID, 64.44% strong Non-IID, and 61.91% strong Non-IID with stronger attack scaling. High clean accuracy is insufficient to establish counterfactual reconstruction.

### Stage 7C: separating label heterogeneity from size imbalance

The final historical study separates two factors:

| Condition | Label heterogeneity | Client size imbalance | Reported FedEraser CF agreement |
|---|---|---|---:|
| D0 | Low | Low | 89.63% |
| D1 | High | Low | 87.09% |
| D2 | Low | High | 81.83% |
| D3 | High | High | 64.13% |

The additive prediction for D3 is 79.29%; the observed agreement is 64.13%, a descriptive interaction of -15.16 percentage points. The historical reports interpret this as evidence that label heterogeneity and aggregation imbalance jointly make reconstruction harder.

Corrections include rerunning canonical D1 with poison ratio 0.1 instead of the exploratory 0.5, fixing D2 geometry/double-scaling and gap definitions, and recovering interrupted D2 training with improved checkpoint safeguards. Final robustness analysis notes that identical attack settings produce different realized ASRs across partitions. The interaction is single-seed descriptive evidence, not a proven general mechanism. No completed new defense, multi-seed replication, or Stage 7D implementation is established by the retained files.

## 6. Folder responsibilities

```text
project/
  configs/                 Current v2 experiment inputs
    _archive/              Older centralized, FL, attack, and staged settings
  docs/                    Protocols, explanations, research reports, audits
  manifests/               Import provenance and inventory notes
  presentation/
    current/               Current 27-slide Phase 1/2 presentation draft
    historical/            Three older proposal/progress decks
  results/                 Phase 1 checkpoints, tables, figures, smoke outputs
    iid_clean/
    iid_attack/
    noniid_clean/
    noniid_attack/
    smoke_test_dir/
  results_phase2/           Logged attacks, unlearning, references, evaluation
    iid_attack_logged/
    noniid_attack_logged/
    iid_counterfactual/
    noniid_counterfactual/
    iid_subtraction/
    noniid_subtraction/
    iid_kd/
    noniid_kd/
    proxy_data/
    charts/
    audit_stage1/
    federaser/             Original, intermediate, and corrected runs
  scripts/                 Training, evaluation, plotting, and audit entrypoints
    _archive/              Historical staged experiment implementation
  src/
    data/                  Dataset loading, partitioning, trigger wrappers
    models/                SmallCNN
    training/              Local optimization, evaluation, FedAvg
    unlearning/            Reusable FedEraser calibration mathematics
    utils/                 Seeding, logging, training plots
  tests/                   Active component and toy integration tests
    historical/            Prototype tests requiring absent old artifacts
```

There is no web application, API service, database, production deployment, or packaged command-line application. The unit of work is an experiment script and its output directory. Subtraction and KD are implemented in scripts rather than dedicated modules under `src/unlearning/`.

## 7. Reading a saved run directory

| Filename | Meaning |
|---|---|
| `config.json` | Settings associated with that run |
| `initial_model.pt` | Starting PyTorch model state dictionary |
| `final_model.pt` | Saved endpoint model state dictionary |
| `metrics_per_round.csv` | Accuracy, trigger rate, and related metrics over rounds |
| `final_results.json` | Final metrics and usually a configuration/runtime record |
| `client_class_histograms.json` | Client sizes and class counts; not exact sample identities |
| `runtime.json` | Logged training wall time |
| `extended_metrics.json` | Extra evaluation such as target recall and Macro-F1 |
| `cf_agreement.json` | Prediction agreement with the matched reference |
| `subtraction_runtime.json` | Subtraction timing and sometimes client-weight metadata |
| `metadata.json` in KD folders | KD settings, losses, runtime, and proxy provenance |
| `eval_results.json` in FedEraser folders | Evaluation of that reconstruction checkpoint |
| `reconstruction_logs.csv` | Per-reconstruction-round timing, not an accuracy trajectory |

The missing `trajectory/round_XXX/` trees would contain individual client deltas and round global checkpoints. The missing `client_partitions.pt` files would contain pickled dataset Subset objects with exact indices and embedded dataset data. Histograms cannot replace those files as exact sample identities.

## 8. Practical limitations and discrepancies found in this checkout

1. **Dependencies and datasets are absent in the inspected local Python environment.** `python` resolves to Python 3.9.13; Torch, Torchvision, and pytest were not found. There is no root `data/` folder. Evaluation requires datasets as well as checkpoints, even when no retraining is required. No dependency installation or model training was performed for this explanation.
2. **`test_cifar.py` is misdocumented.** It checks loading an existing CIFAR-100 dataset after bypassing integrity checks. It does not run a two-round CIFAR-10 FL smoke test. `v2_smoke.yaml` actually specifies three rounds and one local epoch; retained smoke results also reflect that three-round configuration.
3. **`eval_noniid_clean.py` is misnamed.** It loads `results/iid_clean/final_model.pt` and writes IID extended metrics.
4. **`eval_only_federaser.py` is a historical evaluator.** It operates on superseded `iid/` and `noniid/` folders, uses a 7,500-image clean subset alongside full-test ASR, and labels average class accuracy as Macro-F1. It should not be treated as the evaluator for the corrected `_v2` comparison. The corrected v5 runner uses `evaluate_extended` and full current populations.
5. **Source and training history have been imported despite stale placeholder READMEs.** Several files still say imports are pending or suggest a folder structure that does not match the actual implementation. The source import manifest itself mixes earlier and final counts; actual pre-walkthrough inventory is 366 project files totaling 78,677,900 bytes, excluding `.git`.
6. **The checksum manifest is historical, not a clean integrity certificate for this snapshot.** Reading its 51 entries found 29 matching files, six absent partition files, and 16 present files with different hashes. These mismatches are metadata/evaluation files, and this check alone does not establish why they changed. The mismatched paths are listed below. No hashes were rewritten.
7. **Exact replay needs more than saved endpoints.** Trajectories and serialized partitions were intentionally omitted. Current subtraction, FedEraser, and reference runners depend on these files. Full 50-round regeneration is documented as not verified end-to-end after import.
8. **Most launchers hardcode output locations and many helpers execute at import time.** Running them can overwrite curated summaries or models. Archived scripts often still import their previous unarchived paths and reference absent historical result directories. Read their inputs and outputs before execution.
9. **Requirements are unpinned.** Version strings in README/developer notes record a claimed historical environment; installing `requirements.txt` does not reconstruct an exact environment. Those claimed versions were not independently verified here.
10. **Seed caveat needs precision.** The corrected runner uses `hash((42, round_num, client_id)) % 2**32`. Documentation broadly attributes this to Python hash randomization. Here the tuple contains integers, so the ordinary salted-string-hash warning is not directly applicable. A version-independent, explicitly specified seed derivation and controlled backend behavior would still be preferable for a reproducibility claim; this walkthrough does not change the historical seed policy.
11. **Data-preparation names are misleading.** `download_hf_cifar.py` rewrites an existing local CIFAR-100 pickle; it does not download from Hugging Face. Some preparation/KD scripts bypass archive integrity checks, and KD code disables HTTPS certificate verification. These are inherited experiment workarounds, not a complete dataset acquisition pipeline.
12. **Evidence is limited.** The current Non-IID table is one saved split/seed. Historical robustness studies likewise report one seed. Neither low ASR nor agreement with one reference establishes certified unlearning or removal of every possible backdoor behavior.

Checksum mismatches against the existing root manifest:

```text
results_phase2/iid_attack_logged/runtime.json
results_phase2/iid_subtraction/subtraction_runtime.json
results_phase2/iid_kd/metadata.json
results_phase2/iid_counterfactual/config.json
results_phase2/iid_counterfactual/final_results.json
results_phase2/noniid_attack_logged/client_class_histograms.json
results_phase2/noniid_attack_logged/runtime.json
results_phase2/noniid_subtraction/subtraction_runtime.json
results_phase2/noniid_kd/metadata.json
results_phase2/noniid_counterfactual/config.json
results_phase2/noniid_counterfactual/final_results.json
results_phase2/phase2_evaluation_superseded.json
results_phase2/phase2_evaluation_corrected.json
results_phase2/phase2_kd_evaluation.json
results_phase2/noniid_evaluation.json
results_phase2/noniid_kd_evaluation.json
```

## 9. Verification performed for this explanation

Inspected the complete project file inventory; read the core implementation, experiment configurations, major runners/evaluators and research reports; extracted presentation text from all four PowerPoint files; parsed every Python source file with Python's AST parser and every JSON file with the JSON parser; checked the existing 51-entry SHA-256 manifest; and compared retained current result tables with the corrected FedEraser JSONs. Python and JSON parsing reported no errors. Unit tests, checkpoint tensor loading, inference, full dataset verification, and training were not executed because their dependencies/data are unavailable in the inspected environment. Historical reports of 15 passing active tests are historical records, not a new test result.

## 10. Suggested reading sequence

Read the current root README, `docs/EXPERIMENT_PROTOCOL.md`, `src/models/small_cnn.py`, the three data modules, `src/training/fedavg.py`, `scripts/run_experiment_v2.py`, subtraction/KD runners, `scripts/run_federaser_v5.py`, and the corrected comparison table. Then study the Stage 6/7 reports in order, keeping the Stage 7A invalidation and later corrections beside their original reports.

## 11. Complete file-by-file inventory

The inventory below covers every one of the 366 original project files, excluding Git's internal database. It also identifies this newly added guide separately. File roles refer to actual code/content where filenames are misleading. `.git/` contains Git's commit/object/index/configuration metadata and is not experimental source or an output dataset.

| Folder | Original file count |
|---|---:|
| (root) | 7 |
| configs | 12 |
| docs | 36 |
| manifests | 2 |
| presentation | 5 |
| results | 52 |
| results_phase2 | 116 |
| scripts | 115 |
| src | 11 |
| tests | 10 |

Each group names an exact parent directory. Each original file appears once. Definitions identify implemented functions; input/output examples identify the experiment artifacts involved.

### Root files

| File | Explanation |
|---|---|
| `.gitignore` | Dataset/environment/trajectory/partition exclusions and curated checkpoint exceptions. |
| `CONTRIBUTING.md` | Document: Contributing; Pull request expectations; Scientific integrity; Code and data hygiene. |
| `pytest.ini` | Active test discovery/import settings; isolates historical/archive tests. |
| `README.md` | Document: ISP Project: Client-Level Backdoor Unlearning in Federated Learning; 1. Quick Start & Setup; Clone the repository (default branch: main); Create virtual environment and install dependencies. |
| `requirements.txt` | Unpinned dependency list: Torch, Torchvision, NumPy, Pandas, Matplotlib, PyYAML, tqdm, pytest. |
| `sha256_manifest.txt` | Historical 51-entry SHA-256 manifest; current mismatches and omitted paths explained above. |
| `test_cifar.py` | CIFAR-100 local dataset loading probe with integrity bypass; not an FL smoke test. |

### configs

| File | Explanation |
|---|---|
| `README.md` | Document: Experiment configurations. |
| `v2_iid.yaml` | Experiment configuration: v2 iid. Current v2 protocol; smoke config is three rounds. |
| `v2_noniid.yaml` | Experiment configuration: v2 noniid. Current v2 protocol; smoke config is three rounds. |
| `v2_smoke.yaml` | Experiment configuration: v2 smoke. Current v2 protocol; smoke config is three rounds. |

### configs/_archive

| File | Explanation |
|---|---|
| `attack_badnets.yaml` | Experiment configuration: attack badnets. Historical staged protocol. |
| `centralized_baseline.yaml` | Experiment configuration: centralized baseline. Historical staged protocol. |
| `federaser.yaml` | Experiment configuration: federaser. Historical staged protocol. |
| `federated_baseline.yaml` | Experiment configuration: federated baseline. Historical staged protocol. |
| `full_retraining.yaml` | Experiment configuration: full retraining. Historical staged protocol. |
| `stage7c_d1_attacked.yaml` | Experiment configuration: stage7c d1 attacked. Historical staged protocol. |
| `stage7c_d1_attacked_canonical.yaml` | Experiment configuration: stage7c d1 attacked canonical. Historical staged protocol. |
| `stage7c_d1_mcf.yaml` | Experiment configuration: stage7c d1 mcf. Historical staged protocol. |

### docs

| File | Explanation |
|---|---|
| `archive_staging_move_report.md` | Document: Archive Staging Move Report; 1. Initial Disk State; 2. Directories Moved; 3. Move Verification. |
| `canonical_benchmark.md` | Document: Canonical Unlearning Benchmark; 1. Purpose; 2. Benchmark Artifacts; 3. Threat Model and Forgotten Client. |
| `developer_guide.md` | Document: ISP Project: Teammate Developer & Reproducibility Guide; 1. Repository Setup & Dependencies; Clone the repository (default branch: main); 2. Canonical Experiment Protocol & Configurations. |
| `EXPERIMENT_PROTOCOL.md` | Document: Experimental Protocol — Current Baseline Snapshot; Task and question; Dataset and model; Data partitioning. |
| `federaser_adaptation_spec.md` | Document: FedEraser Adaptation Specification; 1. Purpose; 2. Original FedEraser Method; 3. Exact Mathematical Formulation. |
| `federaser_final_report.md` | Document: FedEraser Final Experiment Report; 1. Objective; 2. Experimental Configuration; 3. FedEraser Procedure. |
| `IMPORT_EXISTING_ARTIFACTS.md` | Document: Import the existing working-tree artifacts; 1. Fetch the handoff branch; 2. Import the canonical experiment source; 3. Import presentation materials. |
| `METHODS_AND_METRICS.md` | Document: Methods, Metrics and Interpretation; Methods compared; Metrics; Phase 1 interpretation. |
| `post_b1_research_review.md` | Document: Post-B1 Research Review & Strategic Roadmap; A. Phase B-1 Results Analysis; B. Deltas and Gap Analysis; C. Robust Observations vs. Hypotheses. |
| `REPRODUCIBILITY_AND_ARTIFACTS.md` | Document: Reproducibility and Artifact Audit; What is documented here; Required source artifacts still to import; FedEraser corrected-run record. |
| `RESULTS_SUMMARY.md` | Document: Baseline Results Summary; Phase 1: clean vs attacked; Phase 2: principal endpoints; Takeaways that are supported by these records. |
| `SOURCE_PROJECT_README.md` | Document: Removing Backdoors from Federated Models via Machine Unlearning; Project Purpose; Environment Setup; Project Structure. |
| `stage6d_regeneration_record.md` | Document: Stage 6D Regeneration Record; 1. Stage 6D Objective; 2. Experimental Conditions; 3. Exact Timing Definitions. |
| `stage6d_regeneration_record_final.md` | Document: Stage 6D Final Preservation Record; 1. Objective; 2. Exact Conditions & Timing Definitions; 3. Exact Attack Configuration. |
| `stage7c_d1_analysis.md` | Document: Stage 7C: D1 Analysis & Factorial Decomposition; 1. Objective; 2. Partition Construction & Statistics; 3. BatchNorm Validation. |
| `stage7c_d1_canonical_analysis.md` | Document: Stage 7C: D1 Canonical Analysis & Factorial Evidence; 1. D1 Canonical Objective; 2. Partition and Attack Configuration; 3. Matched Counterfactual (M_CF). |
| `stage7c_d2_analysis.md` | Document: Stage 7C: D2 Downstream Evaluation; 1. D2 Objective; 2. D2 Partition; 3. D2 Attack. |
| `stage7c_d2_analysis_corrected.md` | Document: Stage 7C: D2 Correction & Integrity Audit; 1. Discrepancy Discovered; 2. Frozen Evaluation Split Verification; 3. M_CF Client-0 Exclusion Verification. |
| `stage7c_factorial_synthesis.md` | Document: Stage 7C: Final Factorial Synthesis & Mechanism Audit; 1. Single-Seed Limitation & Statistical Language; 2. Verification of Evaluation Parity; 3. Attack Validity & Configuration Audit. |
| `stage7c_factorial_synthesis_v2.md` | Document: Stage 7C: Final Factorial Synthesis (v2 Canonical); 1. Single-Seed Limitation & Statistical Language; 2. Verification of Evaluation Parity; 3. Final 2x2 Factorial Values (Canonical). |
| `stage7c_final_factorial_robustness_audit.md` | Document: Stage 7C: Final Factorial Robustness Audit (Attack-Strength Normalization); 1. Purpose and Attack-Strength Caveat; 2. Attack Impact Metrics; 3. Unlearning Impact Metrics (FedEraser). |
| `stage_6c_analysis.md` | Document: Stage 6C: Counterfactual Gap & Client-Influence Analysis; 1. Objective; 2. Existing-Artifact Analysis & Canonical Model Comparison; 3. Counterfactual Fidelity (Prediction Level). |
| `stage_6d_experiment_spec.md` | Document: Stage 6D Experiment Specification: Controlled Temporal Influence Study; 1. Scientific Question; 2. Hypothesis; 3. Independent Variable. |
| `stage_6d_phase_a_report.md` | Document: Stage 6D Phase A Report: Attack and Counterfactual Validation; 1. Scientific Question; 2. Experimental Design; 3. Exact Client-0 Participation Schedule. |
| `stage_6d_phase_b_audit.md` | Document: Stage 6D Phase B: Design Audit and Execution Plan; A. CURRENT IMPLEMENTATION AUDIT; B. SCIENTIFIC VALIDITY / CONFOUNDS; C. PHASE B-1 EXPERIMENT MATRIX. |
| `stage_6d_phase_b_report.md` | Document: Phase B-1 Consolidated Report: Temporal Influence & Unlearning Fidelity; 1. Execution Summary; 2. Consolidated Metrics; 3. Measured Results vs Interpretation. |
| `stage_7a_audit_report.md` | Document: Stage 7A Post-Execution Scientific Audit; 1. Executive Verdict; 2. Evaluation Split Audit; 3. Checkpoint Audit. |
| `stage_7a_experiment_spec.md` | Document: Stage 7A: Failure-Boundary Mapping for Federated Client Unlearning; 1. Scientific Question; 2. Hypothesis; 3. Experimental Design. |
| `stage_7a_final_report.md` | Document: Stage 7A Final Report: Failure-Boundary Mapping for Federated Client Unlearning; 1. Objective and Experimental Design; 2. Attack-Validity Check; 3. Unlearning Results & Counterfactual Fidelity. |
| `stage_7a_preflight_report.md` | Document: Stage 7A Pre-Flight Report; 1. Frozen C1-C4 Matrix; 2. Full H0/H1/H2 Partition Statistics (Benign Clients 1-19); 3. Client-0 Statistics. |
| `stage_7b_corrected_report.md` | Document: Stage 7B Corrected Report: Validating the FedEraser Failure Boundary; 1. What was wrong in Stage 7A; 2. Corrections Applied; 3. True $M_{CF}$ Results. |
| `stage_7c_d2_recovery_report.md` | Document: Stage 7C D2 Attacked Run Recovery Report; 1. Why the previous run was not resumable; 2. New checkpointing safeguards; 3. D2 partition identity. |
| `stage_7c_recovery_audit.md` | Document: Stage 7C Recovery Audit; 1. Last Valid Round; 2. Existing Artifacts; 3. Checkpoint Integrity. |
| `storage_archive_verification.md` | Document: Storage Archive Verification; 1. Current Disk State; 2. Dependency Tracing Results; 3. Minimum Local Artifact Set. |
| `storage_optimization_audit.md` | Document: Storage Optimization Audit; 1. Current Disk State; 2. Project-Level Storage Breakdown; 3. Top 10 Largest Storage Consumers. |
| `TEAM_WORKFLOW.md` | Document: Team Workflow; Before changing anything; Branching and pull requests; Experiment record required for every run. |

### manifests

| File | Explanation |
|---|---|
| `README.md` | Document: Run and artifact manifests. |
| `source_import_manifest.md` | Document: Source Import Manifest: ISP Client-Level Backdoor Unlearning; 1. Provenance & Source Metadata; 2. Reconciled Tracked Inventory (Directly from Git Index); 3. Artifacts Intentionally Excluded & Justifications. |

### presentation

| File | Explanation |
|---|---|
| `README.md` | Document: Presentation assets and status; Presentation variants in the working materials; Canonicalization before upload. |

### presentation/current

| File | Explanation |
|---|---|
| `ClientLevel_Backdoor_Unlearning_FL_initial_draft.pptx` | 27-slide current Phase 1/2 results draft. |

### presentation/historical

| File | Explanation |
|---|---|
| `Removing-Backdoors-FL-Machine-Unlearning.pptx` | 13-slide historical proposal/progress deck with earlier protocol/scope. |
| `Removing_Backdoors_from_Federated_Models.pptx` | 15-slide historical proposal/progress deck with earlier protocol/scope. |
| `Removing_Backdoors_from_Federated_Models_via_Machine_Unlearning.pptx` | 11-slide historical proposal/progress deck with earlier protocol/scope. |

### results

| File | Explanation |
|---|---|
| `clean_acc_vs_round.png` | Saved figure: clean acc vs round. |
| `federaser_corrected_runs.csv` | Federaser corrected runs; 4 data rows. Columns: `partition`, `calibration_epochs`, `clean_accuracy_pct`, `asr_ttr_pct`, `utr_pct`, `airplane_recall_pct`, `macro_f1_pct`, `clean_cf_agreement_pct`. |
| `iid_heatmap.png` | Saved figure: iid heatmap. |
| `noniid_heatmap.png` | Saved figure: noniid heatmap. |
| `per_class_asr.csv` | Per class asr; 36 data rows. Columns: `Model`, `SourceClass`, `Rate`. |
| `per_class_asr.png` | Saved figure: per class asr. |
| `phase1_note.md` | Document: Phase 1 Note. |
| `phase1_summary.csv` | Phase1 summary; 4 data rows. Columns: `partition`, `model`, `clean_accuracy_pct`, `asr_ttr_pct`, `utr_pct`. |
| `phase2_iid.csv` | Phase2 iid; 6 data rows. Columns: `model`, `clean_accuracy_pct`, `asr_ttr_pct`, `utr_pct`, `airplane_recall_pct`, `macro_f1_pct`, `macro_f1_approximate`, `clean_cf_agreement_pct`. |
| `phase2_noniid.csv` | Phase2 noniid; 6 data rows. Columns: `model`, `clean_accuracy_pct`, `asr_ttr_pct`, `utr_pct`, `airplane_recall_pct`, `macro_f1_pct`, `clean_cf_agreement_pct`, `triggered_cf_agreement_pct`. |
| `README.md` | Document: Recorded baseline result snapshots; Files; Metric denominators. |
| `results_table.csv` | Results table; 4 data rows. Columns: `Experiment`, `Clean_Acc`, `Clean_Loss`, `Macro_F1`, `Triggered_Rate`, `Untriggered_Rate`, `Runtime_min`. |
| `runtime_ledger.csv` | Runtime ledger; 11 data rows. Columns: `record`, `partition`, `seconds`, `scope`. |
| `smoke_test.json` | Smoke test. Fields: `config`, `client_distributions`, `comparison`, `raw_results`. |
| `triggered_rate_vs_round.png` | Saved figure: triggered rate vs round. |
| `triggered_sample.png` | Saved figure: triggered sample. |

### results/iid_attack

| File | Explanation |
|---|---|
| `client_class_histograms.json` | Client class histograms. Fields: `client_0`, `client_1`, `client_2`, `client_3`, `client_4`, `client_5`, `client_6`, `client_7`, `client_8`, `client_9`. |
| `config.json` | Config. Fields: `num_clients`, `clients_per_round`, `partition_type`, `dirichlet_alpha`, `malicious_client_id`, `poison_ratio`, `trigger_size`, `target_class`, `attack_start_round`, `num_rounds`. |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `final_results.json` | Final results. Fields: `config`, `mode`, `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `runtime_seconds`. Clean accuracy 75.01%. |
| `initial_model.pt` | Initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metrics_per_round.csv` | Metrics per round; 50 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`, `Attack_Active`. |

### results/iid_clean

| File | Explanation |
|---|---|
| `cf_agreement.json` | Cf agreement. Fields: `Clean_CF_Agreement`, `Triggered_CF_Agreement`. |
| `client_class_histograms.json` | Client class histograms. Fields: `client_0`, `client_1`, `client_2`, `client_3`, `client_4`, `client_5`, `client_6`, `client_7`, `client_8`, `client_9`. |
| `config.json` | Config. Fields: `num_clients`, `clients_per_round`, `partition_type`, `dirichlet_alpha`, `malicious_client_id`, `poison_ratio`, `trigger_size`, `target_class`, `attack_start_round`, `num_rounds`. |
| `extended_metrics.json` | Extended metrics. Fields: `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `airplane_recall`. Clean accuracy 75.11%. |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `final_results.json` | Final results. Fields: `config`, `mode`, `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `runtime_seconds`. Clean accuracy 75.11%. |
| `initial_model.pt` | Initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metrics_per_round.csv` | Metrics per round; 50 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`, `Attack_Active`. |

### results/noniid_attack

| File | Explanation |
|---|---|
| `client_class_histograms.json` | Client class histograms. Fields: `client_0`, `client_1`, `client_2`, `client_3`, `client_4`, `client_5`, `client_6`, `client_7`, `client_8`, `client_9`. |
| `config.json` | Config. Fields: `num_clients`, `clients_per_round`, `partition_type`, `dirichlet_alpha`, `malicious_client_id`, `poison_ratio`, `trigger_size`, `target_class`, `attack_start_round`, `num_rounds`. |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `final_results.json` | Final results. Fields: `config`, `mode`, `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `runtime_seconds`. Clean accuracy 69.8%. |
| `initial_model.pt` | Initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metrics_per_round.csv` | Metrics per round; 50 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`, `Attack_Active`. |

### results/noniid_clean

| File | Explanation |
|---|---|
| `cf_agreement.json` | Cf agreement. Fields: `Clean_CF_Agreement`, `Triggered_CF_Agreement`. |
| `client_class_histograms.json` | Client class histograms. Fields: `client_0`, `client_1`, `client_2`, `client_3`, `client_4`, `client_5`, `client_6`, `client_7`, `client_8`, `client_9`. |
| `config.json` | Config. Fields: `num_clients`, `clients_per_round`, `partition_type`, `dirichlet_alpha`, `malicious_client_id`, `poison_ratio`, `trigger_size`, `target_class`, `attack_start_round`, `num_rounds`. |
| `extended_metrics.json` | Extended metrics. Fields: `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `airplane_recall`. Clean accuracy 70.1%. |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `final_results.json` | Final results. Fields: `config`, `mode`, `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `runtime_seconds`. Clean accuracy 70.1%. |
| `initial_model.pt` | Initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metrics_per_round.csv` | Metrics per round; 50 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`, `Attack_Active`. |

### results/smoke_test_dir

| File | Explanation |
|---|---|
| `attack_final_model.pt` | Attack final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `attack_initial_model.pt` | Attack initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `attack_metrics_per_round.csv` | Attack metrics per round; 3 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`, `Attack_Active`. |
| `clean_final_model.pt` | Clean final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `clean_initial_model.pt` | Clean initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `clean_metrics_per_round.csv` | Clean metrics per round; 3 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`, `Attack_Active`. |
| `client_class_histograms.json` | Client class histograms. Fields: `client_0`, `client_1`, `client_2`, `client_3`, `client_4`, `client_5`, `client_6`, `client_7`, `client_8`, `client_9`. |
| `final_results.json` | Final results. Fields: `config`, `comparison`, `raw_results`. |

### results_phase2

| File | Explanation |
|---|---|
| `noniid_evaluation.json` | Noniid evaluation. Fields: `Phase-1 Non-IID Clean`, `Logged Non-IID Attack`, `Non-IID M_sub`, `Non-IID M_CF`. |
| `noniid_kd_evaluation.json` | Noniid kd evaluation. Fields: `Logged Non-IID Attack`, `Non-IID M_sub`, `Non-IID M_KD`, `Non-IID M_CF`. |
| `phase2_evaluation_corrected.json` | Phase2 evaluation corrected. Fields: `Phase-1 Clean`, `Logged IID Attack`, `M_sub`, `M_CF`. |
| `phase2_evaluation_superseded.json` | Phase2 evaluation superseded. Fields: `Phase-1 Clean`, `Logged IID Attack`, `M_sub`, `M_CF`. |
| `phase2_final_report.md` | Document: Phase 2: Final Validation and Findings; 1. Reproducible $M_{CF}$ Aggregation Audit; 2. Characterization of Non-IID $M_{sub}$ Performance Drop; 3. Teacher Predictions on the Proxy Data. |
| `phase2_kd_evaluation.json` | Phase2 kd evaluation. Fields: `Phase-1 Clean`, `Logged IID Attack`, `M_sub`, `M_KD`, `M_CF`. |

### results_phase2/audit_stage1

| File | Explanation |
|---|---|
| `artifact_storage_inventory.csv` | Artifact storage inventory; 6 data rows. Columns: `Description`, `Path`, `Exists`, `File_Count`, `Size_Bytes`, `Size_MiB`, `Size_GiB`. |
| `asr_vs_round.csv` | Asr vs round; 50 data rows. Columns: `Round`, `Clean_Acc_IID`, `Triggered_Target_Rate_IID`, `Untriggered_Airplane_Rate_IID`, `Attack_Active_IID`, `Clean_Acc_NonIID`, `Triggered_Target_Rate_NonIID`, `Untriggered_Airplane_Rate_NonIID`. |
| `asr_vs_round.png` | Saved figure: asr vs round. |
| `canonical_results.csv` | Canonical results; 10 data rows. Columns: `Partition`, `Model`, `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Macro_F1`, `Clean_CF_Agreement`. |
| `canonical_results.md` | Document: . |
| `git_info.txt` | Version-control state captured during the historical audit. |
| `historical_eval_diff.txt` | Recorded evaluation-metric differences. |
| `noniid_m_sub_per_class.csv` | Noniid m sub per class; 10 data rows. Columns: `Class`, `Clean_Accuracy`. |
| `noniid_m_sub_per_class_accuracy.png` | Saved figure: noniid m sub per class accuracy. |
| `noniid_m_sub_predicted_distribution.csv` | Noniid m sub predicted distribution; 10 data rows. Columns: `Predicted_Class`, `Count`, `Percentage`. |
| `noniid_m_sub_predicted_distribution.png` | Saved figure: noniid m sub predicted distribution. |
| `runtime_storage_inventory.csv` | Runtime storage inventory; 10 data rows. Columns: `Component`, `Scope`, `Time_Seconds`. |
| `stage1_audit_report.md` | Document: Stage 1: Final Source-of-Truth Audit; 1. Repository and Frozen State; 2. Canonical Results Table; 3. Non-IID M_sub Collapse. |

### results_phase2/charts

| File | Explanation |
|---|---|
| `airplane_recall.png` | Saved figure: airplane recall. |
| `clean_acc.png` | Saved figure: clean acc. |
| `clean_cf.png` | Saved figure: clean cf. |
| `kd_loss.png` | Saved figure: kd loss. |

### results_phase2/federaser

| File | Explanation |
|---|---|
| `batchnorm_policy.md` | Document: FedEraser BatchNorm Buffer Policy (Pre-Training Documentation); Floating-Point Trainable Parameters (Weights/Biases); BatchNorm Running Statistics and Counters. |
| `controlled_timing.json` | Controlled timing. Fields: `m_cf_total_3_rounds_s`, `m_cf_per_client_epoch_s`, `fe_total_3_rounds_s`, `fe_per_client_epoch_s`, `fe_io_overhead_s`, `fe_compute_only_s`, `note`. |
| `federaser_comparison.csv` | Federaser comparison; 12 data rows. Columns: `Partition`, `Model`, `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Macro_F1`, `Clean_CF_Agreement`. |
| `federaser_comparison.md` | Document: . |
| `federaser_summary.md` | Document: FedEraser Baseline Summary; Algorithm Verification & Constraints; Performance Outcomes; Next Steps . |
| `implementation_notes.md` | Document: FedEraser Implementation Notes; Algorithm Settings; Training Configuration; Structural Constraints & Norm Granularity. |
| `log_feasibility_audit.md` | Document: FedEraser Log Feasibility Audit (Resolved); 1. Initial Algorithmic Assessment; 2. Resolved Alignment; 3. Artifact Validation. |
| `RUN_REGISTRY.md` | Document: FedEraser Run Registry. |
| `seed_manifest.md` | Document: FedEraser Seed Manifest. |

### results_phase2/federaser/4_configs

| File | Explanation |
|---|---|
| `federaser_extended_comparison.csv` | Federaser extended comparison; 14 data rows. Columns: `Partition`, `Model`, `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Macro_F1`, `Clean_CF_Agreement`. |
| `federaser_extended_comparison.md` | Document: . |
| `validation_report.md` | Document: FedEraser Configurations Validation Report; 1. Explicit Configurations Overview; 2. Global L2-Norm Assertions; 3. Pre-Trained Policy Declarations. |

### results_phase2/federaser/corrected_evaluation

| File | Explanation |
|---|---|
| `audit_note.md` | Document: FedEraser Corrected Evaluation Audit; 1. Metric Definitions & Denominators; 2. Partition Equality; 3. Configuration Findings. |
| `federaser_comparison.csv` | Federaser comparison; 12 data rows. Columns: `Partition`, `Model`, `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Macro_F1`, `Clean_CF_Agreement`. |
| `federaser_comparison.md` | Document: . |
| `iid_corrected_evaluation.json` | Iid corrected evaluation. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`. Accuracy 66.4%, ASR 3.5%. |
| `noniid_corrected_evaluation.json` | Noniid corrected evaluation. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`. Accuracy 58.63%, ASR 3.7333%. |

### results_phase2/federaser/corrected_evaluation_v2

| File | Explanation |
|---|---|
| `federaser_comparison.csv` | Federaser comparison; 14 data rows. Columns: `Partition`, `Model`, `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Macro_F1`, `Clean_CF_Agreement`. |
| `federaser_comparison.md` | Document: . |

### results_phase2/federaser/iid

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`, `target_client`, `total_calibration_epochs`, `reconstruction_time_seconds`. **Superseded run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`, `Per_Class_Clean_Accuracy`. Accuracy 66.13333333333334%, ASR 3.5%. **Superseded run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Superseded run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 25 data rows. Columns: `round`, `time_s`. **Superseded run.** |

### results_phase2/federaser/iid_Ecal1

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`, `target_client`, `total_calibration_epochs`, `reconstruction_time_seconds`. **Superseded run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`, `Per_Class_Clean_Accuracy`. Accuracy 72.43%, ASR 3.1778%. **Superseded run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Superseded run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 50 data rows. Columns: `round`, `time_s`. **Superseded run.** |

### results_phase2/federaser/iid_Ecal1_v2

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `weight_decay`, `weight_decay_asserted`, `norm_granularity`, `seed_scheme`, `backend_limitations`, `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`. **Corrected current run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`. Accuracy 72.11%, ASR 2.6667%. **Corrected current run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Corrected current run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 50 data rows. Columns: `round`, `time_s`. **Corrected current run.** |

### results_phase2/federaser/iid_Ecal2

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`, `target_client`, `total_calibration_epochs`, `reconstruction_time_seconds`. **Superseded run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`, `Per_Class_Clean_Accuracy`. Accuracy 74.96%, ASR 2.1778%. **Superseded run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Superseded run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 50 data rows. Columns: `round`, `time_s`. **Superseded run.** |

### results_phase2/federaser/iid_Ecal2_v2

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `weight_decay`, `weight_decay_asserted`, `norm_granularity`, `seed_scheme`, `backend_limitations`, `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`. **Corrected current run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`. Accuracy 74.99%, ASR 2.3444%. **Corrected current run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Corrected current run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 50 data rows. Columns: `round`, `time_s`. **Corrected current run.** |

### results_phase2/federaser/noniid

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`, `target_client`, `total_calibration_epochs`, `reconstruction_time_seconds`. **Superseded run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`, `Per_Class_Clean_Accuracy`. Accuracy 58.666666666666664%, ASR 3.7333%. **Superseded run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Superseded run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 25 data rows. Columns: `round`, `time_s`. **Superseded run.** |

### results_phase2/federaser/noniid_Ecal1

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`, `target_client`, `total_calibration_epochs`, `reconstruction_time_seconds`. **Superseded run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`, `Per_Class_Clean_Accuracy`. Accuracy 65.4%, ASR 4.1222%. **Superseded run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Superseded run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 50 data rows. Columns: `round`, `time_s`. **Superseded run.** |

### results_phase2/federaser/noniid_Ecal1_v2

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `weight_decay`, `weight_decay_asserted`, `norm_granularity`, `seed_scheme`, `backend_limitations`, `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`. **Corrected current run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`. Accuracy 65.75%, ASR 3.5333%. **Corrected current run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Corrected current run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 50 data rows. Columns: `round`, `time_s`. **Corrected current run.** |

### results_phase2/federaser/noniid_Ecal2

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`, `target_client`, `total_calibration_epochs`, `reconstruction_time_seconds`. **Superseded run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`, `Per_Class_Clean_Accuracy`. Accuracy 69.0%, ASR 2.5444%. **Superseded run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Superseded run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 50 data rows. Columns: `round`, `time_s`. **Superseded run.** |

### results_phase2/federaser/noniid_Ecal2_v2

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `weight_decay`, `weight_decay_asserted`, `norm_granularity`, `seed_scheme`, `backend_limitations`, `retained_rounds`, `cali_epochs`, `batch_size`, `lr`, `momentum`. **Corrected current run.** |
| `eval_results.json` | Eval results. Fields: `Clean_Accuracy`, `Clean_Loss`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Clean_CF_Agreement`, `Triggered_CF_Agreement`, `Airplane_Recall`, `Macro_F1`. Accuracy 68.89%, ASR 3.0889%. **Corrected current run.** |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. **Corrected current run.** |
| `reconstruction_logs.csv` | Reconstruction logs; 50 data rows. Columns: `round`, `time_s`. **Corrected current run.** |

### results_phase2/iid_attack_logged

| File | Explanation |
|---|---|
| `client_class_histograms.json` | Client class histograms. Fields: `client_0`, `client_1`, `client_2`, `client_3`, `client_4`, `client_5`, `client_6`, `client_7`, `client_8`, `client_9`. |
| `config.json` | Config. Fields: `num_clients`, `clients_per_round`, `partition_type`, `dirichlet_alpha`, `malicious_client_id`, `poison_ratio`, `trigger_size`, `target_class`, `attack_start_round`, `num_rounds`. |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `final_results.json` | Final results. Fields: `config`, `mode`, `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `runtime_seconds`. Clean accuracy 75.01%. |
| `initial_model.pt` | Initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metrics_per_round.csv` | Metrics per round; 50 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`, `Attack_Active`. |
| `runtime.json` | Runtime. Fields: `runtime_seconds`. |

### results_phase2/iid_counterfactual

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `num_clients`, `clients_per_round`, `partition_type`, `dirichlet_alpha`, `malicious_client_id`, `poison_ratio`, `trigger_size`, `target_class`, `attack_start_round`, `num_rounds`. |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `final_results.json` | Final results. Fields: `config`, `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `runtime_seconds`. Clean accuracy 74.87%. |
| `initial_model.pt` | Initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metrics_per_round.csv` | Metrics per round; 50 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`. |

### results_phase2/iid_kd

| File | Explanation |
|---|---|
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metadata.json` | Metadata. Fields: `epochs`, `batch_size`, `lr`, `momentum`, `T`, `optimizer`, `hyperparameter_note`, `kd_runtime_seconds`, `subtraction_runtime_seconds`, `total_unlearning_runtime`. |

### results_phase2/iid_subtraction

| File | Explanation |
|---|---|
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `subtraction_runtime.json` | Subtraction runtime. Fields: `runtime_seconds`. |

### results_phase2/noniid_attack_logged

| File | Explanation |
|---|---|
| `client_class_histograms.json` | Client class histograms. Fields: `client_0`, `client_1`, `client_2`, `client_3`, `client_4`, `client_5`, `client_6`, `client_7`, `client_8`, `client_9`. |
| `config.json` | Config. Fields: `num_clients`, `clients_per_round`, `partition_type`, `dirichlet_alpha`, `malicious_client_id`, `poison_ratio`, `trigger_size`, `target_class`, `attack_start_round`, `num_rounds`. |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `final_results.json` | Final results. Fields: `config`, `mode`, `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `runtime_seconds`. Clean accuracy 69.8%. |
| `initial_model.pt` | Initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metrics_per_round.csv` | Metrics per round; 50 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`, `Attack_Active`. |
| `runtime.json` | Runtime. Fields: `runtime_seconds`. |

### results_phase2/noniid_counterfactual

| File | Explanation |
|---|---|
| `config.json` | Config. Fields: `num_clients`, `clients_per_round`, `partition_type`, `dirichlet_alpha`, `malicious_client_id`, `poison_ratio`, `trigger_size`, `target_class`, `attack_start_round`, `num_rounds`. |
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `final_results.json` | Final results. Fields: `config`, `clean_accuracy`, `clean_loss`, `clean_macro_f1`, `triggered_rate`, `untriggered_airplane_rate`, `runtime_seconds`. Clean accuracy 69.64%. |
| `initial_model.pt` | Initial model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metrics_per_round.csv` | Metrics per round; 50 data rows. Columns: `Round`, `Clean_Acc`, `Triggered_Target_Rate`, `Untriggered_Airplane_Rate`. |

### results_phase2/noniid_kd

| File | Explanation |
|---|---|
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `metadata.json` | Metadata. Fields: `epochs`, `batch_size`, `lr`, `momentum`, `T`, `optimizer`, `kd_runtime_seconds`, `subtraction_runtime_seconds`, `total_unlearning_runtime`, `kd_loss_history`. |

### results_phase2/noniid_subtraction

| File | Explanation |
|---|---|
| `final_model.pt` | Final model PyTorch state-dictionary checkpoint. See the run-directory explanation above; tensors were not deserialized. |
| `subtraction_runtime.json` | Subtraction runtime. Fields: `runtime_seconds`, `c0_weight`, `n_0`, `N`. |

### results_phase2/proxy_data

| File | Explanation |
|---|---|
| `cifar100_proxy_indices.json` | Cifar100 proxy indices; list of 5000 entries. |
| `proxy_grid.png` | Saved figure: proxy grid. |

### scripts

| File | Explanation |
|---|---|
| `audit_kd.py` | Inspect attacked-teacher predictions on proxy examples and create a sample grid. |
| `calc_storage.py` | Estimate model tensor size and storage for client/global historical updates. |
| `check_batches_tracked.py` | Check corrected checkpoints preserve initial integer BatchNorm counters. |
| `compile_4_configs_results.py` | Compile intermediate four-setting reconstruction comparison. |
| `compile_corrected_results.py` | Compile corrected evaluation of original reconstruction runs. |
| `compile_federaser_results.py` | Compile original reconstruction/Phase 2 comparison. |
| `compile_v2_results.py` | Compile corrected _v2 reconstructions and extended Phase 1 metrics. |
| `compile_v3_results.py` | Compile the current full corrected comparison with Phase 1 clean CF agreement and corrected _v2 reconstructions. |
| `corrected_evaluate_federaser.py` | Correct full-population evaluation and true Macro-F1 for original reconstruction checkpoints. |
| `diff_runs.py` | Compare intermediate versus corrected reconstruction metrics. |
| `download_hf_cifar.py` | Reformats an existing local CIFAR-100 pickle; performs no download. |
| `dump_trigger_image.py` | Save a de-normalized sample containing the white trigger. |
| `eval_noniid_clean.py` | Misnamed: evaluates IID clean checkpoint and writes IID extended metrics. |
| `eval_only_federaser.py` | Older evaluator of superseded iid/noniid runs; mixed populations and incorrectly named Macro-F1. |
| `eval_p1_cf.py` | Compute clean/triggered agreement of Phase 1 clean checkpoints with matching references. |
| `evaluate_noniid_subtraction.py` | Evaluate Non-IID clean/attack/subtraction/reference endpoints, including per-class utility and CF agreement. |
| `evaluate_phase2.py` | Initial IID evaluator with prediction agreement and per-class ASR; its original results are superseded. |
| `evaluate_phase2_corrected.py` | Corrected IID pre-KD evaluator with proper per-class clean accuracy and CF agreement. |
| `final_audit.py` | Audit proxy/model provenance and metrics; generate comparison charts and hashes. |
| `final_cleanup.py` | Consolidate evaluation/chart records and regenerate historical hashes; some inputs are omitted partition files. |
| `final_report_script.py` | Compile subtraction/KD results and comparison/loss charts. |
| `final_verification.py` | Verify poison-selection counts/identities and recompute per-class triggered rates; needs saved partitions. |
| `make_meta.py` | Creates CIFAR-100 metadata with placeholder class names. |
| `patch_cifar.py` | Changes CIFAR100 integrity metadata within this Python process only. |
| `per_class_asr.py` | Calculate trigger target rates separately for each original class. |
| `plot_results.py` | Compile Phase 1 table and plot accuracy/trigger trajectories plus client-class heatmaps. |
| `post_check.py` | Print saved Non-IID Client 0 counts and current attack settings. |
| `post_verification.py` | Compare pre-attack clean/attack trajectories and partition/checkpoint identity. |
| `proxy_audit.py` | Select balanced CIFAR-100 proxy indices, hash raw pixels, and record provenance. |
| `README.md` | Document: Scripts. |
| `run_baseline_federaser.py` | Original reconstruction runner; its outputs are superseded due to optimizer/evaluation issues. |
| `run_experiment_v2.py` | Main clean/attack FL launcher; partitions data, trains all ten clients each round, evaluates, saves endpoints, and optionally logs every delta. |
| `run_federaser_4_configs.py` | Intermediate Ecal=1/2 reconstruction runner predating corrected seeds/BatchNorm counters. |
| `run_federaser_v5.py` | Corrected reconstruction for IID/Non-IID with Ecal=1/2: seeded calibration, norm checks, BN counter preservation, final evaluation and timing. |
| `run_four_experiments.sh` | Bash wrapper launching run four experiments. Read before running: these scripts launch training. |
| `run_kd.py` | IID five-epoch distillation from attacked teacher into subtracted student on balanced CIFAR-100 proxy; BN statistics frozen. |
| `run_m_cf.py` | Train IID client-excluded reference from matching saved initialization/partitions; renormalize weights over Clients 1-9. |
| `run_noniid_experiments.sh` | Bash wrapper launching run noniid experiments. Read before running: these scripts launch training. |
| `run_noniid_m_cf.py` | Train Non-IID client-excluded reference with retained sample-count weights. |
| `run_noniid_stage_abc.py` | Audit logged attack/subtraction prerequisites, run Non-IID KD, evaluate final endpoints. |
| `run_noniid_subtraction.py` | Non-IID subtraction using Client 0 actual sample fraction derived from saved partitions. |
| `run_subtraction.py` | IID historical subtraction across all 50 rounds using Client 0 weight 0.1. |
| `stage1_audit.py` | Original audit of baseline tables, trigger trajectories, class-collapse diagnostics and storage. |
| `stage1_audit_v2.py` | Revised baseline audit with clearer runtime reporting and metric checks. |
| `stage1_audit_v3.py` | Normalize Macro-F1 percentages, separate runtime type/seconds, and validate trajectories. |
| `stage1_audit_v4.py` | Further baseline table revision with corrected KD timing keys/fallbacks. |
| `step3_verify.py` | Check logged IID replay matches original metric trajectories/hashes; needs omitted partitions for full execution. |
| `summarize.py` | Print four Phase 1 endpoint metrics and runtimes. |
| `verify_agreements.py` | Independently count selected IID reference agreement and target predictions. |
| `verify_partitions.py` | Compare regenerated IID/Non-IID sample indices with saved partition objects. |

### scripts/_archive

| File | Explanation |
|---|---|
| `analyze_counterfactual.py` | Historical prediction, parameter, layer-distance and trigger-variant comparison against retraining. |
| `analyze_geometry_d0_d3.py` | Historical experiment utility: analyze geometry d0 d3. Defines `get_trainable_params`, `flatten_dict`, `analyze_round`. Input/output example: `results/stage7a_c1_attacked_20260910_233012`. |
| `analyze_geometry_d1_canonical.py` | Analyze matched-strength D1 update geometry using eligible trainable tensors. |
| `analyze_geometry_d1_corrected.py` | Historical experiment utility: analyze geometry d1 corrected. Defines `get_trainable_params`, `flatten_dict`. Input/output example: `results/stage7c_d1_attacked_20260911_230919`. |
| `analyze_geometry_d2.py` | Historical experiment utility: analyze geometry d2. Defines `get_update_vector`, `analyze_round`. Input/output example: `results/stage7c_d2/geometry_analysis.json`. |
| `analyze_geometry_d2_corrected.py` | Correct D2 update geometry for stored scaling and eligible tensors. |
| `analyze_stage7a_attack_impact.py` | Historical experiment utility: analyze stage7a attack impact. |
| `analyze_stage7a_unlearning.py` | Historical experiment utility: analyze stage7a unlearning. Input/output example: `results/stage7a_unlearning`. |
| `analyze_trajectory.py` | Plot reconstruction trajectories and method cost/fidelity comparisons. |
| `analyze_update_geometry.py` | Original historical norm/dispersion analysis, corrected later to exclude integer buffers. |
| `archive_move.py` | Historical maintenance script moving selected run folders into archive staging and recording manifests. |
| `audit_d2_partition.py` | Historical experiment utility: audit d2 partition. Defines `gini`, `entropy`, `js_div`. Input/output example: `results/stage7c_partitions/d2_distribution.json`. |
| `audit_federaser.py` | Historical experiment utility: audit federaser. Defines `get_flattened_float_vector`, `cosine_similarity`, `run_audit`. Input/output example: `results/attack_scaling_final_gamma5_20260909_015903`. |
| `audit_stage7a_geometry.py` | Historical experiment utility: audit stage7a geometry. Defines `load_updates`. Input/output example: `results/stage7a_audit/update_geometry_audit.csv`. |
| `audit_stage7a_metrics.py` | Historical experiment utility: audit stage7a metrics. Defines `evaluate`, `load_model`. Input/output example: `results/stage7a_audit`. |
| `audit_stage7b_final.py` | Historical experiment utility: audit stage7b final. Defines `evaluate`, `get_kl_div`, `load_model`, `get_latest_dir`. Input/output example: `results/stage7a_c1_attacked_20260910_233012`. |
| `audit_stage7b_geometry.py` | Correct geometry using only trainable floating-point parameters. |
| `audit_stage7b_h2_imbalance.py` | Inspect strongly heterogeneous H2 client-size imbalance. |
| `audit_stage7c_d1_canonical.py` | Historical experiment utility: audit stage7c d1 canonical. Defines `get_predictions`. Input/output example: `results/attack_badnets_20260912_182946/checkpoints/global_round_100.pt`. |
| `audit_stage7c_d1_final.py` | Historical experiment utility: audit stage7c d1 final. Defines `get_predictions`. Input/output example: `results/stage7c_d1/unlearning/federaser/checkpoints/global_round_100.pt`. |
| `audit_stage7c_d2_final.py` | Historical experiment utility: audit stage7c d2 final. Defines `evaluate`. Input/output example: `results/stage7c_d2/unlearning/federaser/checkpoints/federaser_round_100.pt`. |
| `audit_stage7c_d2_final_corrected.py` | Recompute D2 final-checkpoint metrics using frozen evaluation indices. |
| `audit_stage7c_d2_recovery.py` | Historical experiment utility: audit stage7c d2 recovery. Defines `get_latest_dir`. |
| `audit_stage7c_final.py` | Historical experiment utility: audit stage7c final. Defines `evaluate`, `get_kl_div`, `load_model`, `get_latest_dir`. Input/output example: `results/stage7a_c1_attacked_20260910_233012`. |
| `check_batchnorm.py` | Check partition-size/BatchNorm workability for historical factorial conditions. |
| `compile_stage7c_robustness.py` | Compile final factorial results, realized attack impact and descriptive gap ratios. |
| `create_d1_partition.py` | Construct equal-size label-heterogeneous D1 client indices, distributions and manifests. |
| `evaluate_phase_a.py` | Historical experiment utility: evaluate phase a. Defines `load_metrics`. Input/output example: `results/stage_6d_temporal`. |
| `evaluate_phase_b.py` | Historical experiment utility: evaluate phase b. Defines `get_model`, `evaluate_subset`, `get_predictions`, `calc_agreement`. Input/output example: `results/stage6d_phase_b_full_`. |
| `evaluate_predictions.py` | Historical experiment utility: evaluate predictions. Defines `get_model`, `get_predictions`, `calc_agreement`. Input/output example: `results/full_retraining_final_20260909_024627/checkpoints/global_round_100.pt`. |
| `evaluate_stage7a_unlearning.py` | Historical experiment utility: evaluate stage7a unlearning. Defines `evaluate`, `load_model`. Input/output example: `results/stage7a_c1_mcf_20260911_003109`. |
| `plot_stage5a.py` | Historical experiment utility: plot stage5a. Defines `calculate_displacement`, `plot_metrics`. Input/output example: `results/attack_scaling_final_gamma5_20260909_015903/checkpoints/global_round_100.pt`. |
| `quick_mcf_eval.py` | Historical experiment utility: quick mcf eval. Defines `eval_mcf`. Input/output example: `results/stage7b_c1_mcf_20260911_134941/checkpoints/global_round_100.pt`. |
| `run_centralized.py` | Historical conventional single-machine trainer with YAML settings, checkpoints, logs and curves. |
| `run_federaser.py` | Historical reconstruction using retained clients, sampled-client history, stored updates and calibrated directions. |
| `run_federated.py` | Historical 20-client partial-participation trainer with scaling, exclusion, custom partitions and staged-study support. |
| `run_scaling_study.sh` | Bash wrapper launching run scaling study. Read before running: these scripts launch training. |
| `run_stage6d_phase_a.py` | Launch early/middle/late attacks with matched no-attack and client-excluded controls. |
| `run_stage6d_phase_b_dryrun.py` | Small historical unlearning dry run before the timing matrix. |
| `run_stage6d_phase_b_full.py` | Launch three methods across three historical timing conditions. |
| `run_stage7a.py` | Historical experiment utility: run stage7a. Input/output example: `configs/attack_badnets.yaml`. |
| `run_stage7a_unlearning.py` | Historical experiment utility: run stage7a unlearning. Input/output example: `results/stage7a_c1_attacked_20260910_233012`. |
| `run_stage7b_mcf.py` | Corrected historical reference exclusion mapping and reference training. |
| `run_stage7b_unlearning.py` | Rerun historical methods with corrected reference/evaluation/checkpoint policies. |
| `run_stage7c_d1_canonical_unlearning.py` | Historical experiment utility: run stage7c d1 canonical unlearning. Input/output example: `results/attack_badnets_20260912_182946`. |
| `run_stage7c_d1_federaser.py` | Historical experiment utility: run stage7c d1 federaser. Input/output example: `results/stage7c_d1/unlearning`. |
| `run_stage7c_d1_unlearning.py` | Historical experiment utility: run stage7c d1 unlearning. Input/output example: `results/stage7c_d1/unlearning`. |
| `run_stage7c_d2_recovery.py` | Relaunch interrupted D2 attacked training with improved checkpoint safeguards. |
| `run_stage7c_d2_unlearning.py` | Historical experiment utility: run stage7c d2 unlearning. Defines `run_cmd`. Input/output example: `results/stage7c_d2/unlearning`. |
| `run_stage7c_fl.py` | Historical experiment utility: run stage7c fl. Input/output example: `configs/attack_badnets.yaml`. |
| `run_stage7c_unlearning.py` | Historical experiment utility: run stage7c unlearning. Defines `get_latest_dir`. Input/output example: `results/stage7c_unlearning`. |
| `stage7a_generate_partitions.py` | Build controlled H0/H1/H2 partitions with fixed Client 0 data. |
| `stage7c_generate_partitions.py` | Generate D1 label-only and D2 size-only factorial conditions. |
| `storage_audit.py` | Inspect historical directory sizes for artifact-retention planning. |
| `test_batchnorm.py` | Historical BatchNorm workability probe for D2 partitions. |
| `test_run.py` | Historical experiment utility: test run. Input/output example: `configs/stage7c_d1_attacked_canonical.yaml`. |
| `test_sinkhorn.py` | Exploratory matrix-balancing probe for controlled partition construction. |
| `unlearn_stage5a.py` | Historical weighted/scaled Client 0 subtraction with stored-update representation checks. |
| `unlearn_stage5b.py` | Historical KD repair using CIFAR-10 test proxy and separate evaluation subset. |
| `validate_d2_attacked.py` | Historical experiment utility: validate d2 attacked. Defines `evaluate`. Input/output example: `results/stage7c_d2_attacked_recovery_20260911_194422_20260911_194423/checkpoints/global_round_100.pt`. |
| `validate_mcf.py` | Historical experiment utility: validate mcf. Defines `evaluate`. Input/output example: `results/stage7c_d2/mcf_validation.json`. |
| `validate_stage7a.py` | Historical experiment utility: validate stage7a. Defines `check_partitions`. Input/output example: `results/stage7a_partitions`. |
| `validate_stage7c_d1_mcf.py` | Historical experiment utility: validate stage7c d1 mcf. Defines `evaluate`. Input/output example: `results/stage7c_d1_attacked_20260911_230919`. |
| `validate_stage7c_d1_partition.py` | Historical experiment utility: validate stage7c d1 partition. Input/output example: `results/stage7c_partitions/d1_indices.json`. |
| `validate_timing.py` | Verify historical attack windows against Client 0 participation events. |

### src

| File | Explanation |
|---|---|
| `README.md` | Document: Source code. |

### src/data

| File | Explanation |
|---|---|
| `backdoor.py` | Implementation: backdoor. Defines `PoisonedDataset`, `create_asr_dataloader`. |
| `cifar10.py` | Implementation: cifar10. Defines `get_cifar10_dataloaders`. |
| `fl_partition.py` | Implementation: fl partition. Defines `create_iid_partitions`, `save_partition_metadata`, `create_noniid_partitions`. |

### src/models

| File | Explanation |
|---|---|
| `small_cnn.py` | Implementation: small cnn. Defines `SmallCNN`. |

### src/training

| File | Explanation |
|---|---|
| `fedavg.py` | Implementation: fedavg. Defines `compute_model_delta`, `aggregate_fedavg`, `get_communication_bytes`. |
| `train_utils.py` | Implementation: train utils. Defines `train_epoch`, `evaluate_extended`, `evaluate`. |

### src/unlearning

| File | Explanation |
|---|---|
| `federaser.py` | Implementation: federaser. Defines `compute_l2_norm_of_floats`, `calibrate_update`. |

### src/utils

| File | Explanation |
|---|---|
| `logger.py` | Implementation: logger. Defines `CSVLogger`, `save_config`. |
| `seed.py` | Implementation: seed. Defines `set_seed`. |
| `viz.py` | Implementation: viz. Defines `plot_training_curves`. |

### tests

| File | Explanation |
|---|---|
| `test_attack.py` | Implementation: test attack. Defines `clean_dataset`, `test_trigger_geometry`, `test_poison_ratio`, `test_create_asr_dataloader`, `test_update_scaling`. |
| `test_data.py` | Implementation: test data. Defines `test_get_cifar10_dataloaders`. |
| `test_federaser.py` | Implementation: test federaser. Defines `TestFedEraserMath`. |
| `test_federaser_toy.py` | Implementation: test federaser toy. Defines `TinyNet`, `run_toy_fl_test`, `test_toy_fl`. |
| `test_federated.py` | Implementation: test federated. Defines `test_partitioning`, `test_fedavg_arithmetic`, `test_compute_delta`. |
| `test_model.py` | Implementation: test model. Defines `test_small_cnn_instantiation`, `test_small_cnn_forward_shape`. |
| `test_seed.py` | Implementation: test seed. Defines `test_seed_reproducibility`. |

### tests/historical

| File | Explanation |
|---|---|
| `README.md` | Document: Historical Test Suite; Preserved Tests; Running Historical Tests. |
| `test_benchmark_freeze.py` | Implementation: test benchmark freeze. Defines `TestBenchmarkFreeze`. Input/output example: `results/attack_scaling_final_gamma5_20260909_015903`. |
| `test_stage7b_counterfactual.py` | Implementation: test stage7b counterfactual. Defines `test_counterfactual_exclusion`. Input/output example: `configs/federated_baseline.yaml`. |

### Added by this walkthrough

| File | Explanation |
|---|---|
| `docs/PROJECT_WALKTHROUGH.md` | This explanation and complete original-file inventory. |
