# ISP Project: Client-Level Backdoor Unlearning in Federated Learning

Research workspace for the ISP project **Removing Backdoors from Federated Models via Machine Unlearning**.

The core objective is to evaluate whether the influence of a single malicious federated client (Client 0) can be removed to eliminate a backdoor while preserving utility on clean tasks. The repository implements and compares:
1. **Direct Historical-Update Subtraction ($M_{\text{sub}}$)**
2. **Subtraction followed by Knowledge Distillation ($M_{\text{KD}}$)**
3. **FedEraser Reconstruction ($M_{\text{FedEraser}}$ with $E_{\text{cal}} \in \{1, 2\}$)**
4. **Counterfactual Reference Retraining ($M_{\text{CF}}$)**: Independent reference trained from scratch on clean clients (C1–C9).

---

## 1. Quick Start & Setup

### Environment Setup
The project has been tested and verified exclusively on **Python 3.11.6** and **PyTorch 2.14.0** (with Torchvision 0.29.0, NumPy 2.4.6, Pandas 3.0.5, Matplotlib 3.11.1, PyYAML 6.0.3, tqdm 4.70.0, pytest 9.1.1). Compatibility with other Python versions (e.g., 3.10 or 3.12+) or other PyTorch versions is explicitly **unverified**.
```bash
# Clone the repository (default branch: main)
git clone https://github.com/mankiratsaini2005/ISP-Client-Level-Backdoor-Unlearning.git
cd ISP-Client-Level-Backdoor-Unlearning

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests & Verification
```bash
# Run the active unit test suite (15/15 passed)
pytest

# Run the historical test suite (6 expected failures due to archived prototype paths)
pytest tests/historical/

# Run a fast 2-round CIFAR-10 FL smoke test
python test_cifar.py
```

---

## 2. Repository Directory Map

```text
ISP-Client-Level-Backdoor-Unlearning/
├── src/                                     # Core model, data, federated, and unlearning logic
│   ├── data/                                # CIFAR-10 data loaders, backdoor injection, partition logic
│   ├── models/                              # SmallCNN model architecture
│   ├── training/                            # FedAvg aggregation & training loops
│   ├── unlearning/                          # FedEraser calibration (with L2-norm parity assertion)
│   └── utils/                               # Seeding, logging, and visualization utilities
├── scripts/                                 # 114 training, unlearning, evaluation, and audit scripts
│   ├── run_experiment_v2.py                 # Logged FL attack training (generates trajectories)
│   ├── run_m_cf.py & run_noniid_m_cf.py     # Counterfactual reference retraining (M_CF)
│   ├── run_subtraction.py                   # Historical update subtraction (M_sub)
│   ├── run_kd.py                            # Knowledge distillation unlearning (M_KD)
│   ├── run_federaser_v5.py                  # Canonical FedEraser baselines (E_cal in {1, 2})
│   ├── eval_p1_cf.py & eval_noniid_clean.py # CF agreement and extended evaluation scripts
│   └── _archive/                            # Archived historical scripts and prototype runners
├── configs/                                 # Canonical experiment YAML specifications
│   ├── v2_iid.yaml                          # IID 50-round FL specification
│   ├── v2_noniid.yaml                       # Non-IID Dirichlet (alpha=0.5) specification
│   ├── v2_smoke.yaml                        # Fast smoke-test configuration
│   └── _archive/                            # Archived run configurations
├── tests/                                   # Active unit test suite
│   ├── historical/                          # Archived prototype tests (isolated from active suite)
│   └── [test_*.py]                          # Core unit tests (attack, data, model, federaser, etc.)
├── docs/                                    # Detailed documentation, guides, and audit logs
│   ├── developer_guide.md                   # Complete teammate developer & reproducibility guide
│   ├── EXPERIMENT_PROTOCOL.md               # Canonical 50-round experiment specifications
│   ├── METHODS_AND_METRICS.md               # Mathematical formulations and metric definitions
│   └── [analysis reports]                   # Factorial robustness audits and stage reports
├── results/                                 # Phase-1 baseline models, metrics, CSVs, and figures
│   ├── iid_clean/ & iid_attack/             # Model checkpoints, per-round metrics, client histograms
│   ├── noniid_clean/ & noniid_attack/       # Model checkpoints, per-round metrics, client histograms
│   └── [CSVs & PNGs]                        # Baseline metric snapshots, heatmaps, and curves
├── results_phase2/                          # Phase-2 unlearning models, evaluations, and artifacts
│   ├── federaser/                           # Corrected v2 runs, RUN_REGISTRY.md, seed_manifest.md
│   ├── iid_subtraction/ & noniid_subtraction/ # M_sub checkpoints and timing metadata
│   ├── iid_kd/ & noniid_kd/                 # M_KD checkpoints and metadata
│   ├── iid_counterfactual/ & noniid_counterfactual/ # M_CF checkpoints and round metrics
│   ├── audit_stage1/                        # Canonical audit logs and reports
│   └── proxy_data/                          # CIFAR-100 proxy dataset indices and sample grid
├── presentation/                            # Editable PowerPoint slide decks
│   ├── current/                             # ClientLevel_Backdoor_Unlearning_FL_initial_draft.pptx
│   └── historical/                          # Removing_Backdoors_from_Federated_Models.pptx and decks
└── manifests/                               # Provenance and checksum manifests
    └── source_import_manifest.md            # Detailed record of imported files and exclusions
```

---

## 3. Experiment Rerunnability & Trajectory Prerequisites

> [!IMPORTANT]
> **Reproduction Limitation & Trajectory Requirements**:
> All evaluated global model checkpoints (`final_model.pt`, `initial_model.pt`), configurations, evaluation metrics, and audit scripts are committed to Git (~75 MB total).
>
> However, to comply with GitHub storage constraints, the large per-round client delta directories (`trajectory/`, ~2.2 GB) and pickled dataset objects (`client_partitions.pt`, ~1.48 GB) are excluded from the repository.

### Commands That Run Immediately (No Retraining Required)
Using the committed checkpoints in `results/` and `results_phase2/`:
- **Model Evaluation**:
  ```bash
  PYTHONPATH=. python scripts/eval_only_federaser.py
  PYTHONPATH=. python scripts/eval_p1_cf.py
  PYTHONPATH=. python scripts/eval_noniid_clean.py
  PYTHONPATH=. python scripts/evaluate_phase2_corrected.py
  ```
- **Result Summarization & Audit**:
  ```bash
  PYTHONPATH=. python scripts/compile_v3_results.py
  PYTHONPATH=. python scripts/check_batches_tracked.py
  PYTHONPATH=. python scripts/diff_runs.py
  ```
- **Plotting**:
  ```bash
  PYTHONPATH=. python scripts/plot_results.py
  PYTHONPATH=. python scripts/per_class_asr.py
  ```

### Commands Requiring Trajectory Regeneration
FedEraser (`scripts/run_federaser_v5.py`) and historical subtraction (`scripts/run_subtraction.py`) require historical per-round client updates. To regenerate them:
```bash
# Regenerate IID logged attack trajectory (50 rounds):
PYTHONPATH=. python scripts/run_experiment_v2.py \
    --config configs/v2_iid.yaml \
    --mode attack \
    --log_trajectory \
    --output_dir results_phase2/iid_attack_logged

# Regenerate Non-IID logged attack trajectory (50 rounds):
PYTHONPATH=. python scripts/run_experiment_v2.py \
    --config configs/v2_noniid.yaml \
    --mode attack \
    --log_trajectory \
    --output_dir results_phase2/noniid_attack_logged
```

> [!NOTE]
> **Trajectory Output Counts & Validation Status**:
> - **File Counts**: Each trajectory generation command executes 50 rounds with 10 clients per round, producing exactly **500 client-update delta files per setting** (`trajectory/round_{r:03d}/delta_client_{c:02d}.pt` for rounds 1–50 and clients 0–9), or **1,000 client-update files combined** across both IID and Non-IID if all updates are successfully logged. Each round also saves the round-level `global_model.pt` checkpoint, alongside the overall `client_partitions.pt` and `client_class_histograms.json` metadata files.
> - **Validation Status**: The full 50-round trajectory regeneration has **not** been executed end-to-end to avoid long retraining times and to preserve existing baseline artifacts; full 50-round regeneration remains **unverified end-to-end**. However, the script mechanics, directory creation, per-client delta serialization, and partition metadata generation have been verified via a controlled 1-round smoke test with temporary outputs. Note that running trajectory regeneration requires CIFAR-10 data extracted at `./data/cifar-10-batches-py`.

---

## 4. Key Experimental Baseline Observations

- **Clean accuracy alone does not detect the backdoor**: In Phase 1 IID, clean accuracy was virtually identical (75.11% clean vs. 75.01% attack), while the Attack Success Rate (ASR) jumped from 2.73% to 56.07%.
- **Direct Subtraction ($M_{\text{sub}}$)**: Eliminates the backdoor but causes catastrophic utility collapse under Non-IID partitions (28.04% clean accuracy in the saved split).
- **Knowledge Distillation Recovery ($M_{\text{KD}}$)**: Partially recovers clean accuracy (72.82% IID, 60.36% Non-IID) with low ASR.
- **FedEraser Reconstruction**: Reaches 74.99% accuracy / 2.34% ASR in IID ($E_{\text{cal}}=2$) and 68.89% accuracy / 3.09% ASR in the saved Non-IID partition ($E_{\text{cal}}=2$).
- **Counterfactual Model ($M_{\text{CF}}$)**: Serves as an empirical reference point trained without Client 0 (74.87% IID, 69.64% Non-IID).

For complete protocol details, consult [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) and [docs/developer_guide.md](docs/developer_guide.md).

## Checkerboard attack pilot

An optional 4x4 bottom-right black/white checkerboard variant is configured in `configs/v3_checkerboard4_iid.yaml` and `configs/v3_checkerboard4_noniid.yaml`. See [docs/CHECKERBOARD_ATTACK.md](docs/CHECKERBOARD_ATTACK.md) for commands to train only M_attack and inspect its CA/ASR. Each run writes its own `attack_report.md`; reusing a nonempty checkerboard output path automatically selects a fresh numbered sibling. Stronger ASR remains to be measured. The original v2 white-patch configurations retain their trigger behavior.

For subtraction, KD and FedEraser on these new models, use the variant-aware launcher and commands in [docs/CHECKERBOARD_UNLEARNING.md](docs/CHECKERBOARD_UNLEARNING.md). It keeps the new trigger, source histories and result folders consistent without changing the old white-patch launchers.

For selective Conv2 + BatchNorm repair using G10/G20/G29 donors, use [docs/COMPONENT_PATCHING.md](docs/COMPONENT_PATCHING.md). It documents the whole-block interpolation grid, individual-channel rankings and prefix controls, selection/report separation, optional frozen-layer calibration, and commands for both saved checkerboard attack models.

## Extreme label skew stress test

Run `python scripts/run_label_skew_stress.py` in the training environment to compare subtraction, KD and FedEraser on a fresh Dirichlet alpha=0.01 checkerboard attack. Use `--plan` to inspect commands, `--preflight_only` to inspect actual partitions, or `--seeds 42 43 44` for partition sensitivity. See [docs/EXTREME_LABEL_SKEW.md](docs/EXTREME_LABEL_SKEW.md) for output files and interpretation.
