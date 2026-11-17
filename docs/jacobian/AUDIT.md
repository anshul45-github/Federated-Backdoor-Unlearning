# JVP-Cal Experiment Audit
*Branch: jacobian-cal | Audited from code on 2026-11-16*

---

## (a) Local Optimizer Hyperparameters

Sourced from `configs/v2_iid.yaml`, `configs/v2_noniid.yaml`, and confirmed in `scripts/run_experiment_v2.py` (line 135) and `scripts/baseline/run_federaser_v5.py` (lines 24–25, 160–161).

| Parameter | Value | Source |
|---|---|---|
| Optimizer | SGD | `run_experiment_v2.py:135`, `run_federaser_v5.py:160` |
| Learning rate (lr) | 0.01 | `v2_iid.yaml:14`, asserted in runner |
| Momentum | 0.9 | `v2_iid.yaml:15` |
| Weight decay | 0.0 (explicitly set; `weight_decay` absent from YAML → `optim.SGD` default) | `run_federaser_v5.py:24,161` |
| Batch size | 64 | `v2_iid.yaml:12` |
| Local epochs E | 2 | `v2_iid.yaml:11` |
| Rounds | 50 | `v2_iid.yaml:10` |
| Clients per round K | 10 (full participation) | `v2_iid.yaml:1-2` |

**Steps per round per client:**  
Each client runs E=2 local epochs on its partition. IID: 5000 samples ÷ 64 batch = 79 steps/epoch × 2 = **~158 gradient steps per client per round**. Non-IID varies by partition size.

> [!NOTE]
> The YAML does not set `weight_decay`. The SGD call in `run_experiment_v2.py:135` is `optim.SGD(local_model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])` — weight decay defaults to 0.0. The FedEraser runner explicitly sets and asserts `weight_decay=0.0`.

---

## (b) Retained Rounds and Retention Interval

From `run_federaser_v5.py` line 21:
```python
retained_rounds = list(range(1, 51))
```
**All 50 rounds (1–50 inclusive) are retained.** There is no retention interval or subsampling — every round's client deltas are replayed in calibration.

---

## (c) FedEraser Calibration Implementation

Sourced from `src/unlearning/federaser.py` and `scripts/baseline/run_federaser_v5.py`.

### Round 1 (special case)
Round 1 uses historical deltas **unchanged** (no calibration training):
```python
if round_num == 1:
    tilde_delta = hist_delta   # run_federaser_v5.py:149
```

### Rounds 2–50 (calibration)
For each retained client c in rounds 2–50:
1. Load `hist_delta = trajectory/round_{r:03d}/delta_client_{c:02d}.pt`
2. Train a local model from the **current reconstructed global state** for `Ecal` epochs (1 or 2)
3. Compute `cali_delta = local_state - global_state`
4. Apply `calibrate_update(hist_delta, cali_delta)` → `tilde_delta`

### Calibration math (`src/unlearning/federaser.py:calibrate_update`)
```
tilde_delta = ||hist_delta||_2 * (cali_delta / ||cali_delta||_2)
```
- **Norm granularity**: GLOBAL (a single L2 norm computed over all floating-point parameters concatenated, excluding BN buffers). **Not per-layer.**
- Scale factor: `hist_norm / cali_norm`
- Zero-norm safeguard: if `cali_norm < 1e-8`, `scale_factor = 1.0`
- Post-calibration assertion: `|calibrated_norm - hist_norm| < rel_tol=1e-4` (via `math.isclose`)

### Delta_t handling
`hist_delta` contains the original client's delta from the attacked trajectory. `cali_delta` is the new retained-client delta from the reconstructed trajectory. These are **not** the same values: hist_delta is from an attack run including Client 0; cali_delta comes from the ongoing reconstruction without Client 0.

### Aggregation weights
FedEraser uses sample-count-weighted FedAvg over retained clients (C1–C9) only:
```python
sample_counts = [len(partitions[c]) for c in client_ids]   # clients 1–9
weights = [n_k / total_samples for n_k in sample_counts]
```
IID: each retained client has 5000 samples → weight = 5000/45000 ≈ 0.1111 each.

---

## (d) BatchNorm Buffer Handling

SmallCNN has two BatchNorm2d layers (`features.1` and `features.5`). Each has three state dict entries:
- `running_mean` — float32 tensor
- `running_var` — float32 tensor  
- `num_batches_tracked` — int64 scalar

### In delta computation (`src/training/fedavg.py:compute_model_delta`)
All keys are included in the delta (cast to float32 via `.float()`):
```python
delta[key] = local_state[key].float() - global_state[key].float()
```
So BN buffers ARE stored in `delta_client_*.pt` files.

### In calibration (`src/unlearning/federaser.py:calibrate_update`)
BN buffers **bypass** magnitude scaling:
```python
if tensor.is_floating_point() and "running_" not in key and "num_batches_tracked" not in key:
    calibrated_delta[key] = tensor * scale_factor  # weights/biases only
else:
    calibrated_delta[key] = tensor.clone()          # BN buffers: unscaled cali value
```
- `running_mean`, `running_var` → **unscaled calibration delta** (from local training, not rescaled)
- `num_batches_tracked` → **unscaled calibration delta**

### In FedEraser aggregation (`run_federaser_v5.py:aggregate_fedavg_modified`)
`num_batches_tracked` is **completely bypassed** — the global value is preserved unchanged:
```python
if "num_batches_tracked" in key:
    continue   # skip delta aggregation entirely
```
An assertion verifies this after each round (line 175–177). `running_mean` and `running_var` go through the weighted-average delta aggregation.

> [!IMPORTANT]
> This is a correction from earlier superseded runners: original FedEraser runs incorrectly applied the float scale factor to `num_batches_tracked` integer counters. The v5 runner preserves counters at their initial value for all 50 reconstruction rounds.

---

## (e) CA and ASR Computation

### Clean Accuracy (CA)
- Population: full 10,000-image CIFAR-10 test set
- Metric: fraction correctly classified
- Loader: `test_loader` from `get_cifar10_dataloaders` (ToTensor + Normalize, no augmentation)
- Implementation: `evaluate_extended` in `src/training/train_utils.py`

### Attack Success Rate (ASR)
- Population: 9,000 non-Airplane test images (Airplane = class 0 excluded)
- Trigger: **3×3 white patch in bottom-right corner** (default `white_patch` trigger type)
- Trigger value: normalized white pixel = `(1.0 - mean) / std` per channel (CIFAR-10 mean/std)
- Applied at positions `img[:, h-3:, w-3:]` (pixels 29–31, 29–31 for 32×32 images)
- Target class: **class 0 (Airplane)**
- Metric: fraction of triggered non-Airplane images predicted as class 0
- 100% of the 9,000 non-Airplane test images receive the trigger (poison_indices = all_indices in asr_dataset)
- Loader: `create_asr_dataloader` → `PoisonedDataset` wrapping the 9,000-image Subset

### Training-time poison
- During rounds 30–50, Client 0's partition gets `floor(10% × |C0|)` poisoned samples
- IID: 5000 × 0.1 = **500 poisoned images**; Non-IID: 3597 × 0.1 = **359 poisoned images**
- Poison candidates: non-Airplane images from C0's partition (filtered by `filter_non_target_indices`)
- Poison indices fixed once at seed `42 + 999 = 1041` via `torch.randperm`

---

## (f) Trajectory Directory Contents

Path: `{output_dir}/trajectory/round_{r:03d}/` for r in 1..50

Each round directory contains:
- `delta_client_00.pt` through `delta_client_09.pt` — 10 files, one per client
  - Format: Python dict mapping state_dict keys → float32 tensors (local_state - global_state)
  - Includes ALL keys: weights, biases, BN running_mean, BN running_var, BN num_batches_tracked (cast to float32)
  - Empty-partition clients save a zero-delta dict
- `global_model.pt` — the aggregated global state dict after FedAvg for that round

Additionally at the trajectory root:
- `initial_model.pt` — global model state before round 1
- `client_partitions.pt` — pickled list of 10 `torch.utils.data.Subset` objects (the actual dataset indices)
- `client_class_histograms.json` — per-client sample counts and class distributions

**Total delta files**: 50 rounds × 10 clients = **500 delta files** (matches README spec)

---

## (g) Model Architecture and Parameter Count

**SmallCNN** (`src/models/small_cnn.py`):

```
Input: 3 × 32 × 32
features.0: Conv2d(3→32, 3×3, padding=1)    →  864 weights + 32 bias = 896
features.1: BatchNorm2d(32)                   →  32 weight + 32 bias = 64 params
                                                 + running_mean(32), running_var(32), num_batches_tracked(1) [buffers]
features.2: ReLU
features.3: MaxPool2d(2×2) → 32 × 16 × 16
features.4: Conv2d(32→64, 3×3, padding=1)   →  18432 weights + 64 bias = 18496
features.5: BatchNorm2d(64)                   →  64 weight + 64 bias = 128 params
                                                 + running_mean(64), running_var(64), num_batches_tracked(1) [buffers]
features.6: ReLU
features.7: MaxPool2d(2×2) → 64 × 8 × 8
Flatten → 4096
classifier.0: Linear(4096→128)               →  524288 weights + 128 bias = 524416
classifier.1: ReLU
classifier.2: Dropout(p=0.5)
classifier.3: Linear(128→10)                 →  1280 weights + 10 bias = 1290
```

| Count | Value |
|---|---|
| **Trainable parameters** | **545,290** |
| BN buffers (non-trainable) | 194 (running_mean: 96, running_var: 96, num_batches_tracked: 2) |
| Total state_dict entries | 14 keys |

Verified by direct instantiation: `sum(p.numel() for p in SmallCNN().parameters()) = 545290`

---

## (h) Environment Deviations from README

| Item | README/docs claim | Actual environment |
|---|---|---|
| Python version | 3.11.6 (explicitly stated in README, developer_guide) | **3.12.9** (Anaconda) |
| PyTorch version | 2.14.0 | **2.9.0+cu128** |
| Torchvision | 0.29.0 | (not separately checked) |

> [!WARNING]
> README explicitly states: *"compatibility with other Python versions (e.g., 3.10 or 3.12+) or other PyTorch versions is explicitly unverified"*. We are running on Python 3.12.9 and PyTorch 2.9.0, both outside the verified envelope.

> [!NOTE]
> PyTorch 2.9.0 is a **higher** version than the documented 2.14.0 (which was the latest stable at time of writing), not a downgrade. The CUDA version is cu128. The core APIs used (`torch.nn`, `torch.optim.SGD`, `torch.save/load`) are stable across this version range, but deterministic reproducibility cannot be guaranteed.

---

## Summary Table

| Audit item | Key finding |
|---|---|
| Local optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=0.0, batch=64, E=2 epochs |
| K (steps/round) | 10 clients × full participation; ~158 steps/client/round (IID) |
| Retained rounds | All 50 (range(1,51)), no subsampling |
| FedEraser norm | **Global** L2 norm across all float params (not per-layer) |
| BN running_mean/var in calibration | Unscaled calibration value (not norm-rescaled) |
| num_batches_tracked | Completely bypassed in aggregation; held at initial value |
| CA metric | Full 10k test set, fraction correct |
| ASR metric | 9k non-Airplane test images + white 3×3 patch → fraction predicted as class 0 |
| Trajectory contents | 500 delta .pt files + 50 global_model.pt + partitions + histograms |
| Model params | 545,290 trainable |
| Python deviation | 3.12.9 vs documented 3.11.6 |
| PyTorch deviation | 2.9.0+cu128 vs documented 2.14.0 |
