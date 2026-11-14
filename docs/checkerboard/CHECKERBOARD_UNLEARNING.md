# Non-IID attack and unlearning on the new checkerboard models

## Confirmed IID source

The completed IID run found in this checkout is:

```text
results/checkerboard/iid_20261024_015516_rerun_001
```

Its saved final results report **75.26% clean accuracy and 92.1778% checkerboard ASR**. All 500 historical client updates and the serialized partitions are present. The initial `iid_20261024_015516` folder only contains configuration from an earlier attempt; use the completed `_rerun_001` source.

## 1. Train only the Non-IID attacked model

From the project root, activate the same environment that trained IID. In PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$IidAttack = 'results/checkerboard/iid_20261024_015516_rerun_001'
$CheckerRunId = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$NoniidAttack = "results/checkerboard/noniid_$CheckerRunId"
python scripts/run_experiment_v2.py --config configs/v3_checkerboard4_noniid.yaml --mode attack --log_trajectory --output_dir "$NoniidAttack"
```

Wait for successful completion before running Non-IID unlearning. Inspect `$NoniidAttack/attack_report.md` and confirm its final CA and ASR. `--log_trajectory` is required for later subtraction/FedEraser. If the runner prints an automatically adjusted `_rerun_...` output path, update `$NoniidAttack` to that actual path before continuing. If you open a new terminal, set both attack-folder variables again to the completed folders.

The Non-IID command changes partitioning to Dirichlet alpha=0.5 while retaining the same 4x4 bottom-right checkerboard, 10% local poisoning, seed 42, and attack rounds 30-50. It does not train M_clean, M_CF, or any unlearning method.

## 2. Use the variant-specific unlearning launcher

Use `scripts/run_checkerboard_unlearning.py` for these new runs. Do not launch the old hardcoded white-patch subtraction/KD/FedEraser scripts: those still point to original result folders or use the original evaluation trigger.

The new launcher reads the chosen source's trigger, partition, round count and sample counts, checks required historical updates, and evaluates the resulting endpoint with **the same checkerboard**. It supports both IID and Non-IID without manually editing paths or code. It does not launch any clean/reference retraining or restore cost-accounting tooling.

You can check source history and metadata without training or loading tensors:

```powershell
python scripts/run_checkerboard_unlearning.py --source_run "$IidAttack" --method subtraction --check_only
python scripts/run_checkerboard_unlearning.py --source_run "$IidAttack" --method federaser --check_only
python scripts/run_checkerboard_unlearning.py --source_run "$NoniidAttack" --method subtraction --check_only
python scripts/run_checkerboard_unlearning.py --source_run "$NoniidAttack" --method federaser --check_only
```

The checker verifies file presence and provenance. `--check_only` is not a dataset/model numerical validation or a guarantee that all tensors deserialize correctly.

## 3. Prepare CIFAR-100 once for KD

Subtraction and FedEraser need CIFAR-10. KD additionally needs CIFAR-100 training images. The CIFAR-100 train file was not present in the inspected checkout. If missing in your experiment environment, download it once:

```powershell
python -c "from torchvision.datasets import CIFAR100; CIFAR100(root='data', train=True, download=True)"
```

This new launcher uses the standard Torchvision loader without disabling integrity or HTTPS checks. If your existing CIFAR-100 files were reformatted by historical helper scripts and fail integrity verification, prepare the standard dataset rather than silently bypassing the check.

KD deterministically selects 50 training images per CIFAR-100 class, totaling 5,000, using seed 42 for both IID and Non-IID. The exact indices are stored as `proxy_indices.json` in each KD output. For an explicitly shared list, add `--proxy_indices <existing-KD-output>/proxy_indices.json`; it is checked for uniqueness, valid range, and 50 images per class.

## 4. Run the three methods for IID and Non-IID

Choose fresh method output folders for this batch:

```powershell
$UnlearningRunId = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$IidUnlearn = "$IidAttack/unlearning/$UnlearningRunId"
$NoniidUnlearn = "$NoniidAttack/unlearning/$UnlearningRunId"
```

### IID

```powershell
python scripts/run_checkerboard_unlearning.py --source_run "$IidAttack" --method subtraction --output_dir "$IidUnlearn/subtraction"
python scripts/run_checkerboard_unlearning.py --source_run "$IidAttack" --method kd --subtraction_dir "$IidUnlearn/subtraction" --output_dir "$IidUnlearn/kd"
python scripts/run_checkerboard_unlearning.py --source_run "$IidAttack" --method federaser --calibration_epochs 2 --output_dir "$IidUnlearn/federaser_Ecal2"
```

### Non-IID

```powershell
python scripts/run_checkerboard_unlearning.py --source_run "$NoniidAttack" --method subtraction --output_dir "$NoniidUnlearn/subtraction"
python scripts/run_checkerboard_unlearning.py --source_run "$NoniidAttack" --method kd --subtraction_dir "$NoniidUnlearn/subtraction" --output_dir "$NoniidUnlearn/kd"
python scripts/run_checkerboard_unlearning.py --source_run "$NoniidAttack" --method federaser --calibration_epochs 2 --output_dir "$NoniidUnlearn/federaser_Ecal2"
```

Run commands individually and proceed only after successful completion. KD requires that partition's subtraction checkpoint. FedEraser is independent of subtraction and KD and can run immediately after its source attacked run is complete.

The commands use two calibration epochs, matching the corrected Ecal=2 baseline. To test one epoch as well, use the same command with `--calibration_epochs 1` and a separate `federaser_Ecal1` output folder.

If an output folder already contains files, the launcher preserves it and selects a numbered sibling. Watch the printed `Output:` path. If subtraction is redirected, point the following KD command's `--subtraction_dir` at that actual folder. KD verifies the subtraction's source attack hash and its recorded output checkpoint hash, preventing accidental use of old white-patch or opposite-partition subtraction weights.

## Methods retained

- **Subtraction:** subtract C0's weighted historical updates over every source round, including its pre-attack rounds. Weight is derived from the source's client sample counts. All floating-point state is subtracted; integer counters stay unchanged.
- **KD:** teacher is this source's attacked model; student starts from this source's subtracted model. Five epochs, temperature 4, T-squared KL divergence, SGD LR 0.001, momentum 0.9, batch size 64, no weight decay. Student Dropout remains active; student BatchNorm running statistics/counters are frozen and checked. CIFAR-100 labels only select the balanced proxy, not loss targets. The exact new-run proxy indices and raw uint8 NHWC pixel hash are recorded.
- **FedEraser:** reconstruct from the matching initial state using Clients 1-9. Round 1 reuses retained historical updates; subsequent rounds calibrate new retained-client directions against historical global weight/bias L2 norms. BatchNorm running statistics bypass magnitude scaling; integer trackers remain at their initial values. Client sample weights are renormalized over retained clients. Saved partition indices are checked against seed-based regeneration. The existing reusable FedEraser calibration math is used.

These are variant-aware implementations of the existing method definitions, not a promise of bit-identical replay of historical white-patch runs. Device/backend behavior and the new attacked trajectory can change results.

## Optional: evaluate existing M_CF without retraining it

The main commands above evaluate CA/ASR only and do not require a reference. If you also want counterfactual prediction agreement and the existing reference's new-trigger CA/ASR, add:

```powershell
# Add this option to an IID method command:
--reference_model results/phase2/iid_counterfactual/final_model.pt

# Add this option to a Non-IID method command:
--reference_model results/phase2/noniid_counterfactual/final_model.pt
```

Those are command options, not standalone PowerShell commands. Matching training config and initial weights are checked before optimization. Reference weights are loaded solely for evaluation; they are not teachers or optimization targets. Reference ASR is recomputed with the checkerboard instead of copying the old white-patch number.

## Results and provenance

Each method writes only to its own output folder:

| File | Contents |
|---|---|
| `final_model.pt` | Unlearned endpoint weights |
| `run_config.json` | Source config/path/hash, method parameters, output checkpoint hash, and started/model-saved/completed status |
| `evaluation.json` | Clean accuracy, checkerboard ASR, untriggered target rate, class metrics, and optional reference comparison |
| `results.csv` | Source attack versus new method CA/ASR; optional re-evaluated reference row |
| `report.md` | Human-readable CA/ASR comparison for this method |
| `proxy_indices.json` | KD-only exact balanced CIFAR-100 subset |

CA uses 10,000 clean CIFAR-10 test images. ASR uses the 9,000 triggered non-Airplane images predicted as Airplane. The saved final model is evaluated directly; no test-selected intermediate checkpoint is substituted. All percentages use the same new checkerboard protocol within a comparison.

Original white-patch models, tables and launchers remain intact. No unlearning result numbers are filled in until the methods actually execute. The implementation was checked with dependency-free provenance/subtraction/report tests and source-history preflight on the completed IID run. Full tensor/training checks require the experiment environment with Torch; no unlearning runs were launched while preparing these commands.

```powershell
python -m unittest discover -s tests -p test_checkerboard_unlearning.py -v
```
