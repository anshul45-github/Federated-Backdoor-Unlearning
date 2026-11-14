# Bottom-right 4x4 checkerboard attack pilot

## Question and scope

Does replacing the old 3x3 white trigger with a 4x4 black/white checkerboard produce a higher final attack success rate while preserving clean accuracy?

This is a candidate stronger attack setting, not a measured improvement yet. Only `M_attack` is trained in the commands below. Clean and counterfactual model weights do not need retraining for this trigger change. Their old white-patch ASR values are not checkerboard baseline ASRs: re-evaluate those same checkpoints with the new trigger if a matched reference comparison is needed later.

## Exact change

| Setting | Old baseline | New pilot |
|---|---|---|
| Pattern | Solid white | Alternating white/black checkerboard |
| Size | 3x3 | 4x4 |
| Position | Bottom right | Bottom right |
| Raw colors | RGB white = 1 | RGB white = 1, RGB black = 0 |
| Target | Airplane, class 0 | Same |
| Poisoned client | Client 0 | Same |
| Poison ratio | 10% of C0 local data | Same |
| Attack rounds | 30-50 inclusive | Same |
| Training | 10 clients, full participation, 50 rounds, two local epochs | Same |
| Optimizer | SGD, LR 0.01, momentum 0.9, zero weight decay | Same |
| Seed / partitions | 42; IID or Dirichlet alpha=0.5 | Same |
| Update amplification | None in the current runner | None |

The patch is:

```text
W B W B
B W B W
W B W B
B W B W
```

Every RGB channel receives the same pattern. Both white and black values are converted with the existing CIFAR-10 normalization. The patch is applied after training augmentation so it remains exactly in the bottom-right corner. Only the fixed selected non-target-class examples in C0 are poisoned and relabeled. No other attack parameter is increased, so the pilot isolates the trigger change rather than confounding it with stronger poisoning/scaling.

Configurations:

- `configs/v3_checkerboard4_iid.yaml`
- `configs/v3_checkerboard4_noniid.yaml`

Existing v2 configs and callers without `trigger_type` still use the original white patch. The training runner passes the selected pattern into both local poisoning and the triggered test loader.

## Run M_attack

From the project root, activate your existing experiment environment and set the import path in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
```

Use whichever environment already runs your baseline if it is not `.venv`. If dependencies are missing, install the existing `requirements.txt` in that environment. The runner requires extracted CIFAR-10 at `data/cifar-10-batches-py`; it does not download it. CIFAR-100 is not needed for this pilot.

### IID pilot

```powershell
$CheckerRunId = Get-Date -Format 'yyyyMMdd_HHmmss'
$CheckerIidOutput = "results/checkerboard/iid_$CheckerRunId"
python scripts/run_experiment_v2.py --config configs/v3_checkerboard4_iid.yaml --mode attack --log_trajectory --output_dir "$CheckerIidOutput"
```

This runs only attacked training. It still executes the complete 50-round attacked branch, including the 29 pre-attack rounds needed to construct its model. It does not launch a separate clean model, counterfactual reference, subtraction, KD, or FedEraser.

### Optional Non-IID pilot

If Non-IID is the regime you want to assess, run this instead of, or after, the IID pilot:

```powershell
$CheckerRunId = Get-Date -Format 'yyyyMMdd_HHmmss'
$CheckerNoniidOutput = "results/checkerboard/noniid_$CheckerRunId"
python scripts/run_experiment_v2.py --config configs/v3_checkerboard4_noniid.yaml --mode attack --log_trajectory --output_dir "$CheckerNoniidOutput"
```

`--log_trajectory` saves 500 client-update files plus 50 round checkpoints per setting, as well as partitions and endpoint checkpoints. This retains the information needed to attempt historical-update unlearning later without first repeating attacked training. If you only want the endpoint pilot and wish to avoid trajectory storage, omit the flag; historical subtraction/FedEraser would then need a new logged attacked run before proceeding.

Every invocation uses a **new or empty output folder**. If the requested checkerboard folder already contains files, the runner preserves it and automatically selects a numbered sibling such as `iid_20261024_015516_rerun_001`. It prints the actual output path before training; inspect that path rather than assuming the original PowerShell variable still names the selected folder. A path pointing to a file is rejected. Legacy white-patch runs continue to reject nonempty output folders. This starts a new run; no resume behavior is implied.

## Inspect CA and ASR

The training console prints clean accuracy and triggered target rate for every round. After successful completion, inspect the report in that run's folder:

```powershell
Get-Content -LiteralPath "$CheckerIidOutput/attack_report.md"
Get-Content -LiteralPath "$CheckerIidOutput/final_results.json"
```

For a Non-IID run, use `$CheckerNoniidOutput` instead.

| New artifact | Meaning |
|---|---|
| `attack_report.md` | This run's final CA/ASR, untriggered false-target rate, exact protocol, poison count, and diagnostic peak attack-period ASR |
| `final_results.json` | Machine-readable endpoint metrics and config; `clean_accuracy` is CA and `triggered_rate` is ASR, both percentages |
| `metrics_per_round.csv` | CA, ASR, untriggered false-target rate, and attack activation over all 50 rounds |
| `config.json` | Exact configuration including checkerboard type and size |
| `initial_model.pt`, `final_model.pt` | Starting and final attacked-branch weights |
| `client_class_histograms.json` | Client sizes and label counts |
| `client_partitions.pt`, `trajectory/` | Exact partition objects and historical updates for possible later unlearning; trajectories require the flag |

CA uses all 10,000 clean test images. ASR uses the 9,000 non-Airplane test images with this checkerboard applied; success means prediction as Airplane. The untriggered false-target rate uses those same images without a trigger.

Report the final-round checkpoint as the endpoint. A temporary peak ASR is shown separately and is not substituted for final ASR. Decide whether the result is reasonable by examining both trigger success and preserved clean utility; a high ASR paired with a major CA loss is a different outcome from a successful hidden backdoor. No acceptance threshold or successful result is invented before the run.

For context, the saved old white-patch attacked endpoints are:

| Regime | Old CA | Old white-patch ASR |
|---|---:|---:|
| IID | 75.01% | 56.07% |
| Non-IID | 69.80% | 32.21% |

Compare the new run within the same regime. The new report is written only inside its own output directory; the original `results/phase1/iid_attack`, `results/phase1/noniid_attack`, Phase 2 folders, baseline CSVs, and presentations are not overwritten. New result values remain unmeasured until you run the pilot.

## Before later unlearning

The existing white-patch subtraction/KD/FedEraser launchers and some evaluators hardcode old run paths or a 3x3 white trigger. Use the variant-specific `scripts/run_checkerboard_unlearning.py` instead. See [CHECKERBOARD_UNLEARNING.md](CHECKERBOARD_UNLEARNING.md) for the Non-IID attack command, matching IID/Non-IID subtraction/KD/FedEraser commands, and optional evaluation of existing reference weights without retraining.

## Verification status

```powershell
python -m unittest discover -s tests -p test_checkerboard_attack.py -v
python -m pytest tests/test_attack.py
```

The new tests cover the normalized alternating pattern, bottom-right-only changes, unpoisoned samples, matching training/ASR triggers, legacy default behavior, C0/start-round activation, output overwrite protection, and report endpoint/peak separation. Tensor/training-loader tests require Torch and related experiment dependencies; they explicitly skip in the local dependency-free environment. No 50-round pilot or ASR improvement has been claimed without executing it.
