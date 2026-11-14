# Extreme label skew: Dirichlet alpha = 0.01

Run from the repository root in the existing PyTorch training environment:

```powershell
python scripts/run_label_skew_stress.py
```

The default seed is 42. The launcher trains a fresh attacked model with full historical update logging, then runs subtraction, KD and corrected FedEraser (two calibration epochs) on that exact trajectory. The configuration is configs/v4_checkerboard4_extreme_noniid.yaml. It retains the current 4x4 checkerboard, ten clients, 50 rounds, two local epochs, attack rounds 30-50, and 10% local poisoning. Only Dirichlet alpha changes from 0.5 to 0.01 relative to the existing non-IID checkerboard setup.

Inspect the planned commands without dependencies, data or training:

```powershell
python scripts/run_label_skew_stress.py --plan
```

Validate the datasets and measure the actual partition before committing to training:

```powershell
python scripts/run_label_skew_stress.py --preflight_only
```

For sensitivity across independently sampled partitions, specify multiple seeds:

```powershell
python scripts/run_label_skew_stress.py --seeds 42 43 44
```

All requested partitions are checked before training starts. A seed may produce no poisonable examples in Client 0, or fewer than required by the configured poisoning rate. Such a run stops with an explanation and partition reports; the launcher does not cap the poison count, move examples, change the target, or search for a favorable seed. Empty retained clients contribute zero weight; the training runner logs zero deltas and FedEraser skips their calibration loaders. The original Dirichlet sampling algorithm is unchanged, preserving existing partition reproduction.

CIFAR-10 training/test and CIFAR-100 training datasets must be installed under data/ with standard Torchvision integrity checks passing. No dataset downloads occur. KD uses its existing 5,000-image balanced CIFAR-100 proxy and original hyperparameters. A fresh trajectory is required: applying new alpha metadata to saved alpha=0.5 models would not test extreme skew.

Each invocation creates results/label_skew/alpha001_<timestamp>/seed_<seed>/ with:

- config.yaml and partition_report.json: actual client sizes, class counts, dominant-class fractions, entropy, empty clients, C0 aggregation weight, poison count, and classes lost when C0 is removed.
- attack/: new attacked checkpoint and complete trajectory; attack/unlearning/ holds all three method outputs, source hashes and per-class metrics.
- attack.log, subtraction.log, kd.log, federaser.log and status.json: progress and failure details. Tail the current log to follow training. Stages stop on failure.
- comparison.csv and report.md: source and method CA, ASR, untriggered target rate, macro F1 and changes from the attacked endpoint. These appear after all stages succeed.

Use --output_dir to choose a new, empty output folder. Preflight-only output is diagnostic and cannot be resumed as a training run; use a fresh directory for training. There is no automatic resume. Multi-seed runs produce one comparison per seed; they do not aggregate results or silently drop invalid seeds.

Interpret utility and ASR together. Low ASR with collapsed accuracy is not useful unlearning. Extreme Dirichlet partitioning also changes sample sizes and client aggregation weights; inspect C0's weight and per-class recall. If C0 owns nearly all examples of a class, removing it can reduce retained-client class coverage. Compare attacked ASR with untriggered target rate to establish that the source learned a backdoor. No matching alpha=0.01 counterfactual reference is trained here, so prediction agreement with retraining is not measured. Existing alpha=0.5 references are incompatible.

Validation in this checkout: dependency-free launcher and checkerboard tests passed; Torch-dependent tests and full training could not run because the available Python lacks Torch, Torchvision, NumPy and PyYAML. The C: volume was also full. No experimental metric values have been produced for alpha=0.01.
