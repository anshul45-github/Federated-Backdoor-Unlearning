# Experiment commands

Run commands from the project root with the project dependencies installed and the project root on Python's import path. For PowerShell, set `$env:PYTHONPATH = '.'`. For Bash, use `PYTHONPATH=. python …`. The two baseline Bash launchers expect `.venv/bin/python`.

| Task | Entry point | Inputs and outputs |
|---|---|---|
| Federated training | `python scripts/run_experiment_v2.py --help` | Takes an experiment config and output directory. Saves checkpoints, metrics and optional trajectories. |
| Checkerboard unlearning | [Commands](../docs/checkerboard/CHECKERBOARD_UNLEARNING.md) | Takes a saved checkerboard run and method. Saves evaluation and model artifacts. |
| Component patching | [Commands](../docs/checkerboard/COMPONENT_PATCHING.md) | Uses a saved run and donor checkpoints. Saves rankings, metrics and reports. |
| Label-skew comparison | `python scripts/run_label_skew_stress.py --plan` | Prints the commands. A full run saves outputs under `results/label_skew/`. |
| Phase 1 baselines and Phase 2 unlearning | [Baseline scripts](baseline/) and [developer guide](../docs/guides/developer_guide.md) | Uses configs, local CIFAR data, and, for some methods, saved trajectories and exact partitions. Writes to the Phase 1 or Phase 2 results directories. |
| Evaluate saved models | [Evaluation scripts](evaluation/) | Uses saved checkpoints and local CIFAR data. Writes metric JSON to saved run folders. `eval_iid_clean.py` loads the IID clean checkpoint. |
| Tables and figures | [Reporting scripts](reporting/) | Reads saved metrics, checkpoints, or data depending on the script. Several scripts overwrite existing CSV, Markdown or PNG outputs. |
| Verification | [Verification scripts](verification/) | Checks inputs, saved runs, or checkpoint metadata. Some checks require local partition or trajectory files. |
| Local CIFAR-100 loader check | `python scripts/data/check_cifar100.py` | Reads the locally extracted CIFAR-100 dataset. |
| Data and proxy utilities | [Data scripts](data/) | `rewrite_cifar100_train_pickle.py` rewrites the local training pickle in place. `make_meta.py` writes dataset metadata. The proxy utilities can write proxy indices and figures. |

The [run index](../results/README.md) records saved outputs and local trajectory locations. The archive includes the earlier Stage 5–7 scripts, [Phase 2 FedEraser variants](_archive/phase2_federaser/) and [Phase 2 audits and repairs](_archive/phase2_audits/). Some historical commands depend on old inputs or write to saved outputs. The archived `final_cleanup.py` also contains its original malformed checksum output path.
