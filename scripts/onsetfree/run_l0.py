"""Generate trigger-free L0 and control checkpoints from a stored trajectory.

Example:
python -m scripts.onsetfree.run_l0 --run-dir results/checkerboard/iid_... \
    --counts-json results/checkerboard/iid_.../client_class_histograms.json \
    --output-dir results/onsetfree/l0/checkerboard_iid
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.models.small_cnn import SmallCNN
from src.unlearning.onsetfree.l0 import apply_update, contrasts_and_rows, make_updates
from src.unlearning.onsetfree.traj_io import Trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--malicious-client", type=int, default=0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.perf_counter()
    counts_raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in counts_raw.items()}
    trajectory = Trajectory(args.run_dir, counts)
    parameter_keys = set(dict(SmallCNN().named_parameters()))
    final_state, contrasts, benign_rows, malicious_rows, attacker_sum = contrasts_and_rows(
        trajectory, args.malicious_client, parameter_keys
    )
    updates, inside_energy = make_updates(contrasts, benign_rows, malicious_rows, attacker_sum, args.rank, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, update in updates.items():
        torch.save(apply_update(final_state, update, args.scale), output / f"{name}.pt")
    metadata = {
        "source_run": str(Path(args.run_dir).resolve()),
        "counts_json": str(Path(args.counts_json).resolve()),
        "malicious_client": args.malicious_client,
        "rank": args.rank,
        "scale": args.scale,
        "seed": args.seed,
        "bn_policy": "keep final attack running statistics and counters",
        "client_flops": 0,
        "server_wall_seconds": time.perf_counter() - started,
        "inside_benign_subspace_energy_fraction": inside_energy,
        "checkpoints": sorted(updates),
        "metrics_status": "not evaluated",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
