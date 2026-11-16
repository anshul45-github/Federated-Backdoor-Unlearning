"""Check round completeness and reconstruct FedAvg for a saved run."""

import argparse
import json
from pathlib import Path

import torch

from src.unlearning.onsetfree.traj_io import Trajectory, reconstruction_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    counts_raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in counts_raw.items()}
    if len(counts) != 10:
        raise ValueError(f"Expected 10 client counts, got {len(counts)}")
    trajectory = Trajectory(args.run_dir, counts)
    expected_rounds = list(range(1, 51))
    observed_rounds = [int(path.name.split("_")[-1]) for path in trajectory.round_dirs]
    if observed_rounds != expected_rounds:
        raise ValueError(f"Expected rounds 1-50, got {observed_rounds}")
    max_float_error = 0.0
    max_integer_error = 0.0
    for record in trajectory:
        errors = reconstruction_error(record)
        for key, error in errors.items():
            if record.theta_t[key].is_floating_point():
                max_float_error = max(max_float_error, error)
            else:
                max_integer_error = max(max_integer_error, error)
    last = record.theta_next
    endpoint = torch.load(Path(args.run_dir) / "final_model.pt", map_location="cpu", weights_only=True)
    max_endpoint_error = max((last[key].float() - endpoint[key].float()).abs().max().item() for key in last)
    result = {
        "source_run": str(Path(args.run_dir).resolve()),
        "rounds": len(observed_rounds),
        "client_deltas": len(observed_rounds) * len(counts),
        "max_float_reconstruction_error": max_float_error,
        "max_integer_reconstruction_error": max_integer_error,
        "max_endpoint_error": max_endpoint_error,
        "atol": args.atol,
        "passed": max(max_float_error, max_integer_error, max_endpoint_error) <= args.atol,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(f"Trajectory reconstruction failed: {result}")


if __name__ == "__main__":
    main()
