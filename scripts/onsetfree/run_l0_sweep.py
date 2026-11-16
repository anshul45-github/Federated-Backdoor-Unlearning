"""Registered L0 rank/scale sweep and three random-subspace controls.

All choices are generated without attack metadata, validation data, or ASR.
Selection must happen later on a held-out benign validation split.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.models.small_cnn import SmallCNN
from src.unlearning.onsetfree.l0 import (
    apply_update, contrasts_and_rows, project_out, random_basis, right_basis,
)
from src.unlearning.onsetfree.traj_io import Trajectory


RANKS = (4, 8, 16, 32, 64)
SCALES = (0.5, 1.0)
RANDOM_SEEDS = (42, 43, 44)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--malicious-client", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts_raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in counts_raw.items()}
    parameter_keys = sorted(dict(SmallCNN().named_parameters()))
    state, contrasts, benign_rows, malicious_rows, attacker_sum = contrasts_and_rows(
        Trajectory(args.source_run, counts), args.malicious_client, parameter_keys
    )
    total_by_key = {
        key: sum((pm * e for pm, e in rounds), torch.zeros_like(rounds[0][1]))
        for key, rounds in contrasts.items()
    }
    diagnostics = {}
    for rank in RANKS:
        updates = {name: {} for name in ("l0", "c4_malicious", *(f"c3_random_seed{seed}" for seed in RANDOM_SEEDS))}
        diagnostics[str(rank)] = {}
        for key in parameter_keys:
            total = total_by_key[key]
            benign_basis = right_basis(benign_rows[key], rank, seed=42)
            malicious_basis = right_basis(malicious_rows[key], rank, seed=42)
            updates["l0"][key] = -project_out(total, benign_basis)
            updates["c4_malicious"][key] = -project_out(total, malicious_basis)
            inside = total - project_out(total, benign_basis)
            denominator = torch.dot(total, total).item()
            diagnostics[str(rank)][key] = torch.dot(inside, inside).item() / denominator if denominator else 0.0
            for seed in RANDOM_SEEDS:
                basis = random_basis(total.numel(), rank, seed=seed)
                updates[f"c3_random_seed{seed}"][key] = -project_out(total, basis)
        for scale in SCALES:
            for name, update in updates.items():
                torch.save(apply_update(state, update, scale), output / f"{name}_k{rank}_s{scale:g}.pt")
    for scale in SCALES:
        torch.save(apply_update(state, {key: -value for key, value in attacker_sum.items()}, scale), output / f"c1_s{scale:g}.pt")
        torch.save(apply_update(state, {key: -value for key, value in total_by_key.items()}, scale), output / f"c2_s{scale:g}.pt")
    metadata = {
        "source_run": str(Path(args.source_run).resolve()),
        "malicious_client": args.malicious_client,
        "ranks": RANKS,
        "scales": SCALES,
        "random_seeds": RANDOM_SEEDS,
        "bn_policy": "keep final attack running statistics and counters",
        "benign_subspace_inside_energy_fraction": diagnostics,
        "client_flops": 0,
        "server_wall_seconds": time.perf_counter() - started,
        "selection_status": "not selected; never use test ASR to choose a rank or scale",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
