"""Registered L0 flat-term gates from trigger-free detected onset only."""

import argparse
import json
from pathlib import Path

import torch

from src.models.small_cnn import SmallCNN
from src.unlearning.onsetfree.l0 import apply_update, contrasts_and_rows, project_out, right_basis
from src.unlearning.onsetfree.onset import flat_gate_weights
from src.unlearning.onsetfree.traj_io import Trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--detector-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--malicious-client", type=int, default=0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--kappa", default="0.5")
    parser.add_argument("--shifts", type=int, nargs="+", default=[0, 10, 20], help="Prespecified shifts; +20 is deliberately after the evaluation-only true onset")
    args = parser.parse_args()
    detector = json.loads(Path(args.detector_json).read_text())
    estimated = detector["analyses"][args.kappa]["estimated_onset"]
    if estimated is None:
        raise ValueError("Detector had no onset alarm; do not invent a gate")
    counts_raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in counts_raw.items()}
    keys = sorted(dict(SmallCNN().named_parameters()))
    state, contrasts, benign_rows, _, _ = contrasts_and_rows(Trajectory(args.source_run, counts), args.malicious_client, keys)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = {}
    for shift in args.shifts:
        for alpha in (0.0, 0.5, 1.0):
            for margin in (0, 2, 5):
                name = f"gate_shift{shift}_alpha{alpha:g}_margin{margin}"
                variants[name] = {"shift": shift, "alpha": alpha, "margin": margin, "update": {}}
    for key in keys:
        basis = right_basis(benign_rows[key], args.rank, seed=42)
        flat_by_round = [pm * project_out(e, basis) for pm, e in contrasts[key]]
        rounds = list(range(1, len(flat_by_round) + 1))
        for name, variant in variants.items():
            weights = flat_gate_weights(rounds, estimated + variant["shift"], variant["alpha"], variant["margin"])
            variant["update"][key] = -sum((weight * flat for weight, flat in zip(weights, flat_by_round)), torch.zeros_like(flat_by_round[0]))
    for name, variant in variants.items():
        torch.save(apply_update(state, variant["update"]), output / f"{name}.pt")
        del variant["update"]
    (output / "metadata.json").write_text(json.dumps({
        "source_run": str(Path(args.source_run).resolve()),
        "detector_json": str(Path(args.detector_json).resolve()),
        "kappa": args.kappa,
        "estimated_onset": estimated,
        "rank": args.rank,
        "shifts": args.shifts,
        "variants": variants,
        "selection_status": "not selected; gating is an optional ablation",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
