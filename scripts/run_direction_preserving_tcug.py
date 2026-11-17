"""Sweep aligned Client 0 update retention on a saved checkerboard endpoint."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from scripts.run_checkerboard_unlearning import evaluate, sha256, validate_inputs
from src.models.small_cnn import SmallCNN
from src.utils.attack_report import prepare_output_directory


LAMBDAS = (0.0, 0.25, 0.5, 1.0)


def aligned_component(client, trusted, cap_ratio, epsilon):
    trusted_sq = torch.dot(trusted, trusted).item()
    dot = torch.dot(client, trusted).item()
    coefficient = max(0.0, dot) / (trusted_sq + epsilon)
    scale = min(coefficient, cap_ratio) if cap_ratio is not None else coefficient
    return trusted * scale, {
        "dot": dot, "trusted_norm": math.sqrt(trusted_sq),
        "raw_coefficient": coefficient, "applied_coefficient": scale,
        "cap_activated": scale < coefficient,
    }


def accumulate(trajectory, parameters, counts, cap_ratio, epsilon):
    names = tuple(parameters)
    removed = torch.zeros(sum(p.numel() for p in parameters.values()), dtype=torch.float64)
    retained = torch.zeros_like(removed)
    client_weight = counts[0] / sum(counts)
    trusted_count = sum(counts[1:])
    history = []
    for round_idx, paths in trajectory.items():
        trusted = torch.zeros_like(removed)
        for client_id, (path, count) in enumerate(zip(paths, counts)):
            if not count:
                continue
            update = torch.load(path, map_location="cpu", weights_only=True)
            if any(update[key].shape != parameters[key].shape or not update[key].is_floating_point()
                   for key in names):
                raise ValueError(f"Invalid parameter delta in {path}")
            vector = torch.cat([update[key].reshape(-1).double() for key in names])
            if not torch.isfinite(vector).all():
                raise ValueError(f"Non-finite parameter delta in {path}")
            if client_id == 0:
                client = vector
            else:
                trusted.add_(vector, alpha=count / trusted_count)
        aligned, projection = aligned_component(client, trusted, cap_ratio, epsilon)
        removed.add_(client, alpha=client_weight)
        retained.add_(aligned, alpha=client_weight)
        history.append({"round": round_idx, **projection})
    return removed, retained, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    cap = parser.add_mutually_exclusive_group()
    cap.add_argument("--cap_ratio", type=float, default=1.0,
                     help="Cap the projection norm at this multiple of the trusted norm")
    cap.add_argument("--no_cap", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.set_defaults(method="direction_preserving_tcug", subtraction_dir=None, reference_model=None)
    args = parser.parse_args()
    if not math.isfinite(args.cap_ratio) or args.cap_ratio < 0:
        parser.error("--cap_ratio must be finite and nonnegative")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("--epsilon must be finite and positive")
    cap_ratio = None if args.no_cap else args.cap_ratio
    source, config, _, counts, source_hash, _ = validate_inputs(args)
    trajectory = {r: [source / "trajectory" / f"round_{r:03d}" / f"delta_client_{c:02d}.pt"
                      for c in range(len(counts))] for r in range(1, config["num_rounds"] + 1)}
    for paths in trajectory.values():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    attacked = torch.load(source / "final_model.pt", map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    removed, retained, history = accumulate(trajectory, parameters, counts, cap_ratio, args.epsilon)
    output = Path(prepare_output_directory(args.output_dir, unique_on_conflict=True))
    model.load_state_dict(attacked)
    evaluations = {"attacked": evaluate(torch, model, config, args, device)}
    output_hashes = {}
    sizes = [p.numel() for p in parameters.values()]
    for value in LAMBDAS:
        label = f"lambda_{value:g}"
        repaired = attacked.copy()
        correction = value * retained - removed
        for key, delta in zip(parameters, correction.split(sizes)):
            original = attacked[key]
            repaired[key] = (original.double() + delta.reshape_as(original)).to(original.dtype)
        model.load_state_dict(repaired)
        evaluations[label] = evaluate(torch, model, config, args, device)
        checkpoint = output / f"{label}.pt"
        torch.save(repaired, checkpoint)
        output_hashes[label] = sha256(checkpoint)
        print(f"{label}: CA {evaluations[label]['clean_accuracy']:.2f}%, "
              f"ASR {evaluations[label]['attack_success_rate']:.2f}%", flush=True)

    metadata = {
        "method": args.method, "source_run": str(source), "source_config": config,
        "source_model_sha256": source_hash, "checkpoint_sha256": output_hashes,
        "rounds": list(trajectory), "lambdas": LAMBDAS, "client_sample_counts": counts,
        "client_0_fedavg_weight": counts[0] / sum(counts), "cap_ratio": cap_ratio,
        "epsilon": args.epsilon, "evaluation_device": str(device), "torch_version": torch.__version__,
        "state_policy": "trainable parameters corrected; buffers copied from attacked model",
    }
    for name, record in (("run_config", metadata), ("projection_history", history), ("evaluation", evaluations)):
        (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    fields = ("clean_accuracy", "attack_success_rate", "untriggered_target_rate", "macro_f1_pct")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", *fields))
        for label, metrics in evaluations.items():
            writer.writerow((label, *(metrics[field] for field in fields)))


if __name__ == "__main__":
    main()
