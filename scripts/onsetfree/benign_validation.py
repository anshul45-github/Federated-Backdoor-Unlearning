"""Evaluation-only benign validation loss, accuracy and Hessian quadratic.

The validation sample excludes the L1 curvature sample when its metadata is
provided. It never uses attack trigger, target label or test examples.
"""

import argparse
import json
from pathlib import Path

import torch

from scripts.onsetfree.run_l1 import fixed_benign_batches
from src.models.small_cnn import SmallCNN
from src.unlearning.onsetfree.curvature import BenignCurvature, flatten, parameter_layout


def metrics(model, batches, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return {"loss": loss_sum / total, "accuracy_percent": 100 * correct / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--attack-model", required=True)
    parser.add_argument("--method", action="append", required=True, help="NAME=CHECKPOINT; repeat")
    parser.add_argument("--l1-metadata", help="Exclude curvature subsample from validation")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--malicious-client", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    counts_raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in counts_raw.items()}
    excluded = json.loads(Path(args.l1_metadata).read_text())["sample_indices_by_client"] if args.l1_metadata else None
    batches, selected = fixed_benign_batches(Path(args.source_run), counts, args.malicious_client, args.samples, args.seed, args.batch_size, args.data_dir, excluded)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device).eval()
    attack = torch.load(args.attack_model, map_location=device, weights_only=True)
    model.load_state_dict(attack)
    baseline = metrics(model, batches, device)
    curvature = BenignCurvature(model, batches, device)
    layout = parameter_layout(model)
    results = {}
    for entry in args.method:
        name, path = entry.split("=", 1)
        method = torch.load(path, map_location=device, weights_only=True)
        delta = flatten({key: method[key] - attack[key] for key, _, _ in layout}, layout)
        quadratic = 0.5 * torch.dot(delta, curvature.hessian(delta)).item()
        model.load_state_dict(method)
        observed = metrics(model, batches, device)
        results[name] = {
            "checkpoint": str(Path(path).resolve()),
            "loss": observed["loss"],
            "accuracy_percent": observed["accuracy_percent"],
            "loss_change": observed["loss"] - baseline["loss"],
            "accuracy_change_points": observed["accuracy_percent"] - baseline["accuracy_percent"],
            "half_delta_H_delta": quadratic,
        }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "role": "benign validation only; no test or trigger metadata",
        "source_run": str(Path(args.source_run).resolve()),
        "validation_sample_seed": args.seed,
        "validation_sample_indices_by_client": selected,
        "excluded_l1_curvature_sample": bool(args.l1_metadata),
        "baseline": baseline,
        "hessian_vector_products": curvature.calls,
        "methods": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
