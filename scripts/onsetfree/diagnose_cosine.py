"""Evaluation-only cosine to counterfactual displacement, per parameter layer."""

import argparse
import json
from pathlib import Path

import torch

from src.models.small_cnn import SmallCNN


def cosine(a, b):
    denominator = a.norm() * b.norm()
    return (torch.dot(a, b) / denominator).item() if denominator.item() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-model", required=True)
    parser.add_argument("--counterfactual-model", required=True)
    parser.add_argument("--method-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    attack = torch.load(args.attack_model, map_location="cpu", weights_only=True)
    counterfactual = torch.load(args.counterfactual_model, map_location="cpu", weights_only=True)
    keys = tuple(dict(SmallCNN().named_parameters()))
    star = {key: (counterfactual[key] - attack[key]).reshape(-1).float() for key in keys}
    result = {
        "role": "diagnostic only; counterfactual never enters method selection",
        "attack_model": str(Path(args.attack_model).resolve()),
        "counterfactual_model": str(Path(args.counterfactual_model).resolve()),
        "methods": {},
    }
    for path in sorted(Path(args.method_dir).glob("*.pt")):
        method = torch.load(path, map_location="cpu", weights_only=True)
        displacement = {key: (method[key] - attack[key]).reshape(-1).float() for key in keys}
        result["methods"][path.stem] = {
            "per_layer_cosine": {key: cosine(displacement[key], star[key]) for key in keys},
            "whole_vector_cosine": cosine(torch.cat(tuple(displacement.values())), torch.cat(tuple(star.values()))),
        }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
