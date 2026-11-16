"""Evaluation-only source-class ASR on the fixed CIFAR-10 test population."""

import argparse
import json
from pathlib import Path

import torch

from src.data.backdoor import PoisonedDataset
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.small_cnn import SmallCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-config", required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    config = json.loads(Path(args.attack_config).read_text())
    target = config["target_class"]
    size = config["trigger_size"]
    _, test_loader = get_cifar10_dataloaders(args.data_dir, batch_size=config["batch_size"])
    patch = PoisonedDataset(test_loader.dataset, [], target, size, config.get("trigger_type", "white_patch")).trigger_patch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device).eval()
    result = {"role": "evaluation only; attack metadata and test data never enter method construction", "target_class": target, "trigger_size": size, "models": {}}
    for entry in args.model:
        name, path = entry.split("=", 1)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        successes = {str(i): 0 for i in range(10) if i != target}
        counts = {str(i): 0 for i in range(10) if i != target}
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                triggered = images.clone()
                triggered[:, :, -size:, -size:] = patch.to(device=device, dtype=images.dtype)
                predictions = model(triggered).argmax(1).cpu()
                for label in counts:
                    mask = labels == int(label)
                    counts[label] += int(mask.sum())
                    successes[label] += int(((predictions == target) & mask).sum())
        total = sum(counts.values())
        result["models"][name] = {
            "checkpoint": str(Path(path).resolve()),
            "overall_asr_percent": 100 * sum(successes.values()) / total,
            "per_source_class_asr_percent": {label: 100 * successes[label] / counts[label] for label in counts},
            "per_source_class_count": counts,
        }
        print(name, result["models"][name]["overall_asr_percent"], flush=True)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
