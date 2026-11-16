"""Evaluate saved L0 checkpoints; attack metadata is used here only."""

import argparse
import json
from pathlib import Path

import torch

from src.data.backdoor import create_asr_dataloader
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.small_cnn import SmallCNN
from src.training.train_utils import evaluate_extended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--attack-config", required=True, help="Evaluation only; never passed to run_l0")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.attack_config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, clean_loader = get_cifar10_dataloaders(args.data_dir, batch_size=config["batch_size"])
    asr_loader = create_asr_dataloader(
        clean_loader,
        target_class=config["target_class"],
        trigger_size=config["trigger_size"],
        batch_size=config["batch_size"],
        trigger_type=config.get("trigger_type", "white_patch"),
    )
    criterion = torch.nn.CrossEntropyLoss()
    results = {}
    for checkpoint in sorted(Path(args.model_dir).glob("*.pt")):
        model = SmallCNN().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        clean = evaluate_extended(model, clean_loader, criterion, device)
        triggered = evaluate_extended(model, asr_loader, criterion, device)
        results[checkpoint.stem] = {
            "ca_percent": clean["accuracy"],
            "asr_percent": triggered["accuracy"],
            "clean_loss": clean["loss"],
            "clean_count": len(clean_loader.dataset),
            "triggered_non_target_count": len(asr_loader.dataset),
        }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"evaluation_only_attack_config": str(Path(args.attack_config).resolve()), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
