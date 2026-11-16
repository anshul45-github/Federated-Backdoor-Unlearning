"""Matched, resumable FedEraser calibration in results/onsetfree/.

Replicates the archived v5 reconstruction rules without writing its legacy
results paths. The attack config is read only for final evaluation.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.baseline.run_federaser_v5 import (
    aggregate_fedavg_modified, calibrate_update, compute_model_delta, seed_everything,
)
from src.data.backdoor import create_asr_dataloader
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.small_cnn import SmallCNN
from src.training.train_utils import evaluate_extended, train_epoch


def save_progress(output, round_number, state, elapsed_seconds):
    temp = output / "checkpoint.tmp.pt"
    torch.save(state, temp)
    os.replace(temp, output / "checkpoint.pt")
    (output / "progress.json").write_text(json.dumps({"completed_round": round_number, "elapsed_seconds": elapsed_seconds}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ecal", type=int, choices=(1, 2), required=True)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    source = Path(args.source_run)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DONE").exists():
        print(f"Already completed: {output}")
        return
    config = json.loads((source / "config.json").read_text())
    config.setdefault("trigger_type", "white_patch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_partitions = torch.load(source / "client_partitions.pt", weights_only=False)
    partitions = saved_partitions if isinstance(saved_partitions, list) else saved_partitions.get("partitions", saved_partitions.get("client_partitions"))
    malicious = config["malicious_client_id"]
    retained = [i for i in range(config["num_clients"]) if i != malicious]
    counts = [len(partitions[i]) for i in retained]
    model = SmallCNN().to(device)
    progress_path = output / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        start_round = progress["completed_round"] + 1
        elapsed_before = progress["elapsed_seconds"]
        model.load_state_dict(torch.load(output / "checkpoint.pt", map_location=device, weights_only=True))
    else:
        start_round = 1
        elapsed_before = 0.0
        model.load_state_dict(torch.load(source / "initial_model.pt", map_location=device, weights_only=True))
    criterion = torch.nn.CrossEntropyLoss()
    started = time.perf_counter()
    for round_number in range(start_round, config["num_rounds"] + 1):
        round_start = time.perf_counter()
        calibrated = []
        for i in retained:
            historical = torch.load(source / "trajectory" / f"round_{round_number:03d}" / f"delta_client_{i:02d}.pt", map_location=device, weights_only=True)
            if round_number == 1:
                update = historical
            else:
                seed = hash((config["seed"], round_number, i)) % (2**32)
                seed_everything(seed)
                generator = torch.Generator().manual_seed(seed)
                loader = DataLoader(partitions[i], batch_size=config["batch_size"], shuffle=True, generator=generator)
                local = SmallCNN().to(device)
                local.load_state_dict(model.state_dict())
                optimizer = torch.optim.SGD(local.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=0.0)
                for _ in range(args.ecal):
                    train_epoch(local, loader, criterion, optimizer, device)
                update = calibrate_update(historical, compute_model_delta(model.state_dict(), local.state_dict()))
            calibrated.append(update)
        model.load_state_dict(aggregate_fedavg_modified(model.state_dict(), calibrated, counts))
        elapsed = elapsed_before + time.perf_counter() - started
        save_progress(output, round_number, model.state_dict(), elapsed)
        print(f"round {round_number:03d} elapsed={elapsed:.1f}s round_seconds={time.perf_counter()-round_start:.1f}", flush=True)
    _, clean_loader = get_cifar10_dataloaders(args.data_dir, batch_size=config["batch_size"])
    asr_loader = create_asr_dataloader(clean_loader, config["target_class"], config["trigger_size"], config["batch_size"], config["trigger_type"])
    clean = evaluate_extended(model, clean_loader, criterion, device)
    asr = evaluate_extended(model, asr_loader, criterion, device)
    torch.save(model.state_dict(), output / "final_model.pt")
    result = {
        "source_run": str(source.resolve()),
        "ecal": args.ecal,
        "ca_percent": clean["accuracy"],
        "asr_percent": asr["accuracy"],
        "reconstruction_seconds": elapsed_before + time.perf_counter() - started,
        "client_flops": None,
        "flops_status": "pending measured or analytic count",
        "algorithm": "archived v5 whole-vector calibration, all 50 rounds, C1-C9",
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "DONE").write_text("Reconstruction and evaluation completed.\n")


if __name__ == "__main__":
    main()
