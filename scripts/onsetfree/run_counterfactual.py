"""Resumable counterfactual training without the removed client.

Attack metadata is used only for final ASR. The training loop matches
scripts.run_experiment_v2 without its per-round test evaluation.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml

from scripts.run_experiment_v2 import get_client_dataloader, set_seed
from src.data.backdoor import create_asr_dataloader
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.fl_partition import create_iid_partitions, create_noniid_partitions
from src.models.small_cnn import SmallCNN
from src.training.fedavg import aggregate_fedavg, compute_model_delta
from src.training.train_utils import evaluate_extended, train_epoch


def save_progress(output, round_number, state, elapsed_seconds):
    temp = output / "checkpoint.tmp.pt"
    torch.save(state, temp)
    os.replace(temp, output / "checkpoint.pt")
    (output / "progress.json").write_text(json.dumps({"completed_round": round_number, "elapsed_seconds": elapsed_seconds}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DONE").exists():
        print(f"Already completed: {output}")
        return
    config = yaml.safe_load(Path(args.config).read_text())
    config.setdefault("trigger_type", "white_patch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["seed"])
    train_loader, test_loader = get_cifar10_dataloaders(args.data_dir, batch_size=config["batch_size"])
    if config["partition_type"] == "iid":
        partitions, distributions = create_iid_partitions(train_loader.dataset, config["num_clients"], config["seed"])
    else:
        partitions, distributions = create_noniid_partitions(train_loader.dataset, config["num_clients"], config["dirichlet_alpha"], config["seed"])
    (output / "client_class_histograms.json").write_text(json.dumps(distributions, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    retained = tuple(i for i in range(config["num_clients"]) if i != config["malicious_client_id"])
    model = SmallCNN().to(device)
    progress_path = output / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        start_round = progress["completed_round"] + 1
        elapsed_before = progress["elapsed_seconds"]
        state = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
    else:
        start_round = 1
        elapsed_before = 0.0
        state = model.state_dict()
        torch.save(state, output / "initial_model.pt")
    model.load_state_dict(state)
    criterion = torch.nn.CrossEntropyLoss()
    started = time.perf_counter()
    for round_number in range(start_round, config["num_rounds"] + 1):
        deltas = []
        counts = []
        for i in retained:
            loader, _ = get_client_dataloader(config, partitions[i], i, round_number, False, [])
            local = SmallCNN().to(device)
            local.load_state_dict(state)
            optimizer = torch.optim.SGD(local.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
            for _ in range(config["local_epochs"]):
                train_epoch(local, loader, criterion, optimizer, device)
            deltas.append(compute_model_delta(state, local.state_dict()))
            counts.append(len(partitions[i]))
        state = aggregate_fedavg(state, deltas, counts)
        model.load_state_dict(state)
        elapsed = elapsed_before + time.perf_counter() - started
        save_progress(output, round_number, state, elapsed)
        print(f"round {round_number:03d} elapsed={elapsed:.1f}s", flush=True)
    asr_loader = create_asr_dataloader(test_loader, config["target_class"], config["trigger_size"], config["batch_size"], config["trigger_type"])
    clean = evaluate_extended(model, test_loader, criterion, device)
    asr = evaluate_extended(model, asr_loader, criterion, device)
    torch.save(state, output / "final_model.pt")
    (output / "final_results.json").write_text(json.dumps({
        "config": config,
        "excluded_client": config["malicious_client_id"],
        "clean_accuracy": clean["accuracy"],
        "triggered_rate": asr["accuracy"],
        "runtime_seconds": elapsed_before + time.perf_counter() - started,
        "mode": "counterfactual retraining",
    }, indent=2) + "\n")
    (output / "DONE").write_text("Counterfactual retraining and evaluation completed.\n")


if __name__ == "__main__":
    main()
