"""Resumable L3 clean FedAvg repair from an onset-free checkpoint.

Only retained clients train. Trigger and target metadata from the saved
config are used solely for final evaluation after all repair rounds.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from scripts.run_experiment_v2 import get_client_dataloader, set_seed
from src.data.backdoor import create_asr_dataloader
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.small_cnn import SmallCNN
from src.training.fedavg import aggregate_fedavg, compute_model_delta
from src.training.train_utils import evaluate_extended, train_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repair-rounds", type=int, default=3)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cost-json", help="Measured FLOPs for one retained-client epoch")
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
    set_seed(config["seed"])
    _, test_loader = get_cifar10_dataloaders(args.data_dir, batch_size=config["batch_size"])
    partitions = torch.load(source / "client_partitions.pt", weights_only=False)
    if not isinstance(partitions, list):
        partitions = partitions.get("partitions", partitions.get("client_partitions"))
    retained = [i for i in range(config["num_clients"]) if i != config["malicious_client_id"]]
    model = SmallCNN().to(device)
    progress_path = output / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        completed = progress["completed_repair_rounds"]
        elapsed_before = progress["elapsed_seconds"]
        state = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
    else:
        completed = 0
        elapsed_before = 0.0
        state = torch.load(args.input_model, map_location=device, weights_only=True)
        torch.save(state, output / "initial_model.pt")
    model.load_state_dict(state)
    criterion = torch.nn.CrossEntropyLoss()
    started = time.perf_counter()
    for repair_round in range(completed + 1, args.repair_rounds + 1):
        round_number = config["num_rounds"] + repair_round
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
        temp = output / "checkpoint.tmp.pt"
        torch.save(state, temp)
        os.replace(temp, output / "checkpoint.pt")
        elapsed = elapsed_before + time.perf_counter() - started
        progress_path.write_text(json.dumps({"completed_repair_rounds": repair_round, "elapsed_seconds": elapsed}, indent=2) + "\n")
        print(f"repair round {repair_round}/{args.repair_rounds}, global round {round_number}, elapsed={elapsed:.1f}s", flush=True)
    asr_loader = create_asr_dataloader(test_loader, config["target_class"], config["trigger_size"], config["batch_size"], config["trigger_type"])
    clean = evaluate_extended(model, test_loader, criterion, device)
    asr = evaluate_extended(model, asr_loader, criterion, device)
    torch.save(state, output / "final_model.pt")
    cost = json.loads(Path(args.cost_json).read_text()) if args.cost_json else None
    (output / "results.json").write_text(json.dumps({
        "source_run": str(source.resolve()),
        "input_model": str(Path(args.input_model).resolve()),
        "repair_rounds": args.repair_rounds,
        "communication_rounds": args.repair_rounds,
        "local_epochs_per_round": config["local_epochs"],
        "ca_percent": clean["accuracy"],
        "asr_percent": asr["accuracy"],
        "wall_seconds": elapsed_before + time.perf_counter() - started,
        "client_flops": args.repair_rounds * config["local_epochs"] * cost["one_retained_client_epoch_flops"] if cost else None,
        "flops_formula": "repair_rounds * local_epochs_per_round * one_retained_client_epoch_flops; forward/backward ATen FLOPs only",
        "cost_json": str(Path(args.cost_json).resolve()) if args.cost_json else None,
    }, indent=2) + "\n")
    (output / "DONE").write_text("Retained-client repair and final evaluation completed.\n")


if __name__ == "__main__":
    main()
