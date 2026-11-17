"""Run three clean FedAvg rounds from a saved TCUG repair endpoint."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from scripts.run_checkerboard_unlearning import evaluate, read_json, sha256
from scripts.run_experiment_v2 import create_untriggered_baseline_loader, run_fl_simulation
from src.data.backdoor import create_asr_dataloader
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.small_cnn import SmallCNN
from src.training.class_balanced_healing import ClassBalancedHealing
from src.utils.attack_report import prepare_output_directory


def select_repair_checkpoint(repair_dir, metadata, repair_lambda=None):
    method = metadata["method"]
    if method == "training_free_tcug_2":
        if repair_lambda is not None:
            raise ValueError("--repair_lambda applies only to direction-preserving TCUG")
        return repair_dir / "final_model.pt", metadata["output_model_sha256"], method
    if method == "direction_preserving_tcug":
        label = f"lambda_{1 if repair_lambda is None else repair_lambda}"
        return (repair_dir / f"{label}.pt", metadata["checkpoint_sha256"][label],
                f"{method}_{label}")
    raise ValueError(f"Unsupported TCUG repair method: {method}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", type=Path, required=True,
                        help="Original attacked run with the saved client partitions")
    parser.add_argument("--repair_dir", type=Path, required=True,
                        help="Completed training-free TCUG-2 or direction-preserving TCUG output directory")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repair_lambda", type=int, choices=(0, 1),
                        help="Direction-preserving checkpoint to heal (default: 1)")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--cbsh", action="store_true",
                        help="Select healing clients each round using class recall on held-out data")
    parser.add_argument("--cbsh_clients_per_round", type=int, choices=range(1, 10), default=5)
    parser.add_argument("--cbsh_validation_per_class", type=int, default=100)
    parser.add_argument("--cbsh_validation_seed", type=int, default=2026)
    parser.add_argument("--random_client_seed", type=int,
                        help="With --cbsh, sample clients uniformly instead of ranking by validation recall")
    parser.set_defaults(reference_model=None)
    args = parser.parse_args()
    if args.random_client_seed is not None and not args.cbsh:
        parser.error("--random_client_seed requires --cbsh to reuse its validation split")
    if args.cbsh and args.repair_lambda != 0:
        parser.error("--cbsh requires --repair_lambda 0")
    if args.cbsh and args.cbsh_validation_per_class < 1:
        parser.error("--cbsh_validation_per_class must be positive")
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}", flush=True)

    config = read_json(args.source_run / "config.json")
    metadata = read_json(args.repair_dir / "run_config.json")
    checkpoint, expected_hash, endpoint = select_repair_checkpoint(args.repair_dir, metadata, args.repair_lambda)
    repair_hash = sha256(checkpoint)
    if (metadata["source_config"] != config
            or metadata["source_model_sha256"] != sha256(args.source_run / "final_model.pt")
            or expected_hash != repair_hash
            or config["num_rounds"] != 50 or config["num_clients"] != 10
            or config["malicious_client_id"] != 0):
        raise ValueError(f"Expected a verified {endpoint} checkpoint from this 50-round source")

    train, test = get_cifar10_dataloaders(str(args.data_dir), config["batch_size"])
    partitions_file = args.source_run / "client_partitions.pt"
    # Trust the original local run artifact, which contains pickled Subset objects.
    saved = torch.load(partitions_file, map_location="cpu", weights_only=False)
    if isinstance(saved, dict):
        saved = saved.get("partitions", saved.get("client_partitions"))
    partitions = [Subset(train.dataset, subset.indices) for subset in saved]
    indices = sorted(int(i) for subset in partitions for i in subset.indices)
    if (len(partitions) != 10 or indices != list(range(len(train.dataset)))
            or [len(subset) for subset in partitions] != metadata["client_sample_counts"]):
        raise ValueError("Saved partitions must cover CIFAR-10 and match the repaired run's client counts")

    clients = tuple(range(1, 10))
    healing = None
    if args.cbsh:
        healing = ClassBalancedHealing(
            partitions, clients, test.dataset.transform, config["batch_size"], args.cbsh_clients_per_round,
            args.cbsh_validation_per_class, args.cbsh_validation_seed, args.random_client_seed,
        )
        partitions = healing.partitions

    triggered = create_asr_dataloader(test, config["target_class"], config["trigger_size"],
                                     config["batch_size"], config["trigger_type"])
    untriggered = create_untriggered_baseline_loader(test, config["target_class"], config["batch_size"])
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    before = evaluate(torch, model, config, args, device)
    del model

    output = Path(prepare_output_directory(args.output_dir, unique_on_conflict=True))
    print(f"Output: {output.resolve()}", flush=True)
    continuation = {**config, "num_rounds": 53}
    run = run_fl_simulation(continuation, partitions, test, triggered, untriggered,
                            False, output, initial_checkpoint=checkpoint, start_round=51,
                            client_ids=clients, select_clients=healing)
    final = output / "final_model.pt"
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(final, map_location=device, weights_only=True))
    after = evaluate(torch, model, config, args, device)
    method = f"{endpoint}_plus_cbsh_3_rounds" if healing else f"{endpoint}_plus_3_rounds"
    if args.random_client_seed is not None:
        method = f"{endpoint}_plus_random_healing_3_rounds"
    evaluations = {endpoint: before, method: after}
    provenance = {
        "method": method, "source_run": str(args.source_run.resolve()),
        "source_config": config, "repair_dir": str(args.repair_dir.resolve()),
        "repair_checkpoint": checkpoint.name,
        "repair_sha256": repair_hash, "output_model_sha256": sha256(final),
        "client_partitions_sha256": sha256(partitions_file), "rounds": [51, 52, 53],
        "training_clients": clients, "attack_enabled": False, "evaluation_device": str(device),
        "torch_version": torch.__version__, "runtime_seconds": run["runtime_seconds"],
    }
    if healing:
        provenance["training_clients"] = sorted({client for row in healing.history
                                                  for client in row["selected_clients"]})
        provenance["cbsh"] = {
            "eligible_clients": clients, "clients_per_round": args.cbsh_clients_per_round,
            "validation_per_class": args.cbsh_validation_per_class,
            "validation_seed": args.cbsh_validation_seed,
            "validation_source": "stratified retained-client CIFAR-10 training examples",
            "client_score": "sum(class_count * (1 - validation recall))",
            "client_selection": "recall_deficit" if args.random_client_seed is None else "uniform_without_replacement",
            "random_client_seed": args.random_client_seed,
            "aggregation": "sample-weighted over selected healing partitions",
        }
        selection = {
            "validation_indices": healing.validation_indices,
            "healing_class_counts": healing.class_counts, "rounds": healing.history,
            "validation_after_round_53": healing.measure(model, device),
        }
        (output / "selection_history.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    for name, record in (("run_config", provenance), ("evaluation", evaluations)):
        (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    fields = ("clean_accuracy", "attack_success_rate", "untriggered_target_rate", "macro_f1_pct")
    comparison = output / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", *fields))
        for name, metrics in evaluations.items():
            writer.writerow((name, *(metrics[field] for field in fields)))
    print(comparison.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
