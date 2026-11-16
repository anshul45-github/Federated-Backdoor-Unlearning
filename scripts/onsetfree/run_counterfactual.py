"""Matched 50-round retraining without the removed client.

The trigger fields in the config are used only to report ASR after training.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.run_experiment_v2 import create_untriggered_baseline_loader, run_fl_simulation, set_seed
from src.data.backdoor import create_asr_dataloader
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.fl_partition import create_iid_partitions, create_noniid_partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing counterfactual output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    config.setdefault("trigger_type", "white_patch")
    set_seed(config["seed"])
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=config["batch_size"])
    if config["partition_type"] == "iid":
        partitions, distributions = create_iid_partitions(train_loader.dataset, config["num_clients"], config["seed"])
    else:
        partitions, distributions = create_noniid_partitions(train_loader.dataset, config["num_clients"], config["dirichlet_alpha"], config["seed"])
    (output / "client_class_histograms.json").write_text(json.dumps(distributions, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.save(partitions, output / "client_partitions.pt")
    asr_loader = create_asr_dataloader(test_loader, config["target_class"], config["trigger_size"], config["batch_size"], config["trigger_type"])
    untriggered_loader = create_untriggered_baseline_loader(test_loader, config["target_class"], config["batch_size"])
    retained = tuple(i for i in range(config["num_clients"]) if i != config["malicious_client_id"])
    result = run_fl_simulation(
        config, partitions, test_loader, asr_loader, untriggered_loader,
        is_attack=False, output_dir=str(output), client_ids=retained,
    )
    (output / "final_results.json").write_text(json.dumps({
        "config": config,
        "excluded_client": config["malicious_client_id"],
        "clean_accuracy": result["clean_eval"]["accuracy"],
        "triggered_rate": result["triggered_eval"]["accuracy"],
        "runtime_seconds": result["runtime_seconds"],
        "mode": "counterfactual retraining",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
