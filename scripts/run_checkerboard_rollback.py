"""Compare honest continuation from G29 with a saved checkerboard FedEraser model."""

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
from src.unlearning.temporal_grafting import SimilarityGraft, build_static_graft
from src.utils.attack_report import prepare_output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", type=Path, required=True)
    parser.add_argument("--federaser_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--variant", choices=("tcug-0", "tcug-1", "tcug-2"), default="tcug-0")
    parser.add_argument("--graft_window", type=int, default=5,
                        help="Number of consecutive clean C0 donor rounds ending at round 29")
    parser.add_argument("--graft_max_norm", type=float,
                        help="Global L2 cap for the mean; defaults to the median donor norm")
    args = parser.parse_args()
    if not 1 <= args.graft_window <= 29:
        parser.error("--graft_window must be between 1 and 29")
    if args.variant == "tcug-2" and args.graft_window < 3:
        parser.error("TCUG-2 requires --graft_window of at least 3")
    args.reference_model = None

    source = args.source_run
    config = read_json(source / "config.json")
    if (config.get("trigger_type"), config["attack_start_round"], config["num_rounds"],
            config["num_clients"], config["malicious_client_id"]) != ("checkerboard", 30, 50, 10, 0):
        raise ValueError("Expected the 50-round checkerboard protocol with Client 0 poisoning from round 30")

    with (source / "metrics_per_round.csv").open(newline="") as handle:
        rounds = list(csv.DictReader(handle))
    if ([int(row["Round"]) for row in rounds] != list(range(1, 51)) or
            any(row["Attack_Active"].lower() != "false" for row in rounds[:29])):
        raise ValueError("Source must have a complete round log and a clean first 29 rounds")

    donor = source / "trajectory" / "round_029" / "global_model.pt"
    federaser = args.federaser_dir / "final_model.pt"
    federaser_hash = sha256(federaser)
    metadata = read_json(args.federaser_dir / "run_config.json")
    if (metadata["method"] != "federaser" or metadata["status"] != "completed" or
            metadata["calibration_epochs"] != 2 or metadata["source_config"] != config or
            metadata["source_model_sha256"] != sha256(source / "final_model.pt") or
            metadata["output_model_sha256"] != federaser_hash):
        raise ValueError("FedEraser must be the completed, unmodified endpoint from this source run")

    donor_hash = sha256(donor)
    train, test = get_cifar10_dataloaders(str(args.data_dir), config["batch_size"])
    # The trusted local partition artifact contains pickled Subset objects.
    saved = torch.load(source / "client_partitions.pt", map_location="cpu", weights_only=False)
    if isinstance(saved, dict):
        saved = saved.get("partitions", saved.get("client_partitions"))
    partitions = [Subset(train.dataset, subset.indices) for subset in saved]
    all_indices = sorted(int(index) for subset in partitions for index in subset.indices)
    if len(partitions) != 10 or all_indices != list(range(len(train.dataset))):
        raise ValueError("Saved partitions must cover the CIFAR-10 training set exactly once")

    static_update = None
    selector = None
    graft_metadata = None
    if args.variant != "tcug-0":
        if not len(partitions[0]):
            raise ValueError("Grafting requires a nonempty Client 0 partition")
        donor_rounds = list(range(30 - args.graft_window, 30))
        paths = {r: source / "trajectory" / f"round_{r:03d}" / "delta_client_00.pt" for r in donor_rounds}
        parameters = SmallCNN().named_parameters()
        if args.variant == "tcug-1":
            graft, graft_metadata = build_static_graft(paths.values(), parameters, args.graft_max_norm)
            static_update = (graft, len(partitions[0]))
        else:
            selector = SimilarityGraft(paths, parameters, args.graft_max_norm)
            graft_metadata = {"top_k": 3, "similarity": "cosine",
                              "query": "sample_weighted_C1_to_C9_parameter_delta",
                              "max_norm": selector.cap, "donor_norms_by_round": selector.donor_norms}
        graft_metadata.update(donor_rounds=donor_rounds, donor_sha256=[sha256(path) for path in paths.values()],
                              sample_count=len(partitions[0]), fedavg_weight=len(partitions[0]) / len(train.dataset))

    triggered = create_asr_dataloader(test, config["target_class"], config["trigger_size"],
                                     config["batch_size"], config["trigger_type"])
    untriggered = create_untriggered_baseline_loader(test, config["target_class"], config["batch_size"])
    output = Path(prepare_output_directory(args.output_dir, unique_on_conflict=True))
    print(f"Output: {output.resolve()}", flush=True)
    retained = tuple(range(1, 10))
    (output / "run_config.json").write_text(json.dumps({
        "source_run": str(source.resolve()), "source_config": config,
        "round_29_sha256": donor_hash, "federaser_dir": str(args.federaser_dir.resolve()),
        "federaser_sha256": federaser_hash, "first_round": 30, "last_round": 50,
        "retained_clients": retained, "torch_version": torch.__version__,
        "variant": args.variant, "graft": graft_metadata,
    }, indent=2), encoding="utf-8")
    run = run_fl_simulation(config, partitions, test, triggered, untriggered, False, output,
                            initial_checkpoint=donor, start_round=30, client_ids=retained,
                            static_client_update=static_update,
                            adaptive_client_update=(selector, len(partitions[0])) if selector else None)
    if selector is not None:
        (output / "graft_history.json").write_text(json.dumps(selector.records, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    results = {}
    endpoint = {"tcug-0": "rollback_21_honest_rounds", "tcug-1": "tcug_1", "tcug-2": "tcug_2"}[args.variant]
    for label, checkpoint in (("round_29", donor), ("federaser", federaser),
                              (endpoint, output / "final_model.pt")):
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        results[label] = evaluate(torch, model, config, args, device)
    results[endpoint]["runtime_seconds"] = run["runtime_seconds"]
    (output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fields = ("clean_accuracy", "attack_success_rate", "untriggered_target_rate", "macro_f1_pct")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", *fields))
        for label, metrics in results.items():
            writer.writerow((label, *(metrics[field] for field in fields)))
    print((output / "comparison.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
