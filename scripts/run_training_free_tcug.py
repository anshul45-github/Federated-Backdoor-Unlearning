"""Repair a saved checkerboard endpoint with historical Client 0 deltas."""

import argparse
import csv
import json
from pathlib import Path

import torch

from scripts.run_checkerboard_unlearning import evaluate, sha256, validate_inputs, write_results
from src.models.small_cnn import SmallCNN
from src.unlearning.temporal_grafting import SimilarityGraft
from src.utils.attack_report import prepare_output_directory


def repair_endpoint(attacked, trajectory, selector, counts):
    """Replace weighted C0 parameter deltas with round-specific clean grafts."""
    client_weight = counts[0] / sum(counts)
    # Accumulate in float64 and cast once after applying the correction.
    corrections = {key: torch.zeros_like(attacked[key], dtype=torch.float64)
                   for key in selector.parameters}
    for round_idx in range(30, 51):
        paths = trajectory[round_idx]
        deltas = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        graft = selector(round_idx, deltas[1:], counts[1:])
        for key, total in corrections.items():
            poisoned = deltas[0][key]
            if (poisoned.shape != total.shape or not poisoned.is_floating_point()
                    or not torch.isfinite(poisoned).all()):
                raise ValueError(f"Invalid poisoned delta for {key} in {paths[0]}")
            total.add_(graft[key].double() - poisoned.double(), alpha=client_weight)
    repaired = attacked.copy()
    for key, correction in corrections.items():
        repaired[key] = (attacked[key].double() + correction).to(attacked[key].dtype)
    return repaired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--graft_max_norm", type=float,
                        help="Defaults to the median norm of the clean donor deltas")
    parser.set_defaults(method="training_free_tcug_2", reference_model=None, subtraction_dir=None)
    args = parser.parse_args()
    source, config, source_metrics, counts, source_hash, _ = validate_inputs(args)
    if config["attack_start_round"] != 30 or config["num_rounds"] != 50:
        raise ValueError("Expected clean donors at rounds 20–29 and an attack at rounds 30–50")
    with (source / "metrics_per_round.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [(int(row["Round"]), row["Attack_Active"].lower()) for row in rows] != [
        (r, str(r >= 30).lower()) for r in range(1, 51)
    ]:
        raise ValueError("Source round log does not match the expected attack schedule")

    trajectory = source / "trajectory"
    donors = {r: trajectory / f"round_{r:03d}" / "delta_client_00.pt" for r in range(20, 30)}
    continuation = {r: [trajectory / f"round_{r:03d}" / f"delta_client_{c:02d}.pt"
                        for c in range(10)] for r in range(30, 51)}
    donor_hashes = {r: sha256(path) for r, path in donors.items()}
    continuation_hashes = {r: [sha256(path) for path in paths] for r, paths in continuation.items()}
    weight = counts[0] / sum(counts)
    attacked = torch.load(source / "final_model.pt", map_location="cpu", weights_only=True)
    selector = SimilarityGraft(donors, SmallCNN().named_parameters(), args.graft_max_norm)
    repaired = repair_endpoint(attacked, continuation, selector, counts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    model.load_state_dict(repaired)
    metrics = evaluate(torch, model, config, args, device)

    output = Path(prepare_output_directory(args.output_dir, unique_on_conflict=True))
    checkpoint = output / "final_model.pt"
    torch.save(repaired, checkpoint)
    metadata = {
        "method": args.method, "source_run": str(source), "source_config": config,
        "source_model_sha256": source_hash, "output_model_sha256": sha256(checkpoint),
        "clean_rounds": list(range(20, 30)), "replaced_rounds": list(range(30, 51)),
        "client_sample_counts": counts, "client_0_fedavg_weight": weight,
        "donor_delta_sha256_by_round": donor_hashes, "round_delta_sha256_by_client": continuation_hashes,
        "clean_estimator": "mean of top three cosine-similar clean C0 parameter deltas",
        "query": "sample-weighted historical C1–C9 parameter delta for the same round",
        "graft_max_norm": selector.cap, "donor_norms_by_round": selector.donor_norms,
        "state_policy": "trainable parameters corrected; buffers copied from attacked model",
        "evaluation_device": str(device), "torch_version": torch.__version__,
    }
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "graft_history.json").write_text(json.dumps(selector.records, indent=2) + "\n", encoding="utf-8")
    write_results(output, source_metrics, metrics, metadata)
    print(f"{output.resolve()}\n{(output / 'results.csv').read_text(encoding='utf-8')}", flush=True)


if __name__ == "__main__":
    main()
