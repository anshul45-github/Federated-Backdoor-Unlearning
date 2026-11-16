"""Leave-one-out residuals and benign-calibrated CUSUM; no trigger access."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.models.small_cnn import SmallCNN
from src.unlearning.onsetfree.l0 import project_out, right_basis
from src.unlearning.onsetfree.onset import calibrate_from_benign, cusum, robust_z
from src.unlearning.onsetfree.traj_io import Trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--malicious-client", type=int, default=0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--true-onset", type=int, help="Evaluation only; never used by detector")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts_raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in counts_raw.items()}
    records = list(Trajectory(args.source_run, counts))
    benign_ids = [i for i in sorted(counts) if i != args.malicious_client]
    parameter_keys = sorted(dict(SmallCNN().named_parameters()))
    norms_sq = [{i: 0.0 for i in counts} for _ in records]
    for key in parameter_keys:
        rows = [record.deltas[i][key].reshape(-1).float() for record in records for i in benign_ids]
        basis = right_basis(rows, args.rank, seed=42)
        for round_index, record in enumerate(records):
            pm = record.weights[args.malicious_client]
            benign_sum = sum((record.weights[i] * record.deltas[i][key].reshape(-1).float() for i in benign_ids), torch.zeros_like(rows[0]))
            for i in counts:
                ui = record.deltas[i][key].reshape(-1).float()
                if i == args.malicious_client:
                    reference = benign_sum / (1 - pm)
                else:
                    reference = (benign_sum - record.weights[i] * ui) / (1 - pm - record.weights[i])
                residual = project_out(ui - reference, basis)
                norms_sq[round_index][i] += torch.dot(residual, residual).item()
    residuals = [{i: math.sqrt(value) for i, value in round_values.items()} for round_values in norms_sq]
    benign_residuals = [{i: round_values[i] for i in benign_ids} for round_values in residuals]
    z = [robust_z(round_values[args.malicious_client], [round_values[i] for i in benign_ids]) for round_values in residuals]
    analyses = {}
    for kappa in (0.5, 1.0):
        threshold, pseudo_peaks = calibrate_from_benign(benign_residuals, kappa)
        trace, estimated_onset = cusum(z, kappa, threshold)
        analyses[str(kappa)] = {
            "threshold": threshold,
            "benign_pseudo_target_maxima": pseudo_peaks,
            "empirical_benign_false_alarm_rate": sum(value > threshold for value in pseudo_peaks.values()) / len(pseudo_peaks),
            "cusum": trace,
            "estimated_onset": estimated_onset,
            "onset_error_rounds": estimated_onset - args.true_onset if estimated_onset is not None and args.true_onset is not None else None,
        }
    (output / "detector.json").write_text(json.dumps({
        "source_run": str(Path(args.source_run).resolve()),
        "malicious_client": args.malicious_client,
        "rank": args.rank,
        "true_onset_evaluation_only": args.true_onset,
        "residuals": residuals,
        "malicious_robust_z": z,
        "analyses": analyses,
    }, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    rounds = range(1, len(z) + 1)
    axes[0].plot(rounds, z, label="malicious robust z")
    axes[0].set_ylabel("robust z")
    for label, entry in analyses.items():
        axes[1].plot(rounds, entry["cusum"], label=f"CUSUM kappa={label}")
        axes[1].axhline(entry["threshold"], linestyle="--", alpha=0.5)
    if args.true_onset is not None:
        for axis in axes:
            axis.axvline(args.true_onset, color="black", linestyle=":", label="true onset (evaluation)")
    axes[1].set_ylabel("CUSUM")
    axes[1].set_xlabel("round")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "detector.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
