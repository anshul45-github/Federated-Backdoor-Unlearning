"""Time-varying-curvature L2 tail using one/two GGN products per round.

This is the registered Taylor recursion. The effective learning rate under
momentum is an explicit approximation; see metadata and compare to L1.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from scripts.onsetfree.run_l1 import fixed_benign_batches
from src.models.small_cnn import SmallCNN
from src.unlearning.onsetfree.curvature import BenignCurvature, flatten, l2_tail, parameter_layout, spectral_response
from src.unlearning.onsetfree.l0 import apply_update
from src.unlearning.onsetfree.traj_io import Trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--l1-spectrum", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--malicious-client", type=int, default=0)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--curvature-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--second-order", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.momentum < 1:
        raise ValueError("Momentum must be in [0, 1)")
    started = time.perf_counter()
    source = Path(args.source_run)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts_raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in counts_raw.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectrum = torch.load(args.l1_spectrum, map_location=device, weights_only=True)
    basis = spectrum["basis"]
    contractions = spectrum["contractions"]
    model = SmallCNN().to(device).eval()
    layout = parameter_layout(model)
    batches, selected = fixed_benign_batches(source, counts, args.malicious_client, args.curvature_samples, args.seed, args.batch_size, args.data_dir)
    records = list(Trajectory(source, counts))
    if not 1 <= args.window <= len(records):
        raise ValueError("Window must be between 1 and the number of saved rounds")
    forces = []
    for record in records:
        pm = record.weights[args.malicious_client]
        malicious = flatten(record.deltas[args.malicious_client], layout).to(device)
        benign = torch.zeros_like(malicious)
        for i, weight in record.weights.items():
            if i != args.malicious_client:
                benign += (weight / (1 - pm)) * flatten(record.deltas[i], layout).to(device)
        forces.append(pm * (malicious - benign))
    prefix = spectral_response(forces[:-args.window], basis, contractions) if len(forces) > args.window else torch.zeros_like(forces[0])
    benign_total = sum(n for i, n in counts.items() if i != args.malicious_client)
    weighted_steps = sum((n / benign_total) * args.local_epochs * math.ceil(n / args.batch_size) for i, n in counts.items() if i != args.malicious_client)
    eta_eff = args.learning_rate / (1 - args.momentum)
    hvps = []
    operators = []
    for record in records[-args.window:]:
        model.load_state_dict(record.theta_t)
        operator = BenignCurvature(model, batches, device)
        operators.append(operator)
        hvps.append(operator.ggn)
    correction = l2_tail(prefix, forces[-args.window:], hvps, eta_eff, weighted_steps, args.second_order)
    final_state = torch.load(source / "final_model.pt", map_location=device, weights_only=True)
    update = {}
    offset = 0
    for name, shape, width in layout:
        update[name] = correction[offset:offset + width].reshape(shape)
        offset += width
    torch.save(apply_update(final_state, update), output / "l2.pt")
    metadata = {
        "source_run": str(source.resolve()),
        "l1_spectrum": str(Path(args.l1_spectrum).resolve()),
        "window": args.window,
        "curvature_samples": args.curvature_samples,
        "sample_seed": args.seed,
        "sample_indices_by_client": selected,
        "weighted_local_steps": weighted_steps,
        "eta_eff": eta_eff,
        "eta_eff_note": "heuristic eta/(1-momentum); registered first/second-order recursion is not exact for many momentum steps",
        "second_order": args.second_order,
        "ggn_vector_products": sum(operator.calls for operator in operators),
        "bn_policy": "eval-mode fixed stored statistics for GGN; final attacked buffers kept in checkpoint",
        "wall_seconds": time.perf_counter() - started,
        "metrics_status": "not evaluated",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
