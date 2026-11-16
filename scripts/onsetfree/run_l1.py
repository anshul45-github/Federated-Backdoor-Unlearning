"""Default stationary-spectrum L1 from a fixed benign training subsample.

No attack onset, trigger, target label, test data or counterfactual is read.
The model and historical updates are from a completed attack trajectory.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.small_cnn import SmallCNN
from src.unlearning.onsetfree.curvature import (
    BenignCurvature, flatten, heavy_ball_contraction, lanczos_top,
    parameter_layout, spectral_response,
)
from src.unlearning.onsetfree.l0 import apply_update
from src.unlearning.onsetfree.traj_io import Trajectory


def fixed_benign_batches(source, counts, malicious, total_samples, seed, batch_size, data_dir, exclude_by_client=None):
    partitions = torch.load(source / "client_partitions.pt", map_location="cpu", weights_only=False)
    if not isinstance(partitions, list):
        partitions = partitions.get("partitions", partitions.get("client_partitions"))
    benign = [i for i in sorted(counts) if i != malicious]
    total_benign = sum(counts[i] for i in benign)
    quotas = {i: total_samples * counts[i] / total_benign for i in benign}
    allocations = {i: min(counts[i], math.floor(quotas[i])) for i in benign}
    remaining = total_samples - sum(allocations.values())
    for i in sorted(benign, key=lambda i: (-(quotas[i] - math.floor(quotas[i])), i)):
        if remaining and allocations[i] < counts[i]:
            allocations[i] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Not enough benign samples for the fixed curvature subsample")
    generator = torch.Generator().manual_seed(seed)
    exclude_by_client = exclude_by_client or {}
    selected = {}
    indices = []
    for i in benign:
        partition = partitions[i]
        partition_indices = partition.indices if isinstance(partition, Subset) else partition
        excluded = set(exclude_by_client.get(str(i), exclude_by_client.get(i, [])))
        partition_indices = [int(index) for index in partition_indices if int(index) not in excluded]
        if len(partition_indices) < allocations[i]:
            raise ValueError(f"Not enough disjoint benign validation examples for client {i}")
        chosen = torch.randperm(len(partition_indices), generator=generator)[:allocations[i]].tolist()
        selected[i] = [int(partition_indices[j]) for j in chosen]
        indices.extend(selected[i])
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    batches = list(DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0))
    return batches, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--malicious-client", type=int, default=0)
    parser.add_argument("--rank", type=int, default=50)
    parser.add_argument("--curvature-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    started = time.perf_counter()
    source = Path(args.source_run)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts_raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in counts_raw.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device).eval()
    final_state = torch.load(source / "final_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(final_state)
    batches, selected = fixed_benign_batches(source, counts, args.malicious_client, args.curvature_samples, args.seed, args.batch_size, args.data_dir)
    curvature = BenignCurvature(model, batches, device)
    layout = parameter_layout(model)
    dimension = sum(width for _, _, width in layout)
    eigenvalues, basis = lanczos_top(curvature.ggn, dimension, args.rank, device, seed=args.seed)
    benign_total = sum(n for i, n in counts.items() if i != args.malicious_client)
    contractions = torch.zeros_like(eigenvalues)
    for i, n in counts.items():
        if i == args.malicious_client:
            continue
        steps = args.local_epochs * math.ceil(n / args.batch_size)
        contractions += (n / benign_total) * heavy_ball_contraction(eigenvalues, args.learning_rate, args.momentum, steps)
    torch.save({"eigenvalues": eigenvalues.cpu(), "basis": basis.cpu(), "contractions": contractions.cpu()}, output / "spectrum.pt")
    forces = []
    for record in Trajectory(source, counts):
        pm = record.weights[args.malicious_client]
        malicious = flatten(record.deltas[args.malicious_client], layout).to(device)
        benign = torch.zeros_like(malicious)
        for i, weight in record.weights.items():
            if i != args.malicious_client:
                benign += (weight / (1 - pm)) * flatten(record.deltas[i], layout).to(device)
        forces.append(pm * (malicious - benign))
    correction = spectral_response(forces, basis, contractions)
    update = {}
    offset = 0
    for name, shape, width in layout:
        update[name] = correction[offset:offset + width].reshape(shape)
        offset += width
    torch.save(apply_update(final_state, update), output / "l1.pt")
    metadata = {
        "source_run": str(source.resolve()),
        "malicious_client": args.malicious_client,
        "rank": args.rank,
        "curvature_samples": args.curvature_samples,
        "sample_seed": args.seed,
        "sample_indices_by_client": selected,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "contraction_model": "weighted exact constant-curvature heavy-ball perturbation transfer per client, zero initial local momentum",
        "eigenvalues": eigenvalues.detach().cpu().tolist(),
        "contractions": contractions.detach().cpu().tolist(),
        "contractions_below_half": int((contractions < 0.5).sum().item()),
        "ggn_vector_products": curvature.calls,
        "bn_policy": "eval-mode fixed running statistics for GGN; final attacked buffers kept in checkpoint",
        "wall_seconds": time.perf_counter() - started,
        "metrics_status": "not evaluated",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
