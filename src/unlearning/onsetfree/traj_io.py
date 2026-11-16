"""Read the trajectory written by scripts.run_experiment_v2.

Round numbers are one-based: theta_t is the state *before* round t and
theta_next is the saved state after it. Integer buffers require FedAvg's
round-to-integer rule and cannot be checked as ordinary additive tensors.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping

import torch


@dataclass
class RoundRecord:
    round_number: int
    theta_t: Dict[str, torch.Tensor]
    theta_next: Dict[str, torch.Tensor]
    deltas: Dict[int, Dict[str, torch.Tensor]]
    weights: Dict[int, float]


class Trajectory:
    def __init__(self, run_dir, sample_counts: Mapping[int, int]):
        self.run_dir = Path(run_dir)
        self.sample_counts = dict(sample_counts)
        if not self.sample_counts or sum(self.sample_counts.values()) <= 0:
            raise ValueError("Positive client sample counts are required")
        self.round_dirs = sorted((self.run_dir / "trajectory").glob("round_[0-9][0-9][0-9]"))
        if not self.round_dirs:
            raise FileNotFoundError(f"No round directories in {self.run_dir / 'trajectory'}")
        total = sum(self.sample_counts.values())
        self.weights = {i: n / total for i, n in self.sample_counts.items()}

    @staticmethod
    def _load(path):
        return torch.load(path, map_location="cpu", weights_only=True)

    def __iter__(self) -> Iterator[RoundRecord]:
        before = self._load(self.run_dir / "initial_model.pt")
        for round_dir in self.round_dirs:
            number = int(round_dir.name.split("_")[-1])
            deltas = {}
            for i in self.sample_counts:
                path = round_dir / f"delta_client_{i:02d}.pt"
                deltas[i] = self._load(path)
            after = self._load(round_dir / "global_model.pt")
            yield RoundRecord(number, before, after, deltas, self.weights.copy())
            before = after


def reconstruction_error(record: RoundRecord):
    """Maximum absolute error per key after reproducing FedAvg's dtype cast."""
    errors = {}
    for key, before in record.theta_t.items():
        update = torch.zeros_like(before, dtype=torch.float32)
        for i, weight in record.weights.items():
            update += record.deltas[i][key].float() * weight
        predicted = before.float() + update
        if before.is_floating_point():
            predicted = predicted.to(before.dtype)
        else:
            predicted = predicted.round().to(before.dtype)
        errors[key] = (predicted.float() - record.theta_next[key].float()).abs().max().item()
    return errors
