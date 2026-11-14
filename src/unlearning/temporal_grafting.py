"""Build a fixed, norm-clipped mean of a client's historical parameter deltas."""

import math
from statistics import median

import torch


def build_static_graft(checkpoints, parameters, max_norm=None):
    mean = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in parameters}
    norms = []
    for path in checkpoints:
        update = torch.load(path, map_location="cpu", weights_only=True)
        squared_norm = 0.0
        for name, accumulator in mean.items():
            delta = update[name]
            if delta.shape != accumulator.shape or not delta.is_floating_point():
                raise ValueError(f"Invalid parameter delta {name} in {path}")
            squared_norm += delta.double().square().sum().item()
            accumulator.add_(delta)
        if not math.isfinite(squared_norm):
            raise ValueError(f"Non-finite parameter delta in {path}")
        norms.append(math.sqrt(squared_norm))
    if not norms:
        raise ValueError("At least one clean donor round is required")
    cap = median(norms) if max_norm is None else max_norm
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("The graft norm cap must be finite and nonnegative")
    for delta in mean.values():
        delta.div_(len(norms))
    raw_norm = math.sqrt(sum(delta.double().square().sum().item() for delta in mean.values()))
    scale = cap / raw_norm if raw_norm > cap else 1.0
    for delta in mean.values():
        delta.mul_(scale)
    return mean, {
        "donor_norms": norms, "max_norm": cap, "mean_norm": raw_norm,
        "clipped_norm": raw_norm * scale, "clip_scale": scale,
    }
