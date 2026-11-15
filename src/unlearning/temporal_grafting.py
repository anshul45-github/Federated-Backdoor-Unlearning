"""Build static and similarity-selected grafts from clean historical deltas."""

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
    cap = graft_norm_cap(norms, max_norm)
    for delta in mean.values():
        delta.div_(len(norms))
    raw_norm = math.sqrt(sum(delta.double().square().sum().item() for delta in mean.values()))
    scale = clip_scale(raw_norm, cap)
    for delta in mean.values():
        delta.mul_(scale)
    return mean, {
        "donor_norms": norms, "max_norm": cap, "mean_norm": raw_norm,
        "clipped_norm": raw_norm * scale, "clip_scale": scale,
    }


def graft_norm_cap(norms, max_norm):
    cap = median(norms) if max_norm is None else max_norm
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("The graft norm cap must be finite and nonnegative")
    return cap


def clip_scale(norm, cap):
    return cap / norm if norm > cap else 1.0


class SimilarityGraft:
    """Select three clean donors using the current honest FedAvg direction."""

    def __init__(self, donor_paths, parameters, max_norm=None):
        self.parameters = dict(parameters)
        self.rounds = sorted(donor_paths, reverse=True)
        if len(self.rounds) < 3:
            raise ValueError("TCUG-2 requires at least three clean donors")
        vectors = []
        for round_idx in self.rounds:
            update = torch.load(donor_paths[round_idx], map_location="cpu", weights_only=True)
            for name, parameter in self.parameters.items():
                delta = update[name]
                if delta.shape != parameter.shape or not delta.is_floating_point():
                    raise ValueError(f"Invalid parameter delta {name} in donor round {round_idx}")
            vectors.append(self._flatten(update))
        self.vectors = torch.stack(vectors)
        norms = torch.linalg.vector_norm(self.vectors, dim=1, dtype=torch.float64)
        eligible = torch.isfinite(norms) & (norms > 0)
        if not torch.isfinite(norms).all() or eligible.sum().item() < 3:
            raise ValueError("TCUG-2 requires finite donors and at least three nonzero donor norms")
        self.donor_norms = dict(zip(self.rounds, norms.tolist()))
        self.cap = graft_norm_cap(norms[eligible].tolist(), max_norm)
        self.eligible = eligible
        self.directions = self.vectors / torch.where(eligible, norms, 1.0)[:, None]
        self.records = []

    def _flatten(self, update):
        return torch.cat([update[name].detach().float().reshape(-1) for name in self.parameters])

    def __call__(self, round_idx, client_deltas, sample_counts):
        total = sum(sample_counts)
        query = torch.zeros_like(self._flatten(client_deltas[0]))
        for delta, count in zip(client_deltas, sample_counts):
            if count:
                query.add_(self._flatten(delta), alpha=count / total)
        query = query.cpu().double()
        norm = torch.linalg.vector_norm(query).item()
        if not math.isfinite(norm):
            raise ValueError(f"Non-finite honest update at round {round_idx}")
        scores = self.directions.mv(query / norm).clamp(-1, 1) if norm else torch.zeros(len(self.rounds))
        scores = scores.masked_fill(~self.eligible, -torch.inf)
        # Donors are newest first, so a stable sort resolves ties by recency.
        selected = scores.argsort(descending=True, stable=True)[:3]
        mean = self.vectors[selected].mean(dim=0)
        raw_norm = torch.linalg.vector_norm(mean, dtype=torch.float64).item()
        scale = clip_scale(raw_norm, self.cap)
        mean.mul_(scale)
        chunks = mean.split([parameter.numel() for parameter in self.parameters.values()])
        self.records.append({
            "round": round_idx, "honest_update_norm": norm,
            "selected_donor_rounds": [self.rounds[index] for index in selected.tolist()],
            "cosine_similarities": scores[selected].tolist() if norm else None,
            "mean_norm": raw_norm, "max_norm": self.cap, "clip_scale": scale,
            "clipped_norm": raw_norm * scale,
        })
        return {name: chunk.reshape_as(parameter)
                for (name, parameter), chunk in zip(self.parameters.items(), chunks)}
