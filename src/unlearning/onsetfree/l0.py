"""Stored-update L0 estimators. No attack metadata enters this module."""

from collections import defaultdict

import torch


def project_out(vector, basis):
    """Return (I - V V^T) vector without constructing the projector."""
    if basis is None or basis.numel() == 0:
        return vector.clone()
    return vector - basis @ (basis.T @ vector)


def right_basis(rows, rank, seed=42):
    """Uncentered right singular basis of stacked client updates."""
    if rank <= 0 or not rows:
        return None
    matrix = torch.stack([row.reshape(-1).float() for row in rows])
    q = min(rank + 8, *matrix.shape)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        _, _, v = torch.svd_lowrank(matrix, q=q, niter=4)
    return v[:, : min(rank, v.shape[1])]


def random_basis(width, rank, seed=42):
    if rank <= 0:
        return None
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(width, min(width, rank), generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q


def contrasts_and_rows(trajectory, malicious_client, parameter_keys):
    """Accumulate contrast/C1 and rows, requiring one pass over saved rounds."""
    contrasts = defaultdict(list)
    benign_rows = defaultdict(list)
    malicious_rows = defaultdict(list)
    attacker_sum = {}
    final_state = None
    rounds = 0
    for record in trajectory:
        if malicious_client not in record.weights:
            raise ValueError("Malicious client absent from trajectory")
        pm = record.weights[malicious_client]
        if not 0 < pm < 1:
            raise ValueError("Malicious client weight must lie between zero and one")
        rounds += 1
        final_state = record.theta_next
        for key in parameter_keys:
            um = record.deltas[malicious_client][key].reshape(-1).float()
            benign = torch.zeros_like(um)
            for client, weight in record.weights.items():
                if client == malicious_client:
                    continue
                ui = record.deltas[client][key].reshape(-1).float()
                benign += (weight / (1 - pm)) * ui
                benign_rows[key].append(ui)
            contrasts[key].append((pm, um - benign))
            malicious_rows[key].append(um)
            attacker_sum[key] = attacker_sum.get(key, torch.zeros_like(um)) + pm * um
    if not rounds:
        raise ValueError("Empty trajectory")
    return final_state, contrasts, benign_rows, malicious_rows, attacker_sum


def make_updates(contrasts, benign_rows, malicious_rows, attacker_sum, rank=32, seed=42):
    """Return C1-C4 and the L0 projection diagnostics for trainable keys."""
    updates = {name: {} for name in ("c1", "c2", "l0", "c3_random", "c4_malicious")}
    inside_energy = {}
    for key, round_contrasts in contrasts.items():
        total = sum((pm * e for pm, e in round_contrasts), torch.zeros_like(round_contrasts[0][1]))
        v = right_basis(benign_rows[key], rank, seed)
        vr = random_basis(total.numel(), rank, seed)
        vm = right_basis(malicious_rows[key], rank, seed)
        updates["c1"][key] = -attacker_sum[key]
        updates["c2"][key] = -total
        updates["l0"][key] = -project_out(total, v)
        updates["c3_random"][key] = -project_out(total, vr)
        updates["c4_malicious"][key] = -project_out(total, vm)
        denom = torch.dot(total, total).item()
        inside = total - project_out(total, v)
        inside_energy[key] = torch.dot(inside, inside).item() / denom if denom else 0.0
    return updates, inside_energy


def apply_update(final_state, update, scale=1.0):
    """Change parameters only; preserve all BN running buffers and counters."""
    result = {key: value.clone() for key, value in final_state.items()}
    for key, delta in update.items():
        result[key] = (result[key].float() + scale * delta.reshape(result[key].shape)).to(result[key].dtype)
    return result
