"""Conv2/BatchNorm channel patching and selection-only search utilities.

Pure selection helpers do not import Torch. Tensor helpers import it lazily.
"""

import math
import random


CHANNEL_KEYS = ("features.4.weight", "features.4.bias", "features.5.weight", "features.5.bias",
                "features.5.running_mean", "features.5.running_var")
CHANNEL_COUNT = 64


def stratified_split(labels, fraction=0.5, seed=2026):
    if not 0 < fraction < 1:
        raise ValueError("Selection fraction must lie strictly between zero and one")
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    rng = random.Random(seed)
    selection, report = [], []
    for label in sorted(groups):
        indices = list(groups[label])
        if len(indices) < 2:
            raise ValueError("Each class needs at least two images")
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, int(len(indices) * fraction)))
        selection.extend(indices[:count])
        report.extend(indices[count:])
    return sorted(selection), sorted(report)


def effects(ca, asr, baseline, epsilon_pp=1e-6):
    if not all(math.isfinite(x) and 0 <= x <= 100 for x in [ca, asr, baseline["ca"], baseline["asr"]]):
        raise ValueError("CA/ASR must be finite percentages in [0, 100]")
    if not math.isfinite(epsilon_pp) or epsilon_pp <= 0:
        raise ValueError("Score epsilon must be positive and finite")
    delta_ca, delta_asr = ca - baseline["ca"], asr - baseline["asr"]
    lost, drop = max(0.0, -delta_ca), -delta_asr
    return {"ca": ca, "asr": asr, "delta_ca_pp": delta_ca, "delta_asr_pp": delta_asr,
            "ca_lost_pp": lost, "asr_drop_pp": drop, "score": drop / max(lost, epsilon_pp)}


def choose_stage1_setting(rows):
    """Choose one donor/alpha on selection data; full-block budget need not hold.

    A full block can violate the budget while a channel subset remains useful.
    Therefore Stage 1 screens by the same drop/loss score, not feasibility alone.
    """
    if not rows:
        raise ValueError("No Stage 1 candidates")
    return min(rows, key=lambda r: (-r["score"], r["asr"], -r["ca"], -r["donor_round"], r["alpha"]))


def channel_rankings(single_rows, random_seed=314159, shuffles=10):
    if len(single_rows) != CHANNEL_COUNT or {r["channel"] for r in single_rows} != set(range(CHANNEL_COUNT)):
        raise ValueError("Expected one measurement for each of 64 channels")
    if shuffles < 1:
        raise ValueError("At least one random control is required")
    scores = sorted(single_rows, key=lambda r: (-r["score"], -r["asr_drop_pp"], r["ca_lost_pp"], r["channel"]))
    drift = sorted(single_rows, key=lambda r: (-r["weight_drift"], r["channel"]))
    result = {"single_channel_score": [r["channel"] for r in scores],
              "weight_drift": [r["channel"] for r in drift]}
    for index in range(shuffles):
        order = list(range(CHANNEL_COUNT))
        random.Random(random_seed + index).shuffle(order)
        result[f"random_{index:02d}"] = order
    return result


def choose_prefix(rows, baseline, ca_budget_pp):
    if not math.isfinite(ca_budget_pp) or not 0 <= ca_budget_pp <= 100:
        raise ValueError("CA budget must be a finite number of percentage points in [0, 100]")
    feasible = [r for r in rows if baseline["ca"] - r["ca"] <= ca_budget_pp + 1e-9]
    if not feasible:
        raise ValueError("No feasible candidate; include the unchanged k=0 baseline")
    def priority(row):
        ranking = row["ranking"]
        rank_priority = 0 if ranking == "single_channel_score" else 1 if ranking == "weight_drift" else 2
        return (row["asr"], -row["ca"], row["prefix_k"], rank_priority,
                -row["donor_round"], row["alpha"], ranking)
    return min(feasible, key=priority)


def patch_channels(final, donor, channels, alpha):
    """Blend output-channel slices only; scalar BN counters remain from final."""
    import torch
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Alpha must lie in [0, 1]")
    channels = list(channels)
    if len(set(channels)) != len(channels) or any(type(c) is not int or not 0 <= c < CHANNEL_COUNT for c in channels):
        raise ValueError("Channels must be unique integers in [0, 63]")
    if final.keys() != donor.keys():
        raise ValueError("Donor and final state schemas differ")
    for key in CHANNEL_KEYS:
        if key not in final or final[key].shape != donor[key].shape or final[key].shape[0] != CHANNEL_COUNT:
            raise ValueError(f"Unexpected Conv2/BN shape at {key}")
        if not torch.isfinite(final[key]).all() or not torch.isfinite(donor[key]).all():
            raise ValueError(f"Nonfinite component tensor at {key}")
    # Shallow-copy untouched tensors; clone before editing every patched tensor.
    result = dict(final)
    if not channels or alpha == 0:
        return result
    for key in CHANNEL_KEYS:
        index = torch.tensor(channels, dtype=torch.long, device=final[key].device)
        current = final[key].index_select(0, index)
        old = donor[key].to(device=current.device, dtype=current.dtype).index_select(0, index)
        value = old if alpha == 1 else (1 - alpha) * current + alpha * old
        result[key] = final[key].clone()
        result[key].index_copy_(0, index, value)
    return result


def weight_drift(final, donor):
    """Raw per-output-filter Conv2 L2 drift, including Conv2 bias, excluding BN."""
    import torch
    weight = final["features.4.weight"].float() - donor["features.4.weight"].float()
    bias = final["features.4.bias"].float() - donor["features.4.bias"].float()
    return torch.sqrt(weight.flatten(1).square().sum(1) + bias.square()).tolist()


def assert_only_selected_changed(before, after, channels):
    import torch
    keep = [c for c in range(CHANNEL_COUNT) if c not in channels]
    for key in before:
        if key in CHANNEL_KEYS:
            if not torch.equal(before[key][keep], after[key][keep]):
                raise AssertionError(f"Unselected channels changed: {key}")
        elif not torch.equal(before[key], after[key]):
            raise AssertionError(f"Unaffected component changed: {key}")
