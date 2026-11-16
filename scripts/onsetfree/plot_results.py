"""Render measured onset-free diagnostics from JSON; skip unavailable panels."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def plot_energy(root, setting, out):
    data = read(root / "l0" / setting / "metadata.json")
    if not data:
        return
    energy = data["inside_benign_subspace_energy_fraction"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(energy)), list(energy.values()), color="#3478ac")
    ax.set_xticks(range(len(energy)), [key.replace(".weight", "").replace(".bias", " b") for key in energy], rotation=50, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction in benign-update subspace")
    ax.set_title(f"{setting.upper()} L0, rank {data['rank']}")
    save(fig, out / f"l0_energy_{setting}.png")


def plot_cosine(root, setting, out):
    l0 = read(root / "l0" / setting / "cosine.json")
    l1 = read(root / "diagnostics" / f"{setting}_l1_cosine.json")
    if not l0:
        return
    methods = {key: l0["methods"][key]["per_layer_cosine"] for key in ("c1", "l0") if key in l0["methods"]}
    if l1 and "l1" in l1["methods"]:
        methods["l1"] = l1["methods"]["l1"]["per_layer_cosine"]
    if not methods:
        return
    layers = list(next(iter(methods.values())))
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(methods)
    for i, (method, values) in enumerate(methods.items()):
        ax.bar([j - 0.4 + width * (i + 0.5) for j in range(len(layers))], [values[layer] or 0 for layer in layers], width, label=method)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(range(len(layers)), [key.replace(".weight", "").replace(".bias", " b") for key in layers], rotation=50, ha="right")
    ax.set_ylabel("Cosine to counterfactual displacement")
    ax.set_title(f"{setting.upper()} evaluation-only diagnostic")
    ax.legend()
    save(fig, out / f"cosine_{setting}.png")


def plot_spectrum(root, setting, out):
    data = read(root / "l1" / setting / "metadata.json")
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, len(data["eigenvalues"]) + 1), data["eigenvalues"], marker=".")
    axes[0].set_title("GGN eigenvalues")
    axes[1].plot(range(1, len(data["contractions"]) + 1), data["contractions"], marker=".")
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title("Momentum-aware contraction")
    for ax in axes:
        ax.set_xlabel("Rank")
    fig.suptitle(f"{setting.upper()} L1 spectrum, {data['contractions_below_half']} contractions <0.5")
    save(fig, out / f"l1_spectrum_{setting}.png")


def plot_l2(root, setting, out):
    records = []
    for path in (root / "l2" / setting).glob("bc*_w*/evaluation.json"):
        data = read(path)
        if data and "l2" in data["results"]:
            name = path.parent.name
            bc = int(name.split("_w")[0][2:])
            window = int(name.split("_w")[1])
            records.append((bc, window, data["results"]["l2"]))
    if not records:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for bc in sorted({row[0] for row in records}):
        rows = sorted((row for row in records if row[0] == bc), key=lambda row: row[1])
        axes[0].plot([row[1] for row in rows], [row[2]["ca_percent"] for row in rows], marker="o", label=f"Bc={bc}")
        axes[1].plot([row[1] for row in rows], [row[2]["asr_percent"] for row in rows], marker="o", label=f"Bc={bc}")
    for ax, label in zip(axes, ("Clean accuracy (%)", "ASR (%)")):
        ax.set_xlabel("Window W")
        ax.set_ylabel(label)
        ax.legend()
    fig.suptitle(f"{setting.upper()} L2 Taylor window sweep")
    save(fig, out / f"l2_window_{setting}.png")


def plot_tradeoff(root, setting, out):
    cost = read(root / f"{setting}_cost.json")
    l0 = read(root / "l0" / setting / "evaluation.json")
    if not cost or not l0:
        return
    points = []
    for name in ("c1", "l0"):
        value = l0["results"].get(name)
        if value:
            points.append((name, 0, value))
    for ecal in (1, 2):
        value = read(root / f"federaser_{setting}_ecal{ecal}" / "results.json")
        if value:
            points.append((f"FedEraser {ecal}", cost[f"federaser_ecal{ecal}_client_flops"], value))
    for variant in ("default", "half_scale"):
        value = read(root / "l3" / setting / variant / "results.json")
        if value:
            points.append((f"L3 {variant}", value["client_flops"], value))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, flops, value in points:
        x = flops / 1e12
        axes[0].scatter(x, value["ca_percent"])
        axes[1].scatter(x, value["asr_percent"])
        axes[0].annotate(name, (x, value["ca_percent"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axes[1].annotate(name, (x, value["asr_percent"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for ax, label in zip(axes, ("Clean accuracy (%)", "ASR (%)")):
        ax.set_xlabel("Additional client FLOPs (trillion)")
        ax.set_ylabel(label)
    fig.suptitle(f"{setting.upper()} measured cost and test outcomes")
    save(fig, out / f"ca_asr_cost_{setting}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/onsetfree")
    parser.add_argument("--setting", choices=("iid", "noniid"), required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "plots"
    for function in (plot_energy, plot_cosine, plot_spectrum, plot_l2, plot_tradeoff):
        function(root, args.setting, out)


if __name__ == "__main__":
    main()
