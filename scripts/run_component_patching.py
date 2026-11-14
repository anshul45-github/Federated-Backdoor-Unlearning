"""Selection/report-split Conv2+BN historical patch search for checkerboard models."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.attack_report import prepare_output_directory
from src.unlearning.component_patching import (CHANNEL_COUNT, CHANNEL_KEYS, assert_only_selected_changed,
    channel_rankings, choose_prefix, choose_stage1_setting, effects, patch_channels, stratified_split, weight_drift)


METRICS = ["ca", "asr", "delta_ca_pp", "delta_asr_pp", "ca_lost_pp", "asr_drop_pp", "score"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_run", type=Path, required=True)
    p.add_argument("--output_dir", type=Path)
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--stage", choices=["1", "2", "both"], default="both")
    p.add_argument("--donor_rounds", type=int, nargs="+", default=[10, 20, 29])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--stage2_scope", choices=["best", "all"], default="best")
    p.add_argument("--stage2_donor_round", type=int)
    p.add_argument("--stage2_alpha", type=float)
    p.add_argument("--ca_budget_pp", type=float, default=1.0)
    p.add_argument("--score_epsilon_pp", type=float, default=1e-6)
    p.add_argument("--selection_fraction", type=float, default=0.5)
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--random_seed", type=int, default=314159)
    p.add_argument("--random_shuffles", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--calibration_steps", type=int, default=0)
    p.add_argument("--calibration_lr", type=float, default=0.0001)
    p.add_argument("--check_only", action="store_true")
    args = p.parse_args(argv)
    if not math.isfinite(args.ca_budget_pp) or not 0 <= args.ca_budget_pp <= 100:
        p.error("CA budget must be finite percentage points in [0, 100]")
    if not math.isfinite(args.score_epsilon_pp) or args.score_epsilon_pp <= 0:
        p.error("Score epsilon must be positive and finite")
    if not 0 < args.selection_fraction < 1 or args.batch_size < 1 or args.random_shuffles < 1:
        p.error("Invalid split fraction, batch size or shuffle count")
    if args.calibration_steps < 0 or not math.isfinite(args.calibration_lr) or args.calibration_lr <= 0:
        p.error("Invalid calibration steps or learning rate")
    if len(set(args.donor_rounds)) != len(args.donor_rounds) or len(set(args.alphas)) != len(args.alphas):
        p.error("Donor rounds and alphas must be unique")
    if any(not math.isfinite(a) or not 0 < a <= 1 for a in args.alphas):
        p.error("Alphas must lie in (0, 1]")
    if (args.stage2_donor_round is None) != (args.stage2_alpha is None):
        p.error("Supply both --stage2_donor_round and --stage2_alpha")
    if args.stage2_donor_round is not None:
        if args.stage2_donor_round not in args.donor_rounds or args.stage2_alpha not in args.alphas:
            p.error("Explicit Stage 2 donor/alpha must be in the declared grid")
        if args.stage2_scope == "all":
            p.error("Explicit Stage 2 donor/alpha cannot be combined with --stage2_scope all")
    if args.stage == "2" and args.stage2_scope == "best" and args.stage2_donor_round is None:
        p.error("Stage 2 alone requires an explicit donor/alpha, or --stage2_scope all")
    return args


def inspect_source(args):
    source = args.source_run.resolve()
    config = json.loads((source / "config.json").read_text())
    recorded = json.loads((source / "final_results.json").read_text())
    if recorded.get("mode") != "attack" or recorded.get("config") != config:
        raise ValueError("Use a completed attack folder with matching configuration records")
    if config.get("trigger_type") != "checkerboard" or config.get("trigger_size") != 4:
        raise ValueError("This experiment expects the 4x4 checkerboard attack")
    paths = {"final": source / "final_model.pt"}
    for r in args.donor_rounds:
        if not 1 <= r < config["attack_start_round"] or r >= config["num_rounds"]:
            raise ValueError(f"G{r} must be before attack activation and before the final checkpoint")
        paths[f"G{r}"] = source / "trajectory" / f"round_{r:03d}" / "global_model.pt"
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.calibration_steps and not (source / "client_partitions.pt").is_file():
        raise FileNotFoundError("Clean calibration requires source client_partitions.pt to exclude Client 0")
    return source, config, recorded, paths


def load_test_cache(torch, data_dir):
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = datasets.CIFAR10(str(data_dir), train=False, download=False, transform=transform)
    images = torch.empty(len(dataset), 3, 32, 32)
    for index in range(len(dataset)):
        images[index] = dataset[index][0]
    return images, torch.tensor(dataset.targets, dtype=torch.long)


def evaluate_indices(torch, model, images, labels, indices, config, device, batch_size):
    """Only indices explicitly supplied by the caller are ever accessed."""
    from src.data.backdoor import PoisonedDataset
    template = PoisonedDataset([], [], config["target_class"], config["trigger_size"], config["trigger_type"])
    patch = template.trigger_patch.to(device)
    correct, target_predictions, count, triggered_count = 0, 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            x = images[batch_indices].to(device)
            y = labels[batch_indices].to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            count += len(y)
            non_target = y != config["target_class"]
            triggered = x[non_target].clone()
            if len(triggered):
                size = config["trigger_size"]
                triggered[:, :, -size:, -size:] = patch
                target_predictions += (model(triggered).argmax(1) == config["target_class"]).sum().item()
                triggered_count += len(triggered)
    if not count or not triggered_count:
        raise ValueError("Evaluation requires clean and non-target examples")
    return {"ca": 100 * correct / count, "asr": 100 * target_predictions / triggered_count,
            "clean_count": count, "triggered_count": triggered_count}


def csv_writer(path, fields):
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return handle, writer


def setting_id(donor_round, alpha):
    return f"G{donor_round}_alpha{alpha:g}"


def search(torch, model, final, donors, baseline, evaluate_selection, args, output):
    cache = {}
    def measure(r, alpha, channels):
        key = (r, alpha, tuple(sorted(channels)))
        if not channels:
            return baseline
        if key not in cache:
            model.load_state_dict(patch_channels(final, donors[r], channels, alpha))
            cache[key] = evaluate_selection(model)
        return cache[key]
    stage1_rows = []
    if args.stage in {"1", "both"}:
        handle, writer = csv_writer(output / "stage1.csv", ["donor_round", "alpha", "channel_count", *METRICS])
        try:
            for r in args.donor_rounds:
                for alpha in args.alphas:
                    measured = measure(r, alpha, list(range(CHANNEL_COUNT)))
                    row = {"donor_round": r, "alpha": alpha, "channel_count": CHANNEL_COUNT,
                           **effects(measured["ca"], measured["asr"], baseline, args.score_epsilon_pp)}
                    stage1_rows.append(row)
                    writer.writerow(row)
                    handle.flush()
                    print(f"Stage 1 G{r} alpha={alpha:g}: CA={row['ca']:.2f} ASR={row['asr']:.2f} "
                          f"delta_CA={row['delta_ca_pp']:+.2f} pp delta_ASR={row['delta_asr_pp']:+.2f} pp", flush=True)
        finally:
            handle.close()
    if args.stage == "1":
        return None, None
    if args.stage2_scope == "all":
        settings = [(r, a) for r in args.donor_rounds for a in args.alphas]
    elif args.stage2_donor_round is not None:
        settings = [(args.stage2_donor_round, args.stage2_alpha)]
    else:
        chosen = choose_stage1_setting(stage1_rows)
        settings = [(chosen["donor_round"], chosen["alpha"])]
    save_json(output / "stage2_settings.json", {"scope": args.stage2_scope, "settings": settings,
        "selection_rule": "explicit/all grid, otherwise maximal Stage 1 ASR-drop / clamped-CA-loss score"})
    singles_handle, singles_writer = csv_writer(output / "single_channel_scores.csv",
        ["donor_round", "alpha", "channel", "weight_drift", *METRICS])
    curves_handle, curves_writer = csv_writer(output / "channel_curves.csv",
        ["donor_round", "alpha", "ranking", "prefix_k", "channels_json", "within_ca_budget", *METRICS])
    curves, orders, best_per_order = [], {}, []
    try:
        for r, alpha in settings:
            singles = []
            drift = weight_drift(final, donors[r])
            for channel in range(CHANNEL_COUNT):
                measured = measure(r, alpha, [channel])
                row = {"donor_round": r, "alpha": alpha, "channel": channel, "weight_drift": drift[channel],
                       **effects(measured["ca"], measured["asr"], baseline, args.score_epsilon_pp)}
                singles.append(row)
                singles_writer.writerow(row)
                singles_handle.flush()
                print(f"Single G{r} alpha={alpha:g} channel={channel}: CA={row['ca']:.2f} ASR={row['asr']:.2f} score={row['score']:.4g}", flush=True)
            rankings = channel_rankings(singles, args.random_seed, args.random_shuffles)
            orders[setting_id(r, alpha)] = rankings
            save_json(output / "ranking_orders.json", orders)
            for name, order in rankings.items():
                ranking_curves = []
                for k in range(CHANNEL_COUNT + 1):
                    channels = order[:k]
                    measured = measure(r, alpha, channels)
                    row = {"donor_round": r, "alpha": alpha, "ranking": name, "prefix_k": k,
                           "channels_json": json.dumps(channels),
                           "within_ca_budget": baseline["ca"] - measured["ca"] <= args.ca_budget_pp + 1e-9,
                           **effects(measured["ca"], measured["asr"], baseline, args.score_epsilon_pp)}
                    curves.append(row)
                    ranking_curves.append(row)
                    curves_writer.writerow(row)
                    curves_handle.flush()
                    if k % 16 == 0:
                        print(f"Prefix G{r} alpha={alpha:g} {name} k={k}: CA={row['ca']:.2f} ASR={row['asr']:.2f}", flush=True)
                best_per_order.append(choose_prefix(ranking_curves, baseline, args.ca_budget_pp))
    finally:
        singles_handle.close()
        curves_handle.close()
    save_json(output / "best_prefix_per_order.json", best_per_order)
    selected = choose_prefix(curves, baseline, args.ca_budget_pp)
    selected["channels"] = json.loads(selected["channels_json"])
    return selected, curves


def calibrate(torch, model, before, selected_channels, source, config, args, device):
    """Clean retained-client training only; BN stats and all other slices frozen."""
    from torchvision import datasets, transforms
    from src.utils.seed import set_seed
    set_seed(config["seed"] + 100000)
    saved = torch.load(source / "client_partitions.pt", map_location="cpu", weights_only=False)
    if not isinstance(saved, list):
        saved = saved.get("partitions", saved.get("client_partitions"))
    if not isinstance(saved, list) or len(saved) != config["num_clients"]:
        raise ValueError("Unexpected partition serialization")
    indices = [int(i) for c, part in enumerate(saved) if c != config["malicious_client_id"] for i in part.indices]
    forgotten = set(int(i) for i in saved[config["malicious_client_id"]].indices)
    if len(set(indices)) != len(indices) or forgotten.intersection(indices):
        raise ValueError("Calibration client partitions overlap")
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train = datasets.CIFAR10(str(args.data_dir), train=True, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train, indices), batch_size=64, shuffle=True,
                                         generator=torch.Generator().manual_seed(config["seed"] + 100000))
    model.load_state_dict(before)
    model.eval()  # Freeze all BN running statistics and disable Dropout.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hooks, trainable = [], []
    for name, parameter in model.named_parameters():
        if name in CHANNEL_KEYS:
            parameter.requires_grad_(True)
            mask = torch.zeros_like(parameter)
            mask[selected_channels] = 1
            hooks.append(parameter.register_hook(lambda gradient, mask=mask: gradient * mask))
            trainable.append(parameter)
    optimizer = torch.optim.SGD(trainable, lr=args.calibration_lr, momentum=0, weight_decay=0)
    iterator = iter(loader)
    try:
        for step in range(args.calibration_steps):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, labels = next(iterator)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
    finally:
        for hook in hooks:
            hook.remove()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert_only_selected_changed(before, after, selected_channels)
    # BN running statistics are immutable even for selected channels during calibration.
    for key in CHANNEL_KEYS:
        if "running_" in key and not torch.equal(before[key], after[key]):
            raise AssertionError(f"Calibration changed BN statistics: {key}")
    return after


def plot_curves(curves, selected, baseline, budget, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = sorted({(r["donor_round"], r["alpha"]) for r in curves})
    for donor, alpha in settings:
        subset = [r for r in curves if r["donor_round"] == donor and r["alpha"] == alpha]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, metric, ylabel in zip(axes, ["ca", "asr"], ["Clean accuracy (%)", "Attack success rate (%)"]):
            for ranking in ["single_channel_score", "weight_drift"]:
                points = sorted((r for r in subset if r["ranking"] == ranking), key=lambda r: r["prefix_k"])
                ax.plot([r["prefix_k"] for r in points], [r[metric] for r in points], label=ranking.replace("_", " "))
            randoms = [r for r in subset if r["ranking"].startswith("random_")]
            means, spreads = [], []
            for k in range(CHANNEL_COUNT + 1):
                values = [r[metric] for r in randoms if r["prefix_k"] == k]
                means.append(statistics.mean(values))
                spreads.append(statistics.pstdev(values))
            x = list(range(CHANNEL_COUNT + 1))
            ax.plot(x, means, color="gray", linestyle="--", label="random mean")
            ax.fill_between(x, [max(0, m - s) for m, s in zip(means, spreads)],
                            [min(100, m + s) for m, s in zip(means, spreads)], color="gray", alpha=.2, label="random ±1 SD")
            if metric == "ca":
                ax.axhline(baseline["ca"] - budget, color="red", linestyle=":", label="CA budget floor")
            else:
                ax.set_ylim(0, 100)
            if (donor, alpha) == (selected["donor_round"], selected["alpha"]):
                ax.scatter([selected["prefix_k"]], [selected[metric]], marker="*", s=130, color="black", label="selected pre-calibration")
            ax.set(xlabel="Number of patched channels", ylabel=ylabel)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"Selection split only: G{donor}, alpha={alpha:g}")
        fig.tight_layout()
        path = output / f"channel_curves_{setting_id(donor, alpha)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        if (donor, alpha) == (selected["donor_round"], selected["alpha"]):
            shutil.copyfile(path, output / "channel_curves.png")


def main(argv=None):
    args = parse_args(argv)
    source, config, recorded, paths = inspect_source(args)
    inputs = {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}
    if args.calibration_steps:
        partition_path = source / "client_partitions.pt"
        inputs["calibration_partitions"] = {"path": str(partition_path), "sha256": sha256(partition_path)}
    if args.check_only:
        print(json.dumps({"source": str(source), "final_round": config["num_rounds"], "inputs": inputs}, indent=2))
        return
    import torch
    from src.models.small_cnn import SmallCNN
    from src.utils.seed import set_seed
    if args.device == "auto":
        device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    requested = args.output_dir or source / "component_patching"
    if requested.resolve() == source or requested.resolve() in source.parents:
        raise ValueError("Output must not replace the attacked run or its parent")
    output = Path(prepare_output_directory(requested, unique_on_conflict=True))
    print(f"Output: {output.resolve()}", flush=True)
    manifest = {"status": "started", "arguments": vars(args), "source_config": config, "inputs": inputs,
                "source_full_test_results": recorded,
                "device": str(device), "torch_version": str(torch.__version__),
                "selection_rule": "minimum selection ASR within CA budget across tested prefixes, including random controls; k=0 fallback",
                "report_rule": "report split evaluated only after final model selection; no held-out reselection"}
    save_json(output / "manifest.json", manifest)
    set_seed(config["seed"])
    final = torch.load(paths["final"], map_location="cpu", weights_only=True)
    donors = {r: torch.load(paths[f"G{r}"], map_location="cpu", weights_only=True) for r in args.donor_rounds}
    model = SmallCNN().to(device)
    model.load_state_dict(final)
    images, labels = load_test_cache(torch, args.data_dir)
    if len(labels) != 10000 or set(labels.tolist()) != set(range(10)):
        raise ValueError("Expected CIFAR-10's 10,000-image ten-class test set")
    selection, report = stratified_split(labels.tolist(), args.selection_fraction, args.split_seed)
    save_json(output / "split_indices.json", {"selection": selection, "report": report,
        "split_seed": args.split_seed, "selection_fraction": args.selection_fraction,
        "test_labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
        "normalized_test_pixels_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest()})
    def selection_eval(candidate):
        return evaluate_indices(torch, candidate, images, labels, selection, config, device, args.batch_size)
    baseline = selection_eval(model)
    save_json(output / "selection_baseline.json", baseline)
    print(f"Selection baseline: CA={baseline['ca']:.2f}% ASR={baseline['asr']:.2f}%", flush=True)
    selected, curves = search(torch, model, final, donors, baseline, selection_eval, args, output)
    if selected is None:
        manifest["status"] = "stage1_completed"
        save_json(output / "manifest.json", manifest)
        print(f"Stage 1 CSV: {output / 'stage1.csv'}")
        return output
    best = patch_channels(final, donors[selected["donor_round"]], selected["channels"], selected["alpha"])
    assert_only_selected_changed(final, best, selected["channels"])
    calibration = {"requested_steps": args.calibration_steps, "accepted": False}
    final_selection = {"ca": selected["ca"], "asr": selected["asr"]}
    if args.calibration_steps and selected["channels"]:
        torch.save(best, output / "best_patch_uncalibrated.pt")
        after = calibrate(torch, model, best, selected["channels"], source, config, args, device)
        model.load_state_dict(after)
        measured = selection_eval(model)
        calibration["selection_metrics"] = measured
        # A fixed optional post-selection proposal; accept only on selection data.
        if (baseline["ca"] - measured["ca"] <= args.ca_budget_pp + 1e-9 and
                (measured["asr"], -measured["ca"]) < (selected["asr"], -selected["ca"])):
            best, final_selection = after, measured
            calibration["accepted"] = True
    torch.save(best, output / "best_patch_model.pt")
    save_json(output / "best_patch_channels.json", {"source_model": inputs["final"],
        "donor": inputs[f"G{selected['donor_round']}"], "donor_round": selected["donor_round"], "alpha": selected["alpha"],
        "ranking": selected["ranking"], "channels": selected["channels"], "prefix_k": selected["prefix_k"],
        "selection_baseline": baseline, "selection_pre_calibration": selected,
        "selection_final": final_selection, "ca_budget_pp": args.ca_budget_pp, "calibration": calibration,
        "bn_counter_policy": "retain final counter; interpolate selected running mean/variance; freeze stats during calibration",
        "best_model_sha256": sha256(output / "best_patch_model.pt")})
    # No report-split evaluation occurs anywhere above this line.
    model.load_state_dict(final)
    baseline_report = evaluate_indices(torch, model, images, labels, report, config, device, args.batch_size)
    model.load_state_dict(best)
    best_report = evaluate_indices(torch, model, images, labels, report, config, device, args.batch_size)
    held_out = {"baseline": baseline_report, "best_patch": best_report,
                "effects": effects(best_report["ca"], best_report["asr"], baseline_report, args.score_epsilon_pp),
                "within_report_ca_budget": baseline_report["ca"] - best_report["ca"] <= args.ca_budget_pp + 1e-9,
                "no_reselection": True}
    save_json(output / "held_out_results.json", held_out)
    plot_curves(curves, selected, baseline, args.ca_budget_pp, output)
    text = ["# Conv2 + BatchNorm historical patch search", "", f"Source: `{source}` (G{config['num_rounds']}).",
        f"Selected: G{selected['donor_round']}, alpha={selected['alpha']}, ranking={selected['ranking']}, channels={selected['channels']}.",
        f"Selection CA-loss budget: {args.ca_budget_pp} percentage points. Calibration accepted: {calibration['accepted']}.", "",
        "| Split/model | CA % | ASR % |", "|---|---:|---:|",
        f"| Selection / original | {baseline['ca']:.2f} | {baseline['asr']:.2f} |",
        f"| Selection / chosen patch | {final_selection['ca']:.2f} | {final_selection['asr']:.2f} |",
        f"| Held-out report / original | {baseline_report['ca']:.2f} | {baseline_report['asr']:.2f} |",
        f"| Held-out report / chosen patch | {best_report['ca']:.2f} | {best_report['asr']:.2f} |", "",
        "Channel scores, rankings, and prefixes use the selection split only. The report split is not used for choosing or revising the patch.",
        "Random controls participate in selection; a random ranking can win. This search does not by itself establish superiority of score ranking.",
        "These historical donors contain Client 0's pre-attack clean contributions. This is selective backdoor repair, not certified client-level unlearning.", ""]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    manifest["status"] = "completed"
    manifest["selected_model_sha256"] = sha256(output / "best_patch_model.pt")
    save_json(output / "manifest.json", manifest)
    print(f"Held-out patch: CA={best_report['ca']:.2f}% ASR={best_report['asr']:.2f}%; report: {output / 'report.md'}", flush=True)
    return output


if __name__ == "__main__":
    main()
