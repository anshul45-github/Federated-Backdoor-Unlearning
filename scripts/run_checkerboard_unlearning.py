"""Subtraction, KD and corrected FedEraser for one completed checkerboard run.

The old white-patch launchers and artifacts are left intact. Torch is imported
only after input validation, so --check_only works without training dependencies.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.attack_report import prepare_output_directory


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", type=Path, required=True, help="Exact completed checkerboard attack folder")
    parser.add_argument("--method", choices=["subtraction", "kd", "federaser"], required=True)
    parser.add_argument("--output_dir", type=Path, help="Default <source_run>/unlearning/<method>")
    parser.add_argument("--subtraction_dir", type=Path, help="KD input; defaults to <source_run>/unlearning/subtraction")
    parser.add_argument("--calibration_epochs", choices=[1, 2], type=int, default=2)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--proxy_indices", type=Path, help="Optional fixed 5,000-image balanced CIFAR-100 training index list")
    parser.add_argument("--reference_model", type=Path, help="Optional existing M_CF, used only for evaluation")
    parser.add_argument("--check_only", action="store_true", help="Validate run/provenance/history; do not import Torch or train")
    return parser.parse_args(argv)


def validate_inputs(args):
    source = args.source_run.resolve()
    config = read_json(source / "config.json")
    final = read_json(source / "final_results.json")
    if final.get("mode") != "attack":
        raise ValueError("Source must be a completed attacked run")
    if config.get("trigger_type") != "checkerboard" or config.get("trigger_size") != 4:
        raise ValueError("Source must use the 4x4 checkerboard trigger")
    if config.get("partition_type") not in {"iid", "noniid"}:
        raise ValueError("Unsupported source partition")
    if config.get("num_clients") != 10 or config.get("clients_per_round") != 10 or config.get("malicious_client_id") != 0:
        raise ValueError("This launcher supports the current ten-client/full-participation/Client-0 protocol")
    rounds = config["num_rounds"]
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("Source rounds must be positive")
    for name in ["final_model.pt", "initial_model.pt", "client_class_histograms.json"]:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    # Require consistent source records rather than silently mixing a config and endpoint.
    if final.get("config") != config:
        raise ValueError("Source final_results.json and config.json disagree")
    hist = read_json(source / "client_class_histograms.json")
    counts = [hist[f"client_{c}"]["num_samples"] for c in range(10)]
    if any(not isinstance(n, int) or n < 0 for n in counts) or sum(counts) != 50000:
        raise ValueError("Expected nonnegative per-client sample counts summing to 50,000")
    if counts[0] == 0 or sum(counts[1:]) == 0:
        raise ValueError("Unlearning requires data in both Client 0 and the retained clients")
    source_hash = sha256(source / "final_model.pt")
    required_clients = [0] if args.method == "subtraction" else list(range(1, 10)) if args.method == "federaser" else []
    missing = [source / "trajectory" / f"round_{r:03d}" / f"delta_client_{c:02d}.pt"
               for r in range(1, rounds + 1) for c in required_clients
               if not (source / "trajectory" / f"round_{r:03d}" / f"delta_client_{c:02d}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required historical updates, first: {missing[0]}. Use a --log_trajectory attacked run.")
    if args.method == "federaser" and not (source / "client_partitions.pt").is_file():
        raise FileNotFoundError(source / "client_partitions.pt")
    sub = (args.subtraction_dir or source / "unlearning/subtraction").resolve()
    if args.method == "kd":
        metadata = read_json(sub / "run_config.json")
        if metadata.get("method") != "subtraction" or metadata.get("source_model_sha256") != source_hash:
            raise ValueError("KD must use subtraction from this exact attacked checkpoint")
        if metadata.get("output_model_sha256") != sha256(sub / "final_model.pt"):
            raise ValueError("Subtraction checkpoint no longer matches its recorded checksum")
        if args.proxy_indices is not None and not args.proxy_indices.is_file():
            raise FileNotFoundError(args.proxy_indices)
    if args.reference_model is not None:
        if not args.reference_model.is_file():
            raise FileNotFoundError(args.reference_model)
        # Existing reference weights are reusable only for the matching training setup.
        ref_config = read_json(args.reference_model.parent / "config.json")
        for field in ["seed", "partition_type", "num_clients", "num_rounds", "local_epochs", "learning_rate", "momentum", "batch_size"]:
            if ref_config.get(field) != config.get(field):
                raise ValueError(f"Reference training configuration differs on {field}")
        if config["partition_type"] == "noniid" and ref_config.get("dirichlet_alpha") != config["dirichlet_alpha"]:
            raise ValueError("Reference Non-IID alpha differs")
        if not (args.reference_model.parent / "initial_model.pt").is_file():
            raise FileNotFoundError("Reference initial_model.pt is required to verify matching initialization")
    return source, config, final, counts, source_hash, sub


def subtract(torch, source, counts, rounds):
    state = copy.deepcopy(torch.load(source / "final_model.pt", map_location="cpu", weights_only=True))
    weight = counts[0] / sum(counts)
    for r in range(1, rounds + 1):
        delta = torch.load(source / "trajectory" / f"round_{r:03d}" / "delta_client_00.pt", map_location="cpu", weights_only=True)
        for key in state:
            if state[key].is_floating_point():
                state[key] = state[key] - weight * delta[key]
    return state


def aggregate_reconstruction(global_state, deltas, counts):
    from src.training.fedavg import aggregate_fedavg
    # The corrected v5 policy preserves integer BatchNorm trackers globally.
    without_counters = [{k: v for k, v in delta.items() if "num_batches_tracked" not in k} for delta in deltas]
    return aggregate_fedavg(global_state, without_counters, counts)


def reconstruct(torch, source, config, model, device, args):
    from src.data.cifar10 import get_cifar10_dataloaders
    from src.data.fl_partition import create_iid_partitions, create_noniid_partitions
    from src.models.small_cnn import SmallCNN
    from src.training.train_utils import train_epoch
    from src.training.fedavg import compute_model_delta
    from src.unlearning.federaser import calibrate_update
    from src.utils.seed import set_seed
    train, _ = get_cifar10_dataloaders(str(args.data_dir), config["batch_size"])
    saved = torch.load(source / "client_partitions.pt", map_location="cpu", weights_only=False)
    if not isinstance(saved, list):
        saved = saved.get("partitions", saved.get("client_partitions"))
    if config["partition_type"] == "iid":
        regenerated, _ = create_iid_partitions(train.dataset, 10, config["seed"])
    else:
        regenerated, _ = create_noniid_partitions(train.dataset, 10, config["dirichlet_alpha"], config["seed"])
    for c in range(10):
        if sorted(saved[c].indices) != sorted(regenerated[c].indices):
            raise ValueError(f"Saved partition differs for client {c}")
    model.load_state_dict(torch.load(source / "initial_model.pt", map_location=device, weights_only=True))
    initial_counters = {k: v.clone() for k, v in model.state_dict().items() if "num_batches_tracked" in k}
    for r in range(1, config["num_rounds"] + 1):
        deltas = []
        for c in range(1, 10):
            if len(saved[c]) == 0:
                # Empty retained clients have zero weight and no calibration data.
                # Skip their loader without resampling or redistributing the split.
                continue
            historical = torch.load(source / "trajectory" / f"round_{r:03d}" / f"delta_client_{c:02d}.pt",
                                    map_location=device, weights_only=True)
            if r == 1:
                calibrated = historical
            else:
                seed = hash((config["seed"], r, c)) % (2**32)
                set_seed(seed)
                loader = torch.utils.data.DataLoader(saved[c], batch_size=config["batch_size"], shuffle=True,
                                                     generator=torch.Generator().manual_seed(seed))
                local = SmallCNN().to(device)
                local.load_state_dict(model.state_dict())
                optimizer = torch.optim.SGD(local.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=0)
                for _ in range(args.calibration_epochs):
                    train_epoch(local, loader, torch.nn.CrossEntropyLoss(), optimizer, device)
                calibrated = calibrate_update(historical, compute_model_delta(model.state_dict(), local.state_dict()))
            deltas.append(calibrated)
        model.load_state_dict(aggregate_reconstruction(model.state_dict(), deltas, [len(saved[c]) for c in range(1, 10) if len(saved[c]) > 0]))
        for key, value in initial_counters.items():
            if not torch.equal(model.state_dict()[key], value):
                raise AssertionError(f"BatchNorm counter changed: {key}")
        print(f"FedEraser: reconstructed round {r}/{config['num_rounds']}", flush=True)


def distill(torch, source, subtraction, config, model, device, args, output):
    import numpy as np
    from torchvision import datasets, transforms
    from src.models.small_cnn import SmallCNN
    from src.utils.seed import set_seed
    set_seed(config["seed"])
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    proxy = datasets.CIFAR100(str(args.data_dir), train=True, download=False,
                              transform=transforms.Compose([transforms.ToTensor(), normalization]))
    labels = np.asarray(proxy.targets)
    if args.proxy_indices:
        indices = read_json(args.proxy_indices)
    else:
        indices = []
        for cls in range(100):
            candidates = np.flatnonzero(labels == cls)
            np.random.shuffle(candidates)
            indices.extend(candidates[:50].tolist())
        indices.sort()
    if (len(indices) != 5000 or len(set(indices)) != 5000 or
            any(not isinstance(i, int) or not 0 <= i < len(proxy) for i in indices) or
            any(int((labels[indices] == cls).sum()) != 50 for cls in range(100))):
        raise ValueError("Proxy must contain 5,000 unique CIFAR-100 train indices, 50 per class")
    (output / "proxy_indices.json").write_text(json.dumps(indices), encoding="utf-8")
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(proxy, indices), batch_size=64, shuffle=True,
                                         generator=torch.Generator().manual_seed(config["seed"]))
    teacher = SmallCNN().to(device)
    teacher.load_state_dict(torch.load(source / "final_model.pt", map_location=device, weights_only=True))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model.load_state_dict(torch.load(subtraction / "final_model.pt", map_location=device, weights_only=True))
    before = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches_tracked" in k}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0)
    losses = []
    for epoch in range(1, 6):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        total_loss, batches = 0.0, 0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                target = teacher(images)
            student = model(images)
            loss = 16 * torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student / 4, dim=1),
                                                   torch.nn.functional.softmax(target / 4, dim=1), reduction="batchmean")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        losses.append(total_loss / batches)
        print(f"KD epoch {epoch}/5, loss {losses[-1]:.6f}", flush=True)
    for key, value in before.items():
        if not torch.equal(model.state_dict()[key], value):
            raise AssertionError(f"Frozen BatchNorm buffer changed: {key}")
    return {"epochs": 5, "temperature": 4, "learning_rate": 0.001, "momentum": 0.9, "batch_size": 64,
            "loss_history": losses, "proxy_pixel_sha256_uint8_nhwc": hashlib.sha256(proxy.data[indices].tobytes()).hexdigest()}


def evaluate(torch, model, config, args, device):
    from src.data.cifar10 import get_cifar10_dataloaders
    from src.data.backdoor import create_asr_dataloader
    from src.models.small_cnn import SmallCNN
    from src.training.train_utils import evaluate_extended
    _, clean = get_cifar10_dataloaders(str(args.data_dir), batch_size=config["batch_size"])
    triggered = create_asr_dataloader(clean, config["target_class"], config["trigger_size"], config["batch_size"], config["trigger_type"])
    if len(clean.dataset) != 10000 or len(triggered.dataset) != 9000:
        raise ValueError("Expected 10,000 clean and 9,000 triggered non-target test images")
    criterion = torch.nn.CrossEntropyLoss()
    clean_metrics = evaluate_extended(model, clean, criterion, device)
    asr = evaluate_extended(model, triggered, criterion, device)["accuracy"]
    target = config["target_class"]
    utr = 100 * sum(row[target] for i, row in enumerate(clean_metrics["confusion_matrix"]) if i != target) / 9000
    result = {"clean_accuracy": clean_metrics["accuracy"], "attack_success_rate": asr, "triggered_rate": asr,
              "untriggered_target_rate": utr, "macro_f1_pct": 100 * clean_metrics["macro_f1"],
              "clean_loss": clean_metrics["loss"], "per_class": clean_metrics["per_class"],
              "clean_population": 10000, "triggered_population": 9000,
              "trigger_type": config["trigger_type"], "trigger_size": config["trigger_size"]}
    if args.reference_model is not None:
        reference = SmallCNN().to(device)
        reference.load_state_dict(torch.load(args.reference_model, map_location=device, weights_only=True))
        reference.eval()
        model.eval()
        result["reference_metrics"] = {"clean_accuracy": evaluate_extended(reference, clean, criterion, device)["accuracy"],
                                       "attack_success_rate": evaluate_extended(reference, triggered, criterion, device)["accuracy"]}
        for name, loader in [("clean", clean), ("triggered", triggered)]:
            agree, total = 0, 0
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    agree += (model(images).argmax(1) == reference(images).argmax(1)).sum().item()
                    total += len(images)
            result[f"cf_agreement_{name}"] = 100 * agree / total
    return result


def write_results(output, source_metrics, metrics, metadata):
    (output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    rows = [("M_attack (source)", source_metrics["clean_accuracy"], source_metrics["triggered_rate"]),
            (metadata["method"], metrics["clean_accuracy"], metrics["attack_success_rate"])]
    if "reference_metrics" in metrics:
        ref = metrics["reference_metrics"]
        rows.append(("M_CF (existing weights, checkerboard evaluation)", ref["clean_accuracy"], ref["attack_success_rate"]))
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "clean_accuracy_pct", "asr_pct"])
        writer.writerows(rows)
    report = ["# Checkerboard unlearning results", "", f"Source run: `{metadata['source_run']}`.",
              f"Method: {metadata['method']}. Partition: {metadata['source_config']['partition_type']}.", "",
              "| Model | Clean accuracy % | Checkerboard ASR % |", "|---|---:|---:|"]
    report += [f"| {name} | {ca:.2f} | {asr:.2f} |" for name, ca, asr in rows]
    if metadata.get("calibration_epochs") is not None:
        report += ["", f"FedEraser calibration epochs: {metadata['calibration_epochs']}."]
    report += ["", "CA: 10,000 clean images. ASR: 9,000 triggered non-target images. Trigger: 4x4 bottom-right checkerboard.",
               "No clean/reference training is performed. Existing weights are only evaluated if explicitly requested.",
               "See run_config.json for exact source hashes and method settings. These files do not overwrite baseline tables.", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    source, config, source_metrics, counts, source_hash, subtraction = validate_inputs(args)
    check = {"source_run": str(source), "method": args.method, "partition": config["partition_type"],
             "source_ca": source_metrics["clean_accuracy"], "source_asr": source_metrics["triggered_rate"],
             "historical_rounds": config["num_rounds"], "source_model_sha256": source_hash}
    if args.check_only:
        print(json.dumps(check, indent=2))
        return
    if not (args.data_dir / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(f"CIFAR-10 is required at {args.data_dir / 'cifar-10-batches-py'}")
    if args.method == "kd" and not (args.data_dir / "cifar-100-python/train").is_file():
        raise FileNotFoundError("KD needs CIFAR-100 training data; download it once before running KD")
    suffix = f"federaser_Ecal{args.calibration_epochs}" if args.method == "federaser" else args.method
    requested = args.output_dir or source / "unlearning" / suffix
    if requested.resolve() == source or requested.resolve() in source.parents:
        raise ValueError("Output cannot replace the attacked run or its parents")
    output = Path(prepare_output_directory(requested, unique_on_conflict=True))
    metadata = {**check, "source_config": config, "source_model_sha256": source_hash,
                "subtraction_dir": str(subtraction) if args.method == "kd" else None,
                "reference_model": str(args.reference_model.resolve()) if args.reference_model else None,
                "calibration_epochs": args.calibration_epochs if args.method == "federaser" else None,
                "forgotten_client": 0, "status": "started"}
    config_path = output / "run_config.json"
    config_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Output: {output.resolve()}", flush=True)
    import torch
    from src.models.small_cnn import SmallCNN
    from src.utils.seed import set_seed
    device = torch.device("cpu" if args.method == "subtraction" else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(config["seed"])
    model = SmallCNN().to(device)
    if args.reference_model is not None:
        source_init = torch.load(source / "initial_model.pt", map_location="cpu", weights_only=True)
        ref_init = torch.load(args.reference_model.parent / "initial_model.pt", map_location="cpu", weights_only=True)
        if source_init.keys() != ref_init.keys() or any(not torch.equal(source_init[k], ref_init[k]) for k in source_init):
            raise ValueError("Reference initialization does not match this attacked run")
    if args.method == "subtraction":
        model.load_state_dict(subtract(torch, source, counts, config["num_rounds"]))
    elif args.method == "kd":
        metadata["kd"] = distill(torch, source, subtraction, config, model, device, args, output)
    else:
        reconstruct(torch, source, config, model, device, args)
    torch.save(model.state_dict(), output / "final_model.pt")
    metadata["output_model_sha256"] = sha256(output / "final_model.pt")
    metadata["status"] = "model_saved"
    config_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metrics = evaluate(torch, model, config, args, device)
    write_results(output, source_metrics, metrics, metadata)
    metadata["status"] = "completed"
    config_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"CA: {metrics['clean_accuracy']:.2f}% | checkerboard ASR: {metrics['attack_success_rate']:.2f}%", flush=True)
    print(f"Report: {output / 'report.md'}", flush=True)
    return output


if __name__ == "__main__":
    main()
