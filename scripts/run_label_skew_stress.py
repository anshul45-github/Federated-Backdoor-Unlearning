"""Train and compare subtraction, KD and FedEraser at Dirichlet alpha=0.01."""
import argparse
import csv
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.attack_report import prepare_output_directory


def partition_report(hist, config):
    clients = []
    for c in range(10):
        entry = hist[f"client_{c}"]
        counts = [entry["class_distribution"].get(str(k), entry["class_distribution"].get(k, 0)) for k in range(10)]
        n = entry["num_samples"]
        if sum(counts) != n or any(v < 0 for v in counts):
            raise ValueError("Invalid class histogram")
        clients.append(dict(client=c, num_samples=n, class_counts=counts,
                            classes_present=sum(v > 0 for v in counts),
                            dominant_class_fraction=max(counts) / n if n else None,
                            label_entropy_nats=-sum(v/n * math.log(v/n) for v in counts if v) if n else 0))
    totals = [sum(c["class_counts"][k] for c in clients) for k in range(10)]
    if totals != [5000] * 10:
        raise ValueError("Expected all 50,000 CIFAR-10 training examples by class")
    c0 = clients[0]
    poisons = int(c0["num_samples"] * config["poison_ratio"])
    candidates = c0["num_samples"] - c0["class_counts"][config["target_class"]]
    retained = [totals[k] - c0["class_counts"][k] for k in range(10)]
    errors = []
    if poisons < 1 or poisons > candidates:
        errors.append(f"Client 0 needs {poisons} poisoned non-target examples; {candidates} available")
    if sum(retained) == 0:
        errors.append("No retained training data")
    return dict(alpha=config["dirichlet_alpha"], seed=config["seed"], clients=clients,
                empty_clients=[c["client"] for c in clients if c["num_samples"] == 0],
                malicious_client_weight=c0["num_samples"] / 50000, poisoned_samples=poisons,
                non_target_poison_candidates=candidates, retained_class_counts=retained,
                classes_lost_on_removal=[k for k, n in enumerate(retained) if n == 0], errors=errors)


def commands(run):
    source = run / "attack"
    outputs = source / "unlearning"
    common = [sys.executable, "-u", "scripts/run_checkerboard_unlearning.py", "--source_run", str(source)]
    return [
        ("attack", [sys.executable, "-u", "scripts/run_experiment_v2.py", "--config", str(run / "config.yaml"),
                    "--mode", "attack", "--log_trajectory", "--output_dir", str(source)]),
        ("subtraction", common + ["--method", "subtraction", "--output_dir", str(outputs / "subtraction")]),
        ("kd", common + ["--method", "kd", "--subtraction_dir", str(outputs / "subtraction"), "--output_dir", str(outputs / "kd")]),
        ("federaser", common + ["--method", "federaser", "--calibration_epochs", "2", "--output_dir", str(outputs / "federaser_Ecal2")])]


def summarize(run):
    source = run / "attack"
    attack = json.loads((source / "final_results.json").read_text())
    rows = [dict(model="attack", clean_accuracy=attack["clean_accuracy"], attack_success_rate=attack["triggered_rate"],
                 untriggered_target_rate=attack["untriggered_airplane_rate"], macro_f1_pct=100 * attack["clean_macro_f1"])]
    for method in ["subtraction", "kd", "federaser_Ecal2"]:
        metrics = json.loads((source / "unlearning" / method / "evaluation.json").read_text())
        rows.append(dict(model=method, **{k: metrics[k] for k in rows[0] if k != "model"}))
    for row in rows:
        row["ca_change_pp"] = row["clean_accuracy"] - attack["clean_accuracy"]
        row["asr_change_pp"] = row["attack_success_rate"] - attack["triggered_rate"]
    with (run / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Extreme label skew stress test", "", "Dirichlet alpha = 0.01. See partition_report.json for realized skew.", "",
              "| Model | CA % | ASR % | Untriggered target % | Macro F1 % | CA change pp |", "|---|---:|---:|---:|---:|---:|"]
    report += [f"| {r['model']} | {r['clean_accuracy']:.2f} | {r['attack_success_rate']:.2f} | "
               f"{r['untriggered_target_rate']:.2f} | {r['macro_f1_pct']:.2f} | {r['ca_change_pp']:+.2f} |" for r in rows]
    report += ["", "CA: 10,000 clean test images. ASR: 9,000 triggered non-target images.",
               "Compare source ASR with its untriggered target rate to establish whether the backdoor learned.",
               "Low ASR with collapsed CA does not establish useful unlearning. Inspect per-class recall in evaluation.json.",
               "No counterfactual reference is trained; these results measure endpoint utility and backdoor behavior.", ""]
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--plan", action="store_true", help="Print commands without dependencies or writes")
    parser.add_argument("--preflight_only", action="store_true", help="Verify datasets and save partition reports without training")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("Seeds must be unique unsigned 32-bit integers")
    output = (args.output_dir or ROOT / "results/label_skew" / f"alpha001_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    if args.plan:
        for seed in args.seeds:
            print(f"Seed {seed}: validate datasets/partition and generate config.yaml")
            for stage, command in commands(output / f"seed_{seed}"):
                print(f"{stage}: {subprocess.list2cmdline(command)}")
        return
    missing = [m for m in ["torch", "torchvision", "numpy", "yaml"] if importlib.util.find_spec(m) is None]
    if missing:
        parser.error(f"Missing training dependencies in {sys.executable}: {', '.join(missing)}. Use the training environment; --plan works without dependencies.")
    import yaml
    from torchvision.datasets import CIFAR10, CIFAR100
    from src.data.fl_partition import create_noniid_partitions
    config = yaml.safe_load((ROOT / "configs/v4_checkerboard4_extreme_noniid.yaml").read_text())
    if (config["dirichlet_alpha"] != 0.01 or config["partition_type"] != "noniid" or config["num_clients"] != 10 or
            config["clients_per_round"] != 10 or config["malicious_client_id"] != 0 or
            config["trigger_type"] != "checkerboard" or config["trigger_size"] != 4):
        parser.error("Expected alpha=0.01, ten full-participation clients, Client 0 attack and 4x4 checkerboard")
    dataset = CIFAR10(str(ROOT / "data"), train=True, download=False)
    CIFAR10(str(ROOT / "data"), train=False, download=False)
    CIFAR100(str(ROOT / "data"), train=True, download=False)
    output = Path(prepare_output_directory(output, unique_on_conflict=False))
    prepared = []
    for seed in args.seeds:
        run = output / f"seed_{seed}"
        run.mkdir()
        current = dict(config, seed=seed, experiment_name=f"checkerboard4_extreme_noniid_alpha001_seed{seed}")
        (run / "config.yaml").write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
        partitions, hist = create_noniid_partitions(dataset, 10, 0.01, seed)
        indices = [i for p in partitions for i in p.indices]
        if len(indices) != 50000 or len(set(indices)) != 50000:
            raise ValueError("Partition must cover each training image exactly once")
        report = partition_report(hist, current)
        (run / "partition_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (run / "client_class_histograms.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
        print(f"Seed {seed}: sizes {[len(p) for p in partitions]}, C0 weight {report['malicious_client_weight']:.4f}, "
              f"poisons {report['poisoned_samples']}, empty clients {report['empty_clients']}", flush=True)
        prepared.append((run, report))
    errors = [f"{run.name}: {error}" for run, report in prepared for error in report["errors"]]
    if errors:
        raise ValueError("No training launched: " + "; ".join(errors))
    if args.preflight_only:
        print(f"Preflight complete: {output}")
        return
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    for run, _ in prepared:
        for stage, command in commands(run):
            print(f"Starting {run.name}/{stage}; log: {run / (stage + '.log')}", flush=True)
            status = dict(stage=stage, status="running", command=command)
            status_path = run / "status.json"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            with (run / f"{stage}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{run.name}/{stage} failed; inspect its log")
        summarize(run)
    print(f"Completed stress test: {output}", flush=True)


if __name__ == "__main__":
    main()
