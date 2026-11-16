"""Require and collect matched Stage 1 outputs; mark stage done only if complete."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("results/onsetfree")


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    result = {"status": "matched_remote_V2_white_patch", "generated_utc": datetime.now(timezone.utc).isoformat(), "partitions": {}}
    for setting in ("iid", "noniid"):
        attacked_dir = ROOT / f"phase2_{setting}_attack"
        cf_dir = ROOT / f"counterfactual_{setting}"
        ecal_dirs = {n: ROOT / f"federaser_{setting}_ecal{n}" for n in (1, 2)}
        verified = read(ROOT / f"{setting}_trajectory_verification.json")
        if not verified["passed"] or verified["rounds"] != 50 or verified["client_deltas"] != 500:
            raise ValueError(f"{setting} trajectory failed the Stage 1 integrity gate")
        attacked = read(attacked_dir / "final_results.json")
        cf = read(cf_dir / "final_results.json")
        costs = read(ROOT / f"{setting}_cost.json")
        methods = {
            "attack": {
                "ca_percent": attacked["clean_accuracy"], "asr_percent": attacked["triggered_rate"],
                "wall_seconds": attacked["runtime_seconds"], "client_flops": None,
                "checkpoint_sha256": sha256(attacked_dir / "final_model.pt"),
            },
            "counterfactual": {
                "ca_percent": cf["clean_accuracy"], "asr_percent": cf["triggered_rate"],
                "wall_seconds": cf["runtime_seconds"], "client_flops": costs["counterfactual_client_flops"],
                "checkpoint_sha256": sha256(cf_dir / "final_model.pt"),
            },
        }
        for ecal, path in ecal_dirs.items():
            if not (path / "DONE").exists():
                raise ValueError(f"Incomplete FedEraser output: {path}")
            data = read(path / "results.json")
            methods[f"federaser_ecal{ecal}"] = {
                "ca_percent": data["ca_percent"], "asr_percent": data["asr_percent"],
                "wall_seconds": data["reconstruction_seconds"],
                "client_flops": costs[f"federaser_ecal{ecal}_client_flops"],
                "checkpoint_sha256": sha256(path / "final_model.pt"),
            }
        result["partitions"][setting] = {
            "trajectory_verification": str(ROOT / f"{setting}_trajectory_verification.json"),
            "flop_counting_method": costs["counting_method"],
            "methods": methods,
        }
    (ROOT / "baselines.json").write_text(json.dumps(result, indent=2) + "\n")
    (ROOT / "stage1.DONE").write_text("Matched IID and non-IID trajectories, counterfactuals and FedEraser Ecal 1/2 verified; see baselines.json.\n")


if __name__ == "__main__":
    main()
