"""Collect compact measured JSON endpoints without checkpoints or sample lists."""

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("results/onsetfree")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def collect(setting):
    result = {"baselines": read(ROOT / "baselines.json")["partitions"][setting], "evaluations": {}, "costs": {}, "diagnostics": {}}
    for group in ("l0", "l0_sweep", "l1", "l1_sweep", "l1_seeds", "l1_hessian", "l1_bn_train", "l2", "l2_heavy_ball", "l3", "gating"):
        directory = ROOT / group / setting
        for path in sorted(directory.glob("**/evaluation.json")):
            result["evaluations"][str(path.relative_to(ROOT))] = read(path)["results"]
        if group == "l3":
            for path in sorted(directory.glob("**/results.json")):
                result["evaluations"][str(path.relative_to(ROOT))] = read(path)
        for path in sorted(directory.glob("**/cost.json")):
            data = read(path)
            result["costs"][str(path.relative_to(ROOT))] = {key: value for key, value in data.items() if key in ("estimated_total_ggn_flops", "ggn_vector_products", "one_batch_ggn_flops", "curvature_samples")}
    for name in (f"{setting}_benign_validation", f"{setting}_l1_cosine", f"{setting}_per_class_asr"):
        path = ROOT / "diagnostics" / f"{name}.json"
        data = read(path)
        if data:
            if name.endswith("benign_validation"):
                data = {"baseline": data["baseline"], "methods": data["methods"], "sample_count": 512}
            result["diagnostics"][name] = data
    onset = read(ROOT / "onset" / setting / "detector.json")
    if onset:
        result["diagnostics"]["onset"] = {"true_onset_evaluation_only": onset["true_onset_evaluation_only"], "analyses": {key: {field: value for field, value in analysis.items() if field in ("threshold", "empirical_benign_false_alarm_rate", "estimated_onset", "onset_error_rounds")} for key, analysis in onset["analyses"].items()}}
    return result


def main():
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "Matched V2 white-patch trajectories in /DATA/teaching/new-approach/repo; test metrics are evaluation-only",
        "partitions": {setting: collect(setting) for setting in ("iid", "noniid")},
    }
    output = ROOT / "summary_metrics.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
