"""Separate run artifacts and human-readable reports for attack experiments."""

import csv
import os


def prepare_output_directory(output_dir, unique_on_conflict=False):
    """Preserve existing artifacts, optionally reserving a fresh numbered sibling."""
    output_dir = os.path.normpath(os.fspath(output_dir))
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise ValueError(f"Output must be a new or empty directory: {output_dir}. Choose a new run ID.")
        if os.listdir(output_dir):
            if not unique_on_conflict:
                raise ValueError(f"Output must be a new or empty directory: {output_dir}. Choose a new run ID.")
            attempt = 1
            while True:
                candidate = f"{output_dir}_rerun_{attempt:03d}"
                try:
                    os.mkdir(candidate)
                except FileExistsError:
                    attempt += 1
                else:
                    return candidate
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def write_attack_report(output_dir, final_output, config_path, log_trajectory):
    """Create a self-contained report from this run, without editing baseline tables."""
    config = final_output["config"]
    with open(os.path.join(output_dir, "metrics_per_round.csv"), newline="") as handle:
        rounds = list(csv.DictReader(handle))
    attack_rounds = [row for row in rounds if row["Attack_Active"] == "True"]
    peak = max(attack_rounds, key=lambda row: float(row["Triggered_Target_Rate"])) if attack_rounds else None
    report = [
        "# Federated attack experiment", "",
        f"Experiment: `{config.get('experiment_name', os.path.basename(os.path.normpath(output_dir)))}`",
        f"Config: `{config_path}`. Output: `{os.path.abspath(output_dir)}`.", "",
        "| Final endpoint metric | Value |", "|---|---:|",
        f"| Clean accuracy (CA), 10,000 clean images | {final_output['clean_accuracy']:.2f}% |",
        f"| Attack success rate (ASR), 9,000 triggered non-target images | {final_output['triggered_rate']:.2f}% |",
        f"| Untriggered false-target rate, same 9,000 images without trigger | {final_output['untriggered_airplane_rate']:.2f}% |", "",
        "Final metrics use the last training round; no test-selected checkpoint is substituted.", "",
        "## Protocol", "",
        f"- Mode: {final_output['mode']}; partition: {config['partition_type']}; seed: {config['seed']}.",
        f"- Trigger: {config['trigger_type']}, {config['trigger_size']} x {config['trigger_size']}, bottom right; target class {config['target_class']}.",
        f"- Malicious client: {config['malicious_client_id']}; {final_output['poisoned_samples']} fixed poisoned samples out of {final_output['malicious_client_samples']} local samples.",
        f"- Attack rounds: {config['attack_start_round']} through {config['num_rounds']}, inclusive.",
        f"- Clients: {config['num_clients']} with full participation; local epochs: {config['local_epochs']}; batch size: {config['batch_size']}.",
        f"- SGD: learning rate {config['learning_rate']}, momentum {config['momentum']}, weight decay 0; no update amplification.",
        f"- Historical client-update logging enabled: {log_trajectory}.", "",
    ]
    if config["trigger_type"] == "checkerboard":
        report += ["Raw patch pixels (W = white 1.0, B = black 0.0; identical pattern in all RGB channels):", "", "```text"]
        report += [" ".join("W" if (r + c) % 2 == 0 else "B" for c in range(config["trigger_size"]))
                   for r in range(config["trigger_size"])]
        report += ["```", "", "Both colors are normalized using the CIFAR-10 mean/std before insertion.", ""]
    if peak:
        report += [f"Peak attack-period ASR: {float(peak['Triggered_Target_Rate']):.2f}% at round {peak['Round']} (diagnostic only).", ""]
    report += ["## Interpretation", "",
               "Judge CA and final ASR together. A larger checkerboard is a candidate stronger trigger, not an established improvement until measured.",
               "The old white-patch ASR is a different trigger evaluation. Existing clean/reference weights can be reused, but their ASR must be re-evaluated with this trigger for matched comparisons.",
               "This run does not train clean/reference models or perform unlearning. Other legacy evaluators/unlearning launchers still assume the older trigger/paths and must be adapted before use.", ""]
    with open(os.path.join(output_dir, "attack_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))

