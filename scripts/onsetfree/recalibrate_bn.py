"""Re-estimate BN running statistics from a fixed benign training subsample.

This variant has benign-client computation and must not be reported as
zero-client-cost L0. No attack trigger or test set is read.
"""

import argparse
import json
from pathlib import Path

import torch

from scripts.onsetfree.run_l1 import fixed_benign_batches
from src.models.small_cnn import SmallCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--malicious-client", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    counts_raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in counts_raw.items()}
    batches, selected = fixed_benign_batches(Path(args.source_run), counts, args.malicious_client, args.samples, args.seed, args.batch_size, args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(args.input_model, map_location=device, weights_only=True))
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
    with torch.no_grad():
        for x, _ in batches:
            model(x.to(device))
    output = Path(args.output_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    output.with_suffix(".json").write_text(json.dumps({
        "source_model": str(Path(args.input_model).resolve()),
        "source_run": str(Path(args.source_run).resolve()),
        "benign_samples": args.samples,
        "sample_seed": args.seed,
        "sample_indices_by_client": selected,
        "client_flops_status": "benign forwards not yet counted",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
