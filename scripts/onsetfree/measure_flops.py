"""Count model forward/backward FLOPs for matched baseline client training.

This excludes optimizer arithmetic, data augmentation, communication, and
server aggregation. It uses FlopCounterMode's supported ATen operators.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.small_cnn import SmallCNN


def one_step_flops(batch_size):
    model = SmallCNN().train()
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.zeros(batch_size, dtype=torch.long)
    with FlopCounterMode(display=False) as counter:
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
    return counter.get_total_flops()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--malicious-client", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=2)
    args = parser.parse_args()
    raw = json.loads(Path(args.counts_json).read_text())
    counts = {int(key.split("_")[-1]): int(value["num_samples"]) for key, value in raw.items()}
    retained = {i: n for i, n in counts.items() if i != args.malicious_client}
    remainders = {n % args.batch_size for n in retained.values() if n % args.batch_size}
    flops_by_batch = {args.batch_size: one_step_flops(args.batch_size)}
    flops_by_batch.update({r: one_step_flops(r) for r in sorted(remainders)})

    def one_epoch(n):
        full, remainder = divmod(n, args.batch_size)
        return full * flops_by_batch[args.batch_size] + (flops_by_batch[remainder] if remainder else 0)

    epoch_flops = sum(one_epoch(n) for n in retained.values())
    result = {
        "counting_method": "torch.utils.flop_counter.FlopCounterMode, one fwd+backward on SmallCNN per batch size; optimizer/data/server ops excluded",
        "source_counts": str(Path(args.counts_json).resolve()),
        "one_retained_client_epoch_flops": epoch_flops,
        "counterfactual_client_flops": args.rounds * args.local_epochs * epoch_flops,
        "federaser_ecal1_client_flops": (args.rounds - 1) * epoch_flops,
        "federaser_ecal2_client_flops": (args.rounds - 1) * 2 * epoch_flops,
        "l0_client_flops": 0,
        "per_batch_flops": {str(k): v for k, v in flops_by_batch.items()},
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
