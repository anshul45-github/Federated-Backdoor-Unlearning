"""Count GGN-vector-product FLOPs with the same SmallCNN operator as L1/L2."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.small_cnn import SmallCNN
from src.unlearning.onsetfree.curvature import BenignCurvature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="L1 or L2 metadata.json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    data = json.loads(Path(args.metadata).read_text())
    samples = data["curvature_samples"]
    if samples % args.batch_size:
        raise ValueError("This counter currently requires a whole number of equal batches")
    model = SmallCNN().eval()
    x = torch.randn(args.batch_size, 3, 32, 32)
    y = torch.zeros(args.batch_size, dtype=torch.long)
    operator = BenignCurvature(model, [(x, y)])
    vector = torch.randn(sum(parameter.numel() for parameter in model.parameters()))
    with FlopCounterMode(display=False) as counter:
        operator.ggn(vector)
    per_batch = counter.get_total_flops()
    calls = data["ggn_vector_products"]
    result = {
        "metadata": str(Path(args.metadata).resolve()),
        "method": "FlopCounterMode on one eval-mode SmallCNN GGN-vector product, multiplied by fixed subsample batches and logged HVP calls; excludes data transfer and uncounted operations",
        "one_batch_ggn_flops": per_batch,
        "batch_size": args.batch_size,
        "curvature_samples": samples,
        "ggn_vector_products": calls,
        "estimated_total_ggn_flops": per_batch * (samples // args.batch_size) * calls,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
