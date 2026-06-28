from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from anomaly_detection.train import build_parser, run
from anomaly_detection.utils import ensure_dir


def main() -> None:
    parser = build_parser()
    parser.description = "Run baseline and improved anomaly detection experiments"
    parser.set_defaults(output_dir="outputs/comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.output_dir == "outputs/comparison":
        output_dir = Path("outputs") / f"comparison_{args.epochs}ep"
    output_dir = ensure_dir(output_dir)

    summary = {}
    for variant in ["baseline", "improved"]:
        variant_args = copy.deepcopy(args)
        variant_args.variant = variant
        variant_args.output_dir = str(output_dir / variant)
        summary[variant] = run(variant_args)

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
