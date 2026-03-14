#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_modules.prior_knowledge import PriorKnowledgeStatsBuilder  # noqa: E402
import train  # noqa: F401,E402 - triggers dataset registration


def parse_args():
    parser = argparse.ArgumentParser(description="Build class-wise prior statistics.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Name of the registered dataset (e.g., duo_train_sparse_50_30).",
    )
    parser.add_argument(
        "--output",
        default="priors/duo_prior.json",
        help="Path to store the prior JSON file.",
    )
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=5.0,
        help="Lower percentile for feature bounds.",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=95.0,
        help="Upper percentile for feature bounds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    builder = PriorKnowledgeStatsBuilder(
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    builder.build(args.dataset, args.output)
    print(f"Saved priors for {args.dataset} to {args.output}")


if __name__ == "__main__":
    main()

