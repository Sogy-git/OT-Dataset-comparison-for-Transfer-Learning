#!/usr/bin/env python3
"""Run multi-seed, multi-ratio evaluation of OT sample selection methods."""

import argparse
import json
import os
import sys

import pandas as pd

from graphs import generate_all_figures, results_to_dataframe
from pipeline import get_device, run_experiment


DEFAULT_SEEDS = [42, 69, 123, 456, 789]
DEFAULT_RATIOS = [0.10, 0.20, 0.40]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate KNN / OT / OTDD selection across seeds and SEMEION train ratios.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for metrics, models, and figures (default: results)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help=f"Random seeds for SEMEION splits (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=DEFAULT_RATIOS,
        help=f"SEMEION train ratios (default: {DEFAULT_RATIOS})",
    )
    parser.add_argument("--subset-size", type=int, default=5000, help="MNIST samples to select")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs per method")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--pretrained", default="simple_cnn_full.pth",
        help="Pretrained MNIST checkpoint used for embeddings and fine-tuning",
    )
    parser.add_argument(
        "--random-pretrained", default="simple_cnn_random.pth",
        help="Random-subset baseline checkpoint (skipped if missing)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast smoke run: 1 seed, 10%% ratio, 1 epoch",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Load existing results JSON and regenerate figures only",
    )
    parser.add_argument("--show", action="store_true", help="Display figures after saving")
    return parser.parse_args()


def save_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    df = results_to_dataframe(all_results)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)

    return json_path, csv_path


def print_summary(df):
    print("\n" + "=" * 72)
    print("SUMMARY — mean accuracy (± std across seeds)")
    print("=" * 72)

    summary = (
        df.groupby(["semeion_train_ratio", "method"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for ratio in sorted(df["semeion_train_ratio"].unique()):
        pct = int(ratio * 100)
        print(f"\n  SEMEION train ratio: {pct}%")
        subset = summary[summary["semeion_train_ratio"] == ratio].sort_values("mean", ascending=False)
        for _, row in subset.iterrows():
            std = row["std"] if row["count"] > 1 else 0.0
            print(f"    {row['method']:20s}  {row['mean']:.1%} ± {std:.1%}  (n={int(row['count'])})")

    print("=" * 72 + "\n")


def main():
    args = parse_args()

    if args.quick:
        args.seeds = [42]
        args.ratios = [0.10]
        args.epochs = 1

    if not os.path.exists(args.pretrained):
        print(f"Error: pretrained checkpoint not found: {args.pretrained}", file=sys.stderr)
        print("Run supervised_train.py first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "evaluation_results.json")

    if args.skip_training and os.path.exists(json_path):
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {json_path}")
    else:
        print(f"Device: {get_device()}")
        print(f"Seeds: {args.seeds}")
        print(f"SEMEION train ratios: {[f'{r:.0%}' for r in args.ratios]}")
        print(f"Output: {args.output_dir}\n")

        all_results = []
        total = len(args.seeds) * len(args.ratios)
        run_idx = 0

        for seed in args.seeds:
            for ratio in args.ratios:
                run_idx += 1
                print(f"[{run_idx}/{total}] seed={seed}, ratio={ratio:.0%}")

                result = run_experiment(
                    seed=seed,
                    semeion_train_ratio=ratio,
                    output_dir=args.output_dir,
                    subset_size=args.subset_size,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    pretrained_path=args.pretrained,
                    random_pretrained_path=args.random_pretrained,
                    verbose=not args.quick,
                )
                all_results.append(result)

        save_results(all_results, args.output_dir)

    df = results_to_dataframe(all_results)
    print_summary(df)

    figures_dir = generate_all_figures(all_results, args.output_dir)
    print(f"Figures saved to {figures_dir}/")
    print(f"Results saved to {args.output_dir}/evaluation_results.csv")

    if args.show:
        import matplotlib.pyplot as plt
        for fig_file in sorted(os.listdir(figures_dir)):
            if fig_file.endswith(".png"):
                img = plt.imread(os.path.join(figures_dir, fig_file))
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.axis("off")
                plt.title(fig_file)
        plt.show()


if __name__ == "__main__":
    main()
