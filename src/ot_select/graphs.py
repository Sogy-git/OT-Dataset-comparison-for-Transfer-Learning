import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

METHOD_ORDER = ["random_zero_shot", "full_zero_shot", "KNN", "OT", "OTDD"]
METHOD_LABELS = {
    "random_zero_shot": "Random (zero-shot)",
    "full_zero_shot": "Full MNIST (zero-shot)",
    "KNN": "KNN",
    "OT": "OT",
    "OTDD": "OTDD",
}
METHOD_COLORS = {
    "random_zero_shot": "#9e9e9e",
    "full_zero_shot": "#2196f3",
    "KNN": "#4caf50",
    "OT": "#ff9800",
    "OTDD": "#f44336",
}


def results_to_dataframe(all_results):
    """Flatten nested run results into a pandas DataFrame."""
    rows = []
    for run in all_results:
        for method, metrics in run["methods"].items():
            rows.append({
                "seed": run["seed"],
                "semeion_train_ratio": run["semeion_train_ratio"],
                "semeion_train_size": run["semeion_train_size"],
                "semeion_test_size": run["semeion_test_size"],
                "method": method,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
            })
    return pd.DataFrame(rows)


def _ordered_methods(df):
    present = set(df["method"].unique())
    return [m for m in METHOD_ORDER if m in present]


def plot_accuracy_by_ratio(df, save_path, title="SEMEION Test Accuracy vs. Available Target Data"):
    """Line plot: accuracy vs SEMEION train ratio, one line per method, error bars across seeds."""
    methods = _ordered_methods(df)
    ratios = sorted(df["semeion_train_ratio"].unique())
    ratio_labels = [f"{int(r * 100)}%" for r in ratios]

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in methods:
        subset = df[df["method"] == method]
        means = []
        stds = []
        for ratio in ratios:
            accs = subset[subset["semeion_train_ratio"] == ratio]["accuracy"]
            means.append(accs.mean())
            stds.append(accs.std() if len(accs) > 1 else 0.0)

        ax.errorbar(
            ratio_labels, means, yerr=stds,
            marker="o", capsize=4, linewidth=2,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("SEMEION data available for selection (train split)")
    ax.set_ylabel("Accuracy on SEMEION test set")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_accuracy_comparison(df, save_path, ratio=None):
    """Bar chart comparing methods, optionally filtered to one ratio."""
    if ratio is not None:
        df = df[df["semeion_train_ratio"] == ratio]
        title = f"Method Comparison ({int(ratio * 100)}% SEMEION train data)"
    else:
        title = "Method Comparison (all ratios, mean across seeds)"

    summary = (
        df.groupby("method")["accuracy"]
        .agg(["mean", "std"])
        .reindex(_ordered_methods(df))
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    labels = [METHOD_LABELS.get(m, m) for m in summary.index]

    ax.bar(
        x, summary["mean"],
        yerr=summary["std"],
        capsize=4,
        color=[METHOD_COLORS.get(m, "#666666") for m in summary.index],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_loss_by_ratio(df, save_path):
    """Line plot: loss vs SEMEION train ratio with error bars across seeds."""
    methods = _ordered_methods(df)
    ratios = sorted(df["semeion_train_ratio"].unique())
    ratio_labels = [f"{int(r * 100)}%" for r in ratios]

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in methods:
        subset = df[df["method"] == method]
        means = []
        stds = []
        for ratio in ratios:
            losses = subset[subset["semeion_train_ratio"] == ratio]["loss"]
            means.append(losses.mean())
            stds.append(losses.std() if len(losses) > 1 else 0.0)

        ax.errorbar(
            ratio_labels, means, yerr=stds,
            marker="o", capsize=4, linewidth=2,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("SEMEION data available for selection (train split)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("SEMEION Test Loss vs. Available Target Data")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(run_result, save_path):
    """Grouped bar chart of per-class accuracy for each method in one run."""
    methods = _ordered_methods(pd.DataFrame([
        {"method": m} for m in run_result["methods"]
    ]))
    per_method = {}

    for method in methods:
        metrics = run_result["methods"][method]
        targets = np.array(metrics["targets"])
        preds = np.array(metrics["predictions"])
        class_accs = []
        for digit in range(10):
            mask = targets == digit
            class_accs.append((preds[mask] == digit).mean() if mask.any() else 0.0)
        per_method[method] = class_accs

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(10)
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(
            x + offset, per_method[method], width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("Digit class")
    ax.set_ylabel("Per-class accuracy")
    ax.set_title(
        f"Per-class Accuracy (seed={run_result['seed']}, "
        f"{int(run_result['semeion_train_ratio'] * 100)}% SEMEION train)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_seed_stability(df, save_path):
    """Box plot of accuracy per method across all seeds and ratios."""
    methods = _ordered_methods(df)
    data = [df[df["method"] == m]["accuracy"].values for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Distribution Across Seeds and Ratios")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_all_figures(all_results, output_dir):
    """Generate all evaluation figures from a list of run result dicts."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    df = results_to_dataframe(all_results)

    plot_accuracy_by_ratio(df, os.path.join(figures_dir, "accuracy_by_ratio.png"))
    plot_loss_by_ratio(df, os.path.join(figures_dir, "loss_by_ratio.png"))
    plot_accuracy_comparison(df, os.path.join(figures_dir, "accuracy_comparison_overall.png"))
    plot_seed_stability(df, os.path.join(figures_dir, "seed_stability.png"))

    for ratio in sorted(df["semeion_train_ratio"].unique()):
        pct = int(ratio * 100)
        plot_accuracy_comparison(
            df, os.path.join(figures_dir, f"accuracy_comparison_{pct}pct.png"), ratio=ratio,
        )

    # Per-run detail plots for the last run of each ratio (most seeds)
    seen_ratios = set()
    for run in reversed(all_results):
        ratio = run["semeion_train_ratio"]
        if ratio in seen_ratios:
            continue
        seen_ratios.add(ratio)
        pct = int(ratio * 100)
        tag = f"seed{run['seed']}_ratio{pct}"

        plot_per_class_accuracy(run, os.path.join(figures_dir, f"per_class_{tag}.png"))

        for method, metrics in run["methods"].items():
            plot_confusion_matrix(
                metrics["targets"],
                metrics["predictions"],
                os.path.join(figures_dir, f"confusion_{method}_{tag}.png"),
                title=f"{METHOD_LABELS.get(method, method)} — seed={run['seed']}, {pct}% train",
            )

    return figures_dir
