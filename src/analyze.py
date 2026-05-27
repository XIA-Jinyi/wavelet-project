"""Analysis: read CSV results → compute metrics, generate figures."""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                               f1_score, precision_score, recall_score)
from pathlib import Path

from config import RESULT_DIR, FIGURE_DIR, WAVELETS, RATES

MODELS = ["wdcnn", "mlp", "svm", "cnn"]
COLORS = {"wdcnn": "#e74c3c", "mlp": "#3498db", "svm": "#2ecc71", "cnn": "#8e44ad"}


def load_csv(path):
    y_true, y_prob = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["label"]))
            y_prob.append(float(row["predicted_prob"]))
    return np.array(y_true), np.array(y_prob)


def evaluate(y_true, y_pred, y_prob):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=350)
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.n_train}"

    # Collect all results
    all_metrics = []
    roc_curves = {}

    for model in MODELS:
        if model == "cnn":
            for rate in RATES:
                path = RESULT_DIR / f"cnn_{rate:.1f}{suffix}.csv"
                if not path.exists():
                    continue
                y_true, y_prob = load_csv(path)
                y_pred = (y_prob > 0.5).astype(int)
                m = evaluate(y_true, y_pred, y_prob)
                m.update({"model": "cnn", "wavelet": "none", "rate": f"{rate:.1f}"})
                all_metrics.append(m)
                for wavelet in WAVELETS:
                    roc_curves[("cnn", wavelet, rate)] = (y_true.copy(), y_prob.copy())
        else:
            for wavelet in WAVELETS:
                for rate in RATES:
                    path = RESULT_DIR / f"{model}_{wavelet}_{rate:.1f}{suffix}.csv"
                    if not path.exists():
                        continue
                    y_true, y_prob = load_csv(path)
                    y_pred = (y_prob > 0.5).astype(int)
                    m = evaluate(y_true, y_pred, y_prob)
                    m.update({"model": model, "wavelet": wavelet, "rate": f"{rate:.1f}"})
                    all_metrics.append(m)
                    roc_curves[(model, wavelet, rate)] = (y_true, y_prob)

    if not all_metrics:
        print("No result CSVs found. Run 'make test' first.")
        return

    # Print table
    print(f"{'Model':>8s}  {'Wavelet':>7s}  {'Rate':>6s}  {'Acc':>8s}  {'AUC':>8s}  {'F1':>8s}  {'Prec':>8s}  {'Rec':>8s}")
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['model']:>8s}  {m['wavelet']:>7s}  {m['rate']:>6s}  "
              f"{m['acc']:8.4f}  {m['auc']:8.4f}  {m['f1']:8.4f}  "
              f"{m['prec']:8.4f}  {m['rec']:8.4f}")

    # Save summary CSV
    summary_path = RESULT_DIR / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        w.writeheader(); w.writerows(all_metrics)
    print(f"\nSaved summary to {summary_path}")

    # ROC curves — one figure per rate
    for rate_i, rate in enumerate(RATES):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for col, wavelet in enumerate(WAVELETS):
            ax = axes[col]
            for model in MODELS:
                key = (model, wavelet, rate)
                if key in roc_curves:
                    yt, yp = roc_curves[key]
                    fpr, tpr, _ = roc_curve(yt, yp)
                    auc_v = roc_auc_score(yt, yp)
                    ax.plot(fpr, tpr, color=COLORS[model], linewidth=1.8,
                            label=f"{model} AUC={auc_v:.4f}")
            ax.plot([0, 1], [0, 1], "k:", linewidth=0.4)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"{wavelet} @ {rate:.1f} bpp"); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"roc_{rate:.1f}{suffix}.png", dpi=150)
        plt.close()
        print(f"Saved roc_{rate:.1f}{suffix}.png")

    # Comparison bar chart — AUC by model × wavelet × rate
    fig, ax = plt.subplots(figsize=(14, 6))
    x_labels = []
    for m in all_metrics:
        if m["model"] == "cnn":
            x_labels.append(f"cnn\n{m['rate']} bpp")
        else:
            x_labels.append(f"{m['wavelet']}\n{m['rate']} bpp")
    x = np.arange(len(all_metrics))
    bar_colors = [COLORS[m["model"]] for m in all_metrics]
    bars = ax.bar(x, [m["auc"] for m in all_metrics], color=bar_colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("AUC"); ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[m], label=m) for m in MODELS]
    ax.legend(handles=legend_elements)
    ax.set_title(f"AUC comparison ({args.n_train} train samples)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"auc_comparison{suffix}.png", dpi=150)
    plt.close()
    print(f"Saved auc_comparison{suffix}.png")


if __name__ == "__main__":
    main()
