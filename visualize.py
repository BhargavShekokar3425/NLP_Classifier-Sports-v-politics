"""
visualize.py
============
Generate matplotlib charts from experiment results.

Produces:
  1. Grouped bar chart  – test accuracy per (feature, classifier)
  2. Heatmap            – test accuracy matrix (features × classifiers)
  3. Per-class F1 grouped bars
  4. Confusion matrices – one subplot per model combination
  5. Train vs Test accuracy comparison

All figures are saved to an `output/` directory and optionally shown.
"""

import os
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


# ───────────────────────────────────────────────
# 1. Test Accuracy Bar Chart
# ───────────────────────────────────────────────

def plot_accuracy_bars(results, save=True, show=False):
    """Grouped bar chart of test accuracy for each model combo."""
    _ensure_output_dir()

    labels = [f"{r['feature']}\n{r['classifier']}" for r in results]
    accs = [r["test_acc"] * 100 for r in results]

    colors = []
    cmap = {"BoW": "#4C72B0", "TF-IDF": "#55A868", "Bigram": "#C44E52"}
    for r in results:
        colors.append(cmap.get(r["feature"], "#8172B2"))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), accs, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy — All Model Combinations")
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "accuracy_bars.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return OUTPUT_DIR / "accuracy_bars.png"


# ───────────────────────────────────────────────
# 2. Accuracy Heatmap
# ───────────────────────────────────────────────

def plot_accuracy_heatmap(results, save=True, show=False):
    """Feature × Classifier heatmap of test accuracy."""
    _ensure_output_dir()

    features = sorted(set(r["feature"] for r in results),
                      key=lambda f: list(dict.fromkeys(r["feature"] for r in results)).index(f))
    classifiers = sorted(set(r["classifier"] for r in results),
                         key=lambda c: list(dict.fromkeys(r["classifier"] for r in results)).index(c))

    acc_map = {(r["feature"], r["classifier"]): r["test_acc"] for r in results}
    data = [[acc_map.get((f, c), 0) * 100 for c in classifiers] for f in features]
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(classifiers)))
    ax.set_xticklabels(classifiers, fontsize=9)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)

    for i in range(len(features)):
        for j in range(len(classifiers)):
            ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if data[i, j] > 70 else "black")

    ax.set_title("Test Accuracy Heatmap (Features × Classifiers)")
    fig.colorbar(im, ax=ax, label="Accuracy %")
    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "accuracy_heatmap.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return OUTPUT_DIR / "accuracy_heatmap.png"


# ───────────────────────────────────────────────
# 3. Per-class F1 Scores
# ───────────────────────────────────────────────

def plot_f1_scores(results, save=True, show=False):
    """Grouped bar chart of F1 score per class per model combo."""
    _ensure_output_dir()

    classes = results[0]["classes"]
    n = len(results)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, c in enumerate(classes):
        f1s = [r["per_class"][c]["f1"] * 100 for r in results]
        ax.bar(x + i * width, f1s, width, label=c)

    labels = [f"{r['feature']}\n{r['classifier']}" for r in results]
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Per-Class F1 Scores — All Models")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "f1_scores.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return OUTPUT_DIR / "f1_scores.png"


# ───────────────────────────────────────────────
# 4. Confusion Matrices
# ───────────────────────────────────────────────

def plot_confusion_matrices(results, save=True, show=False):
    """Grid of confusion matrices, one per model combination."""
    _ensure_output_dir()

    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        classes = r["classes"]
        cm = r["cm"]
        matrix = np.array([[cm.get((a, p), 0) for p in classes] for a in classes])

        ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{r['feature']} + {r['classifier']}", fontsize=9)

        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    # hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Confusion Matrices", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return OUTPUT_DIR / "confusion_matrices.png"


# ───────────────────────────────────────────────
# 5. Train vs Test Accuracy
# ───────────────────────────────────────────────

def plot_train_vs_test(results, save=True, show=False):
    """Side-by-side bars comparing train and test accuracy."""
    _ensure_output_dir()

    labels = [f"{r['feature']}\n{r['classifier']}" for r in results]
    train_accs = [r["train_acc"] * 100 for r in results]
    test_accs = [r["test_acc"] * 100 for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, train_accs, width, label="Train", color="#4C72B0")
    ax.bar(x + width / 2, test_accs, width, label="Test", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Train vs Test Accuracy")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "train_vs_test.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return OUTPUT_DIR / "train_vs_test.png"


# ───────────────────────────────────────────────
# 6. Dataset class distribution
# ───────────────────────────────────────────────

def plot_class_distribution(labels, save=True, show=False):
    """Pie chart of class distribution in the full dataset."""
    _ensure_output_dir()

    counts = Counter(labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%",
           colors=["#4C72B0", "#C44E52"], startangle=90)
    ax.set_title("Dataset Class Distribution")
    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return OUTPUT_DIR / "class_distribution.png"


# ───────────────────────────────────────────────
# Generate ALL charts at once
# ───────────────────────────────────────────────

def generate_all_charts(results, all_labels, show=False):
    """Produce every chart and return list of saved paths."""
    paths = [
        plot_accuracy_bars(results, show=show),
        plot_accuracy_heatmap(results, show=show),
        plot_f1_scores(results, show=show),
        plot_confusion_matrices(results, show=show),
        plot_train_vs_test(results, show=show),
        plot_class_distribution(all_labels, show=show),
    ]
    return paths
