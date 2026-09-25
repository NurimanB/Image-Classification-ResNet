"""Plotting helpers — each function saves a figure under results/."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend, safe for CI / no-display environments
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def plot_topk_predictions(image_paths: dict[str, str], predictions: dict[str, list[dict]],
                           out_path: str = None) -> str:
    """Image on the left, horizontal top-k confidence bars on the right, one row per image."""
    out_path = out_path or os.path.join(RESULTS_DIR, "topk_predictions.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = len(image_paths)
    fig = plt.figure(figsize=(14, 4 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig, wspace=0.05, hspace=0.5)

    for row, (name, path) in enumerate(image_paths.items()):
        preds = predictions[name]

        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.imshow(Image.open(path).convert("RGB"))
        ax_img.set_title(f"Input: {name}", fontsize=11, fontweight="bold")
        ax_img.axis("off")

        ax_bar = fig.add_subplot(gs[row, 1])
        classes = [p["class"].replace("_", " ") for p in reversed(preds)]
        probs = [p["probability"] for p in reversed(preds)]
        colors = ["#2196F3" if i == len(preds) - 1 else "#90CAF9" for i in range(len(preds))]
        bars = ax_bar.barh(classes, probs, color=colors, edgecolor="white")
        for bar, pct in zip(bars, probs):
            ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}%", va="center", fontsize=9)
        ax_bar.set_xlim(0, max(probs) * 1.2)
        ax_bar.set_xlabel("Confidence (%)")
        ax_bar.set_title(f"Top-{len(preds)} Predictions – {name}", fontsize=10)
        ax_bar.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path


def plot_model_comparison(comparison: dict, out_path: str = None) -> str:
    """Three side-by-side bar charts: params, inference time, published accuracy."""
    out_path = out_path or os.path.join(RESULTS_DIR, "resnet_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model_names = list(comparison.keys())
    params = [comparison[m]["params_M"] for m in model_names]
    inf_time = [comparison[m]["avg_inference_ms"] for m in model_names]
    accuracy = [comparison[m].get("published_top1_acc") for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#42A5F5", "#1E88E5", "#1565C0"][: len(model_names)]

    for ax, data, ylabel, title in zip(
        axes,
        [params, inf_time, accuracy],
        ["Parameters (M)", "Avg Inference (ms, CPU)", "Top-1 Accuracy (%, published)"],
        ["Model Size", "Inference Speed", "ImageNet Accuracy"],
    ):
        bars = ax.bar(model_names, data, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(data) * 0.02,
                    f"{val}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(data) * 1.2)

    fig.suptitle("ResNet Variant Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path


def plot_batch_vs_individual(individual_times: list[float], batch_total_ms: float,
                              speedup: float, out_path: str = None) -> str:
    """Per-image individual timings + total-time comparison against one batched forward pass."""
    out_path = out_path or os.path.join(RESULTS_DIR, "batch_vs_individual.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = len(individual_times)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(1, n + 1), individual_times, color="#42A5F5", label="Individual")
    ax1.axhline(batch_total_ms / n, color="#E53935", linestyle="--", linewidth=2,
                label=f"Batch avg ({batch_total_ms / n:.1f} ms)")
    ax1.set_xlabel("Image index")
    ax1.set_ylabel("Inference time (ms)")
    ax1.set_title("Per-image Inference Time")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    categories = ["Individual\n(total)", "Batch\n(total)"]
    totals = [sum(individual_times), batch_total_ms]
    bars = ax2.bar(categories, totals, color=["#42A5F5", "#43A047"], width=0.4, edgecolor="white")
    for bar, val in zip(bars, totals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{val:.0f} ms", ha="center", fontweight="bold")
    ax2.set_ylabel("Total time (ms)")
    ax2.set_title(f"Total Processing Time (speedup: {speedup:.2f}×)")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Batch vs. Individual Inference ({n} images)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path
