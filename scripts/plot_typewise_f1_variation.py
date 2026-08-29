#!/usr/bin/env python3
"""Create two publication-ready views of type-wise baseline F1 variation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-ist")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "figures"

METHODS = [
    "TF-IDF",
    "BM25",
    "LSI",
    "S-BERT",
    "CodeBERT",
    "Direct",
    "RAG-LLM",
]
TYPES = ["FR", "NFR", "Mixed"]

# Pooled held-out results reported in Table 4 of the manuscript.
F1 = np.array(
    [
        [0.687, 0.677, 0.637],
        [0.684, 0.672, 0.659],
        [0.699, 0.681, 0.635],
        [0.664, 0.627, 0.667],
        [0.666, 0.667, 0.667],
        [0.729, 0.667, 0.724],
        [0.743, 0.664, 0.720],
    ]
)

# Okabe-Ito colors: colorblind-safe and consistent with Figure 1.
COLORS = {
    "FR": "#0072B2",
    "NFR": "#E69F00",
    "Mixed": "#009E73",
}
MARKERS = {
    "FR": "o",
    "NFR": "s",
    "Mixed": "D",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "semibold",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_connected_dots(output_path: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(7.15, 3.45))
    y = np.arange(len(METHODS))

    for row, y_pos in zip(F1, y, strict=True):
        ax.plot(
            [row.min(), row.max()],
            [y_pos, y_pos],
            color="#C8D0D6",
            linewidth=1.6,
            solid_capstyle="round",
            zorder=1,
        )

    for column, req_type in enumerate(TYPES):
        ax.scatter(
            F1[:, column],
            y,
            s=54,
            marker=MARKERS[req_type],
            color=COLORS[req_type],
            edgecolor="white",
            linewidth=0.8,
            label=req_type,
            zorder=3,
        )

    ax.set_yticks(y, METHODS)
    ax.invert_yaxis()
    ax.set_xlim(0.615, 0.755)
    ax.set_xticks(np.arange(0.62, 0.76, 0.02))
    ax.set_xlabel("F1")
    ax.grid(axis="x", color="#D9DEE2", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", length=0)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    for label in ax.get_xticklabels():
        label.set_fontweight("semibold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=2.0,
    )
    for label in legend.get_texts():
        label.set_fontweight("bold")

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.84)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_heatmap(output_path: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(5.7, 3.65))
    cmap = LinearSegmentedColormap.from_list(
        "paper_blue",
        ["#F2F6F8", "#B9D2DF", "#5B91B2", "#1E658F"],
    )
    norm = Normalize(vmin=0.62, vmax=0.75)
    image = ax.imshow(F1, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(TYPES)), TYPES)
    ax.set_yticks(np.arange(len(METHODS)), METHODS)
    ax.tick_params(
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        pad=7,
    )

    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")

    for row in range(F1.shape[0]):
        for column in range(F1.shape[1]):
            value = F1[row, column]
            text_color = "white" if norm(value) >= 0.57 else "#24323B"
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(-0.5, len(TYPES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.set_label("F1", fontweight="bold")
    colorbar.ax.tick_params(length=0, labelsize=8)
    colorbar.outline.set_visible(False)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight("semibold")

    fig.subplots_adjust(left=0.23, right=0.90, bottom=0.08, top=0.86)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    connected_path = OUTPUT_DIR / "typewise_f1_variation_connected_dots.png"
    heatmap_path = OUTPUT_DIR / "typewise_f1_variation_heatmap.png"
    plot_connected_dots(connected_path)
    plot_heatmap(heatmap_path)
    print(f"Wrote {connected_path}")
    print(f"Wrote {heatmap_path}")


if __name__ == "__main__":
    main()
