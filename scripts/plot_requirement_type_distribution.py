#!/usr/bin/env python3
"""Plot the requirement-type composition of the five study datasets."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-ist")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "figures"

DATASETS = [
    ("eTour", "eTour"),
    ("eANCI", "eANCI"),
    ("iTrust", "iTrust"),
    ("LibEST", "LibEST"),
    ("Industrial", "Industrial"),
]
TYPES = ["FR", "NFR", "Mixed"]
COLORS = {
    "FR": "#496B86",
    "NFR": "#D99A3E",
    "Mixed": "#6F9B76",
}


def load_counts() -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for display_name, directory_name in DATASETS:
        path = DATA_ROOT / directory_name / "requirements.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        unknown = sorted(set(frame["Type"].dropna()) - set(TYPES))
        if unknown:
            raise ValueError(f"{display_name}: unsupported requirement types: {unknown}")
        counts = frame["Type"].value_counts()
        row: dict[str, int | str] = {"Dataset": display_name}
        row.update({req_type: int(counts.get(req_type, 0)) for req_type in TYPES})
        row["Total"] = int(len(frame))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_distribution(counts: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "semibold",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.15, 3.25))
    y_positions = list(range(len(counts)))
    left = pd.Series(0.0, index=counts.index)

    for req_type in TYPES:
        shares = counts[req_type] / counts["Total"] * 100
        bars = ax.barh(
            y_positions,
            shares,
            left=left,
            height=0.62,
            color=COLORS[req_type],
            edgecolor="white",
            linewidth=0.9,
            label=req_type,
            zorder=3,
        )

        for index, (bar, share, value) in enumerate(
            zip(bars, shares, counts[req_type], strict=True)
        ):
            if value == 0:
                continue
            center_x = left.iloc[index] + share / 2
            if share >= 6:
                text_color = "#1F1F1F" if req_type == "NFR" else "white"
                ax.text(
                    center_x,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8.5,
                    fontweight="bold",
                    zorder=4,
                )
            else:
                ax.annotate(
                    f"{value}",
                    xy=(center_x, bar.get_y() + bar.get_height() / 2),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#333333",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#666666",
                        "linewidth": 0.6,
                    },
                    zorder=5,
                )
        left = left + shares

    for index, total in enumerate(counts["Total"]):
        ax.text(
            101.5,
            index,
            f"n = {total}",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="semibold",
            color="#4A4A4A",
        )

    ax.set_yticks(y_positions, counts["Dataset"])
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel("Proportion of requirements (%)", labelpad=7)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=7)
    for label in ax.get_xticklabels():
        label.set_fontweight("semibold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.2,
    )
    for label in legend.get_texts():
        label.set_fontweight("bold")

    fig.subplots_adjust(left=0.145, right=0.955, bottom=0.19, top=0.86)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    counts = load_counts()
    output_path = OUTPUT_DIR / "requirement_type_distribution_refined.png"
    plot_distribution(counts, output_path)
    print(counts.to_csv(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
