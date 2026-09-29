#!/usr/bin/env python3
"""Draw the aggregate FLAIR ablation effect with patient-bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=PROJECT / "reports/evaluation_summary.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/flair_effect_ci.png")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    pairs = summary["paired_comparisons"]
    comparisons = [
        ("DWI + FLAIR vs DWI", pairs["503_minus_501"]),
        ("DWI + ADC + FLAIR vs DWI + ADC", pairs["504_minus_502"]),
    ]
    metrics = [
        ("Dice", "mean_dice_difference", "mean_dice_difference_ci95"),
        ("Lesion F1", "mean_lesion_f1_difference", "mean_lesion_f1_difference_ci95"),
        ("Small-lesion recall (<1 mL)", "small_lesion_recall_difference", "small_lesion_recall_difference_ci95"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 2.6), sharey=True, layout="constrained")
    for ax, (title, mean_key, ci_key) in zip(axes, metrics):
        ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
        for y, (label, comparison) in enumerate(comparisons):
            mean = comparison[mean_key]
            low, high = comparison[ci_key]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color="#2b6cb0" if y == 0 else "#c05621",
                capsize=4,
                markersize=7,
                linewidth=1.8,
            )
            ax.annotate(
                f"{mean:+.3f}",
                (mean, y),
                xytext=(0, -16 if y == 0 else 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.07, 0.07)
        ax.set_xticks([-0.06, -0.03, 0.0, 0.03, 0.06])
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("FLAIR addition − baseline")

    axes[0].set_yticks(range(len(comparisons)), labels=[name for name, _ in comparisons])
    axes[0].invert_yaxis()
    fig.suptitle("FLAIR effect on 50 matched validation cases (95% patient bootstrap CI)", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
