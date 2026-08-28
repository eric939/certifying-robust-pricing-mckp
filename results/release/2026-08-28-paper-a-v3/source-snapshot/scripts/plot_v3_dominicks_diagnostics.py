#!/usr/bin/env python3
"""Plot the fixed-roster availability and complete-week shock diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    calibration = payload["calibration"]
    boundary = int(payload["data"]["week_boundaries"]["calibration_end"])
    n_items = int(payload["data"]["modeled_items"])

    weeks = calibration["availability_week_numbers"]
    counts = calibration["positive_price_item_counts"]
    cal_weeks = calibration["calibration_week_numbers"]
    cal_scores = calibration["calibration_scores"]
    test_weeks = calibration["test_week_numbers"]
    test_scores = calibration["test_scores"]
    radius = float(calibration["radius"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.45, 4.7), sharex=True)

    axes[0].step(weeks, counts, where="mid", color="#1f4e79", linewidth=1.2)
    axes[0].axhline(n_items, color="0.45", linewidth=0.8, linestyle="--")
    axes[0].axvline(boundary + 0.5, color="0.25", linewidth=0.8)
    axes[0].set_ylabel("Items with\npositive price")
    axes[0].set_ylim(-0.4, n_items + 0.8)
    axes[0].set_yticks([0, 3, 6, 9, n_items])
    axes[0].text(
        boundary - 2,
        0.7,
        "calibration",
        ha="right",
        va="bottom",
        color="0.35",
    )
    axes[0].text(
        boundary + 3,
        0.7,
        "future block",
        ha="left",
        va="bottom",
        color="0.35",
    )
    axes[0].text(
        weeks[-1] - 1,
        2.0,
        f"{len(test_weeks)}/{weeks[-1] - boundary} weeks complete",
        ha="right",
        va="bottom",
        color="#1f4e79",
    )

    axes[1].scatter(
        cal_weeks,
        cal_scores,
        s=16,
        marker="o",
        facecolor="#5b9bd5",
        edgecolor="none",
        label="Calibration complete weeks",
    )
    axes[1].scatter(
        test_weeks,
        test_scores,
        s=23,
        marker="D",
        facecolor="#c44e52",
        edgecolor="none",
        label="Test complete weeks",
    )
    axes[1].axhline(
        radius,
        color="0.15",
        linewidth=1.0,
        linestyle="--",
        label=f"Fixed radius r={radius:.3f}",
    )
    axes[1].axvline(boundary + 0.5, color="0.25", linewidth=0.8)
    axes[1].set_ylabel(r"Gauge $R_\Gamma(h)$")
    axes[1].set_xlabel("Dominick's database week")
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(loc="lower left", frameon=False, ncol=1)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.6)

    fig.tight_layout(h_pad=0.45)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
