#!/usr/bin/env python3
"""Render the exact three-item certificate mechanism used in Paper A v3."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )
    blue = "#2F5D8A"
    orange = "#C46A2D"
    grey = "#A7AFB7"

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.65))

    # Panel (a): original sorted-Gamma feasibility check.
    selected_losses = np.array([0.5, 1.5, 0.5])
    colors = [grey, orange, grey]
    axes[0].bar(np.arange(3), selected_losses, color=colors, width=0.62)
    axes[0].axhline(1.5, color=orange, linestyle="--", linewidth=0.9)
    axes[0].text(2.45, 1.52, r"top-$\Gamma$ loss $=1.5$", ha="right", va="bottom")
    axes[0].text(0.02, 0.96, "nominal surplus = 3.0",
                 transform=axes[0].transAxes, va="top")
    axes[0].text(0.02, 0.57, "robust residual\n$=3.0-1.5=1.5$",
                 transform=axes[0].transAxes, va="top")
    axes[0].set_xticks(range(3), ["item 1", "item 2", "item 3"])
    axes[0].set_ylabel("selected endpoint loss")
    axes[0].set_ylim(0, 1.9)
    axes[0].set_title(r"(a) Direct robust check ($\Gamma=1$)")

    # Panel (b): all original thresholds, with exact LP upper values.
    theta = np.array([0.0, 0.25, 0.5, 1.5])
    upper = np.array([40.3, 446.0 / 11.0, 163.0 / 4.0, 163.0 / 4.0])
    axes[1].plot(theta, upper, marker="o", color=blue, linewidth=1.4)
    axes[1].scatter([0.5], [40.75], s=55, color=orange, zorder=3)
    axes[1].axhline(40.75, color=orange, linestyle="--", linewidth=0.9)
    axes[1].annotate(
        r"$U=40.75$",
        xy=(0.5, 40.75),
        xytext=(0.67, 40.37),
        arrowprops={"arrowstyle": "->", "color": orange, "lw": 0.8},
    )
    axes[1].set_xlabel(r"original threshold $\theta$")
    axes[1].set_ylabel(r"fixed-family LP value $U_\theta$")
    axes[1].set_xticks(theta, ["0", "0.25", "0.5", "1.5"])
    axes[1].set_ylim(40.2, 40.9)
    axes[1].set_title("(b) Complete threshold family")

    # Panel (c): global interval; Z is deliberately shown as an unknown point.
    axes[2].set_xlim(39.9, 40.86)
    axes[2].set_ylim(-0.55, 0.55)
    axes[2].hlines(0, 40.0, 40.75, color=blue, linewidth=7, alpha=0.22)
    axes[2].plot([40.0, 40.75], [0, 0], "o", color=blue, markersize=6)
    axes[2].annotate(r"$H_x=40$", (40.0, 0), xytext=(40.0, -0.25), ha="center")
    axes[2].annotate(r"$U=40.75$", (40.75, 0), xytext=(40.75, -0.25), ha="center")
    axes[2].text(40.375, 0.12, r"$Z^{\rm rob}$ must lie here", ha="center")
    axes[2].annotate(
        "revenue left on the table\n" r"$\leq U-H_x=0.75$",
        xy=(40.375, -0.02),
        xytext=(40.375, 0.43),
        ha="center",
        arrowprops={"arrowstyle": "-[,widthB=3.0,lengthB=0.4", "lw": 0.8},
    )
    axes[2].set_yticks([])
    axes[2].set_xlabel("nominal revenue")
    axes[2].set_title("(c) Verifiable global interval")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[2].spines["left"].set_visible(False)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
