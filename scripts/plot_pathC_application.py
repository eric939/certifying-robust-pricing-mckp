#!/usr/bin/env python3
"""Generate Path C application figures."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite(vals: Iterable[object]) -> List[float]:
    out = []
    for v in vals:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def med(vals: Iterable[object]) -> float:
    xs = finite(vals)
    return float(statistics.median(xs)) if xs else float("nan")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.5,
        }
    )


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/pathC/semisynthetic_application")
    parser.add_argument("--output-dir", default="paper_versions/v2/figures/pathC_application")
    args = parser.parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir)
    style()
    policy = read_csv(inp / "pathC_policy_results.csv")
    stress = read_csv(inp / "pathC_stress_results.csv")
    segment = read_csv(inp / "pathC_segment_diagnostics.csv")
    exact = read_csv(inp / "pathC_exact_subset_results.csv")

    hull = [r for r in policy if r.get("method") == "HullRound"]
    if hull:
        by_gamma = defaultdict(list)
        for r in hull:
            by_gamma[int(float(r["eval_gamma"]))].append(r)
        gamma_vals = sorted(by_gamma)
        x = [g / max(gamma_vals[-1], 1) for g in gamma_vals]
        rev = [med(r["revenue_ratio"] for r in by_gamma[g]) for g in gamma_vals]
        cert = [med(r["robust_certificate"] for r in by_gamma[g]) for g in gamma_vals]
        changed = [med(r["share_changed_vs_nominal"] for r in by_gamma[g]) for g in gamma_vals]
        stress_by = defaultdict(list)
        for r in stress:
            if r.get("method") == "HullRound":
                stress_by[(int(float(r["eval_gamma"])), r["protocol"])].append(r)
        iid = [med(r["violation_probability"] for r in stress_by[(g, "iid")]) for g in gamma_vals]
        block = [med(r["violation_probability"] for r in stress_by[(g, "segment_block")]) for g in gamma_vals]
        heavy = [med(r["violation_probability"] for r in stress_by[(g, "heavy_tail")]) for g in gamma_vals]

        fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
        axs[0, 0].plot(x, rev, marker="o", color="black")
        axs[0, 0].set_ylabel("Revenue ratio")
        axs[0, 0].set_title("Revenue")
        axs[0, 1].plot(x, cert, marker="s", color="black")
        axs[0, 1].axhline(0, color="0.35", linestyle=":", linewidth=1)
        axs[0, 1].set_ylabel("Certificate")
        axs[0, 1].set_title("Exact robust residual")
        axs[1, 0].plot(x, iid, marker="o", color="black", label="iid")
        axs[1, 0].plot(x, block, marker="s", color="0.35", linestyle="--", label="segment block")
        axs[1, 0].plot(x, heavy, marker="^", color="0.55", linestyle=":", label="heavy tail")
        axs[1, 0].set_ylabel("Violation probability")
        axs[1, 0].set_xlabel(r"$\Gamma/n$")
        axs[1, 0].legend(frameon=False)
        axs[1, 1].plot(x, changed, marker="o", color="black")
        axs[1, 1].set_ylabel("Share changed")
        axs[1, 1].set_xlabel(r"$\Gamma/n$")
        axs[1, 1].set_title("Price movement")
        fig.suptitle("Semi-synthetic robust pricing frontier", y=1.02)
        save(fig, out, "pathC_revenue_certificate_stress")

        fig, ax = plt.subplots(figsize=(5.5, 3.7))
        ax.plot(iid, rev, marker="o", color="black", label="iid stress")
        ax.plot(block, rev, marker="s", color="0.35", linestyle="--", label="segment block")
        ax.set_xlabel("Violation probability")
        ax.set_ylabel("Revenue ratio")
        ax.set_title("Revenue-risk frontier")
        ax.legend(frameon=False)
        save(fig, out, "pathC_robustness_frontier")

    if segment:
        # Use the most common nonzero HullRound gamma closest to sqrt(n).
        hull_seg = [r for r in segment if r.get("method") == "HullRound"]
        gammas = sorted(set(int(float(r["eval_gamma"])) for r in hull_seg if int(float(r["eval_gamma"])) > 0))
        target = gammas[0] if gammas else None
        if target is not None:
            groups = defaultdict(list)
            for r in hull_seg:
                if int(float(r["eval_gamma"])) == target:
                    groups[r["segment"]].append(r)
            labels = sorted(groups)
            signed = [med(r["median_signed_price_change"] for r in groups[s]) for s in labels]
            share = [med(r["share_changed"] for r in groups[s]) for s in labels]
            fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
            axs[0].bar(range(len(labels)), signed, color="0.35")
            axs[0].axhline(0, color="black", linewidth=0.8)
            axs[0].set_xticks(range(len(labels)), [s.replace("_", "\n") for s in labels])
            axs[0].set_ylabel("Median signed price change")
            axs[1].bar(range(len(labels)), share, color="0.65")
            axs[1].set_xticks(range(len(labels)), [s.replace("_", "\n") for s in labels])
            axs[1].set_ylabel("Share changed")
            fig.suptitle("Segment-level price movement")
            save(fig, out, "pathC_segment_impacts")

            fig, ax = plt.subplots(figsize=(5.8, 3.5))
            ax.bar(range(len(labels)), share, color="0.65", label="share changed")
            ax.set_xticks(range(len(labels)), [s.replace("_", "\n") for s in labels])
            ax.set_ylabel("Share changed")
            ax.set_title("Price adjustment diagnostics by segment")
            save(fig, out, "pathC_price_adjustment_diagnostics")

    if exact:
        gaps = [
            float(r["gap_to_exact"]) for r in exact
            if r.get("method") == "HullRound" and r.get("gap_to_exact") not in {"", "nan"} and math.isfinite(float(r["gap_to_exact"]))
        ]
        if gaps:
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            ax.hist([10000 * g for g in gaps], bins=min(10, max(3, len(gaps))), color="0.55", edgecolor="black")
            ax.set_xlabel("HullRound gap to exact (basis points)")
            ax.set_ylabel("Rows")
            ax.set_title("Exact subset quality check")
            save(fig, out, "pathC_exact_subset_gap")

    print(f"Wrote Path C figures to {out}")


if __name__ == "__main__":
    main()
