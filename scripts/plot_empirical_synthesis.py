#!/usr/bin/env python3
"""Generate a compact empirical-synthesis figure for the v2 manuscript.

The figure is intentionally built from preserved CSV outputs rather than
hard-coded numbers. It summarizes the evidence for the paper's main empirical
message: HullRound is a strong robust-feasible primal method, exact $\theta$-B&B
certifies non-tight regimes and exposes tight-capacity limits, and the
semi-synthetic pricing application shows an interpretable revenue-risk frontier.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


FAMILY_LABELS = {
    "economic": "Economic",
    "hull_compression": "Hull\ncompression",
    "adversarial": "Low\ncompression",
    "tight_capacity": "Tight\ncapacity",
    "many_theta": "Many\n$\\theta$",
    "boundary": "Boundary",
}

FAMILY_ORDER = ["economic", "hull_compression", "adversarial", "tight_capacity", "many_theta", "boundary"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ff(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, ""))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def median(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(statistics.median(xs)) if xs else float("nan")


def instance_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str]:
    return (row["family"], row["n"], row["m"], row["gamma"], row["gamma_mode"], row["seed"])


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


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(
    publication_rows: List[Dict[str, str]],
    final_rows: List[Dict[str, str]],
    pathc_policy: List[Dict[str, str]],
    pathc_stress: List[Dict[str, str]],
    out_dir: Path,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8))

    # Panel A: HullRound gap to exact on final non-tight certified rows.
    ax = axs[0, 0]
    hull = {instance_key(r): r for r in final_rows if r.get("method") == "hullround"}
    gaps_by_family: Dict[str, List[float]] = defaultdict(list)
    for row in final_rows:
        if row.get("method") != "global_bnb_cached_cutoff_ordered" or row.get("status") != "optimal":
            continue
        opt = ff(row, "objective")
        hrow = hull.get(instance_key(row))
        hval = ff(hrow, "objective") if hrow else float("nan")
        if opt > 0 and math.isfinite(hval):
            gaps_by_family[row["family"]].append(10000.0 * max(0.0, (opt - hval) / opt))
    fams = [f for f in FAMILY_ORDER if gaps_by_family.get(f)]
    if fams:
        data = [gaps_by_family[f] for f in fams]
        ax.boxplot(
            data,
            positions=np.arange(len(fams)),
            widths=0.52,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.2},
            boxprops={"facecolor": "0.86", "edgecolor": "black"},
            whiskerprops={"color": "black"},
            capprops={"color": "black"},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "black", "markeredgecolor": "black"},
        )
        ax.set_xticks(np.arange(len(fams)), [FAMILY_LABELS.get(f, f) for f in fams], fontsize=7)
    ax.set_ylabel("HullRound gap (basis points)")
    ax.set_title("(a) Near-exact primal quality")

    # Panel B: exact status by family on publication benchmark.
    ax = axs[0, 1]
    opt_rows = [r for r in publication_rows if r.get("method") == "global_bnb_cached_cutoff_ordered"]
    fams = [f for f in FAMILY_ORDER if any(r.get("family") == f for r in opt_rows)]
    certified, limited = [], []
    for fam in fams:
        rs = [r for r in opt_rows if r.get("family") == fam]
        certified.append(sum(1 for r in rs if r.get("status") == "optimal"))
        limited.append(sum(1 for r in rs if r.get("status") in {"time_limit", "node_limit", "timelimit", "userinterrupt"}))
    x = np.arange(len(fams))
    ax.bar(x, certified, color="0.86", edgecolor="black", label="Certified")
    ax.bar(x, limited, bottom=certified, color="0.35", edgecolor="black", label="Limited")
    ax.set_xticks(x, [FAMILY_LABELS.get(f, f) for f in fams], fontsize=7)
    ax.set_ylabel("Optimized exact rows")
    ax.set_title("(b) Certification regimes")
    ax.legend(frameon=False, loc="upper right")

    # Panel C/D: Path C frontier.
    hull_policy = [r for r in pathc_policy if r.get("method") == "HullRound"]
    if hull_policy:
        by_gamma: Dict[int, List[Dict[str, str]]] = defaultdict(list)
        for r in hull_policy:
            by_gamma[int(float(r["eval_gamma"]))].append(r)
        gamma_vals = sorted(by_gamma)
        max_gamma = max(gamma_vals) if gamma_vals else 1
        gx = [g / max_gamma for g in gamma_vals]
        rev = [median(ff(r, "revenue_ratio") for r in by_gamma[g]) for g in gamma_vals]
        changed = [median(ff(r, "share_changed_vs_nominal") for r in by_gamma[g]) for g in gamma_vals]
        stress_by: Dict[Tuple[int, str], List[Dict[str, str]]] = defaultdict(list)
        for r in pathc_stress:
            if r.get("method") == "HullRound":
                stress_by[(int(float(r["eval_gamma"])), r["protocol"])].append(r)
        iid = [median(ff(r, "violation_probability") for r in stress_by[(g, "iid")]) for g in gamma_vals]
        block = [median(ff(r, "violation_probability") for r in stress_by[(g, "segment_block")]) for g in gamma_vals]

        ax = axs[1, 0]
        ax.plot(block, rev, marker="s", color="black", label="Segment-block")
        ax.plot(iid, rev, marker="o", color="0.45", linestyle="--", label="IID")
        for g, bx, ry in zip(gamma_vals, block, rev):
            if g in {0, gamma_vals[min(len(gamma_vals) - 1, 2)], max_gamma}:
                ax.annotate(str(g), (bx, ry), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Stress violation probability")
        ax.set_ylabel("Revenue ratio")
        ax.set_title("(c) Revenue-risk frontier")
        ax.legend(frameon=False, loc="lower right")

        ax = axs[1, 1]
        ax.plot(gx, changed, marker="o", color="black", label="Share changed")
        ax.plot(gx, block, marker="s", color="0.45", linestyle="--", label="Segment-block violation")
        ax.set_xlabel(r"$\Gamma/n$")
        ax.set_ylabel("Median rate")
        ax.set_title("(d) Protection vs. price movement")
        ax.legend(frameon=False, loc="best")

    for ax in axs.flat:
        ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    fig.tight_layout()
    save(fig, out_dir, "empirical_synthesis_dashboard")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publication", default="results/publication_benchmarks/publication_benchmarks.csv")
    parser.add_argument("--final", default="results/final_publication_experiment/publication_benchmarks.csv")
    parser.add_argument("--pathc-dir", default="results/pathC/semisynthetic_application")
    parser.add_argument("--output-dir", default="paper_versions/v2/figures/empirical_synthesis")
    args = parser.parse_args()

    style()
    pathc = Path(args.pathc_dir)
    plot_dashboard(
        read_csv(Path(args.publication)),
        read_csv(Path(args.final)),
        read_csv(pathc / "pathC_policy_results.csv"),
        read_csv(pathc / "pathC_stress_results.csv"),
        Path(args.output_dir),
    )
    print(f"Wrote empirical synthesis figures to {args.output_dir}")


if __name__ == "__main__":
    main()
