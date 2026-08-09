#!/usr/bin/env python3
"""Plot publication benchmark CSV outputs.

The figures generated here are intentionally conservative: grayscale-safe,
single-message plots generated only from the benchmark CSV.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "hullround": "HullRound",
    "global_bnb_baseline": r"Global $\theta$-B&B",
    "global_bnb_cached": r"Global $\theta$-B&B + cache",
    "global_bnb_cached_cutoff": r"Global $\theta$-B&B + cache/cutoff",
    "global_bnb_cached_cutoff_ordered": r"Global $\theta$-B&B + cache/cutoff/order",
    "scip": "SCIP",
    "highs": "HiGHS",
}

METHOD_ORDER = [
    "hullround",
    "global_bnb_baseline",
    "global_bnb_cached_cutoff_ordered",
    "scip",
    "highs",
]

EXACT_METHODS = ["global_bnb_baseline", "global_bnb_cached_cutoff_ordered"]

FAMILY_LABELS = {
    "economic": "Economic",
    "hull_compression": "Hull\ncompression",
    "adversarial": "Low\ncompression",
    "tight_capacity": "Tight\ncapacity",
    "many_theta": "Many\ntheta",
    "boundary": "Boundary\nGamma",
}

FAMILY_ORDER = ["economic", "hull_compression", "adversarial", "tight_capacity", "many_theta", "boundary"]

LINESTYLES = {
    "hullround": ("-", "o"),
    "global_bnb_baseline": ("--", "s"),
    "global_bnb_cached_cutoff_ordered": ("-", "^"),
    "scip": (":", "D"),
    "highs": ("-.", "x"),
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def finite(values: Iterable[float]) -> List[float]:
    vals: List[float] = []
    for value in values:
        try:
            fv = float(value)
        except Exception:
            continue
        if math.isfinite(fv):
            vals.append(fv)
    return vals


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def instance_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str]:
    return (row["family"], row["n"], row["m"], row["gamma"], row["gamma_mode"], row["seed"])


def label_method(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def label_family(family: str) -> str:
    return FAMILY_LABELS.get(family, family.replace("_", "\n"))


def plot_runtime_by_n_clean(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        method = row["method"]
        if method not in METHOD_ORDER:
            continue
        rt = f(row, "runtime_seconds")
        if math.isfinite(rt) and rt > 0:
            grouped[(method, int(float(row["n"])))].append(rt)
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method in METHOD_ORDER:
        ns = sorted(n for m, n in grouped if m == method)
        if len(ns) < 2:
            continue
        ys = [median(grouped[(method, n)]) for n in ns]
        linestyle, marker = LINESTYLES.get(method, ("-", "o"))
        ax.plot(
            ns,
            ys,
            linestyle=linestyle,
            marker=marker,
            color="black",
            linewidth=1.5,
            markersize=5,
            label=label_method(method),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of items $n$")
    ax.set_ylabel("Median runtime (seconds, log scale)")
    ax.grid(True, which="major", axis="y", color="0.82", linewidth=0.6)
    ax.grid(True, which="minor", axis="y", color="0.90", linewidth=0.4)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="best")
    save(fig, out_dir, "runtime_by_n_clean")
    save(fig, out_dir, "exact_solver_runtime_by_n")


def plot_speedup_distribution(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    baseline: Dict[Tuple[str, str, str, str, str, str], float] = {}
    for row in rows:
        if row["method"] == "global_bnb_baseline":
            rt = f(row, "runtime_seconds")
            if math.isfinite(rt) and rt > 0:
                baseline[instance_key(row)] = rt

    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "global_bnb_cached_cutoff_ordered":
            continue
        rt = f(row, "runtime_seconds")
        key = instance_key(row)
        if key in baseline and math.isfinite(rt) and rt > 0:
            grouped[row["family"]].append(baseline[key] / rt)
    families = [fam for fam in FAMILY_ORDER if grouped.get(fam)]
    if not families:
        return

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    data = [grouped[fam] for fam in families]
    bp = ax.boxplot(
        data,
        positions=np.arange(len(families)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"facecolor": "0.86", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )
    del bp
    rng = np.random.default_rng(12345)
    for idx, values in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, s=16, color="black", alpha=0.55, linewidths=0)
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle=":")
    ax.set_xticks(np.arange(len(families)), [label_family(f) for f in families], fontsize=8)
    ax.set_ylabel("Runtime speedup vs uncached exact B&B")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", color="0.85", linewidth=0.6)
    save(fig, out_dir, "speedup_distribution")
    save(fig, out_dir, "speedup_over_baseline")


def plot_certification_by_family(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    optimized = [row for row in rows if row["method"] == "global_bnb_cached_cutoff_ordered"]
    families = [fam for fam in FAMILY_ORDER if any(row["family"] == fam for row in optimized)]
    if not families:
        return
    optimal = []
    limited = []
    for fam in families:
        rs = [row for row in optimized if row["family"] == fam]
        optimal.append(sum(1 for row in rs if row["status"] == "optimal"))
        limited.append(sum(1 for row in rs if row["status"] in {"time_limit", "node_limit", "timelimit", "userinterrupt"}))

    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x, optimal, color="0.88", edgecolor="black", label="Certified optimal")
    ax.bar(x, limited, bottom=optimal, color="0.35", edgecolor="black", label="Limited with valid gap")
    ax.set_xticks(x, [label_family(f) for f in families], fontsize=8)
    ax.set_ylabel("Optimized exact B&B rows")
    ax.set_ylim(0, max(np.array(optimal) + np.array(limited)) + 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    save(fig, out_dir, "certification_by_family")
    save(fig, out_dir, "method_failure_modes")


def plot_hullround_gap_to_exact(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    hull = {instance_key(row): row for row in rows if row["method"] == "hullround"}
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "global_bnb_cached_cutoff_ordered" or row["status"] != "optimal":
            continue
        opt = f(row, "objective")
        hrow = hull.get(instance_key(row))
        hval = f(hrow, "objective") if hrow else float("nan")
        if opt > 0 and math.isfinite(hval):
            # Basis points make small but nonzero gaps visible without overstating them.
            grouped[row["family"]].append(10000.0 * max(0.0, (opt - hval) / opt))
    families = [fam for fam in FAMILY_ORDER if grouped.get(fam)]
    if not families:
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    data = [grouped[fam] for fam in families]
    ax.boxplot(
        data,
        positions=np.arange(len(families)),
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"facecolor": "0.88", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    rng = np.random.default_rng(6789)
    for idx, values in enumerate(data):
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, s=15, color="black", alpha=0.45, linewidths=0)
    ax.set_xticks(np.arange(len(families)), [label_family(f) for f in families], fontsize=8)
    ax.set_ylabel("HullRound gap to exact optimum (basis points)")
    ax.grid(True, axis="y", color="0.86", linewidth=0.6)
    save(fig, out_dir, "hullround_gap_to_exact")


def plot_theta_pruning_by_family(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        if row["method"] not in EXACT_METHODS:
            continue
        total = f(row, "theta_total")
        pruned = f(row, "theta_pruned")
        if total > 0:
            grouped[(row["method"], row["family"])].append(100.0 * pruned / total)
    families = [fam for fam in FAMILY_ORDER if any((method, fam) in grouped for method in EXACT_METHODS)]
    if not families:
        return

    x = np.arange(len(families))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for idx, method in enumerate(EXACT_METHODS):
        vals = [median(grouped.get((method, fam), [])) for fam in families]
        offset = (idx - 0.5) * width
        color = "0.82" if idx == 0 else "0.45"
        ax.bar(x + offset, vals, width=width, color=color, edgecolor="black", label=label_method(method))
    ax.set_xticks(x, [label_family(f) for f in families], fontsize=8)
    ax.set_ylabel("Median theta pruning rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    save(fig, out_dir, "theta_pruning_by_family")
    save(fig, out_dir, "theta_pruning_rate")


def plot_nodes_by_gamma(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        if row["method"] not in EXACT_METHODS:
            continue
        nodes = f(row, "nodes_explored")
        if math.isfinite(nodes) and nodes >= 0:
            grouped[(row["method"], row["gamma_mode"])].append(nodes)
    modes = [mode for mode in ["zero", "sqrt", "ten_percent", "full"] if any((m, mode) in grouped for m in EXACT_METHODS)]
    if not modes:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    for method in EXACT_METHODS:
        ys = [median(grouped.get((method, mode), [])) for mode in modes]
        linestyle, marker = LINESTYLES.get(method, ("-", "o"))
        ax.plot(range(len(modes)), ys, color="black", linestyle=linestyle, marker=marker, label=label_method(method))
    ax.set_xticks(range(len(modes)), ["0", "$\\lfloor\\sqrt{n}\\rfloor$", "$\\lfloor0.1n\\rfloor$", "$n$"][: len(modes)])
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("$\\Gamma$ regime")
    ax.set_ylabel("Median B&B nodes")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="0.86", linewidth=0.6)
    save(fig, out_dir, "bnb_nodes_by_gamma")


def plot_certification_rate_legacy(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    if not methods:
        return
    rates = []
    for method in methods:
        rs = [row for row in rows if row["method"] == method]
        if method == "hullround":
            count = sum(1 for row in rs if row["status"] == "feasible" and row.get("valid_certificate", "").lower() == "true")
        else:
            count = sum(1 for row in rs if row["status"] == "optimal")
        rates.append(count / len(rs) if rs else 0.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(range(len(methods)), rates, color="0.78", edgecolor="black")
    ax.set_xticks(range(len(methods)), [label_method(m).replace(" + ", "\n+ ") for m in methods], fontsize=7)
    ax.set_ylabel("Certified or feasible rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    save(fig, out_dir, "exact_solver_certification_rate")


def plot_staged_runtime_by_n(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    families = [fam for fam in FAMILY_ORDER if any(row["family"] == fam for row in rows)]
    methods = [m for m in ["hullround", "global_bnb_cached_cutoff_ordered", "scip", "highs"] if any(row["method"] == m for row in rows)]
    if not families or not methods:
        return
    fig, axes = plt.subplots(1, len(families), figsize=(4.0 * len(families), 3.6), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        for method in methods:
            grouped: Dict[int, List[float]] = defaultdict(list)
            for row in rows:
                if row["family"] != family or row["method"] != method:
                    continue
                rt = f(row, "runtime_seconds")
                if math.isfinite(rt) and rt > 0:
                    grouped[int(float(row["n"]))].append(rt)
            ns = sorted(grouped)
            if len(ns) < 1:
                continue
            ys = [median(grouped[n]) for n in ns]
            linestyle, marker = LINESTYLES.get(method, ("-", "o"))
            ax.plot(ns, ys, linestyle=linestyle, marker=marker, color="black", linewidth=1.4, markersize=5, label=label_method(method))
        ax.set_title(label_family(family).replace("\n", " "))
        ax.set_xlabel("Number of items $n$")
        ax.set_yscale("log")
        ax.grid(True, which="major", axis="y", color="0.84", linewidth=0.6)
        ax.grid(True, which="minor", axis="y", color="0.91", linewidth=0.4)
    axes[0].set_ylabel("Median runtime (seconds, log scale)")
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    save(fig, out_dir, "staged_runtime_by_n")


def plot_staged_certification_by_family(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    families = [fam for fam in FAMILY_ORDER if any(row["family"] == fam for row in rows)]
    methods = [m for m in ["global_bnb_cached_cutoff_ordered", "scip", "highs"] if any(row["method"] == m for row in rows)]
    if not families or not methods:
        return
    labels = []
    optimal = []
    limited = []
    for family in families:
        for method in methods:
            rs = [row for row in rows if row["family"] == family and row["method"] == method]
            if not rs:
                continue
            labels.append(f"{label_family(family).replace(chr(10), ' ')}\n{label_method(method)}")
            optimal.append(sum(1 for row in rs if row["status"] == "optimal"))
            limited.append(sum(1 for row in rs if row["status"] in {"time_limit", "node_limit", "timelimit", "userinterrupt"}))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6.8, 0.75 * len(labels)), 4.0))
    ax.bar(x, optimal, color="0.88", edgecolor="black", label="Certified optimal")
    ax.bar(x, limited, bottom=optimal, color="0.35", edgecolor="black", label="Limited")
    ax.set_xticks(x, labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Rows")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    save(fig, out_dir, "staged_certification_by_family")


def plot_staged_hullround_gap_to_exact(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    hull = {instance_key(row): row for row in rows if row["method"] == "hullround"}
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "global_bnb_cached_cutoff_ordered" or row["status"] != "optimal":
            continue
        hrow = hull.get(instance_key(row))
        opt = f(row, "objective")
        hr = f(hrow, "objective") if hrow else float("nan")
        if opt > 0 and math.isfinite(hr):
            grouped[(row["family"], row["gamma_mode"])].append(10000.0 * max(0.0, (opt - hr) / opt))
    keys = [(fam, gm) for fam in FAMILY_ORDER for gm in ["sqrt", "ten_percent", "full", "zero"] if grouped.get((fam, gm))]
    if not keys:
        return
    data = [grouped[key] for key in keys]
    labels = [f"{label_family(fam).replace(chr(10), ' ')}\n{gm}" for fam, gm in keys]
    fig, ax = plt.subplots(figsize=(max(6.8, 0.65 * len(labels)), 3.8))
    ax.boxplot(
        data,
        positions=np.arange(len(labels)),
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.3},
        boxprops={"facecolor": "0.88", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("HullRound gap to exact optimum (basis points)")
    ax.grid(True, axis="y", color="0.86", linewidth=0.6)
    save(fig, out_dir, "staged_hullround_gap_to_exact")


def plot_staged_gap_under_limits(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    limited = [
        row
        for row in rows
        if row["method"] == "global_bnb_cached_cutoff_ordered"
        and row["status"] in {"time_limit", "node_limit", "timelimit", "userinterrupt"}
        and math.isfinite(f(row, "relative_gap"))
    ]
    if not limited:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    families = [fam for fam in FAMILY_ORDER if any(row["family"] == fam for row in limited)]
    markers = {"sqrt": "o", "ten_percent": "s", "full": "^", "zero": "D"}
    for idx, family in enumerate(families):
        rs = [row for row in limited if row["family"] == family]
        xs = [f(row, "n") + (idx - 0.5) * 2.0 for row in rs]
        ys = [100.0 * f(row, "relative_gap") for row in rs]
        for gm in sorted({row["gamma_mode"] for row in rs}):
            gm_rows = [row for row in rs if row["gamma_mode"] == gm]
            ax.scatter(
                [f(row, "n") + (idx - 0.5) * 2.0 for row in gm_rows],
                [100.0 * f(row, "relative_gap") for row in gm_rows],
                marker=markers.get(gm, "o"),
                color="black",
                alpha=0.75,
                label=f"{label_family(family).replace(chr(10), ' ')} {gm}",
            )
    ax.set_xlabel("Number of items $n$")
    ax.set_ylabel("Final relative gap under limits (%)")
    ax.grid(True, axis="y", color="0.86", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="best")
    save(fig, out_dir, "staged_gap_under_limits")


def plot_staged_nodes_by_n(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "global_bnb_cached_cutoff_ordered":
            continue
        nodes = f(row, "nodes_explored")
        if math.isfinite(nodes):
            grouped[(row["family"], int(float(row["n"])))].append(nodes)
    families = [fam for fam in FAMILY_ORDER if any(key[0] == fam for key in grouped)]
    if not families:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for family in families:
        ns = sorted(n for fam, n in grouped if fam == family)
        ys = [median(grouped[(family, n)]) for n in ns]
        ax.plot(ns, ys, marker="o", linestyle="-" if family == families[0] else "--", color="black", label=label_family(family).replace("\n", " "))
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Number of items $n$")
    ax.set_ylabel("Optimized exact B&B nodes")
    ax.grid(True, axis="y", color="0.86", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out_dir, "staged_nodes_by_n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    out_dir = Path(args.output_dir)
    if not rows:
        print("No rows to plot.")
        return

    plot_runtime_by_n_clean(rows, out_dir)
    plot_speedup_distribution(rows, out_dir)
    plot_certification_by_family(rows, out_dir)
    plot_hullround_gap_to_exact(rows, out_dir)
    plot_theta_pruning_by_family(rows, out_dir)
    plot_nodes_by_gamma(rows, out_dir)
    plot_certification_rate_legacy(rows, out_dir)
    plot_staged_runtime_by_n(rows, out_dir)
    plot_staged_certification_by_family(rows, out_dir)
    plot_staged_hullround_gap_to_exact(rows, out_dir)
    plot_staged_gap_under_limits(rows, out_dir)
    plot_staged_nodes_by_n(rows, out_dir)
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
