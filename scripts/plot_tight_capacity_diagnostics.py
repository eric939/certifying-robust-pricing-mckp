#!/usr/bin/env python3
"""Plot tight-capacity exact B&B diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt


def _float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _exact_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("method", "").startswith("global_bnb_cached_cutoff_ordered")]


def _label(row: Dict[str, str]) -> str:
    method = row.get("method", "")
    suffix = "ref" if method.endswith("reference_bound") else "fast" if method.endswith("fast_bound") else "bnb"
    return f"{suffix}: n={row.get('n')}, m={row.get('m')}, {row.get('gamma_mode')}, s={row.get('seed')}"


def _save(fig, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def plot_time_breakdown(rows: List[Dict[str, str]], out_dir: Path) -> None:
    exact = _exact_rows(rows)
    if not exact:
        return
    labels = [_label(r) for r in exact]
    components = [
        ("root LP", "time_root_lp_total", "white", "////"),
        ("node LP", "time_node_lp_total", "0.75", ""),
        ("hull build", "time_hull_build_total", "0.55", "\\\\\\\\"),
        ("greedy LP", "time_greedy_lp_total", "0.35", ""),
        ("branch", "time_branching_total", "0.15", "...."),
        ("child gen.", "time_child_generation_total", "0.90", "xxxx"),
    ]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 4.2))
    bottom = [0.0] * len(exact)
    x = list(range(len(exact)))
    for name, key, color, hatch in components:
        vals = [_float(r, key, 0.0) for r in exact]
        ax.bar(x, vals, bottom=bottom, label=name, color=color, edgecolor="black", linewidth=0.4, hatch=hatch)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_yscale("log")
    ax.set_ylabel("Seconds (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.set_title("Tight-capacity B&B timing breakdown")
    _save(fig, out_dir, "tight_capacity_time_breakdown")


def plot_nodes_vs_gap(rows: List[Dict[str, str]], out_dir: Path) -> None:
    exact = _exact_rows(rows)
    if not exact:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for status, marker in [("optimal", "o"), ("time_limit", "s"), ("node_limit", "^")]:
        pts = [r for r in exact if r.get("status") == status]
        if not pts:
            continue
        ax.scatter(
            [_float(r, "total_nodes_explored") for r in pts],
            [_float(r, "relative_gap") for r in pts],
            marker=marker,
            facecolors="white" if status == "optimal" else "0.35",
            edgecolors="black",
            label=status,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes explored (log scale)")
    ax.set_ylabel("Final relative gap (log scale)")
    ax.legend(frameon=False)
    ax.set_title("Nodes versus remaining gap")
    _save(fig, out_dir, "tight_capacity_nodes_vs_gap")


def plot_theta_unresolved(rows: List[Dict[str, str]], out_dir: Path) -> None:
    exact = _exact_rows(rows)
    if not exact:
        return
    labels = [_label(r) for r in exact]
    unresolved = [_float(r, "unresolved_theta_count", 0.0) for r in exact]
    gaps = [_float(r, "hardest_theta_gap", 0.0) for r in exact]
    x = list(range(len(exact)))
    fig, ax1 = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 4.0))
    ax1.bar(x, unresolved, color="0.75", edgecolor="black", label="unresolved theta count")
    ax1.set_ylabel("Unresolved theta count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, gaps, color="black", marker="o", linewidth=1.2, label="hardest theta gap")
    ax2.set_ylabel("Hardest theta absolute gap")
    ax1.set_title("Unresolved theta diagnostics")
    _save(fig, out_dir, "tight_capacity_theta_unresolved")


def plot_pruning_breakdown(rows: List[Dict[str, str]], out_dir: Path) -> None:
    exact = _exact_rows(rows)
    if not exact:
        return
    labels = [_label(r) for r in exact]
    components = [
        ("bound", "total_nodes_pruned_bound", "0.80", ""),
        ("infeasible", "total_nodes_pruned_infeasible", "0.55", "////"),
        ("cutoff", "total_nodes_pruned_cutoff", "0.30", ""),
        ("integral LP", "total_integral_lp_nodes", "white", "\\\\\\\\"),
    ]
    x = list(range(len(exact)))
    bottom = [0.0] * len(exact)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 4.2))
    for name, key, color, hatch in components:
        vals = [_float(r, key, 0.0) for r in exact]
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="black", linewidth=0.4, hatch=hatch, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.set_title("B&B pruning breakdown")
    _save(fig, out_dir, "tight_capacity_pruning_breakdown")


def plot_depth_distribution(json_path: Path, out_dir: Path) -> None:
    if not json_path.exists():
        return
    details = json.loads(json_path.read_text(encoding="utf-8"))
    depths: List[float] = []
    for detail in details:
        for theta in detail.get("per_theta", []):
            samples = theta.get("diagnostics", {}).get("node_depth_samples", [])
            depths.extend(float(v) for v in samples if math.isfinite(float(v)))
    if not depths:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.hist(depths, bins=min(30, max(5, len(set(depths)))), color="0.75", edgecolor="black")
    ax.set_xlabel("Sampled node depth")
    ax.set_ylabel("Frequency")
    ax.set_title("Sampled depth distribution")
    _save(fig, out_dir, "tight_capacity_depth_distribution")


def plot_branching_arity(rows: List[Dict[str, str]], out_dir: Path) -> None:
    exact = _exact_rows(rows)
    if not exact:
        return
    labels = [_label(r) for r in exact]
    vals = [_float(r, "average_branching_arity", 0.0) for r in exact]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.5 * len(labels)), 3.6))
    ax.bar(range(len(vals)), vals, color="0.70", edgecolor="black")
    ax.set_ylabel("Average branching arity")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Branching arity")
    _save(fig, out_dir, "tight_capacity_branching_arity")


def _matched_reference_fast(rows: List[Dict[str, str]]) -> List[tuple[tuple[str, str, str, str], Dict[str, str], Dict[str, str]]]:
    grouped: Dict[tuple[str, str, str, str], Dict[str, Dict[str, str]]] = {}
    for row in rows:
        key = (row.get("n", ""), row.get("m", ""), row.get("gamma_mode", ""), row.get("seed", ""))
        grouped.setdefault(key, {})[row.get("method", "")] = row
    matched = []
    for key, by_method in grouped.items():
        ref = by_method.get("global_bnb_cached_cutoff_ordered_reference_bound")
        fast = by_method.get("global_bnb_cached_cutoff_ordered_fast_bound")
        if ref is not None and fast is not None:
            matched.append((key, ref, fast))
    return sorted(matched)


def plot_node_lp_before_after(rows: List[Dict[str, str]], out_dir: Path) -> None:
    matched = _matched_reference_fast(rows)
    if not matched:
        return
    labels = [f"n={k[0]},m={k[1]},{k[2]}" for k, _r, _f in matched]
    x = list(range(len(matched)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.5, 0.65 * len(labels)), 3.8))
    ax.bar([i - width / 2 for i in x], [_float(r, "time_node_lp_total") for _k, r, _f in matched], width, color="0.75", edgecolor="black", label="reference")
    ax.bar([i + width / 2 for i in x], [_float(f, "time_node_lp_total") for _k, _r, f in matched], width, color="0.35", edgecolor="black", label="fast")
    ax.set_ylabel("Node LP-bound time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(frameon=False)
    ax.set_title("Node LP-bound time before/after fast bound")
    _save(fig, out_dir, "node_lp_time_before_after")


def plot_runtime_before_after(rows: List[Dict[str, str]], out_dir: Path) -> None:
    matched = _matched_reference_fast(rows)
    if not matched:
        return
    labels = [f"n={k[0]},m={k[1]},{k[2]}" for k, _r, _f in matched]
    x = list(range(len(matched)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.5, 0.65 * len(labels)), 3.8))
    ax.bar([i - width / 2 for i in x], [_float(r, "runtime_seconds") for _k, r, _f in matched], width, color="0.75", edgecolor="black", label="reference")
    ax.bar([i + width / 2 for i in x], [_float(f, "runtime_seconds") for _k, _r, f in matched], width, color="0.35", edgecolor="black", label="fast")
    ax.set_ylabel("Runtime (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(frameon=False)
    ax.set_title("Runtime before/after fast bound")
    _save(fig, out_dir, "runtime_before_after")


def plot_nodes_and_gap_before_after(rows: List[Dict[str, str]], out_dir: Path) -> None:
    matched = _matched_reference_fast(rows)
    if not matched:
        return
    labels = [f"n={k[0]},m={k[1]},{k[2]}" for k, _r, _f in matched]
    x = list(range(len(matched)))
    fig, ax1 = plt.subplots(figsize=(max(7.0, 0.7 * len(labels)), 4.0))
    ax1.plot(x, [_float(r, "total_nodes_explored") for _k, r, _f in matched], marker="o", color="0.65", label="ref nodes")
    ax1.plot(x, [_float(f, "total_nodes_explored") for _k, _r, f in matched], marker="s", color="black", label="fast nodes")
    ax1.set_yscale("log")
    ax1.set_ylabel("Nodes explored (log scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, [_float(r, "relative_gap") for _k, r, _f in matched], linestyle="--", marker="o", color="0.65", label="ref gap")
    ax2.plot(x, [_float(f, "relative_gap") for _k, _r, f in matched], linestyle="--", marker="s", color="black", label="fast gap")
    ax2.set_ylabel("Final relative gap")
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, ncol=2, fontsize=8, frameon=False)
    ax1.set_title("Nodes and gaps before/after fast bound")
    _save(fig, out_dir, "nodes_and_gap_before_after")


def plot_pruning_breakdown_fastbound(rows: List[Dict[str, str]], out_dir: Path) -> None:
    fast = [r for r in rows if r.get("method") == "global_bnb_cached_cutoff_ordered_fast_bound"]
    if not fast:
        return
    labels = [f"n={r.get('n')},m={r.get('m')},{r.get('gamma_mode')}" for r in fast]
    components = [
        ("cheap prebound", "cheap_prebound_prunes", "white", "////"),
        ("min-cost", "min_cost_infeasibility_prunes", "0.85", ""),
        ("bound", "total_nodes_pruned_bound", "0.55", "\\\\\\\\"),
        ("cutoff", "total_nodes_pruned_cutoff", "0.30", ""),
        ("integral LP", "total_integral_lp_nodes", "0.15", "...."),
    ]
    x = list(range(len(fast)))
    bottom = [0.0] * len(fast)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.65 * len(labels)), 4.0))
    for name, key, color, hatch in components:
        vals = [_float(r, key, 0.0) for r in fast]
        ax.bar(x, vals, bottom=bottom, label=name, color=color, edgecolor="black", linewidth=0.4, hatch=hatch)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.set_title("Fast-bound pruning breakdown")
    _save(fig, out_dir, "pruning_breakdown_fastbound")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/tight_capacity_diagnostics/tight_capacity_diagnostics.csv")
    parser.add_argument("--json", default="results/tight_capacity_diagnostics/tight_capacity_diagnostics.json")
    parser.add_argument("--output-dir", default="paper_versions/v2/figures/tight_capacity_diagnostics")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"missing input CSV: {in_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with in_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    plot_time_breakdown(rows, out_dir)
    plot_nodes_vs_gap(rows, out_dir)
    plot_theta_unresolved(rows, out_dir)
    plot_pruning_breakdown(rows, out_dir)
    plot_depth_distribution(Path(args.json), out_dir)
    plot_branching_arity(rows, out_dir)
    plot_node_lp_before_after(rows, out_dir)
    plot_runtime_before_after(rows, out_dir)
    plot_nodes_and_gap_before_after(rows, out_dir)
    plot_pruning_breakdown_fastbound(rows, out_dir)


if __name__ == "__main__":
    main()
