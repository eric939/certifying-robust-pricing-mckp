#!/usr/bin/env python3
"""Generate the canonical, controlled computational evidence for Paper A.

All manuscript numbers and plots produced by this driver are derived from the
CSV/JSON files in one immutable dated result directory.  The driver is
deterministic apart from measured wall-clock time.  It deliberately separates
exact-oracle comparisons, HullRound certificate checks, and scalability
measurements so that timing observations are never used as proofs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments_nested"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments_nested._common import (  # noqa: E402
    build_prefix_instance,
    extract_arrays,
    hull_sizes_for_theta,
    make_master_portfolio,
)
from robust_mckp import GlobalThetaBNBConfig, solve  # noqa: E402
from robust_mckp.certificate import certificate_is_feasible, compute_certificate  # noqa: E402
from robust_mckp.exact_bnb import solve_global_theta_bnb  # noqa: E402
from scripts.run_publishable_experiments import (  # noqa: E402
    hullround_metrics,
    solve_full_robust_highs,
)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else float("nan")


def _maximum(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return max(vals) if vals else float("nan")


def _gamma_values(n: int) -> List[int]:
    return sorted({0, int(math.floor(math.sqrt(n))), int(math.floor(0.1 * n))})


def _instance(seed: int, n: int, m: int, gamma: int):
    master = make_master_portfolio(seed=seed, n_max=n, m_max=m, min_admissible_menu=min(8, m))
    return build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma).instance


def exact_comparisons(quick: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seeds = [2001] if quick else list(range(2001, 2006))
    ns = [12] if quick else [12, 20, 30]
    for seed in seeds:
        for n in ns:
            for gamma in _gamma_values(n):
                instance = _instance(seed, n, 8, gamma)
                t0 = time.perf_counter()
                bnb = solve_global_theta_bnb(
                    instance,
                    GlobalThetaBNBConfig(
                        tolerance=1e-9,
                        use_hullround_incumbent=True,
                        use_fast_residual_lp_bound=True,
                    ),
                )
                bnb_time = time.perf_counter() - t0
                highs = solve_full_robust_highs(instance, time_limit=30.0)
                hr = solve(instance)
                opt = float(bnb.objective_value)
                highs_obj = float(highs.get("objective", float("nan")))
                agreement = abs(opt - highs_obj) if math.isfinite(highs_obj) else float("nan")
                hr_gap = max(0.0, (opt - hr.objective) / opt) if opt > 0 else float("nan")
                rows.append(
                    {
                        "seed": seed,
                        "n": n,
                        "m": 8,
                        "gamma": gamma,
                        "bnb_status": bnb.status,
                        "bnb_objective": opt,
                        "bnb_runtime_seconds": bnb_time,
                        "bnb_absolute_gap": float(bnb.absolute_gap),
                        "highs_status": highs.get("status", "unknown"),
                        "highs_certified": bool(highs.get("certified", False)),
                        "highs_objective": highs_obj,
                        "highs_runtime_seconds": float(highs.get("runtime_s", float("nan"))),
                        "bnb_highs_absolute_difference": agreement,
                        "hullround_objective": float(hr.objective),
                        "hullround_relative_gap_to_bnb": hr_gap,
                        "hullround_robust_residual": float(compute_certificate(instance, hr.selections)),
                        "hullround_certificate_feasible": bool(hr.is_feasible),
                    }
                )
    return rows


def certificate_study(quick: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seeds = [1001] if quick else list(range(1001, 1006))
    ns = [30] if quick else [30, 60, 120]
    for seed in seeds:
        for n in ns:
            for gamma in _gamma_values(n):
                instance = _instance(seed, n, 10, gamma)
                metrics = hullround_metrics(instance, validate_lp=True)
                metrics.pop("selections", None)
                rows.append({"seed": seed, "n": n, "m": 10, "gamma": gamma, **metrics})
    return rows


def scalability_study(quick: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seeds = [4001] if quick else [4001, 4002, 4003]
    ns = [30] if quick else [30, 60, 120, 200]
    ms = [8] if quick else [8, 16]
    for seed in seeds:
        for n in ns:
            for m in ms:
                gamma = int(math.floor(math.sqrt(n)))
                instance = _instance(seed, n, m, gamma)
                sol = solve(instance)
                values, margins, uncertainties = extract_arrays(instance)
                raw, undominated, hull = hull_sizes_for_theta(values, margins, uncertainties, sol.theta)
                instr = (sol.metadata or {}).get("instrumentation", {})
                rows.append(
                    {
                        "seed": seed,
                        "n": n,
                        "m": m,
                        "gamma": gamma,
                        "runtime_seconds": float(sol.elapsed),
                        "objective": float(sol.objective),
                        "robust_residual": float(compute_certificate(instance, sol.selections)),
                        "certificate_feasible": bool(
                            sol.is_feasible and certificate_is_feasible(instance, sol.selections)
                        ),
                        "candidate_count": int(instr.get("candidate_count_raw", 0)),
                        "theta_evaluated_count": int(instr.get("theta_evaluated_count", 0)),
                        "median_raw_options": _median(raw),
                        "median_undominated_options": _median(undominated),
                        "median_hull_vertices": _median(hull),
                        "median_hull_fraction": _median(h / r for h, r in zip(hull, raw) if r),
                    }
                )
    return rows


def _summary(exact: Sequence[Dict[str, Any]], cert: Sequence[Dict[str, Any]], scale: Sequence[Dict[str, Any]], audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "exact_comparison_rows": len(exact),
        "exact_bnb_optimal_rows": sum(r["bnb_status"] == "optimal" for r in exact),
        "highs_certified_rows": sum(bool(r["highs_certified"]) for r in exact),
        "max_bnb_highs_absolute_difference": _maximum(r["bnb_highs_absolute_difference"] for r in exact),
        "median_hullround_gap_to_optimum": _median(r["hullround_relative_gap_to_bnb"] for r in exact),
        "max_hullround_gap_to_optimum": _maximum(r["hullround_relative_gap_to_bnb"] for r in exact),
        "certificate_rows": len(cert),
        "certificate_feasible_rows": sum(bool(r["certificate_feasible"]) for r in cert),
        "rounding_bound_rows": sum(float(r["l_rd"]) <= float(r["delta_v_max_theta"]) + 1e-7 for r in cert),
        "max_rounding_loss_over_bound": _maximum(r["l_rd_over_delta"] for r in cert),
        "max_hull_lp_highs_absolute_difference": _maximum(r["lp_highs_abs_diff"] for r in cert),
        "scalability_rows": len(scale),
        "scalability_feasible_rows": sum(bool(r["certificate_feasible"]) for r in scale),
        "largest_tested_n": max(int(r["n"]) for r in scale),
        "largest_tested_m": max(int(r["m"]) for r in scale),
        "median_hull_fraction": _median(r["median_hull_fraction"] for r in scale),
        "mathematical_audit_passed": bool(audit.get("passed", False)),
        "mathematical_audit_counts": audit.get("counts", {}),
    }


def _tex_number(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}g}"


def write_tex(summary: Dict[str, Any], output: Path) -> None:
    counts = summary.get("mathematical_audit_counts", {})
    lines = [
        "% Generated by scripts/run_paper_a_release.py; do not edit.",
        rf"\newcommand{{\AuditGlobalOracleCases}}{{{int(counts.get('global_integer_oracle_cases', 0)) + int(counts.get('global_floating_oracle_cases', 0))}}}",
        rf"\newcommand{{\AuditBreakpointChecks}}{{{int(counts.get('selection_breakpoint_identity_checks', 0))}}}",
        rf"\newcommand{{\AuditLPChecks}}{{{int(counts.get('fixed_theta_lp_oracle_checks', 0))}}}",
        rf"\newcommand{{\ExactComparisonRows}}{{{summary['exact_comparison_rows']}}}",
        rf"\newcommand{{\MaxSolverDifference}}{{{_tex_number(float(summary['max_bnb_highs_absolute_difference']), 3)}}}",
        rf"\newcommand{{\MedianHullRoundGapPct}}{{{_tex_number(100.0 * float(summary['median_hullround_gap_to_optimum']), 3)}\%}}",
        rf"\newcommand{{\MaxHullRoundGapPct}}{{{_tex_number(100.0 * float(summary['max_hullround_gap_to_optimum']), 3)}\%}}",
        rf"\newcommand{{\CertificateRows}}{{{summary['certificate_rows']}}}",
        rf"\newcommand{{\MaxLossBoundRatio}}{{{_tex_number(float(summary['max_rounding_loss_over_bound']), 3)}}}",
        rf"\newcommand{{\MaxLPDifference}}{{{_tex_number(float(summary['max_hull_lp_highs_absolute_difference']), 3)}}}",
        rf"\newcommand{{\ScaleRows}}{{{summary['scalability_rows']}}}",
        rf"\newcommand{{\LargestN}}{{{summary['largest_tested_n']}}}",
        rf"\newcommand{{\LargestM}}{{{summary['largest_tested_m']}}}",
        rf"\newcommand{{\MedianHullFractionPct}}{{{_tex_number(100.0 * float(summary['median_hull_fraction']), 3)}\%}}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_evidence(exact: Sequence[Dict[str, Any]], cert: Sequence[Dict[str, Any]], scale: Sequence[Dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05))

    gaps = 1e4 * np.array([float(r["hullround_relative_gap_to_bnb"]) for r in exact])
    axes[0].hist(gaps, bins=min(12, max(4, len(gaps) // 3)), color="0.78", edgecolor="black")
    axes[0].set_xlabel("HullRound gap (basis points)")
    axes[0].set_ylabel("Certified instances")
    axes[0].set_title("(a) Gap to exact optimum")

    by_n: Dict[int, List[float]] = {}
    for row in cert:
        by_n.setdefault(int(row["n"]), []).append(float(row["l_rd_over_delta"]))
    xs = sorted(by_n)
    axes[1].boxplot([by_n[x] for x in xs], tick_labels=[str(x) for x in xs], showfliers=True)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Number of items $n$")
    axes[1].set_ylabel(r"Round-down loss / $\Delta V_{\max}^{\theta}$")
    axes[1].set_title("(b) One-item certificate")

    for m, marker in [(8, "o"), (16, "s")]:
        mrows = [r for r in scale if int(r["m"]) == m]
        ns = sorted({int(r["n"]) for r in mrows})
        med = [_median(float(r["runtime_seconds"]) for r in mrows if int(r["n"]) == n) for n in ns]
        axes[2].plot(ns, med, marker=marker, color="black", label=f"$m={m}$")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Number of items $n$")
    axes[2].set_ylabel("Median time (s, log scale)")
    axes[2].set_title("(c) Full-breakpoint HullRound")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def write_manifest(directory: Path) -> None:
    entries: List[str] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(directory)}")
    (directory / "MANIFEST.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    source_git_commit = _git("rev-parse", "HEAD")
    source_git_status = _git("status", "--porcelain")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise RuntimeError("mathematical audit did not pass")

    exact = exact_comparisons(args.quick)
    cert = certificate_study(args.quick)
    scale = scalability_study(args.quick)
    summary = _summary(exact, cert, scale, audit)
    if summary["exact_bnb_optimal_rows"] != len(exact) or summary["highs_certified_rows"] != len(exact):
        raise RuntimeError("an exact-comparison solver failed to certify a row")
    if float(summary["max_bnb_highs_absolute_difference"]) > 1e-5:
        raise RuntimeError("branch-and-bound and independent MILP objectives disagree")
    if summary["certificate_feasible_rows"] != len(cert) or summary["rounding_bound_rows"] != len(cert):
        raise RuntimeError("a HullRound feasibility or one-item certificate gate failed")
    if summary["scalability_feasible_rows"] != len(scale):
        raise RuntimeError("a scalability row failed direct robust certification")

    _write_csv(output / "exact_comparisons.csv", exact)
    _write_csv(output / "certificate_study.csv", cert)
    _write_csv(output / "scalability_study.csv", scale)
    (output / "mathematical_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol = {
        "schema_version": 1,
        "purpose": "Canonical final Paper A controlled evidence release",
        "quick": bool(args.quick),
        "git_commit": source_git_commit,
        "git_status_porcelain_at_start": source_git_status,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "structural_threshold_clustering": False,
        "bnb_numerical_tolerance": 1e-9,
        "final_feasibility_check": "exact sign of direct sorted-Gamma certificate over binary64 input coefficients",
        "timing_policy": "single-process wall clock; observational only",
        "design": {
            "exact": "5 seeds x n={12,20,30} x distinct Gamma={0,floor(sqrt(n)),floor(0.1n)}, m=8",
            "certificate": "5 seeds x n={30,60,120} x distinct Gamma={0,floor(sqrt(n)),floor(0.1n)}, m=10",
            "scalability": "3 seeds x n={30,60,120,200} x m={8,16}, Gamma=floor(sqrt(n))",
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tex(summary, output / "generated" / "release_macros.tex")
    plot_evidence(exact, cert, scale, output / "figures" / "release_evidence.pdf")
    write_manifest(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
