"""Nested-prefix Experiment 3: revenue--risk frontier and guarantee tightness."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from robust_mckp import solve

from _common import (
    build_prefix_instance,
    make_master_portfolio,
    make_rng,
    progress,
    simulate_adversarial_exact,
    simulate_iid_exact,
    try_import_matplotlib,
)


# ---------------------------------------------------------------------------
# Publication plot style (consistent with exp1/exp2)
# ---------------------------------------------------------------------------
def apply_pub_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.4,
            "lines.markersize": 6,
        }
    )


COLORS = {0.10: "#1f77b4", 0.20: "#d62728"}
ALPHA_LABELS = {0.10: r"$\alpha=0.10$", 0.20: r"$\alpha=0.20$"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="figs_nested")
    parser.add_argument("--results-dir", type=str, default="results_nested")
    parser.add_argument("--scenarios", type=int, default=10_000)
    parser.add_argument("--scenarios-heatmap", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gamma_grid(n: int) -> List[int]:
    vals = [0, 1, 3, 5, 10, int(math.floor(math.sqrt(n))), 20, 30, 50, 100, n]
    return sorted(set(int(max(0, min(n, g))) for g in vals))


def _frontier_attack(gamma: int, n: int) -> int:
    """Adversarial attack level for frontier plots (stricter than protection)."""
    if gamma <= 0:
        return 0
    return int(min(n, max(gamma, math.floor(1.5 * gamma))))


def _rng_salt(*args) -> int:
    """Combine multiple ints into a single seed-salt via SeedSequence."""
    return int(np.random.SeedSequence(list(args)).generate_state(1)[0])


def _save_csv(path: Path, records: List[Dict[str, object]]) -> None:
    """Save CSV with insertion-ordered columns."""
    if not records:
        return
    all_keys = list(records[0].keys())
    for r in records[1:]:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_frontier(rows, alpha_values, n, output_dir):
    """Panel (a): Revenue--violation frontier as two stacked subplots."""
    plt = try_import_matplotlib()
    if plt is None:
        return

    fig, (ax_rev, ax_viol) = plt.subplots(
        2, 1, figsize=(5.5, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )

    for alpha in alpha_values:
        rws = sorted(
            [r for r in rows if abs(float(r["alpha"]) - alpha) < 1e-12],
            key=lambda r: int(r["gamma"]),
        )
        x = np.array([float(r["gamma"]) / n for r in rws])
        rev = np.array([float(r["revenue_ratio"]) for r in rws])
        adv = np.array([float(r["viol_adv_frontier"]) for r in rws])
        iid = np.array([float(r["viol_iid"]) for r in rws])
        c = COLORS[alpha]
        lab = ALPHA_LABELS[alpha]

        ax_rev.plot(x, rev, marker="o", color=c, label=lab)
        ax_viol.plot(x, adv, marker="s", color=c, linestyle="-", label=f"Adv. {lab}")
        ax_viol.plot(x, iid, marker="s", color=c, linestyle="--", alpha=0.6, label=f"IID {lab}")

    # Revenue panel
    ax_rev.set_ylabel(r"$N(\mathbf{x})\,/\,N(\mathbf{x}^{\Gamma=0})$")
    ax_rev.set_title(f"Revenue--risk frontier ($n={n}$)")
    ax_rev.legend(fontsize=8, loc="lower left", framealpha=0.9)

    # Violation panel
    ax_viol.set_xlabel(r"$\Gamma\,/\,n$")
    ax_viol.set_ylabel(r"Violation probability $\hat{P}$")
    ax_viol.legend(fontsize=7.5, loc="upper right", framealpha=0.9, ncol=2)

    # Remove x-tick labels from top panel
    ax_rev.tick_params(labelbottom=False)

    fig.tight_layout()
    fig.savefig(output_dir / "risk_frontier.pdf")
    plt.close(fig)


def plot_tail_margin(rows, alpha_values, n, output_dir):
    """Panel (b): 5% quantile of realized margin vs Gamma/n."""
    plt = try_import_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for alpha in alpha_values:
        rws = sorted(
            [r for r in rows if abs(float(r["alpha"]) - alpha) < 1e-12],
            key=lambda r: int(r["gamma"]),
        )
        x = np.array([float(r["gamma"]) / n for r in rws])
        q05 = np.array([float(r["q05_adv_frontier"]) for r in rws])
        ax.plot(x, q05, marker="o", color=COLORS[alpha], label=ALPHA_LABELS[alpha])

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"$\Gamma\,/\,n$")
    ax.set_ylabel(r"$5\%$-quantile of realized margin")
    ax.set_title(f"Tail margin vs. protection level ($n={n}$)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "tail_margin.pdf")
    plt.close(fig)


def plot_heatmap(heatmap_rows, n, output_dir):
    """Panel (c): Tightness heatmap with log-norm colorscale."""
    plt = try_import_matplotlib()
    if plt is None:
        return
    from matplotlib.colors import LogNorm, Normalize

    if not heatmap_rows:
        return

    g_protect = sorted(set(int(r["gamma_protect"]) for r in heatmap_rows))
    g_attack = sorted(set(int(r["gamma_attack"]) for r in heatmap_rows))
    mat = np.full((len(g_protect), len(g_attack)), np.nan, dtype=float)
    for r in heatmap_rows:
        i = g_protect.index(int(r["gamma_protect"]))
        j = g_attack.index(int(r["gamma_attack"]))
        mat[i, j] = float(r["violation"])

    # Replace exact zeros with NaN so they render as white (below LogNorm range)
    mat_plot = mat.copy()
    mat_plot[mat_plot <= 0.0] = np.nan

    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    # Use log colorscale; NaN (zero violations) maps to white background
    viol_max = float(np.nanmax(mat_plot)) if np.any(np.isfinite(mat_plot)) else 1.0
    viol_min = 1e-4  # floor for log scale
    if viol_max < viol_min:
        viol_max = 1.0

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        mat_plot,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=LogNorm(vmin=viol_min, vmax=viol_max),
    )

    # Draw diagonal line where attack = protection
    diag_coords = []
    for i, gp in enumerate(g_protect):
        if gp in g_attack:
            j = g_attack.index(gp)
            diag_coords.append((j, i))
    if diag_coords:
        dx, dy = zip(*diag_coords)
        ax.plot(dx, dy, "k--", alpha=0.5, linewidth=1.0, label=r"$\Gamma_{\mathrm{attack}}=\Gamma$")

    ax.set_xticks(range(len(g_attack)))
    ax.set_xticklabels([str(g) for g in g_attack], rotation=45, ha="right")
    ax.set_yticks(range(len(g_protect)))
    ax.set_yticklabels([str(g) for g in g_protect])
    ax.set_xlabel(r"$\Gamma_{\mathrm{attack}}$")
    ax.set_ylabel(r"$\Gamma$ (protection)")
    ax.set_title(f"Tightness heatmap (adversarial, $n={n}$, $\\alpha=0.10$)")

    cb = fig.colorbar(im, ax=ax, extend="min")
    cb.set_label("Violation probability")

    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "tightness_heatmap.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    n = 200
    m = 50
    alpha_values = [0.10, 0.20]

    if args.fast:
        args.scenarios = 3_000
        if args.scenarios_heatmap is None:
            args.scenarios_heatmap = 2_000
    if args.scenarios_heatmap is None:
        args.scenarios_heatmap = args.scenarios

    gammas = _gamma_grid(n)
    master = make_master_portfolio(
        n_max=n, m_max=m, seed=args.seed, sigma=0.10, min_admissible_menu=10
    )

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    tightness_rows: List[Dict[str, object]] = []
    heatmap_rows: List[Dict[str, object]] = []

    for alpha in progress(alpha_values, desc="nested-exp3 alpha", total=len(alpha_values)):
        # Gamma=0 solution for revenue normalization
        prefix0 = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=0)
        sol0 = solve(prefix0.instance)
        base_obj = float(sol0.objective)

        for gamma in progress(
            gammas, desc=f"gamma alpha={alpha:.2f}", total=len(gammas), leave=False
        ):
            prefix = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=gamma)
            sol = solve(prefix.instance)

            gamma_attack_front = _frontier_attack(gamma, n)

            # Distinct RNGs per (alpha, gamma, purpose) — 4 positional salts
            alpha_salt = int(alpha * 100)
            rng_adv_match = make_rng(args.seed, n, alpha_salt, gamma, 1)
            rng_adv_front = make_rng(args.seed, n, alpha_salt, gamma, 2)
            rng_iid = make_rng(args.seed, n, alpha_salt, gamma, 3)

            viol_adv_match, q05_adv_match = simulate_adversarial_exact(
                prefix, sol.selections,
                gamma_attack=gamma, rng=rng_adv_match, scenarios=args.scenarios,
            )
            viol_adv_front, q05_adv_front = simulate_adversarial_exact(
                prefix, sol.selections,
                gamma_attack=gamma_attack_front, rng=rng_adv_front,
                scenarios=args.scenarios,
            )
            viol_iid, q05_iid = simulate_iid_exact(
                prefix, sol.selections, rng=rng_iid, scenarios=args.scenarios,
            )

            rows.append(
                {
                    "n": n,
                    "m": m,
                    "alpha": alpha,
                    "gamma": gamma,
                    "gamma_attack_frontier": gamma_attack_front,
                    "objective": float(sol.objective),
                    "revenue_ratio": (
                        float(sol.objective / base_obj) if base_obj > 0 else np.nan
                    ),
                    "theta": float(sol.theta),
                    "certificate_value": float(sol.certificate_value),
                    "viol_adv_match": viol_adv_match,
                    "q05_adv_match": q05_adv_match,
                    "viol_adv_frontier": viol_adv_front,
                    "q05_adv_frontier": q05_adv_front,
                    "viol_iid": viol_iid,
                    "q05_iid": q05_iid,
                }
            )

            # Tightness grid: {Gamma, Gamma+0.1n, 2*Gamma, n}
            attack_levels = sorted(
                set([
                    gamma,
                    min(n, gamma + int(math.floor(0.1 * n))),
                    min(n, 2 * gamma),
                    n,
                ])
            )
            for gamma_attack in attack_levels:
                rng_t = make_rng(args.seed, n, alpha_salt, gamma, gamma_attack)
                viol_t, q05_t = simulate_adversarial_exact(
                    prefix, sol.selections,
                    gamma_attack=gamma_attack, rng=rng_t, scenarios=args.scenarios,
                )
                tightness_rows.append(
                    {
                        "n": n,
                        "m": m,
                        "alpha": alpha,
                        "gamma_protect": gamma,
                        "gamma_attack": gamma_attack,
                        "violation": viol_t,
                        "q05_margin": q05_t,
                    }
                )

            # Heatmap: alpha=0.10 only (panel c)
            if abs(alpha - 0.10) < 1e-12:
                for gamma_attack in progress(
                    gammas,
                    desc=f"heatmap G={gamma}",
                    total=len(gammas),
                    leave=False,
                ):
                    rng_h = make_rng(args.seed, n, alpha_salt, gamma, gamma_attack)
                    viol_h, _ = simulate_adversarial_exact(
                        prefix, sol.selections,
                        gamma_attack=gamma_attack, rng=rng_h,
                        scenarios=args.scenarios_heatmap,
                    )
                    heatmap_rows.append(
                        {
                            "n": n,
                            "alpha": alpha,
                            "gamma_protect": gamma,
                            "gamma_attack": gamma_attack,
                            "violation": viol_h,
                        }
                    )

    # -----------------------------------------------------------------------
    # Sanity check
    # -----------------------------------------------------------------------
    sanity_fail = [r for r in rows if abs(float(r["viol_adv_match"])) > 0.0]
    if sanity_fail:
        print(
            f"\nWARNING: {len(sanity_fail)} matching-attack violations were nonzero "
            f"(max = {max(float(r['viol_adv_match']) for r in sanity_fail):.4f})."
        )
    else:
        print("\nSanity check passed: zero violations at all matching attack levels.")

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    _save_csv(results_dir / "exp3_risk_frontier.csv", rows)
    _save_csv(results_dir / "exp3_tightness_tests.csv", tightness_rows)
    _save_csv(results_dir / "exp3_tightness_heatmap.csv", heatmap_rows)

    print(f"\nResults saved to {results_dir}/")
    print(f"  exp3_risk_frontier.csv       ({len(rows)} rows)")
    print(f"  exp3_tightness_tests.csv     ({len(tightness_rows)} rows)")
    print(f"  exp3_tightness_heatmap.csv   ({len(heatmap_rows)} rows)")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plt = try_import_matplotlib()
    if plt is None:
        print("matplotlib not available; plots skipped")
        return

    apply_pub_style()
    plot_frontier(rows, alpha_values, n, output_dir)
    plot_tail_margin(rows, alpha_values, n, output_dir)
    plot_heatmap(heatmap_rows, n, output_dir)

    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()