"""Stylized application: category pricing in retail (n=300, m=20).

This script implements the case study described in Section "Stylized Application:
Category Pricing in Retail" and produces drop-in figures:

- ``case_frontier.pdf``
- ``case_margin_dist.pdf``

It also saves CSV outputs for summary, stress tests, and SKU-level diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from robust_mckp import PricingInstance, from_pricing_data, solve
from robust_mckp.greedy import greedy_lp
from robust_mckp.hull import build_upper_hull
from robust_mckp.utils import EPS


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def progress(iterable, total: int | None = None, desc: str | None = None, leave: bool = True):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, leave=leave)
    except Exception:
        return iterable


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


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
            "lines.markersize": 5.5,
        }
    )


def _save_csv(path: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    for r in records[1:]:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _extract_vst(instance: PricingInstance):
    v_list, s_list, t_list = [], [], []
    for group in instance.items:
        v_list.append(np.array([o.value for o in group], dtype=float))
        s_list.append(np.array([o.margin for o in group], dtype=float))
        t_list.append(np.array([o.uncertainty for o in group], dtype=float))
    return v_list, s_list, t_list


def _compute_hulls_for_theta(v_list, s_list, t_list, theta):
    hulls = []
    s_star_sum = 0.0
    for v, s, t in zip(v_list, s_list, t_list):
        penalty = np.maximum(0.0, np.abs(t) - theta)
        s_theta = s - penalty
        j_star = int(np.argmax(s_theta))
        s_star = float(s_theta[j_star])
        s_star_sum += s_star
        costs = np.maximum(s_star - s_theta, 0.0)
        hulls.append(build_upper_hull(costs, v, np.arange(len(v), dtype=int)))
    return hulls, s_star_sum


def _round_down_value(lp_sol, hulls, v_list) -> float:
    value = 0.0
    for i, pos in enumerate(lp_sol.positions):
        vertex_idx = pos.upper_vertex if pos.lambda_ >= 1.0 - EPS else pos.lower_vertex
        opt_idx = int(hulls[i].option_indices[vertex_idx])
        value += float(v_list[i][opt_idx])
    return value


def _delta_v_max(hulls) -> float:
    out = 0.0
    for h in hulls:
        if h.values.size >= 2:
            out = max(out, float(np.max(np.diff(h.values))))
    return out


# ---------------------------------------------------------------------------
# Retail case study data generation
# ---------------------------------------------------------------------------
SEGMENTS = [
    {"id": 0, "name": "Staples", "n": 120, "a_median": 3.0, "a_var": 0.15,
     "eta_low": 1.2, "eta_high": 2.0, "d_median": 150.0, "d_var": 0.30, "alpha": 0.08},
    {"id": 1, "name": "Mainstream", "n": 100, "a_median": 5.5, "a_var": 0.20,
     "eta_low": 2.0, "eta_high": 3.5, "d_median": 60.0, "d_var": 0.35, "alpha": 0.15},
    {"id": 2, "name": "Premium", "n": 50, "a_median": 9.0, "a_var": 0.25,
     "eta_low": 1.5, "eta_high": 2.5, "d_median": 20.0, "d_var": 0.40, "alpha": 0.12},
    {"id": 3, "name": "Private label", "n": 30, "a_median": 3.5, "a_var": 0.10,
     "eta_low": 3.0, "eta_high": 5.0, "d_median": 80.0, "d_var": 0.25, "alpha": 0.25},
]

SEGMENT_COLORS = {
    "Staples": "#4c78a8",
    "Mainstream": "#f58518",
    "Premium": "#54a24b",
    "Private label": "#e45756",
}


@dataclass
class RetailPortfolio:
    reference_prices: np.ndarray
    weights: np.ndarray
    price_menus: List[np.ndarray]
    demands: List[np.ndarray]
    uncertainties: List[np.ndarray]
    margin_target: float
    tolerances: np.ndarray
    segment_names: List[str]
    segment_ids: np.ndarray
    elasticities: np.ndarray
    baseline_volumes: np.ndarray
    uncertainty_levels: np.ndarray
    requested_menu_size: int

    @property
    def n(self) -> int:
        return int(self.reference_prices.size)

    def to_instance(self, gamma: int) -> PricingInstance:
        return from_pricing_data(
            reference_prices=self.reference_prices,
            weights=self.weights,
            price_menus=self.price_menus,
            demands=self.demands,
            uncertainties=self.uncertainties,
            margin_target=self.margin_target,
            tolerances=self.tolerances,
            gamma=int(gamma),
        )


def _psychological_round_x9(target: float, lower: float, upper: float) -> float:
    step = 0.10
    offset = 0.09
    k_lo = int(math.ceil((lower - offset) / step - 1e-12))
    k_hi = int(math.floor((upper - offset) / step + 1e-12))
    if k_lo > k_hi:
        return float(np.round(np.clip(target, lower, upper), 2))
    k_star = int(round((target - offset) / step))
    k_star = min(max(k_star, k_lo), k_hi)
    return float(np.round(offset + step * k_star, 2))


def _build_price_menu_with_psychology(a_i: float, m: int, sigma: float) -> np.ndarray:
    lower = (1.0 - sigma) * a_i
    upper = (1.0 + sigma) * a_i
    base_grid = np.linspace(lower, upper, m)
    rounded = np.array(
        [_psychological_round_x9(float(x), lower, upper) for x in base_grid], dtype=float
    )
    rounded = np.unique(np.round(rounded, 2))
    rounded = rounded[(rounded >= lower - 1e-9) & (rounded <= upper + 1e-9)]
    if rounded.size == 0:
        rounded = np.array([_psychological_round_x9(a_i, lower, upper)], dtype=float)
    return rounded


def generate_retail_portfolio(
    *, seed: int = 42, n_expected: int = 300, m: int = 20,
    sigma: float = 0.10, epsilon: float = 0.02,
) -> RetailPortfolio:
    rng = np.random.default_rng(seed)

    a_list, d_list, eta_list = [], [], []
    seg_name_list, seg_id_list, alpha_list = [], [], []

    for seg in SEGMENTS:
        n_seg = int(seg["n"])
        a_seg = rng.lognormal(math.log(seg["a_median"]), math.sqrt(seg["a_var"]), n_seg)
        d_seg = rng.lognormal(math.log(seg["d_median"]), math.sqrt(seg["d_var"]), n_seg)
        eta_seg = rng.uniform(seg["eta_low"], seg["eta_high"], n_seg)
        a_list.extend(a_seg.tolist())
        d_list.extend(d_seg.tolist())
        eta_list.extend(eta_seg.tolist())
        seg_name_list.extend([seg["name"]] * n_seg)
        seg_id_list.extend([seg["id"]] * n_seg)
        alpha_list.extend([seg["alpha"]] * n_seg)

    a = np.array(a_list)
    d_anchor = np.array(d_list)
    eta = np.array(eta_list)
    seg_ids = np.array(seg_id_list, dtype=int)
    alpha_i = np.array(alpha_list)

    if a.size != n_expected:
        raise RuntimeError(f"Segment counts sum to {a.size}, expected {n_expected}")

    weights = d_anchor / float(np.mean(d_anchor))
    tolerances = np.full(a.size, sigma)

    price_menus, demands, uncertainties = [], [], []
    for i in range(a.size):
        x_menu = _build_price_menu_with_psychology(float(a[i]), m=m, sigma=sigma)
        g_hat = d_anchor[i] * (x_menu / a[i]) ** (-eta[i])
        price_menus.append(x_menu)
        demands.append(g_hat)
        uncertainties.append(alpha_i[i] * g_hat)

    # Calibrate Delta
    tilde_x = np.zeros(a.size)
    tilde_g = np.zeros(a.size)
    for i, x_menu in enumerate(price_menus):
        j = int(np.argmin(np.abs(x_menu - a[i])))
        tilde_x[i] = x_menu[j]
        tilde_g[i] = demands[i][j]
    num = float(np.sum(weights * tilde_x * tilde_g))
    den = float(np.sum(weights * a * tilde_g))
    Delta = (1.0 - epsilon) * num / den

    return RetailPortfolio(
        reference_prices=a, weights=weights, price_menus=price_menus,
        demands=demands, uncertainties=uncertainties, margin_target=Delta,
        tolerances=tolerances, segment_names=seg_name_list, segment_ids=seg_ids,
        elasticities=eta, baseline_volumes=d_anchor, uncertainty_levels=alpha_i,
        requested_menu_size=m,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------
def _selected_raw_arrays(portfolio: RetailPortfolio, selections: Sequence[int]):
    n = portfolio.n
    x = np.zeros(n)
    g = np.zeros(n)
    dlt = np.zeros(n)
    for i, j in enumerate(selections):
        x[i] = portfolio.price_menus[i][j]
        g[i] = portfolio.demands[i][j]
        dlt[i] = portfolio.uncertainties[i][j]
    k = portfolio.weights * (x - portfolio.margin_target * portfolio.reference_prices)
    t_nom = k * dlt
    return {"x": x, "g_hat": g, "delta": dlt, "k": k, "t_nom": t_nom}


def simulate_adversarial_exact(
    portfolio: RetailPortfolio, selections: Sequence[int], *,
    gamma_attack: int, rng: np.random.Generator, scenarios: int, batch: int = 1000,
) -> Tuple[float, float]:
    data = _selected_raw_arrays(portfolio, selections)
    g, dlt, k = data["g_hat"], data["delta"], data["k"]
    sign_t = np.sign(data["t_nom"])
    n = g.size
    gamma_attack = int(max(0, min(gamma_attack, n)))
    margins = []
    for start in range(0, scenarios, batch):
        b = min(batch, scenarios - start)
        if gamma_attack == 0:
            realized = np.broadcast_to(g, (b, n))
        else:
            xi = np.zeros((b, n))
            ranks = rng.random((b, n))
            idx = np.argpartition(ranks, kth=gamma_attack - 1, axis=1)[:, :gamma_attack]
            draws = rng.uniform(-1.0, 0.0, size=(b, gamma_attack))
            xi[np.arange(b)[:, None], idx] = draws * sign_t[idx]
            realized = np.maximum(0.0, g[None, :] + xi * dlt[None, :])
        margins.append(realized @ k)
    arr = np.concatenate(margins) if margins else np.array([], dtype=float)
    if arr.size == 0:
        return 0.0, float("nan")
    return float(np.mean(arr < -EPS)), float(np.quantile(arr, 0.05))


def simulate_iid_exact(
    portfolio: RetailPortfolio, selections: Sequence[int], *,
    rng: np.random.Generator, scenarios: int, batch: int = 1000,
) -> Tuple[float, float]:
    data = _selected_raw_arrays(portfolio, selections)
    g, dlt, k = data["g_hat"], data["delta"], data["k"]
    n = g.size
    margins = []
    for start in range(0, scenarios, batch):
        b = min(batch, scenarios - start)
        xi = rng.uniform(-1.0, 1.0, size=(b, n))
        realized = np.maximum(0.0, g[None, :] + xi * dlt[None, :])
        margins.append(realized @ k)
    arr = np.concatenate(margins) if margins else np.array([], dtype=float)
    if arr.size == 0:
        return 0.0, float("nan")
    return float(np.mean(arr < -EPS)), float(np.quantile(arr, 0.05))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_case_frontier(summary_rows: List[Dict[str, object]], output_dir: Path, n: int):
    """Panel (a): revenue--violation frontier as two stacked subplots."""
    plt = try_import_matplotlib()
    if plt is None or not summary_rows:
        return

    rows = sorted(summary_rows, key=lambda r: int(r["gamma"]))
    x = np.array([int(r["gamma"]) / n for r in rows])
    rev = np.array([float(r["revenue_ratio"]) for r in rows])
    viol_adv = np.array([float(r["viol_adv_frontier"]) for r in rows])
    viol_iid = np.array([float(r["viol_iid"]) for r in rows])

    fig, (ax_rev, ax_viol) = plt.subplots(
        2, 1, figsize=(5.5, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )

    # Revenue panel
    ax_rev.plot(x, rev, color="#1f77b4", marker="o", label="Revenue ratio")
    ax_rev.set_ylabel(r"$N(\mathbf{x})\,/\,N(\mathbf{x}^{\Gamma=0})$")
    ax_rev.set_title(f"Retail category pricing ($n={n}$, $m=20$): revenue--risk frontier")
    ax_rev.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax_rev.tick_params(labelbottom=False)

    # Violation panel
    ax_viol.plot(x, viol_adv, color="#d62728", marker="s", linestyle="-", label="Adversarial")
    ax_viol.plot(x, viol_iid, color="#d62728", marker="s", linestyle="--", alpha=0.6, label="IID")
    ax_viol.set_xlabel(r"$\Gamma\,/\,n$")
    ax_viol.set_ylabel(r"Violation probability $\hat{P}$")
    ax_viol.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_dir / "case_frontier.pdf")
    plt.close(fig)


def plot_case_margin_distribution(
    nominal_sku_rows: List[Dict[str, object]],
    robust_sku_rows: List[Dict[str, object]],
    output_dir: Path,
    gamma_robust: int,
):
    """Panel (b): sorted per-SKU margin contributions, robust colored by segment."""
    plt = try_import_matplotlib()
    if plt is None or not nominal_sku_rows or not robust_sku_rows:
        return

    nominal_sorted = sorted(nominal_sku_rows, key=lambda r: float(r["margin_contribution"]))
    robust_sorted = sorted(robust_sku_rows, key=lambda r: float(r["margin_contribution"]))

    x_nom = np.arange(1, len(nominal_sorted) + 1)
    y_nom = np.array([float(r["margin_contribution"]) for r in nominal_sorted])

    x_rob = np.arange(1, len(robust_sorted) + 1)
    y_rob = np.array([float(r["margin_contribution"]) for r in robust_sorted])
    seg_rob = [str(r["segment"]) for r in robust_sorted]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Nominal as a continuous gray line
    ax.plot(x_nom, y_nom, color="gray", linewidth=1.8, alpha=0.8,
            label=r"Nominal ($\Gamma=0$)")

    # Robust: thin connecting line + colored scatter by segment
    ax.plot(x_rob, y_rob, color="black", linewidth=0.6, alpha=0.25, zorder=1)
    for seg in SEGMENTS:
        name = seg["name"]
        mask = np.array([s == name for s in seg_rob])
        if np.any(mask):
            ax.scatter(
                x_rob[mask], y_rob[mask], s=14, color=SEGMENT_COLORS[name],
                alpha=0.9, label=rf"{name} ($\Gamma={gamma_robust}$)", zorder=2,
            )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xlabel("SKU rank (sorted by margin contribution)")
    ax.set_ylabel(r"Per-SKU margin contribution $s_i(x_i)$")
    ax.set_title("Margin distribution: nominal vs. robust")

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    seen, h_out, l_out = set(), [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            h_out.append(h)
            l_out.append(lab)
    ax.legend(h_out, l_out, fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_dir / "case_margin_dist.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="figs")
    parser.add_argument("--results-dir", type=str, default="results_case")
    parser.add_argument("--scenarios", type=int, default=10_000)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = 300
    m = 20
    sigma = 0.10
    gamma_star = int(math.floor(math.sqrt(n)))
    gamma_grid = [0, 1, 5, 10, 15, gamma_star, 30, 50, 75, 100, n]
    gamma_grid = sorted(set(int(min(max(0, g), n)) for g in gamma_grid))

    if args.fast:
        args.scenarios = 2_000
        gamma_grid = [0, 5, gamma_star, 50, n]

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    portfolio = generate_retail_portfolio(seed=args.seed, n_expected=n, m=m, sigma=sigma)

    menu_sizes = np.array([len(x) for x in portfolio.price_menus], dtype=int)
    print("Retail case portfolio generated:", {
        "n": portfolio.n, "requested_m": m,
        "menu_size_min": int(menu_sizes.min()),
        "menu_size_median": float(np.median(menu_sizes)),
        "menu_size_max": int(menu_sizes.max()),
        "Delta": f"{portfolio.margin_target:.6f}",
    })

    summary_rows: List[Dict[str, object]] = []
    stress_rows: List[Dict[str, object]] = []
    sku_rows: List[Dict[str, object]] = []
    nominal_sku_rows: List[Dict[str, object]] = []
    robust_star_sku_rows: List[Dict[str, object]] = []

    baseline_selections = None
    baseline_objective = float("nan")
    baseline_prices = None
    nominal_s = None
    q25_nominal_positive = float("nan")
    nominal_nontrivial_count = None

    for gamma in progress(gamma_grid, desc="retail-case gamma", total=len(gamma_grid)):
        instance = portfolio.to_instance(gamma)

        t0 = time.perf_counter()
        sol = solve(instance)
        wall = time.perf_counter() - t0

        v_list, s_list, t_list = _extract_vst(instance)
        hulls, s_star_sum = _compute_hulls_for_theta(v_list, s_list, t_list, sol.theta)
        capacity = s_star_sum - gamma * sol.theta
        lp_sol = greedy_lp(hulls, capacity)
        rd_value = _round_down_value(lp_sol, hulls, v_list)
        l_rd = float(lp_sol.lp_value - rd_value)
        bound = _delta_v_max(hulls)
        gap_lp = l_rd / lp_sol.lp_value if abs(lp_sol.lp_value) > EPS else float("nan")

        # Per-SKU selected data
        selected_prices = np.array(
            [portfolio.price_menus[i][j] for i, j in enumerate(sol.selections)]
        )
        selected_margins = np.array(
            [instance.items[i][j].margin for i, j in enumerate(sol.selections)]
        )
        selected_uncertainties = np.array(
            [instance.items[i][j].uncertainty for i, j in enumerate(sol.selections)]
        )

        # Baseline (Γ=0) bookkeeping
        if gamma == 0:
            baseline_selections = list(sol.selections)
            baseline_objective = float(sol.objective)
            baseline_prices = selected_prices.copy()
            nominal_s = selected_margins.copy()
            pos_nom = nominal_s[nominal_s > 0]
            q25_nominal_positive = float(np.quantile(pos_nom, 0.25)) if pos_nom.size else 0.0
            nominal_nontrivial_count = int(np.sum(nominal_s >= q25_nominal_positive - EPS))

        assert baseline_prices is not None
        price_shifts_vs_nominal = int(np.sum(np.abs(selected_prices - baseline_prices) > 1e-9))
        nontrivial_count = int(np.sum(selected_margins >= q25_nominal_positive - EPS))
        nontrivial_increase_pct = (
            100.0 * (nontrivial_count - nominal_nontrivial_count) / nominal_nontrivial_count
            if nominal_nontrivial_count and nominal_nontrivial_count > 0
            else float("nan")
        )

        # Stress tests
        frontier_attack = int(min(n, math.floor(1.5 * gamma)))
        attack_levels = sorted(set([
            int(max(0, min(n, gamma))),
            int(max(0, min(n, frontier_attack))),
            int(max(0, min(n, 2 * gamma))),
            n,
        ]))

        rng_iid = np.random.default_rng(
            np.random.SeedSequence([args.seed, 300, gamma, 999])
        )
        viol_iid, q05_iid = simulate_iid_exact(
            portfolio, sol.selections, rng=rng_iid, scenarios=args.scenarios,
        )

        adv_match_violation = None
        adv_match_q05 = None
        adv_frontier_violation = None
        adv_frontier_q05 = None

        for gamma_attack in progress(
            attack_levels, desc=f"stress G={gamma}", total=len(attack_levels), leave=False,
        ):
            rng_adv = np.random.default_rng(
                np.random.SeedSequence([args.seed, 301, gamma, gamma_attack])
            )
            viol_adv, q05_adv = simulate_adversarial_exact(
                portfolio, sol.selections,
                gamma_attack=gamma_attack, rng=rng_adv, scenarios=args.scenarios,
            )
            stress_rows.append({
                "gamma": gamma, "gamma_attack": gamma_attack,
                "violation_adv": viol_adv, "q05_margin_adv": q05_adv,
                "violation_iid": viol_iid, "q05_margin_iid": q05_iid,
                "objective": float(sol.objective),
                "revenue_ratio": float(sol.objective / baseline_objective) if abs(baseline_objective) > EPS else float("nan"),
            })
            if gamma_attack == gamma:
                adv_match_violation = viol_adv
                adv_match_q05 = q05_adv
            if gamma_attack == frontier_attack:
                adv_frontier_violation = viol_adv
                adv_frontier_q05 = q05_adv

        # Fallbacks
        if adv_frontier_violation is None:
            adv_frontier_violation = adv_match_violation if adv_match_violation is not None else float("nan")
            adv_frontier_q05 = adv_match_q05 if adv_match_q05 is not None else float("nan")
        if adv_match_violation is None:
            adv_match_violation = float("nan")
            adv_match_q05 = float("nan")

        instr = (sol.metadata or {}).get("instrumentation", {})
        summary_rows.append({
            "gamma": gamma, "n": n, "m": m,
            "objective": float(sol.objective),
            "revenue_ratio": float(sol.objective / baseline_objective) if abs(baseline_objective) > EPS else float("nan"),
            "time_s": wall,
            "theta": float(sol.theta),
            "certificate_value": float(sol.certificate_value),
            "lp_value": float(lp_sol.lp_value),
            "l_rd": l_rd, "bound": bound, "gap_lp": gap_lp,
            "price_shifts_vs_nominal": price_shifts_vs_nominal,
            "nontrivial_margin_count": nontrivial_count,
            "nontrivial_increase_pct_vs_nominal": nontrivial_increase_pct,
            "frontier_attack_level": frontier_attack,
            "viol_adv_match": adv_match_violation,
            "q05_adv_match": adv_match_q05,
            "viol_adv_frontier": adv_frontier_violation,
            "q05_adv_frontier": adv_frontier_q05,
            "viol_iid": viol_iid, "q05_iid": q05_iid,
            "candidate_count_raw": instr.get("candidate_count_raw", np.nan),
            "candidate_count_reduced": instr.get("candidate_count_reduced", np.nan),
            "theta_evaluated_count": instr.get("theta_evaluated_count", np.nan),
            "theta_skipped_capacity_count": instr.get("theta_skipped_capacity_count", np.nan),
        })

        # SKU-level diagnostics
        for i in range(n):
            sku_rows.append({
                "gamma": gamma, "sku_index": i,
                "segment_id": int(portfolio.segment_ids[i]),
                "segment": portfolio.segment_names[i],
                "reference_price": float(portfolio.reference_prices[i]),
                "selected_price": float(selected_prices[i]),
                "selected_margin_contribution": float(selected_margins[i]),
                "selected_uncertainty_contribution": float(selected_uncertainties[i]),
                "elasticity": float(portfolio.elasticities[i]),
                "baseline_volume": float(portfolio.baseline_volumes[i]),
                "alpha_i": float(portfolio.uncertainty_levels[i]),
            })
            if gamma == 0:
                nominal_sku_rows.append({
                    "segment": portfolio.segment_names[i],
                    "margin_contribution": float(selected_margins[i]),
                })
            if gamma == gamma_star:
                robust_star_sku_rows.append({
                    "segment": portfolio.segment_names[i],
                    "margin_contribution": float(selected_margins[i]),
                })

    # -----------------------------------------------------------------------
    # Managerial insight metrics
    # -----------------------------------------------------------------------
    summary_by_gamma = {int(r["gamma"]): r for r in summary_rows}
    nominal_row = summary_by_gamma.get(0)
    sqrt_row = summary_by_gamma.get(gamma_star)
    insights_rows: List[Dict[str, object]] = []
    if nominal_row is not None and sqrt_row is not None and nominal_nontrivial_count is not None:
        insights_rows.append({
            "gamma_nominal": 0,
            "gamma_robust": gamma_star,
            "n": n, "m": m,
            "nontrivial_margin_count_nominal": nominal_nontrivial_count,
            "nontrivial_margin_count_robust": int(sqrt_row["nontrivial_margin_count"]),
            "nontrivial_margin_increase_pct": float(sqrt_row["nontrivial_increase_pct_vs_nominal"]),
            "revenue_ratio_robust": float(sqrt_row["revenue_ratio"]),
            "revenue_sacrifice_pct": 100.0 * (1.0 - float(sqrt_row["revenue_ratio"])),
            "price_shifts_vs_nominal_robust": int(sqrt_row["price_shifts_vs_nominal"]),
            "runtime_robust_s": float(sqrt_row["time_s"]),
            "l_rd_robust": float(sqrt_row["l_rd"]),
            "bound_robust": float(sqrt_row["bound"]),
            "bound_slack_ratio": (
                float(sqrt_row["l_rd"]) / float(sqrt_row["bound"])
                if abs(float(sqrt_row["bound"])) > EPS else float("nan")
            ),
        })

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    _save_csv(results_dir / "case_retail_summary.csv", summary_rows)
    _save_csv(results_dir / "case_retail_stress.csv", stress_rows)
    _save_csv(results_dir / "case_retail_sku_details.csv", sku_rows)
    _save_csv(results_dir / "case_retail_insights.csv", insights_rows)

    print(f"\nResults saved to {results_dir}/")
    print(f"  case_retail_summary.csv     ({len(summary_rows)} rows)")
    print(f"  case_retail_stress.csv      ({len(stress_rows)} rows)")
    print(f"  case_retail_sku_details.csv ({len(sku_rows)} rows)")
    if insights_rows:
        print(f"  case_retail_insights.csv    ({len(insights_rows)} rows)")
        for k, v in insights_rows[0].items():
            print(f"    {k}: {v}")

    # Sanity check
    nonzero_match = [r for r in summary_rows if abs(float(r["viol_adv_match"])) > 0.0]
    if nonzero_match:
        worst = max(float(r["viol_adv_match"]) for r in nonzero_match)
        print(f"\nWARNING: Nonzero adversarial violations at matching attack level (max={worst:.4g}).")
    else:
        print("\nSanity check passed: zero adversarial violations at matching attack level.")

    max_rt = max(float(r["time_s"]) for r in summary_rows) if summary_rows else float("nan")
    print(f"Max solve runtime: {max_rt:.3f}s")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plt = try_import_matplotlib()
    if plt is None:
        print("matplotlib not available; plots skipped")
        return
    apply_pub_style()
    plot_case_frontier(summary_rows, output_dir=output_dir, n=n)
    plot_case_margin_distribution(
        nominal_sku_rows=nominal_sku_rows,
        robust_sku_rows=robust_star_sku_rows,
        output_dir=output_dir,
        gamma_robust=gamma_star,
    )
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()