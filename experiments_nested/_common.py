"""Shared utilities for deterministic nested-prefix experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from robust_mckp import PricingInstance, from_pricing_data
from robust_mckp.hull import Point, build_upper_hull, merge_equal_cost, prune_dominated, upper_hull
from robust_mckp.utils import EPS


@dataclass
class MasterPortfolio:
    """Deterministic master portfolio from which nested prefixes are sliced."""

    reference_prices: np.ndarray  # a_i
    weights: np.ndarray  # omega_i
    demand_scales: np.ndarray  # b_i
    elasticities: np.ndarray  # eta_i
    menu_shocks: np.ndarray  # u_{i,j}
    n_max: int
    m_max: int
    sigma: float = 0.10


@dataclass
class GeneratedPrefix:
    """Concrete prefix instance with raw arrays retained for analysis/simulation."""

    instance: PricingInstance
    reference_prices: np.ndarray
    weights: np.ndarray
    price_menus: List[np.ndarray]
    demands: List[np.ndarray]
    uncertainties: List[np.ndarray]
    margin_target: float
    tolerances: np.ndarray


def make_rng(master_seed: int, *keys: int) -> np.random.Generator:
    """Reproducible RNG via SeedSequence."""

    ss = np.random.SeedSequence([master_seed, *keys])
    return np.random.default_rng(ss)


def make_master_portfolio(
    *,
    n_max: int = 500,
    m_max: int = 100,
    seed: int = 42,
    sigma: float = 0.10,
    min_admissible_menu: int = 10,
) -> MasterPortfolio:
    """Generate a single master portfolio used for all nested-prefix experiments.

    Distributions match the paper's protocol:
      a_i ~ exp(N(5, 0.25))
      omega_i ~ exp(N(0, 0.09))
      b_i ~ exp(N(3, 0.16))
      eta_i ~ Uniform(1.5, 3.5)
      u_{i,j} ~ Uniform(-0.15, 0.15)

    Notes:
        NumPy's lognormal uses the *standard deviation* parameter, so the provided
        variances 0.25, 0.09, 0.16 correspond to sigmas 0.5, 0.3, 0.4.
    """

    if min_admissible_menu > m_max:
        raise ValueError("min_admissible_menu must be <= m_max")

    rng = make_rng(seed)
    a = rng.lognormal(mean=5.0, sigma=0.5, size=n_max)
    w = rng.lognormal(mean=0.0, sigma=0.3, size=n_max)
    b = rng.lognormal(mean=3.0, sigma=0.4, size=n_max)
    eta = rng.uniform(1.5, 3.5, size=n_max)
    u = rng.uniform(-0.15, 0.15, size=(n_max, m_max))

    # Ensure each item has at least one admissible option in the smallest m used.
    m_req = int(min_admissible_menu)
    for i in range(n_max):
        while not np.any(np.abs(u[i, :m_req]) <= sigma + EPS):
            u[i, :] = rng.uniform(-0.15, 0.15, size=m_max)

    return MasterPortfolio(
        reference_prices=a,
        weights=w,
        demand_scales=b,
        elasticities=eta,
        menu_shocks=u,
        n_max=n_max,
        m_max=m_max,
        sigma=float(sigma),
    )


def build_prefix_instance(
    master: MasterPortfolio,
    *,
    n: int,
    m: int,
    alpha: float,
    gamma: int,
    sigma: float = 0.10,
    epsilon: float = 0.02,
) -> GeneratedPrefix:
    """Build a nested-prefix instance from the master portfolio.

    The instance uses items 1..n and the first m menu draws per item. The margin
    target Delta is recalibrated at each prefix size using the admissible baseline.
    """

    if n <= 0 or n > master.n_max:
        raise ValueError("n must be in [1, master.n_max]")
    if m <= 0 or m > master.m_max:
        raise ValueError("m must be in [1, master.m_max]")

    a = master.reference_prices[:n].copy()
    w = master.weights[:n].copy()
    b = master.demand_scales[:n]
    eta = master.elasticities[:n]
    u = master.menu_shocks[:n, :m]

    x = a[:, None] * (1.0 + u)
    g_hat = b[:, None] * np.maximum(0.0, 1.0 - eta[:, None] * (x - a[:, None]) / a[:, None])
    delta = float(alpha) * g_hat

    price_menus: List[np.ndarray] = []
    demands: List[np.ndarray] = []
    uncertainties: List[np.ndarray] = []
    tolerances = np.full(n, float(sigma), dtype=float)

    tilde_x = np.zeros(n, dtype=float)
    tilde_g = np.zeros(n, dtype=float)

    for i in range(n):
        mask = (u[i] >= -sigma - EPS) & (u[i] <= sigma + EPS)
        if not np.any(mask):
            raise ValueError(
                f"No admissible option for item {i} in prefix (n={n}, m={m}); "
                "regenerate master portfolio or increase min_admissible_menu."
            )
        xi = x[i, mask]
        gi = g_hat[i, mask]
        di = delta[i, mask]

        price_menus.append(xi)
        demands.append(gi)
        uncertainties.append(di)

        j0 = int(np.argmin(np.abs(xi - a[i])))
        tilde_x[i] = xi[j0]
        tilde_g[i] = gi[j0]

    numerator = float(np.sum(w * tilde_x * tilde_g))
    denominator = float(np.sum(w * a * tilde_g))
    if denominator <= 0:
        raise ValueError("Invalid denominator in Delta calibration")
    Delta = (1.0 - float(epsilon)) * numerator / denominator

    instance = from_pricing_data(
        reference_prices=a,
        weights=w,
        price_menus=price_menus,
        demands=demands,
        uncertainties=uncertainties,
        margin_target=Delta,
        tolerances=tolerances,
        gamma=int(gamma),
    )
    return GeneratedPrefix(
        instance=instance,
        reference_prices=a,
        weights=w,
        price_menus=price_menus,
        demands=demands,
        uncertainties=uncertainties,
        margin_target=float(Delta),
        tolerances=tolerances,
    )


def extract_arrays(instance: PricingInstance) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Extract (v, s, t) arrays from a PricingInstance."""

    v_list: List[np.ndarray] = []
    s_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    for group in instance.items:
        v_list.append(np.array([opt.value for opt in group], dtype=float))
        s_list.append(np.array([opt.margin for opt in group], dtype=float))
        t_list.append(np.array([opt.uncertainty for opt in group], dtype=float))
    return v_list, s_list, t_list


def compute_hulls_for_theta(
    v_list: List[np.ndarray],
    s_list: List[np.ndarray],
    t_list: List[np.ndarray],
    theta: float,
):
    """Rebuild per-item hulls for a fixed theta and return hulls, S_max(theta)."""

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


def hull_sizes_for_theta(
    v_list: List[np.ndarray],
    s_list: List[np.ndarray],
    t_list: List[np.ndarray],
    theta: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Return raw, undominated, and hull sizes per item at a fixed theta."""

    raw_sizes: List[int] = []
    undom_sizes: List[int] = []
    hull_sizes: List[int] = []
    for v, s, t in zip(v_list, s_list, t_list):
        penalty = np.maximum(0.0, np.abs(t) - theta)
        s_theta = s - penalty
        j_star = int(np.argmax(s_theta))
        s_star = float(s_theta[j_star])
        costs = np.maximum(s_star - s_theta, 0.0)

        raw_sizes.append(int(costs.size))
        pts = [Point(float(c), float(val), int(j)) for j, (c, val) in enumerate(zip(costs, v))]
        pts = merge_equal_cost(pts)
        pts = prune_dominated(pts)
        undom_sizes.append(len(pts))
        pts = upper_hull(pts)
        hull_sizes.append(len(pts))
    return raw_sizes, undom_sizes, hull_sizes


def build_theta_costs(
    s_list: List[np.ndarray],
    t_list: List[np.ndarray],
    theta: float,
) -> Tuple[List[np.ndarray], float]:
    """Build fixed-theta knapsack costs c^theta and S_max(theta)."""

    costs_list: List[np.ndarray] = []
    s_star_sum = 0.0
    for s, t in zip(s_list, t_list):
        penalty = np.maximum(0.0, np.abs(t) - theta)
        s_theta = s - penalty
        j_star = int(np.argmax(s_theta))
        s_star_sum += float(s_theta[j_star])
        costs_list.append(np.maximum(s_theta[j_star] - s_theta, 0.0))
    return costs_list, s_star_sum


def round_down_value(lp_sol, hulls, v_list) -> float:
    """Objective of the round-down solution induced by LP positions."""

    value = 0.0
    for i, pos in enumerate(lp_sol.positions):
        vertex_idx = pos.upper_vertex if pos.lambda_ >= 1.0 - EPS else pos.lower_vertex
        opt_idx = int(hulls[i].option_indices[vertex_idx])
        value += float(v_list[i][opt_idx])
    return value


def delta_v_max(hulls) -> float:
    """Maximal adjacent hull-value jump across items."""

    max_jump = 0.0
    for hull in hulls:
        if hull.values.size >= 2:
            max_jump = max(max_jump, float(np.max(np.diff(hull.values))))
    return max_jump


def selected_option_data(
    prefix: GeneratedPrefix,
    selections: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Extract selected option arrays needed for exact stress simulations."""

    n = prefix.instance.n_items
    if len(selections) != n:
        raise ValueError("selection length mismatch")

    x_sel = np.zeros(n, dtype=float)
    g_sel = np.zeros(n, dtype=float)
    d_sel = np.zeros(n, dtype=float)
    w = prefix.weights
    a = prefix.reference_prices
    Delta = prefix.margin_target

    for i, j in enumerate(selections):
        x_sel[i] = prefix.price_menus[i][j]
        g_sel[i] = prefix.demands[i][j]
        d_sel[i] = prefix.uncertainties[i][j]

    k = w * (x_sel - Delta * a)  # coefficient multiplying realized demand
    s_nom = k * g_sel
    t_nom = k * d_sel
    return {
        "x": x_sel,
        "g_hat": g_sel,
        "delta": d_sel,
        "k": k,
        "s_nom": s_nom,
        "t_nom": t_nom,
    }


def simulate_adversarial_exact(
    prefix: GeneratedPrefix,
    selections: Sequence[int],
    gamma_attack: int,
    rng: np.random.Generator,
    scenarios: int,
    *,
    batch: int = 1000,
) -> Tuple[float, float]:
    """Adversarial (worst-case-direction) stress test with clipped demand."""

    data = selected_option_data(prefix, selections)
    g = data["g_hat"]
    d = data["delta"]
    k = data["k"]
    sign_t = np.sign(data["t_nom"])
    n = g.size

    margins: List[np.ndarray] = []
    gamma_attack = int(max(0, min(gamma_attack, n)))
    for start in range(0, int(scenarios), batch):
        b = min(batch, int(scenarios) - start)
        if gamma_attack == 0:
            realized = np.broadcast_to(g, (b, n))
            margin_batch = realized @ k
            margins.append(margin_batch)
            continue

        xi = np.zeros((b, n), dtype=float)
        # Uniform random subset per scenario without replacement via random ranks.
        ranks = rng.random((b, n))
        idx = np.argpartition(ranks, kth=gamma_attack - 1, axis=1)[:, :gamma_attack]
        draws = rng.uniform(-1.0, 0.0, size=(b, gamma_attack))
        rows = np.arange(b)[:, None]
        xi[rows, idx] = draws * sign_t[idx]
        realized = np.maximum(0.0, g[None, :] + xi * d[None, :])
        margin_batch = realized @ k
        margins.append(margin_batch)

    arr = np.concatenate(margins) if margins else np.array([], dtype=float)
    if arr.size == 0:
        return 0.0, float("nan")
    return float(np.mean(arr < -EPS)), float(np.quantile(arr, 0.05))


def simulate_iid_exact(
    prefix: GeneratedPrefix,
    selections: Sequence[int],
    rng: np.random.Generator,
    scenarios: int,
    *,
    batch: int = 1000,
) -> Tuple[float, float]:
    """I.I.D. stress test with ξ_i ~ Uniform(-1,1) and clipped demand."""

    data = selected_option_data(prefix, selections)
    g = data["g_hat"]
    d = data["delta"]
    k = data["k"]
    n = g.size

    margins: List[np.ndarray] = []
    for start in range(0, int(scenarios), batch):
        b = min(batch, int(scenarios) - start)
        xi = rng.uniform(-1.0, 1.0, size=(b, n))
        realized = np.maximum(0.0, g[None, :] + xi * d[None, :])
        margin_batch = realized @ k
        margins.append(margin_batch)

    arr = np.concatenate(margins) if margins else np.array([], dtype=float)
    if arr.size == 0:
        return 0.0, float("nan")
    return float(np.mean(arr < -EPS)), float(np.quantile(arr, 0.05))


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def try_import_scipy():
    try:
        import scipy.optimize as opt

        return opt
    except Exception:
        return None


def progress(iterable, total: Optional[int] = None, desc: Optional[str] = None, leave: bool = True):
    """Optional tqdm progress wrapper."""

    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=leave)
    except Exception:
        return iterable


def solve_mckp_milp(
    values: List[np.ndarray],
    costs: List[np.ndarray],
    capacity: float,
    *,
    time_limit: float = 120.0,
) -> Optional[Dict[str, object]]:
    """Solve fixed-theta discrete MCKP via SciPy HiGHS MILP if available."""

    opt = try_import_scipy()
    if opt is None or not hasattr(opt, "milp"):
        return None

    n_items = len(values)
    sizes = [len(v) for v in values]
    total = int(sum(sizes))

    c = -np.concatenate(values)
    integrality = np.ones(total, dtype=int)
    bounds = opt.Bounds(lb=np.zeros(total), ub=np.ones(total))

    A_eq = np.zeros((n_items, total), dtype=float)
    offset = 0
    for i, m_i in enumerate(sizes):
        A_eq[i, offset : offset + m_i] = 1.0
        offset += m_i
    b_eq = np.ones(n_items, dtype=float)

    A_ub = np.concatenate(costs, dtype=float)[None, :]
    b_ub = np.array([float(capacity)], dtype=float)
    constraints = [
        opt.LinearConstraint(A_eq, b_eq, b_eq),
        opt.LinearConstraint(A_ub, -np.inf, b_ub),
    ]

    res = opt.milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": float(time_limit)},
    )
    if res is None:
        return None

    objective = -float(res.fun) if getattr(res, "fun", None) is not None else float("nan")
    mip_gap = float(getattr(res, "mip_gap", float("nan")))
    status = int(getattr(res, "status", -999))
    success = bool(getattr(res, "success", False))

    return {
        "objective": objective if success else objective,
        "mip_gap": mip_gap,
        "status": status,
        "success": success,
        "message": str(getattr(res, "message", "")),
    }


def solve_lp_highs_full(
    values: List[np.ndarray],
    costs: List[np.ndarray],
    capacity: float,
) -> Optional[Dict[str, float]]:
    """Solve LP relaxation via SciPy linprog (HiGHS), returning objective and duals."""

    opt = try_import_scipy()
    if opt is None or not hasattr(opt, "linprog"):
        return None

    n_items = len(values)
    sizes = [len(v) for v in values]
    total = int(sum(sizes))
    c = -np.concatenate(values)

    A_eq = np.zeros((n_items, total), dtype=float)
    offset = 0
    for i, m_i in enumerate(sizes):
        A_eq[i, offset : offset + m_i] = 1.0
        offset += m_i
    b_eq = np.ones(n_items, dtype=float)

    A_ub = np.concatenate(costs, dtype=float)[None, :]
    b_ub = np.array([float(capacity)], dtype=float)

    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
    if not getattr(res, "success", False):
        return None

    knap_dual_raw = float("nan")
    try:
        # HiGHS marginal for A_ub x <= b_ub in the minimization form.
        # For the maximization knapsack density, the sign is flipped.
        knap_dual_raw = float(res.ineqlin.marginals[0])  # type: ignore[attr-defined]
    except Exception:
        pass

    return {
        "objective": -float(res.fun),
        "knapsack_dual_min_form": knap_dual_raw,
        "knapsack_dual_max_form": -knap_dual_raw if np.isfinite(knap_dual_raw) else float("nan"),
    }
