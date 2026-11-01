"""Panel H generator (protocol v2): dominant-jump, heterogeneous-scale portfolios.

As in Proposition prop:tight, one product carries a large value jump relative to
the rest of the portfolio: its demand scale is a multiple of the largest other
scale and its menu is coarse.  Demand scales of the other products are
log-uniform over three orders of magnitude, the price band is wider than in
Panels S and L, and the margin target is close to the largest feasible ratio,
so many thresholds are near-infeasible.  The output has the same binary64
primitive format as study.instances.generate_base and is encoded by the same
exact primitive path.
"""
from __future__ import annotations

import numpy as np

from .instances import inward_generated_price

PANEL_CODE = 3
SIGMA_H = 0.4
TAU_H = 0.85
DOMINANT_FACTOR = 10.0
DOMINANT_M = 3
EPS_RANGE = (1.5, 3.5)


def _kappa_tau(a, q, w, menus, demands, tau):
    """As study.instances._kappa, with the tightness tau as a parameter."""
    rev = [w[i] * np.asarray(menus[i]) * np.asarray(demands[i]) for i in range(len(a))]
    jr = [int(np.argmax(r)) for r in rev]
    R = sum(rev[i][jr[i]] for i in range(len(a)))
    Q = sum(w[i] * q[i] * demands[i][jr[i]] for i in range(len(a)))
    k_lo = R / Q

    def feasible(k):
        return sum(np.max(w[i] * (np.asarray(menus[i]) - k * q[i]) * np.asarray(demands[i]))
                   for i in range(len(a))) >= 0.0

    lo, hi = k_lo, k_lo
    while feasible(hi):
        hi *= 1.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return float(k_lo + tau * (lo - k_lo))


def _rng(salt, n, m, rep):
    return np.random.default_rng(np.random.SeedSequence([salt, PANEL_CODE, 4, n, m, rep]))


def generate_base_h(salt, n, m, rep, tau=None, factor=None, sigma=None, eps_range=None, dom_m=None):
    tau = TAU_H if tau is None else tau
    factor = DOMINANT_FACTOR if factor is None else factor
    sig = SIGMA_H if sigma is None else sigma
    er = EPS_RANGE if eps_range is None else eps_range
    dm = DOMINANT_M if dom_m is None else dom_m
    rng = _rng(salt, n, m, rep)
    a = rng.uniform(10.0, 60.0, n)
    q = a * rng.uniform(0.55, 0.85, n)
    w = rng.uniform(0.5, 2.0, n)
    eps = rng.uniform(er[0], er[1], n)
    scale = 10.0 ** rng.uniform(1.0, 4.0, n)
    scale[0] = factor * scale[1:].max()
    menus, demands = [], []
    for i in range(n):
        mi = dm if i == 0 else m
        lo, hi = a[i] * (1 - sig), a[i] * (1 + sig)
        grid = lo + (hi - lo) * np.arange(mi) / (mi - 1)
        x = np.asarray([inward_generated_price(float(p), float(a[i]), sig) for p in grid])
        g = scale[i] * (x / a[i]) ** (-eps[i])
        menus.append([float(p) for p in x])
        demands.append([float(d) for d in g])
    kappa = _kappa_tau(a, q, w, menus, demands, tau)
    return dict(
        family="dominant", n=n, m=m, rep=rep, salt=salt, panel_code=PANEL_CODE,
        references=[float(x).hex() for x in a], costs=[float(x).hex() for x in q],
        exposures=[float(x).hex() for x in w], tolerances=[float(sig).hex()] * n,
        menus=[[p.hex() for p in row] for row in menus],
        demands=[[d.hex() for d in row] for row in demands],
        kappa=float(kappa).hex(),
    )
