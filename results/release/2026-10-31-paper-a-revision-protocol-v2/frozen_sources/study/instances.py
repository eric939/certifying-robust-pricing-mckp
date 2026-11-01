"""Instance generation and exact primitive encoding for the TOP-revision study.

Base portfolios are generated from a frozen specification and stored as
binary64 primitives (hex strings).  Coefficients are formed by the corrected
exact primitive-input path of the repository (local commit 75d4704,
`robust_mckp.primitive_certificate.ExactPrimitivePricingData`): prices are
filtered by exact rational admissibility |x-a| <= sigma*a, original menu IDs
are retained, and v, s, t are exact rationals.  Solvers receive the correctly
rounded binary64 values of these rationals; every returned plan is re-checked
in exact rational arithmetic against the primitives.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

EXACT_SRC = Path("/Users/ericshao/.local/share/robust-mckp/worktrees/paper-a-exact-primitive-75d47042/src")
if str(EXACT_SRC) not in sys.path:
    sys.path.insert(0, str(EXACT_SRC))

from robust_mckp.primitive_admissibility import inward_generated_price, price_is_admissible  # noqa: E402
from robust_mckp.primitive_certificate import ExactPrimitivePricingData  # noqa: E402

FAMILIES = {"regular": 1, "charm": 2, "noisy": 3}
SIGMA = 0.2
TAU = 0.5  # margin-target tightness between revenue-maximizing and ratio-maximizing plans
CHARM_ENDINGS = (0.09, 0.19, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.89, 0.99)
CHARM_BONUS = {0.99: 1.06, 0.49: 1.03, 0.95: 1.0}


def _rng(salt, panel_code, family, n, m, rep):
    return np.random.default_rng(np.random.SeedSequence([salt, panel_code, FAMILIES[family], n, m, rep]))


def _kappa(a, q, w, menus, demands):
    """Margin target between the revenue-maximizing and ratio-maximizing plans."""
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
    k_hi = lo
    return float(k_lo + TAU * (k_hi - k_lo))


def generate_base(salt, panel_code, family, n, m, rep):
    """Return the binary64 primitive dictionary of one base portfolio."""
    rng = _rng(salt, panel_code, family, n, m, rep)
    a = rng.uniform(10.0, 60.0, n)
    q = a * rng.uniform(0.55, 0.85, n)
    w = rng.uniform(0.5, 2.0, n)
    eps = rng.uniform(0.6, 2.5, n)
    scale = rng.uniform(50.0, 500.0, n)
    menus, demands = [], []
    for i in range(n):
        lo, hi = a[i] * (1 - SIGMA), a[i] * (1 + SIGMA)
        if family in ("regular", "noisy"):
            grid = lo + (hi - lo) * np.arange(m) / (m - 1)
            prices = [inward_generated_price(float(x), float(a[i]), SIGMA) for x in grid]
            mult = np.ones(m)
            if family == "noisy":
                mult = np.exp(rng.normal(0.0, 0.08, m))
        else:
            cand = []
            for dollar in range(int(math.floor(lo)) - 1, int(math.ceil(hi)) + 1):
                for e in CHARM_ENDINGS:
                    x = float(dollar) + e
                    if x > 0 and price_is_admissible(x, float(a[i]), SIGMA):
                        cand.append((x, e))
            pick = np.sort(rng.choice(len(cand), size=m, replace=False))
            prices = [cand[k][0] for k in pick]
            mult = np.array([CHARM_BONUS.get(cand[k][1], 1.0) for k in pick])
        x = np.asarray(prices)
        g = scale[i] * (x / a[i]) ** (-eps[i]) * mult
        menus.append([float(p) for p in x])
        demands.append([float(d) for d in g])
    kappa = _kappa(a, q, w, menus, demands)
    return dict(
        family=family, n=n, m=m, rep=rep, salt=salt, panel_code=panel_code,
        references=[float(x).hex() for x in a], costs=[float(x).hex() for x in q],
        exposures=[float(x).hex() for x in w], tolerances=[float(SIGMA).hex()] * n,
        menus=[[p.hex() for p in row] for row in menus],
        demands=[[d.hex() for d in row] for row in demands],
        kappa=float(kappa).hex(),
    )


def primitive_data(base, radius, gamma):
    f = float.fromhex
    return ExactPrimitivePricingData(
        references=tuple(f(x) for x in base["references"]),
        costs=tuple(f(x) for x in base["costs"]),
        exposures=tuple(f(x) for x in base["exposures"]),
        tolerances=tuple(f(x) for x in base["tolerances"]),
        menus=tuple(tuple(f(p) for p in row) for row in base["menus"]),
        demands=tuple(tuple(f(d) for d in row) for row in base["demands"]),
        kappa=f(base["kappa"]), radius=float(radius), gamma=int(gamma))


def base_sha256(base):
    return hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Encoded:
    """Exact encoded instance plus its binary64 solver arrays."""

    def __init__(self, data: ExactPrimitivePricingData):
        self.data = data
        inst = data.encoded_instance()
        self.items = inst.items
        self.gamma = inst.gamma
        n = len(inst.items)
        off = np.zeros(n + 1, np.int64)
        v, s, t, orig = [], [], [], []
        for i, group in enumerate(inst.items):
            off[i + 1] = off[i] + len(group)
            for opt in group:
                v.append(float(opt.value))
                s.append(float(opt.margin))
                t.append(float(opt.uncertainty))
                orig.append(opt.original_index)
        self.off = off
        self.v = np.asarray(v)
        self.s = np.asarray(s)
        self.t = np.asarray(t)
        self.orig = np.asarray(orig, np.int64)
        self.n = n
        self.M = int(off[-1])
        self.n_original_options = sum(len(row) for row in data.menus)

    def original_ids(self, sel):
        return [int(self.orig[k]) for k in sel]

    def exact_check(self, sel):
        """Exact admissibility/original-ID binding, robust residual and revenue."""
        if sel is None or len(sel) != self.n:
            return dict(exact_feasible=None)
        res = Fraction(0)
        rev = Fraction(0)
        losses = []
        for i, k in enumerate(sel):
            k = int(k)
            if not (self.off[i] <= k < self.off[i + 1]):
                return dict(exact_feasible=False, exact_error="selection outside group")
            opt = self.items[i][k - int(self.off[i])]
            j = opt.original_index
            p, r, sig = self.data.menus[i][j], self.data.references[i], self.data.tolerances[i]
            if abs(p - r) > sig * r:
                return dict(exact_feasible=False, exact_error="inadmissible original ID")
            res += opt.margin
            rev += opt.value
            losses.append(opt.uncertainty)
        losses.sort(reverse=True)
        res -= sum(losses[: self.gamma], Fraction(0))
        return dict(exact_feasible=bool(res >= 0), exact_residual=float(res), exact_revenue=float(rev))

    def menu_statistics(self):
        """Delta^v_max / V_lower of Corollary 15(a) and related menu data."""
        dmax = 0.0
        vlow = 0.0
        for i in range(self.n):
            vi = self.v[self.off[i]:self.off[i + 1]]
            dmax = max(dmax, float(vi.max() - vi.min()))
            vlow += float(vi.min())
        return dict(delta_v_max=dmax, v_lower=vlow, menu_ratio=dmax / vlow,
                    trivial_bound=float(sum(self.v[self.off[i]:self.off[i + 1]].max() for i in range(self.n))))

    def thresholds(self, v=None, t=None):
        t = self.t if t is None else t
        return np.unique(np.concatenate([[0.0], t]))


def universal_dominance_filter(off, v, s, t):
    """Proposition 19: delete options dominated for every threshold.

    Option a is deleted if some b != a in its group has s_b >= s_a,
    s_b - t_b >= s_a - t_a, v_b >= v_a, with a strict inequality or b < a.
    Returns the reduced arrays and the kept global indices.
    """
    keep = []
    noff = [0]
    for i in range(off.size - 1):
        idx = list(range(off[i], off[i + 1]))
        for a_ in idx:
            dominated = False
            for b_ in idx:
                if b_ == a_:
                    continue
                ge = s[b_] >= s[a_] and s[b_] - t[b_] >= s[a_] - t[a_] and v[b_] >= v[a_]
                if not ge:
                    continue
                strict = s[b_] > s[a_] or s[b_] - t[b_] > s[a_] - t[a_] or v[b_] > v[a_]
                if strict or b_ < a_:
                    dominated = True
                    break
            if not dominated:
                keep.append(a_)
        noff.append(len(keep))
    keep = np.asarray(keep, np.int64)
    return np.asarray(noff, np.int64), v[keep], s[keep], t[keep], keep
