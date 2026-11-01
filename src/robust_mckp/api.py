"""The library interface: build an instance, solve it, verify the decision.

Typical use::

    import robust_mckp as rm

    inst = rm.instance_from_arrays(values, margins, losses, gamma=2)
    res = rm.solve(inst)                    # exact optimum (threshold DnC)
    res = rm.solve(inst, method="certified")  # HullRound interval [H, U], faster
    res.selection, res.revenue, res.upper_bound, res.gap, res.exact.feasible

Every method returns a :class:`Result`.  By default the returned decision is
re-checked in exact rational arithmetic (:func:`verify`), so a decision that is
infeasible because of binary64 rounding in a solver is reported, not hidden.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .fast import methods as _m
from .flat import FlatInstance
from .model import Option, PricingInstance
from .primitive_certificate import ExactPrimitivePricingData

InstanceLike = Union[PricingInstance, ExactPrimitivePricingData, FlatInstance]

#: Methods accepted by :func:`solve`, with the paper's name for each.
METHODS: Dict[str, str] = {
    "exact": "threshold divide and conquer with interval records (optimal)",
    "certified": "HullRound with interval records, Proposition 23 (certified interval)",
    "hullround": "HullRound, Algorithm 1, with the guarded sweep of Proposition 24",
    "hullround-full": "HullRound, Algorithm 1, rebuilding every hull at every threshold",
    "hullround-dominance": "guarded-sweep HullRound after the universal dominance filter",
    "exact-layer": "HullRound followed by the exact full-breakpoint layer (optimal)",
    "milp-highs": "compact robust MILP solved by HiGHS (needs the 'milp' extra)",
    "milp-scip": "compact robust MILP solved by SCIP (needs the 'scip' extra)",
    "bruteforce": "exhaustive enumeration (tiny instances only)",
    "reference": "the v3 pure-Python HullRound reference implementation",
}

_STATUS = {
    "OPTIMAL": "optimal",
    "CERTIFIED": "certified",
    "INFEASIBLE": "infeasible",
    "TIME_LIMIT": "time_limit",
    "TIME_LIMIT_NO_INCUMBENT": "time_limit",
}
_NO_LIMIT = 1e20  # seconds; effectively unlimited for every driver


@dataclass(frozen=True)
class Verification:
    """Exact rational check of one decision.

    Attributes:
        feasible: whether the robust constraint holds exactly.
        residual: exact robust residual (nominal surplus minus the Gamma
            largest endpoint losses); ``None`` after a structural error.
        revenue: exact revenue of the decision.
        error: reason for a structural failure (inadmissible original price).
    """

    feasible: bool
    residual: Optional[Fraction]
    revenue: Optional[Fraction]
    error: Optional[str] = None


@dataclass(frozen=True)
class Result:
    """Outcome of :func:`solve`.

    Attributes:
        method: the method that produced it (a key of :data:`METHODS`).
        status: ``"optimal"`` (exact methods, relative tolerance 1e-6),
            ``"certified"`` (HullRound: ``revenue <= OPT <= upper_bound``),
            ``"feasible"`` (reference method), ``"time_limit"``,
            ``"infeasible"`` or ``"error: ..."``.
        selection: chosen option per group, as an index into that group's menu
            of the solved instance; ``None`` if no decision was found.
        original_ids: the same choice as IDs in the supplied, unfiltered menus.
        revenue: binary64 revenue of the decision (H for HullRound).
        upper_bound: a valid upper bound U on the optimum, when the method has one.
        gap: ``(upper_bound - revenue) / upper_bound``.
        elapsed: wall-clock seconds spent in the method.
        exact: the exact rational check of the decision (``verify=True``).
        details: the method's raw diagnostics (threshold counts, nodes, ...).
    """

    method: str
    status: str
    selection: Optional[Tuple[int, ...]]
    original_ids: Optional[Tuple[int, ...]]
    revenue: Optional[float]
    upper_bound: Optional[float]
    gap: Optional[float]
    elapsed: float
    exact: Optional[Verification] = None
    details: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def feasible(self) -> bool:
        """True if a decision was returned and (when checked) is exactly feasible."""
        if self.selection is None:
            return False
        return True if self.exact is None else self.exact.feasible


def _flat(instance: InstanceLike) -> FlatInstance:
    return instance if isinstance(instance, FlatInstance) else FlatInstance(instance)


def instance_from_arrays(values: Sequence[Sequence[float]], margins: Sequence[Sequence[float]],
                         losses: Sequence[Sequence[float]], gamma: int, *,
                         name: Optional[str] = None) -> PricingInstance:
    """Build an instance directly from the MCKP coefficients.

    ``values[i][j]``, ``margins[i][j]`` and ``losses[i][j]`` are the revenue
    v, nominal surplus s and endpoint loss t >= 0 of option j in group i.  At
    most ``gamma`` groups take their endpoint loss in the robust constraint
    ``sum_i s_i - (sum of the gamma largest t_i) >= 0``.  Floats are read
    as their exact binary64 values; ``fractions.Fraction`` entries are kept
    exactly by :func:`verify` and :func:`certify`.
    """
    if not (len(values) == len(margins) == len(losses)):
        raise ValueError("values, margins and losses need one row per group")
    items = []
    for i, (vr, sr, tr) in enumerate(zip(values, margins, losses)):
        if not (len(vr) == len(sr) == len(tr)):
            raise ValueError(f"group {i}: values, margins and losses differ in length")
        for x in tr:
            if x < 0:
                raise ValueError(f"group {i}: endpoint losses must be nonnegative")
        items.append([Option(value=vj, margin=sj, uncertainty=tj) for vj, sj, tj in zip(vr, sr, tr)])
    return PricingInstance(items=items, gamma=gamma, name=name)


def verify(instance: InstanceLike, selection: Sequence[int]) -> Verification:
    """Check a decision in exact rational arithmetic.

    ``selection[i]`` is an index into group i's menu of ``instance``.  For
    :class:`ExactPrimitivePricingData` the menus are the admissible options and
    the admissibility of every chosen price is re-checked.
    """
    flat = _flat(instance)
    r = flat.exact_check(selection)
    return Verification(feasible=bool(r["feasible"]), residual=r["residual"], revenue=r["revenue"],
                        error=r.get("error"))


_verify = verify  # solve() has a parameter of the same name


def certify(instance: Union[PricingInstance, ExactPrimitivePricingData], selection: Sequence[int], *,
            include_rounding_provenance: bool = False):
    """Exact global primal-dual certificate of a decision (Theorem 12).

    Returns a :class:`~robust_mckp.global_certificate.GlobalDualCertificate`
    whose exact upper bound holds for every threshold of the family; check it
    with :func:`~robust_mckp.verify_global_certificate`.  This runs in exact
    rational arithmetic and is meant for small and medium instances.
    """
    from .global_certificate import build_global_certificate

    inst = instance.encoded_instance() if isinstance(instance, ExactPrimitivePricingData) else instance
    return build_global_certificate(inst, [int(j) for j in selection],
                                    include_rounding_provenance=include_rounding_provenance)


def _run(flat: FlatInstance, method: str, tl: float, threads: int) -> dict:
    if method == "exact":
        r = _m.run_dnc(flat, time_limit=tl)
        # The driver labels an infeasible root TIME_LIMIT_NO_INCUMBENT; a genuine
        # time-out needs elapsed > limit (the relabelling of protocol v2's runner).
        if r["status"] == "TIME_LIMIT_NO_INCUMBENT" and r["elapsed"] <= tl:
            r["status"] = "INFEASIBLE"
        return r
    if method == "certified":
        return _m.run_pipeline(flat, time_limit=tl)
    if method == "hullround":
        return _m.run_hullround(flat, sweep=True, dominance=False, time_limit=tl)
    if method == "hullround-full":
        return _m.run_hullround(flat, sweep=False, dominance=False, time_limit=tl)
    if method == "hullround-dominance":
        return _m.run_hullround(flat, sweep=True, dominance=True, time_limit=tl)
    if method == "exact-layer":
        return _m.run_exact_layer(flat, time_limit=tl)
    if method == "milp-highs":
        return _m.run_milp(flat, time_limit=tl, threads=threads)
    if method == "milp-scip":
        return _m.run_milp_scip(flat, time_limit=tl, threads=threads)
    if method == "bruteforce":
        if flat.M > 0 and np.prod([float(flat.off[i + 1] - flat.off[i]) for i in range(flat.n)]) > 5e6:
            raise ValueError("bruteforce is limited to at most 5e6 plans")
        start = time.perf_counter()
        r = _m.run_bruteforce(flat)
        r.update(status="OPTIMAL" if r["sel"] is not None else "INFEASIBLE", bound=r["objective"],
                 elapsed=time.perf_counter() - start)
        return r
    raise ValueError(f"unknown method {method!r}; choose one of {sorted(METHODS)}")


def solve(instance: InstanceLike, method: str = "exact", *, time_limit: Optional[float] = None,
          verify: bool = True, threads: int = 1) -> Result:
    """Solve a finite-menu Gamma-robust pricing / MCKP instance.

    Args:
        instance: a :class:`PricingInstance` (see :func:`instance_from_arrays`
            and :func:`from_pricing_data`), exact primitive pricing data, or a
            prepared :class:`FlatInstance` (to reuse it across calls).
        method: a key of :data:`METHODS`.  ``"exact"`` (default) returns an
            optimal decision; ``"certified"`` returns HullRound's decision with
            its certified interval at a fraction of the cost.
        time_limit: wall-clock limit in seconds (``None``: no limit).  On
            timeout the best decision and a valid upper bound are returned with
            status ``"time_limit"``.
        verify: re-check the returned decision in exact rational arithmetic.
        threads: threads for the MILP solvers (the other methods are serial).
    """
    if method == "reference":
        return _solve_reference(instance, verify)
    flat = _flat(instance)
    tl = _NO_LIMIT if time_limit is None else float(time_limit)
    r = _run(flat, method, tl, threads)
    raw = r.get("status", "")
    status = _STATUS.get(raw, "error: " + raw.lower())
    if "H" in r:  # HullRound-type result
        revenue, upper = r.get("H"), r.get("U")
    else:
        revenue, upper = r.get("objective"), r.get("bound")
    sel = r.get("sel")
    selection = tuple(flat.local(sel)) if sel is not None else None
    if selection is None:
        revenue = None
    gap = None
    if revenue is not None and upper is not None:
        gap = max(0.0, (upper - revenue) / max(abs(upper), 1e-12))
    exact = None
    if verify and selection is not None:
        exact = _verify(flat, selection)
        if not exact.feasible:
            warnings.warn(f"method {method!r} returned a decision that fails the exact check "
                          f"(residual {exact.residual}); see Result.exact", RuntimeWarning, stacklevel=2)
    details = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()
               if not k.startswith("_") and k not in ("sel", "sel_rd")}
    return Result(method=method, status=status, selection=selection,
                  original_ids=tuple(flat.original_ids(selection)) if selection is not None else None,
                  revenue=revenue, upper_bound=upper, gap=gap, elapsed=float(r.get("elapsed", 0.0)),
                  exact=exact, details=details)


def _solve_reference(instance: InstanceLike, check: bool) -> Result:
    from .solver import solve as legacy

    flat = _flat(instance)
    sol = legacy(flat.instance)
    if not sol.is_feasible:
        return Result(method="reference", status="infeasible", selection=None, original_ids=None,
                      revenue=None, upper_bound=None, gap=None, elapsed=sol.elapsed,
                      details=dict(sol.metadata or {}))
    selection = tuple(int(j) for j in sol.selections)
    return Result(method="reference", status="feasible", selection=selection,
                  original_ids=tuple(flat.original_ids(selection)), revenue=float(sol.objective),
                  upper_bound=None, gap=None, elapsed=sol.elapsed,
                  exact=verify(flat, selection) if check else None,
                  details=dict(sol.metadata or {}, theta=sol.theta, lp_value=sol.lp_value))
