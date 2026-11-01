"""Protocol v2 methods: HullRound with interval records, and the compact MILP solved by SCIP."""
from __future__ import annotations

import heapq
import time

import numpy as np

from . import core
from .methods import EPS, REL_TOL


def run_pipeline(enc, *, time_limit):
    """HullRound with interval records (Proposition prop:interval-record).

    Intervals [b_l, b_r] of the threshold family are processed best bound
    first and split at the median threshold, exactly as in the threshold DnC.
    An interval whose interval-LP record is <= the current H is discarded: its
    record is kept for U.  The split threshold of every processed interval is
    also evaluated, which raises H early.  A single threshold is a HullRound evaluation (hull
    LP, round-down, repair and completion); its LP value is its final record and
    it is not searched further.  The method ends when every remaining interval
    record is <= H, and returns the certified interval [H, U] with
    U = max(H, records of single thresholds, records of discarded intervals).
    No branch-and-bound is used.
    """
    start = time.perf_counter()
    deadline = start + time_limit
    off, v, s, t, gamma = enc.off, enc.v, enc.s, enc.t, enc.gamma
    n, M = enc.n, enc.M
    B = np.unique(np.concatenate([[0.0], t]))
    K = B.size
    bufs = dict(sth=np.empty(M), sbar=np.empty(n), c=np.empty(M), work=np.empty(M, np.int64),
                hidx=np.empty(M, np.int64), hc=np.empty(M), hv=np.empty(M), hlen=np.zeros(n, np.int64),
                seg_slope=np.empty(M), seg_dc=np.empty(M), seg_dv=np.empty(M), ids=np.empty(M, np.int64),
                pos=np.empty(n, np.int64))
    rebuild = np.ones(n, np.bool_)
    cand_rd = np.empty(n, np.int64)
    cand_b = np.empty(n, np.int64)
    st = dict(H=-np.inf, Hrd=-np.inf, sel=None, sel_rd=None, evals=0, singles=0)

    def evaluate(l, r):
        b = bufs
        feas, lpv, lam, frac, alpha, dvf, cap, nh, nb = core.eval_theta(
            off, v, s, t, gamma, B[r], B[l], rebuild, True, b["sth"], b["sbar"], b["c"], b["work"],
            b["hidx"], b["hc"], b["hv"], b["hlen"], b["seg_slope"], b["seg_dc"], b["seg_dv"], b["ids"], b["pos"])
        st["evals"] += 1
        if not feas:
            return -np.inf
        if l == r:  # a family member: HullRound's rounding at this threshold
            st["singles"] += 1
            rd, bv = core.round_candidates(off, b["hlen"], b["hidx"], b["hc"], b["hv"], cap, b["pos"], frac,
                                           EPS, cand_rd, cand_b)
            if rd > st["Hrd"] and core.robust_residual(off, s, t, gamma, cand_rd) >= 0.0:
                st["Hrd"], st["sel_rd"] = rd, cand_rd.copy()
            if bv > st["H"] and core.robust_residual(off, s, t, gamma, cand_b) >= 0.0:
                st["H"], st["sel"] = bv, cand_b.copy()
        return lpv

    heap = []
    resolved = -np.inf      # records of single thresholds (final)
    discarded = -np.inf     # records of intervals discarded against H
    root = evaluate(0, K - 1)
    if np.isfinite(root):
        if K == 1:
            resolved = root
        else:
            heapq.heappush(heap, (-root, 0, K - 1))
    status = "CERTIFIED"
    while heap:
        bnd = -heap[0][0]
        if bnd <= st["H"]:
            discarded = max(discarded, bnd)
            break
        if time.perf_counter() > deadline:
            status = "TIME_LIMIT"
            break
        _, l, r = heapq.heappop(heap)
        mid = (l + r) // 2
        if mid != l:  # round at the split threshold, as the DnC does, to raise H early
            resolved = max(resolved, evaluate(mid, mid))
        for (a, b_) in ((l, mid), (mid + 1, r)):
            val = evaluate(a, b_)
            if not np.isfinite(val):
                continue
            if a == b_:
                resolved = max(resolved, val)
            else:
                heapq.heappush(heap, (-val, a, b_))
    remaining = max([-h[0] for h in heap], default=-np.inf)
    H = st["H"]
    U = max(resolved, discarded, remaining, H)
    if not np.isfinite(U):
        status = "INFEASIBLE" if status == "CERTIFIED" else "TIME_LIMIT_NO_INCUMBENT"
    elif status == "TIME_LIMIT" and not np.isfinite(H):
        status = "TIME_LIMIT_NO_INCUMBENT"
    return dict(status=status, H=H if np.isfinite(H) else None, H_rd=st["Hrd"] if np.isfinite(st["Hrd"]) else None,
                U=U if np.isfinite(U) else None, sel=st["sel"], sel_rd=st["sel_rd"], thresholds=int(K),
                interval_evaluations=st["evals"], thresholds_evaluated=st["singles"],
                open_intervals=len(heap), elapsed=time.perf_counter() - start)


def run_milp_scip(enc, *, time_limit, threads=1):
    """Compact Bertsimas-Sim robust counterpart (as run_milp) solved by SCIP."""
    from pyscipopt import Model, quicksum

    start = time.perf_counter()
    off, v, s, t, gamma = enc.off, enc.v, enc.s, enc.t, enc.gamma
    n, M = enc.n, enc.M
    tmax = float(t.max()) if t.size else 0.0
    m = Model()
    m.hideOutput()
    m.setParam("limits/gap", REL_TOL)
    m.setParam("limits/absgap", 0.0)
    m.setParam("randomization/randomseedshift", 0)
    m.setParam("parallel/maxnthreads", int(threads))
    z = [m.addVar(vtype="B", obj=float(v[k])) for k in range(M)]
    th = m.addVar(lb=0.0, ub=tmax, obj=0.0)
    pi = [m.addVar(lb=0.0, ub=max(tmax, 0.0), obj=0.0) for _ in range(n)]
    m.addCons(quicksum(float(s[k]) * z[k] for k in range(M)) - float(gamma) * th - quicksum(pi) >= 0.0)
    for i in range(n):
        m.addCons(pi[i] + th - quicksum(float(t[k]) * z[k] for k in range(off[i], off[i + 1])) >= 0.0)
        m.addCons(quicksum(z[k] for k in range(off[i], off[i + 1])) == 1.0)
    m.setMaximize()
    build = time.perf_counter() - start
    m.setParam("limits/time", max(1.0, float(time_limit) - build))
    m.optimize()
    elapsed = time.perf_counter() - start
    sstat = m.getStatus()
    has_sol = m.getNSols() > 0
    sel = None
    integral = None
    obj = None
    if has_sol:
        sol = m.getBestSol()
        zv = np.asarray([m.getSolVal(sol, x) for x in z])
        obj = float(m.getSolObjVal(sol))
        sel = np.empty(n, np.int64)
        integral = True
        for i in range(n):
            zi = zv[off[i]:off[i + 1]]
            k = int(np.argmax(zi))
            sel[i] = off[i] + k
            if abs(zi[k] - 1.0) > 1e-6 or np.sum(np.abs(zi)) - abs(zi[k]) > 1e-6:
                integral = False
    if sstat in ("optimal", "gaplimit"):
        status = "OPTIMAL"
    elif sstat == "infeasible":
        status = "INFEASIBLE"
    elif sstat == "timelimit":
        status = "TIME_LIMIT" if has_sol else "TIME_LIMIT_NO_INCUMBENT"
    else:
        status = "OTHER:" + sstat
    db = m.getDualbound()
    bound = float(db) if np.isfinite(db) and abs(db) < 1e19 else None
    return dict(status=status, scip_status=sstat, objective=obj, bound=bound,
                mip_gap=float(m.getGap()) if has_sol and np.isfinite(m.getGap()) else None,
                nodes=int(m.getNNodes()), sel=sel, integral=integral, build_time=build, elapsed=elapsed)
