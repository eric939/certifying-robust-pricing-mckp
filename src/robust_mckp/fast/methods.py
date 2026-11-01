"""Solver drivers of the revised paper, on a flat instance.

Each driver takes an object with attributes ``off, v, s, t, gamma, n, M``
(see :class:`robust_mckp.flat.FlatInstance`) and returns a dictionary; the
public entry point :func:`robust_mckp.solve` wraps them.  The drivers are the
ones of the paper's computational study, unchanged apart from module names:

* run_hullround: Algorithm 1 over the complete threshold family, full
  rebuilding or the guarded sweep (Proposition 24), optionally after the
  universal dominance filter (Proposition 19);
* run_exact_layer: HullRound followed by the exact full-breakpoint layer;
* run_dnc: the threshold divide and conquer with interval records;
* run_pipeline: HullRound with interval records (Proposition 23);
* run_milp / run_milp_scip: the compact robust counterpart (HiGHS / SCIP);
* run_bruteforce: exhaustive enumeration for tiny instances.
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from . import kernels as core

REL_TOL = 1e-6  # optimality tolerance shared by every exact method
EPS = 1e-10     # tie tolerance of the rounding heuristics (as in the repository)


def _gap(lb, ub):
    if lb is None or ub is None or not np.isfinite(lb) or not np.isfinite(ub):
        return None
    return max(0.0, (ub - lb) / max(abs(ub), 1e-12))


def run_hullround(enc, *, sweep, dominance, time_limit, start=None):
    start = time.perf_counter() if start is None else start
    deadline = start + time_limit
    off, v, s, t = enc.off, enc.v, enc.s, enc.t
    kept = None
    if dominance:
        off, v, s, t, kept = universal_dominance_filter(off, v, s, t)
    thetas = np.unique(np.concatenate([[0.0], t]))
    lp = np.full(thetas.size, -np.inf)
    feas = np.zeros(thetas.size, np.bool_)
    n = off.size - 1
    sel_b = -np.ones(n, np.int64)
    sel_rd = -np.ones(n, np.int64)
    st = core.hullround(off, v, s, t, enc.gamma, thetas, sweep, deadline, EPS, lp, feas, sel_b, sel_rd)
    elapsed = time.perf_counter() - start
    completed = bool(st[0])
    nfeas = int(st[2])
    H, Hrd, U = st[3], st[4], st[5]
    if kept is not None:
        sel_b = np.where(sel_b >= 0, kept[np.maximum(sel_b, 0)], -1)
        sel_rd = np.where(sel_rd >= 0, kept[np.maximum(sel_rd, 0)], -1)
    out = dict(
        completed=completed, thetas=int(thetas.size), thetas_evaluated=int(st[1]),
        thetas_feasible=nfeas, H=H if np.isfinite(H) else None,
        H_rd=Hrd if np.isfinite(Hrd) else None, options_after_preprocessing=int(v.size),
        hull_vertices_mean_fraction=(st[9] / (nfeas * v.size)) if nfeas else None,
        hull_builds=int(st[10]), loss_at_thetaU=st[7], jump_at_thetaU=st[8],
        lambda_at_thetaU=st[11], rd_bound_check=bool(st[13]) if completed and nfeas else None,
        rejected_candidates=int(st[14]), elapsed=elapsed,
    )
    # build-count bound of Proposition 24 (sweep): 1 + |B cap [tmin_i, tmax_i)| per group
    if sweep:
        bound = 0
        for i in range(n):
            ti = t[off[i]:off[i + 1]]
            bound += 1 + int(np.count_nonzero((thetas >= ti.min()) & (thetas < ti.max())))
        out["hull_build_bound"] = bound
    if completed:
        if nfeas == 0:
            out.update(status="INFEASIBLE", U=None)
        else:
            iu = int(st[6])
            out.update(status="CERTIFIED", U=U, theta_U=float(thetas[iu]))
            out["dual_check_U"] = float(core.theta_dual_check(off, v, s, t, enc.gamma, thetas[iu], st[11]))
    else:
        # Theorem 21 records: evaluated LP values, trivial bound for the rest
        triv = float(sum(v[off[i]:off[i + 1]].max() for i in range(n)))
        out.update(status="TIME_LIMIT", U=max(triv, float(np.max(lp[:int(st[1])], initial=-np.inf))))
    out["sel"] = sel_b if out["H"] is not None else None
    out["sel_rd"] = sel_rd if out["H_rd"] is not None else None
    out["_lp"] = lp
    out["_feas"] = feas
    out["_thetas"] = thetas
    return out


def run_exact_layer(enc, *, time_limit):
    """HullRound (guarded sweep) followed by the exact full-breakpoint layer."""
    start = time.perf_counter()
    deadline = start + time_limit
    hr = run_hullround(enc, sweep=True, dominance=False, time_limit=time_limit, start=start)
    res = dict(hr_status=hr["status"], hr_H=hr["H"], hr_U=hr["U"], hr_elapsed=hr["elapsed"])
    if hr["status"] == "INFEASIBLE":
        res.update(status="INFEASIBLE", objective=None, bound=None, sel=None,
                   elapsed=time.perf_counter() - start, nodes=0, thetas_searched=0)
        return res
    if hr["status"] != "CERTIFIED":
        res.update(status="TIME_LIMIT" if hr["H"] is not None else "TIME_LIMIT_NO_INCUMBENT",
                   objective=hr["H"], bound=hr["U"], sel=hr["sel"],
                   elapsed=time.perf_counter() - start, nodes=0, thetas_searched=0)
        return res
    lp, feas, thetas = hr["_lp"], hr["_feas"], hr["_thetas"]
    inc = hr["H"]
    sel = hr["sel"].copy()
    order = [q for q in np.argsort(-lp, kind="mergesort") if feas[q]]
    n = enc.n
    buf = -np.ones(n, np.int64)
    nodes = 0
    searched = 0
    status = "OPTIMAL"
    record = -np.inf
    pos = 0
    while pos < len(order):
        q = order[pos]
        if lp[q] <= inc + REL_TOL * max(1.0, abs(inc)):
            break
        if time.perf_counter() > deadline:
            status = "TIME_LIMIT"
            break
        buf[:] = -1
        st, found, rec, nd = core.fixed_theta_bnb(enc.off, enc.v, enc.s, enc.t, enc.gamma, thetas[q],
                                                  inc, REL_TOL, deadline, buf)
        nodes += nd
        searched += 1
        if np.isfinite(found) and found > inc:
            if core.robust_residual(enc.off, enc.s, enc.t, enc.gamma, buf) >= 0.0:
                inc = core.revenue(enc.v, buf)
                sel = buf.copy()
        if st == 1:
            status = "TIME_LIMIT"
            record = max(record, rec)
            pos += 1
            break
        pos += 1
    if status == "TIME_LIMIT":
        rest = [lp[q] for q in order[pos:]]
        ub = max([inc, record] + rest)
    else:
        ub = inc
    res.update(status=status, objective=inc, bound=ub, sel=sel, nodes=nodes, thetas_searched=searched,
               thetas_pruned_by_lp=int(len(order) - searched), elapsed=time.perf_counter() - start)
    return res


def run_dnc(enc, *, time_limit):
    """Threshold-interval divide and conquer (Hansknecht et al. 2018; Buesing et al. 2023).

    For thresholds b_l <= theta <= b_r, every plan feasible at theta satisfies
    sum_i s^{b_r}_{i,j(i)} >= Gamma*b_l because s^theta is nondecreasing in
    theta; the hull-greedy LP of this MCKP bounds the whole interval.
    Intervals are processed best-bound first; single thresholds are solved by
    the same fixed-threshold branch-and-bound as the exact layer, and the
    rounding candidates of every evaluated split threshold supply incumbents.
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
    inc = -np.inf
    sel = None
    evals = 0

    def evaluate(l, r, rounding):
        nonlocal inc, sel, evals
        b = bufs
        feas, lpv, lam, frac, alpha, dvf, cap, nh, nb = core.eval_theta(
            off, v, s, t, gamma, B[r], B[l], rebuild, True, b["sth"], b["sbar"], b["c"], b["work"],
            b["hidx"], b["hc"], b["hv"], b["hlen"], b["seg_slope"], b["seg_dc"], b["seg_dv"], b["ids"], b["pos"])
        evals += 1
        if not feas:
            return -np.inf
        if rounding:  # l == r: a genuine family member, rounding gives feasible plans
            rd, bv = core.round_candidates(off, b["hlen"], b["hidx"], b["hc"], b["hv"], cap, b["pos"], frac,
                                           EPS, cand_rd, cand_b)
            for val, cand in ((bv, cand_b), (rd, cand_rd)):
                if val > inc and core.robust_residual(off, s, t, gamma, cand) >= 0.0:
                    inc = val
                    sel = cand.copy()
        return lpv

    heap = []
    root = evaluate(0, K - 1, K == 1)
    leaves_done = 0
    if np.isfinite(root):
        heapq.heappush(heap, (-root, 0, K - 1))
    status = "OPTIMAL"
    nodes = 0
    record = -np.inf
    buf = -np.ones(n, np.int64)
    while heap:
        negb, l, r = heap[0]
        bnd = -negb
        if np.isfinite(inc) and bnd <= inc + REL_TOL * max(1.0, abs(inc)):
            break
        if time.perf_counter() > deadline:
            status = "TIME_LIMIT"
            break
        heapq.heappop(heap)
        if l == r:
            buf[:] = -1
            st, found, rec, nd = core.fixed_theta_bnb(off, v, s, t, gamma, B[l],
                                                      inc if np.isfinite(inc) else -np.inf,
                                                      REL_TOL, deadline, buf)
            nodes += nd
            leaves_done += 1
            if np.isfinite(found) and found > inc and core.robust_residual(off, s, t, gamma, buf) >= 0.0:
                inc = core.revenue(v, buf)
                sel = buf.copy()
            if st == 1:
                status = "TIME_LIMIT"
                record = rec
                break
            continue
        mid = (l + r) // 2
        evaluate(mid, mid, True) if mid != l else None
        for (a, b_) in ((l, mid), (mid + 1, r)):
            val = evaluate(a, b_, a == b_)
            if np.isfinite(val):
                heapq.heappush(heap, (-val, a, b_))
    elapsed = time.perf_counter() - start
    if status == "OPTIMAL" and not np.isfinite(inc):
        status = "INFEASIBLE" if not heap else "TIME_LIMIT_NO_INCUMBENT"
    if status == "OPTIMAL":
        ub = inc
    else:
        ub = max([record] + [-h[0] for h in heap] + ([inc] if np.isfinite(inc) else []))
        if not np.isfinite(inc):
            status = "TIME_LIMIT_NO_INCUMBENT"
    return dict(status=status, objective=inc if np.isfinite(inc) else None,
                bound=ub if np.isfinite(ub) else None, sel=sel, nodes=nodes,
                interval_evaluations=evals, leaves_solved=leaves_done, thresholds=int(K), elapsed=elapsed)


def run_milp(enc, *, time_limit, threads=1):
    """Compact Bertsimas-Sim robust counterpart solved by HiGHS."""
    import highspy

    start = time.perf_counter()
    off, v, s, t, gamma = enc.off, enc.v, enc.s, enc.t, enc.gamma
    n, M = enc.n, enc.M
    # variables: z (M binaries), theta (1), pi (n)
    nv = M + 1 + n
    tmax = float(t.max()) if t.size else 0.0
    col_cost = np.concatenate([v, [0.0], np.zeros(n)])
    lower = np.zeros(nv)
    upper = np.concatenate([np.ones(M), [tmax], np.full(n, max(tmax, 0.0))])
    rows_lo, rows_hi, starts, idx, val = [], [], [0], [], []
    # surplus row: sum s z - Gamma theta - sum pi >= 0
    idx += list(range(M)) + [M] + list(range(M + 1, M + 1 + n))
    val += list(s) + [-float(gamma)] + [-1.0] * n
    rows_lo.append(0.0)
    rows_hi.append(np.inf)
    starts.append(len(idx))
    for i in range(n):
        # pi_i + theta - sum_j t_ij z_ij >= 0
        idx += [M + 1 + i, M] + list(range(off[i], off[i + 1]))
        val += [1.0, 1.0] + [-x for x in t[off[i]:off[i + 1]]]
        rows_lo.append(0.0)
        rows_hi.append(np.inf)
        starts.append(len(idx))
    for i in range(n):
        idx += list(range(off[i], off[i + 1]))
        val += [1.0] * int(off[i + 1] - off[i])
        rows_lo.append(1.0)
        rows_hi.append(1.0)
        starts.append(len(idx))
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("mip_rel_gap", REL_TOL)
    h.setOptionValue("mip_abs_gap", 0.0)
    h.setOptionValue("random_seed", 0)
    lp = highspy.HighsLp()
    lp.num_col_ = nv
    lp.num_row_ = len(rows_lo)
    lp.col_cost_ = col_cost
    lp.col_lower_ = lower
    lp.col_upper_ = upper
    lp.row_lower_ = np.asarray(rows_lo)
    lp.row_upper_ = np.asarray(rows_hi)
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = np.asarray(starts, np.int32)
    lp.a_matrix_.index_ = np.asarray(idx, np.int32)
    lp.a_matrix_.value_ = np.asarray(val, float)
    lp.integrality_ = [highspy.HighsVarType.kInteger] * M + [highspy.HighsVarType.kContinuous] * (n + 1)
    h.passModel(lp)
    build = time.perf_counter() - start
    h.setOptionValue("time_limit", max(1.0, float(time_limit) - build))
    h.run()
    elapsed = time.perf_counter() - start
    ms = h.getModelStatus()
    info = h.getInfo()
    status_str = h.modelStatusToString(ms)
    sol = h.getSolution()
    has_sol = info.primal_solution_status >= 2
    sel = None
    integral = None
    if has_sol:
        z = np.asarray(sol.col_value[:M])
        sel = np.empty(n, np.int64)
        integral = True
        for i in range(n):
            zi = z[off[i]:off[i + 1]]
            k = int(np.argmax(zi))
            sel[i] = off[i] + k
            if abs(zi[k] - 1.0) > 1e-6 or np.sum(np.abs(zi)) - abs(zi[k]) > 1e-6:
                integral = False
    if ms == highspy.HighsModelStatus.kOptimal:
        status = "OPTIMAL"
    elif ms == highspy.HighsModelStatus.kInfeasible:
        status = "INFEASIBLE"
    elif ms == highspy.HighsModelStatus.kTimeLimit:
        status = "TIME_LIMIT" if has_sol else "TIME_LIMIT_NO_INCUMBENT"
    else:
        status = "OTHER:" + status_str
    obj = float(info.objective_function_value) if has_sol else None
    bound = float(info.mip_dual_bound) if np.isfinite(info.mip_dual_bound) else None
    return dict(status=status, highs_status=status_str, objective=obj, bound=bound,
                mip_gap=float(info.mip_gap) if np.isfinite(info.mip_gap) else None,
                nodes=int(info.mip_node_count), sel=sel, integral=integral,
                build_time=build, elapsed=elapsed)


def run_bruteforce(enc):
    sel = -np.ones(enc.n, np.int64)
    best = core.brute_force(enc.off, enc.v, enc.s, enc.t, enc.gamma, sel)
    return dict(objective=best if np.isfinite(best) else None, sel=sel if np.isfinite(best) else None)


def run_pipeline(enc, *, time_limit):
    """HullRound with interval records (Proposition 23).

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
