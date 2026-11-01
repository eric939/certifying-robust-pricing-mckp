"""Compiled kernels for the TOP-revision computational study of Paper A.

The instance is stored flat: group i owns options off[i]..off[i+1]-1 with
binary64 coefficients v (revenue), s (nominal surplus) and t (endpoint loss).
All kernels are single-threaded.  They implement, in the paper's notation:

* eval_theta: one member of the threshold family (s^theta, baselines,
  c^theta, C^theta), canonical hulls (Definition 6), the hull-greedy LP
  (Theorem 8) by weighted selection, and the rounding of Section 5.4;
* hullround: Algorithm 1 over the complete threshold family, either rebuilding
  every hull at every threshold ("full") or reusing a group's hull when
  Proposition 23 shows its transformed point set is unchanged ("sweep");
* fixed_theta_bnb: depth-first branch-and-bound for one threshold with the
  integer-safe dominance rule (Definition 17, Lemma 18);
* brute_force: exhaustive enumeration for the validation panel.

Times are checked through objmode; kernels stop at a wall-clock deadline
expressed in time.perf_counter() seconds.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, objmode

CACHE = True


@njit(cache=CACHE)
def _now():
    with objmode(t="float64"):
        t = time.perf_counter()
    return t


@njit(cache=CACHE)
def robust_residual(off, s, t, gamma, sel):
    """Res_Gamma of a selection given by global option indices (eq. certify_alg)."""
    n = off.size - 1
    tot = 0.0
    ts = np.empty(n)
    for i in range(n):
        tot += s[sel[i]]
        ts[i] = t[sel[i]]
    if gamma > 0:
        ts = np.sort(ts)
        for k in range(gamma):
            tot -= ts[n - 1 - k]
    return tot


@njit(cache=CACHE)
def revenue(v, sel):
    r = 0.0
    for i in range(sel.size):
        r += v[sel[i]]
    return r


@njit(cache=CACHE)
def transformed(off, s, t, gamma, theta_s, theta_r, sth, sbar, c):
    """Fill s^theta, group baselines and c^theta; return C (rhs uses theta_r)."""
    n = off.size - 1
    cap = 0.0
    for i in range(n):
        best = -np.inf
        for k in range(off[i], off[i + 1]):
            a = s[k]
            b = s[k] - t[k] + theta_s
            x = a if a < b else b
            sth[k] = x
            if x > best:
                best = x
        sbar[i] = best
        cap += best
        for k in range(off[i], off[i + 1]):
            c[k] = best - sth[k]
    return cap - gamma * theta_r


@njit(cache=CACHE)
def _before(ci, vi, ii, cj, vj, ij):
    # order: cost ascending, value descending, index ascending
    if ci < cj:
        return True
    if ci > cj:
        return False
    if vi > vj:
        return True
    if vi < vj:
        return False
    return ii < ij


@njit(cache=CACHE)
def build_group_hull(i, off, c, v, work, hidx, hc, hv):
    """Canonical upper hull of group i (Definition 6); returns its length.

    Hull vertices are written to hidx/hc/hv starting at off[i].
    """
    lo = off[i]
    m = off[i + 1] - lo
    for a in range(m):
        work[lo + a] = lo + a
    # insertion sort of the group's option indices
    for a in range(1, m):
        key = work[lo + a]
        b = a - 1
        while b >= 0 and _before(c[key], v[key], key, c[work[lo + b]], v[work[lo + b]], work[lo + b]):
            work[lo + b + 1] = work[lo + b]
            b -= 1
        work[lo + b + 1] = key
    h = 0
    for a in range(m):
        k = work[lo + a]
        if h > 0 and v[k] <= hv[lo + h - 1]:
            continue  # equal cost or dominated (steps 1 and 2)
        while h >= 2:
            oc = hc[lo + h - 2]
            ov = hv[lo + h - 2]
            cross = (hc[lo + h - 1] - oc) * (v[k] - ov) - (hv[lo + h - 1] - ov) * (c[k] - oc)
            if cross >= 0.0:
                h -= 1  # middle point below or on the chord (step 3)
            else:
                break
        hidx[lo + h] = k
        hc[lo + h] = c[k]
        hv[lo + h] = v[k]
        h += 1
    return h


@njit(cache=CACHE)
def _seg_before(slope, a, b):
    if slope[a] > slope[b]:
        return True
    if slope[a] < slope[b]:
        return False
    return a < b


@njit(cache=CACHE)
def lp_select(off, hlen, hc, hv, cap, seg_slope, seg_dc, seg_dv, ids):
    """Hull-greedy LP by weighted selection in expected O(number of segments).

    Returns (filled value above baselines, critical segment id or -1, capacity
    remaining before the critical segment).  Segment id off[i]+k is the
    segment from hull vertex k to k+1 of group i.
    """
    n = off.size - 1
    nseg = 0
    for i in range(n):
        for k in range(hlen[i] - 1):
            sid = off[i] + k
            seg_dc[sid] = hc[off[i] + k + 1] - hc[off[i] + k]
            seg_dv[sid] = hv[off[i] + k + 1] - hv[off[i] + k]
            seg_slope[sid] = seg_dv[sid] / seg_dc[sid]
            ids[nseg] = sid
            nseg += 1
    lo = 0
    hi = nseg
    rem = cap
    val = 0.0
    while lo < hi:
        mid = (lo + hi) // 2
        piv = ids[mid]
        ids[mid] = ids[hi - 1]
        ids[hi - 1] = piv
        p = lo
        wsum = 0.0
        vsum = 0.0
        for q in range(lo, hi - 1):
            x = ids[q]
            if _seg_before(seg_slope, x, piv):
                ids[q] = ids[p]
                ids[p] = x
                p += 1
                wsum += seg_dc[x]
                vsum += seg_dv[x]
        ids[hi - 1] = ids[p]
        ids[p] = piv
        if wsum > rem:
            hi = p
            continue
        rem -= wsum
        val += vsum
        if seg_dc[piv] > rem:
            return val, piv, rem
        rem -= seg_dc[piv]
        val += seg_dv[piv]
        lo = p + 1
    return val, -1, rem


@njit(cache=CACHE)
def lp_positions(off, hlen, seg_slope, crit, pos):
    """Round-down hull positions of the greedy LP solution."""
    n = off.size - 1
    for i in range(n):
        p = 0
        for k in range(hlen[i] - 1):
            sid = off[i] + k
            if crit < 0 or _seg_before(seg_slope, sid, crit):
                p += 1
            else:
                break
        pos[i] = p


@njit(cache=CACHE)
def _repair(off, hlen, hc, hv, pos, cap, cost, value, exclude, eps):
    n = off.size - 1
    while cost > cap:
        best = -1
        br = np.inf
        for i in range(n):
            if i == exclude or pos[i] <= 0:
                continue
            a = off[i] + pos[i]
            dc = hc[a] - hc[a - 1]
            if dc <= 0.0:
                continue
            r = (hv[a] - hv[a - 1]) / dc
            if r < br - eps:
                br = r
                best = i
        if best < 0:
            return False, cost, value
        a = off[best] + pos[best]
        cost -= hc[a] - hc[a - 1]
        value -= hv[a] - hv[a - 1]
        pos[best] -= 1
    return True, cost, value


@njit(cache=CACHE)
def _complete(off, hlen, hc, hv, pos, cap, cost, value, eps):
    n = off.size - 1
    res = cap - cost
    while res > 0.0:
        best = -1
        br = -np.inf
        bdc = 0.0
        bdv = 0.0
        for i in range(n):
            if pos[i] >= hlen[i] - 1:
                continue
            a = off[i] + pos[i]
            dc = hc[a + 1] - hc[a]
            if dc <= 0.0 or dc > res:
                continue
            r = (hv[a + 1] - hv[a]) / dc
            if r > br + eps:
                br = r
                best = i
                bdc = dc
                bdv = hv[a + 1] - hv[a]
        if best < 0:
            break
        pos[best] += 1
        cost += bdc
        value += bdv
        res = cap - cost
    return cost, value


@njit(cache=CACHE)
def _plan_cost_value(off, hc, hv, pos):
    cost = 0.0
    value = 0.0
    for i in range(off.size - 1):
        cost += hc[off[i] + pos[i]]
        value += hv[off[i] + pos[i]]
    return cost, value


@njit(cache=CACHE)
def round_candidates(off, hlen, hidx, hc, hv, cap, pos_lp, frac, eps, sel_rd, sel_best):
    """Round-down x_down and the repaired/completed candidate of Section 5.4.

    pos_lp holds round-down positions; frac is the fractional group or -1.
    Writes global option indices to sel_rd and sel_best and returns their
    revenues (sel_best is -inf revenue if every candidate failed).
    """
    n = off.size - 1
    pos_d = pos_lp.copy()
    cost_d, val_d = _plan_cost_value(off, hc, hv, pos_d)
    rd_value = val_d
    for i in range(n):
        sel_rd[i] = hidx[off[i] + pos_d[i]]
    ok_d, cost_d, val_d = _repair(off, hlen, hc, hv, pos_d, cap, cost_d, val_d, -1, eps)
    use_up = False
    pos_u = pos_lp.copy()
    if frac >= 0:
        pos_u[frac] += 1
        cost_u, val_u = _plan_cost_value(off, hc, hv, pos_u)
        ok_u, cost_u, val_u = _repair(off, hlen, hc, hv, pos_u, cap, cost_u, val_u, frac, eps)
        if ok_u and (not ok_d or val_u > val_d):
            use_up = True
    if use_up:
        pos_b = pos_u
        cost_b, val_b = cost_u, val_u
    elif ok_d:
        pos_b = pos_d
        cost_b, val_b = cost_d, val_d
    else:
        return rd_value, -np.inf
    cost_b, val_b = _complete(off, hlen, hc, hv, pos_b, cap, cost_b, val_b, eps)
    for i in range(n):
        sel_best[i] = hidx[off[i] + pos_b[i]]
    return rd_value, val_b


@njit(cache=CACHE)
def eval_theta(off, v, s, t, gamma, theta_s, theta_r, rebuild, first,
               sth, sbar, c, work, hidx, hc, hv, hlen,
               seg_slope, seg_dc, seg_dv, ids, pos):
    """Evaluate one threshold (theta_s = theta_r) or one interval relaxation.

    rebuild[i] = False reuses the hull of group i stored in hidx/hc/hv.
    Returns (feasible, LP value, critical slope lambda, fractional group,
    alpha, fractional value jump, capacity, hull vertices, hulls rebuilt).
    """
    n = off.size - 1
    cap = transformed(off, s, t, gamma, theta_s, theta_r, sth, sbar, c)
    if cap < 0.0:
        return False, -np.inf, 0.0, -1, 0.0, 0.0, cap, 0, 0
    nb = 0
    nh = 0
    base = 0.0
    for i in range(n):
        if first or rebuild[i]:
            hlen[i] = build_group_hull(i, off, c, v, work, hidx, hc, hv)
            nb += 1
        nh += hlen[i]
        base += hv[off[i]]
    val, crit, rem = lp_select(off, hlen, hc, hv, cap, seg_slope, seg_dc, seg_dv, ids)
    lp_positions(off, hlen, seg_slope, crit, pos)
    if crit < 0:
        return True, base + val, 0.0, -1, 0.0, 0.0, cap, nh, nb
    alpha = rem / seg_dc[crit]
    g = -1
    # group of the critical segment
    lo = 0
    hi = n
    while hi - lo > 1:
        md = (lo + hi) // 2
        if off[md] <= crit:
            lo = md
        else:
            hi = md
    g = lo
    if alpha <= 0.0:
        return True, base + val, seg_slope[crit], -1, 0.0, seg_dv[crit], cap, nh, nb
    return True, base + val + alpha * seg_dv[crit], seg_slope[crit], g, alpha, seg_dv[crit], cap, nh, nb


@njit(cache=CACHE)
def hullround(off, v, s, t, gamma, thetas, sweep, deadline, eps,
              lp_out, feas_out, sel_best, sel_rd):
    """Algorithm 1 over the sorted threshold family `thetas`.

    Returns a stats vector:
      0 completed (1/0), 1 thetas evaluated, 2 feasible thetas,
      3 H (repair+completion), 4 H_rd (round-down only), 5 U,
      6 index of theta^U, 7 loss l at theta^U, 8 value jump at theta^U,
      9 hull vertices summed over feasible thetas, 10 hulls built,
      11 lambda at theta^U, 12 capacity at theta^U,
      13 round-down loss check flag (H_rd >= U - l - tolerance), 14 rejected
      candidates (float residual < 0).
    """
    n = off.size - 1
    M = v.size
    sth = np.empty(M)
    sbar = np.empty(n)
    c = np.empty(M)
    work = np.empty(M, np.int64)
    hidx = np.empty(M, np.int64)
    hc = np.empty(M)
    hv = np.empty(M)
    hlen = np.zeros(n, np.int64)
    seg_slope = np.empty(M)
    seg_dc = np.empty(M)
    seg_dv = np.empty(M)
    ids = np.empty(M, np.int64)
    pos = np.empty(n, np.int64)
    rebuild = np.ones(n, np.bool_)
    cand_rd = np.empty(n, np.int64)
    cand_b = np.empty(n, np.int64)
    tmin = np.empty(n)
    tmax = np.empty(n)
    for i in range(n):
        lo_t = np.inf
        hi_t = -np.inf
        for k in range(off[i], off[i + 1]):
            if t[k] < lo_t:
                lo_t = t[k]
            if t[k] > hi_t:
                hi_t = t[k]
        tmin[i] = lo_t
        tmax[i] = hi_t
    stats = np.zeros(15)
    H = -np.inf
    Hrd = -np.inf
    U = -np.inf
    iu = -1
    lu = 0.0
    du = 0.0
    lamu = 0.0
    capu = 0.0
    hull_vertices = 0.0
    built = 0.0
    nfeas = 0
    rejected = 0
    first = True
    prev = 0.0
    ndone = 0
    for q in range(thetas.size):
        theta = thetas[q]
        if (q & 63) == 0 and q > 0:
            if _now() > deadline:
                break
        if sweep and not first:
            for i in range(n):
                rebuild[i] = not (tmax[i] <= prev or tmin[i] >= theta)
        feas, lp, lam, frac, alpha, dvf, cap, nh, nb = eval_theta(
            off, v, s, t, gamma, theta, theta, rebuild, first or not sweep,
            sth, sbar, c, work, hidx, hc, hv, hlen, seg_slope, seg_dc, seg_dv, ids, pos)
        built += nb
        prev = theta
        ndone = q + 1
        lp_out[q] = lp
        feas_out[q] = feas
        if not feas:
            # hulls are not built at infeasible thresholds; force a rebuild next time
            first = True
            continue
        first = False
        nfeas += 1
        hull_vertices += nh
        rd_val, b_val = round_candidates(off, hlen, hidx, hc, hv, cap, pos, frac, eps, cand_rd, cand_b)
        if lp > U:
            U = lp
            iu = q
            lu = alpha * dvf if frac >= 0 else 0.0
            du = dvf if frac >= 0 else 0.0
            lamu = lam
            capu = cap
        if rd_val > Hrd:
            if robust_residual(off, s, t, gamma, cand_rd) >= 0.0:
                Hrd = rd_val
                sel_rd[:] = cand_rd
            else:
                rejected += 1
        if b_val > H:
            if robust_residual(off, s, t, gamma, cand_b) >= 0.0:
                H = b_val
                sel_best[:] = cand_b
            else:
                rejected += 1
    stats[0] = 1.0 if ndone == thetas.size else 0.0
    stats[1] = ndone
    stats[2] = nfeas
    stats[3] = H
    stats[4] = Hrd
    stats[5] = U
    stats[6] = iu
    stats[7] = lu
    stats[8] = du
    stats[9] = hull_vertices
    stats[10] = built
    stats[11] = lamu
    stats[12] = capu
    stats[13] = 1.0 if Hrd >= U - lu - 1e-9 * max(1.0, abs(U)) else 0.0
    stats[14] = rejected
    return stats


@njit(cache=CACHE)
def dual_value(off, v, c, cap, lam):
    """D_theta(lambda) of eq. (groupwise-dual), an independent LP upper bound."""
    n = off.size - 1
    tot = lam * cap
    for i in range(n):
        best = -np.inf
        for k in range(off[i], off[i + 1]):
            x = v[k] - lam * c[k]
            if x > best:
                best = x
        tot += best
    return tot


@njit(cache=CACHE)
def theta_dual_check(off, v, s, t, gamma, theta, lam):
    n = off.size - 1
    M = v.size
    sth = np.empty(M)
    sbar = np.empty(n)
    c = np.empty(M)
    cap = transformed(off, s, t, gamma, theta, theta, sth, sbar, c)
    return dual_value(off, v, c, cap, lam)


# ---------------------------------------------------------------- exact layer

@njit(cache=CACHE)
def _node_bound(n, uoff, uv, uc, hv0, sorted_seg, seg_group, seg_dc, seg_dv, nseg,
                fixed, cap_left, base_val):
    """LP bound of a node: fixed groups excluded, free groups greedy-filled."""
    val = base_val
    rem = cap_left
    for q in range(nseg):
        sid = sorted_seg[q]
        g = seg_group[sid]
        if fixed[g] >= 0:
            continue
        dc = seg_dc[sid]
        if dc <= rem:
            rem -= dc
            val += seg_dv[sid]
        else:
            if rem > 0.0:
                return val + (rem / dc) * seg_dv[sid], g
            return val, -1
    return val, -1


@njit(cache=CACHE)
def _tol(level, tol_rel):
    if not np.isfinite(level):
        return 0.0
    return tol_rel * max(1.0, abs(level))


@njit(cache=CACHE)
def fixed_theta_bnb(off, v, s, t, gamma, theta, inc, tol_rel, deadline, sel_out):
    """Exact search for one threshold (Theorem 20) under a wall-clock deadline.

    `inc` is the best robust-feasible revenue known so far; nodes whose bound
    does not exceed inc + tol_rel*|inc| are pruned.  Returns
    (status, best value found (> inc) or -inf, record upper bound, nodes),
    where status is 0 = completed, 1 = deadline, 2 = infeasible threshold.
    The record is a valid upper bound on Z^IP(theta) when status is 0 or 1
    (for status 0, the record is max(found, pruning level)).
    """
    n = off.size - 1
    M = v.size
    sth = np.empty(M)
    sbar = np.empty(n)
    c = np.empty(M)
    cap = transformed(off, s, t, gamma, theta, theta, sth, sbar, c)
    if cap < 0.0:
        return 2, -np.inf, -np.inf, 0
    # integer-safe dominance (Definition 17): undominated options per group,
    # sorted by cost ascending with strictly increasing value
    work = np.empty(M, np.int64)
    uoff = np.empty(n + 1, np.int64)
    uidx = np.empty(M, np.int64)
    uc = np.empty(M)
    uv = np.empty(M)
    uoff[0] = 0
    p = 0
    mmax = 0
    for i in range(n):
        lo = off[i]
        m = off[i + 1] - lo
        for a in range(m):
            work[lo + a] = lo + a
        for a in range(1, m):
            key = work[lo + a]
            b = a - 1
            while b >= 0 and _before(c[key], v[key], key, c[work[lo + b]], v[work[lo + b]], work[lo + b]):
                work[lo + b + 1] = work[lo + b]
                b -= 1
            work[lo + b + 1] = key
        start = p
        for a in range(m):
            k = work[lo + a]
            if p > start and v[k] <= uv[p - 1]:
                continue
            uidx[p] = k
            uc[p] = c[k]
            uv[p] = v[k]
            p += 1
        uoff[i + 1] = p
        if p - start > mmax:
            mmax = p - start
    # canonical hulls of the undominated sets and their segments
    hidx = np.empty(M, np.int64)
    hc = np.empty(M)
    hv = np.empty(M)
    hlen = np.empty(n, np.int64)
    hv0 = np.empty(n)
    for i in range(n):
        hlen[i] = build_group_hull(i, off, c, v, work, hidx, hc, hv)
        hv0[i] = hv[off[i]]
    seg_group = np.empty(M, np.int64)
    seg_dc = np.empty(M)
    seg_dv = np.empty(M)
    seg_slope = np.empty(M)
    seg_list = np.empty(M, np.int64)
    nseg = 0
    for i in range(n):
        for k in range(hlen[i] - 1):
            sid = off[i] + k
            seg_group[sid] = i
            seg_dc[sid] = hc[off[i] + k + 1] - hc[off[i] + k]
            seg_dv[sid] = hv[off[i] + k + 1] - hv[off[i] + k]
            seg_slope[sid] = seg_dv[sid] / seg_dc[sid]
            seg_list[nseg] = sid
            nseg += 1
    order = np.argsort(-seg_slope[seg_list[:nseg]], kind="mergesort")
    sorted_seg = seg_list[:nseg][order]
    total_hv0 = 0.0
    for i in range(n):
        total_hv0 += hv0[i]
    fixed = -np.ones(n, np.int64)
    best = -np.inf
    level = inc
    # frames
    fg = np.empty(n + 1, np.int64)
    fstart = np.empty(n + 1, np.int64)
    flen = np.empty(n + 1, np.int64)
    fptr = np.empty(n + 1, np.int64)
    fbound = np.empty(n + 1)
    fcost = np.empty(n + 1)
    fval = np.empty(n + 1)  # value of fixed groups at the frame's node
    ffree = np.empty(n + 1)  # sum of hv0 over free groups at the frame's node
    chbuf = np.empty((n + 1) * mmax, np.int64)
    nodes = 0
    root, g = _node_bound(n, uoff, uv, uc, hv0, sorted_seg, seg_group, seg_dc, seg_dv, nseg,
                          fixed, cap, total_hv0)
    nodes += 1
    tol = _tol(level, tol_rel)
    if root <= level + tol:
        return 0, -np.inf, root, nodes
    if g < 0:
        # LP integral: hull vertices reached by the greedy fill
        pos = np.zeros(n, np.int64)
        rem = cap
        for q in range(nseg):
            sid = sorted_seg[q]
            if seg_dc[sid] <= rem:
                rem -= seg_dc[sid]
                pos[seg_group[sid]] += 1
            else:
                break
        for i in range(n):
            sel_out[i] = hidx[off[i] + pos[i]]
        return 0, root, root, nodes
    # push root frame
    fg[0] = g
    fcost[0] = 0.0
    fval[0] = 0.0
    ffree[0] = total_hv0
    fbound[0] = root
    k0 = 0
    # children ordered by value descending (undominated: cost descending too)
    for a in range(uoff[g + 1] - 1, uoff[g] - 1, -1):
        chbuf[k0] = a
        k0 += 1
    fstart[0] = 0
    flen[0] = k0
    fptr[0] = 0
    depth = 1
    status = 0
    while depth > 0:
        if (nodes & 255) == 0:
            if _now() > deadline:
                status = 1
                break
        top = depth - 1
        gg = fg[top]
        if fptr[top] >= flen[top]:
            fixed[gg] = -1
            depth -= 1
            continue
        a = chbuf[fstart[top] + fptr[top]]
        fptr[top] += 1
        cost_now = fcost[top] + uc[a]
        if cost_now > cap:
            continue
        fixed[gg] = a
        val_fixed = fval[top] + uv[a]
        free_now = ffree[top] - hv0[gg]
        bnd, g2 = _node_bound(n, uoff, uv, uc, hv0, sorted_seg, seg_group, seg_dc, seg_dv, nseg,
                              fixed, cap - cost_now, val_fixed + free_now)
        nodes += 1
        tol = _tol(level, tol_rel)
        if bnd <= level + tol:
            continue
        if g2 < 0:
            # integral node: fixed options plus greedy hull positions
            pos = np.zeros(n, np.int64)
            rem = cap - cost_now
            for q in range(nseg):
                sid = sorted_seg[q]
                gr = seg_group[sid]
                if fixed[gr] >= 0:
                    continue
                if seg_dc[sid] <= rem:
                    rem -= seg_dc[sid]
                    pos[gr] += 1
                else:
                    break
            for i in range(n):
                if fixed[i] >= 0:
                    sel_out[i] = uidx[fixed[i]]
                else:
                    sel_out[i] = hidx[off[i] + pos[i]]
            best = bnd
            level = bnd
            continue
        # push frame for g2
        d = depth
        fg[d] = g2
        fcost[d] = cost_now
        fval[d] = val_fixed
        ffree[d] = free_now
        fbound[d] = bnd
        st = d * mmax
        k0 = 0
        for a2 in range(uoff[g2 + 1] - 1, uoff[g2] - 1, -1):
            chbuf[st + k0] = a2
            k0 += 1
        fstart[d] = st
        flen[d] = k0
        fptr[d] = 0
        depth += 1
    if status == 1:
        rec = level
        for d in range(depth):
            if fptr[d] < flen[d] and fbound[d] > rec:
                rec = fbound[d]
        return 1, best, rec, nodes
    return 0, best, level, nodes


@njit(cache=CACHE)
def brute_force(off, v, s, t, gamma, sel_out):
    """Exhaustive robust optimum (validation only)."""
    n = off.size - 1
    cur = np.zeros(n, np.int64)
    best = -np.inf
    sel = np.empty(n, np.int64)
    while True:
        for i in range(n):
            sel[i] = off[i] + cur[i]
        if robust_residual(off, s, t, gamma, sel) >= 0.0:
            r = revenue(v, sel)
            if r > best:
                best = r
                sel_out[:] = sel
        i = 0
        while i < n:
            cur[i] += 1
            if cur[i] < off[i + 1] - off[i]:
                break
            cur[i] = 0
            i += 1
        if i == n:
            break
    return best
