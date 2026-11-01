"""Summarize the campaign: complete CSV, LaTeX tables, macros and figure.

Every planned task appears in all_runs.csv; a missing result file is
reported as MISSING and counted as a failure.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from study import protocol as P

METHOD_LABEL = {"HR": "HullRound (Alg.~1)", "HR-sweep": "HullRound, guarded sweep",
                "HR-dom": "HullRound, sweep + dominance", "HR+BB": "HullRound + exact layer",
                "DNC": "Threshold DnC", "MILP": "Compact MILP (HiGHS)"}
FAMILY_LABEL = {"regular": "Regular grid", "charm": "Irregular endings", "noisy": "Noisy demand"}
GAMMA_LABEL = {"0": "$0$", "sqrt": r"$\lfloor\sqrt n\rfloor$", "0.1n": r"$\lfloor 0.1n\rfloor$",
               "0.5n": r"$\lfloor 0.5n\rfloor$", "n": "$n$"}


def load(ev: Path):
    rows = []
    for t in P.tasks():
        f = ev / "raw" / f"{t['task_id']}.json"
        row = dict(t)
        if not f.exists():
            row.update(status="MISSING", failure=True)
            rows.append(row)
            continue
        rec = json.loads(f.read_text())
        if not rec.get("ok"):
            row.update(status="KILLED" if rec.get("killed") else "ERROR", failure=True, error=rec.get("error"))
            rows.append(row)
            continue
        r = rec["result"]
        inst = rec["instance"]
        ex = r.get("exact") or {}
        row.update(status=r["status"], objective=r.get("objective"), bound=r.get("bound"),
                   elapsed=r.get("elapsed"), cpu=r.get("cpu_time"), nodes=r.get("nodes"),
                   exact_feasible=ex.get("exact_feasible"), exact_revenue=ex.get("exact_revenue"),
                   M=inst["M"], thresholds=inst["thresholds"], menu_ratio=inst["menu_ratio"],
                   delta_v_max=inst["delta_v_max"], v_lower=inst["v_lower"],
                   load_before=rec["load_before"][0], encode_time=rec.get("encode_time"))
        for k in ("H_rd", "hull_vertices_mean_fraction", "hull_builds", "hull_build_bound", "thetas_feasible",
                  "options_after_preprocessing", "thetas_evaluated", "rd_bound_check", "rejected_candidates",
                  "dual_check_U", "loss_at_thetaU", "jump_at_thetaU", "thetas_searched", "interval_evaluations",
                  "integral", "mip_gap", "hr_elapsed", "leaves_solved"):
            if k in r:
                row[k] = r[k]
        if r.get("exact_rd"):
            row["exact_rd_feasible"] = r["exact_rd"].get("exact_feasible")
        # an incumbent that fails the exact check is a failure, not a solution
        row["failure"] = bool(row["objective"] is not None and row["exact_feasible"] is False)
        rows.append(row)
    # best known exactly verified revenue per run configuration
    key = lambda r: (r["base_id"], r["gamma_label"], r["radius"])
    best = {}
    for r in rows:
        if r.get("exact_feasible") and r.get("exact_revenue") is not None:
            best[key(r)] = max(best.get(key(r), -math.inf), r["exact_revenue"])
    proven = defaultdict(lambda: False)
    for r in rows:
        if r["status"] in ("OPTIMAL",) and r.get("exact_feasible"):
            proven[key(r)] = True
        if r["status"] == "INFEASIBLE":
            proven[key(r)] = True
    for r in rows:
        b = best.get(key(r))
        r["best_known"] = b
        r["best_proven"] = proven[key(r)]
        ok = r.get("exact_feasible") and b is not None
        r["deficit"] = (b - r["exact_revenue"]) / b if ok else None
        r["ub_excess"] = (r["bound"] - b) / b if (r.get("bound") is not None and b) else None
        if r.get("objective") is not None and r.get("bound") is not None and r["bound"] > 0:
            r["gap"] = max(0.0, (r["bound"] - r["objective"]) / r["bound"])
        else:
            r["gap"] = None
        if r.get("H_rd") is not None and r.get("bound"):
            r["gap_rd"] = max(0.0, (r["bound"] - r["H_rd"]) / r["bound"])
    return rows


def q(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    a = np.asarray(vals, float)
    return dict(n=len(a), med=float(np.median(a)), q1=float(np.percentile(a, 25)),
                q3=float(np.percentile(a, 75)), max=float(a.max()), min=float(a.min()))


def fmt_pct(x, d=4):
    if x is None:
        return "--"
    return f"{100 * x:.{d}f}"


def fmt_e(x):
    """Compact scientific notation for small relative quantities, in percent."""
    if x is None:
        return "--"
    x = 100 * x
    if x == 0:
        return "0"
    if x >= 0.01:
        return f"{x:.3f}"
    e = int(math.floor(math.log10(x)))
    return f"{x / 10 ** e:.1f}e{e}"


def fmt_t(x):
    if x is None:
        return "--"
    if x < 0.1:
        return f"{x:.3f}"
    if x < 10:
        return f"{x:.2f}"
    return f"{x:.0f}"


def triple(s, f):
    if s is None:
        return "-- & -- & --"
    return f"{f(s['med'])} & {f(s['q3'] - s['q1'])} & {f(s['max'])}"


def status_counts(rs):
    c = Counter()
    for r in rs:
        if r.get("failure"):
            c["fail"] += 1
        elif r["status"] in ("OPTIMAL", "CERTIFIED"):
            c["done"] += 1
        elif r["status"] == "INFEASIBLE":
            c["inf"] += 1
        elif r["status"] == "TIME_LIMIT":
            c["tl"] += 1
        elif r["status"] == "TIME_LIMIT_NO_INCUMBENT":
            c["tln"] += 1
        else:
            c["fail"] += 1
    return c


def method_table(rows, caption, label, methods):
    out = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{tabular}{lrrrrrr rrr rrr rrr}", r"\toprule",
           r" & & \multicolumn{5}{c}{Termination} & \multicolumn{3}{c}{Gap $(U-L)/U$ (\%)} & "
           r"\multicolumn{3}{c}{Deficit to best (\%)} & \multicolumn{3}{c}{Time (s)}\\",
           r"\cmidrule(lr){3-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-16}",
           r"Method & Runs & Done & TL & TL$^-$ & Inf & Fail & Med & IQR & Max & Med & IQR & Max & Med & IQR & Max\\",
           r"\midrule"]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        c = status_counts(rs)
        out.append(f"{METHOD_LABEL[m]} & {len(rs)} & {c['done']} & {c['tl']} & {c['tln']} & {c['inf']} & {c['fail']} & "
                   f"{triple(q([r['gap'] for r in rs]), fmt_e)} & {triple(q([r['deficit'] for r in rs]), fmt_e)} & "
                   f"{triple(q([r['elapsed'] for r in rs if r.get('elapsed') is not None]), fmt_t)}\\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main(ev_dir, gen_dir, fig_dir):
    ev = Path(ev_dir)
    gen = Path(gen_dir)
    fig = Path(fig_dir)
    rows = load(ev)
    # complete CSV
    cols = sorted({k for r in rows for k in r})
    with open(ev / "all_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    S = [r for r in rows if r["panel"] == "S"]
    L = [r for r in rows if r["panel"] == "L"]
    macros = {}

    def mac(name, val):
        macros[name] = val

    tex = []
    # ---- Table: method comparison on panel S
    tex.append(method_table(S, r"Panel S ($n=200$, $m=16$; three menu families, five budgets, four radii, four "
                               r"portfolios each; 240 runs per method; limit 60\,s). Done: certified interval "
                               r"(HullRound) or optimal within relative tolerance $10^{-6}$; TL: time limit with "
                               r"incumbent; TL$^-$: without incumbent; Inf: infeasible; Fail: error, kill or an "
                               r"incumbent that fails the exact check. Gap uses each method's own bound; deficit "
                               r"is relative to the best exactly verified revenue of the run.",
                            "tab:panelS-methods", P.METHODS_ALL))
    # ---- Table: sensitivity grid (HullRound gap and exact-layer time)
    out = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
           r"\caption{Sensitivity on Panel S. Each cell: median / maximum over 12 runs (three families, four "
           r"portfolios) of HullRound's certified gap $(U-H)/U$ in units of $10^{-3}\,\%$, and in brackets the "
           r"maximum time of HullRound + exact layer in seconds. $\Gamma=0$ does not depend on $\varrho$.}",
           r"\label{tab:panelS-sensitivity}", r"\begin{tabular}{l" + "c" * len(P.RADII) + "}", r"\toprule",
           r"$\Gamma$ & " + " & ".join(rf"$\varrho={x:.2f}$" for x in P.RADII) + r"\\", r"\midrule"]
    for glab, _ in P.gamma_grid(200):
        cells = []
        for rad in P.RADII:
            rs = [r for r in S if r["gamma_label"] == glab and r["radius"] == rad]
            g = q([r["gap"] for r in rs if r["method"] == "HR"])
            tb = q([r["elapsed"] for r in rs if r["method"] == "HR+BB" and r.get("elapsed") is not None])
            inf = sum(1 for r in rs if r["method"] == "HR" and r["status"] == "INFEASIBLE")
            if g is None:
                cells.append("infeasible" if inf else "--")
            else:
                cell = f"{1e5 * g['med']:.2f} / {1e5 * g['max']:.2f} [{fmt_t(tb['max']) if tb else '--'}]"
                if inf:
                    cell += f" ({inf} inf.)"
                cells.append(cell)
        out.append(GAMMA_LABEL[glab] + " & " + " & ".join(cells) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex.append("\n".join(out))
    # ---- Table: ablations on both panels
    out = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{Component ablations of HullRound (completed runs only; medians with maxima in brackets). "
           r"Enumeration: complete rebuild at every threshold vs.\ the guarded sweep of "
           r"\Cref{prop:sweep_exact}. Dominance: threshold-uniform deletion of \Cref{prop:universal-dominance} "
           r"before forming $\Bset$. Rounding: certified gap with round-down only vs.\ with repair and "
           r"completion.}", r"\label{tab:ablation}",
           r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Panel & Comparison & Pairs & Quantity A & Quantity B & Ratio B/A\\", r"\midrule"]
    for pname, PR in (("S", S), ("L", L)):
        by = defaultdict(dict)
        for r in PR:
            by[(r["base_id"], r["gamma_label"], r["radius"])][r["method"]] = r
        # enumeration
        pairs = [(d["HR"], d["HR-sweep"]) for d in by.values() if "HR" in d and "HR-sweep" in d
                 and d["HR"]["status"] == "CERTIFIED" and d["HR-sweep"]["status"] == "CERTIFIED"]
        if pairs:
            ta = q([a["elapsed"] for a, b in pairs]); tb = q([b["elapsed"] for a, b in pairs])
            hb = q([b["hull_builds"] / a["hull_builds"] for a, b in pairs])
            tr = q([b["elapsed"] / a["elapsed"] for a, b in pairs])
            diff = max(max(abs(a["objective"] - b["objective"]) / a["bound"], abs(a["bound"] - b["bound"]) / a["bound"])
                       for a, b in pairs)
            out.append(f"{pname} & full vs.\\ sweep: time (s) & {len(pairs)} & {fmt_t(ta['med'])} [{fmt_t(ta['max'])}] & "
                       f"{fmt_t(tb['med'])} [{fmt_t(tb['max'])}] & {tr['med']:.2f} [{tr['max']:.2f}]\\\\")
            out.append(f"{pname} & full vs.\\ sweep: hull builds & {len(pairs)} & -- & -- & {hb['med']:.2f} [{hb['max']:.2f}]\\\\")
            mac(f"Abl{pname}SweepPairs", str(len(pairs)))
            mac(f"Abl{pname}SweepMaxDiff", f"{diff:.1e}")
            mac(f"Abl{pname}SweepTimeRatio", f"{tr['med']:.2f}")
            mac(f"Abl{pname}SweepBuildRatio", f"{hb['med']:.2f}")
            sw = [b for a, b in pairs]
            mac(f"Abl{pname}SweepBoundOK", "all" if all(b["hull_builds"] <= b["hull_build_bound"] for b in sw) else "not all")
        # dominance
        pairs = [(d["HR-sweep"], d["HR-dom"]) for d in by.values() if "HR-sweep" in d and "HR-dom" in d
                 and d["HR-sweep"]["status"] == "CERTIFIED" and d["HR-dom"]["status"] == "CERTIFIED"]
        if pairs:
            orr = q([b["options_after_preprocessing"] / a["M"] for a, b in pairs])
            thr = q([b["thetas_evaluated"] / a["thetas_evaluated"] for a, b in pairs])
            tr = q([b["elapsed"] / a["elapsed"] for a, b in pairs])
            lower = sum(1 for a, b in pairs if b["objective"] < a["objective"] * (1 - 1e-12))
            higher = sum(1 for a, b in pairs if b["objective"] > a["objective"] * (1 + 1e-12))
            ubl = sum(1 for a, b in pairs if b["bound"] < a["bound"] * (1 - 1e-12))
            out.append(f"{pname} & dominance off vs.\\ on: options kept & {len(pairs)} & 1 & {orr['med']:.2f} [{orr['max']:.2f}] & --\\\\")
            out.append(f"{pname} & dominance off vs.\\ on: thresholds & {len(pairs)} & 1 & {thr['med']:.2f} [{thr['max']:.2f}] & --\\\\")
            out.append(f"{pname} & dominance off vs.\\ on: time & {len(pairs)} & -- & -- & {tr['med']:.2f} [{tr['max']:.2f}]\\\\")
            mac(f"Abl{pname}DomPairs", str(len(pairs)))
            mac(f"Abl{pname}DomOptions", f"{100 * orr['med']:.0f}\\%")
            mac(f"Abl{pname}DomThresholds", f"{100 * thr['med']:.0f}\\%")
            mac(f"Abl{pname}DomTimeRatio", f"{tr['med']:.2f}")
            mac(f"Abl{pname}DomLowerH", str(lower))
            mac(f"Abl{pname}DomHigherH", str(higher))
            mac(f"Abl{pname}DomLowerU", str(ubl))
        # rounding
        rs = [d["HR"] for d in by.values() if "HR" in d and d["HR"]["status"] == "CERTIFIED"]
        if rs:
            ga = q([r["gap_rd"] for r in rs]); gb = q([r["gap"] for r in rs])
            imp = sum(1 for r in rs if r["objective"] > r["H_rd"] * (1 + 1e-12))
            out.append(f"{pname} & round-down vs.\\ repair: gap (\\%) & {len(rs)} & {fmt_e(ga['med'])} [{fmt_e(ga['max'])}] & "
                       f"{fmt_e(gb['med'])} [{fmt_e(gb['max'])}] & --\\\\")
            mac(f"Abl{pname}RoundRuns", str(len(rs)))
            mac(f"Abl{pname}RoundImproved", str(imp))
            mac(f"Abl{pname}RoundGapRdMed", fmt_e(ga["med"]))
            mac(f"Abl{pname}RoundGapMed", fmt_e(gb["med"]))
            mac(f"Abl{pname}RoundGapRdMax", fmt_e(ga["max"]))
            mac(f"Abl{pname}RoundGapMax", fmt_e(gb["max"]))
            mac(f"Abl{pname}RdBoundOK", "all" if all(r.get("rd_bound_check") for r in rs) else "not all")
        out.append(r"\addlinespace")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex.append("\n".join(out))
    # ---- Table: scale panel
    out = [r"\begin{table}[p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{Panel L (scalability; $\Gamma=\lfloor\sqrt n\rfloor$, $\varrho=0.1$, regular and "
           r"irregular-ending menus, three portfolios each, so six runs per row; limit 600\,s). Done: certified "
           r"(HullRound) or optimal within $10^{-6}$; TL: time-limited; Fail as in \Cref{tab:panelS-methods}. "
           r"Gap $(U-L)/U$ uses each method's own bound.}",
           r"\label{tab:panelL}", r"\begin{tabular}{rrrlrrrrrrr}", r"\toprule",
           r"$n$ & $m$ & $|\Bset|$ & Method & Done & TL & Fail & \multicolumn{2}{c}{Time (s)} & "
           r"\multicolumn{2}{c}{Gap (\%)}\\", r"\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
           r" & & (med) & & & & & Med & Max & Med & Max\\", r"\midrule"]
    lmeth = ("HR", "HR-sweep", "HR+BB", "DNC", "MILP")
    for n, m in P.PANELS["L"]["sizes"]:
        rs = [r for r in L if r["n"] == n and r["m"] == m]
        th = q([r["thresholds"] for r in rs if r.get("thresholds")])
        for k, meth in enumerate(lmeth):
            mr = [r for r in rs if r["method"] == meth]
            c = status_counts(mr)
            t = q([r["elapsed"] for r in mr if r.get("elapsed") is not None])
            g = q([r["gap"] for r in mr])
            lead = f"{n} & {m} & {int(th['med']) if th else '--'}" if k == 0 else " & & "
            out.append(f"{lead} & {METHOD_LABEL[meth]} & {c['done']} & {c['tl'] + c['tln']} & {c['fail']} & "
                       f"{fmt_t(t['med']) if t else '--'} & {fmt_t(t['max']) if t else '--'} & "
                       f"{fmt_e(g['med']) if g else '--'} & {fmt_e(g['max']) if g else '--'}\\\\")
        out.append(r"\addlinespace")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex.append("\n".join(out))
    tex.append(method_table(L, r"Panel L aggregated over all 54 runs per method (limit 600\,s); columns as in "
                               r"\Cref{tab:panelS-methods}.", "tab:panelL-methods", P.METHODS_ALL))
    # ---- Table: menus by family (hull fraction and menu statistic)
    out = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
           r"\caption{Menu structure by generator family (all completed HullRound runs). Hull fraction: "
           r"canonical hull vertices divided by admissible options, averaged over feasible thresholds. "
           r"$\Delta^v_{\max}/\underline V$: the instance-level bound of \Cref{cor:relative}(a), reported for "
           r"every instance in the evidence CSV.}", r"\label{tab:families}",
           r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Panel & Family & Instances & \multicolumn{3}{c}{Hull fraction (med / IQR / min)} & "
           r"\multicolumn{2}{c}{$\Delta^v_{\max}/\underline V$ (med / max)}\\", r"\midrule"]
    for pname, PR in (("S", S), ("L", L)):
        for fam in P.FAMILIES:
            rs = [r for r in PR if r["family"] == fam and r["method"] == "HR" and r["status"] == "CERTIFIED"]
            if not rs:
                continue
            hf = q([r["hull_vertices_mean_fraction"] for r in rs])
            mr = q(list({r["base_id"]: r["menu_ratio"] for r in PR if r["family"] == fam and r.get("menu_ratio")}.values()))
            ninst = len({r["base_id"] for r in PR if r["family"] == fam})
            out.append(f"{pname} & {FAMILY_LABEL[fam]} & {ninst} & {hf['med']:.3f} & {hf['q3'] - hf['q1']:.3f} & "
                       f"{hf['min']:.3f} & {mr['med']:.2e} & {mr['max']:.2e}\\\\")
            mac(f"Hull{pname}{fam.capitalize()}", f"{hf['med']:.2f}")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex.append("\n".join(out))
    # ---- global macros
    allr = rows
    mac("StudyTasks", str(len(allr)))
    mac("StudyMissing", str(sum(r["status"] == "MISSING" for r in allr)))
    mac("StudyFailures", str(sum(bool(r.get("failure")) for r in allr)))
    mac("StudyExactChecked", str(sum(r.get("exact_feasible") is not None for r in allr)))
    mac("StudyExactFail", str(sum(r.get("exact_feasible") is False for r in allr)))
    mac("StudyInstancesS", str(len({r["base_id"] for r in S})))
    mac("StudyInstancesL", str(len({r["base_id"] for r in L})))
    mac("StudyRunsS", str(len({(r["base_id"], r["gamma_label"], r["radius"]) for r in S})))
    mac("StudyRunsL", str(len({(r["base_id"], r["gamma_label"], r["radius"]) for r in L})))
    for pname, PR in (("S", S), ("L", L)):
        for meth in P.METHODS_ALL:
            rs = [r for r in PR if r["method"] == meth]
            c = status_counts(rs)
            tag = meth.replace("+", "P").replace("-", "")
            mac(f"{pname}{tag}Done", str(c["done"]))
            mac(f"{pname}{tag}TL", str(c["tl"] + c["tln"]))
            mac(f"{pname}{tag}TLNoInc", str(c["tln"]))
            mac(f"{pname}{tag}Inf", str(c["inf"]))
            mac(f"{pname}{tag}Fail", str(c["fail"]))
            t = q([r["elapsed"] for r in rs if r.get("elapsed") is not None])
            if t:
                mac(f"{pname}{tag}TimeMed", fmt_t(t["med"]))
                mac(f"{pname}{tag}TimeMax", fmt_t(t["max"]))
            g = q([r["gap"] for r in rs])
            if g:
                mac(f"{pname}{tag}GapMed", fmt_e(g["med"]))
                mac(f"{pname}{tag}GapMax", fmt_e(g["max"]))
            d = q([r["deficit"] for r in rs])
            if d:
                mac(f"{pname}{tag}DefMed", fmt_e(d["med"]))
                mac(f"{pname}{tag}DefMax", fmt_e(d["max"]))
        # Corollary 15(a) check: (U-H)/U <= min(1, Delta/V)
        hr = [r for r in PR if r["method"] == "HR" and r["status"] == "CERTIFIED"]
        if hr:
            ok = sum(1 for r in hr if r["gap"] <= min(1.0, r["menu_ratio"]) * (1 + 1e-9))
            mac(f"{pname}CorCheck", f"{ok}/{len(hr)}")
            mac(f"{pname}CorSlackMed", f"{np.median([r['gap'] / r['menu_ratio'] for r in hr]):.3f}")
            mac(f"{pname}CorSlackMax", f"{max(r['gap'] / r['menu_ratio'] for r in hr):.3f}")
            mac(f"{pname}MenuRatioMin", f"{min(r['menu_ratio'] for r in hr):.1e}")
            mac(f"{pname}MenuRatioMax", f"{max(r['menu_ratio'] for r in hr):.1e}")
            dual = [abs(r["dual_check_U"] - r["bound"]) / r["bound"] for r in hr if r.get("dual_check_U") is not None]
            mac(f"{pname}DualCheckMax", f"{max(dual):.1e}" if dual else "--")
    # HiGHS solutions failing the exact check, etc.
    milp_fail = [r for r in rows if r["method"] == "MILP" and r.get("exact_feasible") is False]
    mac("MILPExactFail", str(len(milp_fail)))
    # validation
    vf = ev / "validation.json"
    if vf.exists():
        v = json.loads(vf.read_text())["summary"]
        mac("ValCases", str(v["cases"]))
        mac("ValPassed", str(v["passed"]))
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "study_tables.tex").write_text("% Generated by study/summarize.py; do not edit.\n" + "\n\n".join(tex) + "\n")
    (gen / "study_macros.tex").write_text("% Generated by study/summarize.py; do not edit.\n" +
                                          "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    (ev / "summary_macros.json").write_text(json.dumps(macros, indent=1, sort_keys=True))
    figure(rows, fig)
    print(json.dumps(macros, indent=1, sort_keys=True))


def figure(rows, fig: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    f, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    colors = {"regular": "#1f77b4", "charm": "#d62728", "noisy": "#2ca02c"}
    hr = [r for r in rows if r["method"] == "HR" and r["status"] == "CERTIFIED"]
    for fam in P.FAMILIES:
        rs = [r for r in hr if r["family"] == fam]
        if rs:
            ax[0].scatter([r["menu_ratio"] for r in rs], [max(r["gap"], 1e-9) for r in rs], s=6, alpha=0.6,
                          color=colors[fam], label={"regular": "regular", "charm": "irregular endings",
                                                    "noisy": "noisy"}[fam], edgecolors="none")
    lo, hi = 1e-4, 3e-2
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"menu statistic $\Delta^v_{\max}/\underline{V}$")
    ax[0].set_ylabel(r"certified gap $(U-H)/U$")
    ax[0].set_title("(a) Cor. 15(a) bound vs. realized", fontsize=8)
    ax[0].legend(fontsize=6, frameon=False, loc="lower right")
    # time vs M for scale panel
    L = [r for r in rows if r["panel"] == "L"]
    marks = {"HR-sweep": ("o", "HullRound (sweep)"), "HR": ("s", "HullRound (full)"),
             "HR+BB": ("^", "+ exact layer"), "DNC": ("v", "threshold DnC"), "MILP": ("x", "MILP (HiGHS)")}
    for meth, (mk, lab) in marks.items():
        rs = [r for r in L if r["method"] == meth and r.get("elapsed") is not None]
        if rs:
            ax[1].scatter([r["M"] for r in rs], [r["elapsed"] for r in rs], s=10, marker=mk, label=lab, alpha=0.7)
    ax[1].axhline(600, color="gray", lw=0.6, ls=":")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"admissible options $M$"); ax[1].set_ylabel("wall-clock time (s)")
    ax[1].set_title("(b) Panel L running time", fontsize=8)
    ax[1].legend(fontsize=5.5, frameon=False, loc="upper left")
    # gap by Gamma and radius (panel S), HR
    S = [r for r in hr if r["panel"] == "S"]
    for k, rad in enumerate(P.RADII):
        meds = []
        for glab, _ in P.gamma_grid(200):
            g = [r["gap"] for r in S if r["radius"] == rad and r["gamma_label"] == glab]
            meds.append(np.median(g) if g else np.nan)
        ax[2].plot(range(5), meds, marker="o", ms=3, label=rf"$\varrho={rad:.2f}$")
    ax[2].set_xticks(range(5)); ax[2].set_xticklabels(["0", r"$\sqrt{n}$", "0.1n", "0.5n", "n"])
    ax[2].set_xlabel(r"budget $\Gamma$"); ax[2].set_ylabel("median certified gap")
    ax[2].set_title("(c) Panel S sensitivity", fontsize=8)
    ax[2].legend(fontsize=6, frameon=False)
    f.tight_layout()
    f.savefig(fig / "study_evidence.pdf")


if __name__ == "__main__":
    main(*sys.argv[1:4])
