"""Protocol v2 reporting (written after the v2 freeze; changes no run).

Implements the planned reporting of PROTOCOL_V2.md: Panel H method table,
pipeline and SCIP comparisons on Panels S, L, H, the price-of-robustness
table, manuscript macros, and the v2 manifest.

Usage: python -m study.report_v2 V2_EVIDENCE_DIR GENERATED_DIR FIGURES_DIR
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from study import protocol as P1
from study import protocol_v2 as P2
from study import summarize as S

TOL = 1e-6
LABEL = dict(S.METHOD_LABEL, **{"PIPE": "HullRound, interval records", "MILP-SCIP": "Compact MILP (SCIP)"})


def _row(task, rec):
    row = dict(task)
    if not rec.get("ok"):
        row.update(status="KILLED" if rec.get("killed") else "ERROR", failure=True)
        return row
    r = rec["result"]
    ex = r.get("exact") or {}
    row.update(status=r["status"], objective=r.get("objective"), bound=r.get("bound"), elapsed=r.get("elapsed"),
               exact_feasible=ex.get("exact_feasible"), exact_revenue=ex.get("exact_revenue"),
               M=rec["instance"]["M"], thresholds=rec["instance"]["thresholds"],
               menu_ratio=rec["instance"]["menu_ratio"], H_rd=r.get("H_rd"),
               thresholds_evaluated=r.get("thresholds_evaluated"), interval_evaluations=r.get("interval_evaluations"),
               status_relabelled=r.get("status_relabelled", False))
    row["failure"] = bool(ex.get("exact_feasible") is False)
    return row


def load(ev2: Path):
    v1 = Path(P2.V1_EVIDENCE)
    rows = []
    for t in P1.tasks():
        if t["method"] == "HR-dom":
            continue
        rows.append(_row(t, json.loads((v1 / "raw" / f"{t['task_id']}.json").read_text())))
    for t in P2.tasks():
        f = ev2 / "raw" / f"{t['task_id']}.json"
        rows.append(_row(t, json.loads(f.read_text())) if f.exists() else dict(t, status="MISSING", failure=True))
    key = lambda r: (r["base_id"], r["gamma_label"], r["radius"])
    best = {}
    for r in rows:
        if r.get("exact_feasible") and r.get("exact_revenue") is not None:
            best[key(r)] = max(best.get(key(r), -math.inf), r["exact_revenue"])
    for r in rows:
        if r["method"] in ("HR+BB", "DNC") and r["status"] == "OPTIMAL" and r.get("objective") is not None:
            r["bound"] = r["objective"] + TOL * max(1.0, abs(r["objective"]))  # proven bound at tolerance
        b = best.get(key(r))
        r["best_known"] = b
        ok = r.get("exact_feasible") and b
        r["deficit"] = (b - r["exact_revenue"]) / b if ok else None
        lb, ub = r.get("objective"), r.get("bound")
        r["gap"] = max(0.0, (ub - lb) / ub) if (lb is not None and ub is not None and ub > 0) else None
    return rows


def counts(rs):
    c = dict(done=0, tl=0, inf=0, fail=0)
    for r in rs:
        st = r["status"]
        if r.get("failure") or st in ("KILLED", "ERROR", "MISSING") or st.startswith("OTHER"):
            c["fail"] += 1
        elif st in ("CERTIFIED", "OPTIMAL"):
            c["done"] += 1
        elif st.startswith("TIME_LIMIT"):
            c["tl"] += 1
        elif st == "INFEASIBLE":
            c["inf"] += 1
        else:
            c["fail"] += 1
    return c


def q(vals):
    return S.q([v for v in vals if v is not None])


def f_e(x):
    """Percent (input is a fraction), as in the v1 tables."""
    return "--" if x is None else S.fmt_e(x)


def f_r(x):
    """Plain ratio, three significant digits."""
    if x is None:
        return "--"
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    if round(x / 10 ** e, 1) >= 10:  # 9.96e-5 would print as 10.0e-5
        e += 1
    return f"{x:.3f}" if x >= 0.01 else f"{x / 10 ** e:.1f}e{e}"


def f_t(x):
    return "--" if x is None else S.fmt_t(x)


def panel_h_table(rows):
    H = [r for r in rows if r["panel"] == "H"]
    out = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Panel H (dominant-jump, heterogeneous-scale portfolios; $n\in\{20,100,500\}$, "
           r"$m\in\{3,4\}$, four portfolios each, $\Gamma\in\{\lfloor\sqrt n\rfloor,\lfloor0.5n\rfloor\}$, "
           r"$\varrho\in\{0.2,0.3\}$; 96 runs per method; limit 300\,s). Columns as in "
           r"\Cref{tab:panelS-methods} (no run ended at the time limit, so the TL$^-$ column is omitted); gaps and deficits over runs with a feasible decision, in \%; "
           r"tol.\ as in \Cref{tab:panelS-methods}.}",
           r"\label{tab:panelH}",
           r"\begin{tabular}{lrrrrr rrr rr rr}", r"\toprule",
           r" & & \multicolumn{4}{c}{Termination} & \multicolumn{3}{c}{Gap $(U-L)/U$ (\%)} & "
           r"\multicolumn{2}{c}{Deficit (\%)} & \multicolumn{2}{c}{Time (s)}\\",
           r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
           r"Method & Runs & Done & TL & Inf & Fail & Med & IQR & Max & Med & Max & Med & Max\\", r"\midrule"]
    for m in P2.METHODS_H:
        rs = [r for r in H if r["method"] == m]
        c = counts(rs)
        g = q([r["gap"] for r in rs])
        d = q([r["deficit"] for r in rs])
        t = q([r.get("elapsed") for r in rs])
        if m in ("HR+BB", "DNC"):
            gcell = r"\multicolumn{3}{c}{$\le$1e-4 (tol.)}"
        else:
            gcell = (f"{f_e(g['med'])} & {f_e(g['q3'] - g['q1'])} & {f_e(g['max'])}"
                     if g else "-- & -- & --")
        out.append(f"{LABEL[m]} & {len(rs)} & {c['done']} & {c['tl']} & {c['inf']} & {c['fail']} & {gcell} & "
                   + (f"{f_e(d['med'])} & {f_e(d['max'])}" if d else "-- & --") + " & "
                   + (f"{f_t(t['med'])} & {f_t(t['max'])}" if t else "-- & --") + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def _pairs(rows, panel, a, b):
    idx = {}
    for r in rows:
        if r["panel"] == panel and r["method"] in (a, b):
            idx.setdefault((r["base_id"], r["gamma_label"], r["radius"]), {})[r["method"]] = r
    return [(d[a], d[b]) for d in idx.values() if a in d and b in d]


def comparisons(rows):
    """PIPE vs HR-sweep and SCIP vs HiGHS per panel."""
    res = {}
    for pan in ("S", "L", "H"):
        pp = [(h, p) for h, p in _pairs(rows, pan, "HR-sweep", "PIPE")
              if h["status"] == "CERTIFIED" and p["status"] == "CERTIFIED"]
        full = [(h, p) for h, p in _pairs(rows, pan, "HR", "PIPE")
                if h.get("elapsed") and p.get("elapsed") and h["status"] in ("CERTIFIED", "TIME_LIMIT")]
        pipe = [r for r in rows if r["panel"] == pan and r["method"] == "PIPE"]
        res[pan] = dict(
            pipe_runs=len(pipe), pipe_counts=counts(pipe), pairs=len(pp),
            sameH=sum(h["objective"] == p["objective"] for h, p in pp),
            sameU=sum(h["bound"] == p["bound"] for h, p in pp),
            maxrelH=max([abs(h["objective"] - p["objective"]) / h["bound"] for h, p in pp], default=None),
            maxrelU=max([abs(h["bound"] - p["bound"]) / h["bound"] for h, p in pp], default=None),
            ratio_sweep=q([p["elapsed"] / h["elapsed"] for h, p in pp]),
            ratio_full=q([p["elapsed"] / h["elapsed"] for h, p in full]),
            frac_eval=q([p["thresholds_evaluated"] / p["thresholds"] for p in pipe
                         if p.get("thresholds_evaluated") is not None]),
            pipe_time=q([p.get("elapsed") for p in pipe]),
        )
        hi = [r for r in rows if r["panel"] == pan and r["method"] == "MILP"]
        sc = [r for r in rows if r["panel"] == pan and r["method"] == "MILP-SCIP"]
        both = [(a, b) for a, b in _pairs(rows, pan, "MILP", "MILP-SCIP")
                if a["status"] == "OPTIMAL" and b["status"] == "OPTIMAL"]
        res[pan].update(highs_counts=counts(hi), scip_counts=counts(sc),
                        highs_time=q([r.get("elapsed") for r in hi]), scip_time=q([r.get("elapsed") for r in sc]),
                        milp_both=len(both),
                        milp_agree=sum(abs(a["exact_revenue"] - b["exact_revenue"]) <= 2 * TOL * a["exact_revenue"]
                                       for a, b in both),
                        scip_faster=sum(b["elapsed"] < a["elapsed"] for a, b in both))
        dn = [r for r in rows if r["panel"] == pan and r["method"] == "DNC"]
        bb = [r for r in rows if r["panel"] == pan and r["method"] == "HR+BB"]
        res[pan].update(dnc_time=q([r.get("elapsed") for r in dn]), bb_time=q([r.get("elapsed") for r in bb]),
                        hr_time=q([r.get("elapsed") for r in rows if r["panel"] == pan and r["method"] == "HR-sweep"]))
    return res


def comparison_table(cmp):
    out = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{HullRound with interval records (\Cref{prop:interval-record}) against HullRound "
           r"(guarded sweep), and SCIP against HiGHS on the compact MILP (second protocol). Pairs: runs in which "
           r"both HullRound variants certified. Same $H$, $U$: identical values in binary64. Time ratio and "
           r"threshold evaluations per threshold of $\Bset$: median [maximum]. MILP: done = optimal within "
           r"$10^{-6}$ with an exactly feasible decision (SCIP/HiGHS); agree = revenues within "
           r"$2\times10^{-6}$ relative where both report optimality.}",
           r"\label{tab:v2-methods}",
           r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
           r" & \multicolumn{5}{c}{Interval records vs.\ HullRound} & \multicolumn{4}{c}{SCIP vs.\ HiGHS}\\",
           r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}",
           r"Panel & Pairs & Same $H$ & Same $U$ & Time ratio & Evals.\ per thr. & Done & "
           r"Time med (s) & Time max (s) & Agree\\", r"\midrule"]
    for pan in ("S", "L", "H"):
        c = cmp[pan]
        rt, fe = c["ratio_sweep"], c["frac_eval"]
        out.append(
            f"{pan} & {c['pairs']} & {c['sameH']} & {c['sameU']} & "
            + (f"{f_r(rt['med'])} [{f_r(rt['max'])}]" if rt else "--") + " & "
            + (f"{f_r(fe['med'])} [{f_r(fe['max'])}]" if fe else "--") + " & "
            + f"{c['scip_counts']['done']}/{c['highs_counts']['done']} & "
            + f"{f_t((c['scip_time'] or {}).get('med'))}/{f_t((c['highs_time'] or {}).get('med'))} & "
            + f"{f_t((c['scip_time'] or {}).get('max'))}/{f_t((c['highs_time'] or {}).get('max'))} & "
            + f"{c['milp_agree']}/{c['milp_both']}" + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def robustness_table(ev2: Path):
    rob = json.loads((ev2 / "robustness.json").read_text())["rows"]
    out = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
           r"\caption{Price of robustness on Panel S (second protocol). Each cell: median over the 12 base "
           r"portfolios of $Z^{\mathrm{rob}}(\Gamma,\varrho)/Z^{\mathrm{nom}}$, and in brackets the median "
           r"out-of-sample violation rate (\%) of the margin constraint under $10^4$ draws of i.i.d.\ "
           r"relative residuals $h_i\sim U[-\varrho,\varrho]$. Decisions: threshold DnC (optimal within "
           r"$10^{-6}$); $Z^{\mathrm{nom}}$ and the nominal row use $\Gamma=0$.}",
           r"\label{tab:robustness}",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r"$\Gamma$ & " + " & ".join(rf"$\varrho={x:.2f}$" for x in P1.RADII) + r"\\", r"\midrule"]
    for glab, _ in P1.gamma_grid(200):
        cells = []
        for rad in P1.RADII:
            rr = [r for r in rob if r["gamma_label"] == glab and r["radius"] == rad]
            cells.append(f"{np.median([r['ratio'] for r in rr]):.4f} [{100 * np.median([r['violation_rob'] for r in rr]):.1f}]")
        out.append(S.GAMMA_LABEL[glab] + " & " + " & ".join(cells) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    stats = {}
    for glab, _ in P1.gamma_grid(200):
        rr = [r for r in rob if r["gamma_label"] == glab]
        stats[glab] = dict(ratio_min=min(r["ratio"] for r in rr), ratio_med=float(np.median([r["ratio"] for r in rr])),
                           viol_max=max(r["violation_rob"] for r in rr))
    stats["nom_viol_min"] = min(r["violation_nom"] for r in rob)
    stats["nom_viol_max"] = max(r["violation_nom"] for r in rob)
    return "\n".join(out), stats


def macros(rows, cmp, rstats):
    m = {}
    tasks = P2.tasks()
    m["VtwoTasks"] = str(len(tasks))
    fails = [r for r in rows if r.get("instance_source") and (r.get("failure") or r["status"] in ("KILLED", "ERROR", "MISSING"))]
    m["VtwoFailures"] = str(len(fails))
    m["VtwoExactChecked"] = str(sum(1 for r in rows if r.get("instance_source") and r.get("exact_feasible") is not None))
    m["VtwoExactFail"] = str(sum(1 for r in rows if r.get("instance_source") and r.get("exact_feasible") is False))
    H = [r for r in rows if r["panel"] == "H"]
    hr = [r for r in H if r["method"] == "HR-sweep"]
    g = q([r["gap"] for r in hr])
    m["HInfeasible"] = str(counts(hr)["inf"])
    m["HFeasible"] = str(counts(hr)["done"])
    if g:
        m["HHRGapMed"] = S.fmt_e(g["med"])
        m["HHRGapMax"] = S.fmt_e(g["max"])
        m["HHRGapOnePct"] = str(sum(1 for r in hr if r.get("gap") is not None and r["gap"] >= 0.01))
        d = q([r["deficit"] for r in hr])
        m["HHRDefMed"] = S.fmt_e(d["med"])
        m["HHRDefMax"] = S.fmt_e(d["max"])
        for n, tag in ((20, "A"), (100, "B"), (500, "C")):
            gn = q([r["gap"] for r in hr if r["n"] == n])
            m[f"HHRGapMed{tag}"] = S.fmt_e(gn["med"])
            m[f"HHRGapMax{tag}"] = S.fmt_e(gn["max"])
        for meth, tag in (("MILP", "MILP"), ("MILP-SCIP", "SCIP")):
            t5 = q([r.get("elapsed") for r in H if r["method"] == meth and r["n"] == 500])
            m[f"H{tag}TimeMedC"] = S.fmt_t(t5["med"])
    for meth, tag in (("DNC", "DNC"), ("HR+BB", "BB"), ("MILP", "MILP"), ("MILP-SCIP", "SCIP"), ("PIPE", "PIPE"),
                      ("HR-sweep", "HRsweep")):
        t = q([r.get("elapsed") for r in H if r["method"] == meth])
        if t:
            m[f"H{tag}TimeMed"] = S.fmt_t(t["med"])
            m[f"H{tag}TimeMax"] = S.fmt_t(t["max"])
        m[f"H{tag}Done"] = str(counts([r for r in H if r["method"] == meth])["done"])
    for pan in ("S", "L", "H"):
        c = cmp[pan]
        m[f"Pipe{pan}Pairs"] = str(c["pairs"])
        m[f"Pipe{pan}SameH"] = str(c["sameH"])
        m[f"Pipe{pan}SameU"] = str(c["sameU"])
        if c["ratio_sweep"]:
            m[f"Pipe{pan}RatioMed"] = f_r(c["ratio_sweep"]["med"])
            m[f"Pipe{pan}RatioMax"] = f_r(c["ratio_sweep"]["max"])
        if c["frac_eval"]:
            m[f"Pipe{pan}EvalMed"] = f_r(c["frac_eval"]["med"])
            m[f"Pipe{pan}EvalMax"] = f_r(c["frac_eval"]["max"])
        if c["pipe_time"]:
            m[f"Pipe{pan}TimeMed"] = S.fmt_t(c["pipe_time"]["med"])
            m[f"Pipe{pan}TimeMax"] = S.fmt_t(c["pipe_time"]["max"])
        m[f"Pipe{pan}Done"] = str(c["pipe_counts"]["done"])
        m[f"Pipe{pan}Runs"] = str(c["pipe_runs"])
        m[f"Scip{pan}Done"] = str(c["scip_counts"]["done"])
        m[f"Scip{pan}TL"] = str(c["scip_counts"]["tl"])
        m[f"Scip{pan}Fail"] = str(c["scip_counts"]["fail"])
        m[f"Scip{pan}TimeMed"] = f_t(c["scip_time"]["med"]) if c["scip_time"] else "--"
        m[f"Scip{pan}TimeMax"] = f_t(c["scip_time"]["max"]) if c["scip_time"] else "--"
        m[f"Scip{pan}Faster"] = str(c["scip_faster"])
        m[f"Scip{pan}Both"] = str(c["milp_both"])
        m[f"Scip{pan}Agree"] = str(c["milp_agree"])
    for glab, tag in (("sqrt", "Sqrt"), ("n", "Full")):
        m[f"Rob{tag}RatioMin"] = f"{rstats[glab]['ratio_min']:.3f}"
        m[f"Rob{tag}RatioMed"] = f"{rstats[glab]['ratio_med']:.4f}"
        m[f"Rob{tag}ViolMax"] = f"{100 * rstats[glab]['viol_max']:.2f}"
    m["RobNomViolMin"] = f"{100 * rstats['nom_viol_min']:.1f}"
    m["RobNomViolMax"] = f"{100 * rstats['nom_viol_max']:.1f}"
    m["RobPosViolMax"] = f"{100 * max(rstats[g]['viol_max'] for g in ('sqrt', '0.1n', '0.5n', 'n')):.2f}"
    return m


def main(ev_dir, gen_dir, fig_dir):
    ev2, gen = Path(ev_dir), Path(gen_dir)
    rows = load(ev2)
    from study import report as R1
    rows1 = R1.proven_bounds(S.load(Path(P2.V1_EVIDENCE)))
    extra = [dict(r, M=r["M"]) for r in rows if r["panel"] == "L" and r["method"] in ("PIPE", "MILP-SCIP")
             and r.get("elapsed") is not None]
    R1.figure(rows1, Path(fig_dir), extra_L=extra)
    cmp = comparisons(rows)
    rob_tab, rstats = robustness_table(ev2)
    hdr = "% Generated by study/report_v2.py; do not edit.\n"
    (gen / "study_table_panelH.tex").write_text(hdr + panel_h_table(rows) + "\n")
    (gen / "study_table_v2methods.tex").write_text(hdr + comparison_table(cmp) + "\n")
    (gen / "study_table_robustness.tex").write_text(hdr + rob_tab + "\n")
    m = macros(rows, cmp, rstats)
    text = hdr + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items()))
    import re
    text = re.sub(r"(?<=\{)(\d+(?:\.\d+)?)e([+-]?\d+)(?=\})",
                  lambda mo: "0" if float(mo.group(1)) == 0 else f"\\ensuremath{{{mo.group(1)}\\times10^{{{int(mo.group(2))}}}}}",
                  text)
    (gen / "study_v2_macros.tex").write_text(text)
    keys = ["task_id", "panel", "family", "n", "m", "rep", "gamma_label", "gamma", "radius", "method", "status",
            "objective", "bound", "gap", "deficit", "best_known", "exact_feasible", "exact_revenue", "elapsed",
            "thresholds", "thresholds_evaluated", "interval_evaluations", "status_relabelled", "instance_source"]
    with open(ev2 / "all_runs_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (ev2 / "summary_v2.json").write_text(json.dumps(dict(macros=m, comparisons=cmp), indent=1, default=str))
    rep = ev2 / "reporting"
    rep.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(Path(__file__), rep / "report_v2.py")
    for f in gen.glob("study_*v2*"):
        shutil.copy2(f, rep / f.name)
    for f in ("study_table_panelH.tex", "study_table_robustness.tex"):
        shutil.copy2(gen / f, rep / f)
    shutil.copy2(Path(fig_dir) / "study_evidence.pdf", rep / "study_evidence.pdf")
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(ev2)}"
             for f in sorted(ev2.rglob("*")) if f.is_file() and f.name != "MANIFEST.sha256"]
    (ev2 / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps(m, indent=1))
    print("manifest files:", len(lines))


if __name__ == "__main__":
    main(*sys.argv[1:4])
