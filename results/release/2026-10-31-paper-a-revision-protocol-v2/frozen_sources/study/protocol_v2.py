"""Frozen design of the TOP-revision computational study, protocol v2.

Protocol v2 adds, without changing any v1 run:
  * method PIPE (HullRound with interval records) and MILP-SCIP on the v1
    Panel S and Panel L instances (read from the v1 evidence, hash-checked);
  * Panel H (dominant-jump, heterogeneous-scale portfolios; study.panel_h)
    with seven methods;
  * the price-of-robustness evaluation of the v1 Panel S decisions
    (study.robustness).
The task list is a pure function of the constants below and of study.protocol.
"""
from __future__ import annotations

import math

from . import protocol as P1

STUDY_SALT_V2 = 20261101      # Panel H instances and robustness draws; pilots used salt 20261030
V1_EVIDENCE = ("/Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/"
               "paper-a-top-revision-study/2026-10-31-protocol-v1")
METHODS_H = ("HR", "HR-sweep", "PIPE", "HR+BB", "DNC", "MILP", "MILP-SCIP")
METHODS_NEW = ("PIPE", "MILP-SCIP")

PANEL_H = dict(code=3, family="dominant", sizes=((20, 3), (20, 4), (100, 3), (100, 4), (500, 3), (500, 4)),
               reps=range(4), gammas=("sqrt", "0.5n"), radii=(0.2, 0.3), methods=METHODS_H, time_limit=300.0)
# Generator constants of study.panel_h (recorded here so the design is complete):
PANEL_H_GENERATOR = dict(sigma=0.4, tau=0.85, dominant_factor=10.0, dominant_m=3, eps_range=(1.5, 3.5),
                         scale="10**U[1,4]")


def _gl(n, which):
    return [(g, v) for g, v in P1.gamma_grid(n) if g in which]


def tasks():
    out = []
    # v1 panels, new methods only
    for t in P1.tasks():
        if t["method"] == "HR":
            for meth in METHODS_NEW:
                tt = dict(t)
                tt["method"] = meth
                tt["task_id"] = t["task_id"][: -len("-HR")] + f"-{meth}"
                tt["instance_source"] = "v1"
                out.append(tt)
    p = PANEL_H
    for (n, m) in p["sizes"]:
        for rep in p["reps"]:
            base_id = f"H-dominant-n{n}-m{m}-r{rep}"
            for glab, g in _gl(n, p["gammas"]):
                for rad in p["radii"]:
                    for meth in p["methods"]:
                        out.append(dict(task_id=f"{base_id}-G{glab}-R{rad:.2f}-{meth}", panel="H",
                                        panel_code=p["code"], family=p["family"], n=n, m=m, rep=rep,
                                        base_id=base_id, gamma_label=glab, gamma=g, radius=rad, method=meth,
                                        time_limit=p["time_limit"], instance_source="v2"))
    return out


def bases_h():
    seen = {}
    for t in tasks():
        if t["panel"] == "H":
            seen.setdefault(t["base_id"], dict(base_id=t["base_id"], n=t["n"], m=t["m"], rep=t["rep"]))
    return list(seen.values())


assert math.isqrt(20) == 4
