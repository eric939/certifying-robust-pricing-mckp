"""Frozen design of the TOP-revision computational study (protocol v1).

Every planned (instance, method) pair is a task.  The task list is a pure
function of the constants below, so freezing this file freezes the design.
"""
from __future__ import annotations

import math

STUDY_SALT = 20261031          # study seeds; the pilot used salt 20261030
METHODS_ALL = ("HR", "HR-sweep", "HR-dom", "HR+BB", "DNC", "MILP")
FAMILIES = ("regular", "charm", "noisy")
RADII = (0.05, 0.10, 0.20, 0.30)


def gamma_grid(n):
    """Gamma in {0, floor(sqrt n), floor(0.1n), floor(0.5n), n}, labelled."""
    return (("0", 0), ("sqrt", int(math.isqrt(n))), ("0.1n", n // 10), ("0.5n", n // 2), ("n", n))


PANELS = {
    # Sensitivity and method comparison at moderate size.
    "S": dict(code=1, families=FAMILIES, sizes=((200, 16),), reps=range(4),
              gammas="grid", radii=RADII, methods=METHODS_ALL, time_limit=60.0),
    # Scalability.
    "L": dict(code=2, families=("regular", "charm"), reps=range(3),
              sizes=tuple((n, m) for n in (1000, 2000, 5000) for m in (8, 16, 32)),
              gammas="sqrt", radii=(0.10,), methods=METHODS_ALL, time_limit=600.0),
}


def tasks():
    out = []
    for pname, p in PANELS.items():
        for fam in p["families"]:
            for (n, m) in p["sizes"]:
                for rep in p["reps"]:
                    base_id = f"{pname}-{fam}-n{n}-m{m}-r{rep}"
                    gl = gamma_grid(n) if p["gammas"] == "grid" else [g for g in gamma_grid(n) if g[0] == p["gammas"]]
                    for glab, g in gl:
                        for rad in p["radii"]:
                            for meth in p["methods"]:
                                tid = f"{base_id}-G{glab}-R{rad:.2f}-{meth}"
                                out.append(dict(task_id=tid, panel=pname, panel_code=p["code"], family=fam,
                                                n=n, m=m, rep=rep, base_id=base_id, gamma_label=glab,
                                                gamma=g, radius=rad, method=meth,
                                                time_limit=p["time_limit"]))
    return out


def bases():
    seen = {}
    for t in tasks():
        seen.setdefault(t["base_id"], dict(base_id=t["base_id"], panel_code=t["panel_code"],
                                           family=t["family"], n=t["n"], m=t["m"], rep=t["rep"]))
    return list(seen.values())
