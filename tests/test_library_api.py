"""The v4 library interface: solve / verify / certify and the fast methods."""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import robust_mckp as rm

DATA = Path(__file__).parent / "data"
COMBINATORIAL = ["exact", "certified", "hullround", "hullround-full", "hullround-dominance", "exact-layer"]


def random_instance(seed: int, n: int = 6, m: int = 4, gamma=None):
    rng = np.random.default_rng(seed)
    v = rng.uniform(5.0, 10.0, (n, m))
    s = rng.uniform(-1.0, 3.0, (n, m))
    t = rng.uniform(0.0, 2.0, (n, m))
    g = int(rng.integers(0, n + 1)) if gamma is None else gamma
    return rm.instance_from_arrays(v.tolist(), s.tolist(), t.tolist(), gamma=g)


@pytest.mark.parametrize("seed", range(12))
def test_methods_agree_with_brute_force(seed):
    inst = random_instance(seed)
    opt = rm.solve(inst, "bruteforce")
    for method in COMBINATORIAL + ["milp-highs", "reference"]:
        if method == "milp-highs":
            pytest.importorskip("highspy")
        res = rm.solve(inst, method)
        if opt.status == "infeasible":
            assert res.selection is None and res.status == "infeasible", method
            continue
        assert res.feasible and res.exact.feasible, method
        assert res.exact.revenue == sum(Fraction(inst.items[i][j].value) for i, j in enumerate(res.selection))
        if res.status == "optimal":
            assert res.revenue == pytest.approx(opt.revenue, rel=1e-6), method
        else:  # HullRound: H <= OPT <= U
            assert res.revenue <= opt.revenue * (1 + 1e-12), method
            if res.upper_bound is not None:
                assert res.upper_bound >= opt.revenue * (1 - 1e-12), method
                assert 0.0 <= res.gap < 1.0


def test_result_fields_and_original_ids():
    inst = rm.from_pricing_data(
        reference_prices=[10.0, 20.0], unit_costs=[6.0, 12.0], weights=[1.0, 1.0],
        price_menus=[[5.0, 9.0, 10.0, 11.0, 15.0], [18.0, 20.0, 22.0]],
        demands=[[9.0, 6.0, 5.0, 4.0, 1.0], [3.0, 2.5, 2.0]],
        uncertainties=[[0.2] * 5, [0.1] * 3], margin_target=1.2, tolerances=[0.15, 0.15], gamma=1)
    res = rm.solve(inst)
    assert res.status == "optimal" and res.method == "exact"
    assert len(res.selection) == 2
    # menu 0 keeps prices 9, 10, 11 (original IDs 1..3): the IDs refer to the supplied menu
    assert res.original_ids[0] == res.selection[0] + 1
    assert res.original_ids[1] == res.selection[1]
    assert isinstance(res.exact.residual, Fraction) and res.exact.residual >= 0
    assert res.details["thresholds"] >= 1


def test_verify_detects_infeasible_decision():
    inst = rm.instance_from_arrays([[1.0, 5.0]], [[1.0, 1.0]], [[0.5, 2.0]], gamma=1)
    assert rm.verify(inst, [0]).feasible and rm.verify(inst, [0]).residual == Fraction(1, 2)
    bad = rm.verify(inst, [1])
    assert not bad.feasible and bad.residual == -1
    assert rm.solve(inst).selection == (0,)
    with pytest.raises(ValueError):
        rm.verify(inst, [2])


def test_fractions_are_checked_exactly():
    third = Fraction(1, 3)
    inst = rm.instance_from_arrays([[1.0, 2.0]], [[third, third]], [[0.0, third]], gamma=1)
    assert rm.verify(inst, [1]).residual == 0
    assert rm.verify(inst, [1]).feasible


def test_infeasible_instance():
    inst = rm.instance_from_arrays([[1.0, 2.0], [1.0, 2.0]], [[-1.0, -2.0], [0.5, 0.2]], [[0.0, 0.0], [0.0, 0.0]],
                                   gamma=0)
    for method in ["exact", "certified", "hullround", "exact-layer", "bruteforce"]:
        res = rm.solve(inst, method)
        assert res.status == "infeasible" and res.selection is None and not res.feasible, method


def test_primitive_data_input():
    data = rm.ExactPrimitivePricingData(
        references=(10.0, 20.0), costs=(6.0, 12.0), exposures=(1.0, 2.0), tolerances=(0.15, 0.15),
        menus=((5.0, 9.0, 10.0, 11.0), (18.0, 20.0, 22.0)), demands=((9.0, 6.0, 5.0, 4.0), (3.0, 2.5, 2.0)),
        kappa=1.2, radius=0.2, gamma=1)
    res = rm.solve(data, "exact")
    cert = rm.solve(data, "certified")
    assert res.status == "optimal" and res.exact.feasible
    assert cert.revenue <= res.revenue * (1 + 1e-12) <= cert.upper_bound * (1 + 1e-12)
    assert res.original_ids[0] in (1, 2, 3)  # price 5 is inadmissible
    payload = rm.build_primitive_certificate(data, list(res.original_ids))
    assert rm.verify_primitive_certificate(data, payload).valid


def test_certify_matches_global_certificate():
    inst = random_instance(3, gamma=2)
    res = rm.solve(inst, "certified")
    cert = rm.certify(inst, res.selection)
    assert rm.verify_global_certificate(inst, cert).valid
    assert float(cert.u) == pytest.approx(res.upper_bound, rel=1e-9)
    assert cert.h == rm.verify(inst, res.selection).revenue


def test_flat_instance_is_reusable_and_time_limit_is_honoured():
    inst = random_instance(5, n=40, m=8, gamma=6)
    flat = rm.FlatInstance(inst)
    a, b = rm.solve(flat, "exact"), rm.solve(flat, "certified")
    assert a.revenue >= b.revenue * (1 - 1e-12)
    quick = rm.solve(flat, "hullround-full", time_limit=0.0)
    assert quick.status in ("time_limit", "certified")
    if quick.status == "time_limit" and quick.upper_bound is not None:
        assert quick.upper_bound >= a.revenue * (1 - 1e-9)


def test_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        rm.solve(random_instance(0), "simplex")
    assert set(COMBINATORIAL) <= set(rm.METHODS)


def test_scip_if_installed():
    pytest.importorskip("pyscipopt")
    inst = random_instance(7, gamma=3)
    assert rm.solve(inst, "milp-scip").revenue == pytest.approx(rm.solve(inst, "bruteforce").revenue, rel=1e-6)


def test_reference_rounds_the_true_fractional_group():
    """v3.1.1 rounded a group with lambda ~ 1e-16 here (studies/paper-a-revision/VALIDATION_NOTE.md)."""
    fx = json.loads((DATA / "greedy_fractional_regression.json").read_text())
    vals = [[[float.fromhex(x) for x in opt] for opt in group] for group in fx["options"]]
    inst = rm.instance_from_arrays([[o[0] for o in g] for g in vals], [[o[1] for o in g] for g in vals],
                                   [[o[2] for o in g] for g in vals], gamma=fx["gamma"])
    ref = rm.solve(inst, "reference")
    assert ref.revenue == pytest.approx(fx["optimum"], rel=1e-12)
    assert not math.isclose(ref.revenue, fx["v3_reference_objective"], rel_tol=1e-6)
    assert rm.solve(inst, "certified").revenue == pytest.approx(fx["optimum"], rel=1e-12)
