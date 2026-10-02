from fractions import Fraction as F
import math

import numpy as np
import pytest

from robust_mckp import Option, PricingInstance, from_pricing_data, filter_admissible_options, solve
from robust_mckp.primitive_admissibility import price_is_admissible, inward_generated_price
from scripts.run_v3_independent_rq_campaign import generate_pricing_data


def test_rounded_lower_endpoint_is_rejected_without_moving_supplied_prices():
    outside = 1.0-0.15
    assert abs(F(outside)-1)-F(0.15) == F(1, 2**55)
    instance = from_pricing_data(reference_prices=[1.0], unit_costs=[0.5], weights=[1.0],
        price_menus=[[outside, 1.0]], demands=[[100.0, 5.0]], uncertainties=[[0.0, 0.0]],
        margin_target=1.0, tolerances=[0.15], gamma=0)
    assert [o.price for o in instance.items[0]] == [1.0]
    solution = solve(instance)
    assert solution.is_feasible
    assert solution.objective == 5.0
    assert solution.selections == [0]
    assert instance.original_selection(solution.selections) == [1]


def test_only_invalid_supplied_menu_is_rejected():
    with pytest.raises(ValueError, match='no admissible'):
        from_pricing_data(reference_prices=[1.0], unit_costs=[0.5], weights=[1.0],
            price_menus=[[1.0-0.15]], demands=[[1.0]], uncertainties=[[0.0]],
            margin_target=1.0, tolerances=[0.15], gamma=0)


def test_repeated_filtering_preserves_original_option_ids():
    raw = PricingInstance(items=[[
        Option(10, 5, 0, price=1.0-0.15), Option(5, 3, 0, price=1), Option(1, 1, 0, price=1.1)
    ]], gamma=0)
    first = filter_admissible_options(raw, [1.0], [0.15])
    second = filter_admissible_options(first, [1.0], [0.0])
    assert [o.original_index for o in first.items[0]] == [1, 2]
    assert [o.original_index for o in second.items[0]] == [1]
    assert second.original_selection([0]) == [1]


@pytest.mark.parametrize('reference', [math.ulp(0.0), 1e-200, 1.0, 1e200])
@pytest.mark.parametrize('tolerance', [0.0, 1e-12, 0.1, 0.15, 1.0, 2.0])
def test_predicate_matches_exact_primitive_condition_over_the_model_tolerance_domain(reference, tolerance):
    candidates = [reference, math.nextafter(reference,0), math.nextafter(reference,math.inf)]
    for p in candidates:
        expected = math.isfinite(p) and p > 0 and abs(F(p)-F(reference)) <= F(tolerance)*F(reference)
        assert price_is_admissible(p, reference, tolerance) == expected


@pytest.mark.parametrize('reference', [math.ulp(0.0), 1e-200, 1.0, 1e200])
def test_generated_lower_upper_endpoints_are_moved_inward_only_when_required(reference):
    sigma = 0.15
    for offset in (-sigma, 0.0, sigma):
        p = reference*(1+offset)
        fixed = inward_generated_price(p,reference,sigma)
        assert price_is_admissible(fixed,reference,sigma)
        if price_is_admissible(p,reference,sigma):
            assert fixed == p
        else:
            assert abs(F(fixed)-F(p)) > 0 and abs(F(fixed)-F(reference)) < abs(F(p)-F(reference))


def test_generator_recomputes_demand_at_strictly_admissible_prices():
    raw = generate_pricing_data(seed=710000,n=12,m=5,regime='favorable')
    assert all(price_is_admissible(p,r,sigma) for r,sigma,menu in
               zip(raw.reference_prices,raw.tolerances,raw.price_menus) for p in menu)
    assert all(np.all(np.isfinite(g)) and np.all(g>0) for g in raw.demands)


@pytest.mark.parametrize('bad_index', [-1, True, 0.5])
def test_original_ids_reject_ambiguous_invalid_values(bad_index):
    with pytest.raises((TypeError,ValueError)):
        Option(1,1,0,price=1,original_index=bad_index)


def test_generated_price_rejects_large_unexplained_movement():
    with pytest.raises(ValueError,match='eight float steps'):
        inward_generated_price(0.5,1,0.1)
