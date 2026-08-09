from __future__ import annotations

import pytest

from robust_mckp import Option, PricingInstance
from robust_mckp.certificate import certificate_is_feasible, compute_certificate
from scripts.run_publishable_experiments import solve_full_robust_highs


@pytest.mark.parametrize(
    ("gamma", "expected"),
    [
        (0, True),
        (1, True),
        (2, False),
        (3, False),
    ],
)
def test_exact_certificate_sign_handles_ties_and_gamma_edges(gamma: int, expected: bool) -> None:
    instance = PricingInstance(
        items=[
            [Option(value=0.0, margin=1.0, uncertainty=3.0)],
            [Option(value=0.0, margin=1.0, uncertainty=-3.0)],
            [Option(value=0.0, margin=1.0, uncertainty=0.0)],
        ],
        gamma=gamma,
    )
    assert certificate_is_feasible(instance, [0, 0, 0]) is expected


def test_exact_certificate_sign_resolves_positive_cancellation_hidden_by_float() -> None:
    instance = PricingInstance(
        items=[
            [Option(value=0.0, margin=1e16, uncertainty=1e16)],
            [Option(value=0.0, margin=1.0, uncertainty=0.0)],
        ],
        gamma=1,
    )
    assert compute_certificate(instance, [0, 0]) == 0.0
    assert certificate_is_feasible(instance, [0, 0])


def test_exact_certificate_sign_resolves_negative_cancellation_hidden_by_float() -> None:
    instance = PricingInstance(
        items=[
            [Option(value=0.0, margin=1e16, uncertainty=1e16)],
            [Option(value=0.0, margin=-1.0, uncertainty=0.0)],
        ],
        gamma=1,
    )
    assert compute_certificate(instance, [0, 0]) == 0.0
    assert not certificate_is_feasible(instance, [0, 0])


def test_independent_milp_does_not_certify_negative_gamma_residual_inside_tolerance() -> None:
    instance = PricingInstance(
        items=[
            [Option(value=1.0, margin=1.0, uncertainty=1.0)],
            [Option(value=1.0, margin=-5e-10, uncertainty=0.0)],
        ],
        gamma=1,
    )
    result = solve_full_robust_highs(instance)
    if result["status"] == "UNAVAILABLE":
        pytest.skip("SciPy/HiGHS is unavailable")
    assert result["selections"] == [0, 0]
    assert float(result["certificate_value"]) < 0.0
    assert not result["certified"]
