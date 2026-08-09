"""Robust MCKP pricing optimization package."""

from .model import Option, PricingInstance, Solution
from .preprocessing import filter_admissible_options, from_pricing_data
from .solver import solve
from .exact_bnb import (
    FixedThetaBNBConfig,
    FixedThetaBNBResult,
    GlobalThetaBNBConfig,
    GlobalThetaBNBResult,
    solve_fixed_theta_bnb,
    solve_global_theta_bnb,
)
from .parametric_sweep import (
    ParametricThetaSweepConfig,
    ParametricThetaSweepResult,
    build_parametric_theta_sweep,
    iter_parametric_theta_states,
    solve_global_theta_bnb_sweep,
)
from .local_budget import (
    SegmentLocalExactConfig,
    SegmentLocalExactResult,
    build_local_theta_candidates,
    robust_certificate_segment_local,
    solve_segment_local_exact,
)
from .milp_baselines import MILPBaselineResult, solve_theta_decomposition_milp_baseline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Option",
    "PricingInstance",
    "Solution",
    "from_pricing_data",
    "filter_admissible_options",
    "solve",
    "FixedThetaBNBConfig",
    "FixedThetaBNBResult",
    "GlobalThetaBNBConfig",
    "GlobalThetaBNBResult",
    "solve_fixed_theta_bnb",
    "solve_global_theta_bnb",
    "ParametricThetaSweepConfig",
    "ParametricThetaSweepResult",
    "build_parametric_theta_sweep",
    "iter_parametric_theta_states",
    "solve_global_theta_bnb_sweep",
    "SegmentLocalExactConfig",
    "SegmentLocalExactResult",
    "build_local_theta_candidates",
    "robust_certificate_segment_local",
    "solve_segment_local_exact",
    "MILPBaselineResult",
    "solve_theta_decomposition_milp_baseline",
]
