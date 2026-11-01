"""Certified solvers for finite-menu Gamma-robust pricing and the robust MCKP.

Start with :func:`solve`, :func:`verify` and :func:`instance_from_arrays` or
:func:`from_pricing_data`; see the README for a tour.
"""

from .model import Option, PricingInstance, Solution
from .preprocessing import filter_admissible_options, from_pricing_data
from .solver import solve as solve_reference
from .api import METHODS, Result, Verification, certify, instance_from_arrays, solve, verify
from .flat import FlatInstance
from .primitive_certificate import (
    ExactPrimitivePricingData,
    build_primitive_certificate,
    verify_primitive_certificate,
)
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
from .global_certificate import (
    ExactLPGroupPosition,
    FixedThetaDualCertificate,
    GlobalCertificateVerification,
    GlobalDualCertificate,
    LPToRoundingProvenance,
    build_global_certificate,
    instance_sha256,
    original_theta_candidates,
    verify_global_certificate,
)

__version__ = "4.0.0"

__all__ = [
    "__version__",
    "solve",
    "verify",
    "certify",
    "instance_from_arrays",
    "METHODS",
    "Result",
    "Verification",
    "FlatInstance",
    "ExactPrimitivePricingData",
    "build_primitive_certificate",
    "verify_primitive_certificate",
    "Option",
    "PricingInstance",
    "Solution",
    "from_pricing_data",
    "filter_admissible_options",
    "solve_reference",
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
    "ExactLPGroupPosition",
    "FixedThetaDualCertificate",
    "GlobalCertificateVerification",
    "GlobalDualCertificate",
    "LPToRoundingProvenance",
    "build_global_certificate",
    "instance_sha256",
    "original_theta_candidates",
    "verify_global_certificate",
]
