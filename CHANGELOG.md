# Changelog

## 4.0.0 — revised paper

The release that matches the revised paper. It turns the
repository into a library: `import robust_mckp as rm; rm.solve(instance)`.

### Added

- `robust_mckp.solve(instance, method=..., time_limit=...)`, one entry point
  for every method of the paper, returning a `Result` with the decision, its
  original menu IDs, revenue, upper bound, gap, status and an exact rational
  check of the decision. Methods: `exact` (threshold divide and conquer, the
  default), `certified` (HullRound with interval records, Proposition 23),
  `hullround` (guarded sweep, Proposition 24), `hullround-full`,
  `hullround-dominance`, `exact-layer`, `milp-highs`, `milp-scip`,
  `bruteforce` and `reference` (the v3 solver).
- `robust_mckp.verify` (exact rational check of any decision),
  `robust_mckp.certify` (exact global certificate, Theorem 12) and
  `robust_mckp.instance_from_arrays`.
- `robust_mckp.fast`: the compiled (Numba) kernels and solver drivers of the
  paper's computational study, unchanged apart from module names.
- `robust_mckp.FlatInstance`: the flat binary64 arrays of an instance with
  the exact coefficients they round.
- Exact primitive inputs (`ExactPrimitivePricingData`,
  `build_primitive_certificate`, `verify_primitive_certificate`): price
  admissibility is decided on the exact rational inputs and original menu IDs
  are kept through filtering (also used by `from_pricing_data` and
  `filter_admissible_options`).
- `studies/paper-a-revision/`: protocols, campaign drivers, report generators
  and `verify_evidence.py` for the revised paper's study.
- `results/release/2026-10-31-paper-a-revision-protocol-v1` and `-v2`: the
  evidence of both protocols (1,764 and 1,260 runs).
- Extras `milp`, `scip` and `study`; `examples/`; Makefile targets
  `study-env`, `verify-evidence`, `verify-evidence-full`, `verify-archive`,
  `examples` and `build`.

### Changed

- `robust_mckp.solve` is the new entry point. The v3 function is available as
  `robust_mckp.solve_reference` (and as `robust_mckp.solver.solve`) with
  unchanged signature and `Solution` result; the v3 drivers import it under
  that name.
- Numba is now a core dependency; Python 3.10 or later.
- The v3 evidence is verified against tag `v3.1.1` (`make verify-archive`),
  because its verifiers pin the hashes of the v3 sources.
- The report generator writes the figure without a creation date, so the
  figure PDF is byte-reproducible.

### Fixed

- `greedy_lp` (used by the v3 reference solver) now records each group's LP
  position while filling, instead of recovering it from accumulated floating
  costs. The recovery could give a group a spurious fractional weight of order
  1e-16, and the rounding step then rounded that group instead of the true
  fractional group. The returned decision was always feasible; on the
  validation instance in `tests/data/greedy_fractional_regression.json` v3.1.1
  returned 122894.32 instead of the optimum 123003.03. The effect on the
  submitted results is in the revised paper, §7.2.

## 3.1.1 and earlier

See the tags `v3.1.1`, `v3.1.0`, `v3.0.1`, `v3.0.0` and `v1.0.x`.
