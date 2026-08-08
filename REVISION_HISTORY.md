# Revision History and Scientific Guardrails

This file replaces earlier transient worklogs, mock-review reports, claim
checklists, and final-gate notes. It is intentionally short and should record
only durable facts needed by future maintainers.

## August 2026 arXiv-v2 correction

Date: 2026-08-09.

During the Paper A/Paper B repository split, the preserved V3 implementation
was found to merge threshold candidates whose difference was at most the solver
tolerance. This conflicted with the full-original-breakpoint theorem. A
100-group, $\Gamma=1$ instance with deviations `1` and `1 + 9e-11` makes the
upper value the only feasible threshold; clustering at `1e-9` removes it and
can produce a false infeasibility report.

The Paper A v2 implementation now preserves every binary64-distinct original
deviation plus zero in both global and segment-local candidate sets. The
parametric sweep activates breakpoints by exact candidate order and reuses an
item hull only when its stored cost--value arrays are exactly unchanged.
Regression tests cover candidate construction, the unique-feasible-breakpoint
global solve, segment-local construction, and swept-state reconstruction. This
is a post-V3 protocol correction and must be disclosed in the arXiv-v2 change
description; no old evidence is modified in place.

## Current Paper Positioning

The manuscript is framed as a certifying decomposition framework for
finite-menu integer-budget \(\Gamma\)-robust MCKP:

- exact full-original-breakpoint \(\theta\) enumeration,
- fixed-\(\theta\) upper-hull LP certificates,
- robust-feasible HullRound one-item additive certificate,
- exact \(\theta\)-enumerated branch-and-bound with valid gap accounting,
- parametric sweep diagnostics over the same full breakpoint set,
- diagnostic insight rather than general-purpose solver replacement.

Discrete pricing is the motivating and illustrative application. The pricing
study is controlled/semi-synthetic and should not be described as
transaction-level scanner-data validation, causal demand estimation, or
commercial pricing performance evidence.

## Non-Negotiable Correctness Rules

- Exact global mode uses the full set
  `B = {0} union {|t_ij|}` over original admissible options.
- Reduced \(\theta\) sets may be used only for heuristic diagnostics, not for
  exactness claims.
- Upper hulls are LP-bound objects. They are not integer-safe preprocessing.
- Exact branch-and-bound must preserve original non-dominated integer options,
  including below-hull options.
- Integer-safe pruning may remove only same-item options with no larger
  fixed-\(\theta\) cost and no smaller value, with at least one strict
  improvement.
- Every reported robust incumbent must pass the direct sorted-\(\Gamma\)
  certificate check.
- A finite limited-run global gap is valid only when every \(\theta\) has a
  valid upper-bound record.
- Limited rows are valid-gap diagnostics, not optima.
- The implemented scope is integer \(\Gamma\in\{0,\ldots,n\}\); fractional
  Bertsimas--Sim budgets require additional theory and tests.

## Latest Revision Pass

Date: 2026-05-22.

Changes:

- Strengthened the manuscript novelty map and global exactness/gap theorem.
- Updated the exact global solver to initialize root LP upper-bound records for
  every \(\theta\) before reporting finite global gaps.
- Added tests for the per-\(\theta\) upper-bound-record invariant under limited
  global runs.
- Added HullRound diagnostics for \(\Delta V_{\max}^{\theta}\), realized
  round-down loss, and scale-normalized certificate strength.
- Added `delta_v_max_over_lp_ub` to publishable experiment CSV generation.
- Clarified the split between public GitHub code and the exact local
  submission artifact snapshot.
- Synced rebuilt public/blind manuscript source and PDFs into
  `submission_ready/`.

Validation commands run:

```bash
.venv/bin/python -m pytest -q
cd paper_versions/v2 && tectonic --keep-logs main_v2.tex
cd paper_versions/v2 && tectonic --keep-logs main_v2_blind.tex
.venv/bin/python scripts/run_clean_repro_check.py --quick
.venv/bin/python scripts/run_solver_benchmarks.py --smoke --exact-only --theta-time-limit 2 --full-time-limit 2 --fixed-time-limit 2
```

Observed results:

- Full test suite passed: 82 tests.
- Public and blind manuscript builds passed.
- Public PDF text grep found no stale section/reference wording checked in
  `SUBMISSION.md`.
- Blind PDF strict identity regex found no author, institution, GitHub, or
  arXiv self-identification leaks.
- Clean reproducibility smoke check passed.
- Open-solver smoke completed; SCIP and HiGHS were available, Gurobi and CPLEX
  were unavailable.

## Remaining Scientific Limits

- Full publication benchmark numbers were not regenerated in the latest pass.
  Do not change numerical claims unless the corresponding scripts are rerun.
- SCIP/HiGHS remain strong median-runtime baselines in the tested environment.
  The paper should not claim general runtime dominance over mature MILP
  solvers.
- Tight-capacity rows are the main exact-solver bottleneck.
- The semi-synthetic pricing application is own-price separable and controlled
  by construction; cross-price substitution and transaction-level validation
  remain future work.
