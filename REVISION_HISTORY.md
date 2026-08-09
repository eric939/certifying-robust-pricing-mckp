# Revision history and correction disclosure

## August 2026 definitive Paper A revision

This revision freezes the contribution as a certifying framework for
finite-menu, integer-budget Γ-robust discrete pricing. It does not claim the
classical Bertsimas–Sim scalar threshold representation, finite deviation
enumeration, or MCKP hull geometry as new in isolation. It also excludes Paper
B's simultaneous interval/group-envelope bound.

The final manuscript replaces inherited single-seed numerical and stylized
application claims with one dated multi-seed controlled evidence release.
Every reported number is generated from its canonical CSV/JSON files. The
paper now distinguishes exact-arithmetic theorems, binary64 implementation
checks, independent solver comparisons, and observational timing.

## Discovered threshold-clustering counterexample

During the Paper A/Paper B split, the recovered implementation was found to
merge threshold candidates within solver tolerance. That contradicted the
full-original-breakpoint theorem. A 100-group, Γ=1 instance with deviations
`1` and `1 + 9e-11` makes the larger breakpoint uniquely necessary for
feasibility; clustering at `1e-9` can therefore produce a false infeasibility
report.

The final implementation preserves zero and every binary64-distinct original
deviation. Structural dominance, equal-cost merging, event activation, and
zero-length tests use exact binary64 comparisons. Numerical tolerances remain
only in solver feasibility/bound comparisons and are disclosed separately.
Permanent regression and mathematical-audit cases record the counterexample.

## Other material corrections

- Corrected Dyer's DOI from an unrelated record to
  `10.1007/BF02591729`.
- Fixed the finite-breakpoint proof on the final unbounded interval.
- Restored the fixed-threshold LP baseline-value constant.
- Clarified that the continuous segment representation has the same optimum;
  density-ordered prefix structure supplies the mapping.
- Defined the efficient concave upper frontier precisely.
- Removed the false guarantee that optional round-up repair must succeed.
- Strengthened the conditions for the conditional relative `O(1/n)` bound.
- Kept below-hull non-dominated options in exact integer search.
- Required an upper-bound record for every original threshold before reporting
  a finite global anytime gap.

The immutable v1 source and PDF remain under `paper/archive/`; they were not
modified.

## Recorded pre-release failed gate

The first 1,000-instance mathematical-audit run on 2026-08-09 failed four LP
checks from one widely scaled floating-point instance. The audit had forced a
zero feasibility tolerance, and algebraically equivalent accumulation paths
differed at roundoff scale; the independent HiGHS LP remained feasible. No
threshold was missing and all global branch-and-bound results still matched
exhaustive enumeration. The audit protocol was corrected to use a separately
recorded `1e-12` LP-feasibility tolerance while keeping candidate construction,
dominance, and breakpoint identity exact. The failed report is retained in the
development record and was not used as release evidence.
