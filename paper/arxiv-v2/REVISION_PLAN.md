# arXiv:2603.18653v2 completed revision record

The checked-in `main.tex` is the definitive version-2 manuscript. The March
2026 version-1 source and PDF remain immutable under `paper/archive/`.

## Scientific identity to preserve

Paper A studies robust finite-menu pricing through full-breakpoint
multiple-choice knapsack decomposition. Its independent contribution is the
pricing reduction, complete fixed-threshold decomposition, LP and rounding
structure, exact certification, and the guarded parametric sweep.

## Completed v2 corrections and upgrades

- [x] Retain every binary64-distinct original deviation plus zero in the exact
  threshold set; reduced sets are heuristic diagnostics only.
- [x] Separate LP-safe upper-hull filtering from integer-safe option dominance.
- [x] State exactness only for procedures that search original integer options.
- [x] Require a valid upper-bound record for every original threshold before a
  finite global optimality gap is reported.
- [x] Validate every incumbent by the exact sign of the original sorted-Gamma
  certificate over the binary64 input coefficients.
- [x] Preserve every positive binary64-distinct hull segment in LP bounds.
- [x] Incorporate the V3 exact branch-and-bound and guarded parametric-sweep
  results without importing Paper B's simultaneous interval envelope.
- [x] Rebuild the contribution map, related work, experiments, limitations, and
  data/code statement around evidence that is present in this repository.
- [x] State the Paper A/Paper B relationship without importing unpublished
  Paper B claims or relying on unfixed Paper B bibliographic metadata.

## Completed repository gates

- [x] Run the complete Paper A test suite and the computational protocol.
- [x] Store changed-algorithm results in a new dated release directory; never change old evidence
  in place.
- [x] Regenerate all numerical paper artifacts from that canonical release.
- [x] Build public and anonymous variants and scan anonymous artifacts for identity leakage.
- [x] Verify that the arXiv source package compiles independently.

The remaining external actions are recorded in the root `SUBMISSION.md`: inspect
the final rendered package in arXiv's own preview, submit it as version 2 of the
existing record, and only then tag and release the exact submitted commit.
