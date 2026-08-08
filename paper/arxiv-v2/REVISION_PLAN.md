# arXiv:2603.18653v2 revision plan

The checked-in `main.tex` currently reproduces the March 2026 v1 source. It is
an editable starting point, not yet the v2 manuscript.

## Scientific identity to preserve

Paper A studies robust finite-menu pricing through full-breakpoint
multiple-choice knapsack decomposition. Its independent contribution is the
pricing reduction, complete fixed-threshold decomposition, LP and rounding
structure, exact certification, and the guarded parametric sweep.

## Required v2 corrections and upgrades

- Retain every binary64-distinct original deviation plus zero in the exact
  threshold set; reduced sets are heuristic diagnostics only.
- Separate LP-safe upper-hull filtering from integer-safe option dominance.
- State exactness only for procedures that search original integer options.
- Require a valid upper-bound record for every original threshold before a
  finite global optimality gap is reported.
- Validate every incumbent directly in the original sorted-Gamma robust
  constraint.
- Incorporate the V3 exact branch-and-bound and guarded parametric-sweep
  results without importing Paper B's simultaneous interval envelope.
- Rebuild the contribution map, related work, experiments, limitations, and
  data/code statement around evidence that is present in this repository.
- Give Paper B a forward citation/relationship statement only after Paper B's
  public metadata are fixed.

## Release gates

- Run the complete Paper A test suite and the frozen computational protocol.
- Store new results in a new dated release directory; never change old evidence
  in place.
- Regenerate all numerical paper artifacts from that canonical release.
- Build public and anonymous variants and inspect the rendered PDFs.
- Scan anonymous artifacts for identity leakage.
- Verify that the arXiv source package compiles independently.
- Freeze the exact submitted commit and package before making the Paper A
  repository public or submitting v2.
