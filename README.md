# A Certifying MCKP Framework for Γ-Robust Discrete Pricing

This is the frozen Paper A project associated with
[arXiv:2603.18653](https://arxiv.org/abs/2603.18653). It studies a finite-menu
pricing problem with a ratio-margin requirement and integer-budget demand
uncertainty.

The paper's contribution is an end-to-end certifying chain:

- an exact pricing-to-MCKP reduction;
- the complete threshold family `B = {0} ∪ {|t_ij|}`;
- fixed-threshold upper-hull LP bounds and robust-feasible one-item rounding;
- exact-arithmetic branch-and-bound theory with full-family gap accounting;
- a binary64 implementation that preserves every distinct input threshold and
  checks incumbents with the original sorted-Γ certificate.

The scalar threshold representation and MCKP LP geometry are classical and are
not claimed as new in isolation.

## Paper A and Paper B

Paper A exposes and certifies the complete family of fixed-threshold MCKPs. The
separate [Paper B repository](https://github.com/eric939/simultaneous-group-envelope-mckp) starts
from that family and develops simultaneous interval/group-envelope bounds over
many thresholds. Paper B therefore has different main theorems, algorithms,
evidence, and a separate arXiv record. See [PAPER_LINEAGE.md](PAPER_LINEAGE.md).

## Definitive files

- `paper/current/`: final Paper A source (`arxiv-v2`).
- `results/release/2026-08-09-paper-a-final/`: canonical evidence release.
- `src/robust_mckp/`: solver implementation.
- `tests/`: regression and correctness tests.
- `research/CLAIM_EVIDENCE_MATRIX.md`: claim-to-proof/evidence map.
- `research/EVIDENCE_LEDGER_A.csv`: primary-source reference ledger.
- `REVISION_HISTORY.md`: correction disclosure and frozen scope.

Historical arXiv-v1 material is immutable under
`paper/archive/arxiv-2603.18653v1/`. The recovered intermediate V3 PDF is kept
only as provenance under `manuscript/`.

## Install and verify

```bash
uv sync --extra experiments --extra validation --extra dev
make verify PYTHON=.venv/bin/python
```

`make verify` runs the full test suite, verifies every SHA-256 evidence-manifest
entry, checks all scientific gates, builds public and anonymous manuscripts,
rejects reference/layout warnings, and scans the anonymous PDF for identity.

Build the deterministic arXiv source archive with:

```bash
make arxiv-package PYTHON=.venv/bin/python
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the evidence-generation
protocol. Generated numerical artifacts are never hand-edited.

## License and citation

The software is MIT licensed. Citation metadata are in [CITATION.cff](CITATION.cff).
