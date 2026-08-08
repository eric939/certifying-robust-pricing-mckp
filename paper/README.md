# Paper A manuscript records

`current` points to the only editable manuscript source, `arxiv-v2`.

- `archive/arxiv-2603.18653v1/` is an immutable copy of the exact March 2026
  arXiv PDF, source archive, extracted source, and code snapshot.
- `arxiv-v2/` is the working replacement source. It began as a byte-for-byte
  copy of the v1 extracted source and must be revised from recorded changes,
  not by altering the archived v1 payload.
- `../manuscript/robust_mckp_v3_full.pdf` is the recovered May 2026 manuscript
  and is a scientific reference for the v2 reconstruction. Its original TeX
  source was not recovered.

Generated tables, figures, numerical macros, and summaries must come from a
canonical dated evidence release. Do not hand-edit generated numerical
artifacts. The v2 source must build in a separate build directory so that the
source tree remains submission-clean.
