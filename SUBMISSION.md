# Paper A submission and freeze record

The definitive public source is `paper/current/main.tex`. The anonymous wrapper
`paper/current/main_blind.tex` defines the blind flag and inputs that same
scientific source.

## Required gates

```bash
make verify PYTHON=.venv/bin/python
make arxiv-package PYTHON=.venv/bin/python
```

The first command builds `paper/current/main.pdf` and
`paper/current/main_blind.pdf` after verifying the canonical evidence. The
second creates `submission_packages/arxiv-2603.18653v2-source.zip` with fixed
member timestamps.

Before upload or tagging:

- render and visually inspect every public and blind PDF page;
- confirm the public PDF contains author, affiliation, acknowledgments, and the
  Paper A repository URL;
- confirm the blind PDF contains none of those identifiers;
- compile the extracted arXiv archive in a clean temporary directory;
- record the exact commit used for the uploaded source;
- upload as version 2 of arXiv:2603.18653, not as a new Paper A record;
- only then create the immutable release tag and GitHub release.

Paper B is a separate repository and future arXiv record. Its bibliography must
cite the frozen Paper A version, and its introduction must state that its
simultaneous all-threshold envelope begins from Paper A's fixed-threshold
decomposition.
