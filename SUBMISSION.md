# Submission Package

This file is the submission/package manifest for Paper A. The editable
arXiv-v2 source is tracked under `paper/current`; generated public/blind PDFs,
tables, and submission bundles must be written to separate build or release
directories.

## Public and Blind Builds

The historical commands below describe the recovered V3 artifact workspace and
are retained for provenance. They are not the build commands for the new
`paper/current` arXiv-v2 source:

```bash
cd paper_versions/v2
tectonic --keep-logs main_v2.tex
cp main_v2.pdf ../../submission_ready/main_v2_public.pdf
cp main_v2.tex ../../submission_ready/main_v2.tex
```

Blind manuscript, from the local artifact workspace:

```bash
cd paper_versions/v2
tectonic --keep-logs main_v2_blind.tex
cp main_v2_blind.pdf ../../submission_ready/main_v2_blind_submission.pdf
cp main_v2_blind.tex ../../submission_ready/main_v2_blind.tex
```

The public build contains the author identity, prior-version disclosure, public
GitHub URL, acknowledgments, and data/code statement. The blind build should
not contain the author name, institutional metadata, GitHub username, arXiv ID,
acknowledgments, or self-citation bibliography item.

## Code and Artifact Availability

Public repository:

```text
https://github.com/eric939/robust_pricing_mckp
```

Verify the current public HEAD with:

```bash
git ls-remote https://github.com/eric939/robust_pricing_mckp.git HEAD
```

The public repository contains the solver package, tests, experiment drivers,
plotting/summarization scripts, and reproducibility instructions. The exact
submission snapshot also includes generated manuscript artifacts that are not
tracked in the public Git repository.

Expected submission artifact contents:

- `submission_ready/main_v2.tex`
- `submission_ready/main_v2_blind.tex`
- `submission_ready/main_v2_public.pdf`
- `submission_ready/main_v2_blind_submission.pdf`
- `submission_ready/auto/`
- `submission_ready/tables/`
- `submission_ready/figures/`
- `submission_ready/result_summaries/`
- `src/`
- `scripts/`
- `tests/`
- `README.md`
- `REPRODUCIBILITY.md`
- `SUBMISSION.md`
- `REVISION_HISTORY.md`

Refresh the lightweight artifact bundle with:

```bash
zip -qr submission_artifacts_bundle.zip \
  submission_ready src scripts tests \
  README.md REPRODUCIBILITY.md SUBMISSION.md REVISION_HISTORY.md \
  -x '*.DS_Store' '*__pycache__*' '*.pyc' '*.log' '*.aux' '*.out' '*.synctex.gz'
```

## Final Checks

```bash
.venv/bin/python -m pytest -q
cd paper_versions/v2 && tectonic --keep-logs main_v2.tex
cd paper_versions/v2 && tectonic --keep-logs main_v2_blind.tex
cd paper_versions/v2 && pdftotext main_v2.pdf /tmp/main_v2_public.txt
cd paper_versions/v2 && pdftotext main_v2_blind.pdf /tmp/main_v2_blind.txt
```

Check for stale manuscript strings:

```bash
grep -ni "Section 9 for extensions\|Sections 4 and 5\|earlier repository drafts\|full-theta\|reduced-theta\|theta-enumerated\|theta enumeration\|nodes/thetas\|TODO\|PLACEHOLDER" /tmp/main_v2_public.txt || true
grep -ni "undefined references\|undefined citation\|multiply defined\|missing file\|rerun to get cross-references" paper_versions/v2/main_v2.log paper_versions/v2/main_v2_blind.log || true
```

Strict blind identity check:

```bash
python3 - <<'PY'
from pathlib import Path
import re
text = Path("/tmp/main_v2_blind.txt").read_text(errors="ignore")
patterns = [
    r"\bEric\b", r"\bShao\b", r"ershao", r"ETH Zürich", r"ETH Zurich",
    r"github\.com/eric939", r"robust_mckp", r"2603\.18653",
    r"Kunoth", r"Mollet", r"Hannoversche", r"University of Cologne",
    r"Department of Mathematics, ETH",
]
for pattern in patterns:
    if re.search(pattern, text, flags=re.I):
        print("HIT", pattern)
PY
```

## Data Policy

No raw transaction data are redistributed. The semi-synthetic application stores
generated calibration records and result summaries only. Gurobi and CPLEX were
unavailable for the reported open-solver benchmark environment; SCIP and HiGHS
are used where available.
