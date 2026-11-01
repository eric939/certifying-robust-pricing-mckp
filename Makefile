PYTHON ?= python3
LOCAL_DIR ?= results/local
STUDY := studies/paper-a-revision
ARCHIVE_TAG ?= v3.1.1
ARCHIVE_DIR ?= $(LOCAL_DIR)/archive-$(ARCHIVE_TAG)

.PHONY: install study-env test verify verify-evidence verify-evidence-full verify-archive \
	reproduce-smoke examples build clean

install:
	uv sync --extra experiments --extra validation --extra dev --extra milp --extra scip

# The exact package versions of the paper's campaigns (Python 3.12).
study-env:
	uv venv --python 3.12 .venv-study
	VIRTUAL_ENV=.venv-study uv pip install -r $(STUDY)/requirements-study.txt -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

# Revised paper: integrity, design and every derived file rebuilt from the raw records.
verify-evidence:
	cd $(STUDY) && PYTHONPATH=. $(abspath $(PYTHON)) verify_evidence.py

# ... plus all base portfolios regenerated from their seeds and 60 re-run runs.
verify-evidence-full:
	cd $(STUDY) && PYTHONPATH=. $(abspath $(PYTHON)) verify_evidence.py --instances --rerun 60

# v3 evidence, checked against the tag that produced it (its verifiers pin v3 sources).
verify-archive:
	rm -rf "$(ARCHIVE_DIR)"
	git worktree add --detach "$(ARCHIVE_DIR)" $(ARCHIVE_TAG)
	cd "$(ARCHIVE_DIR)" && PYTHONPATH=src $(abspath $(PYTHON)) scripts/verify_paper_a_release.py \
		--release-dir results/release/2026-08-09-paper-a-final-r2
	cd "$(ARCHIVE_DIR)" && PYTHONPATH=src $(abspath $(PYTHON)) scripts/verify_v3_release.py \
		--release-dir results/release/2026-08-28-paper-a-v3
	cd "$(ARCHIVE_DIR)" && PYTHONPATH=src $(abspath $(PYTHON)) scripts/verify_v3_independent_rq_release.py \
		--release-dir results/release/2026-08-29-paper-a-v3-independent-rq
	git worktree remove --force "$(ARCHIVE_DIR)"

verify: test verify-evidence verify-archive

examples:
	for f in examples/*.py; do echo "== $$f"; $(PYTHON) $$f || exit 1; done

# A bounded fresh run: the study's validation panel (all methods vs brute force).
reproduce-smoke:
	mkdir -p $(LOCAL_DIR)
	cd $(STUDY) && PYTHONPATH=. $(abspath $(PYTHON)) -m study.validate $(abspath $(LOCAL_DIR))/validation.json

build:
	rm -rf dist
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; p=Path('results/local'); shutil.rmtree(p) if p.exists() else None"
