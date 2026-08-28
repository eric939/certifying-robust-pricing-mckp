PYTHON ?= python3
BASE_RELEASE_DIR ?= results/release/2026-08-09-paper-a-final-r2
V3_RELEASE_DIR ?= results/release/2026-08-28-paper-a-v3
LOCAL_DIR ?= results/local

.PHONY: install test verify reproduce-smoke clean

install:
	uv sync --extra experiments --extra validation --extra dev

test:
	$(PYTHON) -m pytest -q

verify: test
	$(PYTHON) scripts/verify_paper_a_release.py --release-dir $(BASE_RELEASE_DIR)
	$(PYTHON) scripts/verify_v3_release.py --release-dir $(V3_RELEASE_DIR)

reproduce-smoke: test
	mkdir -p $(LOCAL_DIR)
	$(PYTHON) scripts/run_mathematical_audit.py \
		--cases 20 --lp-cases 12 --sweep-cases 8 \
		--output $(LOCAL_DIR)/mathematical_audit_smoke.json
	$(PYTHON) scripts/run_paper_a_release.py --quick \
		--output-dir $(LOCAL_DIR)/paper-a-release-smoke \
		--audit-json $(LOCAL_DIR)/mathematical_audit_smoke.json

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; p=Path('results/local'); shutil.rmtree(p) if p.exists() else None"
