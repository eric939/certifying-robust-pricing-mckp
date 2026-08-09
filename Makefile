PYTHON ?= python3
RELEASE_DIR ?= results/release/2026-08-09-paper-a-final

.PHONY: install-dev test check verify paper arxiv-package clean-check publishable-smoke solver-smoke pathc-smoke clean

install-dev:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[experiments,validation,dev]"

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) -m pytest -q
	$(PYTHON) scripts/run_clean_repro_check.py --quick

paper:
	cd paper/current && tectonic --keep-logs main.tex
	cd paper/current && tectonic --keep-logs main_blind.tex

verify:
	$(PYTHON) -m pytest -q
	$(PYTHON) scripts/verify_paper_a_release.py --release-dir $(RELEASE_DIR)

arxiv-package: verify
	$(PYTHON) scripts/build_arxiv_package.py --output submission_packages/arxiv-2603.18653v2-source.zip

clean-check:
	$(PYTHON) scripts/run_clean_repro_check.py --quick

publishable-smoke:
	$(PYTHON) scripts/run_publication_benchmarks.py --smoke --output-dir results/publication_benchmarks_smoke
	$(PYTHON) scripts/run_publishable_experiments.py --smoke

solver-smoke:
	$(PYTHON) scripts/run_solver_benchmarks.py --smoke

pathc-smoke:
	$(PYTHON) scripts/run_pathC_data_calibration.py --source synthetic_only --output-dir results/pathC/calibration
	$(PYTHON) scripts/run_pathC_semisynthetic_application.py --calibration-dir results/pathC/calibration --output-dir results/pathC/semisynthetic_application_smoke --seeds 1 --n 60 --m 8 --stress-scenarios 200 --gamma-grid 0,sqrt,n --run-exact-small-subset

clean:
	rm -rf output .pytest_cache
