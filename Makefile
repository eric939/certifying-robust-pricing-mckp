PYTHON ?= python3

.PHONY: install-dev test check clean-check publishable-smoke solver-smoke pathc-smoke clean

install-dev:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[experiments,validation,dev]"

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) -m pytest -q
	$(PYTHON) scripts/run_clean_repro_check.py --quick

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
