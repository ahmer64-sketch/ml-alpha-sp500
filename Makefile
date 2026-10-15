# One command reproduces every report and figure from the committed data snapshot (no network):
#     make all        (~35 min: Phase 4 ~5, Phase 6 ~15, hold-out ~10)
# Downloading a fresh snapshot is separate:  make fetch

PY := .venv/bin/python

.PHONY: all setup test reports phase1 phase2 phase3 phase4 phase5 phase6 holdout fetch clean

all: test reports

setup: .venv/.installed

.venv/.installed: requirements.txt pyproject.toml
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	$(PY) -m pip install -q -e .
	touch $@

test: setup
	$(PY) -m pytest -q

reports: phase1 phase2 phase3 phase4 phase5 phase6 holdout

phase1: setup
	$(PY) scripts/run_phase1.py > /dev/null

# also writes data/processed/features_h5.parquet (not in git)
phase2: phase1
	$(PY) scripts/run_phase2.py > /dev/null

phase3: phase2
	$(PY) scripts/run_phase3.py > /dev/null

phase4: phase3
	$(PY) scripts/run_phase4.py > /dev/null
	$(PY) scripts/make_phase4_figure.py > /dev/null

phase5: phase4
	$(PY) scripts/run_phase5.py > /dev/null

# caches each section in data/processed/phase6/ (delete it to recompute)
phase6: phase5
	$(PY) scripts/run_phase6.py > /dev/null

# The hold-out. Re-running it is safe: it refuses to run unless src/alpha/ matches the frozen commit,
# keeps the first-run time in reports/holdout_run.json, and reproduces the same predictions.
holdout: phase6
	$(PY) scripts/run_holdout.py > /dev/null

fetch: setup
	$(PY) scripts/fetch_data.py

clean:
	rm -rf .venv .pytest_cache src/*.egg-info data/processed
