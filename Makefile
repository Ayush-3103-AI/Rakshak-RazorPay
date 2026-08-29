# Rakshak — canonical entry points. `make eval` must stay under 15 min (NFR).
# On Windows without GNU make, use ./make.ps1 <target> — same commands.

SEED ?= 42
PY   ?= python

.PHONY: setup eval figures test lint

setup:
	$(PY) -m pip install -e ".[dev]"

eval:
	$(PY) -m rakshak.eval.harness --seed $(SEED)

figures:
	$(PY) -m rakshak.eval.harness --seed $(SEED) --figures-only

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
