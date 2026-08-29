# Rakshak — canonical entry points. `make eval` must stay under 15 min (NFR).
# On Windows without GNU make, use ./make.ps1 <target> — same commands.

SEED ?= 42
PY   ?= python

.PHONY: setup eval baf figures test lint

setup:
	$(PY) -m pip install -e ".[dev]"

# CLAUDE.md: every number in the README must be regenerable by `make eval`.
# BAF is +16 s. The `-` lets a clean checkout without the git-ignored 558 MB
# download still complete eval; baf.py prints the command that fetches it.
eval:
	$(PY) -m rakshak.eval.harness --seed $(SEED)
	-$(PY) -m rakshak.eval.baf --seed $(SEED)

baf:
	$(PY) -m rakshak.eval.baf --seed $(SEED)

figures:
	$(PY) -m rakshak.eval.harness --seed $(SEED) --figures-only

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
