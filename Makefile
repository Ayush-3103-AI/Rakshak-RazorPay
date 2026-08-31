# Rakshak — canonical entry points. `make eval` must stay under 15 min (NFR).
# On Windows without GNU make, use ./make.ps1 <target> — same commands.

SEED ?= 42
# T-0021: a stock Linux box (this Makefile's actual target — GNU make isn't
# even installed on the Windows dev machine) has no `python` on PATH, only
# `python3` — `make setup` failed at word one on a clean checkout. Override
# with `make PY=python ...` on a system where only `python` exists.
PY   ?= python3

.PHONY: setup eval baf figures test lint

setup:
	# T-0021: Debian/Ubuntu's system pip refuses a bare install with
	# "externally-managed-environment" (PEP 668) on a clean box with no venv
	# active. --break-system-packages is a pip>=23 flag that is a no-op on any
	# environment without that marker (venv, conda, non-Debian Python), so it
	# is safe everywhere this project's pyproject.toml (python>=3.11) runs. If
	# your pip predates 23.0 and rejects the flag, activate a venv first.
	$(PY) -m pip install -e ".[dev]" --break-system-packages

# CLAUDE.md: every number in the README must be regenerable by `make eval`.
# Order matters only for readability; each target writes its own file.
#   generator -> data/synthetic/{transactions,state_paths}.parquet (git-ignored
#                inputs every step below reads; T-0021 found `eval` assumed a
#                prior run had already written these — a clean checkout has none)
#   harness   -> summary.md, sensitivity.md/.csv, figures/sensitivity.png  (validate)
#   verdict   -> verdict.md, sensitivity_test.csv, figures/sensitivity_test.png (test, K2)
#   ablations -> ablations.md    (FR-018, 6 fits, the slowest step at ~70 s)
#   lag_probe -> lag_probe.md    (T-0011 detection-lag attribution + leakage clearance)
#   typology  -> typology_recall.md (FR-005, recall broken down by injected typology)
#   reasons   -> reasons.json    (FR-014, the merchant-facing strings; T-0014 renders it)
# BAF is +16 s. The `-` lets a clean checkout without the git-ignored 558 MB
# download still complete eval; baf.py prints the command that fetches it.
eval:
	$(PY) -m rakshak.generator --seed $(SEED)
	$(PY) -m rakshak.eval.harness --seed $(SEED)
	$(PY) -m rakshak.eval.verdict --seed $(SEED)
	$(PY) -m rakshak.eval.ablations --seed $(SEED)
	$(PY) -m rakshak.eval.lag_probe --seed $(SEED)
	$(PY) -m rakshak.eval.typology --seed $(SEED)
	$(PY) -m rakshak.explain.reasons --seed $(SEED)
	-$(PY) -m rakshak.eval.baf --seed $(SEED)

baf:
	$(PY) -m rakshak.eval.baf --seed $(SEED)

figures:
	$(PY) -m rakshak.eval.harness --seed $(SEED) --figures-only

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
