# Rakshak — canonical entry points. `make eval` must stay under 15 min (NFR).
# On Windows without GNU make, use ./make.ps1 <target> — same commands.

SEED ?= 42
PY   ?= python

.PHONY: setup eval baf figures test lint

setup:
	$(PY) -m pip install -e ".[dev]"

# CLAUDE.md: every number in the README must be regenerable by `make eval`.
# Order matters only for readability; each target writes its own file.
#   harness   -> summary.md, sensitivity.md/.csv, figures/sensitivity.png  (validate)
#   verdict   -> verdict.md, sensitivity_test.csv, figures/sensitivity_test.png (test, K2)
#   ablations -> ablations.md    (FR-018, 6 fits, the slowest step at ~70 s)
#   lag_probe -> lag_probe.md    (T-0011 detection-lag attribution + leakage clearance)
#   typology  -> typology_recall.md (FR-005, recall broken down by injected typology)
#   reasons   -> reasons.json    (FR-014, the merchant-facing strings; T-0014 renders it)
# BAF is +16 s. The `-` lets a clean checkout without the git-ignored 558 MB
# download still complete eval; baf.py prints the command that fetches it.
eval:
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
