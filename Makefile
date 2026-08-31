# Rakshak — canonical entry points. `make eval` must stay under 15 min (NFR).
# On Windows without GNU make, use ./make.ps1 <target> — same commands.

SEED ?= 42
# T-0021: a stock Linux box (this Makefile's actual target — GNU make isn't
# even installed on the Windows dev machine) has no `python` on PATH, only
# `python3` — `make setup` failed at word one on a clean checkout. Override
# with `make PY=python ...` on a system where only `python` exists.
PY   ?= python3

# T-0021: the hand-written Baum-Welch fit sums in an order that depends on how
# many threads the BLAS backend spawns, so `ablations.md`'s two HMM-refit rows
# and the two sensitivity PNGs drift from the committed baseline on a machine
# with a different core count — deterministic on any one box, not across boxes.
# Pinning every BLAS backend to a single thread is what makes the committed
# artifacts byte-reproducible on someone else's machine, which is the whole
# claim `make eval` exists to back. Costs wall-clock; eval still runs in ~5 min
# against NFR-004's 15-minute budget.
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1
export MKL_NUM_THREADS = 1
export NUMEXPR_NUM_THREADS = 1

.PHONY: setup eval baf blackswan profile figures test lint

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
#
# T-0021: three committed `results/` artifacts are deliberately NOT in this
# chain, because each needs an input `make eval` cannot produce on its own.
# They are regenerable, just by a named command rather than by `eval`:
#   blackswan.md              -> `make blackswan` (needs a second, shocked
#                                dataset generated first; the target does both)
#   calibration_gap.md,       -> `make profile` (needs Online Retail II, an
#   calibration_profile.json     external CC BY 4.0 download, git-ignored)
# Stated here rather than left implicit so "every number is regenerable" stays
# a checkable claim rather than one that quietly skips four files.
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

# T-0022c. Generates its own shocked dataset first — blackswan.py raises with
# this exact command if `data/synthetic_shock/` is missing, so the two steps
# are kept together here rather than leaving a trap for the next caller.
blackswan:
	$(PY) -m rakshak.generator.generate --seed $(SEED) --shock-day 194 --shock-magnitude 6.0
	$(PY) -m rakshak.eval.blackswan --seed $(SEED)

# T-0015. Needs data/external/online_retail_ii.parquet, fetched by
# `$(PY) -m rakshak.data.download --dataset online_retail_ii` (CC BY 4.0,
# git-ignored). Not chained into eval: a clean checkout has no external data.
profile:
	$(PY) -m rakshak.data.profile --seed $(SEED)

figures:
	$(PY) -m rakshak.eval.harness --seed $(SEED) --figures-only

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
