# Rakshak v2 — every workflow is a target here. This file is the single source of truth
# for the commands; make.ps1 parses it for Windows rather than restating them, because
# two copies of a command list is one copy too many (v1 shipped two and they drifted).
#
# `make all` must pass from a clean git clone on a fresh env. CI enforces exactly that.

SEED   ?= 42
RUNG   ?= 2
PY     ?= uv run python
PYTEST ?= uv run pytest

.PHONY: gen features gates parity perf train eval report test lint all clean

gen:            ## generate the v2 dataset from configs/scenario_v2.yaml
	$(PY) -m rakshak.cli gen --config configs/scenario_v2.yaml --seed $(SEED)

features:       ## materialise FeatureVectors for the generated stream
	$(PY) -m rakshak.cli features --seed $(SEED)

gates:          ## G1-G5 generator parity gates. MUST be green before any model trains.
	$(PYTEST) tests/gates -v

parity:         ## online vs offline feature agreement, <=1e-9 (NFR-08)
	$(PYTEST) tests/parity -v

perf:           ## latency / memory budget assertions (NFR-01..05)
	$(PYTEST) tests/perf -v

train:          ## train one rung on train+val ONLY
	$(PY) -m rakshak.cli train --rung $(RUNG) --seed $(SEED)

# Refuses to touch the test split unless RAKSHAK_UNLOCK=1. Do not set that before T-151.
eval:
	$(PY) -m rakshak.cli eval --rung $(RUNG) --seed $(SEED)

report:         ## regenerate the results table into docs/
	$(PY) -m rakshak.cli report

test:
	$(PYTEST)

lint:
	uv run ruff check src tests
	uv run mypy src

# The pipeline grows a stage as each lane lands, so a clean-clone CI run is always testing
# everything that currently exists and never a stale subset.
#
# `parity` sits before `gates` because it is fast and needs no dataset, so a broken feature
# runner fails in seconds rather than after `gen` has spent a minute writing 366 MB.
# `gates` depends on `gen` — G1 and G3 read the real dataset, not a fixture.
#
# Still to be appended: `features` (needs a cli `features` subcommand — only `gen` exists
# today, so putting it here would break the clean-clone job) and `perf` at T-150.
all: lint parity gen gates test

clean:
	rm -rf data/v2 .pytest_cache .mypy_cache .ruff_cache
