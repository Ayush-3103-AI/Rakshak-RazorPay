"""Pytest wiring for the perf suite. The measurement machinery is in ``perf_budgets.py``.

Only fixtures and hooks live here — everything importable is next door, because
``pythonpath`` puts ``tests/parity`` on the path and a bare ``import conftest`` from any
suite resolves to *that* one. See ``perf_budgets``' module docstring.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pytest
from parity_harness import end_of_day, synthetic_stream
from perf_budgets import SEED, STREAM_DAYS, rung_artifacts, write_measurements

from rakshak.features import registry
from rakshak.features.cascade import Cascade
from rakshak.features.state import MerchantState
from rakshak.schemas import MerchantProfile, Transaction


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Hand the measurements to ``eval.metrics.PerfBudget``, which refuses to invent them."""
    write_measurements()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures. Session-scoped: warming a state costs a full stream replay, and every budget
# wants the same warm state rather than its own.
#
# The stream is synthetic rather than the generated dataset on purpose — `make all` runs
# from a clean clone where data/v2 does not exist yet, and a perf gate that only runs after
# `make gen` is a perf gate that does not run in CI. The state's *size* is
# volume-independent (every buffer in the layer is bounded by a day count or a bin count,
# never by an event count), so a synthetic stream measures the same steady state a
# ten-thousand-merchant run does. The latency figures are per merchant-epoch and likewise do
# not scale with population.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def stream() -> tuple[list[Transaction], dict[str, MerchantProfile]]:
    # Explicitly seeded and threaded, like every other stochastic call in the repo. The
    # session scope is why this does not use the `rng` fixture, which is function-scoped.
    return synthetic_stream(
        np.random.default_rng(SEED), merchants=4, days=STREAM_DAYS, max_per_day=12
    )


@pytest.fixture(scope="session")
def cascade() -> Cascade:
    return Cascade.from_registry()


@pytest.fixture(scope="session")
def warm(
    stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> tuple[MerchantState, datetime]:
    """The busiest merchant in the stream, replayed to steady state, and the epoch to read.

    M000 transacts every day, so its rings are full — which is the state that costs the most
    to serialize and the most to read, and therefore the one the budgets are owed against.
    """
    txns, profiles = stream
    state = MerchantState(merchant_id="M000", profile=profiles["M000"])
    specs = [registry.REGISTRY[name] for name in registry.ORDER]
    ordered = sorted(txns, key=lambda t: (t.event_time, t.event_id))
    days: list[date] = sorted({t.event_date for t in txns})
    cursor = 0
    for day in days:
        as_of = end_of_day(day)
        while cursor < len(ordered) and ordered[cursor].event_time <= as_of:
            event = ordered[cursor]
            if event.merchant_id == "M000":
                for spec in specs:
                    spec.update(spec.state_of(state), event)
            cursor += 1
        for spec in specs:
            spec.value(spec.state_of(state), as_of)
    return state, end_of_day(days[-1])


# ─────────────────────────────────────────────────────────────────────────────
# The Lane D dependency, handled rather than waited on.
#
# T-150 depends on T-142. Everything that does not need a trained booster — state size,
# feature read latency, the cascade's stage-0 and stage-1 mechanics, the sweep — is asserted
# unconditionally. Everything that does skips with the dependency named, and runs for real
# the moment `make train` has written a booster. Never a silent pass.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def booster() -> tuple[Any, tuple[str, ...]]:
    """The Rung 2 booster and its trained column order, loaded from disk.

    Imported inside the fixture, never at module scope: ``rakshak.models`` is Lane D's, and a
    module-level import would break collection of this whole suite while those files are
    mid-edit.
    """
    import lightgbm as lgb

    model_path, sidecar = rung_artifacts()
    if not model_path.exists():
        pytest.skip(f"no trained booster at {model_path} (T-142, Lane D)")
    columns = tuple(json.loads(sidecar.read_text(encoding="utf-8"))["columns"])
    return lgb.Booster(model_file=str(model_path)), columns
