"""NFR-03 — a full daily sweep of 10,000 merchants in <= 30 s on 4 cores.

**This is the budget the cascade exists to buy**, and it is the ticket's `Done when`.

The sweep is measured in the shape a deployment would run it: chunks of merchants loaded
from their packed state, the day's events folded in, stage 0 over everyone in the chunk,
stage 1 over that chunk's top decile. Chunking is not a testing convenience — 10,000 live
``MerchantState`` objects is several hundred megabytes of Python objects and no server
would hold them all, so the per-chunk decile stands in for a global one. The two differ
only in *which* merchants are promoted, never in how many, and NFR-03 is a compute budget.

Measured single-threaded and asserted against the 4-core budget, which is strictly
conservative: nothing here is parallel, so a real 4-core run can only be faster.

The counterfactual is printed alongside, because it is the argument: a single-stage design
that reads all 28 features for all 10,000 merchants, which is what the register would cost
without the cascade.
"""

from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timedelta

import numpy as np
import pytest
from perf_budgets import assert_budget

from rakshak.features.cascade import SWEEP_BUDGET_S, SWEEP_POPULATION, Cascade
from rakshak.features.state import MerchantState
from rakshak.schemas import MerchantProfile, Transaction

#: 14.82M transactions over 10,000 merchants x 180 days, from STATE.md's generated run.
EVENTS_PER_MERCHANT_PER_DAY = 8
#: Merchants held live at once. Bounds peak memory; see the module docstring.
CHUNK = 500


def _clone(blob: bytes, merchant_id: str) -> MerchantState:
    state = MerchantState.unpack(blob)
    state.merchant_id = merchant_id
    return state


def test_the_cascade_promotes_the_declared_fraction(cascade: Cascade) -> None:
    """Stage 0 promotes 10% — the number the whole compute argument rests on."""
    screen = np.arange(1000, dtype=np.float64)
    promoted = cascade.promote(screen)
    assert promoted.size == 100
    # Ranked best-first, and the best is the largest screen statistic.
    assert promoted[0] == 999
    assert set(promoted.tolist()) == set(range(900, 1000))
    # A population smaller than the fraction still promotes someone rather than nobody.
    assert cascade.promote(np.array([1.0, 2.0])).size == 1
    assert cascade.promote(np.empty(0)).size == 0


def test_a_small_sweep_reaches_the_stages_it_claims(
    cascade: Cascade, warm: tuple[MerchantState, datetime]
) -> None:
    """Cheap enough for the fast loop, and it is what checks the sweep's bookkeeping."""
    state, as_of = warm
    blob = state.pack()
    states = [_clone(blob, f"M{i:05d}") for i in range(50)]
    # Otherwise every screen statistic is identical and `promote` is measuring argsort ties.
    for i, s in enumerate(states):
        s.baseline.mean += i

    result = cascade.sweep(states, as_of)
    assert result.promoted.size == 5
    assert result.features.shape == (5, 28)
    reached = result.stage_reached()
    assert reached.sum() == 5
    assert set(np.flatnonzero(reached).tolist()) == set(result.promoted.tolist())

    vectors = result.vectors()
    assert len(vectors) == 5
    assert all(v.stage_reached == 1 for v in vectors)
    assert all(v.values.dtype == np.float64 for v in vectors)
    # Stage 1's vector is stage 0's vector plus the T2 tail, not a re-read of it.
    head = cascade.stage0(states[result.promoted[0]], as_of)
    assert np.array_equal(result.features[0, : len(cascade.t1)], head)


@pytest.mark.slow
def test_full_daily_sweep_of_ten_thousand_merchants(
    cascade: Cascade,
    warm: tuple[MerchantState, datetime],
    stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    txns, _ = stream
    state, warm_as_of = warm
    blob = state.pack()

    # Tomorrow's events, so folding them is a forward day-roll rather than an out-of-order
    # replay into a state that has already closed that day.
    as_of = warm_as_of + timedelta(days=1)
    source = [t for t in txns if t.merchant_id == "M000"][:EVENTS_PER_MERCHANT_PER_DAY]
    assert len(source) == EVENTS_PER_MERCHANT_PER_DAY, "stream fixture is too sparse to sweep"
    events = [
        dataclasses.replace(
            t, event_time=as_of.replace(hour=12, minute=i), event_date=as_of.date()
        )
        for i, t in enumerate(source)
    ]

    specs = [*cascade.t1, *cascade.t2]
    load_s = ingest_s = 0.0
    sweep_s = 0.0
    promoted_total = 0

    for start in range(0, SWEEP_POPULATION, CHUNK):
        ids = [f"M{i:05d}" for i in range(start, min(start + CHUNK, SWEEP_POPULATION))]

        t0 = time.perf_counter()
        states = [_clone(blob, mid) for mid in ids]
        load_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        for merchant in states:
            for event in events:
                for spec in specs:
                    spec.update(spec.state_of(merchant), event)
        ingest_s += time.perf_counter() - t0

        result = cascade.sweep(states, as_of)
        sweep_s += result.seconds
        promoted_total += int(result.promoted.size)

    total = load_s + ingest_s + sweep_s
    n_events = SWEEP_POPULATION * EVENTS_PER_MERCHANT_PER_DAY

    # The counterfactual, measured rather than inferred: one chunk read at stage-1 depth,
    # extrapolated to the population. This is what the register costs with no cascade, and
    # it is the comparison the design decision rests on.
    counterfactual = [_clone(blob, f"C{i:05d}") for i in range(CHUNK)]
    t0 = time.perf_counter()
    for merchant in counterfactual:
        cascade.stage1(merchant, as_of)
    single_stage_s = (time.perf_counter() - t0) * SWEEP_POPULATION / CHUNK

    print(
        f"\nNFR-03 daily sweep, {SWEEP_POPULATION:,} merchants, {n_events:,} events, "
        f"single-threaded:"
        f"\n  load packed state {load_s:7.2f} s   "
        f"({1000 * load_s / SWEEP_POPULATION:.3f} ms/merchant)"
        f"\n  ingest events     {ingest_s:7.2f} s   ({1000 * ingest_s / n_events:.4f} ms/event)"
        f"\n  stage 0 + stage 1 {sweep_s:7.2f} s   ({promoted_total:,} promoted to stage 1)"
        f"\n  total             {total:7.2f} s   against a {SWEEP_BUDGET_S} s budget"
        f"\n  no cascade: stage-1 depth for all 10,000 would be "
        f"{single_stage_s:.1f} s of feature reads against the {sweep_s:.1f} s above"
    )

    assert promoted_total == SWEEP_POPULATION // 10
    assert_budget("NFR-03", "full daily sweep of 10,000 merchants", total, SWEEP_BUDGET_S, "s")
