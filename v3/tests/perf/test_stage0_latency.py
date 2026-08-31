"""NFR-01 — stage-0 screen, p99 <= 0.5 ms per merchant-epoch, one core.

Stage 0 is the only stage that runs on every merchant every day, so it is the only stage
whose latency multiplies by the population. 0.5 ms x 10,000 merchants is 5 s of the 30 s
sweep budget (NFR-03) — which is why this is the tightest budget in the suite and the one
that decides how many T1 features the register can hold.

Measured on one core, and no threads are involved: every reader here is a scalar fold over
a bounded state object.
"""

from __future__ import annotations

from datetime import datetime

from perf_budgets import assert_budget, measure

from rakshak.features.cascade import STAGE0_BUDGET_MS, Cascade
from rakshak.features.state import MerchantState


def test_stage0_reads_only_tier1(cascade: Cascade) -> None:
    """The budget is meaningless if stage 0 quietly reads a T2 column.

    T2's four divergences are three quarters of a full read; a stage 0 that touched one
    would still pass on a fast day and blow the sweep on a busy one.
    """
    assert len(cascade.t1) == 24
    assert len(cascade.t2) == 4
    assert all(spec.tier.name == "T1" for spec in cascade.t1)


def test_stage0_screen_latency(cascade: Cascade, warm: tuple[MerchantState, datetime]) -> None:
    state, as_of = warm

    def once() -> None:
        values = cascade.stage0(state, as_of)
        cascade.screen(values)

    # Seven batches rather than the default five: stage 0 has the tightest budget in the
    # suite (2x margin) and this tree is built with several agents on one machine, so the
    # minimum needs more chances to land on an uncontended one. Widening the batch count is
    # a measurement choice; widening the budget would not be.
    timing = measure(once, iterations=2000, batches=7, warmup=500)
    print(f"\nNFR-01 stage-0: {timing.line()}")
    assert_budget("NFR-01", "stage-0 screen p99", timing.p99_ms, STAGE0_BUDGET_MS, "ms")
