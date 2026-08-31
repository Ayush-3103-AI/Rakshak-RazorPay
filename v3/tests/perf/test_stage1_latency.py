"""NFR-02 — stage-1 full scoring, p99 <= 10 ms per merchant-epoch, one core.

Stage 1 runs on the tenth of the population stage 0 promoted, so its budget is twenty
times looser than stage 0's for a tenth of the calls. It covers the whole read: T1 + T2 +
the cohort residual + the booster's forward pass.

The feature half is asserted unconditionally. The booster half needs a trained Rung 2 —
T-142, Lane D — and skips with that named until `make train RUNG=2` has run. Never a
silent pass: a skipped budget prints its reason and the reason names what unblocks it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from perf_budgets import assert_budget, measure, needs_trained_rung

from rakshak.features import registry
from rakshak.features.cascade import STAGE1_BUDGET_MS, Cascade
from rakshak.features.state import MerchantState


def test_stage1_vector_matches_the_trained_column_order(cascade: Cascade) -> None:
    """Stage 0's vector concatenated with stage 1's tail must be ``registry.ORDER``.

    09-interfaces.md §9: a model trained on one column order and scored on another fails
    silently. `Cascade.from_registry` refuses to build if this stops holding; this asserts
    the same thing from the outside, on the vector rather than on the spec list.
    """
    assert tuple(s.name for s in (*cascade.t1, *cascade.t2)) == registry.ORDER


def test_stage1_feature_read_latency(
    cascade: Cascade, warm: tuple[MerchantState, datetime]
) -> None:
    """The feature half of stage 1, with no model. Runs on a clean clone."""
    state, as_of = warm
    head = cascade.stage0(state, as_of)

    def once() -> None:
        cascade.stage1(state, as_of, t1_values=head)

    timing = measure(once, iterations=1000, batches=5, warmup=300)
    print(f"\nNFR-02 stage-1 features only: {timing.line()}")
    assert_budget(
        "NFR-02-features", "stage-1 T1+T2 read p99", timing.p99_ms, STAGE1_BUDGET_MS, "ms"
    )


@needs_trained_rung
def test_stage1_scoring_latency(
    cascade: Cascade,
    warm: tuple[MerchantState, datetime],
    booster: tuple[Any, tuple[str, ...]],
) -> None:
    """The whole of stage 1: read every feature, then score one row through the booster.

    One row, not a batch, because NFR-02 is quoted per merchant-epoch and a batched
    forward pass would report the amortised number for a cost the cascade pays per
    merchant. The batched figure is the flattering one and it is not the budget.
    """
    state, as_of = warm
    model, columns = booster
    assert columns == registry.ORDER, (
        f"the trained booster's columns are not registry.ORDER, so this measurement is of "
        f"a different model than the cascade would serve. trained={columns}"
    )

    def once() -> None:
        row = cascade.stage1(state, as_of).reshape(1, -1)
        model.predict(row)

    timing = measure(once, iterations=300, batches=5, warmup=100)
    print(f"\nNFR-02 stage-1 read + score: {timing.line()}")
    assert_budget("NFR-02", "stage-1 read + score p99", timing.p99_ms, STAGE1_BUDGET_MS, "ms")


@needs_trained_rung
def test_the_booster_scores_the_cascades_vector_without_realignment(
    cascade: Cascade,
    warm: tuple[MerchantState, datetime],
    booster: tuple[Any, tuple[str, ...]],
) -> None:
    """A latency number for a vector the model cannot actually score is worth nothing."""
    state, as_of = warm
    model, _ = booster
    score = model.predict(cascade.stage1(state, as_of).reshape(1, -1))
    assert np.isfinite(score).all()
    assert 0.0 <= float(score[0]) <= 1.0
