"""Stage 2 — ``pred_contrib`` reason codes, <= 50 ms per non-``PASS`` decision.

From CLAUDE.md's cascade table. It has no NFR number of its own because it is the one
stage bounded by analyst capacity K rather than by population: at K = 50 reviews per day
per 10,000 merchants, 50 ms x 50 is 2.5 s of a 30 s sweep, and a fiftieth of the daily
explain cost of computing contributions for everyone.

Needs a trained Rung 2 — T-142, Lane D. Skips with that named until one exists.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from perf_budgets import assert_budget, measure, needs_trained_rung

from rakshak.features.cascade import STAGE2_BUDGET_MS, Cascade, stage2_rows
from rakshak.features.state import MerchantState
from rakshak.schemas import Action


def test_stage2_explains_only_the_non_pass_decisions() -> None:
    """The routing rule, which is what makes the 50 ms affordable at all."""
    actions = [Action.PASS, Action.REVIEW, Action.PASS, Action.HOLD]
    assert stage2_rows(actions).tolist() == [1, 3]
    assert stage2_rows([Action.PASS] * 100).size == 0


@needs_trained_rung
def test_stage2_explain_latency(
    cascade: Cascade,
    warm: tuple[MerchantState, datetime],
    booster: tuple[Any, tuple[str, ...]],
) -> None:
    """One decision's reason codes: ``pred_contrib`` plus the top-3 selection (FR-014)."""
    state, as_of = warm
    model, columns = booster
    row = cascade.stage1(state, as_of).reshape(1, -1)

    def once() -> None:
        contrib = np.asarray(model.predict(row, pred_contrib=True), dtype=np.float64)[:, :-1]
        order = np.argsort(-np.abs(contrib), axis=1)[:, :3]
        [columns[j] for j in order[0]]

    timing = measure(once, iterations=200, batches=5, warmup=50)
    print(f"\nstage-2 explain: {timing.line()}")
    assert_budget("STAGE2", "stage-2 explain p99", timing.p99_ms, STAGE2_BUDGET_MS, "ms")


@needs_trained_rung
def test_stage2_produces_three_distinct_reason_codes(
    cascade: Cascade,
    warm: tuple[MerchantState, datetime],
    booster: tuple[Any, tuple[str, ...]],
) -> None:
    """``Decision.__post_init__`` requires exactly three on a non-``PASS``, so three it is.

    Timing an explain path that cannot produce a shippable reason code would be timing the
    wrong thing.
    """
    state, as_of = warm
    model, columns = booster
    row = cascade.stage1(state, as_of).reshape(1, -1)
    contrib = np.asarray(model.predict(row, pred_contrib=True), dtype=np.float64)[:, :-1]
    top = np.argsort(-np.abs(contrib), axis=1)[0, :3]
    names = [columns[j] for j in top]
    assert len(set(names)) == 3

    from rakshak.features import registry

    for j, name in zip(top, names, strict=True):
        assert registry.get(name).explain(float(row[0, j])), f"{name} has no reason sentence"
