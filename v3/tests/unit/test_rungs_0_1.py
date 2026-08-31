"""T-140 - Rung 0 floors and the Rung 1 static rule engine.

The ticket's Done-when is "all four floors and the rule engine score on the VALIDATION
split and produce complete EvalResult rows". Three of the four do, and the fourth cannot:
``all_hold`` alerts on the whole population, so ``alerts_per_day`` is the population size
and ``build_eval_result`` refuses - correctly - to compute metrics above capacity K. That
is asserted here rather than glossed, because a clause that cannot be met is worth more
written down than quietly dropped.

The split-level numbers live in ``docs/logbook/T-140.md``; this file asserts the machinery
that produces them, on a synthetic panel small enough to reason about.
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.eval.metrics import CostParams, PerfBudget, RungOutput, Truth, build_eval_result
from rakshak.eval.oracle import oracle_savings
from rakshak.features import registry
from rakshak.models import rung0_floors, rung1_rules
from rakshak.schemas import Action, EvalResult, TypologyId

K = 3
N_MERCHANTS = 40
N_DAYS = 10


def _panel(rng: np.random.Generator) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    """A synthetic ``(merchant-day x feature)`` block over the real register.

    The first four merchants are the drifters: every rule feature is pushed well past its
    threshold from day 4 onward, so a rule engine that works ranks them at the top and one
    that does not is visibly no better than random.
    """
    columns = tuple(registry.ORDER)
    merchants = np.array([f"M{i:03d}" for i in range(N_MERCHANTS)])
    merchant_id = np.repeat(merchants, N_DAYS)
    day = np.tile(np.arange(N_DAYS), N_MERCHANTS)
    x = rng.normal(0.0, 0.4, size=(N_MERCHANTS * N_DAYS, len(columns)))
    x[:, columns.index("p_declared_monthly_gmv")] = 600_000.0

    drifting = np.isin(merchant_id, merchants[:4]) & (day >= 4)
    for rule in rung1_rules.RULES:
        x[drifting, columns.index(rule.feature)] = rule.threshold + rule.scale
    return x, columns, merchant_id, day


def _truth(merchant_id: np.ndarray, rng: np.random.Generator) -> Truth:
    merchants = np.array(sorted(set(merchant_id.tolist())), dtype=object)
    label = np.array([1 if m in set(merchants[:4]) else 0 for m in merchants], dtype=np.int8)
    return Truth(
        merchant_id=merchants,
        label=label,
        is_censored=np.zeros(merchants.size, dtype=bool),
        loss_inr=np.where(label == 1, 40_000.0, 0.0),
        onset_day=np.where(label == 1, 4.0, np.nan),
        typology=np.where(label == 1, "R1", None),
        volume=rng.uniform(1e5, 1e7, size=merchants.size),
    )


def _row(score: np.ndarray, action: np.ndarray, merchant_id: np.ndarray, day: np.ndarray,
         truth: Truth, rung: int) -> EvalResult:
    from rakshak.eval.metrics import day_labels

    output = RungOutput(merchant_id=merchant_id, day=day, score=score, action=action)
    params = CostParams()
    y, keep = day_labels(output, truth)
    order = np.argsort(truth.merchant_id)
    idx = order[np.searchsorted(truth.merchant_id[order], output.merchant_id)]
    ceiling = oracle_savings(day[keep], y[keep], truth.loss_inr[idx][keep], K, params)
    return build_eval_result(
        rung=rung,
        split="val",
        output=output,
        truth=truth,
        k=K,
        params=params,
        rng=np.random.default_rng(7),
        perf=PerfBudget(p99_latency_ms=0.1, state_bytes_p99=4096.0, model_size_mb=0.0),
        oracle_savings=ceiling,
        eval_lock_sha="0" * 64,
        open_count=0,
        git_sha="test",
    )


# ── Rung 0 ───────────────────────────────────────────────────────────────────


def test_rank_normalise_preserves_order_and_lands_in_the_unit_interval() -> None:
    values = np.array([5.0, -2.0, 100.0, 0.0])
    ranks = rung0_floors.rank_normalise(values)
    assert ranks.min() == 0.0 and ranks.max() == 1.0
    assert np.array_equal(np.argsort(ranks), np.argsort(values))


def test_random_floor_is_reproducible_from_its_seed() -> None:
    """v1's headline finding was that random won on savings. A floor nobody can reproduce
    is a floor nobody can argue about."""
    a = rung0_floors.random_scores(50, np.random.default_rng(11))
    b = rung0_floors.random_scores(50, np.random.default_rng(11))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, rung0_floors.random_scores(50, np.random.default_rng(12)))


def test_floor_actions_never_exceed_capacity() -> None:
    rng = np.random.default_rng(3)
    _, _, merchant_id, day = _panel(rng)
    action = rung0_floors.floor_actions(rng.random(day.size), day, K)
    per_day = np.bincount(day, weights=(action != Action.PASS).astype(float))
    assert per_day.max() <= K
    assert set(np.unique(action)) <= {Action.PASS, Action.REVIEW}


@pytest.mark.parametrize("floor", rung0_floors.ROW_FLOORS)
def test_every_row_floor_produces_a_complete_eval_result(floor: str) -> None:
    """T-140's Done-when, for the three floors that can meet it."""
    rng = np.random.default_rng(5)
    _, _, merchant_id, day = _panel(rng)
    truth = _truth(merchant_id, rng)
    volume = dict(zip(truth.merchant_id, truth.volume, strict=True))

    if floor == "all_pass":
        score = np.zeros(day.size)
        action = rung0_floors.all_pass_actions(day.size)
    elif floor == "random_at_k":
        score = rung0_floors.random_scores(day.size, rng)
        action = rung0_floors.floor_actions(score, day, K)
    else:
        score = rung0_floors.volume_scores(np.array([volume[m] for m in merchant_id]))
        action = rung0_floors.floor_actions(score, day, K)

    result = _row(score, action, merchant_id, day, truth, rung=0)
    assert result.prevalence > 0.0
    assert result.alerts_per_day <= K + 1e-9
    # FR-021: all four floors on every row, without exception.
    assert not np.isnan(result.savings_floor_all_pass)
    assert not np.isnan(result.savings_floor_all_hold)
    assert not np.isnan(result.savings_floor_random)
    assert not np.isnan(result.savings_floor_volume_rank)
    # Per-typology recall is a required column, not a nice-to-have: a single aggregate
    # lets easy R1 hide hard R2 and R7, which is the v1 failure it exists to expose.
    assert set(result.recall_by_typology) == set(TypologyId)


def test_all_hold_cannot_be_a_row_because_it_ignores_capacity() -> None:
    """The fourth floor, and why the Done-when clause is met three-quarters of the way.

    ``all_hold`` is not a ranker. It alerts on every merchant every day, so the harness's
    own capacity assertion - the one that makes every other metric mean something - fires.
    Its savings is reported on every row as ``savings_floor_all_hold``, which is where the
    number the ticket wants actually lives.
    """
    rng = np.random.default_rng(5)
    _, _, merchant_id, day = _panel(rng)
    truth = _truth(merchant_id, rng)
    action = np.full(day.size, Action.HOLD, dtype=object)
    with pytest.raises(ValueError, match="exceeds capacity"):
        _row(np.full(day.size, 0.99), action, merchant_id, day, truth, rung=0)


# ── Rung 1 ───────────────────────────────────────────────────────────────────


def test_every_rule_names_a_registered_feature() -> None:
    """The rule engine and the model read the same register, or they are not comparable."""
    assert {rule.feature for rule in rung1_rules.RULES} <= set(registry.REGISTRY)


def test_rule_score_is_a_bounded_graded_quantity() -> None:
    rng = np.random.default_rng(9)
    x, columns, _, _ = _panel(rng)
    score = rung1_rules.score(x, columns)
    assert score.min() >= 0.0 and score.max() <= 1.0
    # Graded, not binary: a twelve-rule binary engine emits a handful of distinct values
    # and the top-K selection degenerates into a tiebreak on row order.
    assert np.unique(score).size > len(rung1_rules.RULES)


def test_rule_score_is_zero_below_every_threshold() -> None:
    columns = tuple(registry.ORDER)
    x = np.zeros((5, len(columns)))
    for rule in rung1_rules.RULES:
        x[:, columns.index(rule.feature)] = rule.threshold - 1e-6
    assert np.allclose(rung1_rules.score(x, columns), 0.0)


def test_rule_score_ranks_the_drifting_merchants_above_the_rest() -> None:
    rng = np.random.default_rng(13)
    x, columns, merchant_id, day = _panel(rng)
    score = rung1_rules.score(x, columns)
    drifting = np.isin(merchant_id, [f"M{i:03d}" for i in range(4)]) & (day >= 4)
    assert score[drifting].min() > score[~drifting].max()


def test_rule_engine_produces_a_complete_eval_result_and_detects_the_drifters() -> None:
    """T-140's Done-when for the rule engine, plus the reason that it is a rung at all."""
    rng = np.random.default_rng(17)
    x, columns, merchant_id, day = _panel(rng)
    truth = _truth(merchant_id, rng)
    score = rung1_rules.score(x, columns)
    action = rung0_floors.floor_actions(score, day, K)
    result = _row(score, action, merchant_id, day, truth, rung=1)
    assert result.alerts_per_day <= K + 1e-9
    assert result.recall_at_k > 0.0
    assert np.isfinite(result.ttd_median_days)


def test_fired_reasons_are_merchant_readable_and_at_most_three() -> None:
    """FR-033's audit trail for the rung that has no booster to ask."""
    rng = np.random.default_rng(19)
    x, columns, merchant_id, day = _panel(rng)
    row = int(np.flatnonzero((merchant_id == "M000") & (day == 9))[0])
    reasons = rung1_rules.fired_reasons(x, columns, row)
    assert len(reasons) == 3
    assert all(isinstance(text, str) and text.strip() for text in reasons)
    quiet = rung1_rules.fired_reasons(np.zeros((1, len(columns))), columns, 0)
    assert quiet == []
