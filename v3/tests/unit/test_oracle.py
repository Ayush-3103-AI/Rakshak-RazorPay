"""T-132 — the perfect-foresight oracle and the leakage assertion.

The named test is ``test_a_rung_beating_the_oracle_is_leakage``. Everything else here
exists to make sure the ceiling it asserts against is a real ceiling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rakshak.eval.capacity import ActionPolicy, select_actions
from rakshak.eval.metrics import CostParams, savings_of_actions
from rakshak.eval.oracle import (
    LeakageError,
    assert_no_leakage,
    gap_to_oracle,
    intervention_benefit,
    knapsack,
    oracle_actions,
    oracle_savings,
)
from rakshak.schemas import Action

PARAMS = CostParams()
K = 50  # charter §10.4: 50 reviews/day per 10,000 merchants. Load-bearing.


def world(n_merchants: int = 400, n_days: int = 20, prevalence: float = 0.015, seed: int = 3):
    """A merchant-day grid at roughly the real prevalence, with instance-dependent losses."""
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_merchants * prevalence)))
    merchant_is_fraud = np.zeros(n_merchants, dtype=bool)
    merchant_is_fraud[:n_fraud] = True
    onset = rng.integers(0, n_days // 2, size=n_merchants)
    merchant_loss = np.where(merchant_is_fraud, rng.uniform(30_000, 400_000, n_merchants), 0.0)

    day = np.tile(np.arange(n_days), n_merchants)
    who = np.repeat(np.arange(n_merchants), n_days)
    y = (merchant_is_fraud[who] & (day >= onset[who])).astype(np.int8)
    loss = merchant_loss[who]
    return day, y, loss, who, rng


# ─────────────────────────── the knapsack ───────────────────────────


def test_knapsack_solves_a_hand_checked_instance() -> None:
    # The textbook 0/1 instance. Total weight is 6, so capacity 5 cannot take everything:
    # items 2+3 (weight 5, value 220) beat 1+3 (weight 4, value 180) and 1+2 (3, 160).
    values = np.array([60.0, 100.0, 120.0])
    weights = np.array([1, 2, 3])
    assert knapsack(values, weights, 6).tolist() == [True, True, True]
    assert knapsack(values, weights, 5).tolist() == [False, True, True]
    # capacity 4 cannot fit 2+3 (weight 5), so 1+3 (weight 4, value 180) wins
    assert knapsack(values, weights, 4).tolist() == [True, False, True]


def test_knapsack_never_takes_a_negative_value_item() -> None:
    """Spending analyst capacity to lose money is not an optimum the oracle may find."""
    mask = knapsack(np.array([-5.0, 10.0]), np.array([1, 1]), 5)
    assert mask.tolist() == [False, True]


def test_knapsack_respects_the_budget() -> None:
    rng = np.random.default_rng(1)
    values = rng.uniform(1, 100, 40)
    weights = rng.integers(1, 6, 40)
    mask = knapsack(values, weights, 12)
    assert weights[mask].sum() <= 12


def test_knapsack_with_unit_weights_equals_top_k() -> None:
    """The uniform-cost shortcut in oracle_actions is an identity, not an approximation."""
    rng = np.random.default_rng(2)
    values = rng.uniform(1, 100, 30)
    weights = np.ones(30, dtype=int)
    mask = knapsack(values, weights, 7)
    assert mask.sum() == 7
    assert set(np.flatnonzero(mask)) == set(np.argsort(-values)[:7])


def test_knapsack_refuses_a_negative_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be >= 0"):
        knapsack(np.array([1.0]), np.array([1]), -1)


# ─────────────────────────── the ceiling ───────────────────────────


def test_the_oracle_only_touches_fraud() -> None:
    """A clean merchant is never worth a review, even with capacity to spare."""
    day, y, loss, _, _ = world(n_merchants=200, n_days=5)
    actions = oracle_actions(day, y, loss, K, PARAMS)
    assert ((actions != Action.PASS) <= (y == 1)).all()


def test_intervention_benefit_is_negative_for_a_good_merchant() -> None:
    benefit, best = intervention_benefit(
        np.array([0, 1]), np.array([0.0, 100_000.0]), PARAMS
    )
    assert benefit[0] == pytest.approx(-PARAMS.review_cost_inr)
    assert benefit[1] == pytest.approx(100_000.0 - PARAMS.review_cost_inr)
    assert best[1] == Action.HOLD  # HOLD stops a known fraud outright


def test_oracle_respects_capacity() -> None:
    day, y, loss, _, _ = world(n_merchants=2000, n_days=10, prevalence=0.20)
    actions = oracle_actions(day, y, loss, 5, PARAMS)
    for d in np.unique(day):
        assert (actions[day == d] != Action.PASS).sum() <= 5


def test_oracle_with_analyst_hour_weights_agrees_with_the_uniform_path() -> None:
    day, y, loss, _, _ = world(n_merchants=300, n_days=6)
    uniform = oracle_actions(day, y, loss, 4, PARAMS)
    weighted = oracle_actions(day, y, loss, 4, PARAMS, weights=np.ones(day.size, dtype=int))
    assert savings_of_actions(uniform, y, loss, PARAMS) == pytest.approx(
        savings_of_actions(weighted, y, loss, PARAMS)
    )


def test_expensive_reviews_force_the_oracle_to_choose() -> None:
    """With 2-hour reviews and a 3-hour budget, only one of two frauds fits."""
    day = np.zeros(2, dtype=int)
    y = np.ones(2, dtype=np.int8)
    loss = np.array([50_000.0, 90_000.0])
    actions = oracle_actions(day, y, loss, 3, PARAMS, weights=np.array([2, 2]))
    assert (actions != Action.PASS).sum() == 1
    assert actions[1] != Action.PASS, "the oracle left the larger loss on the table"


# ─────────────────────────── the assertion that matters ───────────────────────────


def test_a_rung_beating_the_oracle_is_leakage() -> None:
    """T-132's named test. The oracle is given the labels; nothing that is not can win."""
    with pytest.raises(LeakageError, match="beats the perfect-foresight oracle"):
        assert_no_leakage(0.81, 0.80, label="rung 3")


def test_leakage_check_passes_when_the_rung_is_below_the_ceiling() -> None:
    assert_no_leakage(0.79, 0.80)
    assert_no_leakage(0.80, 0.80)  # exactly at the ceiling is legal, if improbable


def test_leakage_check_is_silent_on_unmeasurable_windows() -> None:
    assert_no_leakage(float("nan"), 0.5)
    assert_no_leakage(0.5, float("nan"))


def test_no_realistic_rung_beats_the_oracle() -> None:
    """The end-to-end version of the assertion, over the real decision layer at K=50."""
    day, y, loss, who, rng = world(n_merchants=1000, n_days=15, prevalence=0.015, seed=5)
    ceiling = oracle_savings(day, y, loss, K, PARAMS)
    exposure = np.full(day.size, float(loss[y == 1].mean()))

    rungs = {
        "random": rng.random(day.size),
        "noisy_but_good": np.clip(0.85 * y + rng.normal(0, 0.15, day.size), 0, 1),
        "constant": np.full(day.size, 0.5),
    }
    for name, score in rungs.items():
        actions = select_actions(score, day, exposure, K, PARAMS)
        savings = savings_of_actions(actions, y, loss, PARAMS)
        assert_no_leakage(savings, ceiling, label=name)
        assert alerts_per_day_ok(actions, day, K), name


def alerts_per_day_ok(actions: np.ndarray, day: np.ndarray, k: int) -> bool:
    return all((actions[day == d] != Action.PASS).sum() <= k for d in np.unique(day))


def test_the_oracle_is_at_least_as_good_as_any_ranking_it_could_have_chosen() -> None:
    day, y, loss, _, rng = world(n_merchants=600, n_days=10, prevalence=0.05, seed=9)
    ceiling = oracle_savings(day, y, loss, 8, PARAMS)
    for seed in range(5):
        r = np.random.default_rng(seed)
        exposure = np.full(day.size, 100_000.0)
        actions = select_actions(r.random(day.size), day, exposure, 8, PARAMS, ActionPolicy())
        assert savings_of_actions(actions, y, loss, PARAMS) <= ceiling + 1e-9


# ─────────────────────────── gap to oracle ───────────────────────────


def test_gap_to_oracle_is_zero_when_the_rung_reaches_the_ceiling() -> None:
    assert gap_to_oracle(0.8, 0.8) == pytest.approx(0.0)
    assert gap_to_oracle(0.4, 0.8) == pytest.approx(0.5)


def test_gap_to_oracle_is_nan_rather_than_a_division_by_zero() -> None:
    assert math.isnan(gap_to_oracle(0.4, 0.0))
    assert math.isnan(gap_to_oracle(0.4, float("nan")))


def test_oracle_savings_is_positive_at_realistic_prevalence() -> None:
    """If the ceiling itself is <= 0 at K=50 and 1.5% prevalence, the cost block is wrong
    and every gap-to-oracle downstream is meaningless."""
    day, y, loss, _, _ = world(n_merchants=10_000 // 25, n_days=10, prevalence=0.015)
    assert oracle_savings(day, y, loss, K, PARAMS) > 0.0
