"""Perfect-foresight ceilings against hand-computable fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.config import P_ANALYST_MISS, REVIEW_CAPACITY_HOURS, TAU_REVIEW_HOURS
from rakshak.eval.metrics import HOLD, PASS, REVIEW
from rakshak.eval.oracle import (
    perfect_hindsight_oracle,
    review_knapsack_oracle,
    review_slots,
)


def test_review_slots_is_floor_of_budget_over_tau() -> None:
    assert review_slots(1.0, 0.25) == 4
    assert review_slots(1.0, 0.3) == 3  # 3.33 -> floor
    assert review_slots(0.0, 0.25) == 0
    assert review_slots() == int(REVIEW_CAPACITY_HOURS // TAU_REVIEW_HOURS)


def test_knapsack_picks_the_largest_losses_first() -> None:
    # Budget for exactly 2 reviews. Bad merchants carry losses 500, 900, 100.
    # The oracle must take 900 and 500 and leave 100.
    y = np.array([1, 1, 1, 0])
    loss = np.array([500.0, 900.0, 100.0, 0.0])
    value = np.zeros(4)
    result = review_knapsack_oracle(y, loss, value, capacity_hours=2.0, tau_hours=1.0)
    assert list(result.actions) == [REVIEW, REVIEW, PASS, PASS]
    assert result.n_reviewed == 2
    assert result.hours_used == pytest.approx(2.0)
    assert result.loss_averted_inr == pytest.approx((1.0 - P_ANALYST_MISS) * 1400.0)
    assert result.capacity_binding is True


def test_knapsack_never_spends_capacity_on_a_healthy_merchant() -> None:
    """Budget of 5 against 2 bad merchants: the oracle reviews 2, not 5."""
    y = np.array([1, 1, 0, 0, 0, 0])
    loss = np.array([100.0, 200.0, 0.0, 0.0, 0.0, 0.0])
    result = review_knapsack_oracle(
        y, loss, np.zeros(6), capacity_hours=5.0, tau_hours=1.0
    )
    assert result.n_reviewed == 2
    assert set(np.flatnonzero(result.actions == REVIEW)) == {0, 1}
    assert result.capacity_binding is False


def test_knapsack_is_deterministic_under_ties() -> None:
    y = np.ones(4)
    loss = np.full(4, 100.0)
    a = review_knapsack_oracle(y, loss, np.zeros(4), capacity_hours=2.0, tau_hours=1.0)
    b = review_knapsack_oracle(y, loss, np.zeros(4), capacity_hours=2.0, tau_hours=1.0)
    assert list(a.actions) == list(b.actions) == [REVIEW, REVIEW, PASS, PASS]


def test_hindsight_oracle_picks_the_cheapest_action_per_merchant() -> None:
    # Bad merchant, L = 10_000: PASS 10_000, REVIEW ~1540, HOLD 1000 -> HOLD.
    # Healthy merchant, V = 0:   PASS 0, REVIEW ~40, HOLD 500       -> PASS.
    y = np.array([1, 0])
    loss = np.array([10_000.0, 0.0])
    result = perfect_hindsight_oracle(y, loss, np.zeros(2))
    assert list(result.actions) == [HOLD, PASS]
    assert result.n_held == 1
    assert result.n_reviewed == 0
    assert result.hours_used == 0.0
    assert result.capacity_binding is False


def test_hindsight_oracle_dominates_the_knapsack_on_savings() -> None:
    """The stated reason the second ceiling exists: the review budget constrains
    REVIEW only, so the knapsack is not the highest-savings hindsight policy."""
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) < 0.2).astype(float)
    loss = y * rng.uniform(5_000, 50_000, n)
    value = rng.uniform(1_000, 20_000, n)
    knapsack = review_knapsack_oracle(y, loss, value, capacity_hours=2.0, tau_hours=1.0)
    hindsight = perfect_hindsight_oracle(y, loss, value)
    assert hindsight.savings >= knapsack.savings


def test_hindsight_savings_are_never_negative() -> None:
    """It picks per-merchant argmin cost, so it cannot lose to a constant policy."""
    rng = np.random.default_rng(1)
    n = 100
    y = (rng.random(n) < 0.3).astype(float)
    loss = y * rng.uniform(100, 10_000, n)
    value = rng.uniform(0, 5_000, n)
    assert perfect_hindsight_oracle(y, loss, value).savings >= -1e-12


def test_oracle_handles_a_split_with_no_bad_merchants() -> None:
    y = np.zeros(5)
    result = review_knapsack_oracle(y, np.zeros(5), np.zeros(5))
    assert result.n_reviewed == 0
    assert result.loss_averted_inr == 0.0
    assert result.capacity_binding is False
