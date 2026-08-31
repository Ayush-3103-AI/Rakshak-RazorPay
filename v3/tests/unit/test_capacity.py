"""T-132 — the capacity-constrained action selector and the cost-asymmetry sweep.

Two properties the spec names explicitly: alerts never exceed K, and the selection is
stable under small score perturbations (which is what feeds ``alert_jaccard_wow``).
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.eval.capacity import (
    ASYMMETRY_RATIOS,
    ActionPolicy,
    expected_costs,
    select_actions,
    sweep_cost_asymmetry,
)
from rakshak.eval.metrics import CostParams, savings_of_actions
from rakshak.schemas import Action

PARAMS = CostParams()
#: charter §10.4 — 50 reviews/day per 10,000 merchants (0.5%). A wrong K changes the
#: ranking of rungs, not just their scores, so it is named at every use.
K = 50


def grid(n_merchants: int = 1000, n_days: int = 20) -> tuple[np.ndarray, np.ndarray]:
    day = np.tile(np.arange(n_days), n_merchants)
    who = np.repeat(np.arange(n_merchants), n_days)
    return day, who


# ─────────────────────────── the hard constraint ───────────────────────────


@pytest.mark.parametrize("k", [0, 1, 7, 50, 500])
def test_alerts_never_exceed_k(k: int) -> None:
    """T-132's done-when: ``alerts_per_day <= K`` always. By construction, not by luck."""
    day, _ = grid(500, 12)
    rng = np.random.default_rng(k)
    score = rng.random(day.size)
    exposure = rng.uniform(10_000, 500_000, day.size)
    actions = select_actions(score, day, exposure, k, PARAMS)
    for d in np.unique(day):
        assert (actions[day == d] != Action.PASS).sum() <= k


def test_capacity_is_a_ceiling_not_a_quota() -> None:
    """A day with only two merchants worth intervening on must emit two alerts, not K.

    "Worth it" is an economic test, not a score threshold: with review_cost=250 and
    p_catch=0.80, even a 0.1%-risk merchant clears it if the exposure is large enough.
    The merchants that must not be alerted on here are the ones with nothing at stake.
    """
    day = np.zeros(100, dtype=int)
    score = np.full(100, 0.001)
    score[:2] = 0.99
    exposure = np.full(100, 1_000.0)
    exposure[:2] = 500_000.0
    actions = select_actions(score, day, exposure, K, PARAMS)
    assert (actions != Action.PASS).sum() == 2


def test_a_zero_capacity_day_passes_everything() -> None:
    day = np.zeros(10, dtype=int)
    actions = select_actions(np.full(10, 0.99), day, np.full(10, 1e6), 0, PARAMS)
    assert (actions == Action.PASS).all()


def test_negative_capacity_is_refused() -> None:
    with pytest.raises(ValueError, match="K must be >= 0"):
        select_actions(np.array([0.5]), np.array([0]), np.array([1.0]), -1, PARAMS)


# ─────────────────────────── the action choice ───────────────────────────


def test_expected_costs_follow_the_spec_formula() -> None:
    score = np.array([0.0, 1.0])
    exposure = np.array([100_000.0, 100_000.0])
    c_pass, c_review, c_hold = expected_costs(score, exposure, PARAMS)
    assert c_pass.tolist() == [0.0, 100_000.0]
    assert c_review[0] == pytest.approx(250.0)
    assert c_review[1] == pytest.approx(250.0 + 0.2 * 100_000.0)
    assert c_hold[0] == pytest.approx(250.0 + 8000.0)  # certainly good -> full churn cost
    assert c_hold[1] == pytest.approx(250.0)  # certainly fraud -> no churn cost


def test_hold_needs_both_a_high_score_and_a_real_exposure() -> None:
    """You do not freeze a merchant over 4,000 rupees of expected exposure."""
    day = np.zeros(3, dtype=int)
    score = np.array([0.99, 0.99, 0.40])
    exposure = np.array([4_000.0, 900_000.0, 900_000.0])
    actions = select_actions(score, day, exposure, 3, PARAMS, ActionPolicy())
    assert actions[0] == Action.REVIEW, "tiny exposure was HELD"
    assert actions[1] == Action.HOLD
    assert actions[2] == Action.REVIEW, "a 0.40 score was HELD"


def test_selected_merchants_get_their_own_cost_minimising_action() -> None:
    day = np.zeros(2, dtype=int)
    actions = select_actions(
        np.array([0.99, 0.60]), day, np.array([900_000.0, 900_000.0]), 2, PARAMS
    )
    assert set(actions.tolist()) <= {Action.HOLD, Action.REVIEW}


def test_a_policy_threshold_outside_zero_one_is_refused() -> None:
    with pytest.raises(ValueError, match="probability"):
        ActionPolicy(hold_score_threshold=1.5)


# ─────────────────────────── stability (feeds FR-024) ───────────────────────────


def test_selection_is_stable_under_small_score_perturbations() -> None:
    """The spec's second required property. An alert list that reshuffles on noise
    produces a low alert_jaccard_wow and is unusable by an ops team."""
    day, _ = grid(400, 10)
    rng = np.random.default_rng(21)
    base = rng.random(day.size)
    exposure = rng.uniform(50_000, 400_000, day.size)

    a = select_actions(base, day, exposure, K, PARAMS)
    perturbed = np.clip(base + rng.normal(0, 1e-4, day.size), 0, 1)
    b = select_actions(perturbed, day, exposure, K, PARAMS)

    set_a = set(np.flatnonzero(a != Action.PASS))
    set_b = set(np.flatnonzero(b != Action.PASS))
    jaccard = len(set_a & set_b) / len(set_a | set_b)
    assert jaccard > 0.90, f"a 1e-4 score jitter reshuffled the alert list (J={jaccard:.3f})"


def test_selection_is_deterministic() -> None:
    day, _ = grid(200, 5)
    rng = np.random.default_rng(1)
    score, exposure = rng.random(day.size), rng.uniform(1e4, 1e5, day.size)
    first = select_actions(score, day, exposure, K, PARAMS)
    second = select_actions(score, day, exposure, K, PARAMS)
    assert (first == second).all()


# ─────────────────────────── the cost-asymmetry sweep ───────────────────────────


def sweep_world(seed: int = 4):
    rng = np.random.default_rng(seed)
    n_merchants, n_days = 800, 15
    day = np.tile(np.arange(n_days), n_merchants)
    who = np.repeat(np.arange(n_merchants), n_days)
    is_fraud = np.zeros(n_merchants, dtype=bool)
    is_fraud[:12] = True
    y = is_fraud[who].astype(np.int8)
    loss = np.where(y == 1, rng.uniform(50_000, 300_000, day.size), 0.0)
    exposure = np.full(day.size, 120_000.0)
    scores = {
        "rung0_random": rng.random(day.size),
        "rung2_lgbm": np.clip(0.8 * y + rng.normal(0, 0.2, day.size), 0, 1),
        "rung1_rules": np.clip(0.5 * y + rng.normal(0, 0.35, day.size), 0, 1),
    }
    return scores, day, y, loss, exposure


def test_the_sweep_produces_a_ranking_at_all_five_ratios() -> None:
    """T-132's done-when."""
    scores, day, y, loss, exposure = sweep_world()
    rows = sweep_cost_asymmetry(scores, day, y, loss, exposure, K, PARAMS)
    assert {r.ratio for r in rows} == set(ASYMMETRY_RATIOS)
    for ratio in ASYMMETRY_RATIOS:
        at_ratio = sorted((r for r in rows if r.ratio == ratio), key=lambda r: r.rank)
        assert [r.rank for r in at_ratio] == [1, 2, 3]
        assert {r.rung for r in at_ratio} == set(scores)


def test_the_sweep_reports_the_ranking_rather_than_asserting_one() -> None:
    """A ranking stable across the sweep is a strong claim; a ranking that flips is the
    finding. Either way the harness reports it and does not smooth it over."""
    scores, day, y, loss, exposure = sweep_world()
    rows = sweep_cost_asymmetry(scores, day, y, loss, exposure, K, PARAMS)
    winners = {r.ratio: r.rung for r in rows if r.rank == 1}
    assert len(winners) == len(ASYMMETRY_RATIOS)
    # The good ranker should win at least somewhere, or the cost model is upside down.
    assert "rung2_lgbm" in winners.values()


def test_the_sweep_varies_only_the_asymmetry() -> None:
    """review_cost and p_catch are held fixed; changing everything at once makes the
    result uninterpretable."""
    scores, day, y, loss, exposure = sweep_world()
    rows = sweep_cost_asymmetry(scores, day, y, loss, exposure, K, PARAMS, ratios=(1.0,))
    baseline = CostParams(false_hold_cost_inr=float(loss[y == 1].mean()))
    expected = savings_of_actions(
        select_actions(scores["rung2_lgbm"], day, exposure, K, baseline),
        y,
        loss,
        baseline,
    )
    got = next(r.savings for r in rows if r.rung == "rung2_lgbm")
    assert got == pytest.approx(expected)


def test_the_sweep_refuses_a_window_with_no_fraud() -> None:
    """The ratio has no denominator, and returning zeros would look like a result."""
    day = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="no fraud rows"):
        sweep_cost_asymmetry(
            {"a": np.full(10, 0.5)},
            day,
            np.zeros(10, np.int8),
            np.zeros(10),
            np.full(10, 1000.0),
            5,
            PARAMS,
        )
