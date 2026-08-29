"""Metrics against hand-computable fixtures.

Every expected value below is derived by hand in the comment above it. If a
metric changes, the comment is the spec that says whether the change is a fix
or a regression.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rakshak.config import (
    COST_REVIEW_INR,
    COST_SUPPORT_INR,
    P_ANALYST_MISS,
    P_CHURN_GIVEN_HOLD,
    RESIDUAL_LEAKAGE_RHO,
)
from rakshak.eval import metrics
from rakshak.eval.metrics import HOLD, PASS, REVIEW

# ---------------------------------------------------------------------------
# Ranking / calibration
# ---------------------------------------------------------------------------


def test_pr_auc_is_one_for_a_perfect_ranking() -> None:
    y = np.array([0, 0, 1, 1])
    assert metrics.pr_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_pr_auc_hand_computed() -> None:
    # scores 0.9, 0.8, 0.7, 0.6 -> labels 1, 0, 1, 0.
    # Average precision = sum over positives of precision at that rank / n_pos
    #   rank 1 (positive): precision 1/1 = 1.0
    #   rank 3 (positive): precision 2/3
    # AP = (1.0 + 2/3) / 2 = 0.8333...
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert metrics.pr_auc(y, scores) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_pr_auc_degenerates_to_prevalence_when_labels_are_constant() -> None:
    assert metrics.pr_auc(np.zeros(5), np.arange(5.0)) == 0.0
    assert metrics.pr_auc(np.ones(5), np.arange(5.0)) == 1.0


def test_precision_at_k_hand_computed() -> None:
    # Top 3 by score are the merchants scoring 0.9, 0.8, 0.7 -> labels 1, 0, 1.
    y = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert metrics.precision_at_k(y, scores, 3) == pytest.approx(2.0 / 3.0)
    assert metrics.precision_at_k(y, scores, 1) == pytest.approx(1.0)
    assert metrics.precision_at_k(y, scores, 0) == 0.0
    # k larger than n clamps to n: precision becomes prevalence.
    assert metrics.precision_at_k(y, scores, 99) == pytest.approx(0.4)


def test_precision_at_k_breaks_ties_deterministically() -> None:
    y = np.array([0, 1, 1, 0])
    flat = np.zeros(4)
    assert metrics.precision_at_k(y, flat, 2) == pytest.approx(0.5)  # indices 0, 1
    assert metrics.precision_at_k(y, flat, 2) == metrics.precision_at_k(y, flat, 2)


def test_brier_score_hand_computed() -> None:
    # ((0.9-1)^2 + (0.2-0)^2) / 2 = (0.01 + 0.04) / 2 = 0.025
    assert metrics.brier_score(np.array([1, 0]), np.array([0.9, 0.2])) == pytest.approx(0.025)


def test_brier_score_is_zero_when_perfect() -> None:
    assert metrics.brier_score(np.array([1, 0, 1]), np.array([1.0, 0.0, 1.0])) == 0.0


# ---------------------------------------------------------------------------
# Detection lag
# ---------------------------------------------------------------------------


def _series(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, index=pd.Index(list(values), name="merchant_id"), dtype="float64")


def test_detection_lag_median_and_flagged_fraction() -> None:
    # Bad merchants: A, B, C. Flagged: A (lag 5), B (lag 11). C never flagged.
    labels = pd.Series({"A": 1, "B": 1, "C": 1, "D": 0})
    transition = _series({"A": 100.0, "B": 100.0, "C": 100.0, "D": math.nan})
    flags = _series({"A": 105.0, "B": 111.0, "C": math.nan, "D": 90.0})
    lag, flagged, n_bad = metrics.detection_lag_days(flags, transition, labels)
    assert lag == pytest.approx(8.0)  # median of {5, 11}
    assert flagged == pytest.approx(2.0 / 3.0)
    assert n_bad == 3


def test_detection_lag_keeps_negative_lags() -> None:
    """A model that fires before the transition must not be silently clipped."""
    labels = pd.Series({"A": 1})
    lag, flagged, n_bad = metrics.detection_lag_days(
        _series({"A": 90.0}), _series({"A": 100.0}), labels
    )
    assert lag == pytest.approx(-10.0)
    assert (flagged, n_bad) == (1.0, 1)


def test_detection_lag_is_nan_when_nothing_was_flagged() -> None:
    labels = pd.Series({"A": 1, "B": 1})
    lag, flagged, n_bad = metrics.detection_lag_days(
        _series({"A": math.nan, "B": math.nan}), _series({"A": 10.0, "B": 20.0}), labels
    )
    assert math.isnan(lag)
    assert (flagged, n_bad) == (0.0, 2)


# ---------------------------------------------------------------------------
# Cost layer (07-math.md §5-6)
# ---------------------------------------------------------------------------


def test_false_positive_cost_matches_the_formula() -> None:
    value = np.array([0.0, 10_000.0])
    expected = P_CHURN_GIVEN_HOLD * value + COST_SUPPORT_INR
    assert metrics.false_positive_cost(value) == pytest.approx(expected)


def test_action_cost_matrix_cell_by_cell() -> None:
    loss = np.full(6, 1000.0)
    value = np.zeros(6)  # c_fp collapses to c_support = 500
    y = np.array([0, 0, 0, 1, 1, 1])
    actions = np.array([PASS, REVIEW, HOLD, PASS, REVIEW, HOLD])
    expected = np.array(
        [
            0.0,
            COST_REVIEW_INR,
            COST_SUPPORT_INR,
            1000.0,
            COST_REVIEW_INR + P_ANALYST_MISS * 1000.0,
            RESIDUAL_LEAKAGE_RHO * 1000.0,
        ]
    )
    assert metrics.action_cost(y, actions, loss, value) == pytest.approx(expected)


def test_savings_is_zero_for_the_better_trivial_policy() -> None:
    """Bahnsen's denominator is min(all PASS, all HOLD), so whichever wins scores 0."""
    y = np.array([1, 0, 0, 0])
    loss = np.array([1000.0, 0.0, 0.0, 0.0])
    value = np.zeros(4)
    n = y.size
    all_pass = metrics.savings_score(y, np.full(n, PASS), loss, value)
    all_hold = metrics.savings_score(y, np.full(n, HOLD), loss, value)
    assert max(all_pass, all_hold) == pytest.approx(0.0)
    assert min(all_pass, all_hold) <= 0.0


def test_savings_score_hand_computed() -> None:
    # One bad merchant (L = 1000) and three healthy (V = 0 -> c_fp = 500).
    #   Cost(all PASS) = 1000
    #   Cost(all HOLD) = 0.10 * 1000 + 3 * 500 = 1600  -> Cost_l = 1000
    #   Perfect policy: HOLD the bad one, PASS the rest = 0.10 * 1000 = 100
    #   Savings = (1000 - 100) / 1000 = 0.9
    y = np.array([1, 0, 0, 0])
    loss = np.array([1000.0, 0.0, 0.0, 0.0])
    value = np.zeros(4)
    actions = np.array([HOLD, PASS, PASS, PASS])
    assert metrics.savings_score(y, actions, loss, value) == pytest.approx(0.9)


def test_savings_score_goes_negative_for_a_worse_than_nothing_policy() -> None:
    # HOLD every healthy merchant, PASS the bad one:
    #   Cost = 1000 + 3 * 500 = 2500 against Cost_l = 1000 -> savings = -1.5
    y = np.array([1, 0, 0, 0])
    loss = np.array([1000.0, 0.0, 0.0, 0.0])
    value = np.zeros(4)
    actions = np.array([PASS, HOLD, HOLD, HOLD])
    assert metrics.savings_score(y, actions, loss, value) == pytest.approx(-1.5)


def test_savings_is_zero_when_there_is_nothing_to_save() -> None:
    y = np.zeros(3)
    assert metrics.savings_score(y, np.full(3, PASS), np.zeros(3), np.zeros(3)) == 0.0


def test_gap_to_oracle() -> None:
    assert metrics.gap_to_oracle(0.5, 1.0) == pytest.approx(0.5)
    assert metrics.gap_to_oracle(1.0, 1.0) == pytest.approx(0.0)
    assert metrics.gap_to_oracle(1.2, 1.0) == pytest.approx(-0.2)
    assert math.isnan(metrics.gap_to_oracle(0.5, 0.0))


def test_roc_auc_and_accuracy_are_not_implemented() -> None:
    """06-requirements.md §3 prohibits them as headline metrics. The cheapest
    guarantee they never reach the headline path is that they do not exist."""
    exported = dir(metrics)
    assert not [name for name in exported if "roc" in name.lower()]
    assert not [name for name in exported if "accuracy" in name.lower()]
