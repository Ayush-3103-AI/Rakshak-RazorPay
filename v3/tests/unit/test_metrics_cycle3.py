"""T-0118 — the three cycle-3 metrics, on synthetic input with no rung present.

Every case here is hand-computable. That is the point: these three names enter the metric
list in ``EVAL-LOCK-CYCLE3.json`` and are what Rungs 6, 7 and 8 will be judged on, so the
definitions have to be pinned by arithmetic somebody can check on paper, not by a smoke
test that only proves the function returns.

Nothing in this file reads the test split (days 300-364) or any generated artifact. The
inputs are literals.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rakshak.eval.metrics import (
    false_hold_coverage,
    onset_localisation_error,
    tpp_rescaled_ks,
)
from rakshak.schemas import Action

# ────────────────────────── false_hold_coverage (Rung 6) ──────────────────────────


def _stratum_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two strata, hand-built.

    ``small_merchant``: 100 negative days, 12 of them HOLD -> realised 0.12.
    ``large_merchant``: 100 negative days,  5 of them HOLD -> realised 0.05.

    Each stratum also carries 10 positive (fraud) days, all HELD. They must not move
    either number: a HOLD on a drifting merchant is a correct HOLD, and counting it would
    turn the metric into an alert rate.
    """
    action, y, stratum = [], [], []
    for name, n_false_hold in (("small_merchant", 12), ("large_merchant", 5)):
        for i in range(100):
            action.append(Action.HOLD if i < n_false_hold else Action.PASS)
            y.append(0)
            stratum.append(name)
        for _ in range(10):
            action.append(Action.HOLD)
            y.append(1)
            stratum.append(name)
    return np.array(action), np.array(y), np.array(stratum)


def test_false_hold_coverage_is_the_hand_computed_rate_per_stratum() -> None:
    action, y, stratum = _stratum_case()
    rows = {r.stratum: r for r in false_hold_coverage(action, y, stratum, alpha=0.10)}

    assert rows["small_merchant"].n_negatives == 100
    assert rows["small_merchant"].n_false_hold == 12
    assert rows["small_merchant"].realised == pytest.approx(0.12)
    assert rows["large_merchant"].n_negatives == 100
    assert rows["large_merchant"].n_false_hold == 5
    assert rows["large_merchant"].realised == pytest.approx(0.05)


def test_a_violated_stratum_is_reported_and_not_clamped() -> None:
    """The criterion T-0118 names, and pre-registration §5's commitment.

    At alpha = 0.10 the small-merchant stratum realises 0.12. The metric must say so:
    ``violated`` True and ``realised`` still 0.12, not clipped to alpha, not shrunk toward
    it, not averaged away against the stratum that passes.
    """
    action, y, stratum = _stratum_case()
    rows = {r.stratum: r for r in false_hold_coverage(action, y, stratum, alpha=0.10)}

    assert rows["small_merchant"].violated is True
    assert rows["small_merchant"].realised == pytest.approx(0.12)
    assert rows["small_merchant"].realised > rows["small_merchant"].alpha
    assert rows["large_merchant"].violated is False

    # And the violation survives aggregation: pooled, 17/200 = 0.085 <= 0.10 would pass.
    pooled = false_hold_coverage(action, y, np.full(stratum.size, "all"), alpha=0.10)
    assert pooled[0].realised == pytest.approx(0.085)
    assert pooled[0].violated is False
    assert any(r.violated for r in false_hold_coverage(action, y, stratum, alpha=0.10))


def test_exact_alpha_is_not_a_violation() -> None:
    """10 false HOLDs in 100 negatives at alpha = 0.10 is coverage met, not breached."""
    action = np.array([Action.HOLD] * 10 + [Action.PASS] * 90)
    y = np.zeros(100, dtype=np.int8)
    (row,) = false_hold_coverage(action, y, np.full(100, "s"), alpha=0.10)
    assert row.realised == pytest.approx(0.10)
    assert row.violated is False


def test_a_stratum_with_no_negatives_reports_nan_rather_than_zero() -> None:
    action = np.array([Action.HOLD, Action.HOLD])
    y = np.ones(2, dtype=np.int8)
    (row,) = false_hold_coverage(action, y, np.full(2, "s"), alpha=0.05)
    assert row.n_negatives == 0
    assert math.isnan(row.realised)
    assert row.violated is False


def test_alpha_outside_the_unit_interval_is_refused() -> None:
    action = np.array([Action.PASS])
    with pytest.raises(ValueError, match="nominal error rate"):
        false_hold_coverage(action, np.zeros(1), np.array(["s"]), alpha=1.5)


# ──────────────────────── onset_localisation_error (Rung 7) ────────────────────────


def test_a_deliberate_offset_is_recovered_exactly() -> None:
    """T-0118's criterion: an estimate offset by a known number of days recovers it."""
    true = np.array([40.0, 90.0, 150.0, 210.0])
    result = onset_localisation_error(true + 4.0, true)

    assert result.n == 4
    assert list(result.error_days) == [4.0, 4.0, 4.0, 4.0]
    assert result.median == pytest.approx(4.0)
    assert result.iqr == pytest.approx(0.0)


def test_the_error_is_signed_early_negative_late_positive() -> None:
    true = np.array([100.0, 100.0])
    result = onset_localisation_error(np.array([93.0, 106.0]), true)
    assert list(result.error_days) == [-7.0, 6.0]


def test_median_and_iqr_on_a_hand_computed_five_point_distribution() -> None:
    """errors = [-2, -1, 0, 3, 5] sorted.

    numpy's linear interpolation puts q25 at index 1 and q75 at index 3 for n = 5, so
    q25 = -1, median = 0, q75 = 3 and IQR = 4. Written out because the IQR convention is
    the kind of thing that silently differs between two implementations of "the same"
    metric.
    """
    true = np.zeros(5)
    result = onset_localisation_error(np.array([-2.0, -1.0, 0.0, 3.0, 5.0]), true)
    assert result.q25 == pytest.approx(-1.0)
    assert result.median == pytest.approx(0.0)
    assert result.q75 == pytest.approx(3.0)
    assert result.iqr == pytest.approx(4.0)


def test_merchants_that_never_drifted_are_dropped_and_undeclared_ones_are_counted() -> None:
    true = np.array([50.0, 60.0, np.nan, 70.0])
    est = np.array([52.0, np.nan, 999.0, 70.0])
    result = onset_localisation_error(est, true)

    assert result.n == 2  # merchants 0 and 3
    assert list(result.error_days) == [2.0, 0.0]
    assert result.n_unlocalised == 1  # merchant 1 drifted; no change-point was declared
    assert result.median == pytest.approx(1.0)


def test_no_localisable_merchant_gives_nan_not_a_flattering_zero() -> None:
    result = onset_localisation_error(np.array([np.nan]), np.array([50.0]))
    assert result.n == 0
    assert result.n_unlocalised == 1
    assert math.isnan(result.median)
    assert math.isnan(result.iqr)


# ───────────────────────────── tpp_rescaled_ks (Rung 8) ─────────────────────────────


def test_a_single_increment_gives_the_hand_computed_ks_statistic() -> None:
    """One increment of ln 2 rescales to u = 1 - exp(-ln 2) = 0.5.

    The empirical CDF of a single point at 0.5 against Uniform(0,1) has
    D+ = 1 - 0.5 = 0.5 and D- = 0.5 - 0 = 0.5, so D = 0.5.
    """
    result = tpp_rescaled_ks(np.array([math.log(2.0)]))
    assert result.n == 1
    assert result.statistic == pytest.approx(0.5)


def test_a_correctly_rescaled_sample_does_not_fire() -> None:
    """Exp(1) increments — the intensity is right, so the fit must not be rejected."""
    rng = np.random.default_rng(42)
    result = tpp_rescaled_ks(rng.exponential(1.0, size=2000))
    assert result.n == 2000
    assert not result.rejects_at(0.05)
    assert result.p_value > 0.05


def test_a_misspecified_intensity_is_rejected() -> None:
    """The same arrivals under an intensity half the true one: every compensator increment
    is doubled, the rescaled times are Exp(1/2), and the KS test must say so."""
    rng = np.random.default_rng(42)
    result = tpp_rescaled_ks(rng.exponential(1.0, size=2000) * 2.0)
    assert result.rejects_at(0.05)
    assert result.p_value < 1e-6


def test_an_empty_process_is_nan_not_a_passing_fit() -> None:
    result = tpp_rescaled_ks(np.array([]))
    assert result.n == 0
    assert math.isnan(result.statistic)
    assert math.isnan(result.p_value)


def test_a_negative_compensator_increment_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative intensity"):
        tpp_rescaled_ks(np.array([0.5, -0.1]))
