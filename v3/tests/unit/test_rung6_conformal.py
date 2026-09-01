"""T-0121 — Rung 6, Mondrian conformal risk control on the three-action decision.

Everything here runs on small synthetic fixtures with known answers. **No number in this
file is measured on ``data/v2/features.parquet``**; the real panel is being regenerated and
scoring happens later, by the lead. The point of a synthetic fixture for a conformal method
is that the exchangeability assumption is *constructed* rather than hoped for, so a
deviation from nominal alpha is attributable to the arithmetic and to nothing else.

The two tests that matter most are the pair at the bottom:
``test_an_exchangeable_calibration_set_lands_on_nominal_alpha`` and
``test_a_non_exchangeable_calibration_set_reports_a_violation``. Pre-registration §5 commits
to a coverage violation being reported rather than clamped, so the second one asserts that
:func:`false_hold_coverage` returns ``violated=True`` with the realised rate visible — not
that the rung suppresses it.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from rakshak.eval.capacity import (
    DEFAULT_DECISION,
    DEFAULT_POLICY,
    ActionPolicy,
    DecisionRequest,
)
from rakshak.eval.metrics import CostParams, false_hold_coverage
from rakshak.features.cohort import assign_cohorts
from rakshak.models.rung6_conformal import (
    ConformalHold,
    MondrianCalibration,
    calibrate,
    split_validation,
    strata_of,
)
from rakshak.schemas import Action, MerchantProfile

ALPHAS = (0.01, 0.05, 0.10)
QUIET = "mcc=grocery|gmv=2|vint=1"
NOISY = "mcc=travel|gmv=8|vint=3"

#: A deliberately degenerate cost setting: a false HOLD is free and exposure is enormous, so
#: the inner selector wants to HOLD every row it is handed. That strips the capacity layer
#: and the HOLD policy out of the way and leaves the conformal gate as the only thing
#: deciding, which is what makes the realised rate in the end-to-end tests comparable to
#: nominal alpha at all. `test_the_wrapper_only_ever_softens` uses the real defaults.
FREE_HOLD = CostParams(false_hold_cost_inr=0.0)
WIDE_OPEN = ActionPolicy(hold_score_threshold=0.0, hold_expected_loss_floor_inr=0.0)

SEVERITY = {Action.PASS: 0, Action.REVIEW: 1, Action.HOLD: 2}


def _request(score: np.ndarray, *, k: int | None = None) -> DecisionRequest:
    """One day, every row selectable, HOLD unconstrained by cost or policy."""
    n = score.size
    return DecisionRequest(
        score=score,
        day=np.zeros(n, dtype=np.int64),
        exposure_inr=np.full(n, 1e9),
        k=n if k is None else k,
        params=FREE_HOLD,
        hold_policy=WIDE_OPEN,
    )


def _severity(action: np.ndarray) -> np.ndarray:
    return np.asarray([SEVERITY[Action(a)] for a in action.tolist()])


# ───────────────────────── the conformal threshold itself ─────────────────────────


@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_exceedance_bound_is_at_most_alpha_and_the_realised_rate_matches_it(
    alpha: float,
) -> None:
    # Calibration and evaluation negatives from the same uniform: exchangeable by
    # construction, so the only thing that can move realised coverage off alpha is the
    # order-statistic arithmetic.
    n = 4000
    rng = np.random.default_rng(7)
    stratum = np.full(n, QUIET)
    cal = calibrate(rng.random(n), np.zeros(n, dtype=np.int64), stratum, alpha)

    assert cal.n_calibration[QUIET] == n
    assert cal.bound(QUIET) <= alpha  # the guarantee, exactly, before any data is scored

    realised = float(cal.permits(rng.random(n), stratum).mean())
    slack = 4.0 * math.sqrt(alpha * (1.0 - alpha) / n)
    assert abs(realised - alpha) <= slack, f"{realised=} {alpha=} {slack=}"


def test_a_stratum_too_small_to_certify_alpha_refuses_to_hold_rather_than_guessing() -> None:
    # ceil((50 + 1) * 0.99) = 51 > 50: there is no order statistic that certifies 1% from
    # 50 points. The honest answer is +inf, not the maximum observed score.
    n, alpha = 50, 0.01
    stratum = np.full(n, QUIET)
    cal = calibrate(np.linspace(0.0, 1.0, n), np.zeros(n, dtype=np.int64), stratum, alpha)

    assert math.isinf(cal.threshold[QUIET])
    assert cal.bound(QUIET) == 0.0
    assert not cal.permits(np.full(n, 1e9), stratum).any()


def test_a_stratum_never_seen_in_calibration_never_permits_a_hold() -> None:
    cal = calibrate(
        np.linspace(0.0, 1.0, 400), np.zeros(400, dtype=np.int64), np.full(400, QUIET), 0.05
    )
    unseen = np.full(10, "mcc=nightlife|gmv=9|vint=0")

    assert not cal.permits(np.ones(10), unseen).any()
    assert cal.bound("mcc=nightlife|gmv=9|vint=0") == 0.0


def test_calibration_uses_the_negatives_only() -> None:
    # Fraud days carry the highest scores. If they entered the quantile the threshold would
    # be pulled up and the rung would look better than it is on the only rate it claims.
    n = 400
    score = np.concatenate([np.linspace(0.0, 0.5, n), np.linspace(0.9, 1.0, n)])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    cal = calibrate(score, y, np.full(2 * n, QUIET), 0.05)

    assert cal.n_calibration[QUIET] == n
    assert cal.threshold[QUIET] < 0.5


def test_mondrian_conditioning_catches_what_a_pooled_threshold_hides() -> None:
    # Two strata with different score scales. A single marginal threshold is satisfied
    # overall by over-holding the noisy stratum at twice alpha — the exact failure
    # stratification exists to expose.
    n, alpha = 4000, 0.05
    rng = np.random.default_rng(11)
    stratum = np.concatenate([np.full(n, QUIET), np.full(n, NOISY)])
    cal_score = np.concatenate([rng.uniform(0.0, 0.5, n), rng.uniform(0.5, 1.0, n)])
    ev_score = np.concatenate([rng.uniform(0.0, 0.5, n), rng.uniform(0.5, 1.0, n)])
    noisy = stratum == NOISY

    pooled = calibrate(cal_score, np.zeros(2 * n, dtype=np.int64), np.full(2 * n, "all"), alpha)
    pooled_rate = float(pooled.permits(ev_score, np.full(2 * n, "all"))[noisy].mean())

    mondrian = calibrate(cal_score, np.zeros(2 * n, dtype=np.int64), stratum, alpha)
    mondrian_rate = float(mondrian.permits(ev_score, stratum)[noisy].mean())

    slack = 4.0 * math.sqrt(alpha * (1.0 - alpha) / n)
    # ~0.097 against a nominal 0.05: the marginal guarantee is not a per-cell one, and the
    # gap is far outside anything sampling noise on 4,000 rows could produce.
    assert pooled_rate > alpha + slack
    assert abs(mondrian_rate - alpha) <= slack


# ───────────────────────── the seam: soften only, K binding ─────────────────────────


def test_the_wrapper_only_ever_softens_and_leaves_the_capacity_budget_untouched() -> None:
    # Real CostParams, real ActionPolicy, real K — the conditions the seam's rule is about.
    n_merchants, n_days, k = 1000, 20, 50
    rng = np.random.default_rng(42)
    day = np.tile(np.arange(n_days), n_merchants)
    score = rng.random(day.size)
    stratum = np.where(rng.random(day.size) < 0.5, QUIET, NOISY)
    request = DecisionRequest(
        score=score,
        day=day,
        exposure_inr=rng.uniform(10_000, 500_000, day.size),
        k=k,
        params=CostParams(),
        hold_policy=DEFAULT_POLICY,
    )
    cal = calibrate(rng.random(4000), np.zeros(4000, dtype=np.int64), stratum[:4000], 0.05)

    before = DEFAULT_DECISION.decide(request)
    after = ConformalHold(DEFAULT_DECISION, cal, stratum).decide(request)

    assert (_severity(after) <= _severity(before)).all()  # never promoted
    # The non-PASS set is identical row for row, so alerts_per_day cannot have moved and no
    # softening can have opened a slot for a K+1th alert.
    assert np.array_equal(before != Action.PASS, after != Action.PASS)
    assert np.bincount(day[after != Action.PASS], minlength=n_days).max() <= k
    assert (after != before).any(), "fixture must actually exercise a softening"
    assert ConformalHold(DEFAULT_DECISION, cal, stratum).name == "crc(capacity_topk, alpha=0.05)"


def test_a_stratum_array_that_does_not_line_up_with_the_request_is_refused() -> None:
    cal = calibrate(
        np.linspace(0.0, 1.0, 200), np.zeros(200, dtype=np.int64), np.full(200, QUIET), 0.05
    )
    policy = ConformalHold(DEFAULT_DECISION, cal, np.full(9, QUIET))
    with pytest.raises(ValueError, match="aligned row-for-row"):
        policy.decide(_request(np.linspace(0.0, 1.0, 10)))


# ───────────────────────── the validation carve, and the split guard ─────────────────


@pytest.mark.parametrize("split", ["train", "test"])
def test_calibration_refuses_every_split_but_validation(split: str) -> None:
    with pytest.raises(ValueError, match="validation split only"):
        calibrate(
            np.linspace(0.0, 1.0, 100),
            np.zeros(100, dtype=np.int64),
            np.full(100, QUIET),
            0.05,
            split=split,  # type: ignore[arg-type]
        )


def test_the_calibration_fold_is_carved_by_merchant_so_no_merchant_is_in_both(
    rng: np.random.Generator,
) -> None:
    merchant_id = np.repeat([f"M{i:03d}" for i in range(200)], 30)
    mask = split_validation(merchant_id, rng, fraction=0.5)

    assert not set(merchant_id[mask].tolist()) & set(merchant_id[~mask].tolist())
    assert np.unique(merchant_id[mask]).size == 100
    # Every day of a merchant travels with it: the fold boundary is never inside a merchant.
    assert all(mask[merchant_id == m].all() or (~mask[merchant_id == m]).all() for m in
               np.unique(merchant_id))


def test_strata_come_from_the_cohort_key_including_its_backed_off_cells() -> None:
    # 40 grocery merchants share a full key; 5 travel merchants cannot fill one and back off
    # past mcc_group to global. Both are ordinary strata here and both get a coverage row.
    at = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    profiles = {
        f"M{i:03d}": MerchantProfile(
            merchant_id=f"M{i:03d}",
            onboarded_at=at,
            mcc="5411",
            mcc_group="grocery" if i < 40 else "travel",
            declared_monthly_gmv=100_000.0,
            kyc_tier=2,
            vintage_months=12,
            city_tier=1,
        )
        for i in range(45)
    }
    cohorts = assign_cohorts(profiles)
    merchant_id = np.repeat(sorted(profiles), 3)
    stratum = strata_of(merchant_id, cohorts)

    assert set(stratum.tolist()) == {cohorts.label["M000"], "global"}
    assert cohorts.level["M044"] == "global"

    cal = calibrate(
        np.linspace(0.0, 1.0, stratum.size), np.zeros(stratum.size, dtype=np.int64), stratum, 0.10
    )
    assert set(cal.threshold) == set(stratum.tolist())


# ─────────── end to end, through the decision layer and false_hold_coverage ───────────


@pytest.mark.parametrize("alpha", ALPHAS)
def test_an_exchangeable_calibration_set_lands_on_nominal_alpha(alpha: float) -> None:
    n = 4000
    rng = np.random.default_rng(3)
    stratum = np.concatenate([np.full(n, QUIET), np.full(n, NOISY)])
    y = np.zeros(2 * n, dtype=np.int64)
    cal = calibrate(
        np.concatenate([rng.uniform(0.0, 0.5, n), rng.uniform(0.5, 1.0, n)]), y, stratum, alpha
    )

    score = np.concatenate([rng.uniform(0.0, 0.5, n), rng.uniform(0.5, 1.0, n)])
    action = ConformalHold(DEFAULT_DECISION, cal, stratum).decide(_request(score))
    rows = false_hold_coverage(action, y, stratum, alpha)

    assert len(rows) == 2  # every Mondrian cell gets its own row; there is no pooled row
    slack = 4.0 * math.sqrt(alpha * (1.0 - alpha) / n)
    for row in rows:
        assert row.n_negatives == n
        assert abs(row.realised - alpha) <= slack, f"{row}"


def test_a_non_exchangeable_calibration_set_reports_a_violation_rather_than_hiding_it() -> None:
    # The ticket's own worry, made concrete: chargeback labels arrive faster for bust-out
    # than for a slow ramp, so the merchants whose labels had landed by the calibration
    # cutoff are not a random sample of negatives. Here they are the quiet half, and the
    # threshold they produce is far too low for the population it is then applied to.
    n, alpha = 4000, 0.05
    rng = np.random.default_rng(5)
    stratum = np.full(n, QUIET)
    y = np.zeros(n, dtype=np.int64)
    cal = calibrate(rng.uniform(0.0, 0.6, n), y, stratum, alpha)

    score = rng.uniform(0.0, 1.0, n)  # the population, not the fast-labelling half of it
    action = ConformalHold(DEFAULT_DECISION, cal, stratum).decide(_request(score))
    (row,) = false_hold_coverage(action, y, stratum, alpha)

    assert row.violated, "a coverage violation must surface, not be clamped away"
    assert row.realised > 5.0 * alpha  # ~0.43 against a nominal 0.05
    assert row.n_negatives == n and row.n_false_hold > 0  # the numbers behind it are visible
    # And nothing in the rung tried to rescue it: the threshold is still the plain order
    # statistic the exchangeable case would have produced, with no shrinkage toward alpha.
    assert cal.threshold[QUIET] == pytest.approx(0.6 * (1.0 - alpha), abs=0.02)


def test_capacity_and_the_hold_policy_only_ever_shrink_the_certified_set() -> None:
    # The guarantee is on {score > t}; the realised HOLD set is a subset of it, because a
    # row must also win a top-K slot and clear ActionPolicy. So a tighter K can only lower
    # realised coverage, never raise it above alpha.
    n, alpha = 2000, 0.10
    rng = np.random.default_rng(13)
    stratum = np.full(n, QUIET)
    y = np.zeros(n, dtype=np.int64)
    cal = calibrate(rng.random(n), y, stratum, alpha)
    score = rng.random(n)
    policy = ConformalHold(DEFAULT_DECISION, cal, stratum)

    unconstrained = policy.decide(_request(score))
    squeezed = policy.decide(_request(score, k=25))

    assert (unconstrained == Action.HOLD).sum() > (squeezed == Action.HOLD).sum() > 0
    assert (squeezed == Action.HOLD)[unconstrained != Action.HOLD].sum() == 0  # a subset
    (tight,) = false_hold_coverage(squeezed, y, stratum, alpha)
    assert not tight.violated and tight.realised < alpha


def test_a_calibration_object_can_be_read_without_running_anything() -> None:
    # The thresholds are the artefact a reviewer checks; they are plain floats per stratum,
    # not something recomputed inside decide().
    cal = MondrianCalibration(alpha=0.05, threshold={QUIET: 0.9}, n_calibration={QUIET: 1000})
    assert cal.bound(QUIET) <= 0.05
    assert cal.permits(np.array([0.89, 0.90, 0.91]), np.full(3, QUIET)).tolist() == [
        False,
        False,
        True,
    ]
