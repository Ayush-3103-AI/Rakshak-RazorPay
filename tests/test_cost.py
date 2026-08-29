"""Corrected cost definitions and the oracle-dominance invariant (T-0007a).

Two things are pinned here:

1. **FR-016** — `c_fp` and `L_m` against hand-computed values on a 2-merchant
   fixture. Every expected number below is a literal with its arithmetic shown in
   the comment beside it. Nothing is compared against the function under test.
2. **The oracle-dominance invariant** — a perfect-foresight ceiling must weakly
   dominate every policy scored in the same run, the trivial ones included. The
   last test in this file re-runs the *old* definitions through the same
   assertion and shows it fires, which is the evidence that this check would have
   caught the T-0006 defect at T-0005.

**Structural note, found while building this and worth stating.** The two
ceilings are not equally strong. `perfect_hindsight_oracle` is a per-merchant
argmin over the full action set, so it dominates every policy *by construction*
under any cost matrix. `review_knapsack_oracle` is capacity-constrained and can
only REVIEW, so it is a ceiling over the review-only, <=K action class the
harness's `budget_policy` produces — nothing forces it above hold-everything.
Whether it clears hold-everything depends on how concentrated realised loss is
in the top-K merchants, which is a property of the data, not of the cost
constants. On the shipping `validate` split the top 5 hold 71% of realised loss
and it clears comfortably (+0.317); on a flat toy population it does not. That is
why the dominance claim is asserted on the real split.
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak import config
from rakshak.config import STATE_PATHS_PARQUET, TRANSACTIONS_PARQUET
from rakshak.decision.cost import (
    assert_oracle_dominance,
    expected_monthly_volume_inr,
    fp_cost_per_100_of_fraud_loss,
    merchant_value_inr,
    realised_loss_inr,
    trivial_policy_savings,
)
from rakshak.eval.metrics import PASS, REVIEW, false_positive_cost, savings_score
from rakshak.eval.oracle import perfect_hindsight_oracle, review_knapsack_oracle
from rakshak.eval.splits import load_split

# ---------------------------------------------------------------------------
# The 2-merchant fixture (FR-016)
# ---------------------------------------------------------------------------
#
# Observation span 210 days == 7 months, matching the `validate` split.
#
# Merchant A — healthy.
#   non-refund volume            2,100,000 INR over 210 d
#   v_A = 2,100,000 / 7                    =   300,000 INR/month
#   V_A = g * v_A * l   = 0.0010 * 300,000 * 30
#                                          =     9,000 INR
#   c_fp(A) = 0.35 * 9,000 + 500           =     3,650 INR
#   G_bad_A = 0  ->  L_A                   =         0 INR
#
# Merchant B — bad.
#   non-refund volume              700,000 INR over 210 d
#   v_B = 700,000 / 7                      =   100,000 INR/month
#   V_B = 0.0010 * 100,000 * 30            =     3,000 INR
#   c_fp(B) = 0.35 * 3,000 + 500           =     1,550 INR
#   G_bad_B                                =   500,000 INR
#   L_B = r_cb * (1 + phi) * G_bad_B
#       = 0.05 * 1.35 * 500,000            =    33,750 INR

OBSERVED_DAYS = 210.0
VOLUME_INR = np.array([2_100_000.0, 700_000.0])
GROSS_BAD_VOLUME_INR = np.array([0.0, 500_000.0])
LABELS = np.array([0.0, 1.0])

EXPECTED_MONTHLY_VOLUME = np.array([300_000.0, 100_000.0])
EXPECTED_VALUE_INR = np.array([9_000.0, 3_000.0])
EXPECTED_C_FP_INR = np.array([3_650.0, 1_550.0])
EXPECTED_LOSS_INR = np.array([0.0, 33_750.0])


def test_shipping_constants_are_the_ones_the_fixture_assumes() -> None:
    """The literals above are only meaningful against these central values."""
    assert config.GROSS_MARGIN_RATE == 0.0010
    assert config.MERCHANT_LIFETIME_MONTHS == 30.0
    assert config.CHARGEBACK_REALISATION_RATE == 0.05
    assert config.ANCILLARY_LOADING_PHI == 0.35
    assert config.P_CHURN_GIVEN_HOLD == 0.35
    assert config.COST_SUPPORT_INR == 500.0
    assert not hasattr(config, "MDR_RATE"), (
        "MDR_RATE is the merchant-facing price, not the platform's gross margin. "
        "It was deleted at T-0007a; use GROSS_MARGIN_RATE."
    )


def test_expected_monthly_volume_is_a_rate_not_a_stock() -> None:
    got = expected_monthly_volume_inr(VOLUME_INR, observed_days=OBSERVED_DAYS)
    assert got == pytest.approx(EXPECTED_MONTHLY_VOLUME)


def test_merchant_value_is_lifetime_gross_margin() -> None:
    """V_m = g * v_m * l_m — hand-computed, see the fixture block above."""
    got = merchant_value_inr(
        expected_monthly_volume_inr(VOLUME_INR, observed_days=OBSERVED_DAYS)
    )
    assert got == pytest.approx(EXPECTED_VALUE_INR)


def test_false_positive_cost_against_hand_computed_values() -> None:
    """FR-016: c_fp(m) = P(churn|hold) * V_m + c_support."""
    got = false_positive_cost(EXPECTED_VALUE_INR)
    assert got == pytest.approx(EXPECTED_C_FP_INR)


def test_realised_loss_is_not_turnover() -> None:
    """FR-016: L_m = r_cb * (1 + phi) * G_bad_m, hand-computed."""
    got = realised_loss_inr(GROSS_BAD_VOLUME_INR)
    assert got == pytest.approx(EXPECTED_LOSS_INR)
    # Turnover is not loss: the bad merchant transacted 500,000 and cost 33,750.
    assert got[1] < GROSS_BAD_VOLUME_INR[1]


def test_both_corrections_are_applied_not_only_the_lifetime_one() -> None:
    """Substituting the lifetime alone leaves V_m overstated ~20x.

    The old form was `MDR_RATE (0.02) * volume`. Lifetime-only would be
    `0.02 * v_m * 30`; the shipped form is `0.0010 * v_m * 30`. Guard the ratio
    so a future edit cannot silently reinstate the merchant-facing price.
    """
    lifetime_only = 0.02 * EXPECTED_MONTHLY_VOLUME * config.MERCHANT_LIFETIME_MONTHS
    assert lifetime_only / EXPECTED_VALUE_INR == pytest.approx(np.full(2, 20.0))


def test_fp_per_100_of_fraud_loss_on_the_fixture() -> None:
    """Reported, never gated. 100 * 3,650 / 33,750 = 10.81 on this fixture."""
    ratio, total_fp, total_loss = fp_cost_per_100_of_fraud_loss(
        LABELS, EXPECTED_LOSS_INR, EXPECTED_VALUE_INR
    )
    assert total_fp == pytest.approx(3_650.0)
    assert total_loss == pytest.approx(33_750.0)
    assert ratio == pytest.approx(100.0 * 3_650.0 / 33_750.0)


def test_every_central_value_lies_inside_its_stated_range() -> None:
    """07-math.md §5's ranges and the shipping constants must not drift apart."""
    for name, (low, high) in config.COST_PRIMITIVE_RANGES.items():
        central = getattr(config, name)
        assert low <= central <= high, f"{name}={central} outside [{low}, {high}]"


# ---------------------------------------------------------------------------
# The oracle-dominance invariant
# ---------------------------------------------------------------------------


def _population(n: int = 100, n_bad: int = 20, seed: int = 7):
    """A deterministic 100-merchant, 20-bad toy population.

    Deliberately NOT a stand-in for the real split: its bad-state volume is
    spread evenly across bad merchants, whereas the generator concentrates ~71%
    of all realised loss in the top 5. That concentration is what decides whether
    the *capacity-constrained* knapsack ceiling clears hold-everything (see the
    structural note in this file's header), so the dominance claim is asserted on
    the real `validate` split below, not here. This fixture exists to prove the
    assertion fires, and to pin determinism.
    """
    rng = np.random.default_rng(seed)
    volume = rng.lognormal(mean=np.log(2_000_000.0), sigma=0.8, size=n)
    y = np.zeros(n)
    y[rng.choice(n, size=n_bad, replace=False)] = 1.0
    gross_bad = y * volume * rng.uniform(0.1, 0.6, size=n)
    return y, volume, gross_bad


def _costs(y, volume, gross_bad):
    """(loss_inr, value_inr) under the corrected T-0007a definitions."""
    loss = realised_loss_inr(gross_bad)
    value = merchant_value_inr(expected_monthly_volume_inr(volume, observed_days=210.0))
    return loss, value


def _top_k_review_policy(scores: np.ndarray, k: int) -> np.ndarray:
    """The harness's `budget_policy`, inlined so this test owns no harness code."""
    order = np.lexsort((np.arange(scores.size), -scores))
    actions = np.full(scores.size, PASS, dtype=int)
    actions[order[:k]] = REVIEW
    return actions


CAPACITY_HOURS = 0.4  # ADR-0008: 4.0 h per 1000 merchants x 100 merchants.


def _oracles(y, loss, value, capacity_hours: float = CAPACITY_HOURS) -> dict[str, float]:
    return {
        o.name: o.savings
        for o in (
            review_knapsack_oracle(y, loss, value, capacity_hours=capacity_hours),
            perfect_hindsight_oracle(y, loss, value),
        )
    }


@pytest.mark.skipif(
    not (TRANSACTIONS_PARQUET.exists() and STATE_PATHS_PARQUET.exists()),
    reason="generated data absent; run `python -m rakshak.generator.generate --seed 42`",
)
def test_oracle_dominance_holds_on_the_real_validate_split() -> None:
    """T-0007a's `Done when`, asserted on the population the harness scores.

    Both ceilings must clear hold-everything and every policy in the run. At
    T-0006 the knapsack ceiling scored -0.678 against hold-everything's 0.000.
    """
    split = load_split("validate")
    y = split.labels.to_numpy(dtype=float)
    loss = split.loss_inr.to_numpy(dtype=float)
    value = split.value_inr.to_numpy(dtype=float)
    capacity_hours = (
        config.REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS * y.size / 1000.0
    )
    ceilings = _oracles(y, loss, value, capacity_hours)

    # Both ceilings above the trivial floor, which is 0.0 by Bahnsen's definition.
    for name, ceiling in ceilings.items():
        assert ceiling >= 0.0, f"{name} scores {ceiling:+.4f}, below hold-everything"

    rng = np.random.default_rng(11)
    k = int(capacity_hours // config.TAU_REVIEW_HOURS)
    scored = {
        "perfect ranking": y + rng.normal(0.0, 0.01, size=y.size),
        "noisy ranking": y + rng.normal(0.0, 0.8, size=y.size),
        "pure noise": rng.normal(0.0, 1.0, size=y.size),
    }
    policies = {
        name: savings_score(y, _top_k_review_policy(scores, k), loss, value)
        for name, scores in scored.items()
    }
    checked = assert_oracle_dominance(y, loss, value, ceilings, policies)
    assert set(policies) <= set(checked)
    assert "trivial: hold-everything" in checked


def test_the_better_trivial_policy_scores_exactly_zero() -> None:
    """Bahnsen's denominator is min(all-PASS, all-HOLD), so one of them is 0.0."""
    y, volume, gross_bad = _population()
    loss, value = _costs(y, volume, gross_bad)
    trivial = trivial_policy_savings(y, loss, value)
    assert max(trivial["trivial: pass-everything"], trivial["trivial: hold-everything"]) == (
        pytest.approx(0.0, abs=1e-12)
    )


def test_the_invariant_would_have_caught_the_old_definitions() -> None:
    """The regression proof: under the pre-T-0007a definitions the check fires.

    Old `L_m` = gross turnover while bad; old `V_m` = 0.02 * loaded-history volume.
    This is the T-0006 defect reproduced, and this assertion is what would have
    caught it at T-0005 instead of at T-0006.
    """
    y, volume, gross_bad = _population()
    old_loss = gross_bad  # turnover charged as loss
    old_value = 0.02 * volume  # merchant-facing MDR charged as platform margin
    with pytest.raises(AssertionError, match="oracle-dominance invariant FAILED"):
        assert_oracle_dominance(
            y, old_loss, old_value, _oracles(y, old_loss, old_value)
        )


def test_invariant_is_deterministic() -> None:
    """NFR-003: the random trivial policy is seeded, so repeats are identical."""
    y, volume, gross_bad = _population()
    loss, value = _costs(y, volume, gross_bad)
    first = trivial_policy_savings(y, loss, value)
    second = trivial_policy_savings(y, loss, value)
    assert first == second


def test_invariant_rejects_an_empty_ceiling_set() -> None:
    y, volume, gross_bad = _population()
    loss, value = _costs(y, volume, gross_bad)
    with pytest.raises(ValueError, match="no ceiling"):
        assert_oracle_dominance(y, loss, value, {})
