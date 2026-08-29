"""Bayes Minimum Risk policy, capacity constraint and cost-asymmetry sweep (T-0007b).

Every expected number in the FR-016 fixture below is a **literal with its
arithmetic written out beside it**, computed by hand from `07-math.md` §5's cost
matrix and §6's argmin. Nothing here is compared against the function under test,
and nothing is recomputed the way the implementation computes it.

The fixture is the same two merchants `tests/test_cost.py` uses for T-0007a, so
`V_m`, `c_fp(m)` and `L_m` carry straight over and only the decision layer is new.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rakshak import config
from rakshak.config import STATE_PATHS_PARQUET, TRANSACTIONS_PARQUET
from rakshak.decision.policy import (
    HOLD,
    PASS,
    REVIEW,
    SWEEP_COLUMNS,
    CostParams,
    apply_capacity,
    asymmetry_range,
    bmr_action,
    bmr_policy,
    expected_costs,
    savings,
    sweep_cost_asymmetry,
)

needs_data = pytest.mark.skipif(
    not (TRANSACTIONS_PARQUET.exists() and STATE_PATHS_PARQUET.exists()),
    reason="generated data absent; run `python -m rakshak.generator.generate --seed 42`",
)

# ---------------------------------------------------------------------------
# The 2-merchant fixture (FR-016 acceptance) — all arithmetic by hand
# ---------------------------------------------------------------------------
#
# Carried over from tests/test_cost.py, observation span 210 d (= validate):
#
#   Merchant A (healthy):  V_A = 9,000   c_fp(A) = 0.35*9,000 + 500 = 3,650   L_A = 0
#   Merchant B (bad):      V_B = 3,000   c_fp(B) = 0.35*3,000 + 500 = 1,550
#                          L_B = 0.05 * 1.35 * 500,000              = 33,750
#
# Shipping constants: c_rev = 0.067 h * 600 INR/h = 40.2 INR, p_miss = 0.15,
# rho = 0.10.
#
# Expected cost of each action at posterior p (07-math.md §6):
#   E[PASS]   = p*L
#   E[REVIEW] = c_rev + p*p_miss*L
#   E[HOLD]   = p*rho*L + (1-p)*c_fp
#
# At p = 0.10:
#   A:  PASS   = 0
#       REVIEW = 40.2 + 0.10*0.15*0            = 40.2
#       HOLD   = 0 + 0.90*3,650               = 3,285
#       -> argmin is PASS
#   B:  PASS   = 0.10*33,750                  = 3,375
#       REVIEW = 40.2 + 0.10*0.15*33,750      =   546.45
#       HOLD   = 0.10*0.10*33,750 + 0.90*1,550 = 337.5 + 1,395 = 1,732.5
#       -> argmin is REVIEW

VALUE_INR = np.array([9_000.0, 3_000.0])
LOSS_INR = np.array([0.0, 33_750.0])
POSTERIOR = np.array([0.10, 0.10])

EXPECTED_COSTS_AT_P10 = np.array(
    [
        [0.0, 40.2, 3_285.0],
        [3_375.0, 546.45, 1_732.5],
    ]
)
EXPECTED_ACTIONS_AT_P10 = np.array([PASS, REVIEW])


def test_shipping_constants_are_the_ones_the_fixture_assumes() -> None:
    """The literals above are only meaningful against these central values."""
    assert config.TAU_REVIEW_HOURS == 0.067
    assert config.WAGE_ANALYST_INR_PER_HOUR == 600.0
    assert config.P_ANALYST_MISS == 0.15
    assert config.RESIDUAL_LEAKAGE_RHO == 0.10
    assert config.P_CHURN_GIVEN_HOLD == 0.35
    assert config.COST_SUPPORT_INR == 500.0


def test_bmr_action_matches_hand_computed_costs_and_argmin() -> None:
    """FR-015 + FR-016 acceptance on the 2-merchant fixture."""
    params = CostParams(loss_inr=LOSS_INR, value_inr=VALUE_INR)
    actions, costs = bmr_action(POSTERIOR, params)

    assert costs == pytest.approx(EXPECTED_COSTS_AT_P10)
    assert np.array_equal(actions, EXPECTED_ACTIONS_AT_P10)
    # FR-015: exactly one action per merchant, and the cost of every alternative.
    assert actions.shape == (2,)
    assert costs.shape == (2, 3)


# ---------------------------------------------------------------------------
# Boundary crossing (FR-015 "Verified by: unit test over the decision boundary")
# ---------------------------------------------------------------------------
#
# Merchant B's two boundaries, solved by hand from the same three expressions.
#
# PASS / REVIEW:  p*L = c_rev + p*p_miss*L
#                 p * 33,750 * (1 - 0.15) = 40.2
#                 p* = 40.2 / 28,687.5                         = 0.00140131...
#
# REVIEW / HOLD:  c_rev + p*p_miss*L = p*rho*L + (1-p)*c_fp
#                 40.2 + 5,062.5 p = 3,375 p + 1,550 - 1,550 p
#                 3,237.5 p = 1,509.8
#                 p* = 1,509.8 / 3,237.5                       = 0.46634749...

PASS_REVIEW_BOUNDARY = 40.2 / 28_687.5
REVIEW_HOLD_BOUNDARY = 1_509.8 / 3_237.5


@pytest.mark.parametrize(
    ("p_bad", "expected"),
    [
        (0.001, PASS),  # below the PASS/REVIEW boundary at 0.0014
        (0.002, REVIEW),  # above it
        (0.460, REVIEW),  # below the REVIEW/HOLD boundary at 0.4663
        (0.470, HOLD),  # above it
    ],
)
def test_a_small_posterior_change_flips_the_action(p_bad: float, expected: int) -> None:
    """The action changes at the cost-matrix boundary, not at a tuned threshold."""
    params = CostParams(loss_inr=LOSS_INR[1:], value_inr=VALUE_INR[1:])
    actions, _ = bmr_action(np.array([p_bad]), params)
    assert actions[0] == expected


def test_the_hand_solved_boundaries_are_where_the_costs_actually_cross() -> None:
    """Straddle each boundary by 1e-6 and require the argmin to differ."""
    params = CostParams(loss_inr=LOSS_INR[1:], value_inr=VALUE_INR[1:])
    for boundary, below, above in (
        (PASS_REVIEW_BOUNDARY, PASS, REVIEW),
        (REVIEW_HOLD_BOUNDARY, REVIEW, HOLD),
    ):
        low, _ = bmr_action(np.array([boundary - 1e-6]), params)
        high, _ = bmr_action(np.array([boundary + 1e-6]), params)
        assert (low[0], high[0]) == (below, above)


# ---------------------------------------------------------------------------
# The cost matrix this module owns must be the one eval/metrics.py already ships
# ---------------------------------------------------------------------------


def test_parameterised_cost_matrix_reproduces_eval_metrics_exactly() -> None:
    """`policy.expected_costs` at default params == `metrics.action_cost`.

    Not a re-derivation: it pins that the sweep's parameterised matrix and the
    frozen scoring matrix are the same object seen twice. If they ever diverge,
    `savings` in the summary and `savings` in the sweep stop being comparable.
    """
    from rakshak.eval import metrics

    rng = np.random.default_rng(3)
    n = 40
    y = (rng.random(n) < 0.3).astype(float)
    loss = rng.lognormal(mean=np.log(30_000.0), sigma=1.0, size=n) * y
    value = rng.lognormal(mean=np.log(8_000.0), sigma=0.5, size=n)
    params = CostParams(loss_inr=loss, value_inr=value)

    got = expected_costs(y, params)
    for action in (PASS, REVIEW, HOLD):
        want = metrics.action_cost(y, np.full(n, action), loss, value)
        assert got[:, action] == pytest.approx(want)

    actions = rng.integers(0, 3, size=n)
    assert savings(y, actions, params) == pytest.approx(
        metrics.savings_score(y, actions, loss, value)
    )


# ---------------------------------------------------------------------------
# FR-017 — the global review-capacity constraint
# ---------------------------------------------------------------------------


def _straining_fixture(n: int = 60, seed: int = 5):
    """A population BMR wants to review far more of than the budget allows."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(float)
    loss = np.where(y > 0, rng.uniform(20_000.0, 90_000.0, size=n), 0.0)
    value = rng.uniform(4_000.0, 12_000.0, size=n)
    # Posterior deliberately mid-range everywhere: REVIEW is the argmin for most
    # merchants, so the unconstrained policy overspends the budget.
    posterior = rng.uniform(0.05, 0.40, size=n)
    return y, CostParams(loss_inr=loss, value_inr=value), posterior


def test_capacity_budget_is_never_exceeded() -> None:
    """FR-017: total review time implied by REVIEW actions stays within K hours."""
    _, params, posterior = _straining_fixture()
    for capacity_hours in (0.0, 0.05, 0.2, 0.4, 1.0, 100.0):
        result = bmr_policy(posterior, params, capacity_hours)
        assert result.hours_used <= capacity_hours + 1e-12, (
            f"budget of {capacity_hours} h exceeded: {result.hours_used} h used"
        )
        assert result.n_reviewed <= result.review_slots
        assert set(np.unique(result.actions)) <= {PASS, REVIEW, HOLD}


def test_the_binding_constraint_is_reported_not_silently_dropped() -> None:
    """FR-017's second half. A tight budget and a slack one must not look alike."""
    _, params, posterior = _straining_fixture()
    actions, costs = bmr_action(posterior, params)
    wanted = int((actions == REVIEW).sum())
    assert wanted > 1, "fixture does not strain the budget; the test would be vacuous"

    tight = apply_capacity(actions, costs, capacity_hours=config.TAU_REVIEW_HOURS * 1)
    assert tight.binding_constraint == "capacity"
    assert tight.n_downgraded == wanted - 1
    assert tight.unconstrained_n_reviewed == wanted

    slack = apply_capacity(actions, costs, capacity_hours=config.TAU_REVIEW_HOURS * 1000)
    assert slack.binding_constraint == "none"
    assert slack.n_downgraded == 0
    assert slack.n_reviewed == wanted


def test_downgraded_reviews_go_to_their_own_cheapest_alternative() -> None:
    """A dropped REVIEW is re-decided, not defaulted to PASS."""
    _, params, posterior = _straining_fixture()
    actions, costs = bmr_action(posterior, params)
    result = apply_capacity(actions, costs, capacity_hours=config.TAU_REVIEW_HOURS * 2)
    downgraded = np.flatnonzero((actions == REVIEW) & (result.actions != REVIEW))
    assert downgraded.size == result.n_downgraded
    for i in downgraded:
        cheaper = PASS if costs[i, PASS] <= costs[i, HOLD] else HOLD
        assert result.actions[i] == cheaper


def test_capacity_keeps_the_highest_regret_reviews() -> None:
    """The kept REVIEWs are the ones it would cost most to not review."""
    _, params, posterior = _straining_fixture()
    actions, costs = bmr_action(posterior, params)
    regret = costs[:, [PASS, HOLD]].min(axis=1) - costs[:, REVIEW]
    result = apply_capacity(actions, costs, capacity_hours=config.TAU_REVIEW_HOURS * 3)
    kept = np.flatnonzero(result.actions == REVIEW)
    dropped = np.flatnonzero((actions == REVIEW) & (result.actions != REVIEW))
    assert kept.size == 3
    assert regret[kept].min() >= regret[dropped].max()


# ---------------------------------------------------------------------------
# FR-020 — the cost-asymmetry sweep
# ---------------------------------------------------------------------------


def _sweep_fixture(n: int = 80, seed: int = 13):
    """A population with concentrated realised loss, like the generator's."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[rng.choice(n, size=16, replace=False)] = 1.0
    volume = rng.lognormal(mean=np.log(300_000.0), sigma=0.9, size=n)
    value = config.GROSS_MARGIN_RATE * volume * config.MERCHANT_LIFETIME_MONTHS
    gross_bad = y * volume * rng.uniform(0.2, 3.0, size=n)
    loss = config.CHARGEBACK_REALISATION_RATE * (1.0 + config.ANCILLARY_LOADING_PHI) * gross_bad
    params = CostParams(loss_inr=loss, value_inr=value)
    posteriors = {
        "rules": np.clip(y * 0.5 + rng.normal(0.0, 0.25, size=n) + 0.2, 0.0, 1.0),
        "hmm": np.clip(y * 0.6 + rng.normal(0.0, 0.20, size=n) + 0.2, 0.0, 1.0),
    }
    return y, params, posteriors


def test_asymmetry_range_brackets_the_central_value_and_is_wide() -> None:
    """The range is derived from COST_PRIMITIVE_RANGES, not chosen for a shape."""
    y, params, _ = _sweep_fixture()
    low, central, high = asymmetry_range(y, params)
    assert low < central < high
    # Six primitives move the ratio; their stated ranges span this factor. It is
    # a consequence of 07-math.md §5, not a target: numerator span
    # (0.60/0.15)*(0.0015/0.0008)*(48/18) and denominator span
    # (0.20/0.02)*(1.50/1.20) both enter multiplicatively.
    assert high / low > 100.0


def test_sweep_table_shape_and_columns() -> None:
    y, params, posteriors = _sweep_fixture()
    frame = sweep_cost_asymmetry(y, posteriors, params, capacity_hours=0.32, n_points=5)
    assert list(frame.columns) == list(SWEEP_COLUMNS)
    # 5 log-spaced points plus the inserted central value, times 2 models.
    assert len(frame) == 6 * len(posteriors)
    assert set(frame["model"]) == set(posteriors)
    assert frame["asymmetry"].nunique() == 6


def test_sweep_is_deterministic_at_a_fixed_seed() -> None:
    """NFR-003."""
    y, params, posteriors = _sweep_fixture()
    first = sweep_cost_asymmetry(y, posteriors, params, capacity_hours=0.32, n_points=5)
    second = sweep_cost_asymmetry(y, posteriors, params, capacity_hours=0.32, n_points=5)
    pd.testing.assert_frame_equal(first, second)


def test_capacity_holds_at_every_point_in_the_sweep() -> None:
    """T-0007b `Done when`: the constraint holds across the whole sweep."""
    y, params, posteriors = _sweep_fixture()
    capacity_hours = 0.32
    frame = sweep_cost_asymmetry(y, posteriors, params, capacity_hours, n_points=7)
    assert (frame["hours_used"] <= capacity_hours + 1e-12).all()
    assert set(frame["binding_constraint"]) <= {"capacity", "none"}


def test_oracle_dominance_holds_at_every_point_in_the_sweep() -> None:
    """T-0007b `Done when`. The sweep raises if it does not, so reaching the
    assertions below already proves it; they pin that the check was live."""
    y, params, posteriors = _sweep_fixture()
    frame = sweep_cost_asymmetry(y, posteriors, params, capacity_hours=0.32, n_points=7)
    assert not frame.empty

    # And the invariant really does fire when it should: hand a policy an
    # impossible savings figure at one swept point.
    from rakshak.decision.cost import assert_oracle_dominance
    from rakshak.decision.policy import savings as policy_savings

    with pytest.raises(AssertionError, match="oracle-dominance invariant FAILED"):
        assert_oracle_dominance(
            y,
            params.loss_inr,
            params.value_inr,
            {"toy ceiling": 0.1},
            {"impossible policy": 0.9},
            savings_fn=lambda actions: policy_savings(y, actions, params),
        )


def test_sweep_reports_where_the_review_only_ceiling_stops_being_a_ceiling() -> None:
    """The finding is emitted as data, not swallowed by an assertion.

    `review_knapsack_oracle` may only PASS and REVIEW. Under a low false-positive
    cost, holding is nearly free and averts nearly all loss, so hold-everything
    beats it and it is no longer an upper bound on anything that can hold. That
    boundary is a deliverable, so the sweep carries it in a column rather than
    raising on it.
    """
    y, params, posteriors = _sweep_fixture()
    frame = sweep_cost_asymmetry(y, posteriors, params, capacity_hours=0.32, n_points=9)
    clears = frame.groupby("asymmetry")["knapsack_clears_hold_everything"].first()
    assert not clears.iloc[0], "fixture no longer exercises the low-asymmetry corner"
    assert clears.iloc[-1], "fixture no longer exercises the high-asymmetry corner"
    # And the hindsight ceiling, which does bound everything, is above every model.
    for _, block in frame.groupby("asymmetry"):
        assert (block["savings"] <= block["hindsight_ceiling"] + 1e-9).all()


# ---------------------------------------------------------------------------
# Integration — the harness scores via BMR, not the top-K placeholder
# ---------------------------------------------------------------------------


@needs_data
def test_harness_scores_through_the_bmr_policy_not_top_k() -> None:
    """`harness.budget_policy` is gone and its behaviour is not reproduced.

    The placeholder always spent exactly K reviews and never held anyone. BMR
    decides how many reviews are worth buying and can hold, so a row that holds
    nobody would mean the replacement did not take.
    """
    from rakshak.eval import harness
    from rakshak.eval.splits import load_split

    assert not hasattr(harness, "budget_policy")

    split = load_split("validate")
    row = harness.evaluate_model("rules", split, seed=config.SEED, k=5)
    assert row["n_held"] > 0, "the policy never held anyone; this is still top-K"
    assert row["n_reviewed"] <= 5
    assert row["binding_constraint"] in {"capacity", "none"}
    assert row["hours_used"] <= 5 * config.TAU_REVIEW_HOURS + 1e-12
    assert len(row["posterior"]) == split.n_merchants
