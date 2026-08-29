"""Three-action Bayes-Minimum-Risk policy under a review-capacity budget (T-0007b).

`cost.py` (T-0007a) defines *what a merchant costs*. This module defines *what to
do about it*: `07-math.md` §6's argmin over {PASS, REVIEW, HOLD}, then `07-math.md`
§7's constraint `f_3 = tau * |{REVIEW}| <= B`, then FR-020's sweep of the whole
thing across the cost asymmetry the primitives are uncertain about.

**One cost matrix, four uses.** `expected_costs(p, params)` is linear in `p`, so
the same function serves as the *expected* cost under a posterior (BMR), the
*realised* cost under a 0/1 label (scoring), the Bahnsen denominator (both trivial
policies), and the perfect-hindsight ceiling (argmin at `p = y`). It reproduces
`eval.metrics.action_cost` exactly at default parameters — pinned by a test — and
exists separately only because the sweep must vary parameters that `metrics` reads
from module-level config.

**The capacity constraint is reported, never silently applied** (FR-017). BMR is
solved unconstrained first; if the implied review hours exceed the budget, the
lowest-regret REVIEWs are downgraded to their best unconstrained alternative and
`PolicyResult.binding_constraint` says so. A run in which capacity did nothing and
a run in which it bound must not look the same from the outside.

**What this module deliberately does not do.** It renders no verdict. Whether the
HMM beats `rules` is T-0011's, on the `test` window, which nothing here touches.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from rakshak.config import (
    ANCILLARY_LOADING_PHI,
    CHARGEBACK_REALISATION_RATE,
    COST_PRIMITIVE_RANGES,
    COST_REVIEW_INR,
    COST_SUPPORT_INR,
    GROSS_MARGIN_RATE,
    MERCHANT_LIFETIME_MONTHS,
    P_ANALYST_MISS,
    P_CHURN_GIVEN_HOLD,
    RESIDUAL_LEAKAGE_RHO,
    RESULTS_DIR,
    SEED,
    TAU_REVIEW_HOURS,
)

PASS: Final[int] = 0
REVIEW: Final[int] = 1
HOLD: Final[int] = 2
ACTION_NAMES: Final[tuple[str, ...]] = ("PASS", "REVIEW", "HOLD")

__all__ = [
    "ACTION_NAMES",
    "HOLD",
    "PASS",
    "REVIEW",
    "SWEEP_COLUMNS",
    "CostParams",
    "PolicyResult",
    "apply_capacity",
    "assert_ceilings_dominate",
    "asymmetry_range",
    "bmr_action",
    "bmr_policy",
    "expected_costs",
    "fp_to_loss_asymmetry",
    "hold_threshold",
    "savings",
    "sweep_cost_asymmetry",
]


# ---------------------------------------------------------------------------
# The cost matrix, parameterised (07-math.md §5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostParams:
    """Per-merchant cost primitives for one scoring run. Units: INR unless stated.

    Defaults are `config.py`'s shipping central values, so `CostParams(loss, value)`
    reproduces `eval.metrics.action_cost` exactly. The sweep varies
    `fp_cost_scale` only; every other field is here so a caller *can* vary it, not
    because anything in this repo does.

    Attributes:
        loss_inr: L_m, realised fraud loss per merchant. Units: INR.
        value_inr: V_m, expected lifetime gross margin per merchant. Units: INR.
        cost_review_inr: c_rev, cost of one REVIEW. Units: INR.
        p_analyst_miss: p_miss, probability a review clears a truly-bad merchant.
        residual_leakage_rho: rho, share of loss still leaking after a HOLD.
        p_churn_given_hold: P(churn | wrongly held).
        cost_support_inr: c_support, escalation handling per HOLD. Units: INR.
        tau_review_hours: tau, analyst hours per review. Units: hours.
        fp_cost_scale: Multiplier on the whole false-positive branch c_fp(m).
            1.0 at the cited central primitives; the FR-020 sweep moves this and
            nothing else, so the asymmetry varies without the *absolute size* of
            fraud loss varying with it.
    """

    loss_inr: np.ndarray
    value_inr: np.ndarray
    cost_review_inr: float = COST_REVIEW_INR
    p_analyst_miss: float = P_ANALYST_MISS
    residual_leakage_rho: float = RESIDUAL_LEAKAGE_RHO
    p_churn_given_hold: float = P_CHURN_GIVEN_HOLD
    cost_support_inr: float = COST_SUPPORT_INR
    tau_review_hours: float = TAU_REVIEW_HOURS
    fp_cost_scale: float = 1.0

    @property
    def fp_cost_inr(self) -> np.ndarray:
        """c_fp(m) = scale * (P(churn|hold) * V_m + c_support). Units: INR."""
        value = np.asarray(self.value_inr, dtype=float)
        return self.fp_cost_scale * (self.p_churn_given_hold * value + self.cost_support_inr)


def expected_costs(p_bad: np.ndarray, params: CostParams) -> np.ndarray:
    """Expected cost of each of PASS / REVIEW / HOLD. Units: INR.

    07-math.md §5's matrix, marginalised over P(bad) = `p_bad`::

        E[PASS]   = p * L_m
        E[REVIEW] = c_rev + p * p_miss * L_m
        E[HOLD]   = p * rho * L_m + (1 - p) * c_fp(m)

    Linear in `p_bad`, so passing a 0/1 label vector returns the *realised* cost
    of each action instead of the expected one. That is not a coincidence to be
    tidied away — it is what lets one function serve BMR, scoring, the Bahnsen
    denominator and the hindsight ceiling.

    Args:
        p_bad: Calibrated P(merchant is bad), shape (n,), or a 0/1 label vector.
        params: Cost primitives; `loss_inr`/`value_inr` must be shape (n,).

    Returns:
        Array of shape (n, 3), column order PASS / REVIEW / HOLD. Units: INR.
    """
    p = np.asarray(p_bad, dtype=float)
    loss = np.asarray(params.loss_inr, dtype=float)
    c_fp = params.fp_cost_inr
    return np.stack(
        [
            p * loss,
            params.cost_review_inr + p * params.p_analyst_miss * loss,
            p * params.residual_leakage_rho * loss + (1.0 - p) * c_fp,
        ],
        axis=-1,
    )


def hold_threshold(params: CostParams) -> np.ndarray:
    """p* at which HOLD becomes cheaper than PASS. Dimensionless, per merchant.

    07-math.md §6, Elkan (2001): the optimal threshold is a function of the cost
    matrix, not a hyperparameter. Solving `p L = p rho L + (1-p) c_fp` gives::

        p* = c_fp(m) / (L_m + c_fp(m) - rho L_m)

    Explicitly merchant-dependent, which is the whole argument for the decision
    layer. Reported per sweep point for FR-020(d).

    Args:
        params: Cost primitives.

    Returns:
        Per-merchant PASS/HOLD boundary in [0, 1]. NaN where both L_m and c_fp
        are zero and the boundary is undefined.
    """
    loss = np.asarray(params.loss_inr, dtype=float)
    c_fp = params.fp_cost_inr
    denominator = loss + c_fp - params.residual_leakage_rho * loss
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0.0, c_fp / denominator, np.nan)


# ---------------------------------------------------------------------------
# FR-015 / FR-016 — Bayes Minimum Risk
# ---------------------------------------------------------------------------


def bmr_action(p_bad: np.ndarray, params: CostParams) -> tuple[np.ndarray, np.ndarray]:
    """Bayes Minimum Risk: the argmin action, plus the cost of every alternative.

    FR-015 requires *exactly one* action and the expected cost of each option;
    FR-016 requires that action to be the argmin under the cost matrix. Ties break
    towards the lower action code (PASS < REVIEW < HOLD), which is
    `np.argmin`'s documented behaviour and is deterministic (NFR-003).

    Args:
        p_bad: Calibrated P(merchant is bad), shape (n,).
        params: Cost primitives, arrays shape (n,).

    Returns:
        `(actions, expected_cost_per_action)`. `actions` is shape (n,) with values
        in {PASS, REVIEW, HOLD}; the cost array is shape (n, 3) in INR.
    """
    costs = expected_costs(p_bad, params)
    return costs.argmin(axis=1).astype(int), costs


# ---------------------------------------------------------------------------
# FR-017 — the global review-capacity constraint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResult:
    """The outcome of one policy decision over a whole merchant population.

    Attributes:
        actions: Chosen action per merchant, shape (n,).
        expected_costs: Expected cost of each action, shape (n, 3). Units: INR.
        n_reviewed: Merchants sent to REVIEW after the capacity constraint.
        n_held: Merchants sent to HOLD (consumes no analyst hours).
        hours_used: tau * n_reviewed. Units: hours.
        capacity_hours: B, the analyst-hour budget for the period. Units: hours.
        review_slots: floor(B / tau) — the budget expressed in merchants.
        binding_constraint: "capacity" when the budget forced a downgrade,
            "none" when unconstrained BMR already fitted inside it. FR-017
            requires this be reported, not inferred.
        n_downgraded: REVIEWs the constraint forced to their next-best action.
        unconstrained_n_reviewed: REVIEWs BMR would have chosen with no budget.
            The size of the gap between this and `n_reviewed` is the capacity
            story the project exists to tell.
    """

    actions: np.ndarray
    expected_costs: np.ndarray
    n_reviewed: int
    n_held: int
    hours_used: float
    capacity_hours: float
    review_slots: int
    binding_constraint: str
    n_downgraded: int
    unconstrained_n_reviewed: int


def apply_capacity(
    actions: np.ndarray,
    costs: np.ndarray,
    capacity_hours: float,
    tau_hours: float = TAU_REVIEW_HOURS,
) -> PolicyResult:
    """Enforce `tau * |{REVIEW}| <= B` and report whether the budget bound.

    Unconstrained BMR can ask for more reviews than the analyst pool has hours
    for. When it does, the REVIEWs kept are the ones with the largest *regret* —
    the extra expected cost of not reviewing them, i.e. the best non-REVIEW
    alternative minus REVIEW. Downgrading the smallest regrets first is exactly
    optimal here because every review costs the same tau (07-math.md §7's unit
    review cost), so this is the same sort the knapsack oracle does, on regret
    instead of on hindsight loss.

    HOLD consumes no analyst hours (07-math.md §7, f_3), so a downgraded merchant
    may still be held. That is the honest reading of the constraint and it is why
    `n_held` is reported beside `n_reviewed`.

    Args:
        actions: Unconstrained BMR actions, shape (n,).
        costs: Expected cost of each action, shape (n, 3). Units: INR.
        capacity_hours: B, the analyst-hour budget. Units: hours.
        tau_hours: tau, hours per review. Units: hours.

    Returns:
        A `PolicyResult` whose `hours_used` never exceeds `capacity_hours`.
    """
    if tau_hours <= 0.0:
        raise ValueError(f"tau_hours must be positive, got {tau_hours!r}")
    actions = np.asarray(actions, dtype=int).copy()
    n = actions.size
    slots = min(int(np.floor(capacity_hours / tau_hours)), n)
    reviewing = np.flatnonzero(actions == REVIEW)
    unconstrained = int(reviewing.size)

    n_downgraded = 0
    if unconstrained > slots:
        alternatives = costs[:, [PASS, HOLD]]
        best_alternative = alternatives.min(axis=1)
        regret = best_alternative - costs[:, REVIEW]
        # Deterministic tie-break: ascending index among equal regrets (NFR-003).
        order = np.lexsort((reviewing, -regret[reviewing]))
        dropped = reviewing[order[slots:]]
        fallback = np.array([PASS, HOLD])[alternatives[dropped].argmin(axis=1)]
        actions[dropped] = fallback
        n_downgraded = int(dropped.size)

    n_reviewed = int((actions == REVIEW).sum())
    return PolicyResult(
        actions=actions,
        expected_costs=costs,
        n_reviewed=n_reviewed,
        n_held=int((actions == HOLD).sum()),
        hours_used=n_reviewed * tau_hours,
        capacity_hours=float(capacity_hours),
        review_slots=slots,
        binding_constraint="capacity" if n_downgraded else "none",
        n_downgraded=n_downgraded,
        unconstrained_n_reviewed=unconstrained,
    )


def bmr_policy(
    p_bad: np.ndarray, params: CostParams, capacity_hours: float
) -> PolicyResult:
    """The scored policy: unconstrained BMR, then the capacity constraint.

    This replaces `eval.harness.budget_policy`, the placeholder that spent the
    whole budget on the top-K scores and never held anyone.

    Args:
        p_bad: Calibrated P(merchant is bad), shape (n,).
        params: Cost primitives, arrays shape (n,).
        capacity_hours: B for this period. Units: hours.

    Returns:
        A `PolicyResult`.
    """
    actions, costs = bmr_action(p_bad, params)
    return apply_capacity(actions, costs, capacity_hours, params.tau_review_hours)


# ---------------------------------------------------------------------------
# Scoring under swept parameters (Bahnsen et al. 2016, 07-math.md §6)
# ---------------------------------------------------------------------------


def savings(y_true: np.ndarray, actions: np.ndarray, params: CostParams) -> float:
    """Bahnsen savings score under `params`. Dimensionless.

    Identical to `eval.metrics.savings_score` at default parameters — that
    equivalence is pinned by a test rather than by comment — and exists because
    the sweep must vary primitives that `metrics` reads from module-level config.

    Args:
        y_true: Binary labels, shape (n,).
        actions: One of PASS/REVIEW/HOLD per merchant, shape (n,).
        params: Cost primitives.

    Returns:
        `(Cost_l - Cost(f)) / Cost_l`, with `Cost_l = min(all-PASS, all-HOLD)`.
        0.0 when `Cost_l` is 0.
    """
    y = np.asarray(y_true, dtype=float)
    realised = expected_costs(y, params)
    rows = np.arange(y.size)
    cost_f = float(realised[rows, np.asarray(actions, dtype=int)].sum())
    cost_l = min(float(realised[:, PASS].sum()), float(realised[:, HOLD].sum()))
    if cost_l <= 0.0:
        return 0.0
    return (cost_l - cost_f) / cost_l


KNAPSACK_CEILING: Final[str] = "oracle (review knapsack, perfect foresight)"
HINDSIGHT_CEILING: Final[str] = "oracle (perfect hindsight, unconstrained)"


def _ceiling_savings(
    y_true: np.ndarray, params: CostParams, capacity_hours: float
) -> dict[str, float]:
    """The two T-0007a ceilings, recomputed under `params`.

    Re-derived here rather than imported from `eval.oracle` for one reason only:
    `oracle.py` scores through `metrics.savings_score`, which reads the cost
    primitives from module-level config and therefore cannot follow the sweep.
    The allocations are the same ones — knapsack on `y * L` (07-math.md §7),
    and the per-merchant argmin under the known label.
    """
    y = np.asarray(y_true, dtype=float)
    loss = np.asarray(params.loss_inr, dtype=float)
    realised = expected_costs(y, params)

    priority = y * loss
    slots = min(int(np.floor(capacity_hours / params.tau_review_hours)), y.size)
    order = np.lexsort((np.arange(y.size), -priority))
    selected = order[:slots]
    selected = selected[priority[selected] > 0.0]
    knapsack = np.full(y.size, PASS, dtype=int)
    knapsack[selected] = REVIEW

    return {
        KNAPSACK_CEILING: savings(y, knapsack, params),
        HINDSIGHT_CEILING: savings(y, realised.argmin(axis=1).astype(int), params),
    }


def assert_ceilings_dominate(
    y_true: np.ndarray,
    params: CostParams,
    ceilings: Mapping[str, float],
    policy_savings: Mapping[str, float],
    seed: int = SEED,
    tol: float = 1e-9,
) -> dict[str, float]:
    """T-0007a's invariant, applied to the action class each ceiling actually bounds.

    **This is a correction T-0007b was forced to make, and it is a finding, not a
    convenience.** T-0007a asserted every policy against *both* ceilings. That was
    sound while the scored policy was `harness.budget_policy`, which only ever
    PASSes and REVIEWs. T-0007b's BMR policy can HOLD, and `review_knapsack_oracle`
    is by construction the best *review-only, <= K* allocation — it is not an upper
    bound on any policy allowed to hold. T-0007a's own test file wrote this down:
    *"nothing forces it above hold-everything"*. Under a low cost asymmetry, holding
    is nearly free and the review-only ceiling falls below hold-everything, so
    asserting a holding policy against it fires on a category error rather than on a
    defect.

    So the invariant is split, not weakened:

    * `HINDSIGHT_CEILING` bounds **every** policy — it is a per-merchant argmin over
      the whole action set. T-0007a already noted that its passing proves little.
    * `KNAPSACK_CEILING` bounds only the review-only class. `pass-everything` is the
      one member of that class present in any run, and it is checked.

    Whether the knapsack ceiling clears hold-everything is reported by the caller at
    every swept point, never asserted away.

    Args:
        y_true: Binary labels, shape (n,).
        params: Cost primitives in force for this scoring.
        ceilings: Ceiling name -> savings, as returned by `_ceiling_savings`.
        policy_savings: Scored model name -> savings, under the same `params`.
        seed: Seed for the random trivial policy (NFR-003).
        tol: Absolute slack, to absorb float noise only.

    Returns:
        The full policy name -> savings mapping that was checked.

    Raises:
        AssertionError: If any policy beats a ceiling that genuinely bounds it.
    """
    from rakshak.decision.cost import assert_oracle_dominance

    y = np.asarray(y_true, dtype=float)
    score = lambda actions: savings(y, actions, params)  # noqa: E731

    checked = assert_oracle_dominance(
        y,
        params.loss_inr,
        params.value_inr,
        {HINDSIGHT_CEILING: ceilings[HINDSIGHT_CEILING]},
        policy_savings,
        seed=seed,
        tol=tol,
        savings_fn=score,
    )

    knapsack = ceilings[KNAPSACK_CEILING]
    pass_everything = checked["trivial: pass-everything"]
    if pass_everything > knapsack + tol:
        raise AssertionError(
            "oracle-dominance invariant FAILED for the review-only class "
            "(T-0007a; 07-math.md §7).\n  'trivial: pass-everything' scores "
            f"{pass_everything:+.4f} > ceiling {KNAPSACK_CEILING!r} "
            f"at {knapsack:+.4f}.\n"
            "Perfect-foresight review allocation cannot be worse than reviewing "
            "nobody. This is a mis-specified cost matrix, not a bad oracle. Do NOT "
            "tune constants until it passes — read 07-math.md §5 and report it."
        )
    return checked


# ---------------------------------------------------------------------------
# FR-020 — the cost-asymmetry sweep
# ---------------------------------------------------------------------------


def fp_to_loss_asymmetry(y_true: np.ndarray, params: CostParams) -> float:
    """INR of false-positive cost per INR 100 of fraud loss, on this population.

    The same quantity `cost.fp_cost_per_100_of_fraud_loss` reports, expressed
    against `params` so the sweep can target it. Numerator: c_fp over every truly
    healthy merchant. Denominator: L_m over every truly bad one.

    Args:
        y_true: Binary labels, shape (n,).
        params: Cost primitives.

    Returns:
        The ratio; NaN when no merchant is bad.
    """
    y = np.asarray(y_true, dtype=float)
    total_fp = float(params.fp_cost_inr[y == 0.0].sum())
    total_loss = float(np.asarray(params.loss_inr, dtype=float)[y == 1.0].sum())
    return 100.0 * total_fp / total_loss if total_loss > 0.0 else float("nan")


def asymmetry_range(y_true: np.ndarray, params: CostParams) -> tuple[float, float, float]:
    """The plausible FP-per-100 asymmetry range implied by `COST_PRIMITIVE_RANGES`.

    **Derived, not chosen.** FR-020 requires the sweep to span "the full plausible
    range implied by the per-primitive ranges in 07-math.md §5", and `T-0007b`
    forbids narrowing it because part of it is unflattering. Six primitives move
    this ratio and no others do:

    * numerator, c_fp(m) = P(churn|hold) * g * v_m * l_m + c_support —
      `P_CHURN_GIVEN_HOLD`, `GROSS_MARGIN_RATE`, `MERCHANT_LIFETIME_MONTHS`,
      `COST_SUPPORT_INR`;
    * denominator, L_m = r_cb * (1 + phi) * G_bad_m —
      `CHARGEBACK_REALISATION_RATE`, `ANCILLARY_LOADING_PHI`.

    `g` and `l_m` enter `V_m` multiplicatively and `r_cb`, `phi` enter `L_m`
    multiplicatively, so moving them is a pure rescale of the shipping
    `value_inr` / `loss_inr` and needs no regeneration of the split.

    The low end puts every numerator primitive at the bottom of its stated range
    and every denominator primitive at the top; the high end reverses it. Those
    two corners are the extreme asymmetries the cited spread admits. Nothing
    between them is excluded, and the endpoints are whatever the population makes
    them — this function has no literal in it.

    Args:
        y_true: Binary labels, shape (n,).
        params: Cost primitives at their central values.

    Returns:
        `(low, central, high)` asymmetry in INR of FP cost per INR 100 of loss.
    """
    lo_of = {name: COST_PRIMITIVE_RANGES[name][0] for name in COST_PRIMITIVE_RANGES}
    hi_of = {name: COST_PRIMITIVE_RANGES[name][1] for name in COST_PRIMITIVE_RANGES}

    def corner(numerator: Mapping[str, float], denominator: Mapping[str, float]) -> float:
        value_scale = (
            numerator["GROSS_MARGIN_RATE"] / GROSS_MARGIN_RATE
        ) * (numerator["MERCHANT_LIFETIME_MONTHS"] / MERCHANT_LIFETIME_MONTHS)
        loss_scale = (
            denominator["CHARGEBACK_REALISATION_RATE"] / CHARGEBACK_REALISATION_RATE
        ) * (
            (1.0 + denominator["ANCILLARY_LOADING_PHI"]) / (1.0 + ANCILLARY_LOADING_PHI)
        )
        return fp_to_loss_asymmetry(
            y_true,
            replace(
                params,
                value_inr=np.asarray(params.value_inr, dtype=float) * value_scale,
                loss_inr=np.asarray(params.loss_inr, dtype=float) * loss_scale,
                p_churn_given_hold=numerator["P_CHURN_GIVEN_HOLD"],
                cost_support_inr=numerator["COST_SUPPORT_INR"],
                fp_cost_scale=1.0,
            ),
        )

    return (
        corner(lo_of, hi_of),
        fp_to_loss_asymmetry(y_true, params),
        corner(hi_of, lo_of),
    )


SWEEP_POINTS: Final[int] = 9
"""Points sampled across the derived asymmetry range, log-spaced.

The range spans more than an order of magnitude, so log spacing gives each
decade the same resolution. 9 is enough to locate a crossing to within a few per
cent of asymmetry and cheap enough to leave `make eval` well inside NFR-004."""


def sweep_cost_asymmetry(
    y_true: np.ndarray,
    posteriors: Mapping[str, np.ndarray],
    params: CostParams,
    capacity_hours: float,
    seed: int = SEED,
    n_points: int = SWEEP_POINTS,
    reference_model: str = "rules",
    proposal_model: str = "hmm",
) -> pd.DataFrame:
    """Re-score every model across the derived cost-asymmetry range (FR-020).

    Holds the analyst-hour budget fixed and moves only the false-positive branch
    of the cost matrix, so the sweep isolates the asymmetry rather than
    confounding it with the absolute size of fraud loss. At each point the
    oracle-dominance invariant is re-checked (T-0007a) before any row is emitted.

    **No verdict is rendered here.** The margin column is a measurement; whether
    it clears NFR-001's >=20% bar is T-0011's call, on the `test` window.

    Args:
        y_true: Binary labels, shape (n,).
        posteriors: model name -> calibrated P(bad), shape (n,), aligned to
            `y_true`. Iteration order is preserved in the output.
        params: Cost primitives at their central values.
        capacity_hours: B, held fixed across the sweep. Units: hours.
        seed: Passed to the oracle-dominance invariant's random trivial policy.
        n_points: Asymmetry points sampled log-uniformly across the derived range.
        reference_model: The floor the margin is measured against (`rules`).
        proposal_model: The model whose margin is reported (`hmm`).

    Returns:
        A tidy frame, one row per (asymmetry, model), with the margin columns
        repeated on every row of a point so the table can be filtered on either
        axis. Columns: asymmetry, fp_cost_scale, model, savings, n_reviewed,
        n_held, hours_used, binding_constraint, hold_threshold_median,
        knapsack_ceiling, hindsight_ceiling, knapsack_clears_hold_everything,
        margin_abs, margin_rel.

    Raises:
        AssertionError: If a policy beats a perfect-foresight ceiling at any
            swept point (T-0007a's invariant).
    """
    y = np.asarray(y_true, dtype=float)
    low, central, high = asymmetry_range(y, params)
    if not np.isfinite(central) or central <= 0.0:
        raise ValueError(f"central asymmetry is not usable: {central!r}")
    # Log-spaced across the derived range, with the central value inserted so the
    # cited operating point is always a row and is never interpolated.
    grid = np.unique(
        np.concatenate(
            [np.geomspace(low, high, num=int(n_points)), np.array([central])]
        )
    )

    rows: list[dict[str, object]] = []
    for asymmetry in grid:
        point = replace(params, fp_cost_scale=float(asymmetry / central))
        ceilings = _ceiling_savings(y, point, capacity_hours)
        # Two medians, because one of them is degenerate and saying so is the
        # point. A merchant with G_bad = 0 has L_m = 0, so p* = c_fp / c_fp = 1
        # exactly — "never hold this merchant", correct but uninformative. 80 of
        # the 100 validate merchants are in that state, so the whole-population
        # median is pinned at 1.0 at every asymmetry and would read as "the
        # threshold does not move". The at-risk median (L_m > 0) is the one
        # FR-020(d) is asking about; both are emitted so neither can be quoted
        # without the other.
        thresholds = hold_threshold(point)
        at_risk = np.asarray(point.loss_inr, dtype=float) > 0.0
        threshold_median = float(np.nanmedian(thresholds))
        threshold_median_at_risk = (
            float(np.nanmedian(thresholds[at_risk])) if at_risk.any() else float("nan")
        )

        scored: dict[str, float] = {}
        detail: dict[str, PolicyResult] = {}
        for name, posterior in posteriors.items():
            result = bmr_policy(np.asarray(posterior, dtype=float), point, capacity_hours)
            detail[name] = result
            scored[name] = savings(y, result.actions, point)

        checked = assert_ceilings_dominate(y, point, ceilings, scored, seed=seed)
        hold_everything = checked["trivial: hold-everything"]

        reference = scored.get(reference_model, float("nan"))
        proposal = scored.get(proposal_model, float("nan"))
        margin_abs = proposal - reference
        # Relative margin is what NFR-001 states, and it is numerically unstable
        # when the reference sits near zero. Reported as NaN rather than as a
        # large number, with the absolute margin beside it, so a divide-by-almost
        # -zero can never be read as a result.
        margin_rel = (
            margin_abs / abs(reference) if abs(reference) > 1e-6 else float("nan")
        )

        for name, value in scored.items():
            result = detail[name]
            rows.append(
                {
                    "asymmetry": float(asymmetry),
                    "fp_cost_scale": point.fp_cost_scale,
                    "model": name,
                    "savings": value,
                    "n_reviewed": result.n_reviewed,
                    "n_held": result.n_held,
                    "hours_used": result.hours_used,
                    "binding_constraint": result.binding_constraint,
                    "hold_threshold_median": threshold_median,
                    "hold_threshold_median_at_risk": threshold_median_at_risk,
                    "knapsack_ceiling": ceilings[KNAPSACK_CEILING],
                    "hindsight_ceiling": ceilings[HINDSIGHT_CEILING],
                    "knapsack_clears_hold_everything": bool(
                        ceilings[KNAPSACK_CEILING] >= hold_everything - 1e-9
                    ),
                    "margin_abs": margin_abs,
                    "margin_rel": margin_rel,
                }
            )
    return pd.DataFrame(rows)


SWEEP_COLUMNS: Final[tuple[str, ...]] = (
    "asymmetry",
    "fp_cost_scale",
    "model",
    "savings",
    "n_reviewed",
    "n_held",
    "hours_used",
    "binding_constraint",
    "hold_threshold_median",
    "hold_threshold_median_at_risk",
    "knapsack_ceiling",
    "hindsight_ceiling",
    "knapsack_clears_hold_everything",
    "margin_abs",
    "margin_rel",
)
"""The sweep's tidy-table contract. Pinned by a test so a rename is visible."""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_sensitivity(frame: pd.DataFrame, y_true: np.ndarray, params: CostParams,
                       capacity_hours: float, seed: int) -> str:
    """Build `results/sensitivity.md`. Byte-identical for a fixed seed (NFR-003)."""
    low, central, high = asymmetry_range(np.asarray(y_true, dtype=float), params)
    models = list(dict.fromkeys(frame["model"]))
    lines: list[str] = []
    add = lines.append

    add("# Rakshak — cost-asymmetry sensitivity (FR-020)")
    add("")
    add(
        "> **Sequence-layer metrics are measured on synthetic merchant streams with "
        "injected typologies; the generator is in this repo.**"
    )
    add("")
    add(
        "**No verdict is rendered here.** This table is the machinery FR-020 requires; "
        "T-0011 runs it on the `test` window and states the boundary. Everything below "
        "is the `validate` window."
    )
    add("")
    add("## The figure (FR-020)")
    add("")
    add("![Cost-asymmetry sensitivity](figures/sensitivity.png)")
    add("")
    add(
        "Drawn by `rakshak.eval.figures` from `results/sensitivity.csv`, which is the "
        "same frame that produced every table below -- the figure computes nothing of "
        "its own and cannot disagree with the tables. Regenerate it alone with "
        "`make figures`, which refits no model. **FR-020's figure clause had no owner "
        "after T-0010 was cut; it was assigned to this renderer on 2026-08-29 rather "
        "than struck.**"
    )
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(f"| Produced by | `python -m rakshak.decision.policy --seed {seed}` |")
    add(f"| Seed | {seed} |")
    add(f"| Analyst-hour budget B | {capacity_hours:.2f} h, held fixed across the sweep |")
    add(f"| Derived asymmetry range | {low:.1f} - {high:.1f} INR FP cost per INR 100 loss |")
    add(f"| Cited central asymmetry | {central:.1f} |")
    add("| 07-math.md §5 commentary band (cross-check, **not** a gate) | 400 - 600 |")
    add("")
    add(
        f"**FR-020(c): the cited primitives produce {central:.1f}, against a commentary "
        "band of 400-600. The divergence is stated, not closed.** The band measures "
        "*falsely declined baskets at checkout*, where the denied item is the full "
        "basket value; this ratio measures *held merchant settlements*, where the cost "
        "is the platform's own ~10 bps margin over that merchant's remaining lifetime "
        "and the fraud side is realised chargebacks rather than an abandoned cart. They "
        "were never the same asymmetry. No primitive was moved toward the band — "
        f"07-math.md §5 forbids it. The swept range runs {low:.1f} - {high:.1f}, so the "
        "highest-asymmetry rows below are the closest this repo can get to what the "
        "commentary band would imply if it did apply here — read them as an "
        "illustration, not as a second operating point."
    )
    add("")
    add(
        "The range is **derived, not chosen**: the low end puts every numerator "
        "primitive (`P_CHURN_GIVEN_HOLD`, `GROSS_MARGIN_RATE`, "
        "`MERCHANT_LIFETIME_MONTHS`, `COST_SUPPORT_INR`) at the bottom of its "
        "`07-math.md` §5 range and every denominator primitive "
        "(`CHARGEBACK_REALISATION_RATE`, `ANCILLARY_LOADING_PHI`) at the top; the "
        "high end reverses it. Nothing between the corners is excluded and no "
        "endpoint is a literal."
    )
    add("")
    add(
        "**How each point is reached is not how the endpoints were derived, and "
        "that is a caveat on this table.** `asymmetry_range` reaches its corners "
        "by rescaling `value_inr` *and* `loss_inr` together with the six "
        "primitives. The sweep reproduces each asymmetry by moving the "
        "false-positive branch alone (`fp_cost_scale`), which isolates the "
        "asymmetry instead of confounding it with the absolute size of fraud "
        "loss. The two routes agree on the **ratio** and not on the whole cost "
        "matrix: `cost_review_inr` is an analyst wage and rescales with neither, "
        "so at a swept point the REVIEW branch sits at a different price "
        "relative to loss than it would at the corner that produced that same "
        "ratio. Read a row as *\"this FP:loss ratio, review priced at its "
        "shipping absolute cost\"*, not as *\"the world in which those six "
        "primitives take their corner values\"*. The savings **ordering** across "
        "models at a point is unaffected -- every model at a point faces the "
        "identical matrix -- but the asymmetry at which a crossing occurs is "
        "specific to this parameterisation."
    )
    add("")

    add("## Savings by asymmetry")
    add("")
    add("| asymmetry | " + " | ".join(models) + " | margin abs (hmm - rules) | margin rel |")
    add("|---|" + "---|" * (len(models) + 2))
    for asymmetry, block in frame.groupby("asymmetry", sort=True):
        by_model = dict(zip(block["model"], block["savings"], strict=True))
        cells = " | ".join(f"{by_model.get(m, float('nan')):+.4f}" for m in models)
        margin_abs = float(block["margin_abs"].iloc[0])
        margin_rel = float(block["margin_rel"].iloc[0])
        rel = "n/a" if np.isnan(margin_rel) else f"{margin_rel:+.1%}"
        add(f"| {asymmetry:.1f} | {cells} | {margin_abs:+.4f} | {rel} |")
    add("")
    add(
        "`margin rel` reads `n/a` where the `rules` baseline sits within 1e-6 of zero: "
        "NFR-001's >=20% bar is a *relative* margin, and a relative margin over a "
        "near-zero denominator is not a number worth printing. The absolute margin is "
        "beside it and is always defined."
    )
    add("")
    if "random" in models:
        add(
            "**Read the `random` column before any other.** It is a uniform random "
            "score, and under BMR it posts positive savings across the whole range and "
            "beats `rules` at the low end. That is the cost matrix earning the savings, "
            "not detection — 07-math.md §6's AP-06 guard, measured. Any margin quoted "
            "off this table must be quoted against the `random` floor, not against zero."
        )
        add("")
    crossings = frame.drop_duplicates("asymmetry").sort_values("asymmetry")
    positive = crossings[crossings["margin_abs"] > 0.0]
    if len(positive) and len(positive) < len(crossings):
        first = float(positive["asymmetry"].iloc[0])
        below = float(crossings[crossings["margin_abs"] <= 0.0]["asymmetry"].max())
        add(
            f"On this split the `hmm - rules` margin turns positive between asymmetry "
            f"{below:.1f} and {first:.1f}, and is negative below that. **The boundary "
            "is the deliverable and it is stated rather than narrowed away** (T-0007b). "
            "It is not a verdict: this is the `validate` window, the sweep varies only "
            "the false-positive branch, and whether `00-charter.md` §2's conditional "
            "claim holds is T-0011's call on `test`."
        )
        add("")

    add("## Capacity and thresholds (FR-017, FR-020(d))")
    add("")
    add(
        f"Budget B = {capacity_hours:.2f} h = {int(capacity_hours // TAU_REVIEW_HOURS)} "
        "review slots, identical at every point."
    )
    add("")
    add(
        "| asymmetry | p* median (at risk) | p* median (all) | capacity binds for | "
        "max reviewed | max held |"
    )
    add("|---|---|---|---|---|---|")
    for asymmetry, block in frame.groupby("asymmetry", sort=True):
        bound = [
            str(m)
            for m, c in zip(block["model"], block["binding_constraint"], strict=True)
            if c == "capacity"
        ]
        add(
            f"| {asymmetry:.1f} | "
            f"{float(block['hold_threshold_median_at_risk'].iloc[0]):.4f} | "
            f"{float(block['hold_threshold_median'].iloc[0]):.4f} | "
            f"{', '.join(bound) if bound else 'none'} | "
            f"{int(block['n_reviewed'].max())} | {int(block['n_held'].max())} |"
        )
    add("")
    add(
        "FR-017: `hours_used` never exceeds B at any point above, and the binding "
        "constraint is reported **per model** rather than inferred — a run where the "
        "budget did nothing and a run where it forced a downgrade must not look alike."
    )
    add("")
    add(
        "FR-020(d): p* = c_fp(m) / (L_m + c_fp(m) - rho L_m) is Elkan (2001)'s "
        "cost-matrix-derived threshold. **The `all` column is pinned at 1.0000 at "
        "every asymmetry and that is not a result** — 80 of these 100 merchants "
        "never transact in a bad state, so L_m = 0 and p* collapses to c_fp/c_fp = 1 "
        "by construction. The `at risk` column (L_m > 0) is the one that moves. Both "
        "are printed so the degenerate one cannot be quoted on its own."
    )
    add("")

    add("## Where the review-only ceiling stops being a ceiling")
    add("")
    points = frame.drop_duplicates("asymmetry").sort_values("asymmetry")
    add("| asymmetry | knapsack ceiling | hindsight ceiling | knapsack >= hold-everything |")
    add("|---|---|---|---|")
    for _, row in points.iterrows():
        add(
            f"| {float(row['asymmetry']):.1f} | {float(row['knapsack_ceiling']):+.4f} | "
            f"{float(row['hindsight_ceiling']):+.4f} | "
            f"{'yes' if bool(row['knapsack_clears_hold_everything']) else '**no**'} |"
        )
    add("")
    invalid = points[~points["knapsack_clears_hold_everything"].astype(bool)]
    if len(invalid):
        add(
            f"**The review-knapsack ceiling falls below hold-everything at every "
            f"asymmetry at or below {float(invalid['asymmetry'].max()):.1f}.** T-0007a "
            "predicted exactly this in `tests/test_cost.py`'s header — *\"nothing "
            "forces it above hold-everything\"* — and this sweep is the first thing to "
            "measure where the boundary is. It is a property of the action class, not "
            "a defect: `review_knapsack_oracle` may only PASS and REVIEW, and under a "
            "low false-positive cost, holding is nearly free and averts nearly all "
            "loss. So the review-only ceiling is beaten by hold-everything, by every "
            "model, and by any policy allowed to hold."
        )
    else:
        add(
            "The review-knapsack ceiling clears hold-everything at every swept point "
            "on this split."
        )
    add("")
    add(
        "**This forced a correction to T-0007a's invariant and it is recorded rather "
        "than smoothed over.** T-0007a asserted every policy against both ceilings, "
        "which was sound while the scored policy was `harness.budget_policy` (PASS and "
        "REVIEW only). T-0007b's BMR policy holds, so asserting it against a "
        "review-only ceiling fires on a category error. `assert_ceilings_dominate` now "
        "checks the hindsight ceiling against every policy and the knapsack ceiling "
        "against the review-only class, and both ceilings are reported at every point. "
        "**No constant was moved and no point was dropped from the sweep.**"
    )
    add("")
    add(
        "With that scoping, the oracle-dominance invariant was re-checked at **every** "
        "point above, against ceilings recomputed under that point's own cost matrix, "
        "and held at every one. The table exists only because it did."
    )
    add("")
    return "\n".join(lines) + "\n"


def run(seed: int = SEED, results_dir: Path = RESULTS_DIR) -> Path:
    """Run the sweep on `validate` and write `sensitivity.md`. Returns the path."""
    from rakshak.eval.harness import EVAL_SPLIT, MODEL_REGISTRY, _model_rng, _normalise
    from rakshak.eval.splits import load_split

    split = load_split(EVAL_SPLIT)
    capacity_hours = review_capacity_hours(split.n_merchants)
    params = CostParams(
        loss_inr=split.loss_inr.to_numpy(dtype=float),
        value_inr=split.value_inr.to_numpy(dtype=float),
    )
    posteriors = {
        name: np.clip(
            _normalise(scorer(split, _model_rng(seed, name)), split)["score"].to_numpy(
                dtype=float
            ),
            0.0,
            1.0,
        )
        for name, scorer in MODEL_REGISTRY.items()
    }
    y = split.labels.to_numpy(dtype=float)
    frame = sweep_cost_asymmetry(y, posteriors, params, capacity_hours, seed=seed)

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "sensitivity.md"
    path.write_text(
        render_sensitivity(frame, y, params, capacity_hours, seed),
        encoding="utf-8",
        newline="\n",
    )
    return path


def review_capacity_hours(n_merchants: int) -> float:
    """B for a population of `n_merchants`. Units: hours. ADR-0008."""
    from rakshak.config import REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS

    return REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS * n_merchants / 1000.0


def main(argv: list[str] | None = None) -> int:
    """Run the cost-asymmetry sweep. Returns a process exit code."""
    from rakshak.cli import base_parser, seed_everything

    args = base_parser("Run the FR-020 cost-asymmetry sweep on the validate split.").parse_args(
        argv
    )
    seed_everything(args.seed)
    path = run(args.seed)
    print(f"rakshak: wrote {path} (seed={args.seed})")
    print(
        "rakshak: validate split only; no verdict is rendered here - T-0011 runs this "
        "on test and states the boundary."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
