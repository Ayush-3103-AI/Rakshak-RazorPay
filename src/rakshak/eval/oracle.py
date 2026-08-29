"""Perfect-foresight ceilings. Every result is reported as gap-to-oracle.

06-requirements.md §3 names one ceiling: a knapsack allocation of the K
available review-hours, computed with full hindsight over which merchants
actually transitioned. 07-math.md §7 gives it in closed form — with a unit
review cost tau it is exactly solvable by sorting on `y_m * L_m` descending and
taking the top floor(B / tau).

This module computes that, and one more.

**Why a second ceiling exists.** The review budget constrains REVIEW only; HOLD
consumes no analyst hours (07-math.md §7, f_3). So the review-knapsack oracle is
*not* the highest-savings policy available to an agent with perfect hindsight —
a hindsight policy that simply HOLDs every bad merchant scores higher and uses
zero capacity. Publishing gap-to-oracle against the knapsack alone would let a
model post a *negative* gap (i.e. "better than perfect foresight"), which reads
as a bug and is really an artefact of which resource is scarce.
`perfect_hindsight_oracle` is the unconstrained Bayes-optimal-with-truth policy
and is the honest upper bound on the savings score. Both go in the summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rakshak.config import P_ANALYST_MISS, REVIEW_CAPACITY_HOURS, TAU_REVIEW_HOURS
from rakshak.eval.metrics import (
    HOLD,
    PASS,
    REVIEW,
    action_cost,
    savings_score,
)


@dataclass(frozen=True)
class OracleResult:
    """A hindsight ceiling.

    Attributes:
        name: Which ceiling this is.
        actions: Chosen action per merchant, aligned to the input arrays.
        n_reviewed: Merchants sent to REVIEW.
        n_held: Merchants sent to HOLD.
        hours_used: Analyst hours consumed. Units: hours.
        loss_averted_inr: Fraud loss avoided relative to passing everything.
            Units: INR.
        savings: Bahnsen savings score of this policy. Dimensionless.
        capacity_binding: True when the budget actually limited the allocation.
            False means the constraint is slack and precision@K degenerates.
    """

    name: str
    actions: np.ndarray
    n_reviewed: int
    n_held: int
    hours_used: float
    loss_averted_inr: float
    savings: float
    capacity_binding: bool


def review_slots(
    capacity_hours: float = REVIEW_CAPACITY_HOURS, tau_hours: float = TAU_REVIEW_HOURS
) -> int:
    """K — how many merchants the analyst pool can review in one period.

    Args:
        capacity_hours: Analyst hours available (config.REVIEW_CAPACITY_HOURS).
        tau_hours: Hours per review (config.TAU_REVIEW_HOURS).

    Returns:
        floor(capacity_hours / tau_hours), the review budget in merchants.
    """
    return int(math.floor(capacity_hours / tau_hours))


def review_knapsack_oracle(
    y_true: np.ndarray,
    loss_inr: np.ndarray,
    value_inr: np.ndarray,
    capacity_hours: float = REVIEW_CAPACITY_HOURS,
    tau_hours: float = TAU_REVIEW_HOURS,
) -> OracleResult:
    """The frozen ceiling: allocate B review-hours with full hindsight.

    max_S sum_{m in S} y_m * L_m  s.t.  tau * |S| <= B  (07-math.md §7).
    Unit review cost makes this a sort, not a solver.

    Args:
        y_true: Binary ground-truth labels, shape (n,).
        loss_inr: L_m per merchant. Units: INR.
        value_inr: V_m per merchant, for the savings score. Units: INR.
        capacity_hours: B. Units: hours.
        tau_hours: tau. Units: hours per review.

    Returns:
        An `OracleResult` whose actions are REVIEW for the selected merchants
        and PASS for everyone else.
    """
    y = np.asarray(y_true, dtype=float)
    loss = np.asarray(loss_inr, dtype=float)
    n = y.size

    priority = y * loss
    k = min(review_slots(capacity_hours, tau_hours), n)
    # Deterministic tie-break: ascending index among equal priorities.
    order = np.lexsort((np.arange(n), -priority))
    selected = order[:k]
    # Reviewing a merchant with zero recoverable loss is free capacity wasted,
    # not a gain; the oracle would not spend an hour on it.
    selected = selected[priority[selected] > 0.0]

    actions = np.full(n, PASS, dtype=int)
    actions[selected] = REVIEW

    n_reviewed = int(selected.size)
    n_bad_with_loss = int((priority > 0.0).sum())
    return OracleResult(
        name="oracle (review knapsack, perfect foresight)",
        actions=actions,
        n_reviewed=n_reviewed,
        n_held=0,
        hours_used=n_reviewed * tau_hours,
        loss_averted_inr=float((1.0 - P_ANALYST_MISS) * priority[selected].sum()),
        savings=savings_score(y, actions, loss, value_inr),
        capacity_binding=k < n_bad_with_loss,
    )


def perfect_hindsight_oracle(
    y_true: np.ndarray, loss_inr: np.ndarray, value_inr: np.ndarray
) -> OracleResult:
    """Unconstrained ceiling: the cheapest action per merchant given the truth.

    No capacity limit, no uncertainty — for each merchant, pick whichever of
    PASS / REVIEW / HOLD costs least under the known label. This is the true
    upper bound on the savings score and the denominator gap-to-oracle should
    be read against when a model is not itself capacity-limited.

    Args:
        y_true: Binary ground-truth labels, shape (n,).
        loss_inr: L_m per merchant. Units: INR.
        value_inr: V_m per merchant. Units: INR.

    Returns:
        An `OracleResult` with `capacity_binding=False` by construction.
    """
    y = np.asarray(y_true, dtype=float)
    loss = np.asarray(loss_inr, dtype=float)
    n = y.size

    costs = np.stack(
        [action_cost(y, np.full(n, a), loss, value_inr) for a in (PASS, REVIEW, HOLD)]
    )
    actions = costs.argmin(axis=0).astype(int)

    passed_loss = float((y * loss).sum())
    residual = float(action_cost(y, actions, loss, value_inr)[y > 0].sum())
    return OracleResult(
        name="oracle (perfect hindsight, unconstrained)",
        actions=actions,
        n_reviewed=int((actions == REVIEW).sum()),
        n_held=int((actions == HOLD).sum()),
        hours_used=float((actions == REVIEW).sum()) * TAU_REVIEW_HOURS,
        loss_averted_inr=passed_loss - residual,
        savings=savings_score(y, actions, loss, value_inr),
        capacity_binding=False,
    )
