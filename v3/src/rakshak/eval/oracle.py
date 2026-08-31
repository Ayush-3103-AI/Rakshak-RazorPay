"""The perfect-foresight oracle: the ceiling, not the target (10-eval-harness-spec.md §3).

Given true labels and ``true_loss_amount_inr``, pick, for each day, the set of merchants
under the analyst budget that maximises prevented loss. Its purpose is to convert every
result into a **gap to what was achievable**, so that "we captured 68% of the achievable
savings" replaces "our savings score was 0.41", which means nothing to anyone.

With a uniform review cost this is a top-K selection. With the budget expressed in analyst
*hours* it is a 0/1 knapsack, and the knapsack is implemented — it is twenty lines, it
generalises, and the uniform case is proven against it in ``tests/unit/test_oracle.py``
rather than assumed.

The sanity assertion is the load-bearing part of this module:

    oracle_savings >= any_rung_savings

The oracle sees the labels. Nothing that does not see the labels can beat it. A rung that
does is leaking, and :func:`assert_no_leakage` turns that from something you might notice
into something that stops the run.
"""

from __future__ import annotations

import math

import numpy as np

from rakshak.eval.metrics import (
    CostParams,
    savings_of_actions,
    top_k_by_day,
)
from rakshak.schemas import Action

__all__ = [
    "LeakageError",
    "assert_no_leakage",
    "gap_to_oracle",
    "intervention_benefit",
    "knapsack",
    "oracle_actions",
    "oracle_savings",
]


class LeakageError(RuntimeError):
    """A rung beat the perfect-foresight oracle.

    This is never a good result and it is never a close call. The oracle is given the
    labels; a model that is not, and still wins, is reading something it should not be
    able to see. Historically this assertion catches: a feature computed from a
    forward-looking window, a label joined on ``label_event_at`` instead of
    ``label_available_at``, and a merchant appearing in two splits.
    """


def knapsack(values: np.ndarray, weights: np.ndarray, capacity: int) -> np.ndarray:
    """Exact 0/1 knapsack. Returns the boolean take-mask maximising total value.

    ``weights`` must be non-negative integers (analyst-hours). Items with non-positive
    value are never taken — spending capacity to lose money is not an optimum the oracle
    is allowed to find.
    """
    if capacity < 0:
        raise ValueError(f"capacity must be >= 0; got {capacity!r}")
    n = values.size
    mask = np.zeros(n, dtype=bool)
    if n == 0 or capacity == 0:
        return mask
    if (weights < 0).any():
        raise ValueError("analyst-hour weights must be non-negative")

    best = np.zeros(capacity + 1)
    take = np.zeros((n, capacity + 1), dtype=bool)
    for i in range(n):
        w = int(weights[i])
        v = float(values[i])
        if w > capacity or v <= 0.0:
            continue
        candidate = np.full(capacity + 1, -np.inf)
        candidate[w:] = best[: capacity + 1 - w] + v
        better = candidate > best
        take[i] = better
        best = np.where(better, candidate, best)

    remaining = capacity
    for i in range(n - 1, -1, -1):
        if take[i, remaining]:
            mask[i] = True
            remaining -= int(weights[i])
    return mask


def intervention_benefit(
    y: np.ndarray, loss: np.ndarray, params: CostParams
) -> tuple[np.ndarray, np.ndarray]:
    """``(benefit, best_action)`` per row, with the truth known.

    ``benefit = cost(PASS) - min(cost(REVIEW), cost(HOLD))``. For a fraud row that is
    ``loss - review_cost`` (HOLD stops it outright); for a good row it is ``-review_cost``,
    which is why the oracle never touches a clean merchant even with capacity to spare.
    """
    fraud = y == 1
    scaled = loss * params.fraud_loss_multiplier

    cost_pass = np.where(fraud, scaled, 0.0)
    cost_review = np.where(
        fraud, params.review_cost_inr + (1.0 - params.p_catch) * scaled, params.review_cost_inr
    )
    cost_hold = np.where(
        fraud, params.review_cost_inr, params.false_hold_cost_inr + params.review_cost_inr
    )
    best_action = np.where(cost_hold <= cost_review, Action.HOLD, Action.REVIEW)
    return cost_pass - np.minimum(cost_review, cost_hold), best_action


def oracle_actions(
    day: np.ndarray,
    y: np.ndarray,
    loss: np.ndarray,
    k: int,
    params: CostParams,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """The best actions a perfectly-informed analyst team could take under capacity K.

    ``weights`` are per-row analyst-hours; ``None`` means one review per merchant-day, in
    which case the knapsack degenerates to top-K by benefit and that faster path is taken.
    The two are asserted equivalent in the tests, so the shortcut is a proven identity
    rather than an approximation.
    """
    if k < 0:
        raise ValueError(f"capacity K must be >= 0; got {k!r}")
    benefit, best_action = intervention_benefit(y, loss, params)
    worth_it = benefit > 0.0

    if weights is None:
        # Uniform weights: top-K by benefit among the rows worth intervening on. Scores
        # below zero are pushed to -inf so they can never be selected to fill quota.
        selected = top_k_by_day(np.where(worth_it, benefit, -np.inf), day, k) & worth_it
    else:
        selected = np.zeros(y.shape, dtype=bool)
        for d in np.unique(day):
            rows = np.flatnonzero(day == d)
            selected[rows] = knapsack(benefit[rows], weights[rows], k)

    return np.where(selected, best_action, Action.PASS)


def oracle_savings(
    day: np.ndarray,
    y: np.ndarray,
    loss: np.ndarray,
    k: int,
    params: CostParams,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """The savings ceiling at capacity K. Not a target — nothing should reach it."""
    return savings_of_actions(
        oracle_actions(day, y, loss, k, params, weights=weights), y, loss, params
    )


def gap_to_oracle(rung_savings: float, ceiling: float) -> float:
    """``(oracle - rung) / oracle``. 0.0 means the rung captured everything achievable."""
    if math.isnan(ceiling) or ceiling == 0.0:
        return float("nan")
    return (ceiling - rung_savings) / ceiling


def assert_no_leakage(rung_savings: float, ceiling: float, *, label: str = "rung") -> None:
    """Run this on **every** eval. It has caught real bugs and it costs one comparison."""
    if math.isnan(rung_savings) or math.isnan(ceiling):
        return
    if rung_savings > ceiling + 1e-9:
        raise LeakageError(
            f"{label} savings {rung_savings:.6f} beats the perfect-foresight oracle "
            f"{ceiling:.6f}. The oracle is given the labels; nothing that is not can beat "
            "it. Suspect a forward-looking feature window, a label joined on "
            "label_event_at instead of label_available_at, or a merchant in two splits."
        )
