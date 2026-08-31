"""The capacity-constrained decision layer (10-eval-harness-spec.md §4; FR-034).

Analyst capacity K is the binding operational constraint, and metrics that ignore it are
decoration. K = **50 reviews/day per 10,000 merchants** (0.5%), confirmed in charter §10.4
and load-bearing: a wrong K changes the *ranking* of rungs, not just their scores, so every
capacity-constrained number names the K it was computed at.

Per epoch, given calibrated scores and capacity K:

1. compute the expected cost of each action per merchant;
2. rank by ``cost(PASS) - min(cost(REVIEW), cost(HOLD))`` — the benefit of intervening;
3. take the top K, each with its own cost-minimising action;
4. everything else PASSes.

``alerts_per_day <= K`` holds by construction here rather than by assertion downstream —
the selection is a top-K, so there is no code path that can emit a K+1th alert.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from rakshak.eval.metrics import CostParams, savings_of_actions, top_k_by_day
from rakshak.schemas import Action

__all__ = [
    "ASYMMETRY_RATIOS",
    "DEFAULT_POLICY",
    "ActionPolicy",
    "SweepRow",
    "expected_costs",
    "select_actions",
    "sweep_cost_asymmetry",
]

#: The five ``false_hold_cost / fraud_loss`` ratios the sweep runs at
#: (10-eval-harness-spec.md §2). v1 measured the asymmetry at 47.5 / 13.1 / 61,368 against
#: a literature band of 400-600 — three orders of magnitude, which means the ratio cannot
#: be assumed, only measured per deployment. A ranking stable across this sweep is a far
#: stronger claim than a win at one guessed ratio; a ranking that flips is itself the
#: finding, and a more interesting one.
ASYMMETRY_RATIOS: Final = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True, slots=True)
class ActionPolicy:
    """When a HOLD is permitted at all.

    ``HOLD`` freezes a merchant's settlements. You do not do that over 4,000 rupees of
    expected exposure however confident the model is, and you do not do it on a 0.3 score
    however large the exposure. Both conditions must hold.

    **Config note:** 10-eval-harness-spec.md §4 says both thresholds live in config, and
    ``configs/scenario_v2.yaml`` does not exist yet (Lane A owns it). These defaults are
    named here so the numbers are visible rather than buried, and they belong under a
    ``decision:`` block in that file when it lands.
    """

    hold_score_threshold: float = 0.90
    hold_expected_loss_floor_inr: float = 25_000.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.hold_score_threshold <= 1.0:
            raise ValueError(
                f"hold_score_threshold is a probability; got {self.hold_score_threshold!r}"
            )
        if self.hold_expected_loss_floor_inr < 0:
            raise ValueError("hold_expected_loss_floor_inr must be >= 0")


#: The default HOLD policy, as a singleton so it can be a function default (ruff B008).
DEFAULT_POLICY: Final = ActionPolicy()


def expected_costs(
    score: np.ndarray, exposure_inr: np.ndarray, params: CostParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expected cost of PASS / REVIEW / HOLD per merchant, under ``p = score``.

    This is where the calibration requirement bites: ``score`` is used as a probability,
    not as a rank. An uncalibrated score makes every number below arithmetic on a
    meaningless quantity, which is why ECE is a reported metric and not a footnote.
    """
    p = np.clip(score, 0.0, 1.0)
    expected_loss = p * exposure_inr * params.fraud_loss_multiplier
    cost_pass = expected_loss
    cost_review = params.review_cost_inr + (1.0 - params.p_catch) * expected_loss
    cost_hold = params.review_cost_inr + (1.0 - p) * params.false_hold_cost_inr
    return cost_pass, cost_review, cost_hold


def select_actions(
    score: np.ndarray,
    day: np.ndarray,
    exposure_inr: np.ndarray,
    k: int,
    params: CostParams,
    policy: ActionPolicy = DEFAULT_POLICY,
) -> np.ndarray:
    """PASS/REVIEW/HOLD per merchant-day, at most K non-PASS actions on any day."""
    if k < 0:
        raise ValueError(f"capacity K must be >= 0; got {k!r}")
    cost_pass, cost_review, cost_hold = expected_costs(score, exposure_inr, params)
    benefit = cost_pass - np.minimum(cost_review, cost_hold)
    worth_it = benefit > 0.0

    # -inf on the not-worth-it rows so a slack day cannot be filled with alerts that cost
    # more than they save. Capacity is a ceiling, never a quota.
    selected = top_k_by_day(np.where(worth_it, benefit, -np.inf), day, k) & worth_it

    wants_hold = cost_hold < cost_review
    permitted = (score >= policy.hold_score_threshold) & (
        score * exposure_inr >= policy.hold_expected_loss_floor_inr
    )
    intervention = np.where(wants_hold & permitted, Action.HOLD, Action.REVIEW)
    chosen: np.ndarray = np.where(selected, intervention, Action.PASS)
    return chosen


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One cell of the cost-asymmetry sweep: a rung's savings at one ratio."""

    ratio: float
    rung: str
    savings: float
    rank: int


def sweep_cost_asymmetry(
    scores: Mapping[str, np.ndarray],
    day: np.ndarray,
    y: np.ndarray,
    loss: np.ndarray,
    exposure_inr: np.ndarray,
    k: int,
    params: CostParams,
    *,
    ratios: tuple[float, ...] = ASYMMETRY_RATIOS,
    policy: ActionPolicy = DEFAULT_POLICY,
) -> list[SweepRow]:
    """Rank every rung at each ``false_hold_cost / fraud_loss`` ratio.

    ``exposure_inr`` is the decision layer's *estimate* of what is at risk and must be a
    feature-derived quantity, never ``true_loss_amount_inr``. Handing the selector the true
    loss gives every rung perfect foresight about magnitude, which inflates all of them and
    can let one beat the oracle.

    The reference fraud loss is the mean ``true_loss_amount_inr`` over the fraud rows in
    the window, so a ratio of 1.0 means "holding a good merchant wrongly costs about what
    one fraud costs". ``review_cost_inr`` and ``p_catch`` are held fixed: the sweep varies
    the asymmetry, not everything at once, or the result is uninterpretable.
    """
    fraud_loss = loss[y == 1]
    reference = float(fraud_loss.mean()) if fraud_loss.size else 0.0
    if reference <= 0.0:
        raise ValueError(
            "the sweep needs a positive reference fraud loss; there are no fraud rows in "
            "this window, so the ratio has no denominator"
        )

    rows: list[SweepRow] = []
    for ratio in ratios:
        swept = CostParams(
            review_cost_inr=params.review_cost_inr,
            false_hold_cost_inr=ratio * reference,
            fraud_loss_multiplier=params.fraud_loss_multiplier,
            p_catch=params.p_catch,
        )
        scored = {
            name: savings_of_actions(
                select_actions(score, day, exposure_inr, k, swept, policy), y, loss, swept
            )
            for name, score in scores.items()
        }
        order = sorted(scored, key=lambda n: (-scored[n], n))
        rows.extend(
            SweepRow(ratio=ratio, rung=name, savings=scored[name], rank=i + 1)
            for i, name in enumerate(order)
        )
    return rows
