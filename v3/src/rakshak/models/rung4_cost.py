"""Rung 4 - the instance-dependent cost, inside the training objective (FR-032).

Rungs 2 and 3 minimise logloss and then hand a probability to a cost-aware decision layer,
which is Bayes-minimum-risk applied *after the fact*. The literature's claim is that
putting the cost inside the fit does better, because the fit then spends its capacity on
the rows where being wrong is expensive rather than on the rows that are merely numerous.
This rung tests that claim on this data.

**It is instance weighting, not a hand-written ``fobj``, and that is a deliberate choice.**
Weighting each row by its cost makes the training objective ``sum_i w_i * logloss_i``,
which is exactly "the cost is in the objective". A custom gradient/Hessian pair would put
the model's output on a raw scale that is no longer a probability, and the decision layer
downstream consumes the score *as* a probability - so the custom objective would win the
letter of the ticket and lose the thing the ticket is for.

**The weight is an exposure estimate, never a realised loss.** A row's cost is bounded by
what the merchant is putting through, and that is an onboarding fact the feature register
already carries. Reaching for the true loss amount would be Prime Directive 3 leakage and
would also be unavailable in production, where nobody knows the loss until it lands.

P2 on the board, first in the cut order. If it loses, it is logged and dropped.
"""

from __future__ import annotations

import numpy as np

from rakshak.models.rung2_lgbm import DEFAULT_PARAMS, HParams, TrainedRung
from rakshak.models.rung2_lgbm import train as _train_lgbm

__all__ = ["DAYS_PER_MONTH", "cost_weights", "train"]

#: The exposure feature is a declared *monthly* volume; the decision is daily.
DAYS_PER_MONTH = 30.0


def cost_weights(
    exposure_inr: np.ndarray, y: np.ndarray, review_cost_inr: float, cap: float = 50.0
) -> np.ndarray:
    """Per-row training weight: what being wrong on this row would cost, in review units.

    A missed fraud costs its exposure; a false alarm costs one review. Expressed as a
    ratio so the weights are scale-free and a change in ``review_cost_inr`` moves them
    coherently rather than rescaling the whole objective.

    ``cap`` exists because the exposure distribution is lognormal over three orders of
    magnitude: uncapped, the top few merchants would carry more weight than the rest of the
    positive class combined, and the rung would be a model of six merchants. 50 keeps the
    heaviest row worth fifty of the lightest, which is a real asymmetry and not a
    degenerate one.
    """
    if review_cost_inr <= 0.0:
        raise ValueError(f"review_cost_inr must be > 0; got {review_cost_inr!r}")
    ratio = np.clip(np.asarray(exposure_inr, dtype=np.float64) / review_cost_inr, 1.0, cap)
    return np.where(y == 1, ratio, 1.0)


def train(
    x: np.ndarray,
    y: np.ndarray,
    columns: tuple[str, ...],
    *,
    exposure_inr: np.ndarray,
    review_cost_inr: float,
    params: HParams = DEFAULT_PARAMS,
    merchant_id: np.ndarray | None = None,
) -> TrainedRung:
    """Rung 3's columns and hyperparameters, with the cost matrix inside the fit."""
    return _train_lgbm(
        x,
        y,
        columns,
        rung=4,
        params=params,
        merchant_id=merchant_id,
        weight=cost_weights(exposure_inr, y, review_cost_inr),
    )
