"""Frozen evaluation metrics (06-requirements.md §3).

Primary:   Bahnsen savings score at equal review budget.
Secondary: PR-AUC, precision@K (K = review budget), Brier score, median
           detection lag in days, gap-to-oracle.

**ROC-AUC and raw accuracy are not implemented here, deliberately.** The frozen
eval lists them as *prohibited as headline metrics* because at this generator's
20% positive rate (config.FRAUD_MERCHANT_RATE) ROC-AUC flatters every model and
accuracy is beaten by "predict healthy". Not implementing them is the cheapest
way to guarantee they never leak into the headline path. If a future ticket
genuinely needs one for a diagnostic, add it there and label it as such — do
not add it to this module.

Every function takes plain numpy/pandas arrays and returns plain floats, so the
harness stays the only place that knows about splits.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
)

from rakshak.config import (
    COST_REVIEW_INR,
    COST_SUPPORT_INR,
    P_ANALYST_MISS,
    P_CHURN_GIVEN_HOLD,
    RESIDUAL_LEAKAGE_RHO,
    WINDOW_DAYS,
)

PASS: Final[int] = 0
REVIEW: Final[int] = 1
HOLD: Final[int] = 2
ACTION_NAMES: Final[tuple[str, ...]] = ("PASS", "REVIEW", "HOLD")


# ---------------------------------------------------------------------------
# Ranking / calibration
# ---------------------------------------------------------------------------


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve).

    Args:
        y_true: Binary labels, shape (n,). 1 == merchant is truly bad.
        y_score: Higher means more suspicious, shape (n,). Need not be calibrated.

    Returns:
        Average precision in [0, 1]. Returns the prevalence (the random-ranking
        expectation) when every label is identical and the metric is undefined.
    """
    y_true = np.asarray(y_true, dtype=float)
    if y_true.min() == y_true.max():
        return float(y_true.mean())
    return float(average_precision_score(y_true, np.asarray(y_score, dtype=float)))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of the top-k ranked merchants that are truly bad.

    K is the review budget: the number of merchants an analyst pool can
    actually look at. Always report this alongside the prevalence — at 20%
    prevalence a precision of 0.30 is a 1.5x lift, not a good absolute number.

    Args:
        y_true: Binary labels, shape (n,).
        y_score: Suspicion scores, shape (n,). Ties are broken by ascending
            index, which is deterministic given a stable input ordering.
        k: Number of merchants reviewed. Clamped to [0, n].

    Returns:
        Precision in [0, 1]; 0.0 when k == 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    k = int(np.clip(k, 0, y_true.size))
    if k == 0:
        return 0.0
    order = np.lexsort((np.arange(y_true.size), -y_score))
    return float(y_true[order[:k]].mean())


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of calibrated probabilities. Lower is better.

    Args:
        y_true: Binary labels, shape (n,).
        y_prob: Calibrated P(bad) in [0, 1], shape (n,).

    Returns:
        Brier score in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# Detection lag (07-math.md §8)
# ---------------------------------------------------------------------------


FLAG_ATTRIBUTION_OFFSET_DAYS: Final[dict[str, int]] = {
    "window_start": 0,
    "window_end": WINDOW_DAYS - 1,
}
"""How a window-based scorer's `flag_day` is turned into a calendar day, in days.

A window-based scorer (`models/gbdt.py`, `models/hmm_score.py`) reports the
**start** day of the `WINDOW_DAYS`-long window whose evidence raised the flag.
The evidence was not complete until that window's **last** day, so a lag measured
against the window start credits the model with up to `WINDOW_DAYS - 1` days of
earliness it did not have. `"window_start"` is the historical convention and is
the default so that nothing already reported moves; `"window_end"` is the
attribution T-0011 measured and recommended (`results/lag_probe.md`).

Day-resolved scorers such as `models/rules.py` already report the last day of
their trailing evidence, so `"window_end"` does not apply to them.
"""


def detection_lag_days(
    flag_day: pd.Series,
    transition_day: pd.Series,
    labels: pd.Series,
    *,
    attribution: str = "window_start",
) -> tuple[float, float, int]:
    """Median detection lag over truly-bad merchants that were eventually flagged.

    Lag is `first_flag_day - true_transition_day`, in days, using the
    generator's state-path transition timestamps as truth. Negative lags (a flag
    raised before the merchant went bad) are kept, not clipped: a model that
    fires early on merchants that later go bad has earned that, and clipping
    would hide a systematically trigger-happy model.

    07-math.md §8 requires the flagged fraction be reported alongside — a median
    over a tiny flagged subset is meaningless without it.

    Args:
        flag_day: First day the model flagged each merchant, NaN if never.
            Indexed by merchant_id. Units: days.
        transition_day: True first-bad day per merchant, NaN if never bad.
            Indexed by merchant_id. Units: days.
        labels: Binary ground-truth labels indexed by merchant_id.
        attribution: Key of `FLAG_ATTRIBUTION_OFFSET_DAYS`. `"window_start"`
            (the default, and the historical convention) takes `flag_day` as
            given; `"window_end"` adds `WINDOW_DAYS - 1` days, attributing a
            window-based flag to the last day of the evidence that raised it.
            Units: days.

    Raises:
        ValueError: if `attribution` is not a known mode.

    Returns:
        `(median_lag_days, flagged_fraction, n_bad)`. `median_lag_days` is NaN
        when no truly-bad merchant was flagged. `flagged_fraction` is the share
        of truly-bad merchants that were flagged at all.
    """
    if attribution not in FLAG_ATTRIBUTION_OFFSET_DAYS:
        raise ValueError(
            f"unknown attribution {attribution!r}; "
            f"expected one of {sorted(FLAG_ATTRIBUTION_OFFSET_DAYS)}"
        )
    offset = FLAG_ATTRIBUTION_OFFSET_DAYS[attribution]
    bad = labels.index[labels.astype(bool)]
    n_bad = len(bad)
    if n_bad == 0:
        return float("nan"), 0.0, 0
    flags = flag_day.reindex(bad)
    truth = transition_day.reindex(bad)
    detected = flags.notna() & truth.notna()
    flagged_fraction = float(detected.mean())
    if not detected.any():
        return float("nan"), flagged_fraction, n_bad
    lag = (flags[detected] + offset - truth[detected]).astype("float64")
    return float(lag.median()), flagged_fraction, n_bad


# ---------------------------------------------------------------------------
# Cost layer (07-math.md §5-6) — Bahnsen et al. 2016
# ---------------------------------------------------------------------------


def false_positive_cost(value_inr: np.ndarray) -> np.ndarray:
    """c_fp(m) = P(churn | hold) * V_m + c_support. Units: INR.

    Args:
        value_inr: Per-merchant value V_m, shape (n,). Units: INR.

    Returns:
        Per-merchant cost of wrongly holding, shape (n,). Units: INR.
    """
    return P_CHURN_GIVEN_HOLD * np.asarray(value_inr, dtype=float) + COST_SUPPORT_INR


def action_cost(
    y_true: np.ndarray, actions: np.ndarray, loss_inr: np.ndarray, value_inr: np.ndarray
) -> np.ndarray:
    """Per-merchant realised cost of the chosen actions (07-math.md §5 matrix).

    |          | true: healthy | true: bad            |
    |----------|---------------|----------------------|
    | PASS     | 0             | L_m                  |
    | REVIEW   | c_rev         | c_rev + p_miss * L_m |
    | HOLD     | c_fp(m)       | rho * L_m            |

    Args:
        y_true: Binary labels, shape (n,).
        actions: One of PASS/REVIEW/HOLD per merchant, shape (n,).
        loss_inr: L_m, fraud loss if the merchant is left alone. Units: INR.
        value_inr: V_m, merchant value used for the churn cost. Units: INR.

    Returns:
        Per-merchant cost, shape (n,). Units: INR.
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(actions, dtype=int)
    loss = np.asarray(loss_inr, dtype=float)
    c_fp = false_positive_cost(value_inr)

    cost_healthy = np.select(
        [a == PASS, a == REVIEW, a == HOLD],
        [np.zeros_like(loss), np.full_like(loss, COST_REVIEW_INR), c_fp],
    )
    cost_bad = np.select(
        [a == PASS, a == REVIEW, a == HOLD],
        [loss, COST_REVIEW_INR + P_ANALYST_MISS * loss, RESIDUAL_LEAKAGE_RHO * loss],
    )
    return y * cost_bad + (1.0 - y) * cost_healthy


def total_cost(
    y_true: np.ndarray, actions: np.ndarray, loss_inr: np.ndarray, value_inr: np.ndarray
) -> float:
    """Sum of `action_cost`. Units: INR."""
    return float(action_cost(y_true, actions, loss_inr, value_inr).sum())


def baseline_cost(y_true: np.ndarray, loss_inr: np.ndarray, value_inr: np.ndarray) -> float:
    """Cost_l = min(Cost(all PASS), Cost(all HOLD)). Units: INR.

    Bahnsen's denominator: the cheaper of the two trivial constant policies.
    Reported explicitly because the savings score is only interpretable against
    it, and because which of the two wins says a lot about the cost matrix.
    """
    n = np.asarray(y_true).size
    all_pass = total_cost(y_true, np.full(n, PASS), loss_inr, value_inr)
    all_hold = total_cost(y_true, np.full(n, HOLD), loss_inr, value_inr)
    return min(all_pass, all_hold)


def savings_score(
    y_true: np.ndarray, actions: np.ndarray, loss_inr: np.ndarray, value_inr: np.ndarray
) -> float:
    """Bahnsen savings: (Cost_l - Cost(f)) / Cost_l. Dimensionless.

    1.0 is a costless policy, 0.0 matches the better trivial policy, negative
    means the policy is worse than doing nothing.

    Guard (AP-06, 07-math.md §6): this score is manipulable through the cost
    matrix. Never report it without PR-AUC beside it.

    Returns:
        Savings score. Returns 0.0 when Cost_l is 0 (nothing to save).
    """
    denominator = baseline_cost(y_true, loss_inr, value_inr)
    if denominator <= 0.0:
        return 0.0
    return float((denominator - total_cost(y_true, actions, loss_inr, value_inr)) / denominator)


def gap_to_oracle(value: float, oracle_value: float) -> float:
    """Fraction of the oracle ceiling left on the table. Dimensionless.

    Args:
        value: The model's savings score.
        oracle_value: The perfect-foresight ceiling's savings score.

    Returns:
        `(oracle - value) / oracle`. 0.0 means the model matched the ceiling;
        negative means it exceeded it (possible when the oracle is
        capacity-constrained and the model is not — see oracle.py). NaN when
        the ceiling is 0.
    """
    if oracle_value == 0.0:
        return float("nan")
    return float((oracle_value - value) / oracle_value)


# ---------------------------------------------------------------------------
# Latent-state recovery (FR-013, as amended 2026-08-28 — see 06-requirements.md)
# ---------------------------------------------------------------------------
#
# FR-013 originally scored state recovery with ARI alone. Romano, Vinh, Bailey &
# Verspoor, "Adjusting for Chance Clustering Comparison Measures", JMLR 17 (2016),
# https://arxiv.org/abs/1512.01286, state the guideline verbatim:
#
#     "ARI should be used when the reference clustering has large equal sized
#      clusters; AMI should be used when the reference clustering is unbalanced
#      and there exist small clusters."
#
# Rakshak's reference partition is roughly HEALTHY 91 / FRAUD 5 / RAMP 2 / DORMANT 1.5
# per cent of windows, i.e. exactly the unbalanced-with-small-clusters case. ARI is
# pair-counting based, so ~83% of the pairs it counts are HEALTHY-HEALTHY and the three
# states the product exists to find barely move it.
#
# ARI IS RETAINED AND REPORTED PERMANENTLY, beside AMI and beside the
# oracle-parameterised ceiling for both. The ceiling is what makes the amendment
# credible rather than convenient: it was measured in T-0004, before the metric was
# touched. Removing, burying or de-emphasising either defeats the purpose.


def adjusted_mutual_information(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Mutual Information between a reference and a recovered partition.

    The chance-corrected index the literature names for an unbalanced reference
    containing small clusters (Romano et al., JMLR 17, 2016). Permutation-invariant
    in both arguments, like ARI, so no state alignment is needed.

    Args:
        y_true: Reference partition labels, shape (n,). Any hashable coding.
        y_pred: Recovered partition labels, shape (n,).

    Returns:
        AMI, dimensionless. 0.0 is chance, 1.0 is a perfect partition; it can be
        slightly negative.
    """
    return float(adjusted_mutual_info_score(np.asarray(y_true), np.asarray(y_pred)))


def align_states(y_true: np.ndarray, y_pred: np.ndarray, n_states: int) -> np.ndarray:
    """Best one-to-one map from recovered state index to reference state index.

    A hidden state has no name: Baum-Welch's state 2 may be the reference's RAMP.
    ARI and AMI do not care, but per-state recall does, so the contingency table is
    solved as a linear assignment problem (Hungarian algorithm) maximising total
    agreement. One-to-one on purpose — a many-to-one map would let one recovered
    state claim credit for two reference states.

    Args:
        y_true: Reference state indices in [0, n_states), shape (n,).
        y_pred: Recovered state indices in [0, n_states), shape (n,).
        n_states: K.

    Returns:
        Integer array `mapping` of shape (n_states,): `mapping[j]` is the reference
        state that recovered state `j` is credited with.
    """
    table = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(table, (np.asarray(y_pred, dtype=int), np.asarray(y_true, dtype=int)), 1)
    pred_idx, true_idx = linear_sum_assignment(-table)
    mapping = np.arange(n_states)
    mapping[pred_idx] = true_idx
    return mapping


def per_state_recall(
    y_true: np.ndarray, y_pred: np.ndarray, n_states: int, mapping: np.ndarray | None = None
) -> np.ndarray:
    """Recall of each reference state after best one-to-one state alignment.

    Reported per state, never only as an average: the 90%-mass HEALTHY class would
    otherwise set the headline for a model that finds none of the rare states. The
    macro average of this vector is balanced accuracy.

    Args:
        y_true: Reference state indices, shape (n,).
        y_pred: Recovered state indices, shape (n,).
        n_states: K.
        mapping: Alignment from `align_states`; computed here when None.

    Returns:
        Array of shape (n_states,), entry k being recall of reference state k, or
        NaN when the reference never visits k.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if mapping is None:
        mapping = align_states(y_true, y_pred, n_states)
    aligned = np.asarray(mapping)[y_pred]
    out = np.full(n_states, np.nan)
    for k in range(n_states):
        members = y_true == k
        if members.any():
            out[k] = float((aligned[members] == k).mean())
    return out


def detection_lag_windows(
    true_codes: np.ndarray, alerted: np.ndarray, healthy_code: int = 0
) -> tuple[float, float, int]:
    """Median windows from a merchant's first non-healthy window to its first alert.

    The window-grid twin of `detection_lag_days`, for the state-recovery report. The
    product claim is earliness, so a recovery metric that ignores time cannot measure
    the thing being claimed. Negative lags (alert raised before the merchant actually
    turns) are kept, not clipped, for the reason given in `detection_lag_days`.

    Args:
        true_codes: Reference state indices, shape (M, W), merchants by windows.
        alerted: Boolean, shape (M, W): True where the model raises an alert.
        healthy_code: Index of the healthy reference state.

    Returns:
        `(median_lag_windows, flagged_fraction, n_bad)`. `median_lag_windows` is NaN
        when no truly-bad merchant is ever alerted; `flagged_fraction` is the share of
        truly-bad merchants alerted at or after their onset. Units: windows.
    """
    true_codes = np.asarray(true_codes)
    alerted = np.asarray(alerted, dtype=bool)
    bad_window = true_codes != healthy_code
    is_bad = bad_window.any(axis=1)
    n_bad = int(is_bad.sum())
    if n_bad == 0:
        return float("nan"), 0.0, 0

    onset = np.argmax(bad_window, axis=1)
    lags: list[int] = []
    n_flagged = 0
    for m in np.flatnonzero(is_bad):
        hits = np.flatnonzero(alerted[m])
        if hits.size == 0:
            continue
        n_flagged += 1
        lags.append(int(hits[0] - onset[m]))
    if not lags:
        return float("nan"), 0.0, n_bad
    return float(np.median(lags)), float(n_flagged / n_bad), n_bad


def state_recovery_report(
    true_codes: np.ndarray,
    decoded: np.ndarray,
    n_states: int,
    healthy_code: int = 0,
    non_healthy_score: np.ndarray | None = None,
) -> dict[str, object]:
    """The amended FR-013 metric suite, computed in one place.

    Every caller reports every key. ARI and the oracle ceiling are not optional
    extras here — see the module section header above.

    Args:
        true_codes: Reference state indices, shape (M, W).
        decoded: Recovered state indices, shape (M, W).
        n_states: K.
        healthy_code: Index of the healthy reference state.
        non_healthy_score: Optional per-window score in [0, 1] for the binary
            "this window is not healthy" task, shape (M, W) — normally
            `1 - P(healthy)` from the HMM posterior. When None the hard decoded
            partition is used as a 0/1 score, which makes PR-AUC a threshold-free
            metric computed on a thresholded input; that is weaker, and is why the
            posterior should be passed whenever it exists.

    Returns:
        Dict with keys `ari`, `ami`, `recall` (per reference state, aligned),
        `macro_recall`, `binary_pr_auc`, `binary_base_rate`, `detection_lag_windows`,
        `flagged_fraction` and `mapping`.
    """
    true_flat = np.asarray(true_codes).ravel()
    dec_flat = np.asarray(decoded).ravel()
    mapping = align_states(true_flat, dec_flat, n_states)
    recall = per_state_recall(true_flat, dec_flat, n_states, mapping=mapping)

    binary_true = (true_flat != healthy_code).astype(float)
    aligned = mapping[dec_flat]
    score = (
        np.asarray(non_healthy_score).ravel()
        if non_healthy_score is not None
        else (aligned != healthy_code).astype(float)
    )
    if np.ndim(true_codes) == 2:
        lag, flagged, _ = detection_lag_windows(
            np.asarray(true_codes),
            mapping[np.asarray(decoded)] != healthy_code,
            healthy_code=healthy_code,
        )
    else:
        # Detection lag needs the (merchant, window) grid. Flattened input is a
        # legitimate call for the partition metrics, so this is NaN, not an error.
        lag, flagged = float("nan"), float("nan")
    return {
        "ari": float(adjusted_rand_score(true_flat, dec_flat)),
        "ami": adjusted_mutual_information(true_flat, dec_flat),
        "recall": recall,
        "macro_recall": float(np.nanmean(recall)),
        "binary_pr_auc": pr_auc(binary_true, score),
        "binary_base_rate": float(binary_true.mean()),
        "detection_lag_windows": lag,
        "flagged_fraction": flagged,
        "mapping": mapping,
    }
