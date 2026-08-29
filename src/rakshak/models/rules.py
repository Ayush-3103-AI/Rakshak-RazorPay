"""The floor: a static rule engine with fixed global thresholds and no learning.

06-requirements.md §3 freezes this baseline before any model exists::

    flag if 7-day transaction velocity exceeds 3x the merchant's trailing 90-day
    mean, OR refund ratio exceeds 15%, OR chargeback ratio exceeds 1%.

Three properties of this module are deliberate and must survive future edits:

1. **The thresholds are constants, not parameters.** Nothing here is fitted, and
   nothing here reads the labels. `score_rules` never touches `split.labels`.
2. **It is not softened.** 06-requirements.md §3: *"If the sophisticated approach
   cannot beat this, that is a bug or a finding — not a result to hide."*
   Weakening the floor to flatter the HMM is the exact dishonesty CLAUDE.md
   forbids, so the thresholds live as module constants that a reader can check
   against the spec in one glance.
3. **It emits a `flag_day`**, so median detection lag is computable for it. A
   rule engine fires on a particular day; a model that only ever returns a
   scalar cannot be compared with it on latency.

Two things the frozen spec does not pin down, chosen here and stated rather than
buried:

* **The ratio windows.** The spec gives a window for velocity (7 vs 90 days) and
  none for the refund and chargeback ratios. Both are measured over a trailing
  30 days: a 7-day chargeback ratio at a 1% threshold is one chargeback in a
  hundred transactions, which for a small merchant is pure Poisson noise, and a
  month is the ordinary risk-ops reporting period.
* **The score.** The rule engine's native output is binary, and a binary score
  ranks badly on PR-AUC and precision@K for reasons that have nothing to do with
  the rules being weak — everything ties. Each rule therefore also reports a
  *severity*, `observed / threshold`, clipped to [0, 2] and rescaled to [0, 1];
  the merchant's score is the mean of the three. Severity uses the same fixed
  thresholds — nothing is fitted — and severity >= 0.5 is exactly the binary rule
  firing, so the flag and the score never disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rakshak.eval.splits import Split

VELOCITY_MULTIPLE: float = 3.0
"""Rule 1: 7-day transaction count over 3x the trailing-90-day expectation."""

VELOCITY_WINDOW_DAYS: int = 7
"""Length of the velocity window, in days."""

VELOCITY_BASELINE_DAYS: int = 90
"""Trailing days used for the baseline rate, ending where the velocity window starts."""

REFUND_RATIO_THRESHOLD: float = 0.15
"""Rule 2: refund transactions over 15% of transactions."""

CHARGEBACK_RATIO_THRESHOLD: float = 0.01
"""Rule 3: chargeback transactions over 1% of transactions."""

RATIO_WINDOW_DAYS: int = 30
"""Trailing days over which the two ratios are measured. See the module docstring."""

RULE_NAMES: tuple[str, str, str] = ("velocity", "refund_ratio", "chargeback_ratio")
"""Rule order, matching the severity columns `rule_severities` returns."""


def _daily_counts(transactions: pd.DataFrame, merchant_ids: tuple[str, ...], n_days: int):
    """Dense per-merchant daily counts of transactions, refunds and chargebacks.

    Args:
        transactions: Split transactions carrying `merchant_id`, `day`, `is_refund`,
            `is_chargeback`.
        merchant_ids: Row order to emit.
        n_days: Number of day columns, i.e. days `[0, n_days)`.

    Returns:
        Three float arrays of shape (M, n_days): transactions, refunds, chargebacks.
        Units: transaction counts per day.
    """
    row_of = {mid: i for i, mid in enumerate(merchant_ids)}
    rows = transactions["merchant_id"].map(row_of).to_numpy()
    days = transactions["day"].to_numpy(dtype=np.int64)
    keep = np.isfinite(rows.astype(float)) & (days >= 0) & (days < n_days)
    rows, days = rows[keep].astype(np.int64), days[keep]

    shape = (len(merchant_ids), n_days)
    flat = rows * n_days + days
    size = shape[0] * shape[1]
    n_txn = np.bincount(flat, minlength=size).astype(float).reshape(shape)
    n_refund = (
        np.bincount(flat, weights=transactions["is_refund"].to_numpy()[keep], minlength=size)
        .astype(float)
        .reshape(shape)
    )
    n_chargeback = (
        np.bincount(flat, weights=transactions["is_chargeback"].to_numpy()[keep], minlength=size)
        .astype(float)
        .reshape(shape)
    )
    return n_txn, n_refund, n_chargeback


def _trailing(cumulative: np.ndarray, end: np.ndarray, length: int) -> np.ndarray:
    """Counts over the `length` days ending the day before `end` (exclusive).

    Args:
        cumulative: Shape (M, n_days + 1); `cumulative[:, d]` is the count over days
            `[0, d)`.
        end: Exclusive window end per column, shape (T,).
        length: Window length in days.

    Returns:
        Array of shape (M, T) of counts.
    """
    hi = np.clip(end, 0, cumulative.shape[1] - 1)
    lo = np.clip(end - length, 0, cumulative.shape[1] - 1)
    return cumulative[:, hi] - cumulative[:, lo]


def rule_severities(split: Split) -> tuple[np.ndarray, np.ndarray]:
    """Per-day severity of each rule over the split's decision window.

    Severity is `observed / threshold`: 1.0 is exactly at the fixed threshold,
    so `severity >= 1` is the rule firing. Nothing here is fitted and the labels
    are never read.

    Args:
        split: The split to score. Its `transactions` carry each merchant's own
            history from day 0, which is what the trailing baselines need.

    Returns:
        `(severity, days)`. `severity` has shape (M, T, 3) in `RULE_NAMES` order,
        dimensionless. `days` has shape (T,), the days in
        `[split.start_day, split.end_day)` the severities are evaluated on.
    """
    n_days = split.end_day
    n_txn, n_refund, n_chargeback = _daily_counts(split.transactions, split.merchant_ids, n_days)
    cum_txn = np.concatenate([np.zeros((n_txn.shape[0], 1)), n_txn.cumsum(axis=1)], axis=1)
    cum_refund = np.concatenate(
        [np.zeros((n_refund.shape[0], 1)), n_refund.cumsum(axis=1)], axis=1
    )
    cum_chargeback = np.concatenate(
        [np.zeros((n_chargeback.shape[0], 1)), n_chargeback.cumsum(axis=1)], axis=1
    )

    days = np.arange(split.start_day, split.end_day)
    end = days + 1  # windows are inclusive of the decision day

    velocity = _trailing(cum_txn, end, VELOCITY_WINDOW_DAYS)
    baseline = _trailing(cum_txn, end - VELOCITY_WINDOW_DAYS, VELOCITY_BASELINE_DAYS)
    expected = baseline * VELOCITY_WINDOW_DAYS / VELOCITY_BASELINE_DAYS
    # No trailing history means no baseline to be 3x of, so the rule cannot fire.
    velocity_severity = np.where(
        expected > 0.0, velocity / np.maximum(expected * VELOCITY_MULTIPLE, 1e-12), 0.0
    )

    txn_ratio_window = _trailing(cum_txn, end, RATIO_WINDOW_DAYS)
    denominator = np.maximum(txn_ratio_window, 1e-12)
    refund_severity = np.where(
        txn_ratio_window > 0.0,
        _trailing(cum_refund, end, RATIO_WINDOW_DAYS) / denominator / REFUND_RATIO_THRESHOLD,
        0.0,
    )
    chargeback_severity = np.where(
        txn_ratio_window > 0.0,
        _trailing(cum_chargeback, end, RATIO_WINDOW_DAYS)
        / denominator
        / CHARGEBACK_RATIO_THRESHOLD,
        0.0,
    )
    return np.stack([velocity_severity, refund_severity, chargeback_severity], axis=-1), days


def score_rules(split: Split, rng: np.random.Generator) -> pd.DataFrame:
    """Score every merchant with the static rule engine (06-requirements.md §3).

    Args:
        split: The split to score.
        rng: Unused — the rule engine is deterministic. Present for the harness
            scorer signature.

    Returns:
        Frame indexed by merchant_id with `score` in [0, 1] (dimensionless mean
        rule severity, 0.5 == exactly at threshold) and `flag_day`, the first day
        in the decision window on which any rule fired (NaN if none ever did).
    """
    del rng
    severity, days = rule_severities(split)
    fired = (severity >= 1.0).any(axis=-1)  # (M, T)
    first = np.argmax(fired, axis=1)
    flag_day = np.where(fired.any(axis=1), days[first], np.nan).astype(float)
    score = np.clip(severity.max(axis=1), 0.0, 2.0).mean(axis=-1) / 2.0
    return pd.DataFrame(
        {"score": score, "flag_day": flag_day},
        index=pd.Index(split.merchant_ids, name="merchant_id"),
    )
