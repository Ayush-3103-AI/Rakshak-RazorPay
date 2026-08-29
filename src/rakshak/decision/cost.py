"""Corrected cost definitions and the oracle-dominance invariant (T-0007a).

This module implements the two boxed redefinitions in `07-math.md` §5 as amended
2026-08-28 by T-0017, and nothing else. The Bayes-Minimum-Risk policy and the
cost-asymmetry sweep are T-0007b.

**What changed and why (definitions, not calibration).**

1. ``V_m = g * v_m * l_m`` — expected *lifetime* gross margin. The previous form,
   ``MDR_RATE * window_volume``, was wrong twice over and the two errors nearly
   cancelled:

   * it charged one window's margin for a churn that costs every remaining month
     of margin (understated by roughly ``l_m`` months);
   * ``MDR_RATE = 0.02`` is the *price the merchant pays*, not the *margin the
     platform keeps*. Almost all of it leaves again as issuer interchange, scheme
     fees and GST. Razorpay's own gross margin is ~10 bps of TPV (overstated by
     roughly 20x).

   Applying only the lifetime fix leaves ``V_m`` overstated ~20x. Both or neither.

2. ``L_m = r_cb * (1 + phi) * G_bad_m`` — realised loss, not turnover. Counting a
   bust-out merchant's whole gross turnover as acquirer loss inflated ``L_m`` by
   more than an order of magnitude and is the defect behind T-0006's negative
   savings on *both* perfect-foresight oracles.

Every primitive lives in `config.py` with a source class ([S]/[D]/[A]), a citation
and a range. This module holds only the arithmetic that combines them.

**No constant in this module or in `config.py` was chosen by the ratio it
produces.** `07-math.md` §5's 400-600 asymmetry is a reported cross-check, not a
gate: `fp_cost_per_100_of_fraud_loss` computes it and callers report whatever it
says, including a divergence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from rakshak.config import (
    ANCILLARY_LOADING_PHI,
    CHARGEBACK_REALISATION_RATE,
    DAYS_PER_MONTH,
    GROSS_MARGIN_RATE,
    MERCHANT_LIFETIME_MONTHS,
    SEED,
)

__all__ = [
    "assert_oracle_dominance",
    "expected_monthly_volume_inr",
    "fp_cost_per_100_of_fraud_loss",
    "merchant_value_inr",
    "realised_loss_inr",
]


# ---------------------------------------------------------------------------
# Definitional fix 1 — V_m is expected lifetime gross margin
# ---------------------------------------------------------------------------


def expected_monthly_volume_inr(
    volume_inr: pd.Series | np.ndarray, observed_days: float
) -> pd.Series | np.ndarray:
    """v_m — merchant's expected monthly gross processed volume. Units: INR/month.

    Estimated as observed non-refund volume divided by the observation span in
    months. Uses no label and no future information, so it is safe on any split.

    It also removes a latent bug: the previous `value_inr` summed the merchant's
    whole loaded history, so V_m silently grew with how many days the split had
    loaded (validate 210 days, test 270 days). A per-month rate does not.

    Args:
        volume_inr: Non-refund gross volume observed per merchant. Units: INR.
        observed_days: Length of the observation span. Units: days.

    Returns:
        Expected monthly volume per merchant. Units: INR / month.
    """
    months = float(observed_days) / DAYS_PER_MONTH
    if months <= 0.0:
        raise ValueError(f"observed_days must be positive, got {observed_days!r}")
    return volume_inr / months


def merchant_value_inr(
    monthly_volume_inr: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """V_m = g * v_m * l_m — expected remaining lifetime gross margin. Units: INR.

    07-math.md §5, definitional fix 1. This is a *stock*, not a rate: it is every
    rupee of margin the platform forgoes if this merchant is held and churns.

    Args:
        monthly_volume_inr: v_m. Units: INR / month.

    Returns:
        Per-merchant lifetime gross margin. Units: INR.
    """
    return GROSS_MARGIN_RATE * monthly_volume_inr * MERCHANT_LIFETIME_MONTHS


# ---------------------------------------------------------------------------
# Definitional fix 2 — L_m is realised loss, not turnover
# ---------------------------------------------------------------------------


def realised_loss_inr(
    gross_bad_volume_inr: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """L_m = r_cb * (1 + phi) * G_bad_m — realised fraud loss. Units: INR.

    07-math.md §5, definitional fix 2. `r_cb` is the fraction of bad-state
    turnover that comes back as chargeback, confirmed-fraud write-off or
    unrecovered negative balance; `phi` loads scheme dispute fees, representment
    handling and monitoring penalties on top. Analyst labour and support are NOT
    in `phi` — the cost matrix charges those separately via c_rev and c_support,
    and double-counting them would be an error.

    Args:
        gross_bad_volume_inr: G_bad_m, gross volume transacted while in a bad
            state. Units: INR.

    Returns:
        Per-merchant realised loss. Units: INR.
    """
    return CHARGEBACK_REALISATION_RATE * (1.0 + ANCILLARY_LOADING_PHI) * gross_bad_volume_inr


# ---------------------------------------------------------------------------
# The 400-600 cross-check — computed and reported, never closed
# ---------------------------------------------------------------------------


def fp_cost_per_100_of_fraud_loss(
    y_true: np.ndarray, loss_inr: np.ndarray, value_inr: np.ndarray
) -> tuple[float, float, float]:
    """INR of false-positive cost per INR 100 of fraud loss, on this population.

    Numerator: c_fp summed over every truly-healthy merchant, i.e. the cost of
    holding all of them. Denominator: L_m summed over every truly-bad merchant,
    i.e. the cost of passing all of them.

    07-math.md §5 demotes the 400-600 commentary band from a gate to a **reported
    cross-check**. Report whatever this returns and state any divergence. Never
    move a primitive to reach a number.

    Args:
        y_true: Binary labels, shape (n,). 1 == truly bad.
        loss_inr: L_m per merchant. Units: INR.
        value_inr: V_m per merchant. Units: INR.

    Returns:
        (ratio, total_fp_cost_inr, total_fraud_loss_inr). `ratio` is NaN when no
        merchant is bad.
    """
    # Local import: `eval.metrics` is the single home of the cost matrix and this
    # module is imported by `eval.splits`, so a module-level import would cycle.
    from rakshak.eval.metrics import false_positive_cost

    y = np.asarray(y_true, dtype=float)
    loss = np.asarray(loss_inr, dtype=float)
    total_fp = float(false_positive_cost(np.asarray(value_inr, dtype=float))[y == 0.0].sum())
    total_loss = float(loss[y == 1.0].sum())
    ratio = 100.0 * total_fp / total_loss if total_loss > 0.0 else float("nan")
    return ratio, total_fp, total_loss


# ---------------------------------------------------------------------------
# The oracle-dominance invariant (T-0007a)
# ---------------------------------------------------------------------------


def trivial_policy_savings(
    y_true: np.ndarray,
    loss_inr: np.ndarray,
    value_inr: np.ndarray,
    seed: int = SEED,
    savings_fn: Callable[[np.ndarray], float] | None = None,
) -> dict[str, float]:
    """Savings of the three trivial policies: pass-everything, hold-everything, random.

    By Bahnsen's definition Cost_l = min(Cost(all PASS), Cost(all HOLD)), so the
    better of the first two scores exactly 0.0 and the worse scores <= 0. They are
    computed anyway rather than asserted, because an oracle below zero is precisely
    the T-0006 defect this invariant exists to catch.

    Args:
        y_true: Binary labels, shape (n,).
        loss_inr: L_m per merchant. Units: INR.
        value_inr: V_m per merchant. Units: INR.
        seed: Seed for the random policy. Determinism is NFR-003.
        savings_fn: Optional `actions -> savings` scorer. Defaults to
            `eval.metrics.savings_score` at the shipping cost primitives. T-0007b's
            FR-020 sweep passes its own so the invariant is evaluated under the
            same cost matrix the policies were, rather than under config's
            central values.

    Returns:
        Policy name -> savings score.
    """
    from rakshak.eval.metrics import HOLD, PASS, REVIEW, savings_score

    y = np.asarray(y_true, dtype=float)
    n = y.size
    score = savings_fn or (lambda actions: savings_score(y, actions, loss_inr, value_inr))
    rng = np.random.default_rng(seed)
    return {
        "trivial: pass-everything": score(np.full(n, PASS)),
        "trivial: hold-everything": score(np.full(n, HOLD)),
        "trivial: random actions": score(rng.choice([PASS, REVIEW, HOLD], size=n)),
    }


def assert_oracle_dominance(
    y_true: np.ndarray,
    loss_inr: np.ndarray,
    value_inr: np.ndarray,
    oracle_savings: Mapping[str, float],
    policy_savings: Mapping[str, float] | None = None,
    seed: int = SEED,
    tol: float = 1e-9,
    savings_fn: Callable[[np.ndarray], float] | None = None,
) -> dict[str, float]:
    """A perfect-foresight ceiling must weakly dominate every policy in the run.

    The executable form of "set the oracle". Run as a test and as a harness
    precondition. At T-0006 the review-knapsack oracle scored -0.678 while
    hold-everything scored 0.000; a ceiling a trivial policy beats is not a
    ceiling, and this assertion would have caught it at T-0005.

    Args:
        y_true: Binary labels, shape (n,).
        loss_inr: L_m per merchant. Units: INR.
        value_inr: V_m per merchant. Units: INR.
        oracle_savings: Ceiling name -> savings. Every one must dominate.
        policy_savings: Scored model name -> savings. The three trivial policies
            are added automatically and need not be passed.
        seed: Seed for the random trivial policy (NFR-003).
        tol: Absolute slack, to absorb float noise only.
        savings_fn: Optional `actions -> savings` scorer for the trivial policies,
            forwarded to `trivial_policy_savings`. Pass it whenever the cost
            primitives in force differ from `config.py`'s central values.

    Returns:
        The full policy name -> savings mapping that was checked, for reporting.

    Raises:
        AssertionError: If any policy strictly beats any ceiling by more than
            `tol`. The message names every violation.
    """
    if not oracle_savings:
        raise ValueError("oracle_savings is empty — there is no ceiling to check")

    policies = dict(policy_savings or {})
    policies.update(
        trivial_policy_savings(
            y_true, loss_inr, value_inr, seed=seed, savings_fn=savings_fn
        )
    )

    violations = [
        f"  {policy!r} scores {value:+.4f} > ceiling {oracle!r} at {ceiling:+.4f}"
        f"  (excess {value - ceiling:+.4f})"
        for oracle, ceiling in oracle_savings.items()
        for policy, value in policies.items()
        if value > ceiling + tol
    ]
    if violations:
        raise AssertionError(
            "oracle-dominance invariant FAILED (T-0007a; 07-math.md §7).\n"
            "A perfect-foresight ceiling must weakly dominate every policy scored "
            "in the same run.\n" + "\n".join(violations) + "\n"
            "This is a mis-specified cost matrix, not a bad oracle. Do NOT tune "
            "constants until it passes — read 07-math.md §5 and report the failure."
        )
    return policies
