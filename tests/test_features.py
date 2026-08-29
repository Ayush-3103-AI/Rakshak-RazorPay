"""T-0004 — the feature layer's own contracts (FR-007 through FR-011).

The headline is `test_relative_behaviour_survives_100x_aov`: FR-007's stated acceptance
test, and the one that proves within-merchant standardisation is actually doing its job
rather than merely being present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rakshak.config import SEED
from rakshak.features import (
    BASE_FEATURES,
    MIN_SEGMENT_MERCHANTS,
    VULCAN_FEATURES,
    build_emissions,
    build_window_features,
    fit_segment_map,
    window_state_labels,
)

_METHODS = ("UPI", "CARD", "NETBANKING")
_EPOCH = pd.Timestamp("2026-01-01")


def synthetic_merchant(
    merchant_id: str,
    aov: float,
    mcc: str = "5411",
    n_windows: int = 20,
    per_window: int = 30,
    seed: int = SEED,
) -> pd.DataFrame:
    """Build one merchant's stream with a fixed relative shape and a chosen ticket scale.

    Two calls differing only in ``aov`` and ``merchant_id`` produce streams that are
    identical up to a multiplicative amount factor: same timing, same payer structure,
    same method mix, same refund and chargeback positions. That is exactly the pair
    FR-007's acceptance test needs.

    Args:
        merchant_id: Identifier to stamp on every row.
        aov: Ticket-size scale in INR; amounts are ``aov`` times a fixed shape.
        mcc: Category code.
        n_windows: Number of 7-day windows to emit.
        per_window: Transactions per window.
        seed: Seed for the shape, shared across calls so the shape is common.

    Returns:
        Frame in the generator's transaction schema, sorted by timestamp.
    """
    rng = np.random.default_rng(seed)
    n = n_windows * per_window
    shape = np.exp(rng.normal(0.0, 0.4, n))
    day = np.repeat(np.arange(n_windows) * 7, per_window) + rng.integers(0, 7, n)
    hour = rng.integers(8, 22, n)
    payer = rng.integers(0, 40, n)
    return pd.DataFrame(
        {
            "merchant_id": merchant_id,
            "timestamp": _EPOCH + pd.to_timedelta(day * 24 + hour, unit="h"),
            "amount": np.round(aov * shape, 2),
            "payer_id": [f"{merchant_id}-P{p}" for p in payer],
            "method": [_METHODS[i] for i in rng.integers(0, len(_METHODS), n)],
            "mcc": mcc,
            "is_refund": rng.random(n) < 0.05,
            "is_chargeback": rng.random(n) < 0.01,
        }
    ).sort_values("timestamp", ignore_index=True)


# ------------------------------------------------------------------------------------
# FR-007 — the acceptance test named in the ticket's "Done when"
# ------------------------------------------------------------------------------------


def test_relative_behaviour_survives_100x_aov() -> None:
    """FR-007: identical relative behaviour, 100x different AOV, same emissions.

    A 300-INR grocer and a 30 000-INR jeweller behaving identically relative to their own
    norms must be indistinguishable in emission space. If this fails, the model measures
    "is a jeweller" instead of "has changed", which is the 2008-era cardholder-HMM
    false-positive failure P-02 exists to prevent.
    """
    small = synthetic_merchant("M_SMALL", aov=300.0)
    large = synthetic_merchant("M_LARGE", aov=30_000.0)
    # Exactly 100x rather than 100x-then-rounded: rounding to paise is coarser relative
    # to a 300-INR ticket than to a 30 000-INR one, and that quantisation alone leaves a
    # 7.5e-5 residual that has nothing to do with standardisation.
    large["amount"] = small["amount"].to_numpy() * 100.0
    assert large["amount"].mean() / small["amount"].mean() == pytest.approx(100.0, rel=1e-12)

    emissions = build_emissions(pd.concat([small, large], ignore_index=True))
    i_small = list(emissions.merchant_ids).index("M_SMALL")
    i_large = list(emissions.merchant_ids).index("M_LARGE")
    gap = np.abs(emissions.X[i_small] - emissions.X[i_large]).max()
    assert gap < 1e-9, (
        f"standardised emissions differ by {gap:.3g} between two merchants whose only "
        "difference is a 100x ticket-size scale; FR-007 is not working"
    )


def test_raw_features_do_differ_without_standardisation() -> None:
    """The FR-007 test must be able to fail: raw log ticket size differs by ln(100)."""
    panel, _ = build_window_features(
        pd.concat(
            [synthetic_merchant("M_SMALL", 300.0), synthetic_merchant("M_LARGE", 30_000.0)],
            ignore_index=True,
        )
    )
    means = panel.groupby(level="merchant_id")["log_amount_mean"].mean()
    assert means["M_LARGE"] - means["M_SMALL"] == pytest.approx(np.log(100.0), abs=1e-6)


# ------------------------------------------------------------------------------------
# FR-008 / FR-009 / FR-010 — emission vector contents
# ------------------------------------------------------------------------------------


def test_emission_vector_carries_every_required_feature() -> None:
    """FR-008 and FR-009 name these features explicitly; all must be present."""
    required = {
        # FR-008 graph-derived scalars (ADR-0002)
        "payer_entropy",
        "repeat_payer_ratio",
        "payer_jaccard_prev",
        "payer_herfindahl",
        # FR-009 behavioural / financial
        "log_amount_mean",
        "log_amount_var",
        "log_velocity",
        "refund_ratio",
        "chargeback_ratio",
        "chargeback_lag_days",
        "hour_entropy",
        "method_entropy",
        "new_payer_ratio",
    }
    assert required <= set(BASE_FEATURES)


def test_vulcan_score_present_and_absent(caplog: pytest.LogCaptureFixture) -> None:
    """FR-010: the Vulcan proxy is consumed when present and logged when absent."""
    txns = synthetic_merchant("M_A", 500.0)
    with caplog.at_level("INFO", logger="rakshak.features.windows"):
        _, without = build_window_features(txns)
    assert not set(VULCAN_FEATURES) & set(without)
    assert "omitted" in caplog.text

    scored = txns.assign(risk_score=np.linspace(0.0, 1.0, len(txns)))
    _, with_vulcan = build_window_features(scored)
    assert set(VULCAN_FEATURES) <= set(with_vulcan)
    assert len(with_vulcan) == len(without) + len(VULCAN_FEATURES)


def test_jaccard_and_repeat_ratio_on_a_hand_checkable_case() -> None:
    """Graph scalars must be arithmetically right, not merely plausible."""
    rows = []
    # Window 0: payers a, b.  Window 1: payers b, c.  Jaccard(w1) = |{b}| / |{a,b,c}|.
    for day, payers in ((0, "ab"), (7, "bc")):
        for p in payers:
            rows.append(
                {
                    "merchant_id": "M",
                    "timestamp": _EPOCH + pd.Timedelta(days=day, hours=10),
                    "amount": 100.0,
                    "payer_id": f"M-P{p}",
                    "method": "UPI",
                    "mcc": "5411",
                    "is_refund": False,
                    "is_chargeback": False,
                }
            )
    panel, _ = build_window_features(pd.DataFrame(rows))
    assert panel.loc[("M", 1), "payer_jaccard_prev"] == pytest.approx(1.0 / 3.0)
    assert panel.loc[("M", 1), "repeat_payer_ratio"] == pytest.approx(0.5)
    assert panel.loc[("M", 0), "payer_jaccard_prev"] == pytest.approx(0.0)
    # Two payers, one transaction each: entropy ln(2), Herfindahl 2 * 0.5^2.
    assert panel.loc[("M", 0), "payer_entropy"] == pytest.approx(np.log(2.0))
    assert panel.loc[("M", 0), "payer_herfindahl"] == pytest.approx(0.5)


def test_empty_windows_are_kept_not_dropped() -> None:
    """08-pseudocode.md §C: dropping a silent window desynchronises the ground truth."""
    txns = synthetic_merchant("M_A", 500.0, n_windows=6)
    txns = txns[(txns["timestamp"] - _EPOCH).dt.days // 7 != 3]  # silence window 3
    panel, names = build_window_features(txns)
    assert len(panel) == 6
    assert panel.loc[("M_A", 3), "sparse"] == 1.0
    assert panel.loc[("M_A", 3), "log_velocity"] == 0.0
    assert not panel[list(names)].isna().to_numpy().any()


# ------------------------------------------------------------------------------------
# FR-011 — segmentation
# ------------------------------------------------------------------------------------


def test_every_segment_meets_the_twenty_merchant_floor() -> None:
    """FR-011: MCC x AOV-band, and no training segment below 20 merchants."""
    rng = np.random.default_rng(SEED)
    mcc = pd.Series(rng.choice(["5411", "5812", "5944"], 300))
    aov = pd.Series(np.exp(rng.normal(7.0, 1.5, 300)))
    labels = fit_segment_map(mcc, aov).assign(mcc, aov)
    counts = labels.value_counts()
    assert counts.min() >= MIN_SEGMENT_MERCHANTS, counts.to_dict()
    assert labels.str.count(":").eq(1).all()


def test_segment_map_is_reusable_on_held_out_merchants() -> None:
    """Band edges fitted on training merchants must apply to unseen ones unchanged."""
    rng = np.random.default_rng(SEED)
    mcc = pd.Series(["5411"] * 100)
    train_aov = pd.Series(np.exp(rng.normal(7.0, 1.0, 100)))
    fitted = fit_segment_map(mcc, train_aov)
    held_out = pd.Series([1.0, float(train_aov.max()) * 10.0], index=[0, 1])
    labels = fitted.assign(pd.Series(["5411", "5411"]), held_out)
    assert labels.iloc[0].endswith("LOW")
    assert labels.iloc[1].endswith("HIGH")


def test_window_state_labels_takes_the_modal_state() -> None:
    """A window spanning a transition is labelled by where the merchant mostly was."""
    paths = pd.DataFrame(
        {
            "merchant_id": ["M", "M"],
            "state": ["HEALTHY", "FRAUD"],
            "start_day": [0, 5],
            "end_day": [5, 14],
        }
    )
    labels = window_state_labels(paths, np.array(["M"]), n_windows=2)
    assert labels[0, 0] == "HEALTHY"  # days 0-4 healthy, 5-6 fraud
    assert labels[0, 1] == "FRAUD"
