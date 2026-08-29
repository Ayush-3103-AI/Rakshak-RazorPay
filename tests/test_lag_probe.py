"""The detection-lag probe (T-0011), against hand-built fixtures.

Three things are pinned here and nothing else, because the probe's expensive
half — fitting LightGBM and the HMM on the full population — is already covered
by `tests/test_gbdt.py` and `tests/test_hmm_score.py`:

1. `metrics.detection_lag_days` is **unchanged at its default**. The attribution
   argument was added to `metrics.py` additively, and if the default ever moves,
   every median lag already printed in `results/summary.md` silently changes.
2. Window-end attribution shifts a window-based lag by **exactly
   `WINDOW_DAYS - 1` days**, no more and no less. That constant is the whole
   finding: it is what turns the reported -1.0 into +5.0.
3. The rendered document reports **both** attributions. The document exists to
   let a reader compare the two conventions; one of them going missing would be
   a silent regression in the only artefact that clears `summary.md`'s numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rakshak.config import WINDOW_DAYS
from rakshak.eval import lag_probe, metrics
from rakshak.eval.splits import Split


def _series(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, index=pd.Index(list(values), name="merchant_id"), dtype="float64")


# ---------------------------------------------------------------------------
# 1 + 2 — the attribution argument
# ---------------------------------------------------------------------------


def test_default_attribution_is_unchanged() -> None:
    """The default must stay window-start, or every published lag moves."""
    labels = pd.Series({"A": 1, "B": 1})
    transition = _series({"A": 190.0, "B": 200.0})
    flags = _series({"A": 189.0, "B": 196.0})
    # Lags 189-190 = -1 and 196-200 = -4; median -2.5. This is the historical
    # convention and reproduces the sign of the -1.0 in results/summary.md.
    explicit = metrics.detection_lag_days(flags, transition, labels, attribution="window_start")
    assert metrics.detection_lag_days(flags, transition, labels) == explicit
    assert explicit[0] == pytest.approx(-2.5)


def test_window_end_attribution_shifts_by_exactly_window_days_minus_one() -> None:
    labels = pd.Series({"A": 1, "B": 1, "C": 1, "D": 0})
    transition = _series({"A": 190.0, "B": 200.0, "C": 205.0, "D": math.nan})
    # A flags from the window opening day 189, i.e. the window covering days
    # 189-195, which contains A's onset on day 190. Window-start attribution
    # records -1; the model could not have fired before day 195.
    flags = _series({"A": 189.0, "B": 196.0, "C": math.nan, "D": 182.0})
    start_lag, start_flagged, n_bad = metrics.detection_lag_days(flags, transition, labels)
    end_lag, end_flagged, _ = metrics.detection_lag_days(
        flags, transition, labels, attribution="window_end"
    )
    assert start_lag == pytest.approx(-2.5)
    assert end_lag - start_lag == pytest.approx(WINDOW_DAYS - 1)
    assert end_lag == pytest.approx(-2.5 + WINDOW_DAYS - 1)
    # Only the day attribution moves: who was flagged at all does not.
    assert (start_flagged, n_bad) == (end_flagged, 3)


def test_unknown_attribution_is_rejected() -> None:
    labels = pd.Series({"A": 1})
    with pytest.raises(ValueError, match="unknown attribution"):
        metrics.detection_lag_days(
            _series({"A": 189.0}), _series({"A": 190.0}), labels, attribution="midpoint"
        )


# ---------------------------------------------------------------------------
# 3 — the rendered document
# ---------------------------------------------------------------------------


def _fake_split(name: str, start_day: int, end_day: int) -> Split:
    """A Split carrying only the fields `lag_probe.render` reads."""
    empty = pd.DataFrame({"merchant_id": [], "day": []})
    blank = pd.Series(dtype="float64")
    return Split(
        name=name,
        start_day=start_day,
        end_day=end_day,
        merchant_ids=("M0",),
        transactions=empty,
        labels=blank,
        transition_day=blank,
        transition_timestamp=blank,
        loss_inr=blank,
        value_inr=blank,
    )


def _fake_lag_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "rules",
                "lag_start_days": 3.0,
                "lag_end_days": 3.0 + WINDOW_DAYS - 1,
                "delta_days": float(WINDOW_DAYS - 1),
                "flagged_fraction": 0.45,
                "n_bad": 20,
                "n_flagged": 9,
                "n_distinct_flag_days": 6,
            },
            {
                "model": "gbdt",
                "lag_start_days": -1.0,
                "lag_end_days": -1.0 + WINDOW_DAYS - 1,
                "delta_days": float(WINDOW_DAYS - 1),
                "flagged_fraction": 0.50,
                "n_bad": 20,
                "n_flagged": 10,
                "n_distinct_flag_days": 4,
            },
            {
                "model": "hmm",
                "lag_start_days": -1.0,
                "lag_end_days": -1.0 + WINDOW_DAYS - 1,
                "delta_days": float(WINDOW_DAYS - 1),
                "flagged_fraction": 0.65,
                "n_bad": 20,
                "n_flagged": 13,
                "n_distinct_flag_days": 4,
            },
        ]
    )


def _rendered() -> str:
    split = _fake_split("validate", 180, 210)
    table = pd.DataFrame(
        {"feature": ["log_velocity"], "auc": [0.52], "abs_effect": [0.02], "naive_z": [0.3]}
    )
    counts = {
        "n_pre_windows": 24,
        "n_pre_merchants": 13,
        "n_healthy_windows": 320,
        "n_healthy_merchants": 80,
        "n_bad_merchants": 20,
        "observed_max_effect": 0.02,
        "null_p95_max_effect": 0.22,
        "p_value": 0.9,
        "n_permutations": lag_probe.N_PERMUTATIONS,
    }
    return lag_probe.render(
        {"validate": _fake_lag_table()},
        {"validate": split},
        {"validate": np.array([182, 189, 196, 203])},
        {"validate": (table, counts)},
        seed=42,
    )


def test_document_reports_both_attributions() -> None:
    document = _rendered()
    assert "window-START" in document
    assert "window-END" in document
    # Both numbers for the same model, on the same row: -1.0 as published and
    # +5.0 as corrected. If either disappears the document stops being a
    # comparison and becomes an assertion.
    assert "| gbdt | " in document
    row = next(line for line in document.splitlines() if line.startswith("| gbdt |"))
    assert "-1.0" in row
    assert f"{-1.0 + WINDOW_DAYS - 1:.1f}" in row
    assert f"{float(WINDOW_DAYS - 1):.1f}" in row


def test_document_states_the_quantisation_and_the_sample_size() -> None:
    document = _rendered()
    assert "4 distinct days" in document  # the validate window grid
    assert "10 of 20" in document  # merchants behind gbdt's median
    assert "not a precise quantity" in document


def test_document_clears_the_leakage_suspicion_when_p_is_large() -> None:
    document = _rendered()
    assert "NOT separable" in document
    assert "STOP" not in document


def test_render_is_deterministic() -> None:
    """NFR-003: no wall-clock time, no dict-order dependence in the document."""
    assert _rendered() == _rendered()
