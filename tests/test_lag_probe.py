"""The detection-lag probe (T-0011), against hand-built fixtures.

Three things are pinned here and nothing else, because the probe's expensive
half — fitting LightGBM and the HMM on the full population — is already covered
by `tests/test_gbdt.py` and `tests/test_hmm_score.py`:

1. `metrics.detection_lag_days` **converts nothing** — it takes `flag_day` as a
   calendar day. A conversion there would apply to `models/rules.py`, which never
   used the window-start convention, and double-count it.
2. The window-based scorers attribute a flag to the **last** day of the window
   that raised it, exactly `WINDOW_DAYS - 1` days after its start. That constant
   is the whole finding: it is what turned the reported -1.0 into +5.0.
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
from rakshak.models import hmm_score


def _series(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, index=pd.Index(list(values), name="merchant_id"), dtype="float64")


# ---------------------------------------------------------------------------
# 1 + 2 — the attribution argument
# ---------------------------------------------------------------------------


def test_detection_lag_takes_flag_day_as_given() -> None:
    """`metrics` must not convert. Every scorer already reports a calendar day.

    A conversion here is how the defect T-0011 found is reintroduced: it would
    apply to `rules`, which never used the window-start convention, as readily
    as to the window-based models.
    """
    labels = pd.Series({"A": 1, "B": 1})
    transition = _series({"A": 190.0, "B": 200.0})
    flags = _series({"A": 195.0, "B": 202.0})
    lag, _, _ = metrics.detection_lag_days(flags, transition, labels)
    assert lag == pytest.approx(3.5)  # 195-190 = 5 and 202-200 = 2; median 3.5


def test_window_scorers_report_the_last_day_of_their_window() -> None:
    """The fix, pinned at source: a window's flag lands on the day it completes.

    `first_flag_day` is handed window START days and must return the last day of
    the window that fired. `models/rules.py` already reported the last day of its
    own trailing evidence; this is what makes the two comparable.
    """
    probability = np.array([0.1, 0.9, 0.95])
    eligible = np.array([True, True, True])
    window_start_day = np.array([182, 189, 196])
    assert hmm_score.first_flag_day(probability, eligible, window_start_day) == pytest.approx(
        189 + WINDOW_DAYS - 1
    )
    assert math.isnan(
        hmm_score.first_flag_day(np.zeros(3), eligible, window_start_day)
    )


def test_superseded_convention_is_reconstructable_and_produces_the_negative_lag() -> None:
    """Subtracting the offset reproduces the -1.0 the repo printed before T-0011.

    A merchant whose onset is day 190 is flagged from the window covering days
    189-195. Shipped, that is day 195 and a lag of +5. Under the superseded
    window-start convention it was day 189 and a lag of -1 — earliness the model
    never had, because the evidence was not complete until day 195.
    """
    labels = pd.Series({"A": 1})
    transition = _series({"A": 190.0})
    shipped = _series({"A": 195.0})
    superseded = shipped - metrics.WINDOW_ATTRIBUTION_OFFSET_DAYS
    assert metrics.detection_lag_days(shipped, transition, labels)[0] == pytest.approx(5.0)
    assert metrics.detection_lag_days(superseded, transition, labels)[0] == pytest.approx(-1.0)


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
