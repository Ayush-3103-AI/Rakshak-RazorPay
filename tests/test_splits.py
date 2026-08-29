"""NFR-002: zero merchant overlap between splits. This file fails the build.

Also pins the temporal boundaries frozen in 06-requirements.md §3 and the
structural lock on the test window.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rakshak.config import STATE_PATHS_PARQUET, TRANSACTIONS_PARQUET
from rakshak.eval.splits import (
    BAD_STATES,
    SPLIT_DAY_BOUNDS,
    assert_no_leakage,
    assert_window_is_frozen,
    assign_merchant_groups,
    load_split,
    split_summary,
)

pytestmark = pytest.mark.skipif(
    not (TRANSACTIONS_PARQUET.exists() and STATE_PATHS_PARQUET.exists()),
    reason="run `python -m rakshak.generator --seed 42` first",
)


@pytest.fixture(scope="module")
def state_paths() -> pd.DataFrame:
    return pd.read_parquet(STATE_PATHS_PARQUET)


# ---------------------------------------------------------------------------
# NFR-002 — the one that fails the build
# ---------------------------------------------------------------------------


def test_no_merchant_id_appears_in_more_than_one_split(state_paths: pd.DataFrame) -> None:
    groups = assign_merchant_groups(state_paths)
    assert not groups.index.has_duplicates
    seen: dict[str, str] = {}
    for split_name in SPLIT_DAY_BOUNDS:
        for merchant_id in groups.index[groups == split_name]:
            assert merchant_id not in seen, (
                f"NFR-002: {merchant_id} is in both {seen[merchant_id]} and {split_name}"
            )
            seen[merchant_id] = split_name
    assert len(seen) == state_paths["merchant_id"].nunique()


def test_loaded_splits_share_no_merchant(state_paths: pd.DataFrame) -> None:
    train = set(load_split("train").merchant_ids)
    validate = set(load_split("validate").merchant_ids)
    test = set(load_split("test", unlock_test="T-0011").merchant_ids)
    assert train & validate == set()
    assert train & test == set()
    assert validate & test == set()
    assert len(train | validate | test) == state_paths["merchant_id"].nunique()


def test_guard_catches_a_crossing_merchant() -> None:
    """The guard must actually fire — a guard that never fails is a comment."""
    bad = pd.Series(
        ["train", "test"], index=pd.Index(["M1", "M1"], name="merchant_id"), name="split"
    )
    with pytest.raises(AssertionError, match="NFR-002"):
        assert_no_leakage(bad)


def test_guard_rejects_unknown_split_name() -> None:
    bad = pd.Series(["holdout"], index=pd.Index(["M1"], name="merchant_id"), name="split")
    with pytest.raises(AssertionError, match="unknown split"):
        assert_no_leakage(bad)


# ---------------------------------------------------------------------------
# Temporal boundaries (06-requirements.md §3)
# ---------------------------------------------------------------------------


def test_temporal_windows_match_the_frozen_spec() -> None:
    assert_window_is_frozen()
    assert SPLIT_DAY_BOUNDS["train"] == (0, 180)
    assert SPLIT_DAY_BOUNDS["validate"] == (180, 210)
    assert SPLIT_DAY_BOUNDS["test"] == (210, 270)


def test_windows_are_contiguous_and_non_overlapping() -> None:
    bounds = [SPLIT_DAY_BOUNDS[n] for n in ("train", "validate", "test")]
    for (_, end), (start, _) in zip(bounds, bounds[1:], strict=False):
        assert end == start


def test_no_transaction_falls_after_its_window_end() -> None:
    for name in ("train", "validate"):
        split = load_split(name)
        assert split.transactions["day"].max() < split.end_day
        window = split.window_transactions
        assert window["day"].min() >= split.start_day
        assert window["day"].max() < split.end_day


# ---------------------------------------------------------------------------
# Test-window lock (06-requirements.md §3: touched exactly once, at the end)
# ---------------------------------------------------------------------------


def test_test_split_is_locked_without_an_authorised_ticket() -> None:
    with pytest.raises(PermissionError, match="T-0011"):
        load_split("test")
    with pytest.raises(PermissionError):
        load_split("test", unlock_test="T-0006")


def test_test_split_opens_for_the_authorised_tickets() -> None:
    for ticket in ("T-0011", "T-0013"):
        assert load_split("test", unlock_test=ticket).n_merchants > 0


# ---------------------------------------------------------------------------
# Labels and stratification
# ---------------------------------------------------------------------------


def test_ramp_is_a_bad_state(state_paths: pd.DataFrame) -> None:
    """T-0003 finding: SLOW_RAMP merchants never leave RAMP. Excluding RAMP from
    the bad set would make the adversarial typology undetectable by construction."""
    assert "RAMP" in BAD_STATES
    slow_ramp = state_paths[state_paths["typology"] == "SLOW_RAMP"]
    assert set(slow_ramp["state"]) == {"HEALTHY", "RAMP"}

    labelled_bad = 0
    for name in ("train", "validate"):
        split = load_split(name)
        ids = set(slow_ramp["merchant_id"]) & set(split.merchant_ids)
        labelled_bad += int(split.labels.reindex(sorted(ids)).sum())
    assert labelled_bad > 0, "SLOW_RAMP merchants must be labelled bad somewhere"


def test_labels_agree_with_the_transition_day() -> None:
    split = load_split("validate")
    assert (split.labels == split.transition_day.notna().astype(int)).all()
    assert split.transition_day.dropna().max() < split.end_day
    # Healthy merchants carry no fraud loss.
    assert float(split.loss_inr[split.labels == 0].sum()) == 0.0


def test_every_typology_appears_in_every_split(state_paths: pd.DataFrame) -> None:
    table = split_summary(state_paths)
    assert (table.drop(index="TOTAL") > 0).all().all()
    assert int(table.loc["TOTAL"].sum()) == state_paths["merchant_id"].nunique()


def test_merchant_group_assignment_is_seed_independent(state_paths: pd.DataFrame) -> None:
    """The frozen eval must not move when someone changes --seed."""
    first = assign_merchant_groups(state_paths)
    shuffled = state_paths.sample(frac=1.0, random_state=7)
    assert first.equals(assign_merchant_groups(shuffled))
