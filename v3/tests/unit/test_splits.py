"""T-130 — the split engine. Synthetic fixtures only; the generator does not exist yet.

That is deliberate (STATE.md): a harness frozen before the generator is finished is
harder to accuse of hindsight than one tuned against data it has already seen.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from rakshak.eval.splits import (
    DEFAULT_BOUNDARIES,
    SplitBoundaries,
    assign_rows,
    available_labels,
    label_coverage,
    merchant_fold,
    split_of_day,
)
from rakshak.schemas import LABEL_SCHEMA

ORIGIN = date(2026, 1, 1)


def _ts(day: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC).replace() + (
        datetime(2026, 1, 2, tzinfo=UTC) - datetime(2026, 1, 1, tzinfo=UTC)
    ) * day


def _labels(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=LABEL_SCHEMA)


# ─────────────────────────── temporal ───────────────────────────


@pytest.mark.parametrize(
    ("day", "expected"),
    [(0, "train"), (119, "train"), (120, "val"), (149, "val"), (150, "test"), (179, "test")],
)
def test_split_of_day_boundaries_are_exact(day: int, expected: str) -> None:
    assert split_of_day(day) == expected


@pytest.mark.parametrize("day", [-1, 180, 10_000])
def test_days_outside_the_window_belong_to_no_split(day: int) -> None:
    assert split_of_day(day) is None


def test_every_day_in_the_window_is_covered_exactly_once() -> None:
    assigned = [split_of_day(d) for d in range(DEFAULT_BOUNDARIES.n_days)]
    assert None not in assigned
    assert len(assigned) == 180


def test_non_contiguous_boundaries_are_rejected() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        SplitBoundaries(origin=ORIGIN, train=(0, 119), val=(125, 149), test=(150, 179))
    with pytest.raises(ValueError, match="contiguous"):
        SplitBoundaries(origin=ORIGIN, train=(0, 130), val=(120, 149), test=(150, 179))


def test_day_index_accepts_date_and_tz_aware_datetime_and_refuses_naive() -> None:
    assert DEFAULT_BOUNDARIES.day_index(date(2026, 1, 31)) == 30
    assert DEFAULT_BOUNDARIES.day_index(datetime(2026, 1, 31, 23, 59, tzinfo=UTC)) == 30
    with pytest.raises(ValueError, match="tz-aware"):
        DEFAULT_BOUNDARIES.day_index(datetime(2026, 1, 31))  # noqa: DTZ001


# ─────────────────────────── merchant-group ───────────────────────────


def test_merchant_fold_is_deterministic_and_not_process_salted() -> None:
    # If this ever depends on PYTHONHASHSEED, merchants move between splits between runs
    # and every comparison across sessions is meaningless.
    assert merchant_fold("M000042") == merchant_fold("M000042")
    assert merchant_fold("M000042") == "test" or merchant_fold("M000042") in {"train", "val"}


def test_merchant_folds_are_roughly_proportional_to_the_temporal_shares() -> None:
    folds = [merchant_fold(f"M{i:06d}") for i in range(10_000)]
    train = folds.count("train") / len(folds)
    assert 0.64 < train < 0.70, f"train fold share drifted to {train}"
    assert set(folds) == {"train", "val", "test"}


def test_no_merchant_id_spans_two_splits() -> None:
    """T-130's done-when. A merchant contributes rows to at most one split, ever."""
    frame = pl.DataFrame(
        {
            "merchant_id": [f"M{i:06d}" for i in range(300) for _ in range(6)],
            "as_of": [d for _ in range(300) for d in (0, 60, 119, 130, 155, 179)],
        }
    )
    assigned = assign_rows(frame, DEFAULT_BOUNDARIES).drop_nulls("split")
    per_merchant = assigned.group_by("merchant_id").agg(pl.col("split").n_unique().alias("n"))
    assert per_merchant["n"].max() == 1
    assert assigned.height > 0


def test_rows_failing_either_constraint_are_excluded_not_reassigned() -> None:
    mid = "M000001"
    fold = merchant_fold(mid)
    other_day = {"train": 130, "val": 10, "test": 10}[fold]
    frame = pl.DataFrame({"merchant_id": [mid], "as_of": [other_day]})
    assert assign_rows(frame)["split"].to_list() == [None]


def test_assign_rows_accepts_a_date_column() -> None:
    frame = pl.DataFrame(
        {"merchant_id": ["M000001", "M000001"], "as_of": [date(2026, 1, 1), date(2026, 6, 29)]}
    )
    out = assign_rows(frame)
    assert out.height == 2
    assert out["split"].null_count() >= 1


# ─────────────────────────── label availability ───────────────────────────


def _three_labels() -> pl.DataFrame:
    return _labels(
        [
            # resolved and available on day 100
            {
                "merchant_id": "M_early",
                "label": 1,
                "label_event_at": _ts(10),
                "label_available_at": _ts(60),
                "label_source": "chargeback",
                "is_censored": False,
                "schema_version": 2,
            },
            # fraud on day 100, chargeback lands day 175 — invisible at day 120
            {
                "merchant_id": "M_late",
                "label": 1,
                "label_event_at": _ts(100),
                "label_available_at": _ts(175),
                "label_source": "chargeback",
                "is_censored": False,
                "schema_version": 2,
            },
            # never resolves
            {
                "merchant_id": "M_censored",
                "label": None,
                "label_event_at": None,
                "label_available_at": None,
                "label_source": "none",
                "is_censored": True,
                "schema_version": 2,
            },
            {
                "merchant_id": "M_good",
                "label": 0,
                "label_event_at": _ts(5),
                "label_available_at": _ts(50),
                "label_source": "manual_review",
                "is_censored": False,
                "schema_version": 2,
            },
        ]
    )


def test_a_label_not_yet_available_is_not_returned() -> None:
    """The defining domain constraint: a day-100 fraud is unlabelled at day 120."""
    got = available_labels(_ts(120), _three_labels()).collect()["merchant_id"].to_list()
    assert got == ["M_early", "M_good"]
    assert "M_late" not in got

    later = available_labels(_ts(179), _three_labels()).collect()["merchant_id"].to_list()
    assert "M_late" in later


def test_the_boundary_is_inclusive_and_off_by_one_is_visible() -> None:
    labels = _three_labels()
    assert available_labels(_ts(175), labels).collect().height == 3
    assert available_labels(_ts(174), labels).collect().height == 2


def test_censored_merchants_are_excluded_and_counted() -> None:
    cov = label_coverage(_ts(179), _three_labels())
    assert cov.n_censored == 1
    assert cov.n_merchants == 4
    assert cov.n_available == 3
    assert cov.n_pending == 0
    assert "M_censored" not in available_labels(_ts(179), _three_labels()).collect()[
        "merchant_id"
    ].to_list()
    # ...and reachable on purpose, for a coverage report, never by default.
    assert (
        available_labels(_ts(179), _three_labels(), include_censored=True).collect().height == 3
    )


def test_pending_labels_are_counted_separately_from_censored() -> None:
    cov = label_coverage(_ts(120), _three_labels())
    assert (cov.n_available, cov.n_censored, cov.n_pending) == (2, 1, 1)


def test_prevalence_is_computed_over_available_uncensored_labels_only() -> None:
    cov = label_coverage(_ts(179), _three_labels())
    assert cov.n_positive == 2
    assert cov.prevalence == pytest.approx(2 / 3)
    assert cov.coverage == pytest.approx(3 / 4)


def test_naive_as_of_is_refused() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        available_labels(datetime(2026, 4, 1), _three_labels())  # noqa: DTZ001
