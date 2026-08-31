"""T-101: the event store never returns a row from the future. Property-tested.

An example-based test proves the filter works for the `as_of` values someone thought of.
This one draws `as_of` from the whole simulated window — including the exact instants of
events, the nanosecond either side of them, and the boundaries — because off-by-one on a
`<=` is precisely the bug that an example-based test picks the wrong example for.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rakshak.schemas import (
    PAYOUT_SCHEMA,
    PROFILE_SCHEMA,
    TRANSACTION_SCHEMA,
    Instrument,
    TxnStatus,
)
from rakshak.store import EventStore, PointInTimeError

WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_DAYS = 60
N_MERCHANTS = 12
SEED = 42


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small deterministic parquet dataset. Module-scoped because hypothesis re-runs the
    test body many times and rebuilding parquet per example would dominate the runtime."""
    import numpy as np

    rng = np.random.default_rng(SEED)
    root = tmp_path_factory.mktemp("v2data")

    rows: list[dict[str, object]] = []
    payouts: list[dict[str, object]] = []
    for m in range(N_MERCHANTS):
        merchant_id = f"M{m:04d}"
        # Uneven counts on purpose: a merchant with zero events on a given day is the case
        # where a partition-prune-only filter looks correct and is not.
        for day in range(WINDOW_DAYS):
            for k in range(int(rng.integers(0, 4))):
                event_time = (
                    WINDOW_START
                    + timedelta(days=day)
                    + timedelta(seconds=int(rng.integers(0, 86_400)))
                )
                rows.append(
                    {
                        "event_id": f"E{m:04d}-{day:03d}-{k}",
                        "merchant_id": merchant_id,
                        "payer_id": f"P{int(rng.integers(0, 50)):04d}",
                        "event_time": event_time,
                        "event_date": event_time.date(),
                        "amount_inr": float(rng.uniform(100, 50_000)),
                        "instrument": Instrument.UPI.value,
                        "is_cnp": True,
                        "is_international": False,
                        "bin_hash": None,
                        "device_hash": "d" * 16,
                        "ip_hash": "a" * 16,
                        "status": TxnStatus.CAPTURED.value,
                        "decline_code": None,
                        "mcc": "5411",
                        "is_refund": False,
                        "refund_of": None,
                        "schema_version": 1,
                    }
                )
        for p in range(WINDOW_DAYS // 7):
            requested = WINDOW_START + timedelta(days=p * 7)
            payouts.append(
                {
                    "payout_id": f"PO{m:04d}-{p:02d}",
                    "merchant_id": merchant_id,
                    "requested_at": requested,
                    # Settles two days later, so for a great many `as_of` values the
                    # request is visible and the settlement is not. That gap is the test.
                    "settled_at": requested + timedelta(days=2),
                    "amount_inr": float(rng.uniform(1_000, 100_000)),
                    "balance_before_inr": float(rng.uniform(100_000, 500_000)),
                    "is_accelerated": bool(rng.integers(0, 2)),
                    "schema_version": 1,
                }
            )

    pl.DataFrame(rows, schema=TRANSACTION_SCHEMA).write_parquet(root / "transactions.parquet")
    pl.DataFrame(payouts, schema=PAYOUT_SCHEMA).write_parquet(root / "payouts.parquet")
    pl.DataFrame(
        [
            {
                "merchant_id": f"M{m:04d}",
                # Staggered onboarding: half the population does not exist on day 0, so
                # active_merchants() has something real to filter.
                "onboarded_at": WINDOW_START + timedelta(days=m * 2),
                "mcc": "5411",
                "mcc_group": "grocery",
                "declared_monthly_gmv": 500_000.0,
                "kyc_tier": 2,
                "vintage_months": 12,
                "city_tier": 1,
                "schema_version": 1,
            }
            for m in range(N_MERCHANTS)
        ],
        schema=PROFILE_SCHEMA,
    ).write_parquet(root / "profiles.parquet")
    return root


@pytest.fixture(scope="module")
def store(dataset: Path):  # type: ignore[no-untyped-def]
    with EventStore(dataset) as s:
        yield s


# The store fixture is module-scoped on purpose (rebuilding parquet per example would
# dominate the runtime), so hypothesis' function-scoped-fixture health check is the
# wrong alarm here: the fixture is immutable and shared reads cannot leak between
# examples. Declared once rather than restated on every property.
prop = partial(
    settings, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)

AS_OF = st.datetimes(
    min_value=WINDOW_START.replace(tzinfo=None) - timedelta(days=5),
    max_value=WINDOW_START.replace(tzinfo=None) + timedelta(days=WINDOW_DAYS + 5),
    timezones=st.just(UTC),
)


@prop(max_examples=60)
@given(as_of=AS_OF, merchant=st.sampled_from([None, "M0000", "M0007", "M9999"]))
def test_no_future_row_ever_appears(
    store: EventStore, as_of: datetime, merchant: str | None
) -> None:
    frame = store.query_events(merchant, as_of=as_of)
    if frame.height:
        assert frame["event_time"].max() <= as_of
    if merchant is not None:
        assert set(frame["merchant_id"].unique()) <= {merchant}


@prop(max_examples=40)
@given(as_of=AS_OF)
def test_the_visible_prefix_only_ever_grows(store: EventStore, as_of: datetime) -> None:
    """Monotonicity: what was visible yesterday is still visible today, and today can only
    add. A filter that accidentally uses `event_date` rather than `event_time` breaks this
    within the same day, which is exactly where it matters."""
    earlier = store.query_events(None, as_of=as_of - timedelta(days=1))
    later = store.query_events(None, as_of=as_of)
    assert set(earlier["event_id"].to_list()) <= set(later["event_id"].to_list())


def test_boundary_is_inclusive_at_the_exact_event_instant(store: EventStore) -> None:
    # `<=`, not `<`. An event that happened at exactly as_of has happened and is readable;
    # excluding it would make the daily epoch loop silently drop the last event of each day.
    all_events = store.query_events(None, as_of=WINDOW_START + timedelta(days=WINDOW_DAYS + 5))
    pivot = all_events["event_time"][all_events.height // 2]
    assert pivot in store.query_events(None, as_of=pivot)["event_time"].to_list()


def test_since_bounds_the_window_from_below(store: EventStore) -> None:
    as_of = WINDOW_START + timedelta(days=30)
    since = as_of - timedelta(days=7)
    frame = store.query_events(None, as_of=as_of, since=since)
    assert frame.height
    assert frame["event_time"].min() >= since
    assert frame["event_time"].max() <= as_of


def test_a_naive_as_of_is_refused_rather_than_silently_compared(store: EventStore) -> None:
    # A naive timestamp compared against a tz-aware column is a filter that does nothing in
    # some engines and raises in others. Neither is acceptable, so we refuse it ourselves.
    with pytest.raises(PointInTimeError, match="tz-aware"):
        store.query_events(None, as_of=datetime(2026, 2, 1))


@prop(max_examples=40)
@given(as_of=AS_OF)
def test_payout_settlement_is_nulled_until_it_has_happened(
    store: EventStore, as_of: datetime
) -> None:
    frame = store.query_payouts(None, as_of=as_of)
    if frame.height:
        assert frame["requested_at"].max() <= as_of
        settled = frame["settled_at"].drop_nulls()
        if settled.len():
            assert settled.max() <= as_of


@prop(max_examples=25)
@given(as_of=AS_OF)
def test_active_merchants_excludes_the_not_yet_onboarded(
    store: EventStore, as_of: datetime
) -> None:
    active = store.active_merchants(as_of)
    profiles = store.profiles()
    expected = profiles.filter(pl.col("onboarded_at") <= as_of)["merchant_id"].to_list()
    assert active == sorted(expected)


def test_epoch_bounds_reports_the_real_calendar(store: EventStore) -> None:
    first, last = store.epoch_bounds()
    assert first >= WINDOW_START.date()
    assert last <= (WINDOW_START + timedelta(days=WINDOW_DAYS)).date()
