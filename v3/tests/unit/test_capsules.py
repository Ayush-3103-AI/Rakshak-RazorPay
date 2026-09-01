"""T-0119: the (payer, day) capsule accessor is point-in-time, single-path and reproducible.

The point-in-time property is property-tested over random `as_of` for the same reason
`test_point_in_time.py` is: an aggregate is a fine place to hide a future row, because the
row stops being visible as a row the moment it is summed into a count.

The property test alone is not enough, because it only ever looks at the output. A
capsule that *consulted* day 40 to compute day 20 emits no row dated after `as_of` and
passes it. `test_a_capsule_is_identical_whether_or_not_the_future_is_on_disk` is the test
that closes that gap: same `as_of`, two stores, one of which physically does not contain
the future, and the bytes must match. Its negative control sits directly beneath it.

The single-path property is asserted at runtime rather than by review. "It calls
`query_events` underneath" is exactly the kind of claim that stays true until somebody
optimises a scan, so the test breaks `duckdb.connect` for the duration of the call and
counts the `query_events` invocations.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rakshak.features.capsules import (
    CAPSULE_COLUMNS,
    CAPSULE_VECTOR_COLUMNS,
    capsules_as_of,
)
from rakshak.schemas import CAPSULE_SCHEMA, TRANSACTION_SCHEMA, Instrument, TxnStatus
from rakshak.store import EventStore, PointInTimeError

WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_DAYS = 45
N_MERCHANTS = 6
SEED = 42

SMOKE = Path(__file__).resolve().parents[2] / "data" / "smoke"

_INSTRUMENTS = [i.value for i in Instrument]
_STATUSES = [s.value for s in TxnStatus]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small deterministic stream with the structure the capsule vector reads.

    Built in tmp rather than read from `data/`, because `data/` is gitignored and a unit
    test that only passes on a machine that has already run `make gen` is not a test the
    clean-clone CI job can run. `test_smoke_dataset_*` below covers the real generator
    output when it happens to be present.

    Payers are drawn from a small pool so that repeat and first-time payers both occur,
    and devices from a smaller one still so that device sharing across payers is real
    rather than an artefact of every row carrying the same constant hash.
    """
    rng = np.random.default_rng(SEED)
    root = tmp_path_factory.mktemp("capsule_data")
    rows: list[dict[str, object]] = []
    for m in range(N_MERCHANTS):
        for day in range(WINDOW_DAYS):
            for k in range(int(rng.integers(0, 6))):
                event_time = (
                    WINDOW_START
                    + timedelta(days=day)
                    + timedelta(seconds=int(rng.integers(0, 86_400)))
                )
                rows.append(
                    {
                        "event_id": f"E{m:04d}-{day:03d}-{k}",
                        "merchant_id": f"M{m:04d}",
                        "payer_id": f"P{int(rng.integers(0, 20)):04d}",
                        "event_time": event_time,
                        "event_date": event_time.date(),
                        "amount_inr": float(rng.uniform(50, 90_000)),
                        "instrument": _INSTRUMENTS[int(rng.integers(0, len(_INSTRUMENTS)))],
                        "is_cnp": True,
                        "is_international": False,
                        "bin_hash": None,
                        "device_hash": f"d{int(rng.integers(0, 8)):015d}",
                        "ip_hash": "a" * 16,
                        "status": _STATUSES[int(rng.integers(0, len(_STATUSES)))],
                        "decline_code": None,
                        "mcc": "5411",
                        # ~10% refunds: the capsule excludes them, and a test that never
                        # sees one cannot notice if it stops.
                        "is_refund": bool(rng.random() < 0.1),
                        "refund_of": None,
                        "schema_version": 1,
                    }
                )
    pl.DataFrame(rows, schema=TRANSACTION_SCHEMA).write_parquet(root / "transactions.parquet")
    return root


@pytest.fixture(scope="module")
def store(dataset: Path):  # type: ignore[no-untyped-def]
    with EventStore(dataset) as s:
        yield s


prop = partial(
    settings, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)

AS_OF = st.datetimes(
    min_value=WINDOW_START.replace(tzinfo=None) - timedelta(days=3),
    max_value=WINDOW_START.replace(tzinfo=None) + timedelta(days=WINDOW_DAYS + 3),
    timezones=st.just(UTC),
)


def _digest(frame: pl.DataFrame) -> str:
    """The same content hash `GeneratedData.sha256` uses for gate G3: schema, then row
    hashes in order. Schema first because it pins dtype and column order, which `equals`
    alone would not; row hashes in order because row order is part of the contract here.

    Serialised bytes were the first attempt and were wrong. polars stores strings as
    Arrow view arrays, and two frames with identical values can carry different view
    buffer layouts, so `write_ipc` disagrees with itself on data that is equal by every
    definition that matters. That is why the generator hashes frames, not files, and the
    note in `engine.sha256` says so.
    """
    digest = hashlib.sha256()
    digest.update(str(frame.schema).encode())
    digest.update(frame.hash_rows().to_numpy().tobytes())
    return digest.hexdigest()


# -- AC 1: no capsule may contain a row from the future -----------------------


@prop(max_examples=60)
@given(as_of=AS_OF, merchant=st.sampled_from([None, "M0000", "M0003", "M9999"]))
def test_no_capsule_carries_an_event_after_as_of(
    store: EventStore, as_of: datetime, merchant: str | None
) -> None:
    frame = capsules_as_of(store, merchant, as_of)
    assert frame.columns == list(CAPSULE_COLUMNS)
    assert dict(frame.schema) == CAPSULE_SCHEMA
    if frame.height:
        # last_event_time is the witness: the aggregate would otherwise hide a future row
        # inside a count, where no assertion could reach it.
        assert frame["last_event_time"].max() <= as_of
        assert frame["event_date"].max() <= as_of.date()
    if merchant is not None:
        assert set(frame["merchant_id"].unique()) <= {merchant}


@prop(max_examples=40)
@given(as_of=AS_OF)
def test_capsule_counts_only_ever_grow(store: EventStore, as_of: datetime) -> None:
    """The same monotonicity `test_point_in_time.py` asserts on the raw prefix, one level
    up. A capsule that shrinks as `as_of` advances means the aggregation is reading
    something other than the visible prefix."""
    key = ["merchant_id", "event_date", "payer_id"]
    earlier = capsules_as_of(store, None, as_of - timedelta(days=1)).select(*key, "txn_count")
    later = capsules_as_of(store, None, as_of).select(*key, "txn_count")
    joined = earlier.join(later, on=key, how="left", suffix="_later")
    assert joined.height == earlier.height
    # A left join keeps the unmatched row with a null, and `Series.all()` ignores nulls,
    # so the disappearance has to be checked before the comparison rather than through it.
    assert joined["txn_count_later"].null_count() == 0, "a capsule visible earlier vanished"
    if joined.height:
        assert (joined["txn_count_later"] >= joined["txn_count"]).all()


@prop(max_examples=30)
@given(as_of=AS_OF)
def test_the_capsule_bag_reconstructs_the_visible_prefix(
    store: EventStore, as_of: datetime
) -> None:
    """Counts and value sum back to the non-refund rows the store returned. This is what
    makes the accessor a reshape rather than a computation with its own opinions."""
    events = store.query_events(None, as_of=as_of).filter(~pl.col("is_refund"))
    frame = capsules_as_of(store, None, as_of)
    assert frame["txn_count"].sum() == float(events.height)
    if events.height:
        assert frame["value_inr"].sum() == pytest.approx(events["amount_inr"].sum())
        assert frame.height == events.select("merchant_id", "event_date", "payer_id").n_unique()


def _truncated(source: Path, dest: Path, cutoff: datetime) -> int:
    """A second dataset holding only what happened at or before `cutoff`. Returns its size."""
    dest.mkdir(parents=True, exist_ok=True)
    visible = pl.read_parquet(source / "transactions.parquet").filter(
        pl.col("event_time") <= cutoff
    )
    visible.write_parquet(dest / "transactions.parquet")
    return visible.height


@pytest.mark.parametrize("cutoff_day", [7, 20, 33])
def test_a_capsule_is_identical_whether_or_not_the_future_is_on_disk(
    dataset: Path, tmp_path: Path, cutoff_day: int
) -> None:
    """Truncation test. The one that matters.

    `test_no_capsule_carries_an_event_after_as_of` proves no future *row* survives into the
    output. It cannot prove that no future row was *consulted* — an aggregate that read day
    40 to decide a number it wrote onto day 20 emits nothing dated after `as_of` and passes
    that test cleanly. So this asks the stronger question directly: run the same `as_of`
    against a store that physically cannot see the future, and demand the same bytes.

    Leakage here would not surface as a failure anywhere above. It would surface as an
    unearned good number, which is the failure mode with no alarm attached to it.
    """
    as_of = WINDOW_START + timedelta(days=cutoff_day)
    root = tmp_path / f"cut{cutoff_day}"
    kept = _truncated(dataset, root, as_of)
    total = pl.read_parquet(dataset / "transactions.parquet").height
    assert 0 < kept < total, "the cutoff removed nothing; this test would prove nothing"

    with EventStore(dataset) as s:
        with_the_future_on_disk = capsules_as_of(s, None, as_of)
    with EventStore(root) as s:
        without_it = capsules_as_of(s, None, as_of)

    assert _digest(with_the_future_on_disk) == _digest(without_it)
    assert with_the_future_on_disk.equals(without_it)


def test_the_truncation_test_is_not_vacuous(dataset: Path) -> None:
    """Negative control for the test above.

    Bit-identity is only evidence if the capsule vector is capable of moving when the
    prefix moves. It is: `device_shared_payers` counts distinct payers per device over the
    whole visible prefix, so a *past* day's capsule takes a different value once a later
    day is admitted. That is the leakage channel this accessor has, and the truncation test
    is what closes it.

    `payer_is_new` deliberately does not appear here. It is `event_date == min(event_date)`
    over the prefix, and extending a prefix forward cannot lower a minimum, so it is
    invariant to future data by construction rather than by test.
    """
    as_of = WINDOW_START + timedelta(days=20)
    key = ["merchant_id", "event_date", "payer_id"]
    with EventStore(dataset) as s:
        at_cutoff = capsules_as_of(s, None, as_of)
        # The same capsules, recomputed by a reader that was allowed to see everything.
        leaky = capsules_as_of(s, None, WINDOW_START + timedelta(days=WINDOW_DAYS)).filter(
            pl.col("event_date") <= as_of.date()
        )

    joined = at_cutoff.join(leaky, on=key, suffix="_leaky")
    assert joined.height == at_cutoff.height
    moved = [c for c in CAPSULE_VECTOR_COLUMNS if not joined[c].equals(joined[f"{c}_leaky"])]
    assert "device_shared_payers" in moved, (
        "no capsule column responds to the size of the prefix, so the truncation test "
        "above is a tautology and proves nothing"
    )
    assert _digest(at_cutoff) != _digest(leaky)


# -- AC 2: one read path, and it is EventStore.query_events -------------------


def test_the_accessor_opens_no_second_connection_and_goes_through_query_events(
    store: EventStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asserted, not reviewed. Both halves matter: the call count proves the accessor uses
    the store's gate, and the poisoned `duckdb.connect` proves it did not quietly open its
    own alongside it."""
    calls: list[tuple[object, ...]] = []
    real = EventStore.query_events

    def counting(self: EventStore, *args: object, **kwargs: object) -> pl.DataFrame:
        calls.append(args)
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    def poisoned(*args: object, **kwargs: object) -> object:
        raise AssertionError("capsules_as_of opened a second duckdb connection")

    monkeypatch.setattr(EventStore, "query_events", counting)
    monkeypatch.setattr(duckdb, "connect", poisoned)

    frame = capsules_as_of(store, "M0000", WINDOW_START + timedelta(days=30))

    assert len(calls) == 1, f"expected exactly one query_events call, got {len(calls)}"
    assert frame.height


def test_the_module_names_no_reader_of_its_own() -> None:
    """The runtime check above catches a second connection on the path it exercises. This
    catches one on a path it does not - a lazy import, a branch behind a flag, a raw
    `read_parquet` bolted on later."""
    module_path = Path(capsules_as_of.__globals__["__file__"])
    code = [
        line
        for line in module_path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    for forbidden in ("import duckdb", "read_parquet", "scan_parquet", "pl.read_", "connect("):
        offenders = [line for line in code if forbidden in line]
        assert not offenders, f"capsules.py reaches for {forbidden!r}: {offenders}"


# -- AC 3: stable and reproducible --------------------------------------------


def test_same_as_of_gives_a_byte_identical_frame(dataset: Path) -> None:
    """Two runs, two stores, two connections, one digest. Fresh stores on purpose: a
    result that is only stable within a warm connection is not reproducible, it is cached.
    """
    as_of = WINDOW_START + timedelta(days=33, hours=7, minutes=41)
    digests = set()
    for _ in range(2):
        with EventStore(dataset) as s:
            digests.add(_digest(capsules_as_of(s, None, as_of)))
    assert len(digests) == 1, "capsule output is not byte-stable across runs"


def test_the_vector_is_fixed_width_and_within_range(store: EventStore) -> None:
    as_of = WINDOW_START + timedelta(days=WINDOW_DAYS)
    frame = capsules_as_of(store, None, as_of)
    assert frame.height
    assert len(CAPSULE_VECTOR_COLUMNS) == 13
    assert all(frame.schema[c] == pl.Float64 for c in CAPSULE_VECTOR_COLUMNS)
    assert frame.select(CAPSULE_VECTOR_COLUMNS).null_count().to_numpy().sum() == 0

    shares = [f"i_{i.value}_share" for i in Instrument]
    mix = frame.select(shares).to_numpy().sum(axis=1)
    np.testing.assert_allclose(mix, 1.0, atol=1e-12)
    for column in ("failure_rate", "payer_is_new", *shares):
        assert frame[column].min() >= 0.0
        assert frame[column].max() <= 1.0
    assert frame["txn_count"].min() >= 1.0
    assert frame["ticket_cv"].min() >= 0.0
    assert frame["device_shared_payers"].min() >= 1.0
    # The fixture has 20 payers on 8 devices, so sharing must actually be observed -
    # otherwise this whole column is a constant 1.0 and proves nothing.
    assert frame["device_shared_payers"].max() > 1.0


def test_a_payer_is_new_exactly_once_per_merchant(store: EventStore) -> None:
    """`payer_is_new` is a property of the visible prefix, so within one `as_of` a payer
    can be new to a merchant on exactly one day - their first."""
    frame = capsules_as_of(store, None, WINDOW_START + timedelta(days=WINDOW_DAYS))
    per_payer = frame.group_by("merchant_id", "payer_id").agg(
        pl.col("payer_is_new").sum().alias("firsts"),
        pl.col("event_date").min().alias("first_day"),
    )
    assert (per_payer["firsts"] == 1.0).all()
    flagged = frame.filter(pl.col("payer_is_new") == 1.0)
    assert flagged.sort("merchant_id", "payer_id")["event_date"].to_list() == (
        per_payer.sort("merchant_id", "payer_id")["first_day"].to_list()
    )


def test_refunds_do_not_enter_a_capsule(store: EventStore) -> None:
    as_of = WINDOW_START + timedelta(days=WINDOW_DAYS)
    events = store.query_events(None, as_of=as_of)
    assert events["is_refund"].sum() > 0, "fixture has no refunds; the test proves nothing"
    frame = capsules_as_of(store, None, as_of)
    assert frame["txn_count"].sum() == float(events.filter(~pl.col("is_refund")).height)


def test_an_as_of_before_the_stream_returns_the_empty_contract(store: EventStore) -> None:
    frame = capsules_as_of(store, None, WINDOW_START - timedelta(days=1))
    assert frame.height == 0
    assert dict(frame.schema) == CAPSULE_SCHEMA


def test_a_naive_as_of_is_refused_by_the_store_underneath(store: EventStore) -> None:
    with pytest.raises(PointInTimeError, match="tz-aware"):
        capsules_as_of(store, None, datetime(2026, 2, 1))


def test_a_hand_computed_capsule(tmp_path: Path) -> None:
    """The invariant tests above pin structure; this one pins arithmetic.

    Six rows, one merchant, one day. P0 pays twice on device D0 (one captured, one
    failed); P1 pays once on that same D0, which is what makes D0 a shared device; P2 pays
    once on D1 of their own. A fourth row for P0 on the previous day is what makes P0 not
    new, and a refund on the day itself is what must not be counted.
    """
    day = datetime(2026, 1, 2, tzinfo=UTC)
    spec = [
        # (payer, device, day_offset, amount, status, is_refund)
        ("P0", "D0", -1, 100.0, TxnStatus.CAPTURED, False),
        ("P0", "D0", 0, 300.0, TxnStatus.CAPTURED, False),
        ("P0", "D0", 0, 500.0, TxnStatus.FAILED, False),
        ("P0", "D0", 0, 9_999.0, TxnStatus.CAPTURED, True),
        ("P1", "D0", 0, 200.0, TxnStatus.CAPTURED, False),
        ("P2", "D1", 0, 700.0, TxnStatus.CAPTURED, False),
    ]
    rows = []
    for i, (payer, device, offset, amount, status, refund) in enumerate(spec):
        when = day + timedelta(days=offset, minutes=i)
        rows.append(
            {
                "event_id": f"E{i}",
                "merchant_id": "M0",
                "payer_id": payer,
                "event_time": when,
                "event_date": when.date(),
                "amount_inr": amount,
                "instrument": Instrument.UPI.value,
                "is_cnp": True,
                "is_international": False,
                "bin_hash": None,
                "device_hash": device,
                "ip_hash": "a" * 16,
                "status": status.value,
                "decline_code": None,
                "mcc": "5411",
                "is_refund": refund,
                "refund_of": None,
                "schema_version": 1,
            }
        )
    pl.DataFrame(rows, schema=TRANSACTION_SCHEMA).write_parquet(tmp_path / "transactions.parquet")

    with EventStore(tmp_path) as s:
        frame = capsules_as_of(s, "M0", day + timedelta(days=1))
    today = frame.filter(pl.col("event_date") == day.date()).sort("payer_id")

    assert today["payer_id"].to_list() == ["P0", "P1", "P2"]
    assert today["txn_count"].to_list() == [2.0, 1.0, 1.0]
    # 9,999 is the refund and it is absent from every one of these.
    assert today["value_inr"].to_list() == [800.0, 200.0, 700.0]
    # P0: mean 400, population sd 100 -> cv 0.25. Singletons have no dispersion.
    assert today["ticket_cv"].to_list() == pytest.approx([0.25, 0.0, 0.0])
    assert today["failure_rate"].to_list() == [0.5, 0.0, 0.0]
    assert today["i_upi_share"].to_list() == [1.0, 1.0, 1.0]
    assert today["i_card_credit_share"].to_list() == [0.0, 0.0, 0.0]
    # P0 transacted yesterday, inside the visible prefix, so P0 is not new today.
    assert today["payer_is_new"].to_list() == [0.0, 1.0, 1.0]
    # D0 carries P0 and P1; D1 carries only P2.
    assert today["device_shared_payers"].to_list() == [2.0, 2.0, 1.0]

    # And the same call one day earlier: P0 is new, because the prefix holds nothing older.
    with EventStore(tmp_path) as s:
        yesterday = capsules_as_of(s, "M0", day - timedelta(hours=1))
    assert yesterday["payer_is_new"].to_list() == [1.0]
    assert yesterday["device_shared_payers"].to_list() == [1.0]


# -- AC 4 lives in tests/gates/test_g4_no_leakage.py, which AST-scans features/ --
# Not restated here: a second, weaker copy of the quarantine scan is how the real one
# gets deleted. G4's scanner rglobs src/rakshak/features/, so it picks capsules.py up
# with no change to the gate.


@pytest.mark.skipif(
    not (SMOKE / "transactions.parquet").exists(),
    reason="data/smoke is gitignored; generate it locally to exercise real output",
)
def test_smoke_dataset_capsules_are_point_in_time_and_stable() -> None:
    """The same invariants against real generator output, when it is on disk.

    Skipped rather than required, because `data/` is gitignored and the clean-clone CI job
    has no dataset. The synthetic fixture above is the one that always runs.
    """
    as_of = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
    with EventStore(SMOKE) as s:
        merchant = s.active_merchants(as_of)[0]
        frame = capsules_as_of(s, merchant, as_of)
        events = s.query_events(merchant, as_of=as_of).filter(~pl.col("is_refund"))
        assert frame.height
        assert frame["last_event_time"].max() <= as_of
        assert frame["txn_count"].sum() == float(events.height)
        assert _digest(frame) == _digest(capsules_as_of(s, merchant, as_of))
