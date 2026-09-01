"""T-0119: instance-level (payer, day) capsules, read through the existing store.

The T1 register emits one vector per merchant-day. Rungs 5, 7 and 8 need what that
aggregate throws away — *which* payers made up the day, and what each of them looked like.
This module reshapes the same rows into that bag, and it does nothing else.

**It is not a second read path.** Every row here comes out of
``EventStore.query_events``, which is where the ``event_time <= as_of`` predicate, the
``SET TimeZone='UTC'`` pin and the single ``_as_contract_dtypes()`` cast live. The two
timezone defects in T-101 were both a second reader that had its own connection and its
own idea of what a timestamp was; opening a duckdb connection in this file would be the
third. ``tests/unit/test_capsules.py`` asserts that, at runtime, rather than trusting a
reviewer to notice.

Three definitional choices, because they are the ones a reader will want to argue with:

**Refunds are excluded.** A capsule describes what a payer *attempted*; a refund is the
merchant reversing one, and it carries a positive ``amount_inr``, so counting it would
inflate both the value and the ticket dispersion of the payer it is attributed to. This
matches the ``~is_refund`` gate the T1 volume features already apply.

**"New to the merchant" and "device reuse" are evaluated against the visible prefix, not
the day.** A payer is new if the merchant's history *as known at ``as_of``* contains no
earlier day for them, and a device's payer count is over that same prefix. Both therefore
grow monotonically with ``as_of`` and neither can see a row the store would not return —
which is what makes them point-in-time safe rather than merely point-in-time shaped.

**Instrument mix is the full share vector over ``Instrument``**, not a one-number
concentration summary. The enum is closed, so the vector is fixed-width either way, and
a payer who moved from UPI to international card is the case the collapsed form loses.

**Prime Directive 4, stated rather than glossed.** Eleven of the thirteen vector columns
are functions of one (merchant, day, payer) group alone — count, value, dispersion,
failure rate, instrument mix — and are therefore bounded and incremental by construction.
Two are not. ``payer_is_new`` needs a per-merchant set of payers already seen, and
``device_shared_payers`` needs a per-device set of payers; both grow with the number of
distinct payers and devices, not with a constant. They are *insert-only* — an online
maintainer would fold each event in once and never re-scan — but they are not bounded, and
``g_payer_hhi``/``g_device_reuse_rate`` were cut from the T1/T2 register in T-122 for
exactly that reason.

So: capsules are not register features, carry no ``MerchantState``, and are not measured
against NFR-04's 4 KB. This module reads the whole visible prefix on every call, and that
is a real backward scan, not an optimisation gap. If Rung 5 is to be servable, these two
columns need a bounded sketch (a per-merchant HLL for payer novelty, a bounded LRU of hot
devices for reuse) and the sketch changes the numbers — so it is a decision to take
deliberately with the rung, not to smuggle in here. Recorded rather than hidden.

Prime Directive 3: nothing here names a field in ``schemas.RADIOACTIVE_FIELDS``. The
accessor sees the transaction table and no other.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from rakshak.schemas import CAPSULE_SCHEMA, Instrument, TxnStatus
from rakshak.store import EventStore

__all__ = [
    "CAPSULE_COLUMNS",
    "CAPSULE_VECTOR_COLUMNS",
    "capsule_aggregation",
    "capsules_as_of",
]

#: The contract, in order. ``CAPSULE_SCHEMA`` is the single definition; these are views of
#: it so a rung can slice keys from vector without re-listing either.
CAPSULE_COLUMNS: tuple[str, ...] = tuple(CAPSULE_SCHEMA)
CAPSULE_VECTOR_COLUMNS: tuple[str, ...] = CAPSULE_COLUMNS[4:]


def capsules_as_of(
    store: EventStore, merchant_id: str | None, as_of: datetime
) -> pl.DataFrame:
    """One row per (merchant, day, payer) visible at ``as_of``, ordered by that key.

    ``merchant_id=None`` returns every merchant's capsules, for the same reason
    ``query_events`` allows it: the daily sweep wants one scan, not ten thousand.

    The result is a pure function of the rows the store returns, and the sort key is
    unique across them, so two calls with the same ``as_of`` against the same dataset are
    byte-identical. There is no RNG in this module and none is threaded in, because
    nothing here is stochastic.
    """
    attempts = store.query_events(merchant_id, as_of=as_of).filter(~pl.col("is_refund"))
    if attempts.is_empty():
        return pl.DataFrame(schema=CAPSULE_SCHEMA)
    return capsule_aggregation(attempts.lazy()).collect()


def capsule_aggregation(attempts: pl.LazyFrame) -> pl.LazyFrame:
    """The capsule reshape itself, over an already-filtered ``attempts`` frame.

    Split out of :func:`capsules_as_of` (T-0120's harness lane) and **not** re-implemented
    beside it, because a column definition written down twice is a column definition that
    drifts (09-interfaces §9). ``capsules_as_of`` is now the store-backed caller of this
    function and nothing else changed about it.

    The reason the split exists: ``capsules_as_of`` gets its rows from
    ``EventStore.query_events``, which materialises the whole visible prefix through duckdb
    in one go. That is right for one merchant and impossible for twenty thousand — at the
    v3 geometry the prefix at the validation boundary is ~50M rows x 18 columns, and the
    box this runs on has under 2 GB free. Rung 5 therefore needs the same expressions
    applied to a *lazily scanned* frame it can stream, and this is that seam. The
    point-in-time predicate stays where it always was: the caller filters, this function
    aggregates.

    ``attempts`` must already have ``is_refund`` rows removed and be bounded above by an
    ``as_of``. This function asserts neither, because it cannot see an ``as_of`` — which is
    exactly why it is private-by-convention to the two callers above and below it.
    """
    # Reuse, within the visible prefix. Nulls are dropped rather than grouped: a null
    # device_hash is "unknown", and grouping the unknowns together would report every
    # payer with a missing device as sharing one enormous device with all the others.
    device_payers = (
        attempts.drop_nulls("device_hash")
        .group_by("merchant_id", "device_hash")
        .agg(pl.col("payer_id").n_unique().alias("_device_payers"))
    )
    first_day = attempts.group_by("merchant_id", "payer_id").agg(
        pl.col("event_date").min().alias("_first_day")
    )

    return (
        attempts.join(device_payers, on=["merchant_id", "device_hash"], how="left")
        .group_by("merchant_id", "event_date", "payer_id")
        .agg(
            pl.col("event_time").max().alias("last_event_time"),
            pl.len().cast(pl.Float64).alias("txn_count"),
            pl.col("amount_inr").sum().alias("value_inr"),
            # Population sigma, not sample: a one-transaction day has zero observed
            # dispersion, and ddof=1 would call it null and push the null into the model.
            pl.col("amount_inr").std(ddof=0).alias("_amount_sd"),
            (pl.col("status") == TxnStatus.FAILED.value).mean().alias("failure_rate"),
            *(
                (pl.col("instrument") == instrument.value)
                .mean()
                .alias(f"i_{instrument.value}_share")
                for instrument in Instrument
            ),
            pl.col("_device_payers").max().alias("device_shared_payers"),
        )
        .join(first_day, on=["merchant_id", "payer_id"], how="left")
        .with_columns(
            # cv = sd / mean, and mean = value / count. Guarded because the schema permits
            # a zero-value day even though the generator does not emit one.
            ticket_cv=pl.when(pl.col("value_inr") > 0.0)
            .then(pl.col("_amount_sd") * pl.col("txn_count") / pl.col("value_inr"))
            .otherwise(0.0),
            payer_is_new=(pl.col("event_date") == pl.col("_first_day")).cast(pl.Float64),
            # No shared-device evidence reads as "seen on one device", not as missing.
            device_shared_payers=pl.col("device_shared_payers").fill_null(1),
        )
        .select(CAPSULE_COLUMNS)
        .cast(CAPSULE_SCHEMA)  # type: ignore[arg-type]
        .sort("merchant_id", "event_date", "payer_id")
    )
