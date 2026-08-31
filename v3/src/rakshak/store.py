"""The event store: parquet on disk, duckdb in front of it, point-in-time by construction.

Every read of the event stream goes through this module, and every read takes an ``as_of``.
There is no method here that returns "all events" — the signature makes the point-in-time
filter mandatory rather than remembered, because the alternative is a filter that is
written correctly in eleven call sites and forgotten in the twelfth.

**Labels are deliberately absent from this module.** ``Label`` and ``GroundTruth`` are read
only through ``eval.splits.available_labels(as_of)``, which applies the
``label_available_at <= as_of`` gate and is enforced as the sole path by an AST scan
(T-130). Adding a label reader here would create a second door into the label table, and
the second door is the one nobody guards.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType

import duckdb
import polars as pl

from rakshak.schemas import TIMESTAMP

__all__ = ["EventStore", "PointInTimeError"]


class PointInTimeError(RuntimeError):
    """Raised when a query would return, or was asked to return, an event from the future.

    This is not a defensive nicety. A single future-dated row inside a feature window is
    label leakage wearing a costume, and it inflates every metric downstream without ever
    failing a test that was not written to catch it.
    """


def _as_contract_dtypes(frame: pl.DataFrame) -> pl.DataFrame:
    """Force every timestamp column back to the contract dtype: UTC, nanosecond.

    duckdb's TIMESTAMPTZ is microsecond-precision, so a round trip through it silently
    downgrades the nanosecond contract in 09-interfaces.md. Nothing in this project needs
    sub-microsecond resolution today — but a column whose dtype disagrees with the schema
    is the kind of mismatch that surfaces as an unrelated join failure four modules away,
    so it is repaired here, at the single point where duckdb hands data back.
    """
    casts = [
        pl.col(name).cast(TIMESTAMP)
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime) and dtype != TIMESTAMP
    ]
    return frame.with_columns(casts) if casts else frame


def _require_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise PointInTimeError(
            f"{name} must be tz-aware UTC — a naive timestamp compared against a tz-aware "
            f"event_time is a silent no-op filter in some engines; got {value!r}"
        )
    return value.astimezone(UTC)


class EventStore:
    """Point-in-time reader over the generated parquet tables.

    Usage::

        with EventStore(Path("data/v2")) as store:
            frame = store.query_events("M00042", as_of=datetime(2026, 3, 1, tzinfo=UTC))

    The store is read-only. The generator writes parquet directly; nothing mutates a table
    through this class, so two lanes can read the same dataset concurrently without a lock.
    """

    def __init__(self, root: Path | str, *, connection: duckdb.DuckDBPyConnection | None = None):
        self.root = Path(root)
        self.transactions_path = self.root / "transactions.parquet"
        self.profiles_path = self.root / "profiles.parquet"
        self.payouts_path = self.root / "payouts.parquet"
        # An in-memory connection over on-disk parquet: duckdb reads the files directly and
        # pushes the as_of predicate into the scan, so we never materialise a full history
        # in order to throw most of it away.
        self._con = connection or duckdb.connect(":memory:")
        self._owns_connection = connection is None
        # duckdb renders TIMESTAMPTZ in the *session* timezone, which defaults to the
        # machine's. On a laptop in Asia/Calcutta that silently hands back
        # Datetime("us", "Asia/Calcutta") and every downstream comparison against a UTC
        # literal either raises or, worse, quietly compares wall-clock times 5.5 hours
        # apart. Pin it, so the store behaves identically on a dev laptop and in CI.
        self._con.execute("SET TimeZone='UTC'")

    # ── lifecycle ────────────────────────────────────────────────────────────

    def __enter__(self) -> EventStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_connection:
            self._con.close()

    # ── the one point-in-time gate ───────────────────────────────────────────

    def query_events(
        self,
        merchant_id: str | None,
        as_of: datetime,
        *,
        since: datetime | None = None,
    ) -> pl.DataFrame:
        """Events for one merchant (or all merchants, if ``merchant_id`` is None) strictly
        at or before ``as_of``, in event-time order.

        ``since`` bounds the window from below for the rolling-window features, which do
        not need a merchant's whole history to compute a 7-day count.

        The ``<=`` is the entire contract of this class. It is written once, here.
        """
        as_of = _require_utc(as_of, "as_of")
        predicates = ["event_time <= ?"]
        params: list[object] = [as_of]
        if merchant_id is not None:
            predicates.append("merchant_id = ?")
            params.append(merchant_id)
        if since is not None:
            predicates.append("event_time >= ?")
            params.append(_require_utc(since, "since"))

        sql = (
            f"SELECT * FROM read_parquet(?) WHERE {' AND '.join(predicates)} "
            "ORDER BY merchant_id, event_time, event_id"
        )
        frame = _as_contract_dtypes(
            self._con.execute(sql, [str(self.transactions_path), *params]).pl()
        )

        # Belt and braces, and worth the microseconds: the predicate above is correct, but
        # this assertion is what a reviewer reads to believe it, and it is what catches a
        # future refactor that "optimises" the filter into a partition prune on event_date.
        # event_date is a *date*, so pruning on it alone admits same-day future rows.
        if frame.height and frame["event_time"].max() > as_of:  # type: ignore[operator]
            raise PointInTimeError(
                f"query_events returned an event at {frame['event_time'].max()!r}, after "
                f"as_of={as_of!r}. The point-in-time filter is broken; every metric "
                f"computed from this frame is invalid."
            )
        return frame

    def query_payouts(self, merchant_id: str | None, as_of: datetime) -> pl.DataFrame:
        """Payouts requested at or before ``as_of``.

        ``settled_at`` is nulled for rows that had not settled by ``as_of``: the row exists
        because the request happened, but its settlement is still in the future and reading
        it would be the same leak in a different column.
        """
        as_of = _require_utc(as_of, "as_of")
        predicates = ["requested_at <= ?"]
        params: list[object] = [as_of]
        if merchant_id is not None:
            predicates.append("merchant_id = ?")
            params.append(merchant_id)

        sql = (
            "SELECT payout_id, merchant_id, requested_at, "
            "  CASE WHEN settled_at <= ? THEN settled_at ELSE NULL END AS settled_at, "
            "  amount_inr, balance_before_inr, is_accelerated, schema_version "
            f"FROM read_parquet(?) WHERE {' AND '.join(predicates)} "
            "ORDER BY merchant_id, requested_at, payout_id"
        )
        frame = _as_contract_dtypes(
            self._con.execute(sql, [as_of, str(self.payouts_path), *params]).pl()
        )

        if frame.height:
            if frame["requested_at"].max() > as_of:  # type: ignore[operator]
                raise PointInTimeError(
                    f"query_payouts returned a request at {frame['requested_at'].max()!r}, "
                    f"after as_of={as_of!r}."
                )
            settled = frame["settled_at"].drop_nulls()
            if settled.len() and settled.max() > as_of:  # type: ignore[operator]
                raise PointInTimeError(
                    f"query_payouts returned a settlement at {settled.max()!r}, after "
                    f"as_of={as_of!r}."
                )
        return frame

    # ── onboarding facts ─────────────────────────────────────────────────────

    def profiles(self, merchant_ids: list[str] | None = None) -> pl.DataFrame:
        """Merchant profiles. These are onboarding-time facts, constant thereafter, so
        there is no ``as_of`` to apply — but a merchant onboarded *after* ``as_of`` should
        not be scored at all, which is the caller's filter to apply on ``onboarded_at``.
        """
        if merchant_ids is None:
            sql = "SELECT * FROM read_parquet(?) ORDER BY merchant_id"
            return _as_contract_dtypes(
                self._con.execute(sql, [str(self.profiles_path)]).pl()
            )
        sql = "SELECT * FROM read_parquet(?) WHERE merchant_id = ANY(?) ORDER BY merchant_id"
        return _as_contract_dtypes(
            self._con.execute(sql, [str(self.profiles_path), merchant_ids]).pl()
        )

    def active_merchants(self, as_of: datetime) -> list[str]:
        """Merchants already onboarded at ``as_of``. The population a rung must score."""
        as_of = _require_utc(as_of, "as_of")
        sql = "SELECT merchant_id FROM read_parquet(?) WHERE onboarded_at <= ? ORDER BY merchant_id"
        return self._con.execute(sql, [str(self.profiles_path), as_of]).pl()[
            "merchant_id"
        ].to_list()

    # ── convenience for the epoch loop ───────────────────────────────────────

    def epoch_bounds(self) -> tuple[date, date]:
        """First and last ``event_date`` in the stream, for driving the daily epoch loop
        without hardcoding the scenario's calendar in two places."""
        row = self._con.execute(
            "SELECT min(event_date), max(event_date) FROM read_parquet(?)",
            [str(self.transactions_path)],
        ).fetchone()
        if row is None or row[0] is None:
            raise PointInTimeError(f"no events in {self.transactions_path}")
        return row[0], row[1]
