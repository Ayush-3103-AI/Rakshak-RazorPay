"""The parity harness: replay a stream through both runners and assert they agree.

Every feature in the register is tested by this, and the assertion is ``<= 1e-9`` at every
epoch for every merchant — not at the final epoch, and not on average. A feature that is
right at the end and wrong in the middle is a feature that would have been wrong on every
day it actually ran.

Lane B's tickets (T-120, T-122) parametrise ``assert_parity`` over the whole registry.
Nothing in here is feature-specific, on purpose.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, time, timedelta

import polars as pl
import pytest

from rakshak.features.spec import PARITY_TOLERANCE, FeatureSpec
from rakshak.features.state import MerchantState
from rakshak.schemas import (
    TRANSACTION_SCHEMA,
    Instrument,
    MerchantProfile,
    Transaction,
    TxnStatus,
)

__all__ = ["ParityFailure", "assert_parity", "epochs_between", "synthetic_stream"]


class ParityFailure(AssertionError):
    """Online and offline disagree. Read the message before assuming the online side."""


def end_of_day(day: date) -> datetime:
    """The instant an epoch is evaluated: the last nanosecond of the day, UTC.

    Defined once, because online and offline reading the epoch boundary differently is the
    single most common cause of a spurious parity failure — and of a real one.
    """
    return datetime.combine(day, time.max, tzinfo=UTC)


def epochs_between(first: date, last: date) -> list[date]:
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


def synthetic_stream(
    rng: object,
    *,
    merchants: int = 6,
    days: int = 21,
    start: date = date(2026, 1, 1),
    max_per_day: int = 5,
) -> tuple[list[Transaction], dict[str, MerchantProfile]]:
    """A small deterministic stream with the awkward shapes a feature must survive.

    Deliberately included, because each one has broken a real feature: merchants with
    empty days (a running mean that only updates on events goes stale), a merchant with
    no events at all after onboarding (division by a zero count), refunds interleaved with
    captures (sign handling), and failed transactions (which count for velocity but not
    for GMV).
    """
    import numpy as np

    assert isinstance(rng, np.random.Generator)
    txns: list[Transaction] = []
    profiles: dict[str, MerchantProfile] = {}

    for m in range(merchants):
        merchant_id = f"M{m:03d}"
        onboarded = datetime.combine(start, time.min, tzinfo=UTC)
        profiles[merchant_id] = MerchantProfile(
            merchant_id=merchant_id,
            onboarded_at=onboarded,
            mcc="5411",
            mcc_group="grocery",
            declared_monthly_gmv=float(rng.uniform(100_000, 2_000_000)),
            kyc_tier=int(rng.integers(1, 4)),
            vintage_months=int(rng.integers(0, 60)),
            city_tier=int(rng.integers(1, 4)),
        )
        # Merchant 0 transacts every day; the last merchant never transacts at all. The
        # rest are sparse. That spread is the test, not the volume.
        for d in range(days):
            if m == merchants - 1:
                continue
            n = max_per_day if m == 0 else int(rng.integers(0, max_per_day))
            for k in range(n):
                when = (
                    datetime.combine(start + timedelta(days=d), time.min, tzinfo=UTC)
                    + timedelta(seconds=int(rng.integers(0, 86_400)))
                )
                failed = bool(rng.random() < 0.08)
                refund = bool(rng.random() < 0.10) and not failed
                txns.append(
                    Transaction(
                        event_id=f"{merchant_id}-{d:03d}-{k}",
                        merchant_id=merchant_id,
                        payer_id=f"P{int(rng.integers(0, 25)):03d}",
                        event_time=when,
                        event_date=when.date(),
                        amount_inr=float(rng.lognormal(7.0, 1.2)),
                        instrument=Instrument.UPI,
                        is_cnp=True,
                        is_international=False,
                        bin_hash=None,
                        device_hash="d" * 16,
                        ip_hash="a" * 16,
                        status=TxnStatus.FAILED if failed else TxnStatus.CAPTURED,
                        decline_code="insufficient_funds" if failed else None,
                        mcc="5411",
                        is_refund=refund,
                        refund_of=f"{merchant_id}-{d:03d}-0" if refund else None,
                    )
                )

    # Time order across the whole population, which is the order the online runner sees.
    txns.sort(key=lambda t: (t.event_time, t.event_id))
    return txns, profiles


def to_frame(txns: Iterable[Transaction]) -> pl.DataFrame:
    rows = [
        {
            "event_id": t.event_id,
            "merchant_id": t.merchant_id,
            "payer_id": t.payer_id,
            "event_time": t.event_time,
            "event_date": t.event_date,
            "amount_inr": t.amount_inr,
            "instrument": t.instrument.value,
            "is_cnp": t.is_cnp,
            "is_international": t.is_international,
            "bin_hash": t.bin_hash,
            "device_hash": t.device_hash,
            "ip_hash": t.ip_hash,
            "status": t.status.value,
            "decline_code": t.decline_code,
            "mcc": t.mcc,
            "is_refund": t.is_refund,
            "refund_of": t.refund_of,
            "schema_version": t.schema_version,
        }
        for t in txns
    ]
    return pl.DataFrame(rows, schema=TRANSACTION_SCHEMA)


def assert_parity(
    spec: FeatureSpec,
    txns: Sequence[Transaction],
    profiles: dict[str, MerchantProfile],
    *,
    tolerance: float = PARITY_TOLERANCE,
) -> None:
    """Replay ``txns`` through ``spec`` both ways and assert agreement at every epoch.

    The online side folds events in time order and is read at each end-of-day. The offline
    side is recomputed from the whole prefix at that same instant. They must agree to
    ``tolerance`` for every merchant on every day.
    """
    frame = to_frame(txns)
    days = epochs_between(
        min(t.event_date for t in txns), max(t.event_date for t in txns)
    )
    states = {mid: MerchantState(merchant_id=mid, profile=p) for mid, p in profiles.items()}

    cursor = 0
    ordered = sorted(txns, key=lambda t: (t.event_time, t.event_id))

    for day in days:
        as_of = end_of_day(day)

        # Online: fold in everything that happened up to and including this instant.
        while cursor < len(ordered) and ordered[cursor].event_time <= as_of:
            event = ordered[cursor]
            spec.update(spec.state_of(states[event.merchant_id]), event)
            cursor += 1
        online = {
            mid: spec.value(spec.state_of(state), as_of) for mid, state in states.items()
        }

        # Offline: recompute from the prefix. The filter is applied here so that a `batch`
        # implementation which ignores `as_of` still cannot see the future — and so that a
        # failure means the arithmetic differs, not that the window did.
        prefix = frame.filter(pl.col("event_time") <= as_of)
        offline_frame = spec.batch(prefix.lazy(), as_of)
        offline = dict(
            zip(
                offline_frame["merchant_id"].to_list(),
                offline_frame[spec.name].to_list(),
                strict=True,
            )
        )

        for mid in sorted(states):
            # A merchant absent from the offline frame has no events yet; the online
            # runner still owes a value for it, and 0.0 is the contract for "nothing seen".
            want = offline.get(mid, 0.0)
            got = online[mid]
            if want is None:
                want = 0.0
            if abs(got - want) > tolerance:
                raise ParityFailure(
                    f"{spec.name}: merchant {mid} disagrees on {day}. "
                    f"online={got!r} offline={want!r} diff={abs(got - want):.3e} "
                    f"(tolerance {tolerance:.1e}).\n"
                    f"Check `batch` first — it is the runner holding the whole history, so "
                    f"it is the one that can see the future without meaning to."
                )


@pytest.fixture
def stream(rng: object) -> tuple[list[Transaction], dict[str, MerchantProfile]]:
    return synthetic_stream(rng)
