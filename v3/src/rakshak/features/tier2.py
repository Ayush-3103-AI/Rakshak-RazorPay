"""Tier-2 features: histogram and divergence based, computed only for the top 10%.

Stage 1 of the cascade. These are allowed a small histogram or ring buffer where T1 is not,
and they answer a different question: T1 asks whether a *level* moved, T2 asks whether a
*shape* did. A merchant can hold its transaction count and its GMV exactly constant while
its ticket-size distribution, its instrument mix and its hour-of-day pattern all change, and
that merchant is invisible to every T1 feature in the register.

Every feature here is a divergence between the merchant's trailing-7-day histogram and its
own frozen baseline histogram — self-referential by construction, which is register rule 1
without having to be bolted on afterwards. All four share one state class, one day roll and
one pair of divergence functions, and **the scalar and vectorised forms of those functions
are literally the same numpy code applied to a 1-row and an M-row matrix**, which is why the
parity differences here are at machine epsilon rather than merely under 1e-9.

**This ticket is a partial delivery and the cut is deliberate.** Four of the register's
eight T2 rows are here. The other four are cut, with reasons, in `docs/logbook/T-122.md`:
`i_bin_hhi`, `g_payer_hhi` and `g_device_reuse_rate` for state budget — NFR-04's 4096 does
not stretch to them once T1's honest declarations are counted — and `d_refund_latency_med`
because it is **not online-computable in bounded state**, which is register rule 2 doing
exactly the job the register says it is there to do.
"""

from __future__ import annotations

import zlib
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import ClassVar

import numpy as np
import polars as pl

from rakshak.features.registry import register
from rakshak.features.spec import FeatureSpec
from rakshak.features.state import FeatureState, MerchantState
from rakshak.features.tier1 import (
    _CAPTURED,
    HIST_BINS,
    HIST_LOG_MAX,
    HIST_LOG_MIN,
    WARMUP_DAYS,
    _bin_expr,
    _bin_of,
    _is_captured,
)
from rakshak.schemas import Instrument, Tier, Transaction, TxnStatus

__all__ = ["DECLINE_BUCKETS", "T2_WINDOW_DAYS", "HistState", "HistogramSpec"]

#: All four features compare a trailing week against the baseline. 7 days is what the
#: register specifies for every T2 row, and it is also the shortest window in which a
#: mid-sized merchant has enough transactions for a 32-bin histogram to mean anything.
T2_WINDOW_DAYS = 7

#: Decline codes are free-form strings whose vocabulary lives in the generator's config and
#: is not knowable here. Hashing them into a fixed number of buckets is the bounded-state
#: answer and it is exactly reproducible on both runners. Collisions can only *lower* the
#: measured entropy, never raise it, so the feature stays conservative: it under-reports
#: decline spread rather than inventing it. Eight buckets caps the feature at 3 bits, which
#: is set by NFR-04 and not by the statistics — see the logbook.
DECLINE_BUCKETS = 8

#: Instrument mix is over the `Instrument` enum in declaration order, which `schemas.py`
#: pins as the contract.
INSTRUMENTS: tuple[Instrument, ...] = tuple(Instrument)
_INSTRUMENT_INDEX = {inst.value: i for i, inst in enumerate(INSTRUMENTS)}

#: Wasserstein is computed in log-rupee space, so the bin width is in log10 decades.
_LOG_BIN_WIDTH = (HIST_LOG_MAX - HIST_LOG_MIN) / HIST_BINS


# ─────────────────────────────────────────────────────────────────────────────
# Divergences. One implementation each, called with a 1-row matrix online and an
# M-row matrix offline, so the two runners cannot drift apart in the arithmetic —
# only in the counts they were handed, which is the thing parity is meant to test.
# ─────────────────────────────────────────────────────────────────────────────


def _normalise(counts: np.ndarray) -> np.ndarray:
    """Rows to probability vectors. An all-zero row stays all-zero rather than becoming
    uniform: "this merchant had no transactions" is not "this merchant used every
    instrument equally", and the divergence functions below treat the two differently."""
    total = counts.sum(axis=1, keepdims=True)
    return np.asarray(np.divide(counts, np.where(total > 0.0, total, 1.0)), dtype=np.float64)


def _kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Where a == 0 the term is 0 by the 0·log0 convention; where a > 0 the mixture b is
    # strictly positive by construction, so there is no division by zero to guard.
    safe_a = np.where(a > 0.0, a, 1.0)
    safe_b = np.where(b > 0.0, b, 1.0)
    terms = np.where(a > 0.0, a * np.log2(safe_a / safe_b), 0.0).sum(axis=1)
    return np.asarray(terms, dtype=np.float64)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """JSD in bits, in [0, 1]. Symmetric and bounded, unlike KL, which is why it is the
    divergence of choice for a feature a tree has to split on."""
    mid = 0.5 * (p + q)
    return 0.5 * _kl(p, mid) + 0.5 * _kl(q, mid)


def wasserstein_binned(p: np.ndarray, q: np.ndarray, width: float) -> np.ndarray:
    """1-Wasserstein between two binned distributions: the L1 gap between their CDFs.

    A single vectorised pass over the bins, which is the whole reason the register
    specifies a fixed histogram rather than a stored sample.
    """
    gap = np.abs(np.cumsum(p, axis=1) - np.cumsum(q, axis=1)).sum(axis=1) * width
    return np.asarray(gap, dtype=np.float64)


def shannon_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy in bits. 0 for a point mass, log2(k) for uniform over k buckets."""
    safe = np.where(p > 0.0, p, 1.0)
    bits = -np.where(p > 0.0, p * np.log2(safe), 0.0).sum(axis=1)
    return np.asarray(bits, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class HistState(FeatureState):
    """A frozen baseline histogram plus a trailing window of completed daily histograms.

    ``cur`` is today's incomplete histogram and is deliberately *not* in ``recent``, for the
    same reason ``DailyState`` keeps the current day out of its ring: every reader adds it
    back itself, which is what lets ``value()`` be correct on a day with no events without
    mutating anything.
    """

    anchor: date | None = None
    day: date | None = None
    cur: list[float] = field(default_factory=list)
    base: list[float] = field(default_factory=list)
    recent: list[tuple[date, list[float]]] = field(default_factory=list)


class HistogramSpec(FeatureSpec):
    """A bucketed histogram of the trailing week, scored against the frozen baseline.

    Subclasses declare ``bins`` and ``bucket``; everything else — the day roll, the warmup
    freeze, the window eviction and both runners' plumbing — is here once.
    """

    tier: ClassVar[Tier] = Tier.T2
    bins: ClassVar[int] = 1
    #: Whether the feature needs a baseline at all. `f_decline_entropy` does not: it is a
    #: property of the current window alone.
    needs_baseline: ClassVar[bool] = True

    name: ClassVar[str] = "__abstract_tier2__"
    family: ClassVar[str] = "F0"
    state_bytes: ClassVar[int] = 1
    human_template: ClassVar[str] = "{value}"

    def __init__(self, warmup_days: int = WARMUP_DAYS) -> None:
        self.warmup_days = warmup_days

    # ── subclass contract ────────────────────────────────────────────────────

    @abstractmethod
    def bucket(self, event: Transaction) -> int | None:
        """Which bucket this event falls in, or None if it does not count."""

    @abstractmethod
    def bucket_expr(self) -> pl.Expr:
        """``bucket`` as a polars expression: the bucket index, or null to skip the row."""

    @abstractmethod
    def score(self, window: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Rows of window counts and baseline counts to one value per row."""

    def comparable(self, window: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Rows that have enough observations for the divergence to mean anything.

        An empty trailing window is **not** zero divergence for free: JSD between an empty
        distribution and a concentrated baseline is 0.5, because the mixture halves the
        baseline's mass. Left alone, every dormant merchant would sit at a constant 0.5 —
        a mix-drift feature reporting dormancy, which `v_dormant_burst` already reports and
        reports better. Masked to 0.0 instead, which is the same "nothing seen" contract the
        rest of the layer uses, and applied identically on both runners so parity is
        untouched.
        """
        ok = window.sum(axis=1) > 0.0
        if self.needs_baseline:
            ok = ok & (base.sum(axis=1) > 0.0)
        return np.asarray(ok, dtype=np.bool_)

    # ── online ───────────────────────────────────────────────────────────────

    def init_state(self) -> FeatureState:
        return HistState(cur=[0.0] * self.bins, base=[0.0] * self.bins)

    def state_of(self, merchant: MerchantState) -> FeatureState:
        if self.name not in merchant.feature_states:
            merchant.feature_states[self.name] = self.init_state()
        return merchant.feature_states[self.name]

    def _in_warmup(self, state: HistState, day: date) -> bool:
        if state.anchor is None:
            return False
        return 0 <= (day - state.anchor).days < self.warmup_days

    def _roll(self, state: HistState, to_day: date) -> None:
        if state.anchor is None:
            state.anchor = to_day
        if state.day is not None:
            if self._in_warmup(state, state.day):
                for i, v in enumerate(state.cur):
                    state.base[i] += v
            state.recent.append((state.day, list(state.cur)))
            cutoff = to_day - timedelta(days=T2_WINDOW_DAYS - 1)
            state.recent = [row for row in state.recent if row[0] >= cutoff]
        state.day = to_day
        state.cur = [0.0] * self.bins

    def update(self, state: FeatureState, event: Transaction) -> None:
        assert isinstance(state, HistState)
        if state.day != event.event_date:
            self._roll(state, event.event_date)
        index = self.bucket(event)
        if index is not None:
            state.cur[index] += 1.0

    def value(self, state: FeatureState, as_of: datetime) -> float:
        assert isinstance(state, HistState)
        today = as_of.date()
        if state.anchor is None:
            return 0.0
        if self.needs_baseline and (today - state.anchor).days < self.warmup_days:
            return 0.0

        lo = today - timedelta(days=T2_WINDOW_DAYS - 1)
        window = np.zeros((1, self.bins), dtype=np.float64)
        for day, counts in state.recent:
            if lo <= day <= today:
                window[0] += counts
        base = np.array([state.base], dtype=np.float64)
        if state.day is not None:
            if lo <= state.day <= today:
                window[0] += state.cur
            if self._in_warmup(state, state.day):
                base[0] += state.cur
        if not self.comparable(window, base)[0]:
            return 0.0
        return float(self.score(window, base)[0])

    # ── offline ──────────────────────────────────────────────────────────────

    def _matrix(
        self, grouped: pl.DataFrame, index: dict[str, int], n: int
    ) -> np.ndarray:
        out = np.zeros((n, self.bins), dtype=np.float64)
        for merchant, bucket, count in grouped.iter_rows():
            out[index[merchant], int(bucket)] = count
        return out

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        today = as_of.date()
        # The anchor is the merchant's first *observed* day over every row, not only the
        # rows this feature buckets — because the online day roll fires on every event.
        anchors = (
            frame.group_by("merchant_id")
            .agg(pl.col("event_date").min().alias("anchor"))
            .collect()
            .sort("merchant_id")
        )
        if anchors.is_empty():
            return pl.DataFrame(
                {"merchant_id": [], self.name: []},
                schema={"merchant_id": pl.String(), self.name: pl.Float64()},
            )
        merchants = anchors["merchant_id"].to_list()
        index = {m: i for i, m in enumerate(merchants)}

        binned = (
            frame.with_columns(self.bucket_expr().alias("bucket"))
            .filter(pl.col("bucket").is_not_null())
            .join(anchors.lazy(), on="merchant_id", how="left")
            .with_columns(
                (pl.col("event_date") - pl.col("anchor")).dt.total_days().alias("age")
            )
            .collect()
        )

        def counts(rows: pl.DataFrame) -> np.ndarray:
            grouped = rows.group_by(["merchant_id", "bucket"]).agg(
                pl.len().cast(pl.Float64).alias("n")
            )
            return self._matrix(grouped, index, len(merchants))

        window = counts(
            binned.filter(
                (pl.col("event_date") <= today)
                & (pl.col("event_date") > today - timedelta(days=T2_WINDOW_DAYS))
            )
        )
        base = counts(
            binned.filter((pl.col("age") >= 0) & (pl.col("age") < self.warmup_days))
        )
        out = np.where(self.comparable(window, base), self.score(window, base), 0.0)

        if self.needs_baseline:
            elapsed = np.array(
                [
                    a is not None and (today - a).days >= self.warmup_days
                    for a in anchors["anchor"].to_list()
                ]
            )
            out = np.where(elapsed, out, 0.0)
        return pl.DataFrame({"merchant_id": merchants, self.name: out})


# ═════════════════════════════════════════════════════════════════════════════
# F2 — ticket-size distribution
# ═════════════════════════════════════════════════════════════════════════════


@register
class TicketWasserstein(HistogramSpec):
    """1-Wasserstein between the trailing week's ticket distribution and the baseline's.

    Computed in **log-rupee space**, on the same 32-bin log histogram `t_p95_median_ratio`
    uses. Log space is the right space twice over: ticket sizes are lognormal, so the bins
    are equal-mass rather than equal-width; and the distance is then in decades rather than
    rupees, which makes it comparable across a kirana and a jeweller. A Wasserstein in
    rupees would be a GMV feature wearing a distribution feature's name — the exact
    absolute-level mistake register rule 1 exists to prevent.
    """

    name = "t_wasserstein_7d"
    family = "F2"
    bins = HIST_BINS
    state_bytes = 1024
    human_template = "ticket-size distribution has moved {value:.2f} decades from baseline"
    has_cohort_residual = True

    def bucket(self, event: Transaction) -> int | None:
        return _bin_of(event.amount_inr) if _is_captured(event) else None

    def bucket_expr(self) -> pl.Expr:
        return pl.when(_CAPTURED).then(_bin_expr(pl.col("amount_inr"))).otherwise(None)

    def score(self, window: np.ndarray, base: np.ndarray) -> np.ndarray:
        return wasserstein_binned(_normalise(window), _normalise(base), _LOG_BIN_WIDTH)


# ═════════════════════════════════════════════════════════════════════════════
# F3 — payment-instrument mix
# ═════════════════════════════════════════════════════════════════════════════


@register
class InstrumentMixJsd(HistogramSpec):
    """Jensen-Shannon divergence of the trailing week's instrument mix against baseline.

    The register calls this the feature "most at risk of firing on confounder P3 (fee
    change) and P4 (new payment method launch)", and says that is intentional: the raw
    feature *should* fire platform-wide and the residual *should not*. It is therefore the
    second test case for the cohort layer after `f_auth_fail_rate_z`, and T-121's finding —
    that P2 is not common-mode because the generator scales it per merchant — should be
    re-measured here, where P3 is a genuine step applied to everyone.
    """

    name = "i_mix_jsd"
    family = "F3"
    bins = len(INSTRUMENTS)
    state_bytes = 232
    human_template = "payment-instrument mix has diverged {value:.3f} bits from baseline"
    has_cohort_residual = True

    def bucket(self, event: Transaction) -> int | None:
        return _INSTRUMENT_INDEX.get(event.instrument.value)

    def bucket_expr(self) -> pl.Expr:
        return pl.col("instrument").replace_strict(
            _INSTRUMENT_INDEX, default=None, return_dtype=pl.Int32
        )

    def score(self, window: np.ndarray, base: np.ndarray) -> np.ndarray:
        return jensen_shannon(_normalise(window), _normalise(base))


# ═════════════════════════════════════════════════════════════════════════════
# F5 — failure and retry signature
# ═════════════════════════════════════════════════════════════════════════════


@register
class DeclineEntropy(HistogramSpec):
    """Shannon entropy of the trailing week's decline-code mix.

    No baseline: this is a property of the current window alone, and a merchant whose
    declines are spread evenly across many reasons is testing many cards regardless of what
    it used to do. One reason code repeated is a broken integration; eight reason codes at
    once is a card-source sweep.

    Codes are hashed into `DECLINE_BUCKETS` buckets, which caps the feature at 3 bits.
    Collisions can only merge two codes into one, which lowers the measured entropy — so
    the feature under-reports decline spread and never invents it.
    """

    name = "f_decline_entropy"
    family = "F5"
    bins = DECLINE_BUCKETS
    needs_baseline = False
    state_bytes = 224
    human_template = "decline reasons are spread across {value:.2f} bits of entropy"
    has_cohort_residual = True

    def bucket(self, event: Transaction) -> int | None:
        if event.status is not TxnStatus.FAILED or event.decline_code is None:
            return None
        return int(zlib.crc32(event.decline_code.encode("utf-8")) % DECLINE_BUCKETS)

    def bucket_expr(self) -> pl.Expr:
        return (
            pl.when(
                (pl.col("status") == TxnStatus.FAILED.value)
                & pl.col("decline_code").is_not_null()
            )
            .then(
                # polars has no crc32; the codebook is small and bounded, so the mapping is
                # materialised from the codes present rather than computed in-expression.
                pl.col("decline_code").map_elements(
                    lambda s: int(zlib.crc32(s.encode("utf-8")) % DECLINE_BUCKETS),
                    return_dtype=pl.Int32,
                )
            )
            .otherwise(None)
        )

    def score(self, window: np.ndarray, base: np.ndarray) -> np.ndarray:
        return shannon_entropy(_normalise(window))


# ═════════════════════════════════════════════════════════════════════════════
# F7 — temporal pattern
# ═════════════════════════════════════════════════════════════════════════════


@register
class HourlyJsd(HistogramSpec):
    """JS divergence of the trailing week's hour-of-day histogram against baseline.

    An operator change or an automation switch moves *when* a merchant transacts without
    moving how much. A full week of buckets is used rather than a single day precisely
    because a day-of-week effect would otherwise dominate: comparing Tuesday against a
    baseline that averages Saturdays would fire on every merchant with a weekend business.
    """

    name = "h_hourly_jsd"
    family = "F7"
    bins = 24
    state_bytes = 768
    human_template = "time-of-day pattern has diverged {value:.3f} bits from baseline"
    has_cohort_residual = True

    def bucket(self, event: Transaction) -> int | None:
        return event.event_time.hour

    def bucket_expr(self) -> pl.Expr:
        return pl.col("event_time").dt.hour().cast(pl.Int32)

    def score(self, window: np.ndarray, base: np.ndarray) -> np.ndarray:
        return jensen_shannon(_normalise(window), _normalise(base))
