"""Tier-1 features: computed for every merchant, every epoch, in Stage 0 of the cascade.

07-feature-register.md is the prose version of this module. Every feature here is written
twice — once as an O(1) fold over a bounded ``DailyState`` and once as a polars expression
over the whole history — and ``tests/parity/test_tier1_parity.py`` asserts the two agree to
1e-9 at every epoch for every merchant.

Three design decisions run through the whole file and are worth reading before the code:

**1. The baseline window is a fixed number of calendar days, not a fixed number of
observations.** ``warmup_days`` calendar days from ``onboarded_at``, inclusive of days on
which nothing happened. That constant denominator is what makes the online form O(1): a day
with no events contributes 0 to the sum and 0 to the sum-of-squares, so the online runner
never has to "catch up" on empty days — it only has to know how many days the window holds,
and it knows that at compile time. It also matches the offline definition exactly, which a
count-of-active-days baseline would not (offline sees which days were empty; online, folding
only on events, does not).

**2. z-scores are 0.0 until the warmup window has elapsed.** Not "shrunk", not "computed
from a partial window" — 0.0. A z against three days of history is noise with a decimal
point on it, and the cohort layer (T-121) is where the cold-start problem is actually
solved, by shrinking toward the cohort prior. Doing something clever here would have meant
doing it twice.

**3. The baseline never moves after warmup**, which is ``BaselineStats``' documented
anti-R2 choice carried down to each feature's own accumulators. Every feature below keeps
its own Welford-equivalent pair rather than sharing ``MerchantState.baseline``, because the
features are z-scoring different quantities and a single shared accumulator could only hold
one of them.

Prime Directive 3: nothing in this module may name a field in ``schemas.RADIOACTIVE_FIELDS``
or import from ``rakshak.generator`` / ``rakshak.eval``. The dependency runs one way.
"""

from __future__ import annotations

import zlib
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import ClassVar

import numpy as np
import polars as pl

from rakshak.features.registry import register
from rakshak.features.spec import FeatureSpec
from rakshak.features.state import FeatureState, MerchantState
from rakshak.schemas import MerchantProfile, Tier, Transaction, TxnStatus

__all__ = [
    "MICRO_TICKET_INR",
    "PROFILES",
    "RETRY_RING",
    "RETRY_WINDOW_S",
    "ROUND_AMOUNTS",
    "WARMUP_DAYS",
    "Z_CLIP",
    "DailyState",
    "Tier1Spec",
    "load_profiles",
    "profiles_frame",
]

# ─────────────────────────────────────────────────────────────────────────────
# Constants.
#
# CLAUDE.md wants these in configs/*.yaml, and they belong there. configs/ is owned by
# another lane this sprint, so they are named module constants with the reasoning attached
# and T-152 should lift them into the scenario manifest. They are read through the
# constructor (`warmup_days`), not hard-wired into the arithmetic, so a config-driven value
# is a one-line change and not a rewrite.
# ─────────────────────────────────────────────────────────────────────────────

#: Calendar days from ``onboarded_at`` over which each feature's own baseline is built,
#: after which it freezes. 30 days is one full monthly business cycle, which is the
#: shortest window that does not read a merchant's weekly rhythm as drift.
WARMUP_DAYS = 30

#: A z-score is divided by max(std, this). A merchant whose warmup window was perfectly
#: flat would otherwise produce an infinite z on its first varied day — a false-positive
#: generator, not a detector. Matches ``BaselineStats.z``'s floor so the two agree.
Z_FLOOR = 1e-9

#: ...and the floor alone is not enough: 1e-9 turns a single rupee of movement into a z of
#: 1e9, which no tree split survives. Both runners clip identically, so parity is unaffected
#: and the model sees a bounded column.
Z_CLIP = 50.0

#: 07-feature-register.md F2: round-value transactions, the card-testing / laundering tell.
ROUND_AMOUNTS: tuple[float, ...] = (100.0, 500.0, 1000.0, 5000.0)

#: Tolerance for "is this amount round". Money is float64 whole rupees (09-interfaces.md),
#: so half a paisa is comfortably tighter than any real rounding and looser than float noise.
ROUND_TOL = 0.005

#: F2 ``t_micro_share``: card-testing probes are sub-₹10 by construction.
MICRO_TICKET_INR = 10.0

#: F5 ``f_retry_burst_rate``: "same payer, >=3 attempts, <=10 min". The ring holds the last
#: ``RETRY_RING`` events for the merchant; an attempt is bursty when at least two of them
#: are the same payer inside ``RETRY_WINDOW_S``. The ring is what makes it O(1) — and it is
#: also what makes it exactly reproducible offline, as ``RETRY_RING`` lagged comparisons.
RETRY_RING = 8
RETRY_WINDOW_S = 600.0

#: F2 ``t_p95_median_ratio``: a fixed log-spaced histogram, ₹1 to ₹10^7 in 32 bins. Bounded,
#: O(1) to update, and identical arithmetic on both runners — a stored sample or a P²
#: estimator would be neither bounded nor order-independent.
HIST_BINS = 32
HIST_LOG_MIN = 0.0  # log10(1)
HIST_LOG_MAX = 7.0  # log10(10_000_000)


# ─────────────────────────────────────────────────────────────────────────────
# The profile store.
#
# `FeatureSpec.batch(frame, as_of)` is handed a transaction frame and nothing else, so a
# feature whose definition involves an onboarding fact — `v_declared_ratio`'s denominator,
# every F9 static, and the warmup window's start date for every z — has no way to reach
# `MerchantProfile` offline. The online runner does: `state_of` receives the whole
# `MerchantState`, so each feature snapshots what it needs into its own state on first
# touch (see `Tier1Spec.state_of`).
#
# This module-level table closes the offline half. It is reference data known at
# onboarding and constant thereafter — the same table a real deployment keeps in memory —
# not mutable state, and not an RNG. It is loaded once per run.
#
# Reported to the lead as an interface gap: the honest fix is `batch(frame, as_of,
# profiles)` in the frozen `spec.py`, which is a Block-1 change and not Lane B's to make.
# ─────────────────────────────────────────────────────────────────────────────

PROFILES: dict[str, MerchantProfile] = {}


def load_profiles(profiles: Mapping[str, MerchantProfile]) -> None:
    """Install the merchant profile table the offline runners read. Idempotent."""
    PROFILES.clear()
    PROFILES.update(profiles)


def profiles_frame() -> pl.DataFrame:
    """``(merchant_id, onboarded_date, declared_monthly_gmv, ...)``, sorted."""
    if not PROFILES:
        raise RuntimeError(
            "rakshak.features.tier1.PROFILES is empty. The offline runner needs the "
            "merchant profile table for the warmup window start and for every F9 static; "
            "call load_profiles(...) once before batch(). The online runner does not need "
            "it — it snapshots the profile through MerchantState."
        )
    return pl.DataFrame(
        {
            "merchant_id": [p.merchant_id for p in PROFILES.values()],
            "onboarded_date": [p.onboarded_at.date() for p in PROFILES.values()],
            "declared_monthly_gmv": [p.declared_monthly_gmv for p in PROFILES.values()],
        },
        schema={
            "merchant_id": pl.String(),
            "onboarded_date": pl.Date(),
            "declared_monthly_gmv": pl.Float64(),
        },
    ).sort("merchant_id")


# ─────────────────────────────────────────────────────────────────────────────
# The shared online state.
#
# One state class for every daily/windowed feature rather than one per feature. A feature
# uses the slots it needs and leaves the rest at their defaults; the alternative is
# fourteen near-identical five-field dataclasses, which is the kind of thing that is right
# in the first three and copy-pasted wrong in the fourth.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DailyState(FeatureState):
    """Bounded per-feature state: today's partial aggregate, the frozen warmup sums, and a
    trailing-window ring of completed days.

    ``day``/``num``/``den``/``aux`` are the *current, incomplete* day. They are never in
    ``hist`` — ``hist`` holds completed days only — so every reader has to add the current
    day back in itself. That is deliberate: it is what lets ``value()`` be correct on a day
    with no events without mutating anything.
    """

    #: First day this merchant was *observed*, which anchors the warmup window. Not
    #: ``onboarded_at``: the event stream starts at simulation day 0 while merchants were
    #: onboarded up to 120 days earlier, so an onboarding-anchored window is empty for most
    #: of the population and every z divides by a zero baseline. The anchor is the first
    #: event's date, which both runners see identically and which never moves once set.
    anchor: date | None = None
    declared_gmv: float = 0.0

    day: date | None = None
    prev_day: date | None = None
    num: float = 0.0
    den: float = 0.0
    aux: float = 0.0

    #: Sums of the *daily value* over the warmup window. Denominator is warmup_days, a
    #: constant, so empty days need no bookkeeping.
    warm_sum: float = 0.0
    warm_sumsq: float = 0.0
    #: Raw component sums over the warmup window, for features whose baseline is a pooled
    #: ratio rather than a mean of daily ratios (h_weekend_share_z).
    warm_num: float = 0.0
    warm_den: float = 0.0

    last_time: datetime | None = None
    hist: list[tuple[date, float, float, float]] = field(default_factory=list)
    ring: list[tuple[str, datetime]] = field(default_factory=list)
    counts: list[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic shared by both runners.
#
# Written once and called from both sides, because parity to 1e-9 is a claim about the
# *same* arithmetic in a different order — not about two independently-derived formulas
# that ought to agree. Where the offline side needs the vectorised form, it is the numpy
# transliteration directly below the scalar one and the operations are in the same order.
# ─────────────────────────────────────────────────────────────────────────────


def _z(x: float, s: float, ss: float, n: int) -> float:
    mean = s / n
    var = ss / n - mean * mean
    std = var**0.5 if var > 0.0 else 0.0
    z = (x - mean) / max(std, Z_FLOOR)
    return float(min(max(z, -Z_CLIP), Z_CLIP))


def _z_vec(x: np.ndarray, s: np.ndarray, ss: np.ndarray, n: int) -> np.ndarray:
    mean = s / n
    var = ss / n - mean * mean
    std = np.where(var > 0.0, np.sqrt(np.where(var > 0.0, var, 0.0)), 0.0)
    z = (x - mean) / np.maximum(std, Z_FLOOR)
    return np.clip(z, -Z_CLIP, Z_CLIP)


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0.0 else 0.0


def _bin_of(amount: float) -> int:
    """Log-spaced bin index for a rupee amount. Shared with tier2's Wasserstein feature."""
    lg = np.log10(max(amount, 1.0))
    idx = int((lg - HIST_LOG_MIN) / (HIST_LOG_MAX - HIST_LOG_MIN) * HIST_BINS)
    return min(max(idx, 0), HIST_BINS - 1)


def _bin_expr(col: pl.Expr) -> pl.Expr:
    """The polars transliteration of ``_bin_of``. Same clamps, same truncation."""
    lg = pl.max_horizontal(col, pl.lit(1.0)).log10()
    idx = ((lg - HIST_LOG_MIN) / (HIST_LOG_MAX - HIST_LOG_MIN) * HIST_BINS).cast(pl.Int32)
    return idx.clip(0, HIST_BINS - 1)


#: Representative value of each histogram bin: its geometric midpoint. Quantiles read off a
#: binned CDF are exact to the bin, and the bin is what both runners share.
_BIN_MID: np.ndarray = 10.0 ** (
    HIST_LOG_MIN
    + (np.arange(HIST_BINS, dtype=np.float64) + 0.5) * (HIST_LOG_MAX - HIST_LOG_MIN) / HIST_BINS
)


def _hist_quantile(counts: np.ndarray, q: float) -> np.ndarray:
    """Quantile per row of a (merchants x bins) count matrix, from the binned CDF.

    Returns 0.0 for a row with no observations. Deterministic and identical on both runners
    because the only inputs are integer bin counts.
    """
    total = counts.sum(axis=1)
    cdf = np.cumsum(counts, axis=1)
    target = q * total
    idx = (cdf < target[:, None]).sum(axis=1)
    idx = np.minimum(idx, HIST_BINS - 1)
    return np.where(total > 0, _BIN_MID[idx], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# The base class
# ─────────────────────────────────────────────────────────────────────────────


class Tier1Spec(FeatureSpec):
    """A daily-aggregate feature over a bounded state.

    Subclasses declare what one event contributes (``contrib``), how a day's contributions
    become one number (``daily`` and its polars twin ``daily_expr``), and how the current
    state reads out (``read``). The day roll, the warmup accumulation and the trailing
    window are handled here, once.
    """

    tier: ClassVar[Tier] = Tier.T1
    #: Trailing window in days, 0 for features that only look at today. Bounds ``hist``.
    window: ClassVar[int] = 0

    # `FeatureSpec.__init_subclass__` means to skip intermediate abstract bases, but the
    # guard reads `cls.__abstractmethods__`, which ABCMeta has not set yet at the point
    # `__init_subclass__` runs — so it fires on every base too. Placeholders satisfy it
    # without changing behaviour: `name` is overridden by every concrete subclass, and an
    # abstract base is never passed to `register`. Reported to the lead; the fix belongs in
    # the frozen `spec.py`, as `if inspect.isabstract(cls): return` after class creation.
    name: ClassVar[str] = "__abstract_tier1__"
    family: ClassVar[str] = "F0"
    state_bytes: ClassVar[int] = 1
    human_template: ClassVar[str] = "{value}"

    def __init__(self, warmup_days: int = WARMUP_DAYS) -> None:
        # Configuration, not state: the registry constructs one shared instance per feature
        # and every merchant is scored through it. Tests construct a short-warmup instance
        # rather than fabricating a six-month stream to see a single z.
        self.warmup_days = warmup_days

    # ── what a subclass provides ─────────────────────────────────────────────

    @abstractmethod
    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        """``(num, den, aux)`` this event adds to today's running aggregate."""

    def daily(self, num: float, den: float, aux: float) -> float:
        """Today's single number. Default: the numerator."""
        return num

    def daily_expr(self) -> pl.Expr:
        """``daily`` as a polars expression over the columns ``num``, ``den``, ``aux``."""
        return pl.col("num")

    @abstractmethod
    def read(self, state: DailyState, today: date) -> float:
        """The feature's value, given a state and the epoch date. Must not mutate."""

    # ── the online runner ────────────────────────────────────────────────────

    def init_state(self) -> FeatureState:
        return DailyState()

    def state_of(self, merchant: MerchantState) -> FeatureState:
        """As the base class, plus the profile snapshot.

        ``update`` and ``value`` are handed a ``FeatureState`` and never the merchant, so
        the onboarding facts a feature needs have to be copied in at creation. This is the
        one hook where the whole ``MerchantState`` is in scope.
        """
        if self.name not in merchant.feature_states:
            fresh = self.init_state()
            assert isinstance(fresh, DailyState)
            fresh.declared_gmv = merchant.profile.declared_monthly_gmv
            merchant.feature_states[self.name] = fresh
        return merchant.feature_states[self.name]

    def _roll(self, state: DailyState, to_day: date) -> None:
        """Close the current day and open ``to_day``."""
        if state.anchor is None:
            state.anchor = to_day
        if state.day is not None:
            value = self.daily(state.num, state.den, state.aux)
            if self._in_warmup(state, state.day):
                state.warm_sum += value
                state.warm_sumsq += value * value
            if self.window:
                state.hist.append((state.day, state.num, state.den, state.aux))
                cutoff = to_day - timedelta(days=self.window - 1)
                state.hist = [row for row in state.hist if row[0] >= cutoff]
            state.prev_day = state.day
        state.day = to_day
        state.num = state.den = state.aux = 0.0

    def _in_warmup(self, state: DailyState, day: date) -> bool:
        if state.anchor is None:
            return False
        return 0 <= (day - state.anchor).days < self.warmup_days

    def warm_elapsed(self, state: DailyState, today: date) -> bool:
        if state.anchor is None:
            return False
        return (today - state.anchor).days >= self.warmup_days

    def update(self, state: FeatureState, event: Transaction) -> None:
        assert isinstance(state, DailyState)
        if state.day != event.event_date:
            self._roll(state, event.event_date)
        num, den, aux = self.contrib(event)
        state.num += num
        state.den += den
        state.aux += aux
        # Raw warmup pooling is a plain sum, so it needs no day-roll bookkeeping.
        if self._in_warmup(state, event.event_date):
            state.warm_num += num
            state.warm_den += den
        state.last_time = event.event_time

    def value(self, state: FeatureState, as_of: datetime) -> float:
        assert isinstance(state, DailyState)
        return self.read(state, as_of.date())

    # ── readers a subclass composes ──────────────────────────────────────────

    def today_value(self, state: DailyState, today: date) -> float:
        """Today's aggregate, or the empty-day value. Never yesterday's."""
        if state.day != today:
            return self.daily(0.0, 0.0, 0.0)
        return self.daily(state.num, state.den, state.aux)

    def warm(self, state: DailyState, today: date) -> tuple[float, float]:
        """``(sum, sumsq)`` of daily values over the warmup window.

        The current day is added back in here because ``_roll`` has not seen it yet: a
        merchant that went dormant partway through its warmup would otherwise have its last
        active day missing from its own baseline, forever.
        """
        s, ss = state.warm_sum, state.warm_sumsq
        if state.day is not None and self._in_warmup(state, state.day):
            v = self.daily(state.num, state.den, state.aux)
            s += v
            ss += v * v
        return s, ss

    def span(self, state: DailyState, lo: date, hi: date) -> tuple[float, float, float]:
        """Summed ``(num, den, aux)`` over completed days in ``[lo, hi]``, plus today."""
        n = d = a = 0.0
        for day, dn, dd, da in state.hist:
            if lo <= day <= hi:
                n += dn
                d += dd
                a += da
        if state.day is not None and lo <= state.day <= hi:
            n += state.num
            d += state.den
            a += state.aux
        return n, d, a

    def trailing(self, state: DailyState, today: date, days: int) -> tuple[float, float, float]:
        return self.span(state, today - timedelta(days=days - 1), today)

    # ── the offline runner ───────────────────────────────────────────────────

    def daily_frame(self, frame: pl.LazyFrame) -> pl.DataFrame:
        """``(merchant_id, event_date, num, den, aux)`` — the offline day roll."""
        num, den, aux = self.contrib_exprs()
        return (
            frame.group_by(["merchant_id", "event_date"])
            .agg(
                num.sum().alias("num"),
                den.sum().alias("den"),
                aux.sum().alias("aux"),
            )
            .collect()
        )

    @abstractmethod
    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """``contrib`` as three polars expressions over a transaction frame."""

    def anchored(self, daily: pl.DataFrame) -> pl.DataFrame:
        """Attach each merchant's first observed day — the offline twin of ``state.anchor``.

        Read from the frame, not from the profile table, precisely so that both runners
        answer "when did this merchant start being visible to us" the same way. It is a
        ``min`` over the prefix, so it is stable once the merchant's first event is inside
        the prefix and it can never see the future.
        """
        first = daily.group_by("merchant_id").agg(pl.col("event_date").min().alias("anchor"))
        return daily.join(first, on="merchant_id", how="left")

    def empty(self, name: str) -> pl.DataFrame:
        return pl.DataFrame(
            {"merchant_id": [], name: []},
            schema={"merchant_id": pl.String(), name: pl.Float64()},
        )

    def z_batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        """The offline z: warmup sums over the fixed calendar window, today's value, clip."""
        daily = self.daily_frame(frame)
        if daily.is_empty():
            return self.empty(self.name)
        daily = self.anchored(daily).with_columns(
            self.daily_expr().alias("v"),
            (pl.col("event_date") - pl.col("anchor")).dt.total_days().alias("age"),
        )
        today = as_of.date()
        merchants = daily.select("merchant_id").unique().sort("merchant_id")

        warm = (
            daily.filter((pl.col("age") >= 0) & (pl.col("age") < self.warmup_days))
            .group_by("merchant_id")
            .agg(pl.col("v").sum().alias("s"), (pl.col("v") * pl.col("v")).sum().alias("ss"))
        )
        cur = (
            daily.filter(pl.col("event_date") == today)
            .group_by("merchant_id")
            .agg(pl.col("v").sum().alias("x"))
        )
        onboard = daily.group_by("merchant_id").agg(pl.col("anchor").first())
        joined = (
            merchants.join(warm, on="merchant_id", how="left")
            .join(cur, on="merchant_id", how="left")
            .join(onboard, on="merchant_id", how="left")
            .with_columns(pl.col("s", "ss", "x").fill_null(0.0))
            .sort("merchant_id")
        )
        z = _z_vec(
            joined["x"].to_numpy(),
            joined["s"].to_numpy(),
            joined["ss"].to_numpy(),
            self.warmup_days,
        )
        elapsed = np.array(
            [
                d is not None and (today - d).days >= self.warmup_days
                for d in joined["anchor"].to_list()
            ]
        )
        return joined.select("merchant_id").with_columns(
            pl.Series(self.name, np.where(elapsed, z, 0.0))
        )

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        return self.z_batch(frame, as_of)

    def window_batch(
        self,
        frame: pl.LazyFrame,
        as_of: datetime,
        days: int,
        expr: pl.Expr,
    ) -> pl.DataFrame:
        """Offline trailing-window reduction: ``expr`` over summed ``num``/``den``/``aux``."""
        daily = self.daily_frame(frame)
        if daily.is_empty():
            return self.empty(self.name)
        today = as_of.date()
        merchants = daily.select("merchant_id").unique().sort("merchant_id")
        win = (
            daily.filter(
                (pl.col("event_date") <= today)
                & (pl.col("event_date") > today - timedelta(days=days))
            )
            .group_by("merchant_id")
            .agg(
                pl.col("num").sum().alias("num"),
                pl.col("den").sum().alias("den"),
                pl.col("aux").sum().alias("aux"),
            )
        )
        return (
            merchants.join(win, on="merchant_id", how="left")
            .fill_null(0.0)
            .with_columns(expr.alias(self.name))
            .select("merchant_id", self.name)
            .sort("merchant_id")
        )


# ─────────────────────────────────────────────────────────────────────────────
# Contribution expressions used by more than one feature
# ─────────────────────────────────────────────────────────────────────────────

# Per-row, not literals: `pl.lit(1.0).sum()` inside a group-by is 1.0, not the group size,
# because a scalar literal is not broadcast to the group's height. That produced a daily
# count of exactly 1.0 for every merchant on every day and it was the first parity failure
# of the ticket. Anchoring both sentinels to a non-null column makes them row-shaped.
_ZERO = pl.col("event_id").is_null().cast(pl.Float64)
_ONE = pl.col("event_id").is_not_null().cast(pl.Float64)
_CAPTURED = (pl.col("status") == TxnStatus.CAPTURED.value) & ~pl.col("is_refund")
_GMV = pl.when(_CAPTURED).then(pl.col("amount_inr")).otherwise(0.0)


def _is_captured(event: Transaction) -> bool:
    return event.status is TxnStatus.CAPTURED and not event.is_refund


# ═════════════════════════════════════════════════════════════════════════════
# F1 — Volume and velocity drift
# ═════════════════════════════════════════════════════════════════════════════


@register
class TxnCountZ(Tier1Spec):
    """Daily transaction count against the merchant's own frozen baseline."""

    name = "v_txn_count_z"
    family = "F1"
    state_bytes = 40
    human_template = "transaction count is {value:.1f}σ from this merchant's own norm"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return 1.0, 0.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return _ONE, _ZERO, _ZERO

    def read(self, state: DailyState, today: date) -> float:
        if not self.warm_elapsed(state, today):
            return 0.0
        s, ss = self.warm(state, today)
        return _z(self.today_value(state, today), s, ss, self.warmup_days)


@register
class GmvZ(TxnCountZ):
    """Daily captured GMV against the merchant's own frozen baseline.

    Failed and pending transactions count for velocity (``v_txn_count_z``) and not for
    value; refunds are their own family. Keeping the three separate is what lets the model
    tell "more attempts" apart from "more money".
    """

    name = "v_gmv_z"
    state_bytes = 40
    human_template = "daily GMV is {value:.1f}σ from this merchant's own norm"

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return (event.amount_inr if _is_captured(event) else 0.0), 0.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return _GMV, _ZERO, _ZERO


@register
class GmvAccel(GmvZ):
    """Second difference of 7-day GMV: A − 2B + C over three consecutive weeks.

    The discriminator against persona L3, the high-growth genuine merchant and the hardest
    negative in the population. Organic growth is linear-to-log and its second difference
    sits near zero; a bust-out ramp is convex and its second difference does not.

    Normalised by the merchant's own baseline weekly GMV so it is a ratio, not rupees —
    register rule 1, self-referential rather than absolute.
    """

    name = "v_gmv_accel"
    family = "F1"
    window = 21
    state_bytes = 208
    human_template = "GMV growth is accelerating at {value:.2f}× its own weekly baseline"
    has_cohort_residual = True

    def read(self, state: DailyState, today: date) -> float:
        if not self.warm_elapsed(state, today):
            return 0.0
        a = self.span(state, today - timedelta(days=6), today)[0]
        b = self.span(state, today - timedelta(days=13), today - timedelta(days=7))[0]
        c = self.span(state, today - timedelta(days=20), today - timedelta(days=14))[0]
        s, _ = self.warm(state, today)
        scale = 7.0 * (s / self.warmup_days)
        return (a - 2.0 * b + c) / max(scale, 1.0)

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        daily = self.daily_frame(frame)
        if daily.is_empty():
            return self.empty(self.name)
        daily = self.anchored(daily).with_columns(
            (pl.col("event_date") - pl.col("anchor")).dt.total_days().alias("age")
        )
        today = as_of.date()
        merchants = daily.select("merchant_id").unique().sort("merchant_id")

        def block(lo: int, hi: int, alias: str) -> pl.DataFrame:
            return (
                daily.filter(
                    (pl.col("event_date") >= today - timedelta(days=lo))
                    & (pl.col("event_date") <= today - timedelta(days=hi))
                )
                .group_by("merchant_id")
                .agg(pl.col("num").sum().alias(alias))
            )

        warm = (
            daily.filter((pl.col("age") >= 0) & (pl.col("age") < self.warmup_days))
            .group_by("merchant_id")
            .agg(pl.col("num").sum().alias("s"))
        )
        onboard = daily.group_by("merchant_id").agg(pl.col("anchor").first())
        joined = (
            merchants.join(block(6, 0, "a"), on="merchant_id", how="left")
            .join(block(13, 7, "b"), on="merchant_id", how="left")
            .join(block(20, 14, "c"), on="merchant_id", how="left")
            .join(warm, on="merchant_id", how="left")
            .join(onboard, on="merchant_id", how="left")
            .with_columns(pl.col("a", "b", "c", "s").fill_null(0.0))
            .sort("merchant_id")
        )
        scale = 7.0 * (joined["s"].to_numpy() / self.warmup_days)
        raw = (
            joined["a"].to_numpy() - 2.0 * joined["b"].to_numpy() + joined["c"].to_numpy()
        ) / np.maximum(scale, 1.0)
        elapsed = np.array(
            [
                d is not None and (today - d).days >= self.warmup_days
                for d in joined["anchor"].to_list()
            ]
        )
        return joined.select("merchant_id").with_columns(
            pl.Series(self.name, np.where(elapsed, raw, 0.0))
        )


@register
class DeclaredRatio(GmvZ):
    """Trailing-30d GMV ÷ the monthly GMV the merchant declared at onboarding.

    The promise-versus-reality gap. This feature exists *only* because the merchant was
    onboarded — a transaction scorer has never seen the declaration and cannot compute it.
    It is the clearest single argument for the post-onboarding surveillance position, and
    it should be named as such in the writeup.
    """

    name = "v_declared_ratio"
    family = "F1"
    window = 30
    state_bytes = 256
    human_template = "trailing-30d GMV is {value:.1f}× the declared monthly GMV"
    has_cohort_residual = True

    def read(self, state: DailyState, today: date) -> float:
        gmv = self.trailing(state, today, 30)[0]
        return gmv / state.declared_gmv if state.declared_gmv > 0.0 else 0.0

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        daily = self.daily_frame(frame)
        if daily.is_empty():
            return self.empty(self.name)
        today = as_of.date()
        merchants = daily.select("merchant_id").unique().sort("merchant_id")
        win = (
            daily.filter(
                (pl.col("event_date") <= today)
                & (pl.col("event_date") > today - timedelta(days=30))
            )
            .group_by("merchant_id")
            .agg(pl.col("num").sum().alias("gmv"))
        )
        prof = profiles_frame().select("merchant_id", "declared_monthly_gmv")
        return (
            merchants.join(win, on="merchant_id", how="left")
            .join(prof, on="merchant_id", how="left")
            .with_columns(pl.col("gmv").fill_null(0.0))
            .with_columns(
                pl.when(pl.col("declared_monthly_gmv") > 0.0)
                .then(pl.col("gmv") / pl.col("declared_monthly_gmv"))
                .otherwise(0.0)
                .alias(self.name)
            )
            .select("merchant_id", self.name)
            .sort("merchant_id")
        )


@register
class FanoTrailing(TxnCountZ):
    """Fano factor (variance ÷ mean) of daily counts over a trailing 28 days.

    v1 measured a Fano of 12.25 in real overdispersed counts against the Poisson assumption
    of 1.0, which is the measurement that killed v1's generator. A *change* in burstiness is
    a different signal from a change in level, and this is the feature that carries it.

    The denominator is 28 calendar days, not 28 active days: an empty day is a real zero
    and dropping it would make a dormant merchant look metronomic.
    """

    name = "v_fano_trailing"
    family = "F1"
    window = 28
    state_bytes = 240
    human_template = "daily-count burstiness (Fano) is {value:.2f}"
    has_cohort_residual = True

    def read(self, state: DailyState, today: date) -> float:
        lo = today - timedelta(days=27)
        total = 0.0
        sumsq = 0.0
        for day, dn, _dd, _da in state.hist:
            if lo <= day <= today:
                total += dn
                sumsq += dn * dn
        if state.day is not None and lo <= state.day <= today:
            total += state.num
            sumsq += state.num * state.num
        mean = total / 28.0
        var = sumsq / 28.0 - mean * mean
        return var / mean if mean > 0.0 else 0.0

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        daily = self.daily_frame(frame)
        if daily.is_empty():
            return self.empty(self.name)
        today = as_of.date()
        merchants = daily.select("merchant_id").unique().sort("merchant_id")
        win = (
            daily.filter(
                (pl.col("event_date") <= today)
                & (pl.col("event_date") > today - timedelta(days=28))
            )
            .group_by("merchant_id")
            .agg(
                pl.col("num").sum().alias("s"),
                (pl.col("num") * pl.col("num")).sum().alias("ss"),
            )
        )
        joined = merchants.join(win, on="merchant_id", how="left").fill_null(0.0)
        mean = joined["s"].to_numpy() / 28.0
        var = joined["ss"].to_numpy() / 28.0 - mean * mean
        return joined.select("merchant_id").with_columns(
            pl.Series(self.name, np.where(mean > 0.0, var / np.where(mean > 0.0, mean, 1.0), 0.0))
        )


@register
class DormantBurst(TxnCountZ):
    """Days dormant × today's count z-score.

    Dormancy is measured against the last active day *strictly before today*, so a merchant
    that wakes up after three weeks and fires scores its dormancy against the three weeks —
    if the gap were measured to the latest event it would collapse to zero on exactly the
    day the burst happens, which is the day that matters.
    """

    name = "v_dormant_burst"
    family = "F1"
    state_bytes = 48
    human_template = "burst after {value:.0f} dormant-day-sigmas of silence"
    has_cohort_residual = True

    def read(self, state: DailyState, today: date) -> float:
        if not self.warm_elapsed(state, today):
            return 0.0
        s, ss = self.warm(state, today)
        z = _z(self.today_value(state, today), s, ss, self.warmup_days)
        last: date | None = state.day if state.day is not None and state.day < today else None
        if last is None:
            last = state.prev_day
        gap = float((today - last).days) if last is not None else 0.0
        return gap * z

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        today = as_of.date()
        z = super().batch(frame, as_of)
        prior = (
            frame.filter(pl.col("event_date") < today)
            .group_by("merchant_id")
            .agg(pl.col("event_date").max().alias("last"))
            .collect()
        )
        joined = z.join(prior, on="merchant_id", how="left")
        gaps = np.array(
            [0.0 if d is None else float((today - d).days) for d in joined["last"].to_list()]
        )
        return joined.select("merchant_id").with_columns(
            pl.Series(self.name, gaps * joined[self.name].to_numpy())
        )


# ═════════════════════════════════════════════════════════════════════════════
# F2 — Ticket-size distribution drift (T1 rows; t_wasserstein_7d is T2)
# ═════════════════════════════════════════════════════════════════════════════


@register
class P95MedianRatio(Tier1Spec):
    """p95 ÷ median ticket, read off a 32-bin log histogram of the merchant's own history.

    The register specifies P² estimators. P² is order-dependent by construction, so its
    offline twin would have to be a Python replay of the same recurrence — which is not an
    offline runner, it is the online one in a costume, and it would make the parity test
    vacuous. A fixed log-spaced histogram is the same 200-ish bytes, is exactly
    reproducible from a group-by, and costs one bin of resolution. Logged as a deliberate
    deviation from the register.
    """

    name = "t_p95_median_ratio"
    family = "F2"
    state_bytes = 72
    human_template = "ticket p95/median is {value:.1f}, a fattening tail"
    has_cohort_residual = True

    def init_state(self) -> FeatureState:
        state = DailyState()
        state.counts = [0.0] * HIST_BINS
        return state

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return _ZERO, _ZERO, _ZERO

    def update(self, state: FeatureState, event: Transaction) -> None:
        assert isinstance(state, DailyState)
        if not _is_captured(event):
            return
        if not state.counts:
            state.counts = [0.0] * HIST_BINS
        state.counts[_bin_of(event.amount_inr)] += 1.0

    def read(self, state: DailyState, today: date) -> float:
        if not state.counts:
            return 0.0
        counts = np.asarray(state.counts, dtype=np.float64)[None, :]
        p95 = float(_hist_quantile(counts, 0.95)[0])
        med = float(_hist_quantile(counts, 0.50)[0])
        return p95 / med if med > 0.0 else 0.0

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        binned = (
            frame.filter(_CAPTURED)
            .with_columns(_bin_expr(pl.col("amount_inr")).alias("bin"))
            .group_by(["merchant_id", "bin"])
            .agg(pl.len().cast(pl.Float64).alias("n"))
            .collect()
        )
        merchants = sorted(
            frame.select("merchant_id").unique().collect()["merchant_id"].to_list()
        )
        if not merchants:
            return self.empty(self.name)
        index = {m: i for i, m in enumerate(merchants)}
        counts = np.zeros((len(merchants), HIST_BINS), dtype=np.float64)
        for m, b, n in binned.iter_rows():
            counts[index[m], int(b)] = n
        p95 = _hist_quantile(counts, 0.95)
        med = _hist_quantile(counts, 0.50)
        out = np.where(med > 0.0, p95 / np.where(med > 0.0, med, 1.0), 0.0)
        return pl.DataFrame({"merchant_id": merchants, self.name: out}).sort("merchant_id")


class DailyShare(Tier1Spec):
    """Share of today's transactions with some property. Base for four features.

    A share, not a z. The self-referential rule is honoured one layer up: the cohort
    residual in T-121 is what turns "8% of this merchant's tickets are round" into "8% more
    round than its cohort on the same day", and doing it twice would double-count.
    """

    family = "F2"
    state_bytes = 24

    def daily(self, num: float, den: float, aux: float) -> float:
        return _ratio(num, den)

    def daily_expr(self) -> pl.Expr:
        return pl.when(pl.col("den") > 0.0).then(pl.col("num") / pl.col("den")).otherwise(0.0)

    def read(self, state: DailyState, today: date) -> float:
        return self.today_value(state, today)

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        daily = self.daily_frame(frame)
        if daily.is_empty():
            return self.empty(self.name)
        merchants = daily.select("merchant_id").unique().sort("merchant_id")
        cur = daily.filter(pl.col("event_date") == as_of.date())
        return (
            merchants.join(cur, on="merchant_id", how="left")
            .with_columns(pl.col("num").fill_null(0.0), pl.col("den").fill_null(0.0))
            .with_columns(self.daily_expr().alias(self.name))
            .select("merchant_id", self.name)
            .sort("merchant_id")
        )


def _is_round(amount: float) -> bool:
    return any(abs(amount - r) < ROUND_TOL for r in ROUND_AMOUNTS)


@register
class RoundAmountShare(DailyShare):
    """Share of today's transactions at round values — card testing and laundering."""

    name = "t_round_amount_share"
    human_template = "{value:.0%} of today's tickets are round-value"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return float(_is_round(event.amount_inr)), 1.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        hit = pl.any_horizontal(
            [(pl.col("amount_inr") - r).abs() < ROUND_TOL for r in ROUND_AMOUNTS]
        )
        return hit.cast(pl.Float64), _ONE, _ZERO


@register
class MicroShare(DailyShare):
    """Share of today's transactions at or below ₹10 — card-testing probes."""

    name = "t_micro_share"
    human_template = "{value:.0%} of today's tickets are sub-₹10 probes"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return float(event.amount_inr <= MICRO_TICKET_INR), 1.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return (pl.col("amount_inr") <= MICRO_TICKET_INR).cast(pl.Float64), _ONE, _ZERO


@register
class NewMaxEvent(Tier1Spec):
    """1 if today's largest capture exceeds 3× everything that came before it.

    Binary and deliberately not cohort-residualised: a residual of an indicator against a
    cohort median that is almost always zero is the indicator back again, with noise.
    """

    name = "t_new_max_event"
    family = "F2"
    state_bytes = 32
    human_template = "today's largest ticket broke this merchant's historic maximum"
    has_cohort_residual = False

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return _ZERO, _ZERO, _ZERO

    def update(self, state: FeatureState, event: Transaction) -> None:
        assert isinstance(state, DailyState)
        if state.day != event.event_date:
            # aux carries the running historic max across the roll; num is today's max.
            if state.day is not None:
                state.aux = max(state.aux, state.num)
                state.prev_day = state.day
            state.day = event.event_date
            state.num = 0.0
        if _is_captured(event):
            state.num = max(state.num, event.amount_inr)

    def read(self, state: DailyState, today: date) -> float:
        if state.day == today:
            historic, current = state.aux, state.num
        else:
            historic, current = max(state.aux, state.num), 0.0
        return 1.0 if historic > 0.0 and current > 3.0 * historic else 0.0

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        today = as_of.date()
        captured = frame.filter(_CAPTURED)
        merchants = frame.select("merchant_id").unique().collect().sort("merchant_id")
        if merchants.is_empty():
            return self.empty(self.name)
        hist = (
            captured.filter(pl.col("event_date") < today)
            .group_by("merchant_id")
            .agg(pl.col("amount_inr").max().alias("historic"))
            .collect()
        )
        cur = (
            captured.filter(pl.col("event_date") == today)
            .group_by("merchant_id")
            .agg(pl.col("amount_inr").max().alias("current"))
            .collect()
        )
        joined = (
            merchants.join(hist, on="merchant_id", how="left")
            .join(cur, on="merchant_id", how="left")
            .fill_null(0.0)
            .sort("merchant_id")
        )
        fired = (joined["historic"].to_numpy() > 0.0) & (
            joined["current"].to_numpy() > 3.0 * joined["historic"].to_numpy()
        )
        return joined.select("merchant_id").with_columns(
            pl.Series(self.name, fired.astype(np.float64))
        )


# ═════════════════════════════════════════════════════════════════════════════
# F3 — Payment-instrument mix drift (T1 rows; the divergence rows are T2)
# ═════════════════════════════════════════════════════════════════════════════


@register
class IntlShare(DailyShare):
    """Share of today's transactions on international cards."""

    name = "i_intl_share"
    family = "F3"
    human_template = "{value:.0%} of today's transactions are international"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return float(event.is_international), 1.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return pl.col("is_international").cast(pl.Float64), _ONE, _ZERO


@register
class CnpShare(DailyShare):
    """Share of today's transactions that are card-not-present."""

    name = "i_cnp_share"
    family = "F3"
    human_template = "{value:.0%} of today's transactions are card-not-present"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return float(event.is_cnp), 1.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return pl.col("is_cnp").cast(pl.Float64), _ONE, _ZERO


# ═════════════════════════════════════════════════════════════════════════════
# F5 — Failure and retry signature
# ═════════════════════════════════════════════════════════════════════════════


@register
class AuthFailRateZ(Tier1Spec):
    """Daily auth-failure rate against the merchant's own frozen baseline.

    This is the feature confounder P2 (gateway outage) slams platform-wide, and therefore
    the cleanest single demonstration of the cohort residual: during P2 every merchant's raw
    z spikes together and every residual stays flat. Gate G5 is built on it first.
    """

    name = "f_auth_fail_rate_z"
    family = "F5"
    state_bytes = 48
    human_template = "auth failure rate is {value:.1f}σ from this merchant's own norm"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return float(event.status is TxnStatus.FAILED), 1.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return (pl.col("status") == TxnStatus.FAILED.value).cast(pl.Float64), _ONE, _ZERO

    def daily(self, num: float, den: float, aux: float) -> float:
        return _ratio(num, den)

    def daily_expr(self) -> pl.Expr:
        return pl.when(pl.col("den") > 0.0).then(pl.col("num") / pl.col("den")).otherwise(0.0)

    def read(self, state: DailyState, today: date) -> float:
        if not self.warm_elapsed(state, today):
            return 0.0
        s, ss = self.warm(state, today)
        return _z(self.today_value(state, today), s, ss, self.warmup_days)


@register
class RetryBurstRate(Tier1Spec):
    """Share of today's attempts that are the third-or-later by the same payer in 10 minutes.

    Bounded by a ring of the last ``RETRY_RING`` events. That bound is not only a memory
    decision: it is also what makes the feature reproducible offline, as exactly
    ``RETRY_RING`` lagged comparisons over the merchant's event order. An unbounded "any
    prior attempt in 10 minutes" would need a per-payer table with no ceiling on it, which
    register rule 2 forbids.
    """

    name = "f_retry_burst_rate"
    family = "F5"
    state_bytes = 88
    human_template = "{value:.0%} of today's attempts are same-payer retry bursts"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return _ZERO, _ZERO, _ZERO

    def daily(self, num: float, den: float, aux: float) -> float:
        return _ratio(num, den)

    def update(self, state: FeatureState, event: Transaction) -> None:
        assert isinstance(state, DailyState)
        if state.day != event.event_date:
            self._roll(state, event.event_date)
        prior = sum(
            1
            for payer, when in state.ring
            if payer == event.payer_id
            and (event.event_time - when).total_seconds() <= RETRY_WINDOW_S
        )
        state.num += float(prior >= 2)
        state.den += 1.0
        state.ring.append((event.payer_id, event.event_time))
        if len(state.ring) > RETRY_RING:
            del state.ring[0]
        state.last_time = event.event_time

    def read(self, state: DailyState, today: date) -> float:
        return self.today_value(state, today)

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        today = as_of.date()
        # The online runner sees events in (event_time, event_id) order per merchant, which
        # is the order the parity harness replays. The lag comparison below is only correct
        # against that same order, so it is imposed here rather than assumed.
        ordered = frame.sort(["merchant_id", "event_time", "event_id"])
        lags = [
            (
                (pl.col("payer_id").shift(k).over("merchant_id") == pl.col("payer_id"))
                & (
                    (pl.col("event_time") - pl.col("event_time").shift(k).over("merchant_id"))
                    .dt.total_nanoseconds()
                    .cast(pl.Float64)
                    / 1e9
                    <= RETRY_WINDOW_S
                )
            )
            .fill_null(False)
            .cast(pl.Int32)
            for k in range(1, RETRY_RING + 1)
        ]
        bursty = (pl.sum_horizontal(lags) >= 2).cast(pl.Float64)
        merchants = frame.select("merchant_id").unique().collect().sort("merchant_id")
        if merchants.is_empty():
            return self.empty(self.name)
        cur = (
            ordered.with_columns(bursty.alias("b"))
            .filter(pl.col("event_date") == today)
            .group_by("merchant_id")
            .agg(pl.col("b").sum().alias("num"), pl.len().cast(pl.Float64).alias("den"))
            .collect()
        )
        return (
            merchants.join(cur, on="merchant_id", how="left")
            .with_columns(pl.col("num").fill_null(0.0), pl.col("den").fill_null(0.0))
            .with_columns(
                pl.when(pl.col("den") > 0.0)
                .then(pl.col("num") / pl.col("den"))
                .otherwise(0.0)
                .alias(self.name)
            )
            .select("merchant_id", self.name)
            .sort("merchant_id")
        )


# ═════════════════════════════════════════════════════════════════════════════
# F6 — Refund and dispute precursors
#
# Refunds only. Chargebacks are the label, delayed; a chargeback-derived feature at
# decision time is either unavailable or fatal leakage, and the schema keeps the two in
# separate tables so the distinction cannot be lost by accident.
# ═════════════════════════════════════════════════════════════════════════════


@register
class RefundRateZ(AuthFailRateZ):
    """Daily refund count ÷ transaction count, against the merchant's own baseline.

    Persona L8 (travel/OTA) is the hard negative here — legitimately high refund rate —
    which is exactly why this is a z against the merchant's own history and not a level.
    """

    name = "d_refund_rate_z"
    family = "F6"
    state_bytes = 48
    human_template = "refund rate is {value:.1f}σ from this merchant's own norm"

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return float(event.is_refund), 1.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return pl.col("is_refund").cast(pl.Float64), _ONE, _ZERO


@register
class RefundAmountRatio(Tier1Spec):
    """Refunded rupees ÷ captured rupees over a trailing 7 days — the value siphon."""

    name = "d_refund_amount_ratio"
    family = "F6"
    window = 7
    state_bytes = 120
    human_template = "{value:.0%} of trailing-7d captured value was refunded"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        refunded = event.amount_inr if event.is_refund else 0.0
        captured = event.amount_inr if _is_captured(event) else 0.0
        return refunded, captured, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        refunded = pl.when(pl.col("is_refund")).then(pl.col("amount_inr")).otherwise(0.0)
        return refunded, _GMV, _ZERO

    def read(self, state: DailyState, today: date) -> float:
        num, den, _ = self.trailing(state, today, 7)
        return _ratio(num, den)

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        expr = (
            pl.when(pl.col("den") > 0.0).then(pl.col("num") / pl.col("den")).otherwise(0.0)
        )
        return self.window_batch(frame, as_of, 7, expr)


# ═════════════════════════════════════════════════════════════════════════════
# F7 — Temporal pattern drift
# ═════════════════════════════════════════════════════════════════════════════


@register
class InterarrivalCv(Tier1Spec):
    """Coefficient of variation of inter-arrival times over a trailing 7 days.

    The one feature that uses the continuous-time structure v1's fixed-window design threw
    away. Human traffic is bursty and has a CV near or above 1; a script is metronomic and
    drives it toward 0. A gap is booked against the day of its *later* event, which is the
    only definition both runners can agree on without one of them looking ahead.
    """

    name = "h_interarrival_cv"
    family = "F7"
    window = 7
    state_bytes = 192
    human_template = "inter-arrival CV is {value:.2f} — {value:.2f} is scripted at 0"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        return _ZERO, _ZERO, _ZERO

    def update(self, state: FeatureState, event: Transaction) -> None:
        assert isinstance(state, DailyState)
        previous = state.last_time
        if state.day != event.event_date:
            self._roll(state, event.event_date)
        if previous is not None:
            gap = (event.event_time - previous).total_seconds()
            state.num += gap
            state.den += 1.0
            state.aux += gap * gap
        state.last_time = event.event_time

    def read(self, state: DailyState, today: date) -> float:
        total, n, sumsq = self.trailing(state, today, 7)
        if n < 2.0:
            return 0.0
        mean = total / n
        var = sumsq / n - mean * mean
        std = var**0.5 if var > 0.0 else 0.0
        return std / mean if mean > 0.0 else 0.0

    def daily_frame(self, frame: pl.LazyFrame) -> pl.DataFrame:
        gap = (
            (pl.col("event_time") - pl.col("event_time").shift(1).over("merchant_id"))
            .dt.total_nanoseconds()
            .cast(pl.Float64)
            / 1e9
        )
        return (
            frame.sort(["merchant_id", "event_time", "event_id"])
            .with_columns(gap.alias("g"))
            .group_by(["merchant_id", "event_date"])
            .agg(
                pl.col("g").sum().alias("num"),
                pl.col("g").is_not_null().sum().cast(pl.Float64).alias("den"),
                (pl.col("g") * pl.col("g")).sum().alias("aux"),
            )
            .collect()
        )

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        daily = self.daily_frame(frame)
        if daily.is_empty():
            return self.empty(self.name)
        today = as_of.date()
        merchants = daily.select("merchant_id").unique().sort("merchant_id")
        win = (
            daily.filter(
                (pl.col("event_date") <= today)
                & (pl.col("event_date") > today - timedelta(days=7))
            )
            .group_by("merchant_id")
            .agg(
                pl.col("num").sum().alias("s"),
                pl.col("den").sum().alias("n"),
                pl.col("aux").sum().alias("ss"),
            )
        )
        joined = merchants.join(win, on="merchant_id", how="left").fill_null(0.0)
        n = joined["n"].to_numpy()
        safe = np.where(n > 0.0, n, 1.0)
        mean = joined["s"].to_numpy() / safe
        var = joined["ss"].to_numpy() / safe - mean * mean
        std = np.sqrt(np.where(var > 0.0, var, 0.0))
        cv = np.where((n >= 2.0) & (mean > 0.0), std / np.where(mean > 0.0, mean, 1.0), 0.0)
        return joined.select("merchant_id").with_columns(pl.Series(self.name, cv))


@register
class WeekendShareZ(Tier1Spec):
    """Trailing-7d weekend GMV share against the merchant's own warmup share.

    A binomial-proportion z rather than a Welford z: the quantity is a proportion, the
    baseline is a pooled proportion over the warmup window, and its standard error is
    known in closed form. A z of daily weekend share would be degenerate — it is 0 on five
    days a week by construction.
    """

    name = "h_weekend_share_z"
    family = "F7"
    window = 7
    state_bytes = 144
    human_template = "weekend GMV share is {value:.1f}σ from this merchant's own pattern"
    has_cohort_residual = True

    def contrib(self, event: Transaction) -> tuple[float, float, float]:
        gmv = event.amount_inr if _is_captured(event) else 0.0
        weekend = gmv if event.event_date.weekday() >= 5 else 0.0
        return weekend, gmv, 0.0

    def contrib_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        weekend = pl.when(pl.col("event_date").dt.weekday() >= 6).then(_GMV).otherwise(0.0)
        return weekend, _GMV, _ZERO

    def read(self, state: DailyState, today: date) -> float:
        if not self.warm_elapsed(state, today):
            return 0.0
        if state.warm_den <= 0.0:
            return 0.0
        p0 = state.warm_num / state.warm_den
        num, den, _ = self.trailing(state, today, 7)
        if den <= 0.0:
            return 0.0
        se = (p0 * (1.0 - p0) / 7.0) ** 0.5
        z = (num / den - p0) / max(se, Z_FLOOR)
        return float(min(max(z, -Z_CLIP), Z_CLIP))

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        daily = self.daily_frame(frame)
        if daily.is_empty():
            return self.empty(self.name)
        daily = self.anchored(daily).with_columns(
            (pl.col("event_date") - pl.col("anchor")).dt.total_days().alias("age")
        )
        today = as_of.date()
        merchants = daily.select("merchant_id").unique().sort("merchant_id")
        warm = (
            daily.filter((pl.col("age") >= 0) & (pl.col("age") < self.warmup_days))
            .group_by("merchant_id")
            .agg(pl.col("num").sum().alias("wn"), pl.col("den").sum().alias("wd"))
        )
        win = (
            daily.filter(
                (pl.col("event_date") <= today)
                & (pl.col("event_date") > today - timedelta(days=7))
            )
            .group_by("merchant_id")
            .agg(pl.col("num").sum().alias("n"), pl.col("den").sum().alias("d"))
        )
        onboard = daily.group_by("merchant_id").agg(pl.col("anchor").first())
        joined = (
            merchants.join(warm, on="merchant_id", how="left")
            .join(win, on="merchant_id", how="left")
            .join(onboard, on="merchant_id", how="left")
            .with_columns(pl.col("wn", "wd", "n", "d").fill_null(0.0))
            .sort("merchant_id")
        )
        wd = joined["wd"].to_numpy()
        d = joined["d"].to_numpy()
        p0 = np.where(wd > 0.0, joined["wn"].to_numpy() / np.where(wd > 0.0, wd, 1.0), 0.0)
        p_hat = np.where(d > 0.0, joined["n"].to_numpy() / np.where(d > 0.0, d, 1.0), 0.0)
        se = np.sqrt(p0 * (1.0 - p0) / 7.0)
        z = np.clip((p_hat - p0) / np.maximum(se, Z_FLOOR), -Z_CLIP, Z_CLIP)
        elapsed = np.array(
            [
                dt is not None and (today - dt).days >= self.warmup_days
                for dt in joined["anchor"].to_list()
            ]
        )
        ok = elapsed & (wd > 0.0) & (d > 0.0)
        return joined.select("merchant_id").with_columns(
            pl.Series(self.name, np.where(ok, z, 0.0))
        )


# ═════════════════════════════════════════════════════════════════════════════
# F9 — Static profile and mismatch
#
# Not drift features: they modulate the others and enter the model directly. Onboarding
# facts, constant for the merchant's life, so the online form is a snapshot taken at first
# touch and the offline form is a read of the profile table.
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class StaticState(FeatureState):
    onboarded: date | None = None
    value: float = 0.0


class StaticSpec(FeatureSpec):
    """An onboarding fact, served identically by both runners."""

    tier: ClassVar[Tier] = Tier.T1
    family: ClassVar[str] = "F9"
    has_cohort_residual: ClassVar[bool] = False
    name: ClassVar[str] = "__abstract_static__"
    state_bytes: ClassVar[int] = 1
    human_template: ClassVar[str] = "{value}"

    @abstractmethod
    def of(self, profile: MerchantProfile, today: date) -> float:
        """The feature's value for this profile at this epoch."""

    def init_state(self) -> FeatureState:
        return StaticState()

    def state_of(self, merchant: MerchantState) -> FeatureState:
        if self.name not in merchant.feature_states:
            merchant.feature_states[self.name] = StaticState(
                onboarded=merchant.profile.onboarded_at.date(),
                value=self.of(merchant.profile, merchant.profile.onboarded_at.date()),
            )
        return merchant.feature_states[self.name]

    def update(self, state: FeatureState, event: Transaction) -> None:
        """Onboarding facts do not move. Nothing to fold."""

    def value(self, state: FeatureState, as_of: datetime) -> float:
        assert isinstance(state, StaticState)
        return state.value

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        today = as_of.date()
        # Every merchant in the profile table, not only those with transactions: a static
        # is defined for a merchant that has never transacted, and the online runner will
        # be reporting it.
        rows = sorted(PROFILES.values(), key=lambda p: p.merchant_id) if PROFILES else []
        if not rows:
            profiles_frame()  # raises with the actionable message
        return pl.DataFrame(
            {
                "merchant_id": [p.merchant_id for p in rows],
                self.name: [self.of(p, today) for p in rows],
            },
            schema={"merchant_id": pl.String(), self.name: pl.Float64()},
        )


@register
class MccGroup(StaticSpec):
    """The merchant's MCC group, as a stable integer code.

    The register says one-hot, and one-hot needs a vocabulary — which lives in the
    generator's config and does not exist until T-112 lands. A CRC32 bucket is stable
    across processes and platforms (unlike ``hash()``), and the model layer can widen it to
    one-hot from the code once the vocabulary is real. Logged as a placeholder.
    """

    name = "p_mcc_group"
    state_bytes = 8
    human_template = "merchant category group code {value:.0f}"

    def of(self, profile: MerchantProfile, today: date) -> float:
        return float(zlib.crc32(profile.mcc_group.encode("utf-8")) % 4096)


@register
class DaysSinceOnboarding(StaticSpec):
    """Days since onboarding.

    Risk is *non-monotonic* in this — a bust-out has a characteristic latency, neither the
    first week nor the tenth month — so it is handed to the tree raw rather than bucketed
    into a shape somebody guessed.
    """

    name = "p_days_since_onboarding"
    state_bytes = 8
    human_template = "{value:.0f} days since onboarding"

    def of(self, profile: MerchantProfile, today: date) -> float:
        return float((today - profile.onboarded_at.date()).days)

    def state_of(self, merchant: MerchantState) -> FeatureState:
        if self.name not in merchant.feature_states:
            merchant.feature_states[self.name] = StaticState(
                onboarded=merchant.profile.onboarded_at.date()
            )
        return merchant.feature_states[self.name]

    def value(self, state: FeatureState, as_of: datetime) -> float:
        # The one "static" that is a function of the clock, which is exactly why `value`
        # takes `as_of`: a merchant with no events still ages one day per epoch.
        assert isinstance(state, StaticState)
        if state.onboarded is None:
            return 0.0
        return float((as_of.date() - state.onboarded).days)


@register
class KycTier(StaticSpec):
    name = "p_kyc_tier"
    state_bytes = 8
    human_template = "KYC tier {value:.0f}"

    def of(self, profile: MerchantProfile, today: date) -> float:
        return float(profile.kyc_tier)


@register
class VintageMonths(StaticSpec):
    name = "p_vintage_months"
    state_bytes = 8
    human_template = "business age {value:.0f} months at onboarding"

    def of(self, profile: MerchantProfile, today: date) -> float:
        return float(profile.vintage_months)


@register
class DeclaredMonthlyGmv(StaticSpec):
    name = "p_declared_monthly_gmv"
    state_bytes = 8
    human_template = "declared monthly GMV ₹{value:,.0f}"

    def of(self, profile: MerchantProfile, today: date) -> float:
        return profile.declared_monthly_gmv


@register
class CityTier(StaticSpec):
    name = "p_city_tier"
    state_bytes = 8
    human_template = "city tier {value:.0f}"

    def of(self, profile: MerchantProfile, today: date) -> float:
        return float(profile.city_tier)
