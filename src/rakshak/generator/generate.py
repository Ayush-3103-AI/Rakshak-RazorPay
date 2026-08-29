"""Synthetic merchant transaction generator — an EVALUATION ARTIFACT, not a fraud toolkit.

SCOPE AND SAFETY (FR-006, CLAUDE.md non-negotiable #2)
------------------------------------------------------
This module exists for exactly one reason: to produce labelled merchant streams so that
Rakshak's detection layer can be *measured* on data whose ground truth we control. It is
defensive tooling.

It produces synthetic rows in a local parquet file. It contains no payment-system client, no
credential handling, no network calls, and no interaction with any live or test payment API.
Its "fraud typologies" are coarse statistical caricatures — volume ramps, payer-graph shape,
ticket-size shifts, refund rates — at the level of abstraction a risk analyst would describe on
a whiteboard. They encode no operational tradecraft and produce nothing usable against a live
system. Read as an attack recipe the file says only "fraud changes a merchant's statistics",
which is the premise of every fraud-detection paper ever published.

Every result measured on this data must be labelled synthetic, per CLAUDE.md non-negotiable #3.

WHAT IT MODELS
--------------
A population of merchants heterogeneous in category, average order value (AOV), monthly volume,
payer loyalty, refund behaviour and organic growth trend (FR-002). Each merchant walks a latent
state path over the horizon; a fraction are assigned one of five typologies (FR-004, FR-005).

Latent states (4, matching `config.N_HIDDEN_STATES`):
    HEALTHY  — normal operation for the merchant's category
    RAMP     — a drift away from the merchant's own baseline, not yet blatant
    FRAUD    — the typology's full expression
    DORMANT  — the merchant has stopped transacting (bust-out aftermath)

Typologies:
    BUST_OUT            legitimate history, then a hard volume ramp, then vanish
    LAUNDERING_ENDPOINT normal tickets, abnormal payer graph: many payers, no repeats,
                        no organic returns
    CATEGORY_DRIFT      a silent shift of ticket-size and time-of-day profile toward a
                        different business category
    REFUND_COLLUSION    the merchant and a small payer set extract value via refunds and
                        chargebacks
    SLOW_RAMP           ADVERSARIAL (FR-005). A monotone, changepoint-free drift whose per-window
                        effect size is comparable to the organic growth/decline that healthy
                        merchants also exhibit. It is here to be REPORTED AS A FAILURE MODE.
                        Do not tune detection to catch it. Do not soften it to flatter a metric.

KNOWN SIMPLIFICATIONS (state them in the README, do not discover them later)
---------------------------------------------------------------------------
* Payer identifiers are merchant-scoped: no payer is shared across merchants, so no
  cross-merchant graph structure exists. This is deliberate — ADR-0002 rejected graph models,
  and a synthetic cross-merchant graph would make graph features circular. Payer-derived
  features must therefore be within-merchant (entropy, repeat ratio, Jaccard drift).
* Refunds and chargebacks are marked by flags, not linked to the original transaction.
* `amount` is always a positive magnitude in INR; `is_refund` carries the direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.cli import base_parser, seed_everything
from rakshak.config import (
    BURN_IN_WINDOWS,
    FRAUD_MERCHANT_RATE,
    GENERATOR_START_DATE,
    HORIZON_DAYS,
    MERCHANT_GROUP_CYCLE,
    N_MERCHANTS,
    SPLIT_DAY_BOUNDS,
    STATE_PATHS_PARQUET,
    SYNTHETIC_DIR,
    TRANSACTIONS_PARQUET,
    WINDOW_DAYS,
)

# --------------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------------

STATES: tuple[str, ...] = ("HEALTHY", "RAMP", "FRAUD", "DORMANT")
"""Latent states, in increasing severity. Length matches `config.N_HIDDEN_STATES`."""

TYPOLOGIES: tuple[str, ...] = (
    "BUST_OUT",
    "LAUNDERING_ENDPOINT",
    "CATEGORY_DRIFT",
    "REFUND_COLLUSION",
    "SLOW_RAMP",
)
"""The four injected typologies (FR-004) plus the adversarial fifth (FR-005)."""

NO_TYPOLOGY: str = "NONE"
"""Typology label carried by merchants that stay healthy for the whole horizon."""

METHODS: tuple[str, ...] = ("UPI", "CARD", "NETBANKING", "WALLET", "EMI")

TRANSACTION_COLUMNS: tuple[str, ...] = (
    "merchant_id",
    "timestamp",
    "amount",
    "payer_id",
    "method",
    "mcc",
    "is_refund",
    "is_chargeback",
)

STATE_PATH_COLUMNS: tuple[str, ...] = (
    "merchant_id",
    "typology",
    "segment_index",
    "state",
    "start_day",
    "end_day",
    "start_timestamp",
    "end_timestamp",
    "start_txn_index",
    "n_txns",
)

_SECONDS_PER_DAY: int = 86_400
_WEEKDAY_FACTOR: np.ndarray = np.array([1.00, 0.98, 1.00, 1.02, 1.15, 1.30, 1.22])
"""Multiplicative daily-volume seasonality, indexed Monday=0."""

_REPEAT_CONCENTRATION: float = 2.5
"""Preferential-attachment exponent for repeat payers. Higher => a smaller, stickier set of
regulars carries more of the volume. Dimensionless."""


# --------------------------------------------------------------------------------------------
# Merchant population
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One merchant category. All monetary values are INR; volumes are transactions per month."""

    mcc: str
    name: str
    aov_lo: float
    aov_hi: float
    volume_lo: float
    volume_hi: float
    amount_sigma: float
    refund_rate: float
    chargeback_rate: float
    unique_payer_frac: float
    peak_hour: float
    method_tilt: tuple[float, float, float, float, float]


SEGMENTS: tuple[Segment, ...] = (
    # mcc     name              aov_lo  aov_hi  vol_lo vol_hi  sig  refund   cb      uniq  hour
    Segment("5411", "GROCERY", 150, 900, 120, 800, 0.50, 0.008, 0.0004, 0.35, 18.5,
            (0.70, 0.15, 0.04, 0.10, 0.01)),
    Segment("5812", "FOOD_DELIVERY", 60, 500, 150, 800, 0.45, 0.015, 0.0006, 0.25, 20.5,
            (0.72, 0.14, 0.02, 0.11, 0.01)),
    Segment("5815", "DIGITAL_GOODS", 60, 1200, 60, 600, 0.70, 0.010, 0.0025, 0.55, 22.0,
            (0.55, 0.30, 0.03, 0.11, 0.01)),
    Segment("5734", "SAAS", 500, 6000, 20, 250, 0.35, 0.020, 0.0010, 0.12, 11.5,
            (0.30, 0.45, 0.15, 0.03, 0.07)),
    Segment("5732", "ELECTRONICS", 2500, 30000, 10, 120, 0.80, 0.050, 0.0020, 0.85, 15.0,
            (0.35, 0.35, 0.10, 0.04, 0.16)),
    Segment("4722", "TRAVEL", 3000, 25000, 8, 90, 0.90, 0.070, 0.0030, 0.92, 13.0,
            (0.30, 0.42, 0.14, 0.03, 0.11)),
    Segment("8299", "EDUCATION", 800, 15000, 6, 70, 0.60, 0.030, 0.0010, 0.80, 17.0,
            (0.38, 0.32, 0.18, 0.03, 0.09)),
    Segment("5944", "JEWELLERY", 5000, 30000, 5, 50, 0.85, 0.040, 0.0040, 0.90, 16.0,
            (0.28, 0.40, 0.12, 0.02, 0.18)),
)
"""Eight categories. AOV spans 60–30 000 INR and volume 5–800 txn/month: >=2 orders of
magnitude on both axes, >=6 distinct MCCs (FR-002)."""


@dataclass(frozen=True)
class GeneratorConfig:
    """Knobs for one generation run.

    Attributes:
        n_merchants: Number of merchants in the population.
        horizon_days: Length of the observation window, in days.
        fraud_rate: Fraction of merchants assigned a typology, spread evenly over `TYPOLOGIES`.
        start_date: ISO date of day 0.
    """

    n_merchants: int = N_MERCHANTS
    horizon_days: int = HORIZON_DAYS
    fraud_rate: float = FRAUD_MERCHANT_RATE
    start_date: str = GENERATOR_START_DATE


@dataclass
class _DayProfile:
    """Per-day parameters for one merchant. Every array has length `horizon_days`."""

    volume_mult: np.ndarray
    amount_mult: np.ndarray
    unique_payer_frac: np.ndarray
    refund_rate: np.ndarray
    chargeback_rate: np.ndarray
    hour_shift: np.ndarray
    collude: np.ndarray
    state: np.ndarray


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Draw one sample uniform in log-space between `low` and `high`."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _baseline_profile(seg: Segment, days: int, rng: np.random.Generator) -> _DayProfile:
    """Build the healthy day-profile for one merchant, including its organic trend.

    Healthy merchants are not flat: each carries a smooth multiplicative growth or decline in
    volume and a smaller drift in ticket size over the horizon. This is what makes SLOW_RAMP
    genuinely hard — a lone within-merchant drift statistic cannot separate a slow evader from
    a business that is simply growing.

    Args:
        seg: The merchant's category.
        days: Horizon length in days.
        rng: Source of randomness.

    Returns:
        A `_DayProfile` in which every day is labelled HEALTHY.
    """
    t = np.arange(days, dtype=float) / max(days - 1, 1)
    volume_trend = np.exp(rng.normal(0.0, 0.45) * t)
    amount_trend = np.exp(rng.normal(0.0, 0.25) * t)
    return _DayProfile(
        volume_mult=volume_trend,
        amount_mult=amount_trend,
        unique_payer_frac=np.full(days, seg.unique_payer_frac),
        refund_rate=np.full(days, seg.refund_rate),
        chargeback_rate=np.full(days, seg.chargeback_rate),
        hour_shift=np.zeros(days),
        collude=np.zeros(days, dtype=bool),
        state=np.full(days, "HEALTHY", dtype=object),
    )


# --------------------------------------------------------------------------------------------
# Onset schedule — WHEN a merchant goes bad
# --------------------------------------------------------------------------------------------
#
# T-0003 drew every onset from a fixed fraction of the horizon (roughly days 67-187 at the
# 270-day default). That does not compose with the frozen evaluation split: train is days
# 0-179, validate 180-209, test 210-269, so EVERY bad merchant in validate and test had
# already gone bad before its own window opened. Detection lag was undefined there and the
# project's headline claim — catching a merchant DRIFTING from good to bad — would have been
# measured on the much easier task of spotting an already-bad, often already-silent merchant.
#
# The fix is to place each merchant's onset inside the window its merchant group is scored
# on. `eval.splits.assign_merchant_groups` deals sorted merchant IDs round-robin over
# `MERCHANT_GROUP_CYCLE` within each typology; `_assign_typologies` below deals typologies in
# the same order, so a merchant's position within its typology block determines both its
# group and, here, its onset window. Every split therefore holds merchants of every typology
# transitioning inside it, and nothing about the frozen split moved.

MIN_ONSET_DAY: int = (BURN_IN_WINDOWS + 1) * WINDOW_DAYS
"""Earliest onset, in days. The feature layer standardises every merchant against its own
first `BURN_IN_WINDOWS * WINDOW_DAYS` = 56 days, so an onset inside that window would
contaminate the baseline the emissions are measured against. 63 days = 9 windows leaves one
clear window of margin."""

MIN_POST_ONSET_DAYS: int = 35
"""Shortest post-onset horizon any merchant is given, in days. A BUST_OUT ramps for at most
25 days before vanishing, so 35 leaves every typology room to express itself and still be
observed. Units: days."""


def onset_window(position: int, days: int) -> tuple[int, int]:
    """Day range [lo, hi) from which merchant `position`'s typology onset is drawn.

    Args:
        position: Index of the merchant within its own typology block, matching the
            round-robin that `eval.splits.assign_merchant_groups` uses.
        days: Horizon length in days. Bounds are scaled from the 270-day default so that a
            shorter run keeps the same shape.

    Returns:
        `(lo, hi)` with `lo < hi`, both in days since day 0.
    """
    scale = days / HORIZON_DAYS
    group = MERCHANT_GROUP_CYCLE[position % len(MERCHANT_GROUP_CYCLE)]
    start, end = SPLIT_DAY_BOUNDS[group]
    lo = max(int(round(start * scale)), int(round(MIN_ONSET_DAY * scale)))
    hi = min(int(round(end * scale)), days - int(round(MIN_POST_ONSET_DAYS * scale)))
    lo = min(lo, max(hi - 1, 1))
    return lo, max(hi, lo + 1)


# --------------------------------------------------------------------------------------------
# Typologies
# --------------------------------------------------------------------------------------------


def _ramp(days: int, start: int, stop: int, lo: float, hi: float) -> np.ndarray:
    """Return a length-`days` array that is `lo` before `start` and glides to `hi` by `stop`."""
    out = np.full(days, lo)
    if stop <= start:
        return out
    out[start:stop] = np.linspace(lo, hi, stop - start, endpoint=False)
    out[stop:] = hi
    return out


def _inject_bust_out(
    p: _DayProfile, seg: Segment, days: int, onset: int, rng: np.random.Generator
) -> None:
    """Legitimate history, a hard volume ramp, then the merchant vanishes."""
    ramp_len = int(rng.integers(10, 26))
    vanish = min(onset + ramp_len, days)
    mid = onset + ramp_len // 2

    p.volume_mult[onset:vanish] *= _ramp(days, onset, vanish, 1.0, rng.uniform(8.0, 18.0))[
        onset:vanish
    ]
    p.volume_mult[vanish:] = 0.0
    p.amount_mult[onset:vanish] *= _ramp(days, onset, vanish, 1.0, rng.uniform(1.5, 3.0))[
        onset:vanish
    ]
    p.unique_payer_frac[onset:vanish] = 0.90
    p.chargeback_rate[mid:vanish] = rng.uniform(0.04, 0.09)
    p.refund_rate[onset:vanish] = seg.refund_rate * 0.3

    p.state[onset:mid] = "RAMP"
    p.state[mid:vanish] = "FRAUD"
    p.state[vanish:] = "DORMANT"


def _inject_laundering(
    p: _DayProfile, seg: Segment, days: int, onset: int, rng: np.random.Generator
) -> None:
    """Normal ticket sizes, but the payer graph goes flat: many payers, no repeats, no returns."""
    p.volume_mult[onset:] *= rng.uniform(1.3, 2.2)
    p.unique_payer_frac[onset:] = 0.995
    p.refund_rate[onset:] = 0.0005
    p.chargeback_rate[onset:] = seg.chargeback_rate * 0.3
    p.state[onset:] = "FRAUD"


def _inject_category_drift(
    p: _DayProfile, seg: Segment, days: int, onset: int, rng: np.random.Generator
) -> None:
    """Ticket-size and time-of-day profile glide silently toward a different category."""
    glide = min(onset + int(rng.integers(15, 36)), days)

    own_aov = np.sqrt(seg.aov_lo * seg.aov_hi)
    candidates = [
        s for s in SEGMENTS if abs(np.log10(np.sqrt(s.aov_lo * s.aov_hi) / own_aov)) >= 0.5
    ]
    target = candidates[int(rng.integers(0, len(candidates)))]
    ratio = np.sqrt(target.aov_lo * target.aov_hi) / own_aov

    p.amount_mult *= _ramp(days, onset, glide, 1.0, float(ratio))
    p.hour_shift = _ramp(days, onset, glide, 0.0, target.peak_hour - seg.peak_hour)
    p.refund_rate[glide:] = target.refund_rate
    p.state[onset:glide] = "RAMP"
    p.state[glide:] = "FRAUD"


def _inject_refund_collusion(
    p: _DayProfile, seg: Segment, days: int, onset: int, rng: np.random.Generator
) -> None:
    """A small colluding payer set extracts value through refunds and chargebacks."""
    p.amount_mult[onset:] *= rng.uniform(1.2, 2.0)
    p.refund_rate[onset:] = rng.uniform(0.35, 0.55)
    p.chargeback_rate[onset:] = rng.uniform(0.03, 0.06)
    p.unique_payer_frac[onset:] = 0.02
    p.collude[onset:] = True
    p.state[onset:] = "FRAUD"


def _inject_slow_ramp(
    p: _DayProfile, seg: Segment, days: int, onset: int, rng: np.random.Generator
) -> None:
    """ADVERSARIAL (FR-005). A monotone drift hidden inside ordinary business growth.

    Every parameter moves smoothly and by a margin comparable to the organic trend that healthy
    merchants also carry, so there is no changepoint for BOCPD to find and no window statistic
    that clears the healthy population's own spread. The merchant never reaches the FRAUD state
    inside the horizon; it is labelled RAMP throughout, which is the honest label.

    DO NOT WEAKEN THIS TO IMPROVE A METRIC. Its purpose is to be reported as a failure mode.
    """
    p.volume_mult *= _ramp(days, onset, days, 1.0, rng.uniform(1.15, 1.30))
    p.amount_mult *= _ramp(days, onset, days, 1.0, rng.uniform(1.15, 1.35))
    p.unique_payer_frac = np.minimum(
        p.unique_payer_frac * _ramp(days, onset, days, 1.0, 1.25), 0.995
    )
    p.chargeback_rate *= _ramp(days, onset, days, 1.0, 2.5)
    p.state[onset:] = "RAMP"


_INJECTORS = {
    "BUST_OUT": _inject_bust_out,
    "LAUNDERING_ENDPOINT": _inject_laundering,
    "CATEGORY_DRIFT": _inject_category_drift,
    "REFUND_COLLUSION": _inject_refund_collusion,
    "SLOW_RAMP": _inject_slow_ramp,
}


# --------------------------------------------------------------------------------------------
# Per-merchant stream
# --------------------------------------------------------------------------------------------


def _generate_merchant(
    merchant_id: str,
    seg: Segment,
    typology: str,
    position: int,
    days: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate one merchant's transaction stream and its per-day latent state path.

    Args:
        merchant_id: Stable identifier, e.g. "M0042".
        seg: The merchant's category.
        typology: One of `TYPOLOGIES`, or `NO_TYPOLOGY`.
        position: Index of this merchant within its typology block; sets which evaluation
            window its onset is drawn from (`onset_window`). Ignored when healthy.
        days: Horizon length in days.
        rng: Source of randomness.

    Returns:
        A tuple of (column arrays for this merchant's transactions, per-day state labels of
        length `days`, per-transaction day index of length n_txns). `timestamp` is expressed in
        integer seconds since day 0; the caller converts to datetimes.
    """
    profile = _baseline_profile(seg, days, rng)
    if typology != NO_TYPOLOGY:
        lo, hi = onset_window(position, days)
        onset = int(rng.integers(lo, hi))
        _INJECTORS[typology](profile, seg, days, onset, rng)

    aov = _loguniform(rng, seg.aov_lo, seg.aov_hi)
    monthly_volume = _loguniform(rng, seg.volume_lo, seg.volume_hi)
    daily_rate = monthly_volume / 30.0

    weekday = (np.arange(days) + 3) % 7  # day 0 of GENERATOR_START_DATE is a Thursday
    lam = daily_rate * profile.volume_mult * _WEEKDAY_FACTOR[weekday]
    counts = rng.poisson(np.maximum(lam, 0.0))
    n = int(counts.sum())

    day_index = np.repeat(np.arange(days), counts)
    if n == 0:
        empty = {c: np.array([], dtype=object) for c in TRANSACTION_COLUMNS}
        return empty, profile.state, day_index

    hour = (seg.peak_hour + profile.hour_shift[day_index] + rng.normal(0.0, 2.6, n)) % 24.0
    seconds = (day_index * _SECONDS_PER_DAY + hour * 3600.0).astype(np.int64)
    order = np.argsort(seconds, kind="stable")
    seconds = seconds[order]
    day_index = day_index[order]

    amount = np.exp(
        np.log(aov * profile.amount_mult[day_index]) + seg.amount_sigma * rng.standard_normal(n)
    )
    amount = np.round(np.maximum(amount, 1.0), 2)

    # Payer graph. A transaction either mints a new payer or repeats an earlier one. Repeats
    # follow preferential attachment (u**gamma concentrates on the earliest payers), so every
    # merchant accumulates a stable base of regulars rather than a uniform smear. That base is
    # what LAUNDERING_ENDPOINT destroys; without it the repeat-payer signal does not exist.
    is_new = rng.random(n) < profile.unique_payer_frac[day_index]
    is_new[0] = True
    minted = np.cumsum(is_new)
    repeat_pick = np.ceil(minted * rng.random(n) ** _REPEAT_CONCENTRATION)
    payer_local = np.where(is_new, minted, np.maximum(repeat_pick, 1.0).astype(np.int64))
    colluding = profile.collude[day_index]
    if colluding.any():
        pool_size = int(rng.integers(4, 10))
        payer_local = payer_local.copy()
        payer_local[colluding] = 1_000_000 + rng.integers(0, pool_size, int(colluding.sum()))

    method_p = rng.dirichlet(np.asarray(seg.method_tilt) * 40.0)
    method = np.asarray(METHODS, dtype=object)[rng.choice(len(METHODS), size=n, p=method_p)]

    is_refund = rng.random(n) < profile.refund_rate[day_index]
    is_chargeback = (rng.random(n) < profile.chargeback_rate[day_index]) & ~is_refund

    columns = {
        "merchant_id": np.full(n, merchant_id, dtype=object),
        "timestamp": seconds,
        "amount": amount,
        "payer_id": np.char.add(f"{merchant_id}-P", payer_local.astype(str)).astype(object),
        "method": method,
        "mcc": np.full(n, seg.mcc, dtype=object),
        "is_refund": is_refund,
        "is_chargeback": is_chargeback,
    }
    return columns, profile.state, day_index


def _state_segments(
    merchant_id: str,
    typology: str,
    state_by_day: np.ndarray,
    day_index: np.ndarray,
    days: int,
) -> list[dict[str, object]]:
    """Run-length encode a day-level state path into ground-truth transition records.

    FR-003: each record carries the exact day, wall-clock timestamp and within-merchant
    transaction index at which the latent state changed. `start_txn_index` indexes the
    merchant's own timestamp-sorted stream; a segment with `n_txns == 0` (a DORMANT tail) still
    records its transition time.
    """
    change = np.flatnonzero(state_by_day[1:] != state_by_day[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [days]))

    records: list[dict[str, object]] = []
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        first = int(np.searchsorted(day_index, start, side="left"))
        last = int(np.searchsorted(day_index, end, side="left"))
        records.append(
            {
                "merchant_id": merchant_id,
                "typology": typology,
                "segment_index": i,
                "state": str(state_by_day[start]),
                "start_day": int(start),
                "end_day": int(end),
                "start_timestamp": int(start) * _SECONDS_PER_DAY,
                "end_timestamp": int(end) * _SECONDS_PER_DAY,
                "start_txn_index": first,
                "n_txns": last - first,
            }
        )
    return records


# --------------------------------------------------------------------------------------------
# Population-level entry point
# --------------------------------------------------------------------------------------------


def _assign_typologies(
    n_merchants: int, fraud_rate: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each merchant a typology label, spreading the fraudulent ones evenly.

    Returns:
        `(labels, positions)`. `positions[m]` is the merchant's index within its own
        typology block, counted over ascending merchant index — the same ordering
        `eval.splits.assign_merchant_groups` uses to deal merchant groups, which is what
        makes `onset_window` place the onset inside the window this merchant is scored on.
        `positions` is 0 for healthy merchants and unused there.
    """
    labels = np.full(n_merchants, NO_TYPOLOGY, dtype=object)
    positions = np.zeros(n_merchants, dtype=np.int64)
    n_fraud = int(round(n_merchants * fraud_rate))
    chosen = rng.permutation(n_merchants)[:n_fraud]
    for i, m in enumerate(np.sort(chosen)):
        labels[m] = TYPOLOGIES[i % len(TYPOLOGIES)]
        positions[m] = i // len(TYPOLOGIES)
    return labels, positions


def generate(
    config: GeneratorConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the transaction stream and the ground-truth latent state paths.

    Args:
        config: Population size, horizon and fraud prevalence.
        rng: Seeded generator; the same seed yields byte-identical frames.

    Returns:
        `(transactions, state_paths)`. `transactions` has columns `TRANSACTION_COLUMNS`, sorted
        by (merchant_id, timestamp), with `amount` in INR and `timestamp` a naive datetime64[ns].
        `state_paths` has columns `STATE_PATH_COLUMNS`, one row per contiguous latent-state
        segment, and every merchant appears at least once.
    """
    days = config.horizon_days
    typologies, positions = _assign_typologies(config.n_merchants, config.fraud_rate, rng)
    segment_ids = rng.integers(0, len(SEGMENTS), config.n_merchants)

    chunks: list[dict[str, np.ndarray]] = []
    records: list[dict[str, object]] = []
    for m in range(config.n_merchants):
        merchant_id = f"M{m:05d}"
        columns, state_by_day, day_index = _generate_merchant(
            merchant_id,
            SEGMENTS[segment_ids[m]],
            str(typologies[m]),
            int(positions[m]),
            days,
            rng,
        )
        if len(columns["timestamp"]):
            chunks.append(columns)
        records.extend(
            _state_segments(merchant_id, str(typologies[m]), state_by_day, day_index, days)
        )

    epoch = pd.Timestamp(config.start_date)
    transactions = pd.DataFrame(
        {c: np.concatenate([chunk[c] for chunk in chunks]) for c in TRANSACTION_COLUMNS}
    )
    transactions["timestamp"] = epoch + pd.to_timedelta(transactions["timestamp"], unit="s")
    transactions = transactions.sort_values(
        ["merchant_id", "timestamp"], kind="stable", ignore_index=True
    )

    state_paths = pd.DataFrame.from_records(records, columns=list(STATE_PATH_COLUMNS))
    for col in ("start_timestamp", "end_timestamp"):
        state_paths[col] = epoch + pd.to_timedelta(state_paths[col], unit="s")
    return transactions, state_paths


def write_outputs(
    transactions: pd.DataFrame, state_paths: pd.DataFrame, out_dir: Path | None = None
) -> tuple[Path, Path]:
    """Write both frames to parquet.

    Args:
        transactions: Frame from `generate`.
        state_paths: Frame from `generate`.
        out_dir: Destination directory; defaults to `config.SYNTHETIC_DIR`.

    Returns:
        `(transactions_path, state_paths_path)`.
    """
    if out_dir is None:
        txn_path, path_path = TRANSACTIONS_PARQUET, STATE_PATHS_PARQUET
    else:
        txn_path, path_path = out_dir / "transactions.parquet", out_dir / "state_paths.parquet"
    txn_path.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_parquet(txn_path, index=False)
    state_paths.to_parquet(path_path, index=False)
    return txn_path, path_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = base_parser(
        "Generate synthetic merchant streams for evaluating Rakshak. "
        "Evaluation artifact only — see the module docstring."
    )
    parser.add_argument("--merchants", type=int, default=N_MERCHANTS, help="Merchant count.")
    parser.add_argument("--days", type=int, default=HORIZON_DAYS, help="Horizon length in days.")
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=FRAUD_MERCHANT_RATE,
        help="Fraction of merchants assigned a typology.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=SYNTHETIC_DIR, help="Output directory for the parquets."
    )
    args = parser.parse_args(argv)

    rng = seed_everything(args.seed)
    config = GeneratorConfig(
        n_merchants=args.merchants, horizon_days=args.days, fraud_rate=args.fraud_rate
    )
    transactions, state_paths = generate(config, rng)
    txn_path, path_path = write_outputs(transactions, state_paths, args.out_dir)

    counts = state_paths.groupby("merchant_id")["typology"].first().value_counts()
    print(
        f"rakshak.generator: seed={args.seed} merchants={config.n_merchants} "
        f"days={config.horizon_days}"
    )
    print(f"  transactions: {len(transactions):,} rows -> {txn_path}")
    print(f"  state paths:  {len(state_paths):,} segments -> {path_path}")
    for typology, count in counts.items():
        print(f"    {typology:<20} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
