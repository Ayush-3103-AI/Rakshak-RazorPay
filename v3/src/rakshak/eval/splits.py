"""The split engine: temporal + merchant-group disjoint + label-availability aware.

Three constraints applied *simultaneously* (10-eval-harness-spec.md §1, FR-020). Any one
of them alone leaks:

- **Temporal** — days 0-119 train, 120-149 val, 150-179 test.
- **Merchant-group** — merchants are hashed to folds, so no ``merchant_id`` appears in
  two splits. Without it a model memorises merchant identity and reports inflated numbers
  on the same merchants it trained on.
- **Label-availability** — training at decision time ``t`` may use only labels with
  ``label_available_at <= t``. With a 45-120 day chargeback delay, a merchant that turned
  fraudulent on day 100 is *not labelled* during a day-120 run. A harness that hands the
  model that label is measuring a system that cannot exist.

``available_labels()`` is the **only** path to the label table anywhere in this repo.
``store.py`` deliberately has no label reader, and ``tests/gates/test_label_access.py``
AST-scans ``src/`` to assert no second door was opened. One implementation, one place to
get it wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import blake2b
from pathlib import Path
from typing import Final, get_args

import polars as pl

from rakshak.schemas import Split

__all__ = [
    "DEFAULT_BOUNDARIES",
    "DEFAULT_LABEL_PATH",
    "LabelCoverage",
    "SplitBoundaries",
    "assign_rows",
    "available_labels",
    "label_coverage",
    "merchant_fold",
    "split_of_day",
]

#: Where the generator writes the label table. Only this module reads it.
DEFAULT_LABEL_PATH: Final = Path("data/v2/labels.parquet")

#: Fixed salt for the merchant fold hash. Python's ``hash()`` is randomised per process,
#: so a fold assignment built on it would silently change between runs and quietly move
#: merchants across splits. blake2b with a constant salt is stable forever.
_FOLD_SALT: Final = b"rakshak-v2-merchant-fold"

SPLITS: Final[tuple[Split, ...]] = get_args(Split)


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """Day ranges (inclusive, 0-based from ``origin``) and the merchant fold shares.

    The fold shares mirror the temporal shares so that the two constraints do not fight
    each other: a 2/3 temporal train split paired with a 1/2 merchant train fold would
    throw away a third of the training days for no reason.
    """

    origin: date
    train: tuple[int, int] = (0, 119)
    val: tuple[int, int] = (120, 149)
    test: tuple[int, int] = (150, 179)

    def __post_init__(self) -> None:
        spans = (self.train, self.val, self.test)
        for lo, hi in spans:
            if lo > hi:
                raise ValueError(f"split span is empty or reversed: {(lo, hi)!r}")
        for (_, a_hi), (b_lo, _) in zip(spans, spans[1:], strict=False):
            if b_lo != a_hi + 1:
                raise ValueError(
                    "split spans must be contiguous and ordered train < val < test; got "
                    f"{spans!r} — a gap silently drops days and an overlap leaks them"
                )

    @property
    def n_days(self) -> int:
        return self.test[1] + 1

    @property
    def fold_shares(self) -> tuple[float, float, float]:
        total = float(self.n_days)
        return tuple(  # type: ignore[return-value]
            (hi - lo + 1) / total for lo, hi in (self.train, self.val, self.test)
        )

    def day_index(self, as_of: date | datetime) -> int:
        """Days since ``origin``. Accepts a date or a tz-aware UTC datetime."""
        if isinstance(as_of, datetime):
            if as_of.tzinfo is None:
                raise ValueError(f"as_of must be tz-aware UTC; got naive {as_of!r}")
            as_of = as_of.astimezone(UTC).date()
        return (as_of - self.origin).days


#: The boundaries hashed into EVAL-LOCK.json. ``origin`` is the generator's day 0.
DEFAULT_BOUNDARIES: Final = SplitBoundaries(origin=date(2026, 1, 1))


def split_of_day(day: int, boundaries: SplitBoundaries = DEFAULT_BOUNDARIES) -> Split | None:
    """Which split a day index falls in, or ``None`` if it is outside the window."""
    spans = (boundaries.train, boundaries.val, boundaries.test)
    for name, (lo, hi) in zip(SPLITS, spans, strict=True):
        if lo <= day <= hi:
            return name
    return None


def merchant_fold(merchant_id: str, boundaries: SplitBoundaries = DEFAULT_BOUNDARIES) -> Split:
    """Deterministic merchant -> fold assignment. Stable across processes and machines."""
    digest = blake2b(merchant_id.encode(), key=_FOLD_SALT, digest_size=8).digest()
    u = int.from_bytes(digest, "big") / 2**64
    cumulative = 0.0
    for name, share in zip(SPLITS, boundaries.fold_shares, strict=True):
        cumulative += share
        if u < cumulative:
            return name
    return SPLITS[-1]


def assign_rows(
    frame: pl.DataFrame,
    boundaries: SplitBoundaries = DEFAULT_BOUNDARIES,
    *,
    merchant_col: str = "merchant_id",
    day_col: str = "as_of",
) -> pl.DataFrame:
    """Attach a ``split`` column: non-null only where *both* constraints agree.

    A row is in split ``S`` iff its merchant's fold is ``S`` **and** its day falls in
    ``S``'s temporal span. Everything else gets ``None`` and is excluded. This is the
    strict reading of "both constraints simultaneously", and it makes "no merchant_id
    spans two splits" true by construction rather than by a later assertion.
    """
    days = frame.get_column(day_col)
    if days.dtype in (pl.Date, pl.Datetime):
        day_index = days.to_frame().select(
            (pl.col(day_col).cast(pl.Date) - pl.lit(boundaries.origin)).dt.total_days()
        ).to_series()
    else:
        day_index = days.cast(pl.Int64)

    fold = frame.get_column(merchant_col).map_elements(
        lambda m: merchant_fold(m, boundaries), return_dtype=pl.String
    )
    temporal = day_index.map_elements(
        lambda d: split_of_day(int(d), boundaries), return_dtype=pl.String
    )
    return frame.with_columns(
        pl.when(fold == temporal).then(temporal).otherwise(None).alias("split")
    )


# ─────────────────────────────────────────────────────────────────────────────
# The label door. There is exactly one.
# ─────────────────────────────────────────────────────────────────────────────


LabelSource = pl.LazyFrame | pl.DataFrame | Path | str


def _scan(labels: LabelSource) -> pl.LazyFrame:
    if isinstance(labels, pl.LazyFrame):
        return labels
    if isinstance(labels, pl.DataFrame):
        return labels.lazy()
    return pl.scan_parquet(labels)


def available_labels(
    as_of: datetime,
    labels: LabelSource = DEFAULT_LABEL_PATH,
    *,
    include_censored: bool = False,
) -> pl.LazyFrame:
    """Labels the system is permitted to know at ``as_of``. The ONLY way to read labels.

    Applies both gates that a hand-written filter forgets:

    1. ``label_available_at <= as_of`` — a label the operator could not have had yet is
       not a label, it is a leak.
    2. ``is_censored`` rows are dropped. Silently keeping them as negatives deflates
       recall; silently dropping them without saying so inflates prevalence. They are
       dropped here and **counted** by :func:`label_coverage`, which is the difference.
    """
    if as_of.tzinfo is None:
        raise ValueError(
            f"as_of must be tz-aware UTC — a naive comparison against a tz-aware "
            f"label_available_at is a silent no-op filter in some engines; got {as_of!r}"
        )
    frame = _scan(labels).filter(
        pl.col("label_available_at").is_not_null() & (pl.col("label_available_at") <= as_of)
    )
    if not include_censored:
        frame = frame.filter(~pl.col("is_censored"))
    return frame


@dataclass(frozen=True, slots=True)
class LabelCoverage:
    """What the harness knew at ``as_of``, and what it did not. Reported, never dropped."""

    as_of: datetime
    n_merchants: int
    n_available: int
    n_censored: int
    n_pending: int
    n_positive: int

    @property
    def prevalence(self) -> float:
        """Observed prevalence among *available, uncensored* labels.

        This is the number that goes in ``EvalResult.prevalence``. v1 reported a headline
        computed at 20% against a real rate near 1.5% and did not say so; the field exists
        so that omission is impossible.
        """
        return self.n_positive / self.n_available if self.n_available else 0.0

    @property
    def coverage(self) -> float:
        return self.n_available / self.n_merchants if self.n_merchants else 0.0


def label_coverage(as_of: datetime, labels: LabelSource = DEFAULT_LABEL_PATH) -> LabelCoverage:
    """Count what was resolved, censored, and still pending at ``as_of``."""
    frame = _scan(labels).select("merchant_id", "label", "label_available_at", "is_censored")
    counts = frame.select(
        pl.len().alias("n_merchants"),
        pl.col("is_censored").sum().alias("n_censored"),
        (
            pl.col("label_available_at").is_not_null()
            & (pl.col("label_available_at") <= as_of)
            & ~pl.col("is_censored")
        )
        .sum()
        .alias("n_available"),
    ).collect()
    n_available = int(counts["n_available"][0])
    n_positive = int(
        available_labels(as_of, labels).select(pl.col("label").sum()).collect()["label"][0] or 0
    )
    n_merchants = int(counts["n_merchants"][0])
    n_censored = int(counts["n_censored"][0])
    return LabelCoverage(
        as_of=as_of,
        n_merchants=n_merchants,
        n_available=n_available,
        n_censored=n_censored,
        n_pending=n_merchants - n_censored - n_available,
        n_positive=n_positive,
    )
