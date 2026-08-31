"""The bounded online state a merchant carries between epochs.

Everything a feature needs to produce today's value from today's event and nothing else
lives here. The 4 KB ceiling (NFR-04) is not a performance preference — it is what makes
the claim "this system is servable" checkable rather than asserted. A feature that cannot
fit is a feature that does not ship, and that trade is made here rather than argued later.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import date, datetime

from rakshak.schemas import SCHEMA_VERSION, MerchantProfile

__all__ = ["BaselineStats", "FeatureState", "MerchantState", "STATE_BYTES_BUDGET"]

#: NFR-04. A serialized MerchantState must fit in this, asserted in tests/perf/.
STATE_BYTES_BUDGET = 4096


@dataclass(slots=True)
class FeatureState:
    """Base for a feature's own online state.

    Subclasses add their own slotted fields. There is no required interface beyond being
    picklable, because forcing every feature to hand-write a size accessor is boilerplate
    that would be wrong by the third feature; ``nbytes`` measures the real thing instead.
    """

    def nbytes(self) -> int:
        """Actual serialized size. The registry checks *declared* ``state_bytes`` at import
        so the budget fails at startup; this measures what the declaration promised, so a
        feature that quietly outgrows its declaration is caught in tests/perf/."""
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))


@dataclass(slots=True)
class BaselineStats:
    """Welford accumulators over the merchant's own post-onboarding warmup window.

    **Frozen after ``warmup_days``, deliberately.** A rolling baseline lets a slow-ramp
    adversary walk the baseline along with it — the merchant is always "normal relative to
    last month" while last month keeps getting worse. That is precisely how typology R2
    defeats naive drift detection, so the baseline stops moving and the z-scores are taken
    against a fixed reference. Name this choice in the writeup; it is a design decision,
    not an implementation shortcut.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    frozen: bool = False

    def observe(self, x: float) -> None:
        if self.frozen:
            return
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def freeze(self) -> None:
        self.frozen = True

    @property
    def variance(self) -> float:
        # Population variance over the warmup sample. n<2 has no dispersion to speak of.
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return float(self.variance**0.5)

    def z(self, x: float, *, floor: float = 1e-9) -> float:
        """z-score of ``x`` against the frozen baseline.

        ``floor`` keeps a merchant whose warmup window was perfectly flat from producing
        an infinite z on its first varied day. Without it, a dormant merchant's first
        transaction is an alert every time, which is a false-positive generator, not a
        detector.
        """
        if self.count == 0:
            return 0.0
        return (x - self.mean) / max(self.std, floor)


@dataclass(slots=True)
class MerchantState:
    """The whole of what the online path remembers about one merchant.

    Serialized size <= ``STATE_BYTES_BUDGET`` (NFR-04). The registry sums each feature's
    *declared* ``state_bytes`` at import time and refuses to load if the total exceeds the
    budget — fail at startup, not at scale.
    """

    merchant_id: str
    profile: MerchantProfile
    baseline: BaselineStats = field(default_factory=BaselineStats)
    feature_states: dict[str, FeatureState] = field(default_factory=dict)
    last_event_time: datetime | None = None
    schema_version: int = SCHEMA_VERSION

    def warmup_elapsed(self, as_of: datetime, warmup_days: int) -> bool:
        return (as_of - self.profile.onboarded_at).days >= warmup_days

    def maybe_freeze_baseline(self, as_of: datetime, warmup_days: int) -> None:
        if not self.baseline.frozen and self.warmup_elapsed(as_of, warmup_days):
            self.baseline.freeze()

    def nbytes(self) -> int:
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))


def day_of(when: datetime | date) -> date:
    """The epoch a timestamp belongs to. Daily epochs, defined once so that a feature
    cannot accidentally bucket by local date on one path and UTC date on the other —
    which is a parity failure that only appears for events near midnight."""
    return when.date() if isinstance(when, datetime) else when
