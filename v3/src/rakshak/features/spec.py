"""The dual-runner contract. One feature definition, two runners, parity asserted to 1e-9.

This is the single most important interface in the repo, and it exists to prevent one
specific failure: a feature that is easy to compute over a full history and impossible to
compute from a stream. A system built out of those features demos beautifully and cannot
be deployed, which is the exact gap between v1's claim and v1's reality.

So every feature is written twice, in the same class:

- ``update`` / ``value`` — O(1) per event, from a bounded ``MerchantState``. This is what
  production would run.
- ``batch`` — the offline equivalent over a polars frame. This is what training runs.

The parity harness replays a stream through both and asserts they agree at every epoch for
every merchant. **When they disagree it is almost always ``batch`` that is wrong**, because
``batch`` is the one with the whole history in front of it and therefore the one that can
quietly see the future.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar

import polars as pl

from rakshak.features.state import FeatureState, MerchantState
from rakshak.schemas import Tier, Transaction

__all__ = ["FeatureSpec", "PARITY_TOLERANCE"]

#: NFR-08. Not "close enough" — the two runners compute the same arithmetic in a different
#: order, so float64 associativity is the only difference either is allowed to have.
PARITY_TOLERANCE = 1e-9


class FeatureSpec(ABC):
    """One feature, defined once, runnable both ways.

    Subclasses declare their metadata as class attributes and implement the four methods.
    Instances are stateless — all mutable state lives in ``MerchantState.feature_states``,
    keyed by ``name`` — so the registry can hold a single shared instance per feature and
    score ten thousand merchants with it.
    """

    #: Matches an ID in 07-feature-register.md. The register and the code agree by this
    #: string or they do not agree at all.
    name: ClassVar[str]
    tier: ClassVar[Tier]
    #: "F1".."F9", the family grouping from the register.
    family: ClassVar[str]
    #: Declared online state budget in bytes. Summed at import; see registry.py.
    state_bytes: ClassVar[int]
    #: The merchant-facing sentence this feature contributes to a reason code. Formatted
    #: with the feature's own value, e.g. "GMV is {value:.1f}σ above this merchant's norm".
    human_template: ClassVar[str]
    #: Whether the cohort layer computes a leave-one-out residual companion for this
    #: feature. Only meaningful for z-scored features (07-feature-register.md).
    has_cohort_residual: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # `cls.__abstractmethods__` is NOT readable here: ABCMeta populates it after
        # `type.__new__` returns, so it is always None inside `__init_subclass__` and a
        # guard written against it never fires. (Found by Lane B in T-120, which had to
        # give its intermediate bases placeholder metadata to get past it.) Walking the
        # MRO for unimplemented abstract methods answers the same question, correctly.
        if any(
            getattr(getattr(cls, attr, None), "__isabstractmethod__", False) for attr in dir(cls)
        ):
            return  # an intermediate abstract base, not a concrete feature
        missing = [
            attr
            for attr in ("name", "tier", "family", "state_bytes", "human_template")
            if not hasattr(cls, attr)
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__} is a concrete FeatureSpec but declares no {missing}. Every "
                f"field is load-bearing: `name` ties it to the register, `state_bytes` to "
                f"the NFR-04 budget, `human_template` to FR-014's reason codes."
            )
        if cls.state_bytes <= 0:
            raise TypeError(
                f"{cls.__name__}.state_bytes must be a positive declared budget; got "
                f"{cls.state_bytes!r}. A feature that claims to be free is a feature whose "
                f"state nobody counted."
            )

    # ── the online runner ────────────────────────────────────────────────────

    @abstractmethod
    def init_state(self) -> FeatureState:
        """A fresh per-merchant state for this feature, as of onboarding."""

    @abstractmethod
    def update(self, state: FeatureState, event: Transaction) -> None:
        """Fold one event into the state. O(1), mutates in place, called in time order.

        Must not look at anything but ``state`` and ``event``. A peek at a wider frame here
        is exactly the leak the whole dual-runner design exists to make impossible.
        """

    @abstractmethod
    def value(self, state: FeatureState, as_of: datetime) -> float:
        """Read the current value. **Must not mutate state.**

        Called once per epoch per merchant. Takes ``as_of`` because several features are
        functions of elapsed time as well as of events — days-since-last-transaction keeps
        rising on a day with no events at all, and a runner that only moved on events would
        report a stale value for exactly the dormant merchants that matter.
        """

    # ── the offline runner ───────────────────────────────────────────────────

    @abstractmethod
    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        """Offline equivalent over the full history, for every merchant at once.

        Returns a two-column frame ``(merchant_id, <name>)``, one row per merchant present
        in ``frame``, sorted by ``merchant_id``.

        ``frame`` is already filtered to ``event_time <= as_of`` by the caller — but do not
        rely on that as your only defence. If your expression would produce a different
        answer given future rows, it is a leak waiting for the one call site that forgets.
        """

    # ── shared helpers ───────────────────────────────────────────────────────

    def state_of(self, merchant: MerchantState) -> FeatureState:
        """This feature's slot in a merchant's state, created on first touch."""
        if self.name not in merchant.feature_states:
            merchant.feature_states[self.name] = self.init_state()
        return merchant.feature_states[self.name]

    def explain(self, value: float) -> str:
        """The merchant-facing sentence for this feature at this value (FR-014)."""
        return self.human_template.format(value=value)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} {self.tier.name} {self.state_bytes}B>"
