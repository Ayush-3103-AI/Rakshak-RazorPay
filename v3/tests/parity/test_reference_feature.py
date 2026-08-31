"""T-102: the dual-runner framework works, proven on a reference feature.

Daily transaction count is chosen deliberately: it is trivial enough that a disagreement
can only be the *framework*, not the feature's arithmetic. Lane B's real features
(T-120, T-122) reuse ``assert_parity`` unchanged.

The last three tests are the ones that matter. They construct a feature that is subtly
wrong in each of the three ways a real feature goes wrong, and assert the harness catches
it. A parity harness nobody has watched fail is a parity harness nobody should trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import numpy as np
import polars as pl
import pytest

from conftest import ParityFailure, assert_parity, synthetic_stream, to_frame
from rakshak.features import registry
from rakshak.features.spec import FeatureSpec
from rakshak.features.state import STATE_BYTES_BUDGET, FeatureState, MerchantState
from rakshak.schemas import MerchantProfile, Tier, Transaction


@dataclass(slots=True)
class CountState(FeatureState):
    """Two scalars. This is what "bounded per-merchant state" looks like."""

    day: date | None = None
    count: float = 0.0


class DailyTxnCount(FeatureSpec):
    """Transactions on the current epoch's date. The reference implementation."""

    name = "ref_txn_count"
    tier = Tier.T1
    family = "F1"
    state_bytes = 24
    human_template = "{value:.0f} transactions today"
    has_cohort_residual = False

    def init_state(self) -> FeatureState:
        return CountState()

    def update(self, state: FeatureState, event: Transaction) -> None:
        assert isinstance(state, CountState)
        # The day roll is the whole subtlety: the counter resets when the calendar does,
        # not when a fixed number of events have arrived.
        if state.day != event.event_date:
            state.day = event.event_date
            state.count = 0.0
        state.count += 1.0

    def value(self, state: FeatureState, as_of: datetime) -> float:
        assert isinstance(state, CountState)
        # Reading on a day with no events must return 0, not yesterday's count. A runner
        # that only moves on events reports stale values for exactly the dormant merchants
        # the `v_dormant_burst` family is built to catch.
        return state.count if state.day == as_of.date() else 0.0

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        return (
            frame.filter(pl.col("event_date") == as_of.date())
            .group_by("merchant_id")
            .agg(pl.len().cast(pl.Float64).alias(self.name))
            .sort("merchant_id")
            .collect()
        )


@pytest.fixture(autouse=True)
def clean_registry() -> object:
    """Registration is global and import-time. Snapshot and restore it so a test that
    registers a throwaway feature cannot leak into the next one."""
    saved_registry = dict(registry.REGISTRY)
    saved_order = registry.ORDER
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved_registry)
    registry.ORDER = saved_order


# ── the framework itself ──────────────────────────────────────────────────────


def test_reference_feature_agrees_at_every_epoch(
    stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    txns, profiles = stream
    assert_parity(DailyTxnCount(), txns, profiles)


def test_parity_holds_for_a_merchant_with_no_events_at_all(
    stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    # synthetic_stream's last merchant never transacts. It is absent from every offline
    # frame and must still read 0.0 online, on every epoch.
    txns, profiles = stream
    silent = max(profiles)
    assert not [t for t in txns if t.merchant_id == silent]
    spec = DailyTxnCount()
    state = MerchantState(merchant_id=silent, profile=profiles[silent])
    at = datetime.combine(date(2026, 1, 10), time.max, tzinfo=UTC)
    assert spec.value(spec.state_of(state), at) == 0.0


def test_value_does_not_mutate_state(
    stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    # `value` is called once per merchant per epoch and sometimes more in the cascade. A
    # mutating reader makes the feature depend on how many times it was read, which is a
    # bug that only appears once stage 1 starts re-reading stage 0's features.
    txns, profiles = stream
    spec = DailyTxnCount()
    state = MerchantState(merchant_id=txns[0].merchant_id, profile=profiles[txns[0].merchant_id])
    for event in txns[:20]:
        if event.merchant_id == state.merchant_id:
            spec.update(spec.state_of(state), event)
    at = datetime.combine(txns[0].event_date, time.max, tzinfo=UTC)
    first = spec.value(spec.state_of(state), at)
    for _ in range(5):
        assert spec.value(spec.state_of(state), at) == first


# ── the harness must actually fail ────────────────────────────────────────────


class LeakyBatch(DailyTxnCount):
    """`batch` ignores `as_of` and counts the merchant's whole history.

    This is the archetypal offline-only bug: correct-looking polars that quietly answers a
    different question than the online runner does.
    """

    name = "ref_leaky"

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
        return (
            frame.group_by("merchant_id")
            .agg(pl.len().cast(pl.Float64).alias(self.name))
            .sort("merchant_id")
            .collect()
        )


class StaleOnline(DailyTxnCount):
    """`value` returns the last day's count on a day with no events."""

    name = "ref_stale"

    def value(self, state: FeatureState, as_of: datetime) -> float:
        assert isinstance(state, CountState)
        return state.count


class OffByOne(DailyTxnCount):
    """A 1e-6 bias — far too small to notice by eye, far too large for 1e-9."""

    name = "ref_offby"

    def value(self, state: FeatureState, as_of: datetime) -> float:
        return super().value(state, as_of) + 1e-6


@pytest.mark.parametrize("broken", [LeakyBatch, StaleOnline, OffByOne])
def test_harness_catches_a_broken_feature(
    broken: type[FeatureSpec],
    stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    txns, profiles = stream
    with pytest.raises(ParityFailure):
        assert_parity(broken(), txns, profiles)


# ── the registry and its budget ───────────────────────────────────────────────


def test_registration_records_order_and_budget() -> None:
    registry.reset_for_testing()
    registry.register(DailyTxnCount)
    assert registry.ORDER == ("ref_txn_count",)
    assert registry.declared_state_bytes() == 24
    assert registry.of_tier(Tier.T1) == (registry.get("ref_txn_count"),)
    assert registry.of_tier(Tier.T2) == ()


def test_a_duplicate_name_is_refused() -> None:
    registry.reset_for_testing()
    registry.register(DailyTxnCount)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(DailyTxnCount)


def test_registry_refuses_to_exceed_the_state_budget() -> None:
    # NFR-04, enforced at import rather than discovered at scale.
    registry.reset_for_testing()

    class Hog(DailyTxnCount):
        name = "ref_hog"
        state_bytes = STATE_BYTES_BUDGET + 1

    with pytest.raises(registry.StateBudgetExceeded, match="NFR-04"):
        registry.register(Hog)
    # And the failed registration must leave nothing behind, or the next import sees a
    # half-registered registry and treats it as valid.
    assert "ref_hog" not in registry.REGISTRY
    assert "ref_hog" not in registry.ORDER
    assert registry.declared_state_bytes() == 0


def test_a_feature_missing_its_metadata_is_refused_at_class_creation() -> None:
    with pytest.raises(TypeError, match="load-bearing"):

        class Nameless(FeatureSpec):
            def init_state(self) -> FeatureState:
                return CountState()

            def update(self, state: FeatureState, event: Transaction) -> None: ...

            def value(self, state: FeatureState, as_of: datetime) -> float:
                return 0.0

            def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.DataFrame:
                return pl.DataFrame()


def test_declared_state_bytes_must_be_positive() -> None:
    with pytest.raises(TypeError, match="positive declared budget"):

        class Free(DailyTxnCount):
            name = "ref_free"
            state_bytes = 0


def test_human_template_renders_a_merchant_readable_string() -> None:
    # FR-014. The reason code a merchant reads when they call and shout.
    assert DailyTxnCount().explain(42.0) == "42 transactions today"


def test_state_stays_inside_its_declared_budget() -> None:
    # The declaration is checked at import; this checks the declaration was honest.
    rng = np.random.default_rng(7)
    txns, profiles = synthetic_stream(rng)
    spec = DailyTxnCount()
    state = MerchantState(merchant_id="M000", profile=profiles["M000"])
    for event in txns:
        if event.merchant_id == "M000":
            spec.update(spec.state_of(state), event)
    assert spec.state_of(state).nbytes() <= STATE_BYTES_BUDGET
    assert state.nbytes() <= STATE_BYTES_BUDGET


def test_the_stream_fixture_contains_the_awkward_cases() -> None:
    # If this ever stops being true the parity suite quietly gets easier, and every
    # feature that follows is tested against a stream with no edges in it.
    rng = np.random.default_rng(7)
    txns, profiles = synthetic_stream(rng)
    frame = to_frame(txns)
    assert frame["is_refund"].sum() > 0, "no refunds in the parity stream"
    assert (frame["status"] == "failed").sum() > 0, "no failed transactions"
    assert len(set(profiles) - set(frame["merchant_id"].unique())) == 1, "no silent merchant"
    per_day = frame.group_by(["merchant_id", "event_date"]).len()
    assert per_day["len"].min() >= 1
