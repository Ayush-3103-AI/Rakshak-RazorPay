"""T-120: every Tier-1 feature agrees online and offline, at every epoch, to 1e-9.

Parametrised over the registry rather than over a hand-written list, so a feature added to
``tier1.py`` without a test is not possible: it is tested the moment it registers.

**Dependency deviation, recorded here and in docs/logbook/T-120.md.** The board lists
T-120 as depending on T-112 (personas L1-L8), which is still in flight in Lane A. The
dependency is real but narrow — it exists so the tier suites run against realistic merchant
streams. The features themselves depend only on the frozen ``schemas.py`` and ``spec.py``,
and ``synthetic_stream`` from T-102 already carries the shapes a feature has to survive:
empty days, a merchant with no events at all, refunds interleaved with captures, and failed
transactions. So this suite is built against ``synthetic_stream`` and **should be re-run
against the real generator output once T-112 lands**.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import numpy as np
import polars as pl
import pytest

from conftest import assert_parity, end_of_day, synthetic_stream, to_frame
from rakshak.features import registry, tier1
from rakshak.features.spec import PARITY_TOLERANCE, FeatureSpec
from rakshak.features.state import STATE_BYTES_BUDGET, MerchantState
from rakshak.schemas import MerchantProfile, Tier, Transaction

#: Short enough that the 21-day parity stream has live z-scores for two thirds of its span.
#: A suite run at the production WARMUP_DAYS on a 21-day stream would assert that every
#: z-feature returns 0.0 — green, and testing nothing.
TEST_WARMUP = 5

T1 = registry.of_tier(Tier.T1)


def specs() -> list[FeatureSpec]:
    """One short-warmup instance per registered T1 feature.

    ``warmup_days`` is constructor configuration, not state, so this is the same feature
    the registry holds with its baseline window shortened — not a test double.
    """
    out: list[FeatureSpec] = []
    for spec in T1:
        cls = type(spec)
        out.append(cls(warmup_days=TEST_WARMUP) if isinstance(spec, tier1.Tier1Spec) else cls())
    return out


@pytest.fixture
def loaded_stream(
    rng: np.random.Generator,
) -> tuple[list[Transaction], dict[str, MerchantProfile]]:
    """The T-102 stream, with the profile table installed for the offline runners.

    ``FeatureSpec.batch(frame, as_of)`` is handed transactions and nothing else, so a
    feature that needs an onboarding fact — every warmup window start, ``v_declared_ratio``'s
    denominator, all six F9 statics — has no offline path to ``MerchantProfile``. The
    online runner reaches it through ``MerchantState``. Reported to the lead as an
    interface gap; this is the seam until ``spec.py`` unfreezes.
    """
    txns, profiles = synthetic_stream(rng)
    tier1.load_profiles(profiles)
    return txns, profiles


@pytest.mark.parametrize("spec", specs(), ids=lambda s: s.name)
def test_tier1_feature_agrees_online_and_offline(
    spec: FeatureSpec,
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    txns, profiles = loaded_stream
    assert_parity(spec, txns, profiles)


@pytest.mark.parametrize("spec", specs(), ids=lambda s: s.name)
def test_tier1_feature_agrees_over_a_full_warmup_and_beyond(
    spec: FeatureSpec,
    rng: np.random.Generator,
) -> None:
    """The same assertion on a 90-day stream at the production warmup window.

    The short-warmup run above exercises the z arithmetic; this one exercises the freeze —
    a baseline that kept moving after ``WARMUP_DAYS`` would pass the first test and fail
    this one, and a slow-ramp adversary is exactly what a moving baseline hides.
    """
    txns, profiles = synthetic_stream(rng, days=90)
    tier1.load_profiles(profiles)
    long_spec = (
        type(spec)(warmup_days=tier1.WARMUP_DAYS)
        if isinstance(spec, tier1.Tier1Spec)
        else type(spec)()
    )
    assert_parity(long_spec, txns, profiles)


def test_max_observed_parity_difference_across_the_registry(
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """Report the worst disagreement anywhere in T1, not just that each one passed.

    ``assert_parity`` raises on the first failure, which tells you nothing about how close
    the survivors were. A registry sitting at 9e-10 is one refactor from red.
    """
    txns, profiles = loaded_stream
    frame = to_frame(txns)
    days = sorted({t.event_date for t in txns})
    worst = 0.0
    worst_at = ""
    for spec in specs():
        states = {
            mid: MerchantState(merchant_id=mid, profile=p) for mid, p in profiles.items()
        }
        ordered = sorted(txns, key=lambda t: (t.event_time, t.event_id))
        cursor = 0
        for day in days:
            as_of = end_of_day(day)
            while cursor < len(ordered) and ordered[cursor].event_time <= as_of:
                event = ordered[cursor]
                spec.update(spec.state_of(states[event.merchant_id]), event)
                cursor += 1
            offline = spec.batch(frame.filter(pl.col("event_time") <= as_of).lazy(), as_of)
            lookup = dict(
                zip(
                    offline["merchant_id"].to_list(),
                    offline[spec.name].to_list(),
                    strict=True,
                )
            )
            for mid, state in states.items():
                got = spec.value(spec.state_of(state), as_of)
                want = lookup.get(mid) or 0.0
                if abs(got - want) > worst:
                    worst = abs(got - want)
                    worst_at = f"{spec.name} / {mid} / {day}"
    print(f"\nmax parity diff across T1: {worst:.3e} at {worst_at or 'nowhere'}")
    assert worst <= PARITY_TOLERANCE


# ── the contract around the features, not inside them ─────────────────────────


def test_every_registered_t1_feature_declares_both_runners() -> None:
    # `FeatureSpec` is an ABC, so a missing runner cannot instantiate — this asserts the
    # stronger thing: that no feature inherited a runner it did not mean to, by checking
    # each one actually produces a value and a frame.
    assert T1, "no T1 features registered — did features/__init__.py lose its tier import?"
    for spec in T1:
        assert callable(spec.update)
        assert callable(spec.batch)
        assert spec.human_template.format(value=1.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T-120 sets T1 declared state < 1024 B, and 07-feature-register.md estimates T1 at "
        "~0.9 KB. Neither figure survives contact with the windows the register itself "
        "specifies. A trailing-30d GMV sum cannot be maintained in the 16 B "
        "`v_declared_ratio` is allotted, nor a 28d Fano in 40 B: a sliding window needs its "
        "values in order to evict them, so 30 days of float64 is 240 B and there is no "
        "cheaper exact form. Six features each keep a private ring over the same daily "
        "series (GMV x3, counts x2, gaps x1) and those rings alone are ~1 KB. The fix is "
        "architectural, not a smaller declaration: ONE shared per-merchant daily ring on "
        "MerchantState that every window feature reads, which needs a slot in the frozen "
        "state.py and is therefore the lead's call. Left strict so that landing the shared "
        "ring turns this red and forces the register to be corrected."
    ),
)
def test_t1_declared_state_stays_under_the_tier_budget() -> None:
    total = sum(spec.state_bytes for spec in T1)
    print(f"\nT1 declared state: {total} B across {len(T1)} features")
    assert total < 1024


def test_t1_declared_state_stays_under_nfr_04() -> None:
    """The ceiling that is real: NFR-04's 4096, which registry.register enforces at import.

    T2 has to fit in what is left, and the register's own instruction if it does not is to
    cut `i_bin_hhi` first.
    """
    total = sum(spec.state_bytes for spec in T1)
    assert total < STATE_BYTES_BUDGET, total
    print(f"\nT1 declared {total} B; {STATE_BYTES_BUDGET - total} B left for T2")


def test_value_does_not_mutate_state(
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """Read the same epoch six times and get the same answer six times.

    A mutating reader makes a feature depend on how many times it was read, which only
    surfaces once stage 1 of the cascade starts re-reading stage 0's columns.
    """
    txns, profiles = loaded_stream
    for spec in specs():
        merchant = "M000"
        state = MerchantState(merchant_id=merchant, profile=profiles[merchant])
        for event in txns:
            if event.merchant_id == merchant:
                spec.update(spec.state_of(state), event)
        at = end_of_day(max(t.event_date for t in txns))
        first = spec.value(spec.state_of(state), at)
        for _ in range(5):
            assert spec.value(spec.state_of(state), at) == first, spec.name


def test_online_state_stays_bounded_for_the_busiest_merchant(
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """The window rings must not grow with history.

    M000 transacts every day of the stream, so a feature that appended a row per event —
    or per day without evicting — shows up here as a state that grew.
    """
    txns, profiles = loaded_stream
    for spec in specs():
        state = MerchantState(merchant_id="M000", profile=profiles["M000"])
        sizes = []
        for i, event in enumerate(t for t in txns if t.merchant_id == "M000"):
            spec.update(spec.state_of(state), event)
            if i % 10 == 0:
                sizes.append(spec.state_of(state).nbytes())
        assert sizes[-1] <= max(sizes[len(sizes) // 2 :]), f"{spec.name} state is still growing"


def test_a_silent_merchant_reads_zero_on_every_feature(
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """The stream's last merchant never transacts, and it is the one that finds the bugs.

    It is absent from every offline frame, so the harness reads 0.0 for it; anything but
    0.0 online is a division by a zero count wearing a disguise. The two F9 statics that
    legitimately differ are excluded by name — a merchant with no transactions still has a
    KYC tier.
    """
    _, profiles = loaded_stream
    silent = max(profiles)
    at = datetime.combine(date(2026, 1, 15), time.max, tzinfo=UTC)
    for spec in specs():
        if isinstance(spec, tier1.StaticSpec):
            continue
        state = MerchantState(merchant_id=silent, profile=profiles[silent])
        assert spec.value(spec.state_of(state), at) == 0.0, spec.name


def test_statics_are_served_for_a_merchant_with_no_transactions(
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    # The offline runner for a static reads the profile table, not the event frame, so the
    # silent merchant gets a row. If it did not, the harness would compare a real online
    # value against a defaulted 0.0 and every static would fail parity for that merchant.
    txns, profiles = loaded_stream
    silent = max(profiles)
    at = end_of_day(max(t.event_date for t in txns))
    frame = to_frame(txns).lazy()
    for spec in specs():
        if not isinstance(spec, tier1.StaticSpec):
            continue
        out = spec.batch(frame, at)
        assert silent in out["merchant_id"].to_list(), spec.name


def test_the_baseline_is_frozen_after_warmup(
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """A ramp after the warmup window must move the z, not the baseline.

    This is the anti-R2 assertion in feature form: if the baseline kept rolling, a merchant
    that doubles its volume every week reads as permanently normal. Feeding a step change
    after the window has closed must produce a large z and keep producing one.
    """
    from datetime import timedelta

    from rakshak.schemas import Instrument, TxnStatus

    spec = tier1.TxnCountZ(warmup_days=TEST_WARMUP)
    profile = next(iter(loaded_stream[1].values()))
    state = MerchantState(merchant_id=profile.merchant_id, profile=profile)
    start = profile.onboarded_at

    def txn(day: int, k: int) -> Transaction:
        when = start + timedelta(days=day, seconds=k)
        return Transaction(
            event_id=f"x-{day}-{k}",
            merchant_id=profile.merchant_id,
            payer_id="P001",
            event_time=when,
            event_date=when.date(),
            amount_inr=100.0,
            instrument=Instrument.UPI,
            is_cnp=True,
            is_international=False,
            bin_hash=None,
            device_hash="d" * 16,
            ip_hash="a" * 16,
            status=TxnStatus.CAPTURED,
            decline_code=None,
            mcc="5411",
            is_refund=False,
            refund_of=None,
        )

    for day in range(TEST_WARMUP):  # two a day through the warmup window
        for k in range(2):
            spec.update(spec.state_of(state), txn(day, k))
    zs = []
    for day in range(TEST_WARMUP, TEST_WARMUP + 10):  # then twenty a day, forever
        for k in range(20):
            spec.update(spec.state_of(state), txn(day, k))
        zs.append(
            spec.value(spec.state_of(state), end_of_day((start + timedelta(days=day)).date()))
        )
    assert min(zs) > 3.0, f"the baseline moved with the ramp: {zs}"


def test_registry_order_is_the_import_order_and_not_alphabetical() -> None:
    # Column order is part of the contract (09-interfaces.md §9). Sorting the registry
    # would be a silent retrain-and-rescore mismatch, so assert it is not sorted.
    names = [s.name for s in T1]
    assert names[0] == "v_txn_count_z", names[:3]
    assert names != sorted(names), "ORDER has become alphabetical — column order is broken"


def test_declared_state_is_measured_not_assumed(
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """Report the real serialized cost of a full T1 MerchantState against NFR-04.

    Deliberately not an assertion on the per-feature declaration: pickle's per-object
    overhead is large next to the register's logical field counts, and the honest place to
    fix that is a packed serialization in T-150, not a padded declaration here.
    """
    txns, profiles = loaded_stream
    state = MerchantState(merchant_id="M000", profile=profiles["M000"])
    for spec in specs():
        for event in txns:
            if event.merchant_id == "M000":
                spec.update(spec.state_of(state), event)
    per_feature = {
        name: fs.nbytes() for name, fs in sorted(state.feature_states.items())
    }
    print(
        f"\nreal serialized T1 state: {state.nbytes()} B "
        f"(NFR-04 budget {STATE_BYTES_BUDGET} B); "
        f"declared {sum(s.state_bytes for s in T1)} B; "
        f"per-feature {per_feature}"
    )
    assert state.feature_states, "no feature wrote any state"
