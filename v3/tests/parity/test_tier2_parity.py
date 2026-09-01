"""T-122: Tier-2 divergence features agree online and offline, to 1e-9.

Same harness as T-120, plus two things that suite could not do.

The first is variety. `synthetic_stream` emits one instrument, one decline code and one
device, so `i_mix_jsd` and `f_decline_entropy` see a single bucket in it and their parity is
real but their *range* is not. T-112 and T-114 had landed by the time this ticket ran, so
the last test in this file replays real generator output — eight personas, six confounders,
seven instruments — through **every registered feature, T1 and T2**, both ways. That is the
"re-run the tier suites against the real generator" note T-120's logbook left behind, paid
off in the same sprint rather than deferred.

The second is a sanity floor. A divergence that is always zero passes parity perfectly, so
each feature also has to demonstrate it can be non-zero for the right reason.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import polars as pl
import pytest
from parity_harness import assert_parity, end_of_day, synthetic_stream

from rakshak.features import registry, tier1, tier2
from rakshak.features.spec import PARITY_TOLERANCE, FeatureSpec
from rakshak.features.state import STATE_BYTES_BUDGET, MerchantState
from rakshak.schemas import MerchantProfile, Tier, Transaction

TEST_WARMUP = 5

T2 = registry.of_tier(Tier.T2)


def specs() -> list[FeatureSpec]:
    return [type(spec)(warmup_days=TEST_WARMUP) for spec in T2]


@pytest.fixture
def loaded_stream(
    rng: np.random.Generator,
) -> tuple[list[Transaction], dict[str, MerchantProfile]]:
    txns, profiles = synthetic_stream(rng, days=40)
    tier1.load_profiles(profiles)
    return txns, profiles


@pytest.mark.parametrize("spec", specs(), ids=lambda s: s.name)
def test_tier2_feature_agrees_online_and_offline(
    spec: FeatureSpec,
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    txns, profiles = loaded_stream
    assert_parity(spec, txns, profiles)


@pytest.mark.parametrize("spec", specs(), ids=lambda s: s.name)
def test_tier2_feature_agrees_at_the_production_warmup(
    spec: FeatureSpec,
    rng: np.random.Generator,
) -> None:
    txns, profiles = synthetic_stream(rng, days=90)
    tier1.load_profiles(profiles)
    assert_parity(type(spec)(warmup_days=tier1.WARMUP_DAYS), txns, profiles)


def test_the_declared_t2_state_fits_what_t1_left(loaded_stream: Any) -> None:
    """NFR-04 is a budget for T1 **and** T2 together, and T1 spends most of it.

    07-feature-register.md estimates T1+T2 at ~3.8 KB with 5% headroom, on a T1 figure of
    ~0.9 KB that T-120 measured at 1720 B. That overrun is why four of the register's eight
    T2 rows are cut rather than shrunk — see docs/logbook/T-122.md. The registry enforces
    this at import, so this test is a witness rather than the guard.
    """
    t1 = sum(s.state_bytes for s in registry.of_tier(Tier.T1))
    t2 = sum(s.state_bytes for s in T2)
    print(f"\nT1 {t1} B + T2 {t2} B = {t1 + t2} B of {STATE_BYTES_BUDGET} B (NFR-04)")
    assert t1 + t2 <= STATE_BYTES_BUDGET
    assert len(T2) == 4, [s.name for s in T2]


def test_the_online_histograms_stay_bounded(
    loaded_stream: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """The trailing-window ring must evict, or a 180-day run carries 180 histograms.

    This is the failure mode a state budget exists to catch and the one a parity test never
    would: an unbounded ring is perfectly correct and completely unservable.
    """
    txns, profiles = loaded_stream
    for spec in specs():
        state = MerchantState(merchant_id="M000", profile=profiles["M000"])
        sizes = []
        for i, event in enumerate(t for t in txns if t.merchant_id == "M000"):
            spec.update(spec.state_of(state), event)
            if i % 15 == 0:
                sizes.append(spec.state_of(state).nbytes())
        assert len(spec.state_of(state).recent) <= tier2.T2_WINDOW_DAYS  # type: ignore[attr-defined]
        assert sizes[-1] <= max(sizes[len(sizes) // 2 :]), spec.name


# ── the divergences must be able to be non-zero, for the right reason ─────────


def test_a_divergence_of_a_distribution_with_itself_is_zero() -> None:
    p = np.array([[0.1, 0.2, 0.7]])
    assert tier2.jensen_shannon(p, p)[0] == pytest.approx(0.0, abs=1e-15)
    assert tier2.wasserstein_binned(p, p, 1.0)[0] == pytest.approx(0.0, abs=1e-15)


def test_jensen_shannon_is_bounded_symmetric_and_maximal_on_disjoint_support() -> None:
    # Bounded at 1 bit is what makes it splittable by a tree without a scale hyperparameter;
    # KL would be unbounded and would put a single merchant at +inf.
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert tier2.jensen_shannon(a, b)[0] == pytest.approx(1.0)
    assert tier2.jensen_shannon(b, a)[0] == pytest.approx(1.0)
    mid = np.array([[0.5, 0.5]])
    assert 0.0 < tier2.jensen_shannon(a, mid)[0] < 1.0


def test_shannon_entropy_spans_point_mass_to_uniform() -> None:
    point = np.zeros((1, tier2.DECLINE_BUCKETS))
    point[0, 0] = 1.0
    assert tier2.shannon_entropy(point)[0] == pytest.approx(0.0)
    uniform = np.full((1, tier2.DECLINE_BUCKETS), 1.0 / tier2.DECLINE_BUCKETS)
    assert tier2.shannon_entropy(uniform)[0] == pytest.approx(
        np.log2(tier2.DECLINE_BUCKETS)
    )


def test_a_dormant_merchant_does_not_read_as_a_mix_change() -> None:
    """The trap this file's `comparable` mask exists for.

    An empty window is not free zero divergence. JSD between an empty distribution and a
    concentrated baseline is exactly 0.5 — the mixture halves the baseline's mass — so
    every merchant that stopped trading would park at a constant 0.5 and `i_mix_jsd` would
    be reporting dormancy, which `v_dormant_burst` already reports and reports better.
    """
    empty = np.zeros((1, 4))
    base = np.array([[10.0, 0.0, 0.0, 0.0]])
    raw = tier2.jensen_shannon(tier2._normalise(empty), tier2._normalise(base))[0]
    assert raw == pytest.approx(0.5), "the trap is gone; the mask below may be dead code"

    spec = tier2.InstrumentMixJsd()
    assert not spec.comparable(np.zeros((1, spec.bins)), np.ones((1, spec.bins)))[0]
    assert not spec.comparable(np.ones((1, spec.bins)), np.zeros((1, spec.bins)))[0]
    assert spec.comparable(np.ones((1, spec.bins)), np.ones((1, spec.bins)))[0]


def test_wasserstein_moves_with_the_size_of_the_shift() -> None:
    from rakshak.features.tier1 import HIST_BINS

    near = np.zeros((1, HIST_BINS))
    far = np.zeros((1, HIST_BINS))
    mid = np.zeros((1, HIST_BINS))
    near[0, 5] = 1.0
    mid[0, 8] = 1.0
    far[0, 20] = 1.0
    d_small = tier2.wasserstein_binned(near, mid, 1.0)[0]
    d_big = tier2.wasserstein_binned(near, far, 1.0)[0]
    assert 0.0 < d_small < d_big


# ── the real generator, both runners, every registered feature ────────────────


@pytest.fixture(scope="module")
def generated() -> Any:
    """A small but *varied* population: eight personas, six confounders, seven instruments.

    prevalence is left at the scenario's own 1.47% rather than zeroed — parity has nothing
    to do with labels, and a stream containing typologies exercises wider feature ranges
    than one without.
    """
    from rakshak.generator import config as gen_config
    from rakshak.generator import engine

    cfg = gen_config.load_scenario("configs/scenario_v2.yaml")
    n_days = 45
    pop = dataclasses.replace(cfg.population, n_merchants=60, n_days=n_days)
    # The validator requires `splits.test_end_day == population.n_days - 1`, so shrinking
    # the horizon without shrinking the splits leaves the real 365-day boundaries behind a
    # 45-day window and raises ConfigError before a single event is generated — the fixture
    # errors and these tests do not run at all, which reads as an error rather than as a
    # failure and is easy to skim past. `tests/unit/test_cohort.py` hit exactly this and
    # fixed it there; this copy was left behind. Scale with the window so the miniature
    # keeps the real proportions rather than inventing new ones.
    s = cfg.splits
    scale = n_days / cfg.population.n_days
    splits = dataclasses.replace(
        s,
        train_end_day=round((s.train_end_day + 1) * scale) - 1,
        val_end_day=round((s.val_end_day + 1) * scale) - 1,
        test_end_day=n_days - 1,
    )
    data = engine.generate(
        dataclasses.replace(cfg, population=pop, splits=splits), np.random.default_rng(3)
    )
    profiles = {r["merchant_id"]: MerchantProfile(**r) for r in data.profiles.to_dicts()}
    tier1.load_profiles(profiles)
    return data, profiles


@pytest.mark.slow
def test_every_registered_feature_agrees_on_real_generator_output(generated: Any) -> None:
    """The strongest single assertion in the suite: 28 features, both runners, real data.

    `synthetic_stream` is one instrument, one decline code, one device hash and one MCC
    group. Everything below only *exists* in generator output: instrument mix, international
    and CNP flags, decline-code spread, hour-of-day structure from the arrival process,
    dormancy, and confounder days where the whole population moves at once.

    Epochs are sampled rather than swept — every fifth day, warmup onward — because the
    offline runner recomputes the full prefix for every feature at every epoch and the
    sweep is quadratic in the stream length. The T-120 and T-122 suites above already sweep
    every epoch on the synthetic stream; this one is about breadth of *input*, not of time.
    """
    data, profiles = generated
    frame: pl.DataFrame = data.transactions
    txns_by_merchant = frame.sort(["event_time", "event_id"])
    days = sorted(frame["event_date"].unique().to_list())
    specs_all = [
        type(s)(warmup_days=10) if isinstance(s, tier1.Tier1Spec | tier2.HistogramSpec)
        else type(s)()
        for s in registry.REGISTRY.values()
    ]

    states = {mid: MerchantState(merchant_id=mid, profile=p) for mid, p in profiles.items()}
    rows = txns_by_merchant.to_dicts()
    from rakshak.schemas import Instrument, TxnStatus

    events = [
        Transaction(
            event_id=r["event_id"],
            merchant_id=r["merchant_id"],
            payer_id=r["payer_id"],
            event_time=r["event_time"],
            event_date=r["event_date"],
            amount_inr=r["amount_inr"],
            instrument=Instrument(r["instrument"]),
            is_cnp=r["is_cnp"],
            is_international=r["is_international"],
            bin_hash=r["bin_hash"],
            device_hash=r["device_hash"],
            ip_hash=r["ip_hash"],
            status=TxnStatus(r["status"]),
            decline_code=r["decline_code"],
            mcc=r["mcc"],
            is_refund=r["is_refund"],
            refund_of=r["refund_of"],
        )
        for r in rows
    ]

    checkpoints = set(days[10::5]) | {days[-1]}
    worst = 0.0
    worst_at = ""
    cursor = 0
    for day in days:
        as_of = end_of_day(day)
        while cursor < len(events) and events[cursor].event_time <= as_of:
            event = events[cursor]
            for spec in specs_all:
                spec.update(spec.state_of(states[event.merchant_id]), event)
            cursor += 1
        if day not in checkpoints:
            continue
        prefix = frame.filter(pl.col("event_time") <= as_of).lazy()
        for spec in specs_all:
            out = spec.batch(prefix, as_of)
            lookup = dict(
                zip(out["merchant_id"].to_list(), out[spec.name].to_list(), strict=True)
            )
            for mid, state in states.items():
                got = spec.value(spec.state_of(state), as_of)
                want = lookup.get(mid) or 0.0
                if abs(got - want) > worst:
                    worst, worst_at = abs(got - want), f"{spec.name}/{mid}/{day}"

    print(
        f"\nreal-generator parity over {len(specs_all)} features, "
        f"{len(profiles)} merchants, {len(checkpoints)} epochs: "
        f"max diff {worst:.3e} at {worst_at or 'nowhere'}"
    )
    assert worst <= PARITY_TOLERANCE


@pytest.mark.slow
def test_the_generator_stream_actually_exercises_the_t2_buckets(generated: Any) -> None:
    """Guard against the previous test being green because every bucket was empty.

    If this ever fails, the breadth claim above is false and the T2 features are being
    parity-tested on a single bucket exactly as they were on `synthetic_stream`.
    """
    data, _ = generated
    frame: pl.DataFrame = data.transactions
    assert frame["instrument"].n_unique() >= 4, "instrument mix is degenerate"
    assert frame.filter(pl.col("decline_code").is_not_null())["decline_code"].n_unique() >= 2
    assert frame["event_time"].dt.hour().n_unique() >= 12, "hour-of-day is degenerate"
    assert frame["is_international"].sum() > 0
    assert frame["is_cnp"].n_unique() == 2
