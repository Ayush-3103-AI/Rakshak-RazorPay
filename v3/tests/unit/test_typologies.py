"""T-113: all nine typologies emit, carry an onset and a loss, and keep their signatures.

08-generator-v2-spec.md §3. Fraud is never one undifferentiated class here — per-typology
recall is a required output precisely so that a rung which wins on the average while
missing R2 entirely cannot hide behind that average. That only works if each typology
actually has the behaviour its row in the spec table claims, which is what this file
checks.

**The R2 assertion is the one that matters.** R2 is the slow-ramp bust-out, the typology
v1 failed on, and its defining property is that no single week looks alarming. If R2's
week-over-week change ever exceeds one sigma of its own baseline, R2 has quietly become a
second R6 and the hardest case in the population has been deleted — while every aggregate
metric improves, which is what makes it dangerous.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rakshak.generator.config import load_scenario
from rakshak.generator.engine import GeneratedData, generate
from rakshak.generator.typologies import assign_typologies, ramp_progress
from rakshak.schemas import TypologyId

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenario_v2.yaml"
START = datetime(2026, 1, 1, tzinfo=UTC)
SEED = 11
N_MERCHANTS = 2_500
#: Far above the real 1.47%: this suite measures per-typology shape, and R7 at 7% of
#: 1.47% of 2,500 would be a single merchant.
TEST_PREVALENCE = 0.30


@pytest.fixture(scope="module")
def data() -> GeneratedData:
    config = load_scenario(CONFIG_PATH)
    config = dataclasses.replace(
        config,
        population=dataclasses.replace(
            config.population, n_merchants=N_MERCHANTS, prevalence=TEST_PREVALENCE
        ),
        confounders=dataclasses.replace(config.confounders, enabled=False),
    )
    return generate(config, np.random.default_rng(SEED))


@pytest.fixture(scope="module")
def joined(data: GeneratedData) -> pl.DataFrame:
    truth = data.ground_truth.select(
        "merchant_id",
        "risk_typology_id",
        onset=(pl.col("drift_onset_at") - START).dt.total_days().cast(pl.Int32),
    )
    return (
        data.transactions.with_columns(
            day=(pl.col("event_time") - START).dt.total_days().cast(pl.Int32)
        )
        .join(truth, on="merchant_id")
        .with_columns(rel=pl.col("day") - pl.col("onset"))
    )


def post(frame: pl.DataFrame, typology: TypologyId) -> pl.DataFrame:
    """Post-onset rows for one typology, at or past full ramp is *not* required — the
    signature has to be visible while the ramp is still running or the typology is only
    detectable at the moment it stops mattering."""
    return frame.filter((pl.col("risk_typology_id") == typology.value) & (pl.col("rel") >= 0))


def legit(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("risk_typology_id").is_null())


# ─────────────────────────────────────────────────────────────────────────────
# Done-when clause
# ─────────────────────────────────────────────────────────────────────────────


def test_all_nine_typologies_emit(joined: pl.DataFrame) -> None:
    present = set(joined["risk_typology_id"].drop_nulls().unique().to_list())
    assert present == {t.value for t in TypologyId}


def test_every_fraud_merchant_has_onset_after_onboarding_and_a_positive_loss(
    data: GeneratedData,
) -> None:
    """Both legs of the ticket's second clause, plus the reason for the second one:
    ``true_loss_amount_inr`` is the weight in the oracle knapsack, and a zero there
    lowers the ceiling and flatters every rung measured against it."""
    fraud = data.ground_truth.filter(pl.col("risk_typology_id").is_not_null()).join(
        data.profiles.select("merchant_id", "onboarded_at"), on="merchant_id"
    )
    assert fraud.height > 0
    assert fraud.filter(pl.col("drift_onset_at") <= pl.col("onboarded_at")).height == 0
    assert float(fraud["true_loss_amount_inr"].min()) > 0.0
    # And the contrapositive, which schemas.GroundTruth also enforces per row.
    clean = data.ground_truth.filter(pl.col("risk_typology_id").is_null())
    assert float(clean["true_loss_amount_inr"].abs().max()) == 0.0
    assert clean["drift_onset_at"].null_count() == clean.height


def _weekly_gmv(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.filter((pl.col("status") == "captured") & ~pl.col("is_refund"))
        .with_columns(week=pl.col("day") // 7, rel_week=pl.col("rel") // 7)
        .group_by(["merchant_id", "week"])
        .agg(
            pl.col("amount_inr").sum().alias("gmv"),
            pl.col("rel_week").min().alias("rel_week"),
        )
        .sort(["merchant_id", "week"])
    )


N_POST_WEEKS = 9


def _normalised_weeks(weekly: pl.DataFrame) -> tuple[np.ndarray, list[float]]:
    """Each merchant's weekly GMV divided by its own pre-onset mean, keyed by week
    relative to onset, plus the per-merchant baseline coefficient of variation.

    **Scale-free on purpose, and this is where the first attempt at this test was wrong.**
    A raw cohort mean is dominated by the L4 and L8 merchants, whose tickets are two
    orders of magnitude above L1's, while the median baseline sigma describes a typical
    small merchant — so the ratio of the two measured the persona mix and not the ramp,
    and reported R2 at 6.8 sigma when its actual systematic step is a third of one.
    """
    tracks: list[np.ndarray] = []
    cvs: list[float] = []
    for (_merchant,), rows in weekly.group_by(["merchant_id"]):
        pre = rows.filter(pl.col("rel_week") < 0)["gmv"].to_numpy()
        if pre.size < 3:
            continue
        mean = float(pre.mean())
        if mean <= 0.0:
            continue
        cvs.append(float(pre.std(ddof=1) / mean))
        after = rows.filter(
            (pl.col("rel_week") >= 0) & (pl.col("rel_week") < N_POST_WEEKS)
        ).sort("rel_week")
        # Only merchants observed across the whole post-onset window. Otherwise the
        # cohort composition changes from week to week and the resulting "step" is
        # merchants entering and leaving the average, not the ramp.
        if after.height != N_POST_WEEKS:
            continue
        tracks.append(after["gmv"].to_numpy() / mean)
    if not tracks:
        return np.empty((0, N_POST_WEEKS)), cvs
    return np.vstack(tracks), cvs


def test_r2_weekly_change_stays_under_one_baseline_sigma(joined: pl.DataFrame) -> None:
    """**The key assertion of this ticket.**

    R2's *systematic* weekly step, measured on the onset-aligned cohort mean of each
    merchant's weekly GMV normalised by its own pre-onset mean, against the median
    per-merchant baseline coefficient of variation. Both sides are scale-free, so the
    comparison is about the ramp rather than about which personas happened to turn.

    The cohort mean, not a single merchant's series, and that distinction is the whole
    subtlety here. At Fano 12.25 one R2 merchant's weekly GMV swings by several baseline
    sigma from arrival noise alone — and so does a legitimate merchant's, which is the
    companion test below and the operational form of the claim. Asserting a per-merchant
    bound would be asserting that the arrival process is quieter than v1 measured it to
    be, and quiet arrivals are the misspecification v2 exists to fix.
    """
    r2 = joined.filter(pl.col("risk_typology_id") == TypologyId.R2.value)
    tracks, cvs = _normalised_weeks(_weekly_gmv(r2))
    assert len(cvs) >= 5, "not enough R2 merchants with a usable pre-onset baseline"
    sigma = float(np.median(cvs))

    assert tracks.shape[0] >= 5, "not enough R2 merchants observed across the whole ramp"
    # Median across merchants, not mean: each track is divided by its own noisy pre-onset
    # mean, and at Fano 12.25 a merchant whose baseline weeks came in low produces a
    # ratio large enough to drag a cohort *mean* around on its own.
    series = np.median(tracks, axis=0)

    # The *fitted* weekly step, not the largest observed one. Even a cohort median over
    # tens of merchants carries enough residual arrival noise that its biggest single
    # week-to-week move is mostly sampling; the systematic increment R2 introduces is the
    # slope. The observed maximum is reported alongside so the gap is visible rather than
    # hidden by the choice of estimator.
    weeks = np.arange(series.size, dtype=np.float64)
    slope, _ = np.polyfit(weeks, series, 1)
    step = abs(float(slope))
    observed_max = float(np.abs(np.diff(series)).max())

    assert step < sigma, (
        f"R2's systematic weekly step is {step:.3f} of its own baseline against a baseline "
        f"CV of {sigma:.3f} ({step / sigma:.2f} sigma; largest observed single-week move "
        f"{observed_max:.3f}). Above 1.0 it is no longer a slow ramp and the hardest "
        f"typology in the population has become a second R6."
    )


def test_r2_is_indistinguishable_from_a_legitimate_merchant_week_to_week(
    joined: pl.DataFrame,
) -> None:
    """The operational form of the R2 claim, and the reason R2 is hard.

    A weekly-change monitor watching R2 sees very nearly the distribution it sees on a
    merchant doing nothing wrong. Changes are compared as log ratios, because that is
    what a z-score against a *trailing* baseline actually measures: as R2's level grows,
    so does its noise, and an absolute threshold anchored to the pre-onset level would
    fire on the growth rather than on the step.
    """

    def relative_steps(frame: pl.DataFrame) -> np.ndarray:
        weekly = _weekly_gmv(frame)
        out = []
        for (_merchant,), rows in weekly.group_by(["merchant_id"]):
            after = rows.filter(pl.col("rel_week") >= 0)["gmv"].to_numpy()
            after = after[after > 0]
            if after.size >= 3:
                out.append(float(np.abs(np.diff(np.log(after))).max()))
        return np.array(out)

    r2_steps = relative_steps(joined.filter(pl.col("risk_typology_id") == TypologyId.R2.value))
    # The same measurement on legitimate merchants, given a fake onset at the median R2
    # onset day so the pre/post split is comparable.
    control = legit(joined).with_columns(rel=pl.col("day") - 55)
    control_steps = relative_steps(control)

    assert r2_steps.size >= 5 and control_steps.size >= 20
    ratio = float(np.median(r2_steps) / np.median(control_steps))
    assert ratio < 1.5, (
        f"R2's weekly-change distribution is {ratio:.2f}x a legitimate merchant's "
        f"(median {np.median(r2_steps):.3f} vs {np.median(control_steps):.3f} in log "
        f"points). If a weekly monitor can see R2, R2 has stopped being the hard case."
    )


def test_r2_still_ends_up_materially_larger(joined: pl.DataFrame) -> None:
    """The other half of the R2 property, and the reason the one above is not vacuous: a
    typology that never grows would also pass a weekly-change test."""
    r2 = post(joined, TypologyId.R2)
    late = r2.filter(pl.col("rel") >= 55)
    early = joined.filter(
        (pl.col("risk_typology_id") == TypologyId.R2.value) & (pl.col("rel") < 0)
    )
    per_day_late = late.height / max(late["day"].n_unique(), 1)
    per_day_early = early.height / max(early["day"].n_unique(), 1)
    assert per_day_late > 1.5 * per_day_early


# ─────────────────────────────────────────────────────────────────────────────
# One signature per typology
# ─────────────────────────────────────────────────────────────────────────────


def test_r1_ramps_then_vanishes(joined: pl.DataFrame) -> None:
    r1 = joined.filter(pl.col("risk_typology_id") == TypologyId.R1.value)
    peak = r1.filter((pl.col("rel") >= 10) & (pl.col("rel") < 21)).height / 11
    after = r1.filter((pl.col("rel") >= 30) & (pl.col("rel") < 60)).height / 30
    assert after < 0.2 * peak, f"R1 did not vanish: {after:.1f}/day after vs {peak:.1f} at peak"


def test_r3_is_micro_amount_high_failure_and_bin_concentrated(joined: pl.DataFrame) -> None:
    """R3's three dominant features at once. The BIN concentration is the one that is
    easy to forget and is what ``i_bin_hhi`` exists for: a card-testing host is spending
    someone else's stolen book, and a stolen book is narrow."""
    r3, base = post(joined, TypologyId.R3), legit(joined)
    assert float((r3["amount_inr"] <= 10).mean()) > 0.3
    assert float((r3["status"] == "failed").mean()) > 4 * float(
        (base["status"] == "failed").mean()
    )
    r3_bins_per_txn = r3["bin_hash"].n_unique() / r3.height
    base_bins_per_txn = base["bin_hash"].n_unique() / base.height
    assert r3_bins_per_txn < base_bins_per_txn


def test_r4_shifts_international_and_drifts_out_of_its_declared_mcc(
    joined: pl.DataFrame, data: GeneratedData
) -> None:
    r4 = post(joined, TypologyId.R4)
    base_intl = float(legit(joined)["is_international"].mean())
    assert float(r4["is_international"].mean()) > 4 * base_intl
    declared = data.profiles.select("merchant_id", declared_mcc=pl.col("mcc"))
    off_basket = r4.join(declared, on="merchant_id").filter(pl.col("mcc") != pl.col("declared_mcc"))
    assert off_basket.height / r4.height > 0.1


def test_r5_refunds_hard_to_a_narrow_payer_set(joined: pl.DataFrame) -> None:
    r5, base = post(joined, TypologyId.R5), legit(joined)
    assert float(r5["is_refund"].mean()) > 3 * float(base["is_refund"].mean())
    # Payer concentration: distinct payers per transaction, low means a repeat set.
    assert r5["payer_id"].n_unique() / r5.height < base["payer_id"].n_unique() / base.height


def test_r6_is_abrupt(joined: pl.DataFrame) -> None:
    """R6 is account takeover: the change lands overnight on a long clean history. Its
    ramp is one day, so the day-after/day-before ratio is the whole signature."""
    r6 = joined.filter(pl.col("risk_typology_id") == TypologyId.R6.value)
    before = r6.filter((pl.col("rel") >= -14) & (pl.col("rel") < 0)).height / 14
    after = r6.filter((pl.col("rel") >= 1) & (pl.col("rel") <= 14)).height / 14
    assert after > 3.0 * before, f"R6 was not abrupt: {before:.1f}/day -> {after:.1f}/day"


def test_r6_flips_its_hour_of_day(joined: pl.DataFrame) -> None:
    r6 = joined.filter(pl.col("risk_typology_id") == TypologyId.R6.value).with_columns(
        hour=pl.col("event_time").dt.hour()
    )
    before = r6.filter(pl.col("rel") < 0)["hour"].mean()
    after = r6.filter(pl.col("rel") >= 1)["hour"].mean()
    assert abs(float(after) - float(before)) > 2.0


def test_r7_shares_payers_across_the_ring(joined: pl.DataFrame) -> None:
    """R7's whole detectability. Individually unremarkable, jointly a ring — which is why
    it needs T3 cross-merchant features and why its per-typology recall is expected to be
    poor and is reported separately rather than averaged away."""
    r7 = post(joined, TypologyId.R7)
    shared = (
        r7.group_by("payer_id").agg(pl.col("merchant_id").n_unique().alias("merchants"))
    )
    base_shared = (
        legit(joined).group_by("payer_id").agg(pl.col("merchant_id").n_unique().alias("merchants"))
    )
    assert float((shared["merchants"] > 1).mean()) > 10 * float(
        (base_shared["merchants"] > 1).mean()
    )


def test_r8_is_low_ticket_and_refund_heavy(joined: pl.DataFrame) -> None:
    r8, base = post(joined, TypologyId.R8), legit(joined)
    assert float(r8["amount_inr"].median()) < float(base["amount_inr"].median())
    assert float(r8["is_refund"].mean()) > 2 * float(base["is_refund"].mean())


def test_r9_drifts_its_basket_the_furthest(joined: pl.DataFrame, data: GeneratedData) -> None:
    declared = data.profiles.select("merchant_id", declared_mcc=pl.col("mcc"))

    def off_rate(typology: TypologyId) -> float:
        rows = post(joined, typology).join(declared, on="merchant_id")
        if rows.height == 0:
            return 0.0
        return float((rows["mcc"] != rows["declared_mcc"]).mean())

    assert off_rate(TypologyId.R9) > off_rate(TypologyId.R4)
    assert off_rate(TypologyId.R1) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# The assignment layer itself
# ─────────────────────────────────────────────────────────────────────────────


def test_prevalence_is_exact_not_binomial(rng: np.random.Generator) -> None:
    """Prevalence is the number v1 got wrong. It is the last thing that should be left to
    sampling noise, so the positive count is round(n * prevalence) exactly."""
    config = load_scenario(CONFIG_PATH)
    assignment = assign_typologies(rng, 10_000, 0.0147, config.typologies)
    assert assignment.n_fraud == 147


def test_prevalence_zero_produces_no_fraud(rng: np.random.Generator) -> None:
    """Gate G5 runs here. A single fraud merchant would make every alert in that run
    ambiguous."""
    config = load_scenario(CONFIG_PATH)
    assignment = assign_typologies(rng, 5_000, 0.0, config.typologies)
    assert assignment.n_fraud == 0
    assert float(ramp_progress(assignment, config.population.n_days).max()) == 0.0


def test_progress_is_zero_before_onset_and_one_after_the_ramp(
    rng: np.random.Generator,
) -> None:
    config = load_scenario(CONFIG_PATH)
    n_days = config.population.n_days
    assignment = assign_typologies(rng, 2_000, 0.5, config.typologies)
    progress = ramp_progress(assignment, n_days)
    fraud = np.flatnonzero(assignment.is_fraud)
    for m in fraud[:50]:
        onset, ramp = int(assignment.onset_day[m]), int(assignment.ramp_days[m])
        assert progress[m, :onset].max(initial=0.0) == 0.0
        assert progress[m, onset] == pytest.approx(0.0)
        if onset + ramp < n_days:
            assert progress[m, onset + ramp] == pytest.approx(1.0)
    assert progress[~assignment.is_fraud].max() == 0.0
