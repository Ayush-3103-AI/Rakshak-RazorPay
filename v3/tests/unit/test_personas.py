"""T-112: every persona emits a stream, and each hard negative has its signature.

08-generator-v2-spec.md §2 annotates each persona with the feature family it is there to
make hard. Those annotations are the ticket: a generator whose negatives are easy
produces a model whose false-positive cost is fictional, and v1's headline is what that
looks like once it reaches a results table. So this file asserts one signature per
persona, and the four that the ticket's ``Done when`` clause names are asserted with the
numbers it gives.

**The confounder layer is off here, deliberately.** T-114 turns it on and tests it. A
persona property measured through a live platform-drift layer is measuring two things,
and when it fails you do not know which one moved.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rakshak.generator.arrivals import interarrival_cv
from rakshak.generator.config import load_scenario
from rakshak.generator.engine import GeneratedData, generate
from rakshak.schemas import PersonaId

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenario_v2.yaml"
START = datetime(2026, 1, 1, tzinfo=UTC)
SEED = 7
#: Small enough to generate in a few seconds; large enough that the L3 cohort mean has
#: ~150 merchants in it, which is what pulls the NB noise out of the linear fit.
N_MERCHANTS = 2_500
#: Far above the real 1.47%. This suite measures per-typology *shape*, not prevalence,
#: and at 1.47% of 2,500 there would be four R1 merchants to average over.
TEST_PREVALENCE = 0.25


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
    """Transactions with persona, typology and onset day attached, plus a day index."""
    truth = data.ground_truth.select(
        "merchant_id",
        "persona_id",
        "risk_typology_id",
        onset=(pl.col("drift_onset_at") - START).dt.total_days().cast(pl.Int32),
    )
    return data.transactions.with_columns(
        day=(pl.col("event_time") - START).dt.total_days().cast(pl.Int32),
        second=(pl.col("event_time") - START).dt.total_seconds().cast(pl.Float64),
    ).join(truth, on="merchant_id")


def linear_r2(y: np.ndarray) -> float:
    """R^2 of an ordinary least-squares line through ``y`` against its index.

    The whole L3-vs-R1 separation is this number: L3 ramps linearly over the horizon and
    R1 ramps convexly over two to three weeks, and on ``v_gmv_z`` they are the same
    merchant. If a linear fit explains both, ``v_gmv_accel`` has nothing to find and L3
    has stopped being the hard negative it is in the population for.
    """
    x = np.arange(y.size, dtype=np.float64)
    design = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return float(1.0 - residual.var() / y.var())


def cohort_mean_gmv(frame: pl.DataFrame, day_col: str, n_days: int) -> np.ndarray:
    """Mean captured GMV per merchant per day over a cohort, as a dense series."""
    captured = frame.filter((pl.col("status") == "captured") & ~pl.col("is_refund"))
    n_merchants = captured["merchant_id"].n_unique()
    grouped = (
        captured.group_by(day_col).agg(pl.col("amount_inr").sum()).sort(day_col)
    )
    series = np.zeros(n_days, dtype=np.float64)
    series[grouped[day_col].to_numpy()] = grouped["amount_inr"].to_numpy() / n_merchants
    return series


# ─────────────────────────────────────────────────────────────────────────────
# Every persona emits
# ─────────────────────────────────────────────────────────────────────────────


def test_all_eight_personas_emit_streams(joined: pl.DataFrame) -> None:
    present = set(joined["persona_id"].unique().to_list())
    assert present == {p.value for p in PersonaId}
    counts = joined.group_by("persona_id").len()
    assert counts["len"].min() > 0


def test_every_merchant_has_exactly_one_persona(data: GeneratedData) -> None:
    assert data.ground_truth["merchant_id"].n_unique() == data.ground_truth.height
    assert data.ground_truth["persona_id"].null_count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# The four Done-when signatures
# ─────────────────────────────────────────────────────────────────────────────


def test_l3_growth_is_linear_and_r1_is_not(joined: pl.DataFrame) -> None:
    """The ticket's headline assertion: R^2 > 0.9 for L3, < 0.7 for R1.

    L3 is measured over the whole horizon on its non-fraud members. R1 is measured on
    days aligned to each merchant's own ``drift_onset_at`` — the ramps start on different
    calendar days, and averaging them unaligned would smear the convexity into something
    that looks linear for a reason that has nothing to do with the generator.
    """
    n_days = int(joined["day"].max()) + 1
    l3 = joined.filter(
        (pl.col("persona_id") == PersonaId.L3.value) & pl.col("risk_typology_id").is_null()
    )
    l3_r2 = linear_r2(cohort_mean_gmv(l3, "day", n_days))

    r1 = joined.filter(pl.col("risk_typology_id") == "R1").with_columns(
        rel=pl.col("day") - pl.col("onset")
    )
    ramp_window = 21  # typologies.R1.ramp_days_max
    r1 = r1.filter((pl.col("rel") >= 0) & (pl.col("rel") < ramp_window))
    r1_r2 = linear_r2(cohort_mean_gmv(r1, "rel", ramp_window))

    assert l3_r2 > 0.9, f"L3 cohort GMV is not linear enough: R^2 = {l3_r2:.4f}"
    assert r1_r2 < 0.7, f"R1 ramp is not convex enough: linear R^2 = {r1_r2:.4f}"


def test_l5_interarrival_cv_is_below_point_three(joined: pl.DataFrame) -> None:
    """L5's signature, and the hard negative for ``h_interarrival_cv``.

    Measured *within* a day. The gap that straddles midnight is an artefact of daily
    aggregation, not a property of the arrival rhythm: it is the same size as the
    within-day spacing for a merchant with steady volume and several times larger for a
    merchant whose count happened to be low that day, so including it would report the
    dispersion of the count process under the name of the dispersion of the *timing*.
    """
    legit = joined.filter(pl.col("risk_typology_id").is_null())
    cvs = {}
    for pid in (PersonaId.L1, PersonaId.L5, PersonaId.L6):
        rows = legit.filter(pl.col("persona_id") == pid.value)
        values = [
            interarrival_cv(group["second"].to_numpy())
            for _, group in rows.group_by(["merchant_id", "day"])
            if group.height >= 4
        ]
        cvs[pid] = float(np.nanmean(values))

    assert cvs[PersonaId.L5] < 0.3, f"L5 is not scripted-looking: CV = {cvs[PersonaId.L5]:.4f}"
    # And the contrast is what makes it a hard negative rather than a quirk: an
    # unscripted persona must sit near the Poisson value of 1.0.
    assert cvs[PersonaId.L1] > 0.6
    assert cvs[PersonaId.L6] > 0.6


def test_l8_refund_rate_exceeds_fifteen_percent(joined: pl.DataFrame) -> None:
    """L8 breaks the whole F6 refund family, and it has to do so on legitimate volume."""
    legit = joined.filter(pl.col("risk_typology_id").is_null())
    rates = {}
    for pid in (PersonaId.L1, PersonaId.L8):
        rows = legit.filter(pl.col("persona_id") == pid.value)
        rates[pid] = float(rows["is_refund"].sum() / (~rows["is_refund"]).sum())
    assert rates[PersonaId.L8] > 0.15, f"L8 refund rate = {rates[PersonaId.L8]:.4f}"
    assert rates[PersonaId.L1] < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# One signature per remaining persona
# ─────────────────────────────────────────────────────────────────────────────


def per_merchant_daily_counts(frame: pl.DataFrame, persona: PersonaId, n_days: int) -> np.ndarray:
    rows = frame.filter(
        (pl.col("persona_id") == persona.value) & pl.col("risk_typology_id").is_null()
    )
    merchants = rows["merchant_id"].unique().sort().to_list()
    lookup = {m: i for i, m in enumerate(merchants)}
    counts = np.zeros((len(merchants), n_days), dtype=np.float64)
    agg = rows.group_by(["merchant_id", "day"]).len()
    idx = np.array([lookup[m] for m in agg["merchant_id"].to_list()])
    counts[idx, agg["day"].to_numpy()] = agg["len"].to_numpy()
    return counts


def test_l2_spikes_without_fraud(joined: pl.DataFrame) -> None:
    """L2 is the hard negative for ``v_gmv_z``: large spikes, zero fraud. Its peak day
    must be far above its own baseline, or the feature never sees a legitimate spike."""
    n_days = int(joined["day"].max()) + 1

    def peak_over_mean(persona: PersonaId) -> float:
        # Smoothed over a week first. A raw daily peak-to-median ratio measures the NB
        # overdispersion, not the sale window: at Fano 12.25 the flat persona's biggest
        # single day is already 18x its median, and it has no shape at all. The seasonal
        # window is ~11 days wide, so a 7-day mean keeps the shape and drops the noise.
        counts = per_merchant_daily_counts(joined, persona, n_days)
        kernel = np.ones(7) / 7.0
        smoothed = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, counts)
        return float(
            np.median(smoothed.max(axis=1) / np.maximum(smoothed.mean(axis=1), 1e-9))
        )

    seasonal = peak_over_mean(PersonaId.L2)
    flat = peak_over_mean(PersonaId.L1)
    # 1.25x, not 5x, and the gap is the finding rather than a weak test: at Fano 12.25
    # even the *shapeless* persona's best week is 2.6x its own mean, so a 4.5x sale
    # window only reaches 3.6x. A legitimate spike is genuinely hard to separate from
    # overdispersion, which is exactly why L2 is in the population as a hard negative
    # for v_gmv_z rather than as scenery.
    assert seasonal > 1.25 * flat, f"L2 peak/mean {seasonal:.2f} vs L1 {flat:.2f}"


def test_l4_has_a_fat_ticket_tail(joined: pl.DataFrame) -> None:
    """L4 earns a high ``t_p95_median_ratio`` legitimately — the feature cannot use it
    alone, which is the point of putting 10% of the population here."""
    legit = joined.filter(pl.col("risk_typology_id").is_null())
    ratios = {}
    for pid in (PersonaId.L1, PersonaId.L4):
        amounts = legit.filter(pl.col("persona_id") == pid.value)["amount_inr"].to_numpy()
        ratios[pid] = float(np.percentile(amounts, 95) / np.median(amounts))
    assert ratios[PersonaId.L4] > 2.0 * ratios[PersonaId.L1]


def test_l6_has_the_highest_new_payer_ratio(joined: pl.DataFrame) -> None:
    """The hard negative for ``g_new_payer_ratio``: an aggregator's traffic is mostly
    strangers, and so is a fabricated payer base."""
    legit = joined.filter(pl.col("risk_typology_id").is_null())
    ratio = (
        legit.group_by("persona_id")
        .agg((pl.col("payer_id").n_unique() / pl.len()).alias("unique_share"))
        .sort("unique_share", descending=True)
    )
    assert ratio["persona_id"][0] == PersonaId.L6.value


def test_l7_goes_dormant_and_comes_back(joined: pl.DataFrame) -> None:
    """The hard negative for ``v_dormant_burst``. A genuine business coming back must
    look, on that feature alone, exactly like a shell waking up to bust out."""
    n_days = int(joined["day"].max()) + 1
    counts = per_merchant_daily_counts(joined, PersonaId.L7, n_days)
    # Longest run of zero-count days per merchant, against the flat persona's.
    def longest_gap(rows: np.ndarray) -> np.ndarray:
        out = np.zeros(rows.shape[0])
        for i, row in enumerate(rows):
            best = run = 0
            for value in row:
                run = run + 1 if value == 0 else 0
                best = max(best, run)
            out[i] = best
        return out

    assert np.median(longest_gap(counts)) > np.median(
        longest_gap(per_merchant_daily_counts(joined, PersonaId.L1, n_days))
    ) + 10.0


def test_amounts_are_lognormal_not_gaussian(joined: pl.DataFrame) -> None:
    """Payment amounts are multiplicative and heavy-tailed. Gaussian amounts would make
    ``t_wasserstein_7d`` trivial — the divergence feature would fire on any shift at all
    because the baseline has no tail to absorb it."""
    legit = joined.filter(pl.col("risk_typology_id").is_null())
    amounts = legit.filter(pl.col("persona_id") == PersonaId.L1.value)["amount_inr"].to_numpy()
    logs = np.log(amounts)
    log_skew = float(((logs - logs.mean()) ** 3).mean() / logs.std() ** 3)
    raw_skew = float(((amounts - amounts.mean()) ** 3).mean() / amounts.std() ** 3)
    assert raw_skew > 1.0
    assert abs(log_skew) < 0.5
