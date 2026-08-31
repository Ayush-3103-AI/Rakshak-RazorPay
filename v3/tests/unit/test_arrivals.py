"""T-111: the arrival process is overdispersed to a *target*, not merely to "more than
Poisson", and the Hawkes overlay actually clusters events.

FR-003's acceptance clause is `GIVEN target_fano=12.25 THEN the realised Fano factor over
the population is 12.25 +/- 1.0`. The test runs it at three targets — 1.0 (Poisson, the
v1 process), 5.0, and 12.25 (the v1 measurement) — because a calibration that only holds
at the number it was tuned on is a constant, not a calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.generator.arrivals import (
    SECONDS_PER_DAY,
    dispersion_for_fano,
    fano_factor,
    hawkes_overlay,
    interarrival_cv,
    nb_daily_counts,
    regular_day_times,
    within_day_times,
)

TARGETS = [1.0, 5.0, 12.25]
TOLERANCE = 1.0  # FR-003
N_MERCHANTS = 300
N_DAYS = 730  # two simulated years: enough that a per-merchant Fano estimate is stable


def heterogeneous_lambda(rng: np.random.Generator) -> np.ndarray:
    """Intensities spanning the real population — L4 at ~2/day up to L6 at ~45/day.

    Homogeneous lambdas would let a bug that ties the dispersion to a fixed rate pass.
    """
    return np.asarray(rng.uniform(2.0, 45.0, size=N_MERCHANTS)[:, None] * np.ones(N_DAYS))


@pytest.mark.parametrize("target", TARGETS)
def test_realised_fano_hits_target(target: float, rng: np.random.Generator) -> None:
    lam = heterogeneous_lambda(rng)
    counts = nb_daily_counts(rng, lam, target)
    realised = fano_factor(counts)
    assert abs(realised - target) <= TOLERANCE, (
        f"target_fano={target} produced a realised Fano of {realised:.3f} "
        f"(tolerance +/- {TOLERANCE})"
    )


def test_mean_is_preserved_while_variance_is_inflated(rng: np.random.Generator) -> None:
    """Overdispersion must move the variance and leave the mean alone. A process that
    also shifts the mean is not more variable, it is a different merchant."""
    lam = np.full((N_MERCHANTS, N_DAYS), 20.0)
    poisson = nb_daily_counts(rng, lam, 1.0)
    overdispersed = nb_daily_counts(rng, lam, 12.25)
    assert poisson.mean() == pytest.approx(20.0, rel=0.02)
    assert overdispersed.mean() == pytest.approx(20.0, rel=0.02)
    assert overdispersed.var() > 8.0 * poisson.var()


def test_fano_is_flat_across_intensity(rng: np.random.Generator) -> None:
    """r scales with lambda, so a quiet merchant is as overdispersed as a busy one. If
    this drifts, the population Fano becomes an artefact of the persona mix."""
    quiet = nb_daily_counts(rng, np.full((N_MERCHANTS, N_DAYS), 2.0), 12.25)
    busy = nb_daily_counts(rng, np.full((N_MERCHANTS, N_DAYS), 45.0), 12.25)
    assert abs(fano_factor(quiet) - fano_factor(busy)) <= TOLERANCE


def test_dispersion_formula_matches_the_spec(rng: np.random.Generator) -> None:
    lam = np.array([1.0, 10.0, 100.0])
    r = dispersion_for_fano(lam, 12.25)
    np.testing.assert_allclose(1.0 + lam / r, 12.25)
    # Fano <= 1 has no NB parameterisation; the caller falls back to Poisson.
    assert np.all(dispersion_for_fano(lam, 1.0) == 0.0)


def test_zero_and_negative_intensity_yield_no_arrivals(rng: np.random.Generator) -> None:
    """A merchant that has not onboarded, or is inside an L7 dormancy, has lambda 0. That
    must be zero counts, not a numpy domain error four modules downstream."""
    lam = np.array([0.0, -3.0, 5.0])
    counts = nb_daily_counts(rng, lam, 12.25)
    assert counts[0] == 0 and counts[1] == 0
    assert counts.dtype == np.int64


def test_under_dispersed_target_is_rejected(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="under-dispersed"):
        nb_daily_counts(rng, np.array([5.0]), 0.5)


def test_fano_ignores_between_merchant_spread(rng: np.random.Generator) -> None:
    """Two Poisson merchants at wildly different rates are still Fano 1. This is the
    exact confusion the per-merchant definition exists to prevent."""
    lam = np.concatenate(
        [np.full((50, N_DAYS), 2.0), np.full((50, N_DAYS), 45.0)], axis=0
    )
    assert fano_factor(nb_daily_counts(rng, lam, 1.0)) == pytest.approx(1.0, abs=0.15)


# ─────────────────────────────────────────────────────────────────────────────
# Hawkes overlay
# ─────────────────────────────────────────────────────────────────────────────


def bin_counts(times_s: np.ndarray, bin_s: float, horizon_s: float) -> np.ndarray:
    edges = np.arange(0.0, horizon_s + bin_s, bin_s)
    return np.histogram(times_s, bins=edges)[0].astype(np.float64)


def lag1_autocorr(series: np.ndarray) -> float:
    centred = series - series.mean()
    denom = float((centred**2).sum())
    if denom == 0.0:
        return 0.0
    return float((centred[:-1] * centred[1:]).sum() / denom)


def test_hawkes_raises_short_lag_autocorrelation(rng: np.random.Generator) -> None:
    """The Done-when clause. Bursts are the point: without self-excitation, "card
    testing" is just a higher rate and f_retry_burst_rate reads noise."""
    horizon = SECONDS_PER_DAY
    hour_weights = np.ones(24)
    plain_ac, hawkes_ac = [], []
    for _ in range(30):
        base = within_day_times(rng, 300, hour_weights)
        excited = hawkes_overlay(
            rng,
            base,
            excitation=0.35,
            decay_minutes=3.0,
            window_minutes=10.0,
            max_generations=40,
        )
        plain_ac.append(lag1_autocorr(bin_counts(base, 600.0, horizon)))
        hawkes_ac.append(lag1_autocorr(bin_counts(excited, 600.0, horizon)))
    assert np.mean(hawkes_ac) > np.mean(plain_ac) + 0.05, (
        f"Hawkes overlay did not cluster: lag-1 autocorrelation "
        f"{np.mean(hawkes_ac):.4f} vs plain {np.mean(plain_ac):.4f}"
    )


def test_hawkes_adds_events_and_keeps_them_sorted(rng: np.random.Generator) -> None:
    base = within_day_times(rng, 500, np.ones(24))
    excited = hawkes_overlay(
        rng,
        base,
        excitation=0.5,
        decay_minutes=3.0,
        window_minutes=10.0,
        max_generations=40,
    )
    assert excited.size > base.size
    assert np.all(np.diff(excited) >= 0)


def test_hawkes_at_zero_excitation_is_a_no_op(rng: np.random.Generator) -> None:
    base = within_day_times(rng, 100, np.ones(24))
    out = hawkes_overlay(
        rng, base, excitation=0.0, decay_minutes=3.0, window_minutes=10.0, max_generations=40
    )
    np.testing.assert_array_equal(out, base)


def test_explosive_excitation_is_rejected(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="explosive"):
        hawkes_overlay(
            rng,
            np.array([0.0]),
            excitation=1.0,
            decay_minutes=3.0,
            window_minutes=10.0,
            max_generations=40,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Within-day times
# ─────────────────────────────────────────────────────────────────────────────


def test_within_day_times_are_sorted_and_inside_the_day(rng: np.random.Generator) -> None:
    times = within_day_times(rng, 2000, np.ones(24))
    assert np.all(np.diff(times) >= 0)
    assert times.min() >= 0.0 and times.max() < SECONDS_PER_DAY


def test_hour_weights_shape_the_histogram(rng: np.random.Generator) -> None:
    weights = np.zeros(24)
    weights[9:12] = 1.0
    times = within_day_times(rng, 5000, weights)
    hours = (times // 3600).astype(int)
    assert set(np.unique(hours).tolist()) <= {9, 10, 11}


def test_regular_arrivals_have_low_interarrival_cv(rng: np.random.Generator) -> None:
    """L5's signature. Poisson-shaped arrivals sit near CV 1.0; the subscription persona
    must sit well below it or it is not a hard negative for h_interarrival_cv."""
    regular = regular_day_times(rng, 60, jitter_s=120.0)
    irregular = within_day_times(rng, 60, np.ones(24))
    assert interarrival_cv(regular) < 0.3
    assert interarrival_cv(irregular) > 0.6


def test_interarrival_cv_is_nan_when_undefined() -> None:
    assert np.isnan(interarrival_cv(np.array([1.0, 2.0])))
