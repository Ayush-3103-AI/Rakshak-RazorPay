"""Rung 8 — the Hawkes/NB fit, its analytic gradient, and the time-rescaling transform.

The first acceptance criterion of GitHub #59 is a *self-consistency* claim and not a
detection claim: "time-rescaled inter-arrival times from a correctly-specified fit pass a KS
test against unit-rate exponential". So the process here is simulated from the branching
construction the generator itself uses (``generator.arrivals.hawkes_overlay``) rather than
from a second implementation written for the test — a simulator that agreed with the
estimator only because the same author wrote both would prove nothing.

``test_a_baseline_fit_rejects_a_doubled_rate`` is the paired power check. Without it, a
statistic that never rejects anything would pass the criterion above trivially.
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.eval.metrics import tpp_rescaled_ks
from rakshak.generator.arrivals import SECONDS_PER_DAY, hawkes_overlay
from rakshak.models.rung8_tpp import (
    HOURS_PER_DAY,
    MIN_EVENTS,
    background_integral,
    compensator_increments,
    fit,
    hour_shape,
    nb_dispersion,
)
from rakshak.models.rung8_tpp import _neg_loglik as neg_loglik

FLAT = np.ones(HOURS_PER_DAY, dtype=np.float64)

#: The manifest's own kernel: hawkes_decay_minutes 3.0 -> beta = 1440/3 = 480 per day.
DECAY_MINUTES = 3.0
BETA = 24.0 * 60.0 / DECAY_MINUTES


def simulate(
    rng: np.random.Generator, *, mu: float, alpha: float, horizon: float
) -> np.ndarray:
    """A Hawkes process on ``[0, horizon]`` days with a flat background, in days.

    Immigrants are a homogeneous Poisson process of rate ``mu``; offspring come from
    ``hawkes_overlay``, which draws ``Poisson(alpha)`` children per event at ``Exp(decay)``
    delays. That is exactly the kernel ``alpha * beta * exp(-beta * t)`` the estimator
    assumes, which is what makes the fit *correctly specified* rather than merely close.

    The excitation window is set far beyond the horizon so the overlay's truncation never
    bites; a truncated kernel would be a different process from the one being fitted.
    """
    n_immigrants = int(rng.poisson(mu * horizon))
    immigrants = np.sort(rng.uniform(0.0, horizon, size=n_immigrants)) * SECONDS_PER_DAY
    events = hawkes_overlay(
        rng,
        immigrants,
        excitation=alpha,
        decay_minutes=DECAY_MINUTES,
        window_minutes=1e9,
        max_generations=60,
    )
    days = events / SECONDS_PER_DAY
    return np.asarray(days[days <= horizon], dtype=np.float64)


def test_the_hour_shape_normalises_to_mean_one() -> None:
    rng = np.random.default_rng(8)
    shape = hour_shape(rng.uniform(0.0, 30.0, size=5_000))
    assert shape.mean() == pytest.approx(1.0)
    # A whole day of that shape integrates to exactly one day's worth of background, which
    # is what makes ``mu`` events-per-day rather than events-per-day-times-an-unknown.
    assert float(background_integral(np.array([1.0]), shape)[0]) == pytest.approx(1.0)
    assert float(background_integral(np.array([7.0]), shape)[0]) == pytest.approx(7.0)


def test_the_analytic_gradient_matches_finite_differences() -> None:
    """The gradient is hand-written, so it is the one thing here that can be silently wrong.

    A wrong gradient does not crash: L-BFGS-B simply stops somewhere and returns a fit that
    looks like a fit. This is the check that fails instead.
    """
    rng = np.random.default_rng(125)
    times = simulate(rng, mu=30.0, alpha=0.3, horizon=20.0)
    span = float(background_integral(np.array([20.0]), FLAT)[0])
    args = (times, np.ones_like(times), span, 20.0)

    theta = np.array([25.0, 0.28, 400.0])
    _, analytic = neg_loglik(theta, *args)
    for i, step in enumerate((1e-5, 1e-7, 1e-3)):
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        numeric = (neg_loglik(up, *args)[0] - neg_loglik(down, *args)[0]) / (2.0 * step)
        assert analytic[i] == pytest.approx(numeric, rel=2e-4), f"parameter {i}"


def test_a_correctly_specified_fit_is_not_rejected_by_the_rescaling_test() -> None:
    """GitHub #59 acceptance criterion 1, measured rather than asserted in prose.

    The fit recovers the truth and the compensator increments are not distinguishable from
    Exponential(1). This is the *only* claim Rung 8 makes that is not contingent on the
    circularity mitigations in ``scripts/rung8_score.py``.
    """
    rng = np.random.default_rng(20260902)
    horizon = 120.0
    times = simulate(rng, mu=20.0, alpha=0.30, horizon=horizon)
    fitted = fit(times, horizon_days=horizon, shape=FLAT)

    assert fitted.converged
    assert fitted.mu == pytest.approx(20.0, rel=0.15)
    assert fitted.alpha == pytest.approx(0.30, abs=0.06)
    assert fitted.beta == pytest.approx(BETA, rel=0.35)

    result = tpp_rescaled_ks(compensator_increments(times, fitted))
    assert result.n == times.size - 1
    assert not result.rejects_at(0.05), (
        f"a correctly specified fit was rejected: KS={result.statistic:.4f} "
        f"p={result.p_value:.4f} on n={result.n}"
    )


def test_a_baseline_fit_rejects_a_doubled_rate() -> None:
    """The paired power check. A statistic that rejects nothing passes criterion 1 for free."""
    rng = np.random.default_rng(4242)
    horizon = 120.0
    baseline = simulate(rng, mu=20.0, alpha=0.30, horizon=horizon)
    fitted = fit(baseline, horizon_days=horizon, shape=FLAT)

    drifted = simulate(rng, mu=40.0, alpha=0.30, horizon=horizon)
    result = tpp_rescaled_ks(compensator_increments(drifted, fitted))
    assert result.rejects_at(0.05), (
        f"a doubled arrival rate was not detected: KS={result.statistic:.4f} "
        f"p={result.p_value:.4f}"
    )


def test_compensator_increments_are_one_per_gap_and_non_negative() -> None:
    rng = np.random.default_rng(11)
    times = simulate(rng, mu=15.0, alpha=0.2, horizon=30.0)
    fitted = fit(times, horizon_days=30.0, shape=FLAT)
    increments = compensator_increments(times, fitted)
    assert increments.size == times.size - 1
    assert bool((increments >= 0.0).all())
    # tpp_rescaled_ks raises on a negative increment; this is the guarantee it relies on.
    tpp_rescaled_ks(increments)
    assert compensator_increments(times[:1], fitted).size == 0


def test_the_nb_layer_is_reported_and_not_fitted() -> None:
    """The daily-count overdispersion the conditional intensity structurally cannot hold.

    Poisson counts give a Fano near 1 and no finite ``r``; NB counts at the manifest's
    target give a Fano near 12.25. Both are *reported* — neither enters the intensity, and
    that gap is the misspecification Rung 8's null run measures.
    """
    rng = np.random.default_rng(3)
    poisson_r, poisson_fano = nb_dispersion(rng.poisson(6.0, size=365))
    assert poisson_fano == pytest.approx(1.0, abs=0.25)
    assert np.isinf(poisson_r)

    over = rng.negative_binomial(6.0 / (12.25 - 1.0), 1.0 / 12.25, size=2_000)
    _, fano = nb_dispersion(over)
    assert fano == pytest.approx(12.25, rel=0.2)


def test_fit_refuses_a_window_it_cannot_identify_three_parameters_from() -> None:
    with pytest.raises(ValueError, match="MIN_EVENTS"):
        fit(np.linspace(0.0, 1.0, MIN_EVENTS - 1), horizon_days=1.0)
    with pytest.raises(ValueError, match="sorted"):
        fit(np.linspace(1.0, 0.0, MIN_EVENTS + 5), horizon_days=1.0)
    with pytest.raises(ValueError, match=r"must lie in"):
        fit(np.linspace(0.0, 2.0, MIN_EVENTS + 5), horizon_days=1.0)
