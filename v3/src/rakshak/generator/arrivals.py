"""The marked point process: overdispersed daily counts, plus within-day arrival times.

v1 drew arrivals from a Poisson-ish process. Real merchant daily counts are heavily
overdispersed — the v1 retrospective measured a Fano factor of **12.25** against
Poisson's 1.0 — and that misspecification was a named cause of the v1 sequence model's
failure. This module is the fix, and FR-003 is its acceptance criterion.

Three layers, per 08-generator-v2-spec.md §1:

1. **Daily intensity** ``lambda(m, d)`` — computed by the caller (personas x typologies x
   confounders x day-of-week). This module never decides a merchant's intensity; it only
   turns an intensity into counts and times.
2. **Overdispersed counts** — ``n ~ NegBinomial`` parameterised so the realised Fano
   factor is exactly ``target_fano``, at every intensity.
3. **Within-day times** — drawn from the persona's 24-bin hour-of-day categorical, with
   an optional Hawkes self-excitation overlay for the bursty typologies.

Every function here takes ``rng: np.random.Generator``. There is no module-level RNG.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "SECONDS_PER_DAY",
    "dispersion_for_fano",
    "nb_fano_for_target",
    "fano_factor",
    "hawkes_overlay",
    "interarrival_cv",
    "nb_daily_counts",
    "regular_day_times",
    "within_day_times",
]

SECONDS_PER_DAY = 86_400.0

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def dispersion_for_fano(lam: F64, target_fano: float) -> F64:
    """The NB dispersion ``r`` that puts the Fano factor at ``target_fano``.

    A negative binomial with mean ``lambda`` has variance ``lambda(1 + lambda/r)``, so
    ``Fano = 1 + lambda/r`` and ``r = lambda / (target_fano - 1)`` — 08-generator-v2-spec
    §1, stated there and derived here so the two cannot drift apart.

    Because ``r`` scales with ``lambda``, the resulting Fano is *constant across
    intensities*: a quiet merchant and a busy one are equally overdispersed, which is
    what makes the population-level Fano a meaningful calibration target rather than an
    artefact of the intensity mix.
    """
    if target_fano <= 1.0:
        return np.zeros_like(lam)
    return lam / (target_fano - 1.0)


def nb_fano_for_target(lam: F64, target_fano: float) -> F64:
    """The per-merchant NB Fano that makes the *realised* Fano equal ``target_fano``.

    By the law of total variance, for a merchant whose intensity varies across days::

        Var(N) = E[Var(N | lambda)] + Var(lambda)
               = F_nb * E[lambda] + Var(lambda)

    so the realised Fano is ``F_nb + Var(lambda)/E[lambda]``. A merchant's own
    non-stationarity — an L3 growth ramp, an L2 sale window, a typology ramp, a festival
    — contributes dispersion *on top of* whatever the count process supplies.

    This matters because it is the difference between hitting the target and missing it
    by 23%. Drawing counts at ``F_nb = 12.25`` over the composed intensity produced a
    realised population Fano of **15.11**, and gate G1 was right to call that RED. The
    v1 measurement of 12.25 was taken on real merchant counts, which already contain
    real seasonality, real growth and real platform events — so 12.25 is a property of
    the *composed* series, and the count process is the knob that has to give.

    ``lam`` is ``(n_merchants, n_days)``; the result is ``(n_merchants, 1)``, ready to
    broadcast. Clipped at 1.0: a merchant whose shape alone is more dispersed than the
    target (a deep L7 dormancy, a very lumpy L4) gets Poisson counts and still overshoots.
    Those merchants are the reason the realised figure lands slightly above target rather
    than exactly on it, and that residual is reported by G1 rather than tuned away.
    """
    mean = lam.mean(axis=1)
    excess = np.divide(
        lam.var(axis=1), mean, out=np.zeros_like(mean), where=mean > 0.0
    )
    return np.asarray(np.clip(target_fano - excess, 1.0, None))[:, None]


def nb_daily_counts(
    rng: np.random.Generator, lam: F64, target_fano: float | F64
) -> I64:
    """Draw daily transaction counts with a realised Fano factor of ``target_fano``.

    ``lam`` may be any shape; the return has the same shape. Intensities <= 0 yield 0
    (a merchant that has not onboarded, or is inside an L7 dormancy, has no arrivals —
    not a NB draw with a degenerate parameter that numpy would reject).

    ``target_fano == 1.0`` degenerates to Poisson, which is exactly right: it is the
    v1 process, and keeping it reachable is what lets the Fano assertion be tested at
    three targets rather than one.

    ``target_fano`` may be an array broadcastable to ``lam`` — that is how the engine
    passes a per-merchant dispersion from ``nb_fano_for_target``, so that merchants whose
    intensity is non-stationary get less from the count process and the *composed* series
    still lands on the target.
    """
    lam = np.asarray(lam, dtype=np.float64)
    fano = np.broadcast_to(np.asarray(target_fano, dtype=np.float64), lam.shape)
    if np.any(fano < 1.0):
        raise ValueError(
            f"target_fano must be >= 1.0; NB has no under-dispersed parameterisation "
            f"(got a minimum of {float(fano.min())!r}). Use binomial thinning if you ever "
            f"need Fano < 1."
        )
    out = np.zeros(lam.shape, dtype=np.int64)
    active = lam > 0.0
    if not np.any(active):
        return out

    poisson_mask = active & (fano == 1.0)
    if np.any(poisson_mask):
        out[poisson_mask] = rng.poisson(lam[poisson_mask])

    nb_mask = active & (fano > 1.0)
    if np.any(nb_mask):
        # numpy's negative_binomial(n, p) has mean n(1-p)/p and variance n(1-p)/p^2, so
        # Fano = 1/p exactly. p carries the target and n carries lambda.
        p = 1.0 / fano[nb_mask]
        n = lam[nb_mask] / (fano[nb_mask] - 1.0)
        out[nb_mask] = rng.negative_binomial(n, p)
    return out


def fano_factor(counts: npt.NDArray[np.integer] | F64) -> float:
    """Realised Fano factor: the mean of the **per-merchant** Fano over days.

    ``counts`` is ``(n_merchants, n_days)`` (a 1-D array is treated as one merchant).

    Per-merchant, not pooled, and that choice is load-bearing. Pooling every merchant's
    days into one variance folds the *between-merchant* intensity spread into the
    statistic, so a population of identical Poisson merchants with different rates
    reports a Fano far above 1 and the number stops measuring the arrival process at
    all. FR-003 is about the process, so the statistic is computed within a merchant and
    then averaged. Merchants with no activity are skipped — 0/0 is not a Fano of zero.
    """
    arr = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    means = arr.mean(axis=1)
    variances = arr.var(axis=1, ddof=1)
    active = means > 0.0
    if not np.any(active):
        raise ValueError("cannot compute a Fano factor: every merchant has zero activity")
    return float(np.mean(variances[active] / means[active]))


def within_day_times(
    rng: np.random.Generator, n: int, hour_weights: F64, day_start_s: float = 0.0
) -> F64:
    """``n`` sorted arrival offsets in seconds, drawn from a 24-bin hour-of-day shape.

    The hour is categorical and the position inside it is uniform, which gives the
    hour-of-day histogram its shape (what ``h_hourly_jsd`` reads) without pretending to
    a within-hour structure the generator has no basis for.
    """
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    weights = np.asarray(hour_weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("hour_weights must contain at least one positive weight")
    hours = rng.choice(24, size=n, p=weights / total)
    offsets = day_start_s + hours * 3600.0 + rng.uniform(0.0, 3600.0, size=n)
    offsets.sort()
    return np.asarray(offsets, dtype=np.float64)


def regular_day_times(
    rng: np.random.Generator, n: int, jitter_s: float, day_start_s: float = 0.0
) -> F64:
    """Near-deterministic arrivals: evenly spaced through the day with small jitter.

    This is L5 (subscription/recurring), the hard negative for ``h_interarrival_cv``.
    Without it the only near-zero-CV merchants in the population are the scripted fraud
    ones, and the feature becomes a free win that would not survive contact with a real
    portfolio.
    """
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    spacing = SECONDS_PER_DAY / n
    base = day_start_s + spacing * (np.arange(n, dtype=np.float64) + 0.5)
    offsets = base + rng.normal(0.0, jitter_s, size=n)
    offsets.sort()
    return np.asarray(offsets, dtype=np.float64)


def hawkes_overlay(
    rng: np.random.Generator,
    parents: F64,
    *,
    excitation: float,
    decay_minutes: float,
    window_minutes: float,
    max_generations: int,
) -> F64:
    """Overlay self-excited children on ``parents``; returns parents + children, sorted.

    Each event raises the intensity for the next ``window_minutes`` by ``excitation``,
    decaying exponentially — a branching process where a parent spawns
    ``Poisson(excitation)`` children at delays ``Exp(decay_minutes)``, truncated at the
    window.

    This is what makes ``f_retry_burst_rate`` and ``h_interarrival_cv`` *detectable* for
    R3. Without it, "card testing" is just a higher rate, the bursts are an artefact of
    the count distribution, and both features are noise dressed as signal.

    ``excitation`` is a branching ratio: at 0.35 the expected number of descendants per
    parent is 0.35/(1-0.35) ~ 0.54, and the process is sub-critical. ``max_generations``
    is a hard stop so a mis-set excitation >= 1 fails loudly instead of hanging.
    """
    if not 0.0 <= excitation < 1.0:
        raise ValueError(
            f"excitation is a branching ratio and must be in [0,1) or the process is "
            f"explosive; got {excitation!r}"
        )
    if parents.size == 0 or excitation == 0.0:
        return np.sort(parents)

    decay_s = decay_minutes * 60.0
    window_s = window_minutes * 60.0
    pieces: list[F64] = [parents]
    generation = parents
    for _ in range(max_generations):
        counts = rng.poisson(excitation, size=generation.size)
        n_children = int(counts.sum())
        if n_children == 0:
            break
        parent_times = np.repeat(generation, counts)
        delays = rng.exponential(decay_s, size=n_children)
        keep = delays <= window_s
        children = parent_times[keep] + delays[keep]
        if children.size == 0:
            break
        pieces.append(children)
        generation = children
    return np.sort(np.concatenate(pieces))


def interarrival_cv(times_s: F64) -> float:
    """Coefficient of variation of inter-arrival times. NaN with fewer than 3 events.

    Poisson inter-arrivals have CV 1.0; a scripted process approaches 0; a bursty
    self-excited one exceeds 1. Lives here rather than in a test because both
    ``tests/unit/test_personas.py`` (L5 < 0.3) and the gates read it, and two copies of
    a statistic is one copy too many.
    """
    if times_s.size < 3:
        return float("nan")
    gaps = np.diff(np.sort(times_s))
    mean = float(gaps.mean())
    if mean <= 0.0:
        return float("nan")
    return float(gaps.std(ddof=1) / mean)
