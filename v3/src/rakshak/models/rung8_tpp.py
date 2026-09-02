"""Rung 8 — a parametric Hawkes/NB temporal point process, fit per merchant by L-BFGS.

Named in GitHub #59 (T-0125) and in ``configs/rung_roster.yaml`` as ``tpp_hawkes_nb``. The
metric it feeds, ``eval.metrics.tpp_rescaled_ks``, was pre-registered in
``docs/PRE-REGISTRATION-CYCLE3-2026-08-31.md`` §2 and is inside ``eval_module_sha256``;
nothing here touches it. This module's whole job is to produce that function's input.

**Not a neural TPP.** ADR-V3-001 holds this cycle: no autograd, no GPU, no dependency
beyond ``scipy``, which is already pinned. The conditional intensity is written out by hand
and its gradient is written out beside it, so ``scipy.optimize.minimize(..., jac=True)``
gets an exact derivative rather than a finite-difference one.

The model
---------
Over a window that starts at a UTC day boundary, with time measured in **days**::

    lambda(t) = mu * s(t)  +  sum_{t_i < t} alpha * beta * exp(-beta * (t - t_i))

* ``s(t)`` — the merchant's own 24-bin hour-of-day shape, normalised to mean 1 and
  estimated by counting. It is held **fixed** during the optimisation: it is a shape, not a
  rate, and folding 24 more free parameters into a per-merchant fit over a 30-day window
  would be fitting noise.
* ``mu`` — background events per day.
* ``alpha`` — the branching ratio. Expected offspring per event; bounded below 1 or the
  process is explosive, exactly as ``generator/arrivals.py::hawkes_overlay`` requires.
* ``beta`` — exponential decay, per day. The manifest's ``hawkes_decay_minutes: 3.0`` is
  ``beta = 480``.

That is the generator's own arrival model with one deliberate omission, and the omission is
the honest part: **the negative-binomial layer is not in the intensity.** The generator
draws a day's *count* from a NB with the target Fano and then places those events by the
hour shape, which is a Cox process with an i.i.d. gamma multiplier per day. A conditional
intensity cannot see that multiplier — it is latent, i.i.d., and carries no history — so a
history-based compensator cannot absorb it. ``nb_dispersion`` measures it on the daily
counts and it is reported beside every fit rather than folded in, because the size of that
residual dispersion is precisely the size of the model misspecification the KS test will
register on a merchant that has not drifted at all.

The circularity objection, stated here rather than discovered later
------------------------------------------------------------------
Fitting an NB/Hawkes intensity to NB/Hawkes data and calling the misfit "anomaly" is
well-specified-model-on-well-specified-data. It is the same objection ADR-0002 used to
reject GNNs in v1 — *"the only merchant x payer graph available is the one this repo's
generator writes, so a GNN would be scored on how well it learned our own graph
assumptions. A win would prove nothing"* — and it is an evaluation-validity problem, not a
compute problem. ``scripts/rung8_score.py`` runs the two mitigations #59 makes mandatory
(the ``prevalence = 0`` null with confounders on, and BAF test-size calibration) and
``LIMITATIONS.md`` §11 records what they found.

Cost and state
--------------
One fit is three parameters over one merchant's baseline window; the recursions below are
O(n) per likelihood evaluation and the loop is Python, which is the ceiling — a merchant
with 10^5 baseline events costs ~0.3 s to fit. Nothing here is in the per-epoch hot path
as written: the fit is offline and the per-epoch cost is one compensator pass plus one KS
call. The online form (carry ``sum exp(-beta(t - t_i))`` as one float64 in
``MerchantState``) is a real extension and is **not** done here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

__all__ = [
    "BOUNDS",
    "HOURS_PER_DAY",
    "HOUR_SMOOTHING",
    "INITIAL_ALPHA",
    "INITIAL_BETA_PER_DAY",
    "MIN_EVENTS",
    "HawkesNbFit",
    "background_integral",
    "compensator_increments",
    "fit",
    "hour_shape",
    "nb_dispersion",
    "shape_at",
]

F64 = npt.NDArray[np.float64]

HOURS_PER_DAY = 24

#: Add-one smoothing on the hour histogram. An hour a merchant simply never traded in gets
#: a zero weight otherwise, and an event that later lands there makes ``log lambda`` -inf —
#: the fit would then be decided by one arrival at 04:00 rather than by the process.
HOUR_SMOOTHING = 1.0

#: Fewer events than this and three parameters are not identified from the window. The
#: caller is expected to skip the merchant rather than take a number from a fit that had
#: nothing to fit to; ``fit`` raises instead of returning a plausible-looking answer.
MIN_EVENTS = 20

#: ``(mu, alpha, beta)`` box for L-BFGS-B. ``alpha < 1`` is the sub-criticality condition
#: ``hawkes_overlay`` states in the same words; 0.95 keeps the optimiser off the boundary
#: where the compensator diverges. ``beta`` in [1e-3, 1e5] per day spans a 16-hour half-life
#: to a 1-second one, which brackets the manifest's 3 minutes by a wide margin either side.
BOUNDS: tuple[tuple[float, float], ...] = ((1e-8, 1e7), (0.0, 0.95), (1e-3, 1e5))

#: Starting branching ratio. The manifest's ``hawkes_excitation`` is 0.35 but only for the
#: bursty typologies, so starting there would start every legitimate merchant in the wrong
#: place; 0.2 is between the two and the likelihood is well behaved from either side.
INITIAL_ALPHA = 0.2

#: Starting decay: ``configs/scenario_v2.yaml::arrivals.hawkes_decay_minutes`` is 3.0, and
#: 1440/3 = 480 per day. Read as a starting point, not as knowledge — the fit moves it.
INITIAL_BETA_PER_DAY = 480.0


@dataclass(frozen=True, slots=True)
class HawkesNbFit:
    """One merchant's fitted intensity, plus what the intensity could not represent.

    ``nb_fano`` and ``nb_dispersion`` are carried on the fit rather than computed by the
    caller so that no result can be reported without the size of the model's own
    misspecification sitting next to it.
    """

    mu: float
    alpha: float
    beta: float
    hour_shape: F64
    nb_dispersion: float
    nb_fano: float
    loglik: float
    n_events: int
    horizon_days: float
    converged: bool


def hour_shape(times_days: F64) -> F64:
    """24 hourly weights from a merchant's own arrival times, normalised to mean 1.

    Mean 1 rather than sum 1 is what makes ``mu`` interpretable as events per day: the
    integral of ``s`` over any whole day is then exactly 1.0, so ``mu`` and the shape do not
    trade off against each other and the optimiser has one fewer flat direction to wander
    along.
    """
    t = np.asarray(times_days, dtype=np.float64)
    hours = np.floor(np.mod(t, 1.0) * HOURS_PER_DAY).astype(np.int64)
    counts = (
        np.bincount(np.clip(hours, 0, HOURS_PER_DAY - 1), minlength=HOURS_PER_DAY).astype(
            np.float64
        )
        + HOUR_SMOOTHING
    )
    return np.asarray(counts * (HOURS_PER_DAY / counts.sum()), dtype=np.float64)


def _cumulative_shape(shape: F64) -> F64:
    """``cum[h]`` = integral of ``s`` over the first ``h`` hours of a day, in day units."""
    return np.concatenate([[0.0], np.cumsum(shape) / HOURS_PER_DAY])


def background_integral(t: F64, shape: F64) -> F64:
    """``S(t) = integral from 0 to t of s(u) du``, with ``t`` in days from a day boundary.

    Exact rather than quadrature: ``s`` is piecewise constant on hours, so the integral is
    whole days (each worth exactly 1.0, by the mean-1 normalisation) plus a closed-form
    partial hour. A quadrature here would put a tolerance between the likelihood and its
    gradient, and the gradient check in ``tests/unit/test_rung8.py`` would be measuring the
    quadrature.
    """
    cum = _cumulative_shape(np.asarray(shape, dtype=np.float64))
    t_arr = np.asarray(t, dtype=np.float64)
    whole = np.floor(t_arr)
    position = (t_arr - whole) * HOURS_PER_DAY
    hour = np.minimum(np.floor(position).astype(np.int64), HOURS_PER_DAY - 1)
    within = position - hour
    return np.asarray(
        whole + cum[hour] + np.asarray(shape)[hour] * within / HOURS_PER_DAY,
        dtype=np.float64,
    )


def shape_at(t: F64, shape: F64) -> F64:
    """``s(t)``: the hourly weight in force at each time."""
    t_arr = np.asarray(t, dtype=np.float64)
    hour = np.floor(np.mod(t_arr, 1.0) * HOURS_PER_DAY).astype(np.int64)
    return np.asarray(np.asarray(shape)[np.clip(hour, 0, HOURS_PER_DAY - 1)], dtype=np.float64)


def nb_dispersion(daily_counts: npt.NDArray[np.integer] | F64) -> tuple[float, float]:
    """``(r, Fano)`` for the daily counts, by moments: ``Fano = var/mean``, ``r = mean/(F-1)``.

    Moments rather than an NB MLE on purpose. This number is not used to score anything and
    is not optimised over; it exists to report how much day-to-day dispersion the
    conditional intensity cannot represent, and for that a moment estimator is the honest
    instrument — an MLE would imply the NB layer had been fitted into the model, which it
    has not.

    Returns ``(inf, Fano)`` when the counts are at or below Poisson dispersion: there is no
    finite NB ``r`` for that, and returning a large number instead of ``inf`` would read as
    a measurement.
    """
    counts = np.asarray(daily_counts, dtype=np.float64)
    if counts.size < 2:
        return float("nan"), float("nan")
    mean = float(counts.mean())
    if mean <= 0.0:
        return float("nan"), float("nan")
    fano = float(counts.var(ddof=1) / mean)
    if fano <= 1.0:
        return float("inf"), fano
    return mean / (fano - 1.0), fano


def _recursions(times: F64, beta: float) -> tuple[F64, F64]:
    """``R_k = sum_{i<k} exp(-beta(t_k - t_i))`` and ``dR_k/dbeta``, both in one O(n) pass.

    The closed form ``R_k = exp(-beta t_k) * sum_{i<k} exp(beta t_i)`` is the vectorised
    version and it overflows: at the manifest's ``beta = 480`` per day over a 30-day window,
    ``exp(beta t)`` is ``exp(14400)``. The recursion is unconditionally stable because every
    factor is in [0, 1], so it is a Python loop and the loop is the accepted ceiling. See
    the module docstring.
    """
    n = times.size
    r_values = np.zeros(n, dtype=np.float64)
    r_grads = np.zeros(n, dtype=np.float64)
    gaps = np.diff(times)
    r = 0.0
    r_grad = 0.0
    for k in range(1, n):
        decay = float(np.exp(-beta * gaps[k - 1]))
        r = decay * (1.0 + r)
        r_grad = -gaps[k - 1] * r + decay * r_grad
        r_values[k] = r
        r_grads[k] = r_grad
    return r_values, r_grads


def _neg_loglik(
    theta: F64, times: F64, s_values: F64, span: float, horizon: float
) -> tuple[float, F64]:
    """Negative log-likelihood and its exact gradient in ``(mu, alpha, beta)``.

    ``loglik = sum_k log lambda(t_k) - mu * S(T) - alpha * sum_k (1 - exp(-beta(T - t_k)))``,
    the second term being the compensator over the whole window: the background contributes
    ``mu * S(T)`` and each event contributes its kernel mass not yet spent by ``T``.
    """
    mu, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    r_values, r_grads = _recursions(times, beta)
    lam = mu * s_values + alpha * beta * r_values
    tail = np.exp(-beta * (horizon - times))
    unspent = 1.0 - tail

    loglik = float(np.log(lam).sum()) - mu * span - alpha * float(unspent.sum())
    inv = 1.0 / lam
    grad_mu = float((s_values * inv).sum()) - span
    grad_alpha = float((beta * r_values * inv).sum()) - float(unspent.sum())
    grad_beta = float((alpha * (r_values + beta * r_grads) * inv).sum()) - alpha * float(
        ((horizon - times) * tail).sum()
    )
    return -loglik, -np.array([grad_mu, grad_alpha, grad_beta], dtype=np.float64)


def fit(
    times_days: F64,
    *,
    horizon_days: float,
    daily_counts: npt.NDArray[np.integer] | F64 | None = None,
    shape: F64 | None = None,
) -> HawkesNbFit:
    """Maximise the Hawkes log-likelihood over ``(mu, alpha, beta)`` by L-BFGS-B.

    ``times_days`` must be sorted, non-negative, and measured in days from the **start of
    the window**, which must itself be a UTC day boundary — otherwise ``mod(t, 1)`` is not
    the hour of day and the shape is applied at the wrong phase.

    ``shape`` overrides the estimated hour shape; it exists so a test can hold the shape at
    a known truth and vary only the three fitted parameters.
    """
    times = np.asarray(times_days, dtype=np.float64)
    if times.size < MIN_EVENTS:
        raise ValueError(
            f"{times.size} events is below MIN_EVENTS={MIN_EVENTS}; three parameters are "
            "not identified from that window. Skip the merchant rather than fitting it."
        )
    if not np.all(np.diff(times) >= 0.0):
        raise ValueError("times_days must be sorted ascending")
    if times[0] < 0.0 or times[-1] > horizon_days:
        raise ValueError(
            f"times_days must lie in [0, {horizon_days}]; got "
            f"[{times[0]!r}, {times[-1]!r}]. Times are relative to the window start."
        )

    hourly = hour_shape(times) if shape is None else np.asarray(shape, dtype=np.float64)
    s_values = shape_at(times, hourly)
    span = float(background_integral(np.array([horizon_days]), hourly)[0])

    # Start where the background alone explains the events the branching term is not
    # expected to: with a branching ratio of alpha, a fraction (1 - alpha) of events are
    # immigrants, so mu * S(T) = n * (1 - alpha) is the moment-matched starting rate.
    x0 = np.array(
        [max(times.size * (1.0 - INITIAL_ALPHA) / max(span, 1e-9), BOUNDS[0][0]),
         INITIAL_ALPHA,
         INITIAL_BETA_PER_DAY],
        dtype=np.float64,
    )
    result = minimize(
        _neg_loglik,
        x0,
        args=(times, s_values, span, float(horizon_days)),
        method="L-BFGS-B",
        jac=True,
        bounds=BOUNDS,
    )
    dispersion, fano = (
        nb_dispersion(daily_counts) if daily_counts is not None else (float("nan"),) * 2
    )
    return HawkesNbFit(
        mu=float(result.x[0]),
        alpha=float(result.x[1]),
        beta=float(result.x[2]),
        hour_shape=hourly,
        nb_dispersion=dispersion,
        nb_fano=fano,
        loglik=-float(result.fun),
        n_events=int(times.size),
        horizon_days=float(horizon_days),
        converged=bool(result.success),
    )


def compensator_increments(times_days: F64, fitted: HawkesNbFit) -> F64:
    """``Lambda_k`` between consecutive events — the input ``tpp_rescaled_ks`` takes.

    ``Lambda_k = mu * (S(t_k) - S(t_{k-1})) + alpha * (1 - exp(-beta * gap_k)) * Q_{k-1}``,
    where ``Q_{k-1} = sum_{i <= k-1} exp(-beta (t_{k-1} - t_i))`` accumulates by the same
    recursion the likelihood uses.

    **The history before ``times_days[0]`` is dropped**, so ``Q`` starts at 1 (the first
    event exciting itself). Over any window longer than a few multiples of ``1/beta`` — 3
    minutes at the manifest's decay — that is numerically nothing; over a window shorter
    than that it is not, and the caller should not use one.

    Returns ``n - 1`` values, one per inter-arrival interval, non-negative by construction.
    """
    times = np.asarray(times_days, dtype=np.float64)
    if times.size < 2:
        return np.empty(0, dtype=np.float64)
    background = fitted.mu * np.diff(background_integral(times, fitted.hour_shape))
    gaps = np.diff(times)
    excitation = np.empty(times.size - 1, dtype=np.float64)
    q = 1.0
    for k in range(times.size - 1):
        decay = float(np.exp(-fitted.beta * gaps[k]))
        excitation[k] = fitted.alpha * (1.0 - decay) * q
        q = decay * q + 1.0
    # Clipped at 0 only against float round-off on a strictly non-negative quantity:
    # tpp_rescaled_ks raises on a negative increment, and it is right to.
    return np.asarray(np.maximum(background + excitation, 0.0), dtype=np.float64)
