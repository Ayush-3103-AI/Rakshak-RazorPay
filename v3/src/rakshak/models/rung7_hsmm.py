"""Rung 7a — the HSMM-NB inference core, hand-written in numpy (T-0123, #57).

Rung 7's claim is not that it separates a drifting merchant from a healthy one — the
scoring rungs already do that, and v2 measured that they do it better. Its claim is that
it can say **when** the drift began. That is why the duration model is explicit and not
implicit: a Markov chain's dwell time is geometric, geometric is memoryless, and a
memoryless dwell is a bad model of a ramp that has a characteristic length. The explicit
duration distribution is the entire difference between this rung and the v1 HMM that was
already tried and falsified.

**Scope.** This module is the inference core only: emissions, forward, backward, Viterbi,
EM. Onset localisation and the segmented narrative are T-0124 (#58), and the
:func:`rakshak.eval.metrics.onset_localisation_error` metric that judges them was declared
and sealed into ``EVAL-LOCK-CYCLE3.json`` before this file existed.

**Parameterisation.** ``K`` states (four in the intended configuration —
:data:`STATE_NAMES`), ``D`` dwell-time support (``d = 1..D``), ``C`` independent count
channels.

- ``start`` — ``(K,)`` initial state distribution.
- ``trans`` — ``(K, K)`` with a **zero diagonal**. A self-transition in an HSMM is not
  identifiable: it is indistinguishable from a longer dwell, and allowing one lets the
  transition matrix absorb exactly the structure the duration model exists to hold.
- ``durations`` — ``(K, D)`` **non-parametric** dwell pmf per state, truncated at ``D``
  and normalised. Non-parametric because its M-step is closed-form; a parametric family
  would need an inner optimiser for no gain at ``D`` = 60.
- ``nb_mean`` / ``nb_dispersion`` — ``(K, C)`` negative-binomial mean ``mu`` and size
  ``r``, so ``Var = mu + mu^2/r`` and ``Fano = 1 + mu/r``. Poisson is the ``r -> inf``
  limit and is *wrong here by measurement*: the v2 diagnosis is overdispersed counts
  (Fano 12.25 observed, 13.040 realised by the generator), and a Poisson emission would
  rebuild the precise defect v2 exists to fix. Channels are independent given the state.

**Residual-time formulation, and the complexity that follows.** The chain is expanded to
``(state k, residual r)`` where ``r`` counts the steps remaining in the current segment
*including the current one*. ``r > 1`` decrements deterministically; ``r = 1`` draws a new
state from ``trans`` and a new dwell from ``durations``. Written this way, forward,
backward and Viterbi are each

    time  O(T * K * (K + D))        memory  O(T * K * D)

rather than the ``O(T * D^2 * K^2)`` of the segment-form recursion that the K1 survey
§2.2 rejected on cost. At ``K=4``, ``D=60``, ``T=365`` that is a few hundred thousand
float operations over arrays of 88k cells — a fixed, vectorised numpy cost per merchant,
**linear rather than quadratic in D**. This rung runs at **Stage 2 only**, on merchants a
cheaper rung already promoted; it is never in the scoring path.

**The 50 ms Stage-2 budget is NOT certified by this ticket.** The only box available while
T-0123 was built was simultaneously running the feature-materialisation job, and its
timings are worthless: repeated runs of identical code swung by 2x, and ``D=30`` measured
*slower* than ``D=60``, which is impossible. Best-case observed at ``K=4, D=60, T=365``
was 73 ms for :meth:`HsmmNb.smooth` and 9 ms for :meth:`HsmmNb.decode`. What is left is
Python-level call overhead across the ``T``-step loop — roughly 3,000 numpy calls on
240-cell arrays — which is where an idle machine should land far below a contended one.
The budget must be asserted in ``tests/perf/`` on a quiet box before Rung 7 is adopted;
#51 §4 already names the remedy if it fails, which is to reduce ``D`` and log the
reduction rather than to quietly miss the NFR.

**Everything is in log space.** At realistic sequence lengths a linear-space forward
underflows to exactly zero — float64 stops at ``e^-745`` and a year-long merchant sits far
below that. Scaling factors would also work but hide the failure mode; log-domain
arithmetic with a stable ``logsumexp`` makes it structural.
``tests/unit/test_rung7_hsmm.py::test_long_sequence_underflows_a_linear_forward`` runs a
naive linear forward beside this one on the same fixture and asserts the naive one returns
exactly 0.0 while this one still agrees with an independent log-space HMM.

**Prime Directive 3.** Nothing here imports or names a radioactive field. The core infers
change-points from observed counts alone; the comparison against ``drift_onset_at``
happens on the eval side, inside the metric.

**Not a Scorer.** This class deliberately has no ``predict``. See
:mod:`rakshak.explain.registry` — anything satisfying ``Scorer`` is refused registration
as an explainer, and ``tests/unit/test_rung7_hsmm.py`` asserts structurally that the HSMM
cannot reach the scoring path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.special import gammaln

__all__ = [
    "MAX_DISPERSION",
    "MAX_DURATION",
    "MIN_DISPERSION",
    "STATE_NAMES",
    "EmFit",
    "HsmmNb",
    "fit",
    "geometric_durations",
    "sample",
    "seed_model",
]

#: The intended four-state configuration. The core is K-agnostic; these are the names the
#: narrative layer (#58) attaches to the states of a K=4 fit.
STATE_NAMES: Final[tuple[str, ...]] = ("HEALTHY", "RAMP", "EXFIL", "BURNT")

#: Dwell support cap, in days. #51 §4 sets it here and commits to *logging* any reduction
#: rather than quietly shrinking it if the Stage-2 budget is missed.
MAX_DURATION: Final = 60

#: Dispersion bounds. ``r -> inf`` is Poisson; the clip stops an underdispersed state
#: (sample variance below its mean, which happens on short weighted segments) from
#: producing a negative or infinite size.
MIN_DISPERSION: Final = 1e-3
MAX_DISPERSION: Final = 1e6

_MIN_MEAN: Final = 1e-9
_SIMPLEX_TOL: Final = 1e-9


def _log(p: np.ndarray) -> np.ndarray:
    """``log`` with zeros mapped to ``-inf`` rather than to a warning."""
    with np.errstate(divide="ignore"):
        return np.asarray(np.log(p))


def _logsumexp(a: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    """Stable ``log(sum(exp(a)))``, and the reason this is not ``scipy.special.logsumexp``.

    The recursions call this three times per day on arrays of at most ``K*D`` cells, so
    what dominates is per-call overhead, not arithmetic. ``scipy``'s general version costs
    tens of microseconds a call, which at ``T=365`` is most of the Stage-2 budget before
    the backward pass has started: swapping it for these six lines took ``log_likelihood``
    at ``K=4, D=60, T=365`` from 100.7 ms to 37.5 ms and the full posterior from 653.7 ms
    to ~90 ms, on a box that was simultaneously running the feature-materialisation job.
    Those figures are indicative only and are **not** an NFR certification — see the
    complexity note in the module docstring.

    An all-``-inf`` slice — a state the model gives no mass to — returns ``-inf`` rather
    than ``nan``, which is why the peak is forced to zero when it is not finite.
    """
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        total = np.log(np.sum(np.exp(a - peak), axis=axis, keepdims=True))
    return np.asarray(np.squeeze(peak + total, axis=axis))


def _as_counts(obs: np.ndarray) -> np.ndarray:
    """``(T, C)`` float64 view of a count sequence, validated.

    NB emissions are a count family. Handing them a rate, a z-score or a negative residual
    produces a number rather than an error, and that number would be meaningless — so the
    check lives at the one place every path funnels through, not at each caller.
    """
    y = np.asarray(obs, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] == 0:
        raise ValueError(f"observations must be (T,) or (T, C) with T >= 1; got {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("observations contain nan or inf")
    if np.any(y < 0.0) or np.any(y != np.floor(y)):
        raise ValueError(
            "negative-binomial emissions are counts: observations must be non-negative "
            "integers. Scale or bin a continuous feature before handing it to Rung 7."
        )
    return y


@dataclass(frozen=True, slots=True, eq=False)
class HsmmNb:
    """An HSMM with negative-binomial emissions and a non-parametric dwell distribution.

    Validated at construction: a model that cannot be a distribution should not be able to
    exist and then produce a plausible-looking likelihood. ``eq=False`` because the fields
    are arrays and ``==`` on them is ambiguous.
    """

    start: np.ndarray
    trans: np.ndarray
    durations: np.ndarray
    nb_mean: np.ndarray
    nb_dispersion: np.ndarray

    def __post_init__(self) -> None:
        if self.durations.ndim != 2:
            raise ValueError(f"durations must be (K, D); got shape {self.durations.shape}")
        k, d = self.durations.shape
        if k < 2:
            raise ValueError(
                f"an HSMM needs at least 2 states (self-transitions are barred, so K=1 "
                f"could never transition anywhere); got K={k}"
            )
        if d < 1:
            raise ValueError(f"the dwell support must be at least 1 day; got D={d}")
        if self.start.shape != (k,):
            raise ValueError(f"start must be (K,)=({k},); got {self.start.shape}")
        if self.trans.shape != (k, k):
            raise ValueError(f"trans must be (K, K)=({k}, {k}); got {self.trans.shape}")
        if self.nb_mean.ndim != 2 or self.nb_mean.shape[0] != k:
            raise ValueError(f"nb_mean must be (K, C) with K={k}; got {self.nb_mean.shape}")
        if self.nb_dispersion.shape != self.nb_mean.shape:
            raise ValueError(
                f"nb_dispersion must match nb_mean {self.nb_mean.shape}; "
                f"got {self.nb_dispersion.shape}"
            )
        if np.any(np.diagonal(self.trans) != 0.0):
            raise ValueError(
                "trans must have a zero diagonal. A self-transition is unidentifiable "
                "against a longer dwell, so it would absorb exactly the structure the "
                "duration model exists to represent."
            )
        for name, arr in (
            ("start", self.start[None, :]),
            ("trans", self.trans),
            ("durations", self.durations),
        ):
            if np.any(arr < 0.0):
                raise ValueError(f"{name} carries a negative probability")
            sums = arr.sum(axis=1)
            if not np.allclose(sums, 1.0, atol=_SIMPLEX_TOL):
                raise ValueError(f"{name} rows must sum to 1; got {sums}")
        if np.any(self.nb_mean <= 0.0):
            raise ValueError("nb_mean must be strictly positive (it is a count mean)")
        if np.any(self.nb_dispersion <= 0.0):
            raise ValueError("nb_dispersion must be strictly positive (it is the NB size)")

    @property
    def n_states(self) -> int:
        return int(self.durations.shape[0])

    @property
    def max_duration(self) -> int:
        return int(self.durations.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.nb_mean.shape[1])

    @property
    def mean_dwell(self) -> np.ndarray:
        """``(K,)`` expected dwell in days, under the truncated pmf."""
        support = np.arange(1, self.max_duration + 1, dtype=np.float64)
        return np.asarray(self.durations @ support)

    # ------------------------------------------------------------------ emissions

    def log_emissions(self, obs: np.ndarray) -> np.ndarray:
        """``(T, K)`` log NB pmf of each observation under each state.

        ``obs`` is ``(T,)`` for a single channel or ``(T, C)``; values are counts.
        """
        y = _as_counts(obs)
        if y.shape[1] != self.n_channels:
            raise ValueError(
                f"observations have {y.shape[1]} channels; the model has {self.n_channels}"
            )
        mu = np.maximum(self.nb_mean, _MIN_MEAN)[None, :, :]
        r = self.nb_dispersion[None, :, :]
        yy = y[:, None, :]
        log_denom = np.log(r + mu)
        per_channel = (
            gammaln(yy + r)
            - gammaln(r)
            - gammaln(yy + 1.0)
            + r * (np.log(r) - log_denom)
            + yy * (np.log(mu) - log_denom)
        )
        return np.asarray(per_channel.sum(axis=2), dtype=np.float64)

    # ------------------------------------------------------------------ recursions

    def _log_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _log(self.start), _log(self.trans), _log(self.durations)

    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(alpha, entry)``, both in log space.

        ``alpha[t, k, r-1] = log P(o_1..o_t, S_t=k, R_t=r)`` and
        ``entry[t, k] = log P(o_1..o_{t-1}, a segment of state k begins at t)``.
        """
        n, k = log_b.shape
        log_start, log_trans, log_dur = self._log_params()
        alpha = np.empty((n, k, self.max_duration), dtype=np.float64)
        entry = np.empty((n, k), dtype=np.float64)
        entry[0] = log_start
        alpha[0] = log_start[:, None] + log_dur + log_b[0][:, None]
        pad = np.full((k, 1), -np.inf)
        for t in range(1, n):
            entry[t] = _logsumexp(alpha[t - 1][:, 0][:, None] + log_trans, axis=0)
            # residual r at t was r+1 at t-1; r = D can only have been entered, never
            # continued, so the shifted column is -inf.
            continued = np.concatenate([alpha[t - 1][:, 1:], pad], axis=1)
            alpha[t] = log_b[t][:, None] + np.logaddexp(continued, entry[t][:, None] + log_dur)
        return alpha, entry

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        """``beta[t, k, r-1] = log P(o_{t+1}..o_T | S_t=k, R_t=r)``.

        ``beta[T-1] = 0`` for every residual, and that is what right-censors the final
        segment: a sequence ending mid-dwell is evidence about the dwell, not a
        contradiction to be normalised away.
        """
        n, k = log_b.shape
        _, log_trans, log_dur = self._log_params()
        beta = np.zeros((n, k, self.max_duration), dtype=np.float64)
        for t in range(n - 2, -1, -1):
            # value of *entering* state j at t+1, marginalised over its new dwell
            enter_value = log_b[t + 1] + _logsumexp(log_dur + beta[t + 1], axis=1)
            beta[t, :, 0] = _logsumexp(log_trans + enter_value[None, :], axis=1)
            beta[t, :, 1:] = log_b[t + 1][:, None] + beta[t + 1][:, :-1]
        return beta

    def log_likelihood(self, obs: np.ndarray) -> float:
        """Log P(obs) under the model, right-censored at the end of the sequence."""
        alpha, _ = self._forward(self.log_emissions(obs))
        return float(_logsumexp(alpha[-1].ravel(), axis=0))

    def smooth(self, obs: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """``(log_likelihood, state_posterior, segment_start_posterior)`` in one pass.

        ``state_posterior`` is ``(T, K)`` and its rows sum to 1. ``segment_start_posterior``
        is ``(T, K)``: the probability that a segment of state ``k`` *begins* on day ``t``,
        which is the quantity #58 turns into an onset estimate — exposed here because it
        falls out of the E-step for free, and because a change-point posterior the core
        cannot produce is one nobody can check.

        Both come from a single forward-backward. The Stage-2 caller wants both, and
        running the recursions twice to fetch them separately doubles the per-merchant
        latency for nothing.
        """
        loglik, gamma, eta, _, _ = self._estep(obs, want_transitions=False)
        return loglik, gamma, np.asarray(eta.sum(axis=2))

    def posterior_states(self, obs: np.ndarray) -> np.ndarray:
        """``(T, K)`` smoothed state posterior. Rows sum to 1."""
        return self.smooth(obs)[1]

    def posterior_segment_starts(self, obs: np.ndarray) -> np.ndarray:
        """``(T, K)`` posterior that a segment of state ``k`` *begins* on day ``t``."""
        return self.smooth(obs)[2]

    def decode(self, obs: np.ndarray) -> np.ndarray:
        """MAP state path, ``(T,)`` int64 — Viterbi over the residual-time chain.

        Consecutive equal states are one segment by construction (the diagonal of
        ``trans`` is zero), so run-length encoding the path recovers the segmentation
        without a second backpointer array.
        """
        log_b = self.log_emissions(obs)
        n, k = log_b.shape
        d = self.max_duration
        log_start, log_trans, log_dur = self._log_params()
        delta = np.empty((n, k, d), dtype=np.float64)
        came_from = np.zeros((n, k), dtype=np.int64)
        entered = np.zeros((n, k, d), dtype=np.bool_)
        delta[0] = log_start[:, None] + log_dur + log_b[0][:, None]
        entered[0] = True
        pad = np.full((k, 1), -np.inf)
        for t in range(1, n):
            transit = delta[t - 1][:, 0][:, None] + log_trans
            came_from[t] = np.argmax(transit, axis=0)
            enter = transit.max(axis=0)[:, None] + log_dur
            continued = np.concatenate([delta[t - 1][:, 1:], pad], axis=1)
            entered[t] = enter > continued
            delta[t] = log_b[t][:, None] + np.where(entered[t], enter, continued)
        if not np.isfinite(delta[-1].max()):
            raise ValueError(
                "no state path has non-zero probability for this sequence — the model "
                "assigns it zero mass, so there is nothing to decode"
            )
        best = np.unravel_index(int(np.argmax(delta[-1])), (k, d))
        state, residual = int(best[0]), int(best[1])
        path = np.empty(n, dtype=np.int64)
        for t in range(n - 1, -1, -1):
            path[t] = state
            if t == 0:
                break
            if entered[t, state, residual]:
                state, residual = int(came_from[t, state]), 0
            else:
                residual += 1
        return path

    # ------------------------------------------------------------------ EM

    def _estep(
        self, obs: np.ndarray, *, want_transitions: bool = True
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(loglik, gamma, eta, xi, y)`` for one sequence.

        ``gamma`` is ``(T, K)`` state occupancy, ``eta`` is ``(T, K, D)`` the posterior
        that a segment of state ``k`` with dwell ``d`` begins at ``t``, ``xi`` is
        ``(K, K)`` expected transition counts, and ``y`` is the ``(T, C)`` observation
        matrix, returned so the emission M-step does not re-derive it.

        ``want_transitions=False`` skips ``xi`` — EM needs it, an explanation does not,
        and it is the most expensive of the three sufficient statistics.
        """
        y = _as_counts(obs)
        log_b = self.log_emissions(y)
        alpha, entry = self._forward(log_b)
        beta = self._backward(log_b)
        loglik = float(_logsumexp(alpha[-1].ravel(), axis=0))
        _, log_trans, log_dur = self._log_params()

        gamma = np.exp(_logsumexp(alpha + beta, axis=2) - loglik)
        eta = np.exp(entry[:, :, None] + log_dur[None] + log_b[:, :, None] + beta - loglik)
        xi = np.zeros((self.n_states, self.n_states))
        if want_transitions:
            # Value of entering state k at t, marginalised over its dwell — the same
            # quantity the backward pass builds, written once so the two cannot disagree.
            enter_value = log_b + _logsumexp(log_dur[None] + beta, axis=2)
            xi = np.exp(
                _logsumexp(
                    alpha[:-1, :, 0][:, :, None] + log_trans[None] + enter_value[1:][:, None, :],
                    axis=0,
                )
                - loglik
            )
        return loglik, gamma, eta, xi, y


@dataclass(frozen=True, slots=True, eq=False)
class EmFit:
    """What :func:`fit` returns: the model, and the evidence it was fitted honestly."""

    model: HsmmNb
    log_likelihoods: tuple[float, ...]
    converged: bool

    @property
    def n_iter(self) -> int:
        return len(self.log_likelihoods)


def fit(
    sequences: list[np.ndarray],
    init: HsmmNb,
    *,
    n_iter: int = 50,
    tol: float = 1e-6,
    duration_prior: float = 1e-2,
    fit_dispersion: bool = True,
) -> EmFit:
    """EM over pooled sequences. Returns the fitted model and its likelihood trace.

    ``duration_prior`` is a flat Dirichlet pseudocount spread over the dwell support. With
    ``K*D`` free duration parameters an unsmoothed M-step will happily put probability 1
    on a single observed dwell and ``-inf`` on every other length; the pseudocount is what
    stops one merchant's segmentation becoming a hard constraint on all the others.

    ``fit_dispersion`` re-estimates ``r`` by **weighted method of moments**, not by its
    weighted MLE — the MLE has no closed form and an inner Newton solve is not worth the
    failure modes here. That makes the overall iteration generalised-EM: ``start``,
    ``trans``, ``durations`` and ``nb_mean`` are exact M-steps and are monotone, while the
    moment-matched ``r`` is not, so the likelihood trace can dip. With
    ``fit_dispersion=False`` the iteration is plain EM and the trace is non-decreasing —
    which is what ``test_em_log_likelihood_is_monotone`` asserts, because a monotonicity
    test that is allowed to fail legitimately is not a test.

    Complexity: ``O(n_iter * sum_i T_i * K * (K + D))``.
    """
    if not sequences:
        raise ValueError("fit needs at least one sequence")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1; got {n_iter}")
    if duration_prior < 0.0:
        raise ValueError(f"duration_prior must be >= 0; got {duration_prior}")

    model = init
    k, d = model.n_states, model.max_duration
    history: list[float] = []
    converged = False
    for _ in range(n_iter):
        total_ll = 0.0
        start_acc = np.zeros(k)
        eta_acc = np.zeros((k, d))
        xi_acc = np.zeros((k, k))
        weight = np.zeros(k)
        wy = np.zeros((k, model.n_channels))
        wyy = np.zeros((k, model.n_channels))
        for obs in sequences:
            loglik, gamma, eta, xi, y = model._estep(obs)
            total_ll += loglik
            start_acc += eta[0].sum(axis=1)
            eta_acc += eta.sum(axis=0)
            xi_acc += xi
            weight += gamma.sum(axis=0)
            wy += gamma.T @ y
            wyy += gamma.T @ (y * y)
        history.append(total_ll)
        model = _mstep(
            model,
            start_acc=start_acc,
            eta_acc=eta_acc,
            xi_acc=xi_acc,
            weight=weight,
            wy=wy,
            wyy=wyy,
            duration_prior=duration_prior,
            fit_dispersion=fit_dispersion,
        )
        if len(history) > 1 and abs(history[-1] - history[-2]) <= tol * max(1.0, abs(history[-2])):
            converged = True
            break
    return EmFit(model=model, log_likelihoods=tuple(history), converged=converged)


def _mstep(
    model: HsmmNb,
    *,
    start_acc: np.ndarray,
    eta_acc: np.ndarray,
    xi_acc: np.ndarray,
    weight: np.ndarray,
    wy: np.ndarray,
    wyy: np.ndarray,
    duration_prior: float,
    fit_dispersion: bool,
) -> HsmmNb:
    """Closed-form updates, with every degenerate case falling back to the current value.

    A state no sequence visits has no sufficient statistics. Re-estimating it from a zero
    denominator produces ``nan`` and poisons every later iteration, so an unvisited state
    keeps its parameters and the fit stays interpretable instead of silently dying.
    """
    k = model.n_states
    start = _normalise_rows(start_acc[None, :], model.start[None, :])[0]
    trans = _normalise_rows(xi_acc * (1.0 - np.eye(k)), model.trans)
    durations = _normalise_rows(eta_acc + duration_prior, model.durations)

    visited = weight > 0.0
    safe_weight = np.where(visited, weight, 1.0)[:, None]
    mean = np.where(visited[:, None], wy / safe_weight, model.nb_mean)
    mean = np.maximum(mean, _MIN_MEAN)
    dispersion = model.nb_dispersion
    if fit_dispersion:
        second = np.where(visited[:, None], wyy / safe_weight, 0.0)
        var = second - mean * mean
        # Under- or equi-dispersed states are the Poisson limit, not an error.
        overdispersed = visited[:, None] & (var > mean * (1.0 + 1e-9))
        dispersion = np.where(
            overdispersed,
            mean * mean / np.where(overdispersed, var - mean, 1.0),
            np.where(visited[:, None], MAX_DISPERSION, model.nb_dispersion),
        )
        dispersion = np.clip(dispersion, MIN_DISPERSION, MAX_DISPERSION)
    return HsmmNb(
        start=start,
        trans=trans,
        durations=durations,
        nb_mean=mean,
        nb_dispersion=dispersion,
    )


def _normalise_rows(counts: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    ok = totals > 0.0
    return np.asarray(np.where(ok, counts / np.where(ok, totals, 1.0), fallback))


# -------------------------------------------------------------- construction helpers


def geometric_durations(p: np.ndarray, max_duration: int = MAX_DURATION) -> np.ndarray:
    """``(K, D)`` truncated, renormalised geometric dwell pmf, ``P(d) ~ (1-p)^(d-1) p``.

    A geometric dwell is exactly what an ordinary HMM implies, so this is also the
    construction that makes the HSMM-reduces-to-an-HMM cross-check possible. Truncation at
    ``D`` discards ``(1-p)^D`` of the mass, which is why that test picks a ``p`` large
    enough for the discarded tail to sit below float64 resolution, rather than papering
    over it with a loose tolerance.
    """
    prob = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if np.any((prob <= 0.0) | (prob > 1.0)):
        raise ValueError(f"p must be in (0, 1]; got {prob}")
    support = np.arange(max_duration, dtype=np.float64)[None, :]
    pmf = (1.0 - prob)[:, None] ** support * prob[:, None]
    return np.asarray(pmf / pmf.sum(axis=1, keepdims=True))


def seed_model(
    sequences: list[np.ndarray],
    n_states: int,
    rng: np.random.Generator,
    *,
    max_duration: int = MAX_DURATION,
    mean_dwell: float | None = None,
) -> HsmmNb:
    """A diffuse starting point for :func:`fit`, from the pooled moments of ``sequences``.

    States are separated only by a spread of emission means around the pooled mean, and
    every state's dwell distribution starts identical — so anything the fit then says
    about *dwell* was learned from the data rather than smuggled in through the
    initialisation. That matters here specifically: dwell is the property Rung 7 is judged
    on, and an informative dwell prior would let the test pass without the model working.
    """
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2; got {n_states}")
    pooled = np.concatenate([_as_counts(s) for s in sequences], axis=0)
    mu = np.maximum(pooled.mean(axis=0), 1.0)
    var = np.maximum(pooled.var(axis=0), mu * 1.001)
    r = np.clip(mu * mu / (var - mu), MIN_DISPERSION, MAX_DISPERSION)
    spread = np.geomspace(0.5, 2.0, n_states)[:, None]
    jitter = rng.uniform(0.9, 1.1, size=(n_states, pooled.shape[1]))
    dwell = float(max_duration) / 4.0 if mean_dwell is None else mean_dwell
    prob = np.full(n_states, min(1.0, 1.0 / max(dwell, 1.0)))
    off_diagonal = (1.0 - np.eye(n_states)) / (n_states - 1)
    return HsmmNb(
        start=np.full(n_states, 1.0 / n_states),
        trans=off_diagonal,
        durations=geometric_durations(prob, max_duration),
        nb_mean=mu[None, :] * spread * jitter,
        nb_dispersion=np.repeat(r[None, :], n_states, axis=0),
    )


def sample(
    model: HsmmNb, n_steps: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``(states, obs)`` from ``model``: ``(T,)`` int64 and ``(T, C)`` int64 counts.

    A module function rather than a method so that :class:`HsmmNb` stays a parameter
    container with inference on it, and so the fixture generator for the recovery test is
    obviously the same model definition the recursions use.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    states = np.empty(n_steps, dtype=np.int64)
    t = 0
    state = int(rng.choice(model.n_states, p=model.start))
    while t < n_steps:
        dwell = int(rng.choice(model.max_duration, p=model.durations[state])) + 1
        end = min(t + dwell, n_steps)
        states[t:end] = state
        t = end
        if t < n_steps:
            state = int(rng.choice(model.n_states, p=model.trans[state]))
    mu = model.nb_mean[states]
    r = model.nb_dispersion[states]
    obs = rng.negative_binomial(n=r, p=r / (r + mu))
    return states, np.asarray(obs, dtype=np.int64)
