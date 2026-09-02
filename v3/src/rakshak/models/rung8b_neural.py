"""Rung 8b — a *neural* conditional intensity, replacing Rung 8's parametric Hawkes/NB fit.

Named in GitHub #66 (T-0132) and in ``configs/rung_roster.yaml`` as ``tpp_neural_intensity``.
It exists to answer one question and it is not a flattering one: **does more model capacity
fix what Rung 8 could not fix, or does it make the circularity objection worse?** GitHub #66
predicts worse, in its own text — *"a flexible neural intensity fits the generator's own
process even more exactly than a parametric one does, which makes the circularity objection
in #125 worse, not better"* — and ``LIMITATIONS.md`` §12.4 explains structurally why neither
form can win: the generator draws each day's *count* from a negative binomial and only then
places the events, which is a Cox process with an i.i.d. latent gamma multiplier per day. The
multiplier carries no history, so **no** conditional intensity — parametric, neural, or
otherwise — can absorb it.

This module does not touch ``rung8_tpp.py``. That rung's result is this one's baseline, and
ADR-V3-001's amendment says in as many words that no existing rung is to be rewritten onto
``torch``. What is shared is imported (``MIN_EVENTS``, ``nb_dispersion``) rather than copied.

Why ``torch`` is here at all
----------------------------
ADR-V3-001 was **REVERSED for this rung and one other** by lead decision on 2026-09-02; the
amendment at the foot of ``docs/adr/ADR-V3-001-no-autograd.md`` is the authority and it is
worth reading before extending this file. It records two things this module inherits: the
ADR's own revisit trigger did **not** fire, and §Reversal item 3 — that the label constraint
no longer binds — is **NOT SATISFIED and waived**. It still binds at ~234 trainable positive
merchants. ``N_PARAMETERS`` below is 209. That comparison is reported, not buried.

Autograd is load-bearing here rather than decorative: the model parameterises the
**cumulative** hazard and obtains the intensity by differentiating it. That derivative is
what ``torch.autograd.grad`` computes, inside the training loop, with ``create_graph=True``
so the log-likelihood's own gradient flows back through it. Written by hand it would be a
second analytic derivative to keep correct beside the first; that is exactly the
"hand-written backpropagation for one layer" cost the ADR body priced and declined.

The model
---------
Following Omi, Ueda & Aihara (NeurIPS 2019), *Fully Neural Network based Model for General
Temporal Point Processes*: parameterise the **integrated** conditional intensity directly
with a network that is monotone in elapsed time, and recover the intensity by
differentiation. For an inter-arrival gap ``tau`` after event ``k`` with history embedding
``h_k``::

    Phi(tau, h)  = softplus( tanh( g(tau) @ W_tau+ + h @ W_h + b ) @ w_out+ + b_out )
    Lambda(tau|h) = Phi(tau, h) - Phi(0, h)          >= 0, and Lambda(0|h) = 0
    lambda(tau|h) = d Lambda / d tau                  > 0, by autograd

with ``g(tau) = [tau * 1440, log1p(tau * 1440)]`` — both strictly increasing in ``tau``, in
minutes — and ``+`` marking weights passed through ``softplus`` so they are non-negative.
Monotone transforms composed with non-negative weights are monotone, so ``Lambda`` increases
in ``tau`` **by construction** and its derivative is a valid intensity. Nothing has to be
checked at runtime for that to hold; ``tests/unit/test_rung8b.py`` checks it anyway, because
a constraint that is only argued for in a docstring is not a constraint.

Two monotone transforms of ``tau`` rather than one because the gaps span five orders of
magnitude — seconds to days — and a linear input alone gives the first layer nothing to
resolve the short end with.

The history embedding is nine numbers, computed in closed form rather than by an RNN::

    log1p(R_j(t_k))  for six fixed decay rates beta_j  (2880, 1440, 480, 96, 24, 4 per day)
    log1p(gap to the previous event, in minutes)
    sin(2 pi * hour-of-day),  cos(2 pi * hour-of-day)

``R_j(t_k) = sum_{i <= k} exp(-beta_j (t_k - t_i))`` is the same self-exciting memory
Rung 8's Hawkes term carries, computed by the same stable recursion, at six timescales
instead of one fitted one. The grid brackets the manifest's ``hawkes_decay_minutes: 3.0``
(``beta = 480``) by two decades either side.

**This is strictly more expressive than Rung 8's intensity, which is the point.** The
parametric form is close to the special case: one timescale, hazard linear in ``tau``,
excitation entering linearly. Here six timescales enter a non-linear monotone hazard whose
shape in ``tau`` is learned per merchant. If capacity were the binding constraint, this is
the model that would show it.

**A GRU history encoder was tried first and rejected on cost, not on principle.** Measured
on this machine: ``torch.nn.GRU`` forward+backward over one merchant's baseline sequence is
200–550 ms per epoch, so 200 epochs x 1,191 merchants is 13–36 hours for one mitigation run.
The closed-form memory above is 2–3 s per merchant. The recurrence being fixed rather than
learned is a real reduction in capacity and is stated as one — but it is a reduction relative
to a *neural* encoder, not relative to Rung 8, which this still strictly contains.

What is fitted, and on what
---------------------------
One network **per merchant**, on that merchant's own baseline window — the same window,
the same events and the same ``compensator_increments`` contract Rung 8 uses, so
``eval.metrics.tpp_rescaled_ks`` sees the two rungs on identical framing. That is a
like-for-like swap of the intensity and nothing else. It is also 209 parameters where
Rung 8 had 3, on a window that typically holds 180–1,500 events.

Optimiser settings are **declared here and were fixed before any mitigation was run**,
chosen only on the simulated-recovery check in ``tests/unit/test_rung8b.py`` (which is
acceptance criterion 1, a self-consistency claim). Adam, ``lr = 0.05``, 200 epochs,
``weight_decay = 1e-4``, one fixed init seed. Prime Directive 5: nothing below is re-chosen
after seeing a null-run or validation number.

Cost and state
--------------
Full-batch Adam on dense ``(n, 9)`` and ``(n, 2)`` matmuls: ~2–3 s per merchant fit on one
CPU core, float64 throughout, CPU-only — ``torch.cuda`` is never touched. As with Rung 8 the
fit is offline; the per-epoch cost is one compensator pass plus one KS call. No online form
is provided, and ``MerchantState`` is not touched.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from rakshak.models.rung8_tpp import MIN_EVENTS, nb_dispersion

__all__ = [
    "BETAS_PER_DAY",
    "EPOCHS",
    "FIT_SEED",
    "HIDDEN",
    "LEARNING_RATE",
    "N_HISTORY",
    "N_PARAMETERS",
    "TIME_SCALE",
    "WEIGHT_DECAY",
    "HazardNet",
    "NeuralIntensityFit",
    "compensator_increments",
    "fit",
    "history_features",
]

F64 = npt.NDArray[np.float64]

#: Fixed decay rates for the self-exciting memory, per day. 480 is the manifest's own
#: ``hawkes_decay_minutes: 3.0``; the grid spans a 30-second half-life to a 4-hour one, two
#: decades either side of it. Fixed and not fitted: the network weights the six channels, so
#: the timescales are a basis rather than parameters, and a basis cannot wander into the
#: degenerate ``beta -> 0`` corner a fitted decay can.
BETAS_PER_DAY: F64 = np.array([2880.0, 1440.0, 480.0, 96.0, 24.0, 4.0], dtype=np.float64)

#: Six memory channels + log gap + two hour-of-day harmonics.
N_HISTORY = int(BETAS_PER_DAY.size) + 3

#: Width of the single monotone hidden layer.
HIDDEN = 16

#: ``tau`` is in days; the network sees minutes. Inter-arrivals on a live merchant are
#: minutes to hours, and a unit of "days" puts every observation inside the first 1e-3 of
#: the input range, where ``tanh`` is linear and the layer has nothing to work with.
TIME_SCALE = 1440.0

#: Declared in advance (Prime Directive 5) and chosen only on the simulated-recovery check.
#: ``lr = 0.02`` is visibly under-trained at 200 epochs (KS 0.028, p 0.005 on the same
#: process where 0.05 gives KS 0.016, p 0.34); 0.05 converges. ``weight_decay`` is a
#: standard default carried so that a 209-parameter model is not reported as unregularised —
#: on the recovery check it moves KS by 0.0002 and it was kept rather than tuned away.
EPOCHS = 200
LEARNING_RATE = 0.05
WEIGHT_DECAY = 1e-4
FIT_SEED = 20260902

#: ``W_tau (2 x H) + W_h (N_HISTORY x H) + b (H) + w_out (H) + b_out``. Read this next to
#: ADR-V3-001's ~234 trainable positive merchants and next to Rung 8's 3 parameters. The
#: ADR records the label constraint as an unmet precondition, waived by the lead, and this
#: is the number that makes the waiver concrete.
N_PARAMETERS = 2 * HIDDEN + N_HISTORY * HIDDEN + HIDDEN + HIDDEN + 1


@contextmanager
def _single_threaded() -> Iterator[None]:
    """Pin ``torch`` to one intra-op thread for the duration, then restore the caller's.

    Not a performance tweak — the matrices here are ``(n, 9)`` and ``(n, 2)``, far below the
    size where intra-op parallelism pays for its own synchronisation, so one thread is if
    anything faster. It is a **determinism** requirement. A multi-threaded float64 reduction
    splits the sum differently depending on the thread count, and while the difference is
    ~1e-15 per step, 200 Adam steps amplify it into a visibly different local optimum:
    measured on the simulated-recovery process, four threads give KS 0.0155 and one thread
    KS 0.0203 from the identical seed and data. Both pass; a rung whose reported number
    depends on how busy the machine was does not.

    Restoring the previous value matters because this module is imported into processes that
    run other rungs — LightGBM's and Rung 5b's matmuls are large enough to want the threads.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@dataclass(frozen=True, slots=True)
class HazardNet:
    """The monotone cumulative-hazard network's weights, detached from the graph.

    ``w_tau`` and ``w_out`` are stored **raw**: ``softplus`` is applied on use, which is
    what makes the effective weights non-negative and therefore ``cumulative`` monotone in
    ``tau``. Storing the raw parameters rather than their softplus keeps the stored object
    and the object the optimiser saw identical.
    """

    w_tau: torch.Tensor
    w_history: torch.Tensor
    bias: torch.Tensor
    w_out: torch.Tensor
    bias_out: torch.Tensor

    def cumulative(self, tau: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``Lambda(tau | h)`` — the compensator increment. Non-negative, zero at ``tau=0``."""
        return _cumulative(tau, h, self.w_tau, self.w_history, self.bias, self.w_out, self.bias_out)


@dataclass(frozen=True, slots=True)
class NeuralIntensityFit:
    """One merchant's fitted neural intensity, plus what it still could not represent.

    ``nb_dispersion`` and ``nb_fano`` are carried for the same reason ``HawkesNbFit`` carries
    them and are computed by the same function: the daily-count overdispersion is the part
    of the generator no conditional intensity can hold, and adding capacity does not change
    that. Reporting a neural fit without it beside would be reporting the upgrade as if the
    upgrade addressed the diagnosis.
    """

    net: HazardNet
    mean: F64
    std: F64
    nb_dispersion: float
    nb_fano: float
    loglik: float
    n_events: int
    horizon_days: float
    n_parameters: int
    epochs: int
    converged: bool


def _phi(
    tau: torch.Tensor,
    h: torch.Tensor,
    w_tau: torch.Tensor,
    w_history: torch.Tensor,
    bias: torch.Tensor,
    w_out: torch.Tensor,
    bias_out: torch.Tensor,
) -> torch.Tensor:
    """``Phi(tau, h)``, monotone increasing in ``tau`` by construction. See the docstring."""
    g = torch.stack([tau * TIME_SCALE, torch.log1p(tau * TIME_SCALE)], dim=-1)
    inner = g @ torch.nn.functional.softplus(w_tau) + h @ w_history + bias
    outer = torch.tanh(inner) @ torch.nn.functional.softplus(w_out) + bias_out
    return torch.nn.functional.softplus(outer)


def _cumulative(
    tau: torch.Tensor,
    h: torch.Tensor,
    w_tau: torch.Tensor,
    w_history: torch.Tensor,
    bias: torch.Tensor,
    w_out: torch.Tensor,
    bias_out: torch.Tensor,
) -> torch.Tensor:
    """``Lambda(tau|h) = Phi(tau,h) - Phi(0,h)``.

    The subtraction is what pins ``Lambda(0|h) = 0``, which the time-rescaling theorem
    requires and which ``softplus``'s strictly positive output would otherwise violate. It
    preserves monotonicity because ``Phi(0, h)`` does not depend on ``tau``.
    """
    args = (w_tau, w_history, bias, w_out, bias_out)
    return _phi(tau, h, *args) - _phi(torch.zeros_like(tau), h, *args)


def history_features(times_days: F64) -> F64:
    """``(n, N_HISTORY)`` history embedding at each event, using only the past and the present.

    Row ``k`` is measurable at ``t_k``: the memory channels include event ``k`` itself, which
    is correct — the embedding conditions the hazard for the interval that *starts* at
    ``t_k``, and event ``k`` has happened by then.

    The recursion is the one ``rung8_tpp._recursions`` uses and for the same reason: the
    closed form ``exp(-beta t_k) * sum exp(beta t_i)`` overflows at ``beta = 2880`` over a
    30-day window, while every factor in the recursion is in [0, 1].

    **History before ``times_days[0]`` is dropped**, so every channel starts at 1 — the
    first event exciting itself. Over a window longer than a few multiples of ``1/beta`` that
    is nothing; the slowest channel here has a 4-hour timescale, so a caller passing a window
    shorter than about a day is measuring the initial condition. ``compensator_increments``
    carries the same caveat as Rung 8's does, for the same reason.
    """
    times = np.asarray(times_days, dtype=np.float64)
    n = times.size
    memory = np.zeros((n, BETAS_PER_DAY.size), dtype=np.float64)
    if n == 0:
        return np.zeros((0, N_HISTORY), dtype=np.float64)
    gaps = np.diff(times)
    current = np.ones_like(BETAS_PER_DAY)
    memory[0] = current
    for k in range(1, n):
        current = np.exp(-BETAS_PER_DAY * gaps[k - 1]) * current + 1.0
        memory[k] = current
    # The first event has no predecessor; one minute is the placeholder and it is the only
    # row where this column is not the observed gap. It is one row in hundreds and the
    # alternative -- dropping the event -- would change n between the two rungs.
    previous_gap = np.concatenate([[1.0 / TIME_SCALE], gaps])
    hour = np.mod(times, 1.0)
    return np.concatenate(
        [
            np.log1p(memory),
            np.log1p(previous_gap * TIME_SCALE)[:, None],
            np.sin(2.0 * np.pi * hour)[:, None],
            np.cos(2.0 * np.pi * hour)[:, None],
        ],
        axis=1,
    )


def _standardised(features: F64) -> tuple[F64, F64, F64]:
    """Per-merchant z-scoring of the history embedding. Returns ``(z, mean, std)``.

    Per merchant and not per population: the fit is per merchant, so a population statistic
    would put every other merchant's data inside one merchant's fit. A channel that never
    varies (a merchant whose slowest memory channel is saturated) gets ``std = 1`` rather
    than a division by zero.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (features - mean) / std, mean, std


def _validate(times: F64, horizon_days: float) -> None:
    if times.size < MIN_EVENTS:
        raise ValueError(
            f"{times.size} events is below MIN_EVENTS={MIN_EVENTS}; {N_PARAMETERS} "
            "parameters are not identified from that window. Skip the merchant rather "
            "than fitting it."
        )
    if not np.all(np.diff(times) >= 0.0):
        raise ValueError("times_days must be sorted ascending")
    if times[0] < 0.0 or times[-1] > horizon_days:
        raise ValueError(
            f"times_days must lie in [0, {horizon_days}]; got "
            f"[{times[0]!r}, {times[-1]!r}]. Times are relative to the window start."
        )


def fit(
    times_days: F64,
    *,
    horizon_days: float,
    daily_counts: npt.NDArray[np.integer] | F64 | None = None,
    seed: int = FIT_SEED,
    epochs: int = EPOCHS,
) -> NeuralIntensityFit:
    """Maximise the point-process log-likelihood over the network weights, by Adam.

    ``times_days`` must be sorted, non-negative, and measured in days from the **start of
    the window**, which must be a UTC day boundary — otherwise ``mod(t, 1)`` is not the hour
    of day and the two harmonic features are at the wrong phase. Same contract as
    ``rung8_tpp.fit``, deliberately: the two rungs are handed the identical window.

    The objective is the standard TPP negative log-likelihood::

        -[ sum_k log lambda(tau_k | h_{k-1}) - sum_k Lambda(tau_k | h_{k-1})
           - Lambda(T - t_n | h_n) ]

    normalised by the number of gaps so the learning rate means the same thing on a merchant
    with 200 events and one with 2,000. The final term is the censored interval from the last
    event to the horizon — Rung 8's likelihood carries the same term and dropping it here
    would make the two objectives different objects.

    ``lambda`` comes from ``torch.autograd.grad(..., create_graph=True)``: it is a derivative
    of the network output that must itself stay differentiable, because it appears inside the
    loss. That is the whole reason this rung needs an autograd framework.

    Seeded through an explicit ``torch.Generator``; torch's global RNG is never touched.
    """
    times = np.asarray(times_days, dtype=np.float64)
    _validate(times, horizon_days)

    raw = history_features(times)
    z, mean, std = _standardised(raw)
    h = torch.as_tensor(z, dtype=torch.float64)
    tau = torch.as_tensor(np.diff(times), dtype=torch.float64)
    tail = torch.as_tensor([float(horizon_days) - float(times[-1])], dtype=torch.float64)
    h_past, h_last = h[:-1], h[-1:]

    generator = torch.Generator().manual_seed(int(seed))

    def draw(*shape: int) -> torch.Tensor:
        # Uniform(-0.5, 0.5) on every parameter, generator-fed. A fan-in scheme buys nothing
        # at this width and would add a second thing to keep identical between runs.
        tensor = torch.empty(*shape, dtype=torch.float64).uniform_(-0.5, 0.5, generator=generator)
        return tensor.requires_grad_(True)

    w_tau = draw(2, HIDDEN)
    w_history = draw(N_HISTORY, HIDDEN)
    bias = draw(HIDDEN)
    w_out = draw(HIDDEN)
    bias_out = draw(1)
    weights = [w_tau, w_history, bias, w_out, bias_out]

    optimiser = torch.optim.Adam(weights, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loglik = float("nan")
    with _single_threaded():
        for _ in range(int(epochs)):
            optimiser.zero_grad(set_to_none=True)
            t = tau.detach().requires_grad_(True)
            increments = _cumulative(t, h_past, *weights)
            (intensity,) = torch.autograd.grad(increments.sum(), t, create_graph=True)
            # clamp_min, not an eps-add: the intensity is positive by construction and the
            # clamp only guards float64 underflow on a gap the network has driven flat.
            total = (
                torch.log(intensity.clamp_min(1e-300)).sum()
                - increments.sum()
                - _cumulative(tail, h_last, *weights).sum()
            )
            loss = -total / tau.numel()
            # torch ships `Tensor.backward` untyped, so --strict flags the call, not the code.
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            loglik = -float(loss.detach()) * int(tau.numel())

    dispersion, fano = (
        nb_dispersion(daily_counts) if daily_counts is not None else (float("nan"),) * 2
    )
    return NeuralIntensityFit(
        net=HazardNet(
            w_tau=w_tau.detach(),
            w_history=w_history.detach(),
            bias=bias.detach(),
            w_out=w_out.detach(),
            bias_out=bias_out.detach(),
        ),
        mean=mean,
        std=std,
        nb_dispersion=dispersion,
        nb_fano=fano,
        loglik=loglik,
        n_events=int(times.size),
        horizon_days=float(horizon_days),
        n_parameters=N_PARAMETERS,
        epochs=int(epochs),
        # Adam has no convergence test to report. This says only that the objective is a
        # number -- a NaN loss is the failure mode worth refusing downstream, and calling it
        # `converged` for symmetry with HawkesNbFit while meaning more would be dishonest.
        converged=bool(np.isfinite(loglik)),
    )


def compensator_increments(times_days: F64, fitted: NeuralIntensityFit) -> F64:
    """``Lambda_k`` between consecutive events — the input ``tpp_rescaled_ks`` takes.

    Signature and semantics are ``rung8_tpp.compensator_increments``'s, unchanged: ``n - 1``
    non-negative values, one per inter-arrival interval, from a fit produced on some earlier
    window. That identical contract is the whole basis of the comparison — the two rungs
    differ in the intensity and in nothing else that reaches the locked metric.

    ``Lambda_k = Lambda(t_k - t_{k-1} | h_{k-1})`` is read straight off the network, because
    the network parameterises the integral rather than the intensity. There is no quadrature
    here and no integration error to bound.

    **History before ``times_days[0]`` is dropped**, exactly as in Rung 8: the memory
    channels restart at 1. The slowest channel decays with a 4-hour timescale, so a window
    of a day or more is unaffected and a shorter one should not be passed.
    """
    times = np.asarray(times_days, dtype=np.float64)
    if times.size < 2:
        return np.empty(0, dtype=np.float64)
    z = (history_features(times) - fitted.mean) / fitted.std
    with _single_threaded(), torch.no_grad():
        increments = fitted.net.cumulative(
            torch.as_tensor(np.diff(times), dtype=torch.float64),
            torch.as_tensor(z[:-1], dtype=torch.float64),
        )
    # Clipped at 0 only against float round-off on a quantity that is non-negative by
    # construction: tpp_rescaled_ks raises on a negative increment, and it is right to.
    return np.asarray(np.maximum(increments.numpy(), 0.0), dtype=np.float64)
