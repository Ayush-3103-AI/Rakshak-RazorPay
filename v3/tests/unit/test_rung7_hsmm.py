"""T-0123 — the HSMM-NB inference core.

Every number a Rung 7 explanation will ever produce comes out of these recursions, so the
tests here are built around one rule: **the recursion is never allowed to be its own
witness.** Three independent computations check it.

1. ``test_two_observation_likelihood_matches_hand_computation`` — a closed form written
   out by hand in the docstring, evaluated with plain Python floats.
2. ``test_forward_matches_brute_force_enumeration`` — a linear-space enumeration over
   *segmentations*, which is the other standard way to define an HSMM. It shares no code
   with the residual-time recursion, so agreement between them is evidence that the
   reformulation is correct and not merely self-consistent.
3. ``test_geometric_durations_reduce_to_an_hmm`` — with geometric dwells the HSMM *is* an
   HMM, and a five-line log-space HMM forward is obviously correct in a way a
   duration-augmented recursion is not.

The third one is also run at ``T = 1000`` beside a naive linear-space forward, which is
the underflow case: the naive version returns exactly 0.0 and the log-space one is still
right. That test is the point of the ticket, not decoration.

Fixtures are deliberately tiny. A feature-materialisation job owns this machine.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from rakshak.explain.registry import NotAnExplainerError, Scorer, register
from rakshak.models.rung7_hsmm import (
    MAX_DURATION,
    STATE_NAMES,
    HsmmNb,
    fit,
    geometric_durations,
    sample,
    seed_model,
)

MODELS = Path(__file__).resolve().parents[2] / "src" / "rakshak" / "models"


# --------------------------------------------------------------------------- fixtures


def _tiny_model(max_duration: int = 3) -> HsmmNb:
    """K=2, C=1, small D. Every probability is a round number so the hand computation is
    checkable by eye."""
    durations = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])[:, :max_duration]
    durations = durations / durations.sum(axis=1, keepdims=True)
    return HsmmNb(
        start=np.array([0.7, 0.3]),
        trans=np.array([[0.0, 1.0], [1.0, 0.0]]),
        durations=durations,
        nb_mean=np.array([[3.0], [11.0]]),
        nb_dispersion=np.array([[2.0], [5.0]]),
    )


def _three_state_model() -> HsmmNb:
    trans = np.array([[0.0, 0.6, 0.4], [0.25, 0.0, 0.75], [0.5, 0.5, 0.0]])
    durations = np.array([[0.4, 0.35, 0.15, 0.1], [0.1, 0.2, 0.3, 0.4], [0.7, 0.2, 0.05, 0.05]])
    return HsmmNb(
        start=np.array([0.2, 0.5, 0.3]),
        trans=trans,
        durations=durations,
        nb_mean=np.array([[2.0, 1.0], [9.0, 4.0], [25.0, 2.0]]),
        nb_dispersion=np.array([[1.5, 3.0], [4.0, 2.0], [8.0, 1.0]]),
    )


# ----------------------------------------------------------------- independent oracles


def _brute_force_likelihood(model: HsmmNb, obs: np.ndarray) -> float:
    """Enumerate every segmentation of ``obs`` in linear space. No shared code.

    A segmentation is a run of ``(state, dwell)`` pairs with adjacent states differing,
    covering day 0 onwards, where the final segment is allowed to run past the end of the
    sequence — that last part is the right-censoring, and it is the piece a duration model
    is easiest to get wrong.
    """
    emissions = np.exp(model.log_emissions(obs))
    n_obs, n_states = emissions.shape
    total = 0.0
    frontier: list[tuple[int, int | None, float]] = [(0, None, 1.0)]
    while frontier:
        start_day, previous, acc = frontier.pop()
        for state in range(n_states):
            if previous is not None and state == previous:
                continue
            entry = model.start[state] if previous is None else model.trans[previous, state]
            if entry == 0.0:
                continue
            for dwell in range(1, model.max_duration + 1):
                p_dwell = model.durations[state, dwell - 1]
                if p_dwell == 0.0:
                    continue
                end = start_day + dwell
                weight = acc * entry * p_dwell
                for day in range(start_day, min(end, n_obs)):
                    weight *= emissions[day, state]
                if end >= n_obs:
                    total += weight
                else:
                    frontier.append((end, state, weight))
    return total


def _hmm_log_likelihood(
    start: np.ndarray, trans: np.ndarray, log_emissions: np.ndarray
) -> float:
    """A plain log-space HMM forward. Five lines, and obviously correct."""
    log_alpha = np.log(start) + log_emissions[0]
    log_trans = np.log(trans)
    for t in range(1, log_emissions.shape[0]):
        log_alpha = log_emissions[t] + logsumexp(log_alpha[:, None] + log_trans, axis=0)
    return float(logsumexp(log_alpha))


def _naive_linear_forward(model: HsmmNb, obs: np.ndarray) -> float:
    """The same residual-time recursion, in linear space. Here to be seen failing."""
    emissions = np.exp(model.log_emissions(obs))
    n_obs, n_states = emissions.shape
    alpha = model.start[:, None] * model.durations * emissions[0][:, None]
    pad = np.zeros((n_states, 1))
    for t in range(1, n_obs):
        entry = alpha[:, 0] @ model.trans
        continued = np.concatenate([alpha[:, 1:], pad], axis=1)
        alpha = emissions[t][:, None] * (continued + entry[:, None] * model.durations)
    return float(alpha.sum())


def _as_hmm(model: HsmmNb, p: np.ndarray) -> np.ndarray:
    """The transition matrix of the HMM whose dwell times are ``Geometric(p)``."""
    return (1.0 - p)[:, None] * np.eye(model.n_states) + p[:, None] * model.trans


# ------------------------------------------------------------------ correctness proofs


def test_two_observation_likelihood_matches_hand_computation() -> None:
    """With T=2 and D=2 the whole likelihood fits on one line.

    The chain is in state k on day 0 with residual r. If r = 1 the segment ends and a new
    state j != k begins on day 1 with some dwell, which marginalises to 1. If r = 2 the
    chain stays in k. So

        L = sum_k pi_k b_k(o_0) [ P_k(1) * sum_{j != k} A_kj b_j(o_1) + P_k(2) b_k(o_1) ]
    """
    model = _tiny_model(max_duration=2)
    obs = np.array([4, 12])
    emissions = np.exp(model.log_emissions(obs))

    expected = 0.0
    for k in range(2):
        switch = sum(
            model.trans[k, j] * emissions[1, j] for j in range(2) if j != k
        )
        expected += (
            model.start[k]
            * emissions[0, k]
            * (model.durations[k, 0] * switch + model.durations[k, 1] * emissions[1, k])
        )

    assert model.log_likelihood(obs) == pytest.approx(math.log(expected), rel=1e-12)


def test_forward_matches_brute_force_enumeration() -> None:
    """The AC's three-observation toy sequence, plus a wider model for good measure."""
    model = _tiny_model()
    obs = np.array([2, 3, 14])
    assert model.log_likelihood(obs) == pytest.approx(
        math.log(_brute_force_likelihood(model, obs)), rel=1e-12
    )
    # Pinned so a future edit that changes both implementations in the same direction is
    # still caught. Produced by this test at first green, on the fixture above.
    assert model.log_likelihood(obs) == pytest.approx(-7.996515584365415, rel=1e-12)


def test_forward_matches_brute_force_on_a_three_state_two_channel_model() -> None:
    model = _three_state_model()
    obs = np.array([[3, 1], [10, 5], [24, 2], [22, 3], [2, 0]])
    assert model.log_likelihood(obs) == pytest.approx(
        math.log(_brute_force_likelihood(model, obs)), rel=1e-11
    )


def test_forward_matches_brute_force_when_a_dwell_is_impossible() -> None:
    """A zero in the dwell pmf is a hard constraint, and ``log 0 = -inf`` is the case a
    log-space recursion is most likely to turn into a ``nan``."""
    model = _tiny_model()
    model = HsmmNb(
        start=model.start,
        trans=model.trans,
        durations=np.array([[0.0, 0.0, 1.0], [0.6, 0.0, 0.4]]),
        nb_mean=model.nb_mean,
        nb_dispersion=model.nb_dispersion,
    )
    obs = np.array([2, 3, 14, 4])
    value = model.log_likelihood(obs)
    assert np.isfinite(value)
    assert value == pytest.approx(math.log(_brute_force_likelihood(model, obs)), rel=1e-12)


def test_geometric_durations_reduce_to_an_hmm() -> None:
    """An HSMM with geometric dwells is an HMM. If it is not, the recursion is wrong.

    The HSMM truncates the dwell at D and the HMM does not, so they agree only up to the
    discarded tail ``(1-p)^D`` per segment. At D=60 and p>=0.45 that is 6e-16 — below
    float64 resolution — so the two must agree to numerical noise rather than to a
    tolerance chosen to make the test pass. (At D=12 the same test fails by 0.036 nats,
    which is the truncation being visible, not the recursion being wrong.)
    """
    rng = np.random.default_rng(20260901)
    p = np.array([0.5, 0.45])
    base = _tiny_model()
    model = HsmmNb(
        start=base.start,
        trans=base.trans,
        durations=geometric_durations(p, max_duration=MAX_DURATION),
        nb_mean=base.nb_mean,
        nb_dispersion=base.nb_dispersion,
    )
    obs = rng.integers(0, 20, size=25)
    reference = _hmm_log_likelihood(model.start, _as_hmm(model, p), model.log_emissions(obs))
    assert model.log_likelihood(obs) == pytest.approx(reference, abs=1e-9)


def test_long_sequence_underflows_a_linear_forward() -> None:
    """T=1000. The naive linear recursion returns exactly zero; the log one is still right.

    This is the ticket's central assertion. The reference is the independent log-space HMM
    from the reduction above, so the long case is checked against a computation that does
    not share the residual-time machinery, not merely asserted to be finite.
    """
    rng = np.random.default_rng(7)
    p = np.array([0.5, 0.45])
    base = _tiny_model()
    model = HsmmNb(
        start=base.start,
        trans=base.trans,
        durations=geometric_durations(p, max_duration=MAX_DURATION),
        nb_mean=base.nb_mean,
        nb_dispersion=base.nb_dispersion,
    )
    _, obs = sample(model, 1000, rng)

    assert _naive_linear_forward(model, obs) == 0.0, "fixture no longer underflows"

    value = model.log_likelihood(obs)
    assert np.isfinite(value)
    assert value < -700.0, "a sequence this cheap would not have underflowed"
    reference = _hmm_log_likelihood(model.start, _as_hmm(model, p), model.log_emissions(obs))
    assert value == pytest.approx(reference, abs=1e-6)


def test_forward_and_backward_agree_at_every_step() -> None:
    """``logsumexp_k,r (alpha_t + beta_t) = log L`` for every t, not just at the end.

    An off-by-one in the residual index still produces a plausible likelihood at T; it
    does not survive this.
    """
    rng = np.random.default_rng(11)
    model = _three_state_model()
    _, obs = sample(model, 60, rng)
    log_emissions = model.log_emissions(obs)
    alpha, _ = model._forward(log_emissions)
    beta = model._backward(log_emissions)
    total = model.log_likelihood(obs)

    per_step = logsumexp(alpha + beta, axis=(1, 2))
    assert np.allclose(per_step, total, atol=1e-8)

    gamma = model.posterior_states(obs)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(gamma >= -1e-12)


def test_exactly_one_segment_starts_on_the_first_day() -> None:
    """Some segment must begin on day 0, and the segment-start posterior has to know it."""
    rng = np.random.default_rng(3)
    model = _three_state_model()
    _, obs = sample(model, 40, rng)
    starts = model.posterior_segment_starts(obs)
    assert starts[0].sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(starts >= -1e-12)


# ------------------------------------------------------------------------ decoding


def test_viterbi_recovers_a_clean_path() -> None:
    """Well-separated emissions, long dwells: the MAP path should be the sampled one."""
    model = HsmmNb(
        start=np.array([1.0, 0.0]),
        trans=np.array([[0.0, 1.0], [1.0, 0.0]]),
        durations=geometric_durations(np.array([0.06, 0.06]), max_duration=40),
        nb_mean=np.array([[4.0], [80.0]]),
        nb_dispersion=np.array([[30.0], [30.0]]),
    )
    rng = np.random.default_rng(19)
    states, obs = sample(model, 150, rng)
    assert (model.decode(obs) == states).mean() > 0.95


def test_viterbi_cannot_produce_a_segment_the_duration_model_forbids() -> None:
    """The whole reason for an HSMM: a one-day blip cannot become a one-day state.

    Dwell mass is zero below five days, so no interior run in the decoded path may be
    shorter than that however tempting the emission on a single day is. An HMM has no way
    to express this constraint, which is exactly why it localises onset badly.
    """
    durations = np.zeros((2, 20))
    durations[:, 4:] = 1.0 / 16.0
    model = HsmmNb(
        start=np.array([1.0, 0.0]),
        trans=np.array([[0.0, 1.0], [1.0, 0.0]]),
        durations=durations,
        nb_mean=np.array([[5.0], [60.0]]),
        nb_dispersion=np.array([[20.0], [20.0]]),
    )
    obs = np.full(40, 5, dtype=np.int64)
    obs[20] = 300  # a single day that screams state 1
    path = model.decode(obs)

    boundaries = np.flatnonzero(np.diff(path)) + 1
    runs = np.diff(np.concatenate([[0], boundaries, [len(path)]]))
    interior = runs[1:-1] if len(runs) > 2 else np.array([], dtype=np.int64)
    assert np.all(interior >= 5), f"decoded runs {runs} violate the dwell support"


# ------------------------------------------------------------------------ EM


def _recovery_fixture() -> tuple[HsmmNb, list[np.ndarray]]:
    truth = HsmmNb(
        start=np.array([1.0, 0.0]),
        trans=np.array([[0.0, 1.0], [1.0, 0.0]]),
        durations=geometric_durations(np.array([1.0 / 5.0, 1.0 / 15.0]), max_duration=30),
        nb_mean=np.array([[5.0], [40.0]]),
        nb_dispersion=np.array([[10.0], [10.0]]),
    )
    rng = np.random.default_rng(2026)
    sequences = [sample(truth, 100, rng)[1] for _ in range(16)]
    return truth, sequences


def test_em_recovers_the_dwell_parameters() -> None:
    """Sample from a known HSMM, refit from a dwell-uninformative start, recover the dwell.

    The initialisation gives every state the *same* uniform dwell distribution, so a pass
    here cannot come from the seed: whatever the fit knows about dwell, it learned. States
    are matched by emission mean because EM has no idea which label is which.
    """
    truth, sequences = _recovery_fixture()
    init = HsmmNb(
        start=np.array([0.5, 0.5]),
        trans=np.array([[0.0, 1.0], [1.0, 0.0]]),
        durations=np.full((2, 30), 1.0 / 30.0),
        nb_mean=np.array([[9.0], [25.0]]),
        nb_dispersion=np.array([[3.0], [3.0]]),
    )
    result = fit(sequences, init, n_iter=40, tol=1e-7)

    order = np.argsort(result.model.nb_mean[:, 0])
    recovered = result.model.mean_dwell[order]
    expected = truth.mean_dwell[np.argsort(truth.nb_mean[:, 0])]
    assert recovered == pytest.approx(expected, rel=0.25), (
        f"dwell {recovered} against truth {expected}"
    )
    assert result.model.nb_mean[order, 0] == pytest.approx(truth.nb_mean[:, 0], rel=0.2)


def test_em_log_likelihood_is_monotone() -> None:
    """Plain EM must not go downhill. Dispersion is held fixed because its update is
    moment-matching rather than a maximisation step — see ``fit``'s docstring."""
    _, sequences = _recovery_fixture()
    rng = np.random.default_rng(5)
    init = seed_model(sequences, n_states=2, rng=rng, max_duration=20, mean_dwell=8.0)
    result = fit(sequences, init, n_iter=12, tol=0.0, fit_dispersion=False)
    trace = np.asarray(result.log_likelihoods)
    steps = np.diff(trace)
    assert np.all(steps >= -1e-6), f"EM went downhill: {steps}"
    assert trace[-1] > trace[0]


def test_fit_is_deterministic() -> None:
    """Same sequences, same init, same numbers. Determinism is a hard requirement."""
    _, sequences = _recovery_fixture()
    sequences = sequences[:4]  # determinism does not need the whole fixture
    init = seed_model(sequences, 2, np.random.default_rng(1), max_duration=20)
    first = fit(sequences, init, n_iter=3)
    second = fit(sequences, init, n_iter=3)
    assert first.log_likelihoods == second.log_likelihoods
    assert np.array_equal(first.model.durations, second.model.durations)


def test_fit_survives_a_state_no_sequence_visits() -> None:
    """A dead state keeps its parameters instead of becoming ``nan`` and poisoning the fit."""
    _, sequences = _recovery_fixture()
    init = HsmmNb(
        start=np.array([0.5, 0.5, 0.0]),
        trans=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]),
        durations=np.full((3, 20), 1.0 / 20.0),
        nb_mean=np.array([[6.0], [30.0], [5000.0]]),
        nb_dispersion=np.array([[5.0], [5.0], [5.0]]),
    )
    result = fit(sequences, init, n_iter=4)
    assert np.all(np.isfinite(result.model.nb_mean))
    assert np.all(np.isfinite(result.model.durations))
    assert np.isfinite(result.log_likelihoods[-1])


# ------------------------------------------------------------------ contracts & walls


def test_hsmm_is_not_reachable_from_the_scoring_path() -> None:
    """Prime Directive: Rung 7 explains, it does not score.

    ``Scorer`` is structural, so the check is too — ``HsmmNb`` must not grow a ``predict``.
    The core is not an ``Explainer`` either: the narrative that makes it one is #58's, and
    registering a stub now would put a name in the register that no reason string can cite.
    """
    model = _tiny_model()
    assert not isinstance(model, Scorer)
    assert not hasattr(model, "predict")
    with pytest.raises(NotAnExplainerError):
        register(model)  # type: ignore[arg-type]


def test_no_scoring_rung_imports_the_hsmm() -> None:
    """The other half of the wall: nothing in the scoring package may depend on Rung 7."""
    offenders = []
    for path in sorted(MODELS.glob("*.py")):
        if path.name == "rung7_hsmm.py":
            continue
        if "rung7_hsmm" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, f"scoring modules referencing Rung 7: {offenders}"


def test_module_imports_nothing_radioactive() -> None:
    """Prime Directive 3, checked locally as well as by the repo-wide AST gate."""
    tree = ast.parse((MODELS / "rung7_hsmm.py").read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not [m for m in modules if m.startswith(("rakshak.eval", "rakshak.generator"))]
    source = (MODELS / "rung7_hsmm.py").read_text(encoding="utf-8")
    for field in ("drift_onset_at", "persona_id", "risk_typology_id", "ground_truth"):
        assert field not in source.replace("``drift_onset_at``", ""), field


def test_state_names_are_the_four_declared_states() -> None:
    assert STATE_NAMES == ("HEALTHY", "RAMP", "EXFIL", "BURNT")
    assert MAX_DURATION == 60


# ------------------------------------------------------------------------ validation


def test_a_self_transition_is_refused() -> None:
    with pytest.raises(ValueError, match="zero diagonal"):
        HsmmNb(
            start=np.array([0.5, 0.5]),
            trans=np.array([[0.2, 0.8], [1.0, 0.0]]),
            durations=np.full((2, 3), 1.0 / 3.0),
            nb_mean=np.array([[1.0], [2.0]]),
            nb_dispersion=np.array([[1.0], [1.0]]),
        )


def test_non_count_observations_are_refused() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="counts"):
        model.log_likelihood(np.array([1.5, 2.0]))
    with pytest.raises(ValueError, match="counts"):
        model.log_likelihood(np.array([-1, 2]))


def test_channel_count_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="channels"):
        _tiny_model().log_likelihood(np.array([[1, 2], [3, 4]]))
