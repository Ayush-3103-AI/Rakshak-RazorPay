"""T-0002 — the three "Done when" clauses for the hand-written HMM (FR-012, FR-013).

The toy generator in this file is a **throwaway spike fixture**. It exists only
to answer kill criterion K1 before the real generator is built. It must not be
imported by, or reused in, `src/rakshak/generator/` (T-0003).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from rakshak.cli import seed_everything
from rakshak.config import ARI_RECOVERY_THRESHOLD, SEED
from rakshak.models.hmm import HMM

# Toy ground truth: 3 sticky states, 2-D emissions, separated by ~3 sigma.
TRUE_PI = np.array([1.0, 0.0, 0.0])
TRUE_A = np.array(
    [
        [0.94, 0.05, 0.01],
        [0.04, 0.92, 0.04],
        [0.01, 0.05, 0.94],
    ]
)
TRUE_MU = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
TRUE_SD = np.full((3, 2), 1.0)


def sample_toy(
    rng: np.random.Generator, n_seq: int, n_obs: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Draw sequences from the toy HMM above.

    Args:
        rng: Seeded generator.
        n_seq: Number of sequences (stand-ins for merchants).
        n_obs: Observations per sequence.

    Returns:
        ``(sequences, paths)`` — observation arrays of shape (n_obs, 2) and the
        matching ground-truth state arrays of shape (n_obs,).
    """
    sequences, paths = [], []
    for _ in range(n_seq):
        z = np.empty(n_obs, dtype=np.int64)
        z[0] = rng.choice(3, p=TRUE_PI)
        for t in range(1, n_obs):
            z[t] = rng.choice(3, p=TRUE_A[z[t - 1]])
        x = TRUE_MU[z] + rng.normal(scale=TRUE_SD[z])
        sequences.append(x)
        paths.append(z)
    return sequences, paths


@pytest.fixture(scope="module")
def fitted() -> tuple[HMM, list[float], np.ndarray, np.ndarray]:
    """Fit the HMM once on the toy data; shared by the monotonicity and ARI tests."""
    rng = seed_everything(SEED)
    sequences, paths = sample_toy(rng, n_seq=20, n_obs=200)
    model = HMM(n_states=3, n_features=2)
    history = model.fit(sequences, rng=rng)
    decoded = np.concatenate([model.viterbi(x) for x in sequences])
    return model, history, decoded, np.concatenate(paths)


def test_baum_welch_loglik_is_monotone(fitted) -> None:
    """Done-when (a): log-likelihood never decreases across EM iterations."""
    _, history, _, _ = fitted
    assert len(history) >= 3, "EM stopped too early to say anything about monotonicity"
    deltas = np.diff(history)
    assert (deltas >= -1e-6).all(), f"log-likelihood decreased: {history}"


def test_viterbi_matches_brute_force() -> None:
    """Done-when (b): exact agreement with enumeration of all 3^5 paths."""
    rng = seed_everything(SEED)
    model = HMM(n_states=3, n_features=2)
    # Deliberately asymmetric, non-degenerate parameters — a uniform model
    # would make this test pass for the wrong reason.
    model.log_pi = np.log(np.array([0.5, 0.3, 0.2]))
    model.log_A = np.log(
        np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.25, 0.25, 0.5]])
    )
    model.mu = TRUE_MU.copy()
    model.var = np.array([[1.0, 1.5], [2.0, 1.0], [1.2, 1.2]])

    X = rng.normal(loc=1.0, scale=2.0, size=(5, 2))
    log_b = model.log_emission(X)

    best_path, best_score = None, -np.inf
    for path in itertools.product(range(3), repeat=5):
        score = model.log_pi[path[0]] + log_b[0, path[0]]
        for t in range(1, 5):
            score += model.log_A[path[t - 1], path[t]] + log_b[t, path[t]]
        if score > best_score:
            best_path, best_score = path, score

    assert model.viterbi(X).tolist() == list(best_path)


def test_state_recovery_ari(fitted) -> None:
    """Done-when (c): ARI between recovered and true states clears FR-013's gate."""
    _, _, decoded, truth = fitted
    ari = adjusted_rand_score(truth, decoded)
    assert ari > ARI_RECOVERY_THRESHOLD, f"ARI {ari:.3f} <= {ARI_RECOVERY_THRESHOLD} (FR-013)"


def test_fit_rejects_empty_sequence_list() -> None:
    """`fit` on nothing must raise, not silently return an untrained model."""
    with pytest.raises(ValueError):
        HMM(n_states=3, n_features=2).fit([])
