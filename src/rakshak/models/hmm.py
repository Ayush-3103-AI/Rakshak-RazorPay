"""Hand-written Hidden Markov Model with diagonal-covariance Gaussian emissions.

Implements forward, backward, Viterbi and Baum-Welch, all in log space (FR-012).
Written by hand rather than taken from `hmmlearn` per ADR-0001.

Maths: `07-math.md` §1. Pseudocode: `08-pseudocode.md` §D.

Conventions used throughout:
    K  number of hidden states
    D  emission dimensionality
    T  length of one observation sequence
    X  observation array, shape (T, D), dtype float64, arbitrary units
       (in production these are the within-merchant standardised window
       features of `features/`, i.e. dimensionless z-scores)

Every quantity named `log_*` is a natural logarithm. Log-likelihoods are in
nats, never bits.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

# Numerical guards, all mandated by 07-math.md §1 "Numerical stability notes".
VAR_FLOOR: float = 1e-6
"""Added to every diagonal variance each M-step, so a low-occupancy state
cannot collapse to a delta function and drive the log-likelihood to +inf."""

TRANSITION_FLOOR: float = 1e-8
"""Minimum transition probability, so no transition becomes permanently
impossible once EM has driven it to zero."""

MIN_OCCUPANCY: float = 1.0
"""A state whose total posterior occupancy falls below this (in expected
observation counts) is reinitialised rather than left to die."""

UNLABELLED: int = -1
"""Sentinel in a label sequence: this timestep's state is unknown, so its
posterior stays free. Everything not on the TRAINING split must be this."""

LABEL_CLAMP_LOG: float = -1e6
"""Log-density added to every state a label forbids (T-0004b, item 2). Finite
rather than -inf so that no forward/backward term can become inf - inf; at this
magnitude ``exp`` underflows to exactly 0, so the clamp is hard in practice."""


def _log_normalise(log_w: np.ndarray, axis: int) -> np.ndarray:
    """Normalise log-weights so that ``exp`` sums to one along ``axis``.

    Args:
        log_w: Unnormalised log-weights, any shape.
        axis: Axis along which the normalised weights must sum to one.

    Returns:
        Array of the same shape as ``log_w``, normalised in log space.
    """
    return log_w - logsumexp(log_w, axis=axis, keepdims=True)


class HMM:
    """Gaussian-emission HMM with diagonal covariance.

    Diagonal, not full: with D around 15 and per-state sample counts in the
    hundreds a full covariance is under-determined and Baum-Welch produces
    singular matrices (07-math.md §1).

    Attributes:
        n_states: K, the number of hidden states.
        n_features: D, the emission dimensionality.
        log_pi: Log initial-state distribution, shape (K,), nats.
        log_A: Log transition matrix, shape (K, K), nats. Row i sums to one
            in probability space: ``log_A[i, j] = log P(z_t = j | z_{t-1} = i)``.
        mu: Emission means, shape (K, D), same units as the observations.
        var: Emission variances (diagonal covariance), shape (K, D),
            squared observation units. Strictly positive.
    """

    def __init__(self, n_states: int, n_features: int) -> None:
        """Create an HMM with uniform initial and near-diagonal transitions.

        Parameters are placeholders until `fit` is called; `fit` overwrites
        all four from a k-means initialisation.

        Args:
            n_states: K, number of hidden states. Must be >= 1.
            n_features: D, emission dimensionality. Must be >= 1.
        """
        if n_states < 1 or n_features < 1:
            raise ValueError("n_states and n_features must both be >= 1")
        self.n_states = n_states
        self.n_features = n_features
        self.log_pi: np.ndarray = np.full(n_states, -np.log(n_states))
        self.log_A: np.ndarray = _log_normalise(
            np.log(np.eye(n_states) * 0.9 + 0.1 / n_states), axis=1
        )
        self.mu: np.ndarray = np.zeros((n_states, n_features))
        self.var: np.ndarray = np.ones((n_states, n_features))

    # ------------------------------------------------------------------
    # Emissions
    # ------------------------------------------------------------------

    def log_emission(self, X: np.ndarray) -> np.ndarray:
        """Log density of each observation under each state's Gaussian.

        Args:
            X: Observations, shape (T, D).

        Returns:
            Array of shape (T, K), in nats: entry (t, k) is
            ``log N(x_t | mu_k, diag(var_k))``.
        """
        X = np.asarray(X, dtype=float)
        # (T, 1, D) against (1, K, D) -> (T, K, D), summed over D.
        dev2 = (X[:, None, :] - self.mu[None, :, :]) ** 2
        quad = dev2 / self.var[None, :, :]
        norm = np.log(2.0 * np.pi * self.var)[None, :, :]
        return -0.5 * (norm + quad).sum(axis=2)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward recursion in log space. This is the online belief update.

        Args:
            X: Observations, shape (T, D).

        Returns:
            ``(log_alpha, loglik)`` where ``log_alpha`` has shape (T, K) with
            ``log_alpha[t, k] = log P(x_1..x_t, z_t = k)`` in nats, and
            ``loglik`` is the sequence log-likelihood ``log P(x_1..x_T)`` in nats.
        """
        return self._forward_logb(self.log_emission(X))

    def _forward_logb(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        """`forward` on a pre-computed (T, K) log-emission array."""
        n_obs = log_b.shape[0]
        log_alpha = np.empty_like(log_b)
        log_alpha[0] = self.log_pi + log_b[0]
        for t in range(1, n_obs):
            log_alpha[t] = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + self.log_A, axis=0)
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def backward(self, X: np.ndarray) -> np.ndarray:
        """Backward recursion in log space.

        Args:
            X: Observations, shape (T, D).

        Returns:
            ``log_beta`` of shape (T, K) in nats, with
            ``log_beta[t, k] = log P(x_{t+1}..x_T | z_t = k)`` and
            ``log_beta[T-1, :] = 0``.
        """
        return self._backward_logb(self.log_emission(X))

    def _backward_logb(self, log_b: np.ndarray) -> np.ndarray:
        """`backward` on a pre-computed (T, K) log-emission array."""
        n_obs = log_b.shape[0]
        log_beta = np.zeros_like(log_b)
        for t in range(n_obs - 2, -1, -1):
            log_beta[t] = logsumexp(self.log_A + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """MAP state path — the audit trail behind every reason string.

        Args:
            X: Observations, shape (T, D).

        Returns:
            Integer array of shape (T,) with values in ``[0, K)``: the single
            most probable joint state sequence, not the per-timestep argmax.
        """
        log_b = self.log_emission(X)
        n_obs = log_b.shape[0]
        delta = np.empty_like(log_b)
        psi = np.zeros((n_obs, self.n_states), dtype=np.int64)
        delta[0] = self.log_pi + log_b[0]
        for t in range(1, n_obs):
            scores = delta[t - 1][:, None] + self.log_A
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = log_b[t] + scores.max(axis=0)
        path = np.empty(n_obs, dtype=np.int64)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(n_obs - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def posterior(self, X: np.ndarray) -> np.ndarray:
        """Smoothed state posterior gamma.

        Args:
            X: Observations, shape (T, D).

        Returns:
            Array of shape (T, K), each row a probability distribution:
            ``P(z_t = k | x_1..x_T)``.
        """
        log_alpha, _ = self.forward(X)
        log_beta = self.backward(X)
        return np.exp(_log_normalise(log_alpha + log_beta, axis=1))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _init_from_data(self, X_pooled: np.ndarray, rng: np.random.Generator) -> None:
        """Initialise mu/var by k-means on pooled emissions; pi/A stay as constructed.

        Args:
            X_pooled: All observations from all sequences stacked, shape (N, D).
            rng: Generator used only to seed k-means, so the fit is reproducible.
        """
        seed = int(rng.integers(0, 2**31 - 1))
        labels = KMeans(n_clusters=self.n_states, n_init=10, random_state=seed).fit_predict(
            X_pooled
        )
        pooled_var = X_pooled.var(axis=0) + VAR_FLOOR
        for k in range(self.n_states):
            members = X_pooled[labels == k]
            self.mu[k] = members.mean(axis=0) if len(members) else X_pooled.mean(axis=0)
            self.var[k] = pooled_var

    def fit(
        self,
        sequences: list[np.ndarray],
        max_iter: int = 100,
        tol: float = 1e-4,
        rng: np.random.Generator | None = None,
    ) -> list[float]:
        """Fully unsupervised Baum-Welch (EM), pooled across all sequences.

        Sufficient statistics are accumulated over every sequence before a
        single M-step, so all merchants in a segment share one parameter set.

        This is the ablation baseline for T-0011. It is a thin call into `_em`
        with every T-0004b extension switched off, so its behaviour is identical
        to the pre-T-0004b implementation and must stay that way.

        Args:
            sequences: One (T_m, D) observation array per merchant. T_m may
                differ between merchants but must be >= 2. Must be non-empty.
            max_iter: Hard cap on EM iterations.
            tol: Stop when the relative log-likelihood improvement
                ``(ll - ll_prev) / |ll_prev|`` falls below this.
            rng: Generator for k-means seeding and degenerate-state
                reinitialisation. Defaults to `numpy.random.default_rng()`.

        Returns:
            Total log-likelihood (nats, summed over sequences) after each
            E-step, in iteration order. Must be non-decreasing; a decrease is
            always an M-step bug (07-math.md 1).

        Raises:
            ValueError: If ``sequences`` is empty, or any sequence has the
                wrong feature count or fewer than two observations.
        """
        return self._em(sequences, labels=None, max_iter=max_iter, tol=tol, rng=rng)

    def fit_partial(
        self,
        sequences: list[np.ndarray],
        labels: list[np.ndarray],
        *,
        state_weights: np.ndarray | None = None,
        dirichlet_alpha: float = 0.0,
        sticky_kappa: float = 0.0,
        var_floor_scale: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        rng: np.random.Generator | None = None,
    ) -> list[float]:
        """Partially-supervised, label-weighted Baum-Welch (T-0004b, item 2).

        Two departures from `fit`, both taken from Sidrow, Heckman, McRae,
        Volpov, Trites, Fortune and Auger-Methe, *Incorporating sparse labels
        into hidden Markov models using weighted likelihoods*, PLOS ONE
        20(6):e0325321 (2025), arXiv:2409.18091, and motivated by Elworthy
        (ANLP 1994), Merialdo (Computational Linguistics 1994) and Li, Zhou and
        Wang (arXiv:2405.16859):

        1. **Clamping.** On a timestep whose label is known, every state the
           label forbids gets `LABEL_CLAMP_LOG` added to its log-emission. The
           clamp is applied *inside* forward-backward rather than to gamma after
           the fact, so the transition posterior xi is clamped consistently too.
        2. **Weighting.** Labelled timesteps enter the M-step with a per-state
           weight, so a rare labelled state is not drowned by unlabelled
           majority mass. Default weights are inverse-frequency over the
           labelled timesteps, normalised to leave total labelled weight
           unchanged.

        LEAKAGE. This method has no idea what a split is and cannot defend
        itself. ``labels`` must be `UNLABELLED` (-1) at every timestep belonging
        to a merchant outside the TRAINING split. `eval/splits.py` is what
        decides that; `tests/test_hmm_recovery_fullscale.py` asserts it,
        including a test that corrupting held-out labels leaves the fitted
        parameters bit-identical.

        Because the M-step is weighted, the reported log-likelihood is the
        *clamped, weighted* likelihood and is NOT guaranteed monotone when the
        weights are not all 1. That is a property of weighted likelihood, not a
        bug; `fit` keeps the monotonicity guarantee and the monotonicity test.

        Args:
            sequences: One (T_m, D) observation array per merchant.
            labels: One (T_m,) integer array per merchant, values in [0, K) for
                a known state or `UNLABELLED` for a free one. Same length and
                order as ``sequences``.
            state_weights: Optional (K,) M-step weights for labelled timesteps.
                None computes inverse-frequency weights over the labels given.
            dirichlet_alpha: Symmetric Dirichlet pseudo-count added to every
                expected transition count in the M-step. 0 disables. Units:
                expected transition counts.
            sticky_kappa: Extra pseudo-count added to the transition diagonal
                only (Fox, Sudderth, Jordan and Willsky, ICML 2008). 0 disables.
                Units: expected transition counts.
            var_floor_scale: Variance floor as a fraction of each feature's
                pooled variance. 0 leaves only the absolute `VAR_FLOOR`.
            max_iter: Hard cap on EM iterations.
            tol: Relative-improvement stopping tolerance.
            rng: Generator for k-means seeding and state revival.

        Returns:
            Clamped, weighted total log-likelihood after each E-step, in nats.

        Raises:
            ValueError: If ``labels`` does not match ``sequences`` in length or
                shape, or holds a state index outside [0, K).
        """
        if len(labels) != len(sequences):
            raise ValueError(f"got {len(labels)} label arrays for {len(sequences)} sequences")
        for i, (X, y) in enumerate(zip(sequences, labels, strict=True)):
            y = np.asarray(y)
            if y.ndim != 1 or y.shape[0] != X.shape[0]:
                raise ValueError(f"labels[{i}] has shape {y.shape}, expected ({X.shape[0]},)")
            known = y != UNLABELLED
            if known.any() and (y[known].min() < 0 or y[known].max() >= self.n_states):
                raise ValueError(f"labels[{i}] holds a state outside [0, {self.n_states})")
        return self._em(
            sequences,
            labels=[np.asarray(y, dtype=np.int64) for y in labels],
            state_weights=state_weights,
            dirichlet_alpha=dirichlet_alpha,
            sticky_kappa=sticky_kappa,
            var_floor_scale=var_floor_scale,
            max_iter=max_iter,
            tol=tol,
            rng=rng,
        )

    def _default_state_weights(self, labels: list[np.ndarray]) -> np.ndarray:
        """Inverse-frequency M-step weights over the labelled timesteps.

        Normalised so that the total weight carried by labelled timesteps is
        unchanged: the weighting redistributes influence between states, it does
        not silently inflate the labelled set against the unlabelled one.

        Args:
            labels: Label arrays, `UNLABELLED` where the state is unknown.

        Returns:
            Weights of shape (K,), dimensionless. All ones when nothing is
            labelled.
        """
        counts = np.zeros(self.n_states)
        for y in labels:
            known = y[y != UNLABELLED]
            if known.size:
                counts += np.bincount(known, minlength=self.n_states)
        total = counts.sum()
        if total <= 0:
            return np.ones(self.n_states)
        inverse = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
        return inverse * (total / max(float((counts * inverse).sum()), 1e-12))

    def _em(
        self,
        sequences: list[np.ndarray],
        labels: list[np.ndarray] | None = None,
        *,
        state_weights: np.ndarray | None = None,
        dirichlet_alpha: float = 0.0,
        sticky_kappa: float = 0.0,
        var_floor_scale: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        rng: np.random.Generator | None = None,
    ) -> list[float]:
        """The shared EM loop behind `fit` and `fit_partial`.

        With ``labels=None`` and every extension argument at its default this is
        textbook Baum-Welch and reproduces the pre-T-0004b `fit` exactly.

        Returns:
            Total log-likelihood after each E-step, in nats.
        """
        if not sequences:
            raise ValueError("fit requires at least one sequence; got none")
        for i, X in enumerate(sequences):
            if X.ndim != 2 or X.shape[1] != self.n_features:
                raise ValueError(
                    f"sequence {i} has shape {X.shape}, expected (T, {self.n_features})"
                )
            if X.shape[0] < 2:
                raise ValueError(f"sequence {i} has {X.shape[0]} observations; need >= 2")

        rng = rng if rng is not None else np.random.default_rng()
        X_pooled = np.concatenate(sequences, axis=0)
        global_mean = X_pooled.mean(axis=0)
        global_var = X_pooled.var(axis=0) + VAR_FLOOR
        self._init_from_data(X_pooled, rng)

        var_floor = np.maximum(VAR_FLOOR, var_floor_scale * global_var)
        transition_prior = dirichlet_alpha + sticky_kappa * np.eye(self.n_states)

        # Per-sequence clamp offsets and M-step weights, computed once.
        clamps: list[np.ndarray | None] = [None] * len(sequences)
        weights: list[np.ndarray | None] = [None] * len(sequences)
        if labels is not None:
            if state_weights is None:
                state_weights = self._default_state_weights(labels)
            state_weights = np.asarray(state_weights, dtype=float)
            for i, y in enumerate(labels):
                known = y != UNLABELLED
                if not known.any():
                    continue
                clamp = np.zeros((y.shape[0], self.n_states))
                clamp[known] = LABEL_CLAMP_LOG
                clamp[np.flatnonzero(known), y[known]] = 0.0
                w = np.ones(y.shape[0])
                w[known] = state_weights[y[known]]
                clamps[i] = clamp
                weights[i] = w

        history: list[float] = []
        for _ in range(max_iter):
            # --- E-step: accumulate sufficient statistics over all sequences.
            total_ll = 0.0
            xi_sum = np.zeros((self.n_states, self.n_states))
            gamma_first = np.zeros(self.n_states)
            occupancy = np.zeros(self.n_states)
            weighted_x = np.zeros((self.n_states, self.n_features))
            weighted_xx = np.zeros((self.n_states, self.n_features))

            for X, clamp, w in zip(sequences, clamps, weights, strict=True):
                log_b = self.log_emission(X)
                if clamp is not None:
                    log_b = log_b + clamp
                log_alpha, ll = self._forward_logb(log_b)
                log_beta = self._backward_logb(log_b)
                total_ll += ll

                gamma = np.exp(log_alpha + log_beta - ll)  # (T, K)
                # log_xi[t, i, j] = alpha_t(i) + logA_ij + logb_{t+1}(j) + beta_{t+1}(j) - ll
                log_xi = (
                    log_alpha[:-1, :, None]
                    + self.log_A[None, :, :]
                    + (log_b[1:] + log_beta[1:])[:, None, :]
                    - ll
                )
                xi = np.exp(log_xi)
                if w is not None:
                    gamma = gamma * w[:, None]
                    xi = xi * w[:-1, None, None]
                xi_sum += xi.sum(axis=0)
                gamma_first += gamma[0]
                occupancy += gamma.sum(axis=0)
                weighted_x += gamma.T @ X
                weighted_xx += gamma.T @ (X**2)

            history.append(total_ll)
            if len(history) >= 2:
                prev = history[-2]
                if (total_ll - prev) / max(abs(prev), 1e-12) < tol:
                    break

            # --- M-step.
            self.log_pi = _log_normalise(np.log(gamma_first + TRANSITION_FLOOR), axis=0)
            self.log_A = _log_normalise(
                np.log(xi_sum + transition_prior + TRANSITION_FLOOR), axis=1
            )
            # Floor transitions and renormalise: no transition may become
            # permanently impossible.
            self.log_A = _log_normalise(np.logaddexp(self.log_A, np.log(TRANSITION_FLOOR)), axis=1)

            denom = occupancy[:, None]
            safe = np.maximum(denom, 1e-12)
            self.mu = weighted_x / safe
            self.var = weighted_xx / safe - self.mu**2 + VAR_FLOOR
            self.var = np.maximum(self.var, var_floor)

            # Degeneracy handling: revive rather than lose a starved state.
            dead = occupancy < MIN_OCCUPANCY
            if dead.any():
                n_dead = int(dead.sum())
                self.mu[dead] = global_mean + rng.normal(scale=0.1, size=(n_dead, self.n_features))
                self.var[dead] = global_var

        return history

    def score(self, X: np.ndarray) -> float:
        """Sequence log-likelihood.

        Args:
            X: Observations, shape (T, D).

        Returns:
            ``log P(x_1..x_T)`` in nats.
        """
        return self.forward(X)[1]
