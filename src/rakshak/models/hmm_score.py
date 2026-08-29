"""The proposal, scored — HMM filtered posterior to risk score and `flag_day` (T-0006b).

`models/hmm.py` is the mathematics; this module is the only place that turns it
into the two numbers the harness contract asks for. Until this landed,
`MODEL_REGISTRY` held `random`, `rules` and `gbdt` and the model the project's
hypothesis is about had never produced a row.

**The one constraint this module exists to get right.**

`flag_day` comes from the **forward-only filtered posterior**,
``P(z_t = bad | x_1..x_t)``, and never from the smoothed forward-backward
posterior. A smoothed posterior at window *t* conditions on windows *after* *t*,
so using it to decide when the model "first flagged" is deciding with
information from the future, and it manufactures exactly the negative detection
lag that is already under investigation elsewhere. The difference is one call
site: `HMM.forward` here, not `HMM.posterior`.

`filtered_bad_probability` is the **only** scoring path in this module, so both
reported quantities carry the same temporal guarantee:

* `score`   — max filtered bad-state probability over the decision windows.
              Feeds PR-AUC, precision@K and Brier. Ranking at end-of-window
              would have permitted the smoothed posterior; it is not used, so
              the row needs no footnote about which metric saw what.
* `flag_day`— start day of the first decision window whose filtered bad-state
              probability reaches `FLAG_THRESHOLD`. Feeds median detection lag.

`tests/test_hmm_score.py::test_filtered_posterior_ignores_the_future` is the
proof, and it carries a negative control showing the smoothed posterior fails
the identical assertion. The comment is not the proof; the test is.

**The fit.** T-0004b's shipping configuration: `HMM.fit_partial` — label
clamping plus inverse-frequency M-step weighting (items 1 and 2) — pooled over
the whole training population rather than per segment, `N_HIDDEN_STATES` = 4, no
Dirichlet/sticky transition prior, no variance floor beyond the absolute one, no
deterministic DORMANT rule. Items 3 and 4 were measured and did not land; see
`logbook-entries/T-0004b.md`. The RAMP-recall regression (0.328 -> 0.234) is a
reported finding of that ticket and is not touched here.

**Where this differs from T-0004b, deliberately and in the conservative
direction.** T-0004b fitted transductively — over every merchant's emissions,
with labels restricted to the training group — and flagged that residual
exposure in its own logbook entry. A harness scorer must not do that: it fits on
`load_split("train")` and nothing else, and the held-out split is only ever
passed through `forward`. One consequence has to be stated rather than
discovered: inside the training split nearly every window satisfies both label
conditions, so the "partially" in partially-supervised is doing much less work
here than it did at T-0004b (~96% of training windows carry a label, against 38%
of all windows there). The fit is closer to weighted supervised MLE with a
Markov prior. That is what "fit on train only" implies for this model, it is the
same supervision LightGBM gets from `gbdt.build_window_matrix`, and it is
reported, not hidden.

**Fairness.** The design matrix, the segmentation and the decision-window mask
all come from `models/gbdt.py`, so the HMM and the incumbent it is being
compared against see byte-identical inputs. A separate feature path here would
make any difference in the rows unattributable.

**Savings.** The `savings` column for this row is unreadable, exactly as it is
for every other row, until T-0007a fixes the cost matrix. See the FAIL verdict
in the cost-matrix sanity check of `results/summary.md`.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import softmax

from rakshak.config import (
    BAD_STATES,
    N_HIDDEN_STATES,
    SEED,
    SPLIT_DAY_BOUNDS,
    STATE_PATHS_PARQUET,
    WINDOW_DAYS,
)
from rakshak.eval.splits import Split, load_split
from rakshak.features import SegmentMap, window_state_labels
from rakshak.models.gbdt import WindowMatrix, build_window_matrix, decision_mask
from rakshak.models.hmm import HMM, UNLABELLED

LOGGER = logging.getLogger(__name__)

STATE_ORDER: tuple[str, ...] = ("HEALTHY", "RAMP", "FRAUD", "DORMANT")
"""Hidden-state index -> generator state name. Label clamping pins the identity:
state k of the fitted model is the state whose label code is k, so no post-hoc
Hungarian alignment is needed to know which columns are bad. Must stay identical
to `tests/test_hmm_recovery_fullscale.py::STATE_ORDER`."""

BAD_COLUMNS: tuple[int, ...] = tuple(
    k for k, name in enumerate(STATE_ORDER) if name in BAD_STATES
)
"""Columns of the posterior summed into the risk score — RAMP, FRAUD, DORMANT,
matching `config.BAD_STATES` and the "Bad states" row of `results/summary.md`."""

FLAG_THRESHOLD: float = 0.5
"""Filtered bad-state probability at which a flag is raised, for detection lag
only. Not a decision threshold — the harness's budget policy ranks on `score`.
Same value as `gbdt.FLAG_THRESHOLD`, so the two lag numbers are comparable."""

if len(STATE_ORDER) != N_HIDDEN_STATES:  # pragma: no cover - configuration guard
    raise ValueError(
        f"STATE_ORDER has {len(STATE_ORDER)} states but N_HIDDEN_STATES is "
        f"{N_HIDDEN_STATES}; the label codes and the model's states would not line up"
    )


# ---------------------------------------------------------------------------
# The temporally honest scoring path
# ---------------------------------------------------------------------------


def filtered_bad_probability(model: HMM, X: np.ndarray) -> np.ndarray:
    """``P(z_t in BAD_STATES | x_1..x_t)`` — forward only, no future information.

    This is the whole point of the module. `HMM.forward` returns
    ``log_alpha[t, k] = log P(x_1..x_t, z_t = k)``; normalising each row gives the
    filtered belief at *t* conditioned only on windows up to and including *t*.
    `HMM.posterior` would return the smoothed belief and is deliberately not
    called here.

    Args:
        model: A fitted HMM whose state indices follow `STATE_ORDER`.
        X: One merchant's emissions, shape (T, D), dimensionless z-scores.

    Returns:
        Array of shape (T,) in [0, 1]. Entry *t* depends on ``X[:t + 1]`` only,
        bitwise — see `tests/test_hmm_score.py`.
    """
    log_alpha, _ = model.forward(X)
    # Clipped only because summing K-1 float64 columns of a normalised distribution
    # can land a hair above 1.0; the harness's Brier term needs a genuine probability.
    return np.clip(softmax(log_alpha, axis=1)[:, BAD_COLUMNS].sum(axis=1), 0.0, 1.0)


def first_flag_day(
    probability: np.ndarray, eligible: np.ndarray, window_start_day: np.ndarray
) -> float:
    """Start day of the first eligible window at or above `FLAG_THRESHOLD`.

    Args:
        probability: Filtered bad-state probability per window, shape (T,).
        eligible: True where the window lies inside the split's decision window,
            shape (T,).
        window_start_day: Absolute day each window starts on, shape (T,). Units: days.

    Returns:
        The flag day in days since `GENERATOR_START_DATE`, or NaN if the merchant
        was never flagged. Depends only on windows at or before the returned day.
    """
    fired = eligible & (probability >= FLAG_THRESHOLD)
    if not fired.any():
        return float("nan")
    return float(window_start_day[int(np.argmax(fired))])


def _panel(matrix: WindowMatrix) -> np.ndarray:
    """Reshape a flattened `WindowMatrix` back to (M, W, D) sequences."""
    n_merchants = len(matrix.merchant_ids)
    n_rows, n_features = matrix.X.shape
    return matrix.X.reshape(n_merchants, n_rows // n_merchants, n_features)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def training_label_grid(matrix: WindowMatrix, state_paths: pd.DataFrame) -> np.ndarray:
    """The (M, W) label grid handed to `HMM.fit_partial`, `UNLABELLED` where forbidden.

    A window may carry its ground-truth state only if both split dimensions allow
    it (06-requirements.md §3): its merchant is in the training group — guaranteed
    here because `matrix` is built from `load_split("train")` — and the window ends
    before the training temporal window does. Everything else is `UNLABELLED`.

    Args:
        matrix: Design matrix built from the training split.
        state_paths: The generator's state-path frame.

    Returns:
        Int64 array of shape (M, W), values in [0, K) or `UNLABELLED`.
    """
    n_merchants = len(matrix.merchant_ids)
    n_windows = matrix.X.shape[0] // n_merchants
    names = window_state_labels(
        state_paths, matrix.merchant_ids, n_windows, window_days=WINDOW_DAYS
    )
    codes = np.vectorize(STATE_ORDER.index)(names)
    start = matrix.window_start_day.reshape(n_merchants, n_windows)
    labelled = (start + WINDOW_DAYS) <= SPLIT_DAY_BOUNDS["train"][1]
    return np.where(labelled, codes, UNLABELLED).astype(np.int64)


def fit(
    seed: int = SEED,
    drop_features: tuple[str, ...] = (),
    standardise: bool = True,
) -> tuple[HMM, SegmentMap, tuple[str, ...]]:
    """Fit the shipping T-0004b configuration on the training split and nothing else.

    Args:
        seed: Determinism seed; feeds k-means initialisation and state revival.
        drop_features: FR-018 ablation only - emission columns to omit. Defaults to `()`,
            the shipping vector. Label clamping and `STATE_ORDER` are untouched by this:
            the ablation changes the emission dimension D, never the state space K.
        standardise: FR-018 ablation only - False refits on raw per-window features.
            Defaults to True (FR-007).

    Returns:
        `(model, segment_map, feature_names)`. The segment map is fitted on the
        training merchants and must be passed to every held-out build, exactly as
        `gbdt.fit` requires.
    """
    train_split = load_split("train")
    matrix = build_window_matrix(
        train_split, drop_features=drop_features, standardise=standardise
    )
    sequences = _panel(matrix)
    n_merchants, n_windows, n_features = sequences.shape

    labels = training_label_grid(matrix, pd.read_parquet(STATE_PATHS_PARQUET))
    model = HMM(n_states=N_HIDDEN_STATES, n_features=n_features)
    history = model.fit_partial(
        [sequences[i] for i in range(n_merchants)],
        [labels[i] for i in range(n_merchants)],
        rng=np.random.default_rng(seed),
    )
    LOGGER.info(
        "hmm: %d merchants x %d windows, %d of %d windows labelled, %d EM iterations",
        n_merchants,
        n_windows,
        int((labels != UNLABELLED).sum()),
        labels.size,
        len(history),
    )
    return model, matrix.segment_map, matrix.feature_names


@lru_cache(maxsize=8)
def _fitted(
    seed: int, drop_features: tuple[str, ...] = (), standardise: bool = True
) -> tuple[HMM, SegmentMap, tuple[str, ...]]:
    """Memoised `fit` — the harness may score the same seed more than once.

    The ablation variant is part of the cache key (FR-018), so a variant fit can never
    be served to the shipping path.
    """
    return fit(seed, drop_features=drop_features, standardise=standardise)


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


def score_hmm(
    split: Split,
    rng: np.random.Generator,
    drop_features: tuple[str, ...] = (),
    standardise: bool = True,
) -> pd.DataFrame:
    """Score merchants with the per-merchant HMM belief over latent risk states.

    Args:
        split: The split to score. Must not be the training split.
        rng: Used only to derive the fit seed, so the harness's per-model RNG
            still controls determinism (NFR-003).
        drop_features: FR-018 ablation only - applied identically to the fit and to this
            score, so the feature-name guard below stays meaningful. Defaults to `()`.
        standardise: FR-018 ablation only - applied identically to fit and score.
            Defaults to True.

    Returns:
        Frame indexed by merchant_id with `score`, the maximum **filtered**
        bad-state probability over the decision windows (dimensionless, [0, 1]),
        and `flag_day`, the start day of the first decision window whose filtered
        bad-state probability reached `FLAG_THRESHOLD` (days, NaN if never).
        Neither column uses any information from after the window it reports on.
    """
    seed = int(rng.integers(0, 2**31 - 1))
    model, segment_map, feature_names = _fitted(seed, drop_features, standardise)

    matrix = build_window_matrix(
        split,
        segment_map=segment_map,
        drop_features=drop_features,
        standardise=standardise,
    )
    if matrix.feature_names != feature_names:
        raise ValueError(
            f"feature mismatch: fitted on {feature_names}, scoring {matrix.feature_names}"
        )
    sequences = _panel(matrix)
    n_merchants, n_windows, _ = sequences.shape
    eligible = decision_mask(matrix, split).reshape(n_merchants, n_windows)
    if not eligible.any():
        raise ValueError(f"split {split.name!r} has no whole window inside its decision window")
    days = matrix.window_start_day.reshape(n_merchants, n_windows)

    scores = np.empty(n_merchants)
    flags = np.empty(n_merchants)
    for i in range(n_merchants):
        probability = filtered_bad_probability(model, sequences[i])
        scores[i] = probability[eligible[i]].max()
        flags[i] = first_flag_day(probability, eligible[i], days[i])

    return pd.DataFrame(
        {"score": scores, "flag_day": flags},
        index=pd.Index(matrix.merchant_ids, name="merchant_id"),
    ).reindex(pd.Index(split.merchant_ids, name="merchant_id"))
