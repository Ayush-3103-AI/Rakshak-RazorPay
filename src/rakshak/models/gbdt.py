"""LightGBM on windowed aggregates — the discriminative incumbent, no HMM.

06-requirements.md §3 lists this baseline as the one that isolates *"whether
latent-state modelling earns its place (A-005)"*. That question is only worth
asking if the baseline is strong, so this module deliberately gives LightGBM
every advantage the HMM gets and none it does not:

* **The same emissions.** It is trained on `features.build_emissions` output —
  the identical standardised, segment-shrunk window vectors the HMM consumes.
  Feeding it raw un-standardised aggregates would have handed it a grocer's
  velocity and a jeweller's velocity in the same column and produced a strawman.
* **Per-window supervision.** One row per (merchant, window) with the generator's
  window-level bad-state label, so it sees ~25x more labelled examples than a
  per-merchant framing would give it.
* **Class weighting** via `scale_pos_weight`, computed from the training rows.
* **Early stopping** on the validation split, per the frozen eval's "all
  hyperparameters and thresholds chosen on the validation window".

**Leakage.** Training reads `load_split("train")` and nothing else. The segment
map is fitted on training merchants and *passed in* to the held-out build, so no
held-out merchant contributes to any standardisation constant. Within-merchant
standardisation uses each merchant's own first 8 windows (days 0-55), which end
before the earliest possible typology onset and before every decision window.
The test split is never opened here; T-0011 owns that.

**One caveat that must not be lost.** Early stopping selects the iteration count
on the same `validate` split the harness currently reports on, so the `gbdt` row
in `results/summary.md` is mildly optimistic for as long as the harness reports
validate. It is clean at T-0011, where the reported window is `test` and
validate is only the early-stopping set. This is stated in the summary too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from rakshak.config import (
    BURN_IN_WINDOWS,
    GENERATOR_START_DATE,
    SEED,
    WINDOW_DAYS,
)
from rakshak.eval.splits import (
    BAD_STATES,
    Split,
    active_state_paths_path,
    load_split,
)
from rakshak.features import SegmentMap, build_emissions, window_state_labels

LOGGER = logging.getLogger(__name__)

FLAG_THRESHOLD: float = 0.5
"""Per-window probability at which a flag is raised, for detection lag only.
Not a decision threshold — the harness's budget policy ranks on `score`."""

PARAMS: dict[str, object] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}
"""Ordinary LightGBM defaults for a small tabular problem. `deterministic`,
`force_row_wise` and `num_threads=1` are what make NFR-003 hold — LightGBM's
histogram construction is otherwise thread-order dependent."""

N_ROUNDS: int = 400
EARLY_STOPPING_ROUNDS: int = 40


@dataclass(frozen=True)
class WindowMatrix:
    """A split flattened to one row per (merchant, window).

    Attributes:
        X: Standardised emissions, shape (M*W, D), dimensionless z-scores.
        y: Window-level bad-state labels, shape (M*W,), dimensionless {0, 1}.
        merchant_row: Index into `merchant_ids` per row, shape (M*W,).
        window_start_day: Day the row's window starts on, shape (M*W,). Units: days.
        merchant_ids: Merchant ids in row-block order, shape (M,).
        feature_names: Column order of `X`.
        segment_map: The segmentation used, reusable on a held-out population.
    """

    X: np.ndarray
    y: np.ndarray
    merchant_row: np.ndarray
    window_start_day: np.ndarray
    merchant_ids: np.ndarray
    feature_names: tuple[str, ...]
    segment_map: SegmentMap


def build_window_matrix(
    split: Split,
    segment_map: SegmentMap | None = None,
    state_paths: pd.DataFrame | None = None,
    drop_features: tuple[str, ...] = (),
    standardise: bool = True,
) -> WindowMatrix:
    """Flatten a split into the (merchant, window) design matrix LightGBM trains on.

    Args:
        split: The split to flatten. Uses `split.transactions`, i.e. each merchant's
            own history up to the window end.
        segment_map: Segmentation fitted on the training population. None fits one
            here, which is only correct for the training split itself.
        state_paths: Override for the generator's state-path frame (tests use this),
            mirroring `eval.splits.load_split`.
        drop_features: FR-018 ablation only - emission columns to omit. Defaults to `()`,
            the shipping vector. Must match between fit and score; `score_gbdt` raises on
            a feature-name mismatch.
        standardise: FR-018 ablation only - False passes the raw per-window features
            through with no within-merchant standardisation. Defaults to True (FR-007).

    Returns:
        A populated `WindowMatrix`.
    """
    emissions = build_emissions(
        split.transactions,
        segment_map=segment_map,
        drop_features=drop_features,
        standardise=standardise,
    )
    n_merchants, n_windows, n_features = emissions.X.shape

    if state_paths is None:
        # T-0022b: the active dataset, not the config constant, so a shock-dataset
        # run fits on the shock dataset instead of silently on data/synthetic/.
        state_paths = pd.read_parquet(active_state_paths_path())
    labels = window_state_labels(
        state_paths, emissions.merchant_ids, n_windows, window_days=WINDOW_DAYS
    )
    y = np.isin(labels, list(BAD_STATES)).astype(np.int64).ravel()

    # build_window_features indexes windows from the frame's own first day, matching
    # its epoch convention; convert back to absolute days since GENERATOR_START_DATE.
    epoch_offset = int(
        (
            split.transactions["timestamp"].min().normalize()
            - pd.Timestamp(GENERATOR_START_DATE)
        ).days
    )
    window_start_day = np.tile(
        emissions.window_index * WINDOW_DAYS + epoch_offset, n_merchants
    ).astype(np.int64)

    return WindowMatrix(
        X=emissions.X.reshape(n_merchants * n_windows, n_features),
        y=y,
        merchant_row=np.repeat(np.arange(n_merchants), n_windows),
        window_start_day=window_start_day,
        merchant_ids=emissions.merchant_ids,
        feature_names=emissions.feature_names,
        segment_map=emissions.segment_map,
    )


def decision_mask(matrix: WindowMatrix, split: Split) -> np.ndarray:
    """Rows whose window lies inside the split's decision window.

    Windows that straddle `start_day` are excluded: a window half outside the
    decision window is half a different split's evidence.
    """
    return (matrix.window_start_day >= split.start_day) & (
        matrix.window_start_day + WINDOW_DAYS <= split.end_day
    )


def train_mask(matrix: WindowMatrix) -> np.ndarray:
    """Training rows: everything after the burn-in the standardiser consumed.

    The burn-in windows are z-scored against themselves and are HEALTHY for every
    merchant by construction (config.BURN_IN_WINDOWS), so they carry no signal and
    would only inflate the negative class.
    """
    return matrix.window_start_day >= BURN_IN_WINDOWS * WINDOW_DAYS


def fit(
    seed: int = SEED,
    drop_features: tuple[str, ...] = (),
    standardise: bool = True,
) -> tuple[lgb.Booster, SegmentMap, tuple[str, ...]]:
    """Fit LightGBM on the train split, early-stopping on the validate split.

    The test split is never loaded. See the module docstring on the one place this
    is optimistic (reporting on validate while early-stopping on it).

    Args:
        seed: Determinism seed; feeds LightGBM's bagging RNG.
        drop_features: FR-018 ablation only - emission columns to omit from both the
            training and the early-stopping matrix. Defaults to `()`, the shipping vector.
        standardise: FR-018 ablation only - False refits on raw per-window features.
            Defaults to True (FR-007).

    Returns:
        `(booster, segment_map, feature_names)`. The segment map is fitted on the
        training merchants and must be passed to every held-out build.
    """
    variant = {"drop_features": drop_features, "standardise": standardise}
    train_split = load_split("train")
    train_matrix = build_window_matrix(train_split, **variant)
    valid_split = load_split("validate")
    valid_matrix = build_window_matrix(
        valid_split, segment_map=train_matrix.segment_map, **variant
    )

    fit_rows = train_mask(train_matrix)
    stop_rows = decision_mask(valid_matrix, valid_split)
    n_pos = int(train_matrix.y[fit_rows].sum())
    n_neg = int(fit_rows.sum() - n_pos)

    params = dict(PARAMS)
    params["seed"] = seed
    params["scale_pos_weight"] = float(n_neg) / float(max(n_pos, 1))
    LOGGER.info(
        "gbdt: %d train rows (%d positive), %d early-stopping rows, scale_pos_weight=%.2f",
        int(fit_rows.sum()),
        n_pos,
        int(stop_rows.sum()),
        params["scale_pos_weight"],
    )

    names = list(train_matrix.feature_names)
    booster = lgb.train(
        params,
        lgb.Dataset(train_matrix.X[fit_rows], label=train_matrix.y[fit_rows], feature_name=names),
        num_boost_round=N_ROUNDS,
        valid_sets=[
            lgb.Dataset(
                valid_matrix.X[stop_rows], label=valid_matrix.y[stop_rows], feature_name=names
            )
        ],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return booster, train_matrix.segment_map, train_matrix.feature_names


@lru_cache(maxsize=8)
def _fitted(
    seed: int, drop_features: tuple[str, ...] = (), standardise: bool = True
) -> tuple[lgb.Booster, SegmentMap, tuple[str, ...]]:
    """Memoised `fit` - the harness may score the same seed more than once.

    The ablation variant is part of the cache key (FR-018): a fit on a reduced or
    unstandardised emission vector must never be handed back to the shipping path,
    which is what a seed-only key would do.
    """
    return fit(seed, drop_features=drop_features, standardise=standardise)


def score_gbdt(
    split: Split,
    rng: np.random.Generator,
    drop_features: tuple[str, ...] = (),
    standardise: bool = True,
) -> pd.DataFrame:
    """Score merchants with LightGBM over windowed aggregates (06-requirements.md §3).

    Args:
        split: The split to score. Must not be the training split.
        rng: Used only to derive the fit seed, so that the harness's per-model RNG
            still controls determinism.
        drop_features: FR-018 ablation only - applied identically to the fit and to this
            score, so the feature-name guard below stays meaningful. Defaults to `()`.
        standardise: FR-018 ablation only - applied identically to fit and score.
            Defaults to True.

    Returns:
        Frame indexed by merchant_id with `score`, the maximum per-window bad-state
        probability over the decision window (dimensionless, [0, 1]), and `flag_day`,
        the **last** day of the first window scoring above `FLAG_THRESHOLD` — the
        earliest day that window's evidence was complete. See
        `hmm_score.first_flag_day` and `results/lag_probe.md`: window-start
        attribution was what produced the -1.0 median lag, and `models/rules.py`
        has always reported the last day of its own evidence.
    """
    seed = int(rng.integers(0, 2**31 - 1))
    booster, segment_map, feature_names = _fitted(seed, drop_features, standardise)

    matrix = build_window_matrix(
        split,
        segment_map=segment_map,
        drop_features=drop_features,
        standardise=standardise,
    )
    if matrix.feature_names != feature_names:
        raise ValueError(
            f"feature mismatch: trained on {feature_names}, scoring {matrix.feature_names}"
        )
    rows = decision_mask(matrix, split)
    if not rows.any():
        raise ValueError(f"split {split.name!r} has no whole window inside its decision window")

    proba = booster.predict(matrix.X[rows], num_iteration=booster.best_iteration)
    frame = pd.DataFrame(
        {
            "merchant_row": matrix.merchant_row[rows],
            "day": matrix.window_start_day[rows],
            "proba": np.asarray(proba, dtype=float),
        }
    )
    score = frame.groupby("merchant_row")["proba"].max()
    # + WINDOW_DAYS - 1: attribute the flag to the last day of the window that
    # raised it, matching `models/rules.py`. See the docstring above.
    flagged = (
        frame[frame["proba"] >= FLAG_THRESHOLD].groupby("merchant_row")["day"].min()
        + WINDOW_DAYS
        - 1
    )

    index = pd.Index(matrix.merchant_ids, name="merchant_id")
    positions = pd.RangeIndex(len(index))
    return pd.DataFrame(
        {
            "score": score.reindex(positions).to_numpy(dtype=float),
            "flag_day": flagged.reindex(positions).to_numpy(dtype=float),
        },
        index=index,
    ).reindex(pd.Index(split.merchant_ids, name="merchant_id"))
