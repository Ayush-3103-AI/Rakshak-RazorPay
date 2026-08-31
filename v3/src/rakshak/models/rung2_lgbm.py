"""Rung 2 - LightGBM on windowed aggregates. **The incumbent and the bar.**

Charter §: the bar is LightGBM, not the rule engine. v1's hand-written HMM lost to plain
LightGBM on windowed aggregates by 0.3176 PR-AUC, and v2 re-races against that same model
on a corrected generator rather than against something easier.

Three decisions are load-bearing and each one is a decision *not* to do something:

**No post-hoc calibrator.** LightGBM's binary objective is logloss, which is a proper
scoring rule, so the raw sigmoid output is already a probability estimate. Under the
delayed-label regime this dataset actually has, a held-out calibration slice would contain
one or two positive merchants, and an isotonic fit on two positives is not a calibration -
it is a step function with a random step. ECE is reported on the raw probability, which is
the honest number.

**No ``scale_pos_weight`` and no ``is_unbalance``.** Both improve ranking metrics slightly
and destroy calibration completely, and the decision layer consumes the score *as a
probability* to compute expected cost. A rung that ranks well and is uncalibrated makes
every rupee figure downstream arithmetic on a meaningless quantity.

**No early stopping on the validation split.** Early stopping is model selection; doing it
against the split the result is reported on makes the reported number optimistic by an
unknown amount. The tree count is fixed, declared here, and regularised hard instead.

Prime Directive 3: this module never sees a label table, a typology or an onset. It is
handed ``x`` and ``y`` as arrays by the CLI, which lives on the eval side of the
quarantine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import numpy as np

from rakshak.features import registry
from rakshak.models.dataset import RESIDUAL_PREFIX

__all__ = [
    "DEFAULT_PARAMS",
    "HParams",
    "TrainedRung",
    "explain_column",
    "train",
]


@dataclass(frozen=True, slots=True)
class HParams:
    """Declared before any result was seen, and identical for Rungs 2, 3 and 4.

    That identity is what makes the Rung-3 delta attributable to the residual columns and
    to nothing else (FR-031). If a hyperparameter ever needs to move, it moves for every
    rung at once or the comparison stops meaning anything.
    """

    num_leaves: int = 15
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 10.0
    min_split_gain: float = 0.0
    num_threads: int = 4
    seed: int = 42

    def booster_params(self) -> dict[str, Any]:
        params = asdict(self)
        trees = params.pop("n_estimators")
        params.pop("seed")
        return {
            **params,
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "seed": self.seed,
            "bagging_seed": self.seed + 1,
            "feature_fraction_seed": self.seed + 2,
            "data_random_seed": self.seed + 3,
            "deterministic": True,
            "force_row_wise": True,
            "_n_estimators": trees,
        }

    def with_seed(self, seed: int) -> HParams:
        return replace(self, seed=seed)


#: NFR-06 gives 20 minutes on 4 cores and NFR-05 gives 20 MB; 300 trees of 15 leaves over
#: 28-49 columns lands three orders of magnitude inside both, which is deliberate. The
#: binding constraint on this rung is labelled positives, not compute, and spending the
#: budget on capacity the labels cannot support would only fit noise harder.
DEFAULT_PARAMS: Final = HParams()


def explain_column(column: str, value: float) -> str:
    """One merchant-readable sentence for a feature or for its cohort residual (FR-033).

    The residual's sentence carries the cohort clause, because "you went up and your peers
    did not" is the half of the explanation that survives a merchant dispute
    (10-eval-harness-spec.md §5).
    """
    if column.startswith(RESIDUAL_PREFIX):
        base = column[len(RESIDUAL_PREFIX) :]
        return (
            f"{registry.get(base).explain(value)}"
            " - and comparable merchants did not move with it"
        )
    return registry.get(column).explain(value)


@dataclass(frozen=True, slots=True)
class TrainedRung:
    """A fitted booster plus the column order it was fitted on.

    ``columns`` travels with the model rather than being looked up at score time. A model
    trained on one column order and scored on another fails silently (09-interfaces.md §9),
    and silent is the worst failure mode available in this repo.
    """

    rung: int
    booster: lgb.Booster
    columns: tuple[str, ...]
    params: HParams
    n_train_rows: int
    n_train_positive_rows: int
    n_train_positive_merchants: int
    train_seconds: float

    def _aligned(self, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
        if columns == self.columns:
            return x
        index = {name: i for i, name in enumerate(columns)}
        missing = [c for c in self.columns if c not in index]
        if missing:
            raise KeyError(f"scoring frame is missing trained columns {missing}")
        return x[:, [index[c] for c in self.columns]]

    def predict(self, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
        scores = self.booster.predict(self._aligned(x, columns))
        clipped: np.ndarray = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
        return clipped

    def contributions(self, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
        """``pred_contrib`` without the bias term: ``(rows x len(columns))``."""
        raw = np.asarray(
            self.booster.predict(self._aligned(x, columns), pred_contrib=True),
            dtype=np.float64,
        )
        return raw[:, :-1]

    def reason_codes(
        self, x: np.ndarray, columns: tuple[str, ...], rows: np.ndarray, top: int = 3
    ) -> list[list[str]]:
        """Top-``top`` features by absolute contribution, per row (FR-033).

        Native ``pred_contrib``, no SHAP dependency. Computed only for the rows asked for,
        which in the cascade is stage 2 - the non-PASS decisions - and nothing else.
        """
        if rows.size == 0:
            return []
        aligned = self._aligned(x, columns)[rows]
        contrib = self.contributions(aligned, self.columns)
        order = np.argsort(-np.abs(contrib), axis=1)[:, :top]
        return [
            [explain_column(self.columns[j], float(aligned[i, j])) for j in order[i]]
            for i in range(len(rows))
        ]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        return path

    def size_mb(self, path: Path) -> float:
        """NFR-05: the artifact on disk, not an estimate of it."""
        return path.stat().st_size / 1_048_576


def train(
    x: np.ndarray,
    y: np.ndarray,
    columns: tuple[str, ...],
    *,
    rung: int = 2,
    params: HParams = DEFAULT_PARAMS,
    merchant_id: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> TrainedRung:
    """Fit the booster on training rows only. No validation data enters this function.

    ``merchant_id`` is optional and is used for reporting the number of distinct positive
    merchants, which under a delayed-label regime is the number that actually bounds what
    the rung can learn - the positive *row* count is that number times a window length and
    flatters the setup considerably.

    ``weight`` is ``None`` for Rungs 2 and 3 and is Rung 4's whole contribution (FR-032):
    per-instance costs inside the objective rather than Bayes-minimum-risk after it.
    """
    import time as _time

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x has {x.shape[0]} rows and y has {y.shape[0]}")
    if x.shape[1] != len(columns):
        raise ValueError(f"x has {x.shape[1]} columns and {len(columns)} names were given")
    positives = y == 1
    if not positives.any():
        raise ValueError(
            "no positive training rows. Under the label-availability rule (FR-020) a "
            "merchant is trainable only once its dispute has resolved; if this fires, the "
            "training as_of is earlier than the first available label and the rung has "
            "nothing to fit."
        )

    booster_params = params.booster_params()
    n_estimators = int(booster_params.pop("_n_estimators"))
    dataset = lgb.Dataset(
        x, label=y.astype(np.float64), weight=weight, feature_name=list(columns)
    )
    started = _time.perf_counter()
    booster = lgb.train(booster_params, dataset, num_boost_round=n_estimators)
    elapsed = _time.perf_counter() - started

    n_merchants = (
        int(np.unique(merchant_id[positives]).size) if merchant_id is not None else 0
    )
    return TrainedRung(
        rung=rung,
        booster=booster,
        columns=columns,
        params=params,
        n_train_rows=int(x.shape[0]),
        n_train_positive_rows=int(positives.sum()),
        n_train_positive_merchants=n_merchants,
        train_seconds=round(elapsed, 2),
    )
