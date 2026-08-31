"""T-141 - Rung 2, the LightGBM incumbent and the bar.

The split-level numbers (PR-AUC, ECE, savings against all four floors, TTD, train time,
model size) live in ``docs/logbook/T-141.md`` and are produced by ``make eval RUNG=2``.
What is asserted here is everything about the rung that must be true regardless of how it
scores: that it is deterministic, that its column order travels with it, that its output
is a probability rather than a rank, and that every non-PASS decision can be explained.

The NFR-05 and NFR-06 budgets are asserted here on a small fit and again on the real one
in the logbook, because a budget only checked once is a budget nobody notices breaking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rakshak.features import registry
from rakshak.models import rung2_lgbm
from rakshak.models.dataset import base_columns, residual_columns

N = 4000


def _fit(seed: int = 42, rows: int = N) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """A synthetic panel with real signal in ``v_gmv_z`` and noise everywhere else."""
    rng = np.random.default_rng(seed)
    columns = base_columns()
    x = rng.normal(size=(rows, len(columns)))
    logit = -5.0 + 1.6 * x[:, columns.index("v_gmv_z")]
    y = (rng.random(rows) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int8)
    return x, y, columns


def test_train_refuses_when_the_delayed_labels_have_not_landed() -> None:
    """The failure mode this dataset actually has. Under FR-020 a merchant is trainable
    only once its dispute has resolved, so "no positives" is a real state and it must be a
    loud one rather than a booster that fits the intercept and reports a number."""
    x, _, columns = _fit()
    with pytest.raises(ValueError, match="no positive training rows"):
        rung2_lgbm.train(x, np.zeros(x.shape[0], dtype=np.int8), columns)


def test_the_same_seed_gives_bit_identical_predictions() -> None:
    """A rung whose number moves between runs cannot be compared to the rung below it."""
    x, y, columns = _fit()
    a = rung2_lgbm.train(x, y, columns).predict(x, columns)
    b = rung2_lgbm.train(x, y, columns).predict(x, columns)
    assert np.array_equal(a, b)


def test_the_score_is_a_probability_and_is_roughly_calibrated() -> None:
    """No post-hoc calibrator, on purpose: LightGBM's binary objective is a proper scoring
    rule, and the cost layer consumes the score AS a probability. If the mean prediction
    drifted from the base rate, every rupee figure downstream would be arithmetic on a
    meaningless quantity."""
    x, y, columns = _fit()
    p = rung2_lgbm.train(x, y, columns).predict(x, columns)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert abs(float(p.mean()) - float(y.mean())) < 0.02


def test_column_order_travels_with_the_model() -> None:
    """09-interfaces.md §9: a model trained on one order and scored on another fails
    silently, which is the worst failure mode available in this repo."""
    x, y, columns = _fit()
    model = rung2_lgbm.train(x, y, columns)
    order = np.random.default_rng(1).permutation(len(columns))
    shuffled_columns = tuple(columns[i] for i in order)
    assert np.allclose(model.predict(x[:, order], shuffled_columns), model.predict(x, columns))

    with pytest.raises(KeyError, match="missing trained columns"):
        model.predict(x[:, :5], columns[:5])


def test_reason_codes_are_three_merchant_readable_strings_from_pred_contrib() -> None:
    """FR-033. Native ``pred_contrib``, no separate SHAP dependency."""
    x, y, columns = _fit()
    model = rung2_lgbm.train(x, y, columns)
    rows = np.array([0, 1, 2, 3])
    codes = model.reason_codes(x, columns, rows)
    assert len(codes) == len(rows)
    for row in codes:
        assert len(row) == 3
        assert all(isinstance(text, str) and text.strip() for text in row)
    assert model.reason_codes(x, columns, np.array([], dtype=int)) == []


def test_a_residual_column_explains_itself_with_the_cohort_clause() -> None:
    """"You went up and your peers did not" is the half of the explanation that survives a
    merchant dispute (10-eval-harness-spec.md §5)."""
    base = "v_gmv_z"
    plain = rung2_lgbm.explain_column(base, 4.2)
    assert plain == registry.get(base).explain(4.2)
    residual = rung2_lgbm.explain_column(residual_columns()[0], 4.2)
    assert "comparable merchants" in residual


def test_the_artifact_and_the_fit_stay_inside_their_nfr_budgets(tmp_path: Path) -> None:
    """NFR-05 (<=20 MB) and NFR-06 (<=20 min on 4 cores), asserted rather than hoped."""
    x, y, columns = _fit()
    model = rung2_lgbm.train(x, y, columns)
    path = model.save(tmp_path / "rung2.txt")
    assert model.size_mb(path) <= 20.0
    assert model.train_seconds <= 20 * 60


def test_positive_merchants_are_counted_not_positive_rows() -> None:
    """Under a delayed-label regime the row count is the merchant count times a window
    length, and quoting it makes eight labelled merchants look like a thousand."""
    x, y, columns = _fit()
    merchant_id = np.array([f"M{i % 20:03d}" for i in range(x.shape[0])])
    model = rung2_lgbm.train(x, y, columns, merchant_id=merchant_id)
    assert model.n_train_positive_merchants <= 20
    assert model.n_train_positive_rows == int(y.sum())


def test_hyperparameters_carry_every_seed_lightgbm_reads() -> None:
    """Bagging, feature sampling and data ordering each have their own seed; leaving one
    on its default makes "same seed, same model" false in a way that only shows up as a
    number that will not reproduce."""
    params = rung2_lgbm.DEFAULT_PARAMS.booster_params()
    assert {"seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"} <= set(params)
    assert params["objective"] == "binary"
    assert params["deterministic"] is True
