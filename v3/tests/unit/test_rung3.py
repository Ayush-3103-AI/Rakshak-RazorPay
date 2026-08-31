"""T-142 - Rung 3, and the single-variable claim the K-1 verdict rests on.

The ticket's Test field is explicit: *asserts the feature sets differ by exactly the
residual columns*. That is what most of this file does, from both directions - a column
added, a column dropped, a residual renamed - because "identical except for X" is a claim
that is easy to make and easy to stop being true after one edit.

The delta itself, and whether charter K-1 fired, is in ``docs/logbook/T-142.md`` and in
``LIMITATIONS.md``. It is not asserted here: a test that requires the hypothesis to hold
would have to be deleted the day it does not, and deleting a test to record a falsification
is exactly backwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.features import registry
from rakshak.models import rung2_lgbm, rung3_cohort
from rakshak.models.dataset import RESIDUAL_PREFIX, base_columns, residual_columns


def test_rung3_is_rung2_plus_exactly_the_residual_block() -> None:
    base = base_columns()
    full = rung3_cohort.feature_columns()
    assert full[: len(base)] == base
    assert tuple(c for c in full if c not in set(base)) == residual_columns()
    assert rung3_cohort.assert_single_variable(base, full) == residual_columns()


def test_every_residual_names_a_feature_that_declared_one() -> None:
    """The residual block is the register's ``has_cohort_residual`` flag and nothing else.
    A hand-maintained list would drift from the register the first time one moved."""
    flagged = {n for n in registry.ORDER if registry.REGISTRY[n].has_cohort_residual}
    assert {c[len(RESIDUAL_PREFIX) :] for c in residual_columns()} == flagged
    assert len(residual_columns()) == len(flagged)


def test_an_extra_column_is_refused() -> None:
    base = base_columns()
    with pytest.raises(ValueError, match="exactly the registered cohort-residual"):
        rung3_cohort.assert_single_variable(base, (*base, *residual_columns(), "a_smuggled_one"))


def test_a_dropped_column_is_refused() -> None:
    """Dropping a Rung-2 column would make the delta attributable to the removal as much
    as to the residuals, which is the same failure wearing the opposite sign."""
    base = base_columns()
    with pytest.raises(ValueError, match="dropped"):
        rung3_cohort.assert_single_variable(base, (*base[:-1], *residual_columns()))


def test_a_renamed_residual_is_refused() -> None:
    base = base_columns()
    wrong = (*residual_columns()[:-1], "r_not_a_feature")
    with pytest.raises(ValueError, match="exactly the registered cohort-residual"):
        rung3_cohort.assert_single_variable(base, (*base, *wrong))


def test_rung3_trains_through_rung2s_code_path_with_rung2s_hyperparameters() -> None:
    """A thin delegation on purpose. If the two rungs could drift apart, the
    single-variable claim would rest on nobody having edited one of them."""
    columns = rung3_cohort.feature_columns()
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3000, len(columns)))
    logit = -5.0 + 1.6 * x[:, columns.index("v_gmv_z")]
    y = (rng.random(3000) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int8)

    model = rung3_cohort.train(x, y, columns, rung2_columns=base_columns())
    assert model.rung == 3
    assert model.columns == columns
    assert model.params == rung2_lgbm.DEFAULT_PARAMS

    narrow = rung2_lgbm.train(x[:, : len(base_columns())], y, base_columns())
    assert narrow.params.seed == model.params.seed
    assert narrow.params == model.params


def test_training_rung3_on_a_column_set_that_is_not_single_variable_refuses() -> None:
    """The guard runs before the fit, not before the write-up. A single-variable claim
    checked after the fact is a claim about a number already written down."""
    columns = (*base_columns(), *residual_columns(), "an_extra")
    x = np.zeros((10, len(columns)))
    y = np.ones(10, dtype=np.int8)
    with pytest.raises(ValueError, match="exactly the registered cohort-residual"):
        rung3_cohort.train(x, y, columns, rung2_columns=base_columns())
