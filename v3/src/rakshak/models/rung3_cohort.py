"""Rung 3 - Rung 2 plus the cohort-residual columns. **This is the K-1 test.**

Charter K-1 is the sprint's central hypothesis: that subtracting a merchant's cohort's
own movement separates adversarial drift from platform drift well enough to be worth
building. T-121 proved the *mechanism* works - the common mode is removed exactly, median
z +1.790 becoming median residual +0.005. This rung asks the only question that matters
after that: **does it move a number a risk team would act on.**

So the experiment is single-variable by construction, and this module exists mostly to
make that checkable rather than promised:

- the same ``HParams`` instance as Rung 2, including the seed;
- the same rows, the same split, the same labels, the same training as_of;
- ``assert_single_variable`` refuses to train if the two column sets differ by anything
  other than exactly the registered residual columns.

The last one is the point. "We added the cohort features and also bumped num_leaves"
produces a delta attributable to nothing, and a falsification that cannot be attributed
is not a falsification.

**If the validation delta is under 5% relative, K-1 has fired.** Write the number into
``LIMITATIONS.md`` and stop. Do not add features, do not tune, do not re-run at a
different seed to rescue it. Prime Directive 6: a rung that loses is a finding.

Read ``LIMITATIONS.md`` §3 and §6 before reading the result. Two things are already
measured and both bear on it: the residual layer cuts confounder-driven alerts only from
39.0% to 23.1%, because it cannot remove per-merchant heterogeneity; and gate G5 already
passes for the *raw* detector on this generator (+1.27pp against a +2pp allowance). A
small Rung-3 gain is therefore **consistent with what G5 already said**, not a surprise.
"""

from __future__ import annotations

import numpy as np

from rakshak.models.dataset import base_columns, residual_columns
from rakshak.models.rung2_lgbm import DEFAULT_PARAMS, HParams, TrainedRung
from rakshak.models.rung2_lgbm import train as _train_lgbm

__all__ = ["assert_single_variable", "feature_columns", "train"]


def feature_columns() -> tuple[str, ...]:
    """Rung 2's columns, then the residual block, in ``registry.ORDER`` both times."""
    return (*base_columns(), *residual_columns())


def assert_single_variable(
    rung2_columns: tuple[str, ...], rung3_columns: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the added columns, or raise if the two sets differ by anything else.

    Called before training, not after reporting. A single-variable claim checked after the
    fact is a claim about a number that has already been written down.
    """
    added = tuple(c for c in rung3_columns if c not in set(rung2_columns))
    removed = tuple(c for c in rung2_columns if c not in set(rung3_columns))
    expected = residual_columns()
    if removed:
        raise ValueError(
            f"Rung 3 dropped {len(removed)} of Rung 2's columns {removed[:5]}. The delta "
            "would then be attributable to the removal as much as to the residuals."
        )
    if added != expected:
        raise ValueError(
            "Rung 3 must differ from Rung 2 by exactly the registered cohort-residual "
            f"columns (FR-031). Expected {len(expected)} added columns, got {len(added)}; "
            f"unexpected: {sorted(set(added) ^ set(expected))}"
        )
    return added


def train(
    x: np.ndarray,
    y: np.ndarray,
    columns: tuple[str, ...],
    *,
    rung2_columns: tuple[str, ...],
    params: HParams = DEFAULT_PARAMS,
    merchant_id: np.ndarray | None = None,
) -> TrainedRung:
    """Identical to :func:`rakshak.models.rung2_lgbm.train`, on the wider column set.

    Deliberately a thin delegation rather than a copy. If the two rungs' training code
    could drift apart, the single-variable claim would rest on nobody having edited one of
    them, which is not a claim - it is a hope.
    """
    assert_single_variable(rung2_columns, columns)
    return _train_lgbm(x, y, columns, rung=3, params=params, merchant_id=merchant_id)
