"""``scripts/cost_sweep.py`` duplicates one thing, and this is the test that pins it.

``sweep_cost_asymmetry`` builds its swept ``CostParams`` internally and does not export
them, so the floor side of the sweep (Table B) has to rebuild the same object. Adding an
exported helper to ``eval/capacity.py`` was the obvious alternative and is not available:
``capacity.py`` is inside ``eval_module_sha256``, and changing that hash voids the cycle's
central claim that the harness which scored cycle 3 scored cycle 4 byte-identically.

So the duplication is deliberate, four lines long, and checked here rather than trusted.
If ``sweep_cost_asymmetry`` ever changes how it derives ``false_hold_cost_inr`` from the
ratio, this fails — which is the point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from rakshak.eval.capacity import (
    ASYMMETRY_RATIOS,
    DEFAULT_POLICY,
    select_actions,
    sweep_cost_asymmetry,
)
from rakshak.eval.metrics import CostParams, savings_of_actions

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from cost_sweep import _swept  # noqa: E402

PARAMS = CostParams(review_cost_inr=250.0, false_hold_cost_inr=8000.0, p_catch=0.8)


def _fixture() -> tuple[np.ndarray, ...]:
    """A small window with real fraud in it: 40 merchant-days over 4 days, 8 positive."""
    rng = np.random.default_rng(7)
    n = 40
    day = np.repeat(np.arange(4), 10)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, 8, replace=False)] = 1
    loss = np.where(y == 1, rng.uniform(50_000, 400_000, n), 0.0)
    score = np.clip(0.15 + 0.7 * y + rng.normal(0, 0.12, n), 0.0, 1.0)
    exposure = rng.uniform(20_000, 900_000, n)
    return day, y, loss, score, exposure


@pytest.mark.parametrize("ratio", ASYMMETRY_RATIOS)
def test_swept_matches_the_sweeps_own_cost_params(ratio: float) -> None:
    """Scoring one rung through ``_swept`` by hand reproduces the sweep's own row exactly."""
    day, y, loss, score, exposure = _fixture()
    k = 3

    (row,) = sweep_cost_asymmetry(
        {"only": score}, day, y, loss, exposure, k, PARAMS, ratios=(ratio,)
    )

    swept = _swept(PARAMS, ratio, loss, y)
    by_hand = savings_of_actions(
        select_actions(score, day, exposure, k, swept, DEFAULT_POLICY), y, loss, swept
    )

    assert row.savings == by_hand, (
        f"the floor side of the sweep is pricing at a different cost matrix from the rung "
        f"side at ratio {ratio}: {row.savings} vs {by_hand}"
    )


def test_swept_reference_is_the_mean_fraud_loss() -> None:
    """The ratio is against mean fraud loss, not mean loss over all rows."""
    _day, y, loss, _score, _exposure = _fixture()
    assert _swept(PARAMS, 2.0, loss, y).false_hold_cost_inr == pytest.approx(
        2.0 * loss[y == 1].mean()
    )
    # The distinction is not cosmetic here: 8 of 40 rows carry a loss, so a mean over all
    # rows would be 5x smaller and every swept cost matrix would be wrong by that factor.
    assert loss.mean() < loss[y == 1].mean()


def test_hold_is_unreachable_under_table_cs_policy() -> None:
    """Table C's isolation actually forbids HOLD, rather than merely discouraging it."""
    from rakshak.eval.capacity import ActionPolicy
    from rakshak.schemas import Action

    day, y, loss, score, exposure = _fixture()
    no_hold = ActionPolicy(
        hold_score_threshold=DEFAULT_POLICY.hold_score_threshold,
        hold_expected_loss_floor_inr=float("inf"),
    )
    swept = _swept(PARAMS, ASYMMETRY_RATIOS[0], loss, y)
    actions = select_actions(score, day, exposure, 3, swept, no_hold)
    assert not (actions == Action.HOLD).any()
    # ... and the comparison is only worth making if HOLD is otherwise reachable at this
    # ratio, so assert the control fires too. A "no HOLDs" result on a run that would never
    # have held anything proves nothing.
    permissive = select_actions(score, day, exposure, 3, swept, ActionPolicy(0.0, 0.0))
    assert (permissive == Action.HOLD).any()
