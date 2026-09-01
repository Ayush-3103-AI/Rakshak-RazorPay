"""T-0118 — the decision-policy seam, and the golden-output regression on it.

T-0118 asks for "a byte-identical ``summary.md`` at a fixed seed". No ``summary.md`` exists
in this tree (nothing under ``src/rakshak/eval/report.py`` writes one), so the regression is
run against the thing that number is actually made of: the action array. If
``DEFAULT_DECISION`` emits the same actions as the pre-refactor ``select_actions`` on the
same fixed-seed input, every downstream number — savings, floors, P@K, TTD and any report
rendered from them — is identical by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.eval.capacity import (
    DEFAULT_DECISION,
    ActionPolicy,
    CapacityTopK,
    DecisionPolicy,
    DecisionRequest,
    select_actions,
    sweep_cost_asymmetry,
)
from rakshak.eval.metrics import CostParams
from rakshak.schemas import Action

PARAMS = CostParams()
K = 50


def fixture(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1,000 merchants x 20 days at a fixed seed. Synthetic; no split is read."""
    n_merchants, n_days = 1000, 20
    day = np.tile(np.arange(n_days), n_merchants)
    rng = np.random.default_rng(seed)
    return rng.random(day.size), day, rng.uniform(10_000, 500_000, day.size)


# ──────────────────────── the golden-output regression ────────────────────────


@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
def test_the_unwrapped_policy_is_identical_to_the_pre_refactor_selector(seed: int) -> None:
    score, day, exposure = fixture(seed)
    before = select_actions(score, day, exposure, K, PARAMS)
    after = DEFAULT_DECISION.decide(
        DecisionRequest(score=score, day=day, exposure_inr=exposure, k=K, params=PARAMS)
    )
    assert np.array_equal(before, after)


def test_the_seam_carries_the_hold_policy_through_unchanged() -> None:
    score, day, exposure = fixture()
    strict = ActionPolicy(hold_score_threshold=0.999, hold_expected_loss_floor_inr=1e9)
    before = select_actions(score, day, exposure, K, PARAMS, strict)
    after = DEFAULT_DECISION.decide(
        DecisionRequest(
            score=score,
            day=day,
            exposure_inr=exposure,
            k=K,
            params=PARAMS,
            hold_policy=strict,
        )
    )
    assert np.array_equal(before, after)
    assert not (after == Action.HOLD).any()  # the strict policy permits no HOLD at all


def test_the_sweep_is_unmoved_by_being_routed_through_the_seam() -> None:
    """Rungs 0-4 keep the savings they were scored on (pre-registration §3)."""
    score, day, exposure = fixture()
    rng = np.random.default_rng(7)
    y = (rng.random(day.size) < 0.015).astype(np.int8)
    loss = np.where(y == 1, rng.uniform(50_000, 400_000, day.size), 0.0)

    explicit = sweep_cost_asymmetry(
        {"a": score, "b": 1.0 - score},
        day,
        y,
        loss,
        exposure,
        K,
        PARAMS,
        decision=CapacityTopK(),
    )
    default = sweep_cost_asymmetry(
        {"a": score, "b": 1.0 - score}, day, y, loss, exposure, K, PARAMS
    )
    assert explicit == default


# ──────────────────────────── the seam is wrappable ────────────────────────────


class _SoftenAllHolds:
    """The smallest possible wrapper — the shape Rung 6's conformal layer will take.

    It softens HOLD to REVIEW and nothing else, which is the only direction the
    ``DecisionPolicy`` contract permits: softening cannot breach K, promoting can.
    """

    def __init__(self, inner: DecisionPolicy) -> None:
        self.inner = inner

    @property
    def name(self) -> str:
        return f"soften({self.inner.name})"

    def decide(self, request: DecisionRequest) -> np.ndarray:
        inner: np.ndarray = self.inner.decide(request)
        return np.where(inner == Action.HOLD, Action.REVIEW, inner)


def test_a_wrapper_can_soften_without_touching_the_selector() -> None:
    score, day, exposure = fixture()
    wrapped: DecisionPolicy = _SoftenAllHolds(DEFAULT_DECISION)
    base = DEFAULT_DECISION.decide(
        DecisionRequest(score=score, day=day, exposure_inr=exposure, k=K, params=PARAMS)
    )
    out = wrapped.decide(
        DecisionRequest(score=score, day=day, exposure_inr=exposure, k=K, params=PARAMS)
    )

    assert (base == Action.HOLD).any(), "fixture must produce HOLDs or this proves nothing"
    assert not (out == Action.HOLD).any()
    assert np.array_equal(out != Action.PASS, base != Action.PASS)  # alert set unchanged
    assert wrapped.name == "soften(capacity_topk)"


def test_a_wrapper_cannot_smuggle_extra_alerts_past_capacity() -> None:
    """The K ceiling is a property of the emitted actions, so it is checkable on any
    policy, wrapped or not — the seam does not create a route around it."""
    score, day, exposure = fixture()
    out = _SoftenAllHolds(DEFAULT_DECISION).decide(
        DecisionRequest(score=score, day=day, exposure_inr=exposure, k=K, params=PARAMS)
    )
    for d in np.unique(day):
        assert int((out[day == d] != Action.PASS).sum()) <= K


def test_the_default_policy_is_named() -> None:
    assert DEFAULT_DECISION.name == "capacity_topk"


def test_the_request_carries_no_label_and_no_loss() -> None:
    """Prime Directive 3, enforced by the shape of the seam rather than by review."""
    banned = {"y", "label", "loss", "loss_inr", "true_loss_amount_inr", "truth", "onset"}
    assert banned.isdisjoint(DecisionRequest.__dataclass_fields__)
