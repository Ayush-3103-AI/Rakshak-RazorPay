"""Rung 5b - gated-attention MIL pooling (T-0131, GitHub #65).

Four things are worth a runnable check here, and nothing else is:

1. **The pooling reduces to things we can compute by hand.** A flat gate is the arithmetic
   mean and a saturated gate is the argmax instance - the same two limits ``rung5_mil``
   asserts for ``tau -> 0`` and ``tau -> inf``. If attention could not reach them it would
   not nest the baseline it is being measured against, and a loss would be uninterpretable.
2. **Empty bags agree with Rung 5 exactly.** The two rungs must price a zero-payer
   merchant-day identically or the pooled PR-AUC comparison is over different bag universes
   and the whole adoption gate is meaningless.
3. **The fit is deterministic at a fixed seed**, because a rung whose number moves between
   runs is not a result.
4. **The gradient path actually learns.** On a toy whose label IS witness-driven, learned
   pooling must beat BOTH endpoints of the fixed family - which separates "the optimiser
   ran" from "the optimiser worked". It says nothing about the real data.

The parameter count is asserted too, because it is the number the ADR amendment says
decides this rung, and a silent change to ``HIDDEN_DIM`` would move it without failing
anything else.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rakshak.eval.metrics import pr_auc
from rakshak.features.capsules import CAPSULE_VECTOR_COLUMNS
from rakshak.models.rung2_lgbm import DEFAULT_PARAMS
from rakshak.models.rung5_mil import EMPTY_BAG_SCORE, pool
from rakshak.models.rung5_mil import train as train_mil
from rakshak.models.rung5b_attention import (
    HIDDEN_DIM,
    AttentionGate,
    gate_inputs,
    pool_attention,
    train_attention,
)

N_FEATURES = len(CAPSULE_VECTOR_COLUMNS)


def _t(a: list[float]) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float64)


def _bags(a: list[int]) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.int64)


def test_flat_gate_is_exactly_mean_pooling() -> None:
    """Equal logits => equal attention => the arithmetic mean, which is ``rung5_mil``'s
    ``tau = 0`` branch. Asserted against that branch rather than against a hand-typed
    number, so the two rungs are held to the same definition of mean-pooling."""
    p = np.array([0.1, 0.9, 0.4, 0.7, 0.2])
    bag = np.array([0, 0, 0, 1, 1])
    score, a = pool_attention(torch.zeros(5, dtype=torch.float64), _t(list(p)), _bags(list(bag)), 2)
    expected = pool(p, bag, 2, tau=0.0)
    assert np.allclose(score.numpy(), expected, atol=1e-12)
    assert np.allclose(a.numpy(), [1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])


def test_saturated_gate_is_exactly_max_pooling_when_it_points_at_the_max() -> None:
    """A gate that concentrates on the largest ``p`` reproduces ``tau = inf``. This is the
    other endpoint of the family Rung 5 fitted over, so attention spans it."""
    p = np.array([0.1, 0.9, 0.4, 0.2, 0.7])
    bag = np.array([0, 0, 0, 1, 1])
    e = _t([0.0, 500.0, 0.0, 0.0, 500.0])  # 500 would overflow exp() without the max-shift
    score, a = pool_attention(e, _t(list(p)), _bags(list(bag)), 2)
    assert np.allclose(score.numpy(), pool(p, bag, 2, tau=float("inf")), atol=1e-12)
    assert np.isfinite(a.numpy()).all()


def test_attention_weights_sum_to_one_within_every_bag() -> None:
    rng = np.random.default_rng(0)
    bag = np.repeat(np.arange(7), rng.integers(1, 6, size=7))
    e = torch.as_tensor(rng.normal(scale=4.0, size=bag.size), dtype=torch.float64)
    p = torch.as_tensor(rng.uniform(size=bag.size), dtype=torch.float64)
    score, a = pool_attention(e, p, _bags(bag.tolist()), 7)
    sums = np.bincount(bag, weights=a.numpy(), minlength=7)
    assert np.allclose(sums, 1.0)
    # A convex combination of probabilities is a probability, which is what the decision
    # layer consumes; if this drifts the expected-cost arithmetic downstream is wrong.
    assert score.numpy().min() >= 0.0 and score.numpy().max() <= 1.0


def test_empty_bags_score_the_same_as_rung5() -> None:
    """Bag 1 has no instances. Rung 5 gives it ``EMPTY_BAG_SCORE``; so must 5b."""
    p = np.array([0.3, 0.8])
    bag = np.array([0, 2])
    score, _ = pool_attention(torch.zeros(2, dtype=torch.float64), _t(list(p)), _bags([0, 2]), 3)
    assert score.numpy()[1] == EMPTY_BAG_SCORE
    assert np.allclose(score.numpy(), pool(p, bag, 3, tau=0.0))


def test_gate_input_rejects_a_matrix_that_is_not_the_capsule_contract() -> None:
    with pytest.raises(ValueError, match="CAPSULE_VECTOR_COLUMNS"):
        gate_inputs(np.zeros((4, 3)), np.zeros(4))


def test_gate_inputs_reuse_training_moments_rather_than_refitting() -> None:
    """Validation instances standardised by their own moments is a leak that produces a
    perfectly plausible number, so it is checked rather than trusted."""
    rng = np.random.default_rng(1)
    x_tr = rng.normal(size=(50, N_FEATURES))
    p_tr = rng.uniform(0.1, 0.9, size=50)
    _, mean, std = gate_inputs(x_tr, p_tr)
    x_val = rng.normal(loc=5.0, size=(20, N_FEATURES))
    p_val = rng.uniform(0.1, 0.9, size=20)
    h_val, _, _ = gate_inputs(x_val, p_val, mean, std)
    assert not np.allclose(h_val.mean(axis=0), 0.0, atol=0.1)  # shifted, as it should be


def test_parameter_count_is_the_number_the_adr_gate_is_argued_against() -> None:
    """232 parameters against ~234 trainable positive merchants. ADR-V3-001's amendment
    records item 3 (the label constraint) as NOT SATISFIED, and this is that number."""
    gate = AttentionGate(
        v=torch.zeros(HIDDEN_DIM, N_FEATURES + 1, dtype=torch.float64),
        u=torch.zeros(HIDDEN_DIM, N_FEATURES + 1, dtype=torch.float64),
        w=torch.zeros(HIDDEN_DIM, dtype=torch.float64),
        mean=np.zeros(N_FEATURES + 1),
        std=np.ones(N_FEATURES + 1),
    )
    assert gate.n_parameters == 2 * HIDDEN_DIM * (N_FEATURES + 1) + HIDDEN_DIM == 232


def _planted_bags(
    rng: np.random.Generator, n_bags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bags of 6 instances. A positive bag contains exactly one witness, flagged in
    column 0. Column 1 is noise correlated with nothing."""
    per = 6
    x = rng.normal(size=(n_bags * per, N_FEATURES))
    x[:, 0] = 0.0
    y = (rng.uniform(size=n_bags) < 0.35).astype(np.int8)
    for b in np.flatnonzero(y):
        x[b * per + int(rng.integers(per)), 0] = 1.0
    return x, np.repeat(np.arange(n_bags), per), y


def test_training_is_deterministic_and_beats_both_fixed_endpoints() -> None:
    rng = np.random.default_rng(7)
    x, bag, y = _planted_bags(rng, 300)
    columns = CAPSULE_VECTOR_COLUMNS
    instance = train_mil(x, bag, y, columns, params=DEFAULT_PARAMS.with_seed(42))

    a = train_attention(instance, x, bag, y, columns, seed=42)
    b = train_attention(instance, x, bag, y, columns, seed=42)
    assert torch.equal(a.gate.v, b.gate.v) and torch.equal(a.gate.w, b.gate.w)

    # The gate must not stay flat: a uniform 1/6 everywhere is what an untrained gate
    # gives, so a non-degenerate spread is the evidence the optimiser did something.
    weights, p = a.attention(x, columns, bag, y.size)
    assert weights.std() > 0.05

    # And on a bag whose label IS witness-driven by construction, learned pooling must
    # beat BOTH endpoints of the fixed family Rung 5 fits over. This is the check that
    # separates "the optimiser ran" from "the optimiser worked"; note that it says nothing
    # about the real data, where the same comparison is the adoption gate and is measured,
    # not assumed. (`weights[x[:, 0] == 1].mean()` is deliberately NOT asserted: the
    # instance LightGBM does not itself isolate the witness from 12 noise columns under
    # bag-label propagation, so the gate wins by re-weighting, not by finding the witness.)
    score = a.predict(x, columns, bag_index=bag, n_bags=y.size)
    assert pr_auc(y, score) > pr_auc(y, pool(p, bag, y.size, tau=0.0))
    assert pr_auc(y, score) > pr_auc(y, pool(p, bag, y.size, tau=float("inf")))
    assert a.gate.n_parameters == 232
