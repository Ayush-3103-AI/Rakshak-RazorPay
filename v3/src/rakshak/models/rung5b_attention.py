"""Rung 5b - gated-attention MIL pooling over payer capsules (T-0131, GitHub #65).

Rung 5 (``rung5_mil``) scores each (payer, day) capsule with an instance-level LightGBM
and then collapses the bag through a **fixed-form** pooling: log-sum-exp with one fitted
scalar ``tau``. Rung 5b replaces *only that last step* with the gated attention mechanism
of Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance Learning" (ICML 2018),
so the bag score is a **learned** convex combination of the same instance probabilities::

    e_k     = w^T ( tanh(V h_k)  (*)  sigmoid(U h_k) )        (Ilse eq. 9, gated)
    a_k     = softmax_k(e_k)  within the bag
    s(bag)  = sum_k a_k * p_k

``(*)`` is the elementwise product. ``p_k`` is the **identical** LightGBM instance
probability Rung 5 pools; ``h_k`` is the capsule's own feature vector (below). This is
Ilse's *instance-level* aggregation variant rather than the embedding-level one, and that
choice is what makes the comparison single-variable: the instance scorer, the bags, the
splits, the seeds and the labels are all Rung 5's, and the only thing that moves is the
pooling function. An embedding-level variant would replace the LightGBM scorer too, and a
win could then be a win for the scorer rather than for attention.

## Why this nests the baseline rather than merely competing with it

LSE pooling is not a weighted mean, but its derivative is: ``d s / d p_k`` is
``softmax(tau * p_k)`` within the bag - the quantity ``rung5_mil._responsibilities``
already computes and calls the instance's "responsibility". Attention pooling makes that
weighting **explicit and learned**. Because ``logit(p_k)`` is one of the gate's inputs, a
gate of the form ``w^T tanh(V h)`` restricted to that column can reproduce a monotone
function of ``p_k``, so the attention family approximately contains the fixed-tau
weighting it is being measured against. If it loses, it is not because it was denied the
baseline's hypothesis.

## The instance representation

``h_k`` is the 13-column capsule vector (``CAPSULE_VECTOR_COLUMNS``, the same contract
``rung5_mil.feature_columns`` publishes) with ``logit(p_k)`` appended, all standardised by
means and standard deviations fitted on **training instances only**. 14 inputs.

## Parameter count, stated against the constraint that binds

``V`` and ``U`` are ``(L, 14)`` and ``w`` is ``(L,)``, with :data:`HIDDEN_DIM` = 8 and no
bias terms (Ilse eq. 9 has none): **232 trainable parameters**.

ADR-V3-001's §Reversal item 3 - "evidence that the label constraint no longer binds" - is
recorded in that ADR's amendment as **NOT SATISFIED and waived**. The binding constraint
is ~234 trainable positive merchants against 40,000 at the cycle-4 day-239 boundary. 232
parameters against 234 positives is roughly one free parameter per positive merchant.
That is the reason this rung is expected to fail its gate, it was written down before the
gate was run, and :func:`train_attention` reports the count on every fit so the ratio is
on the artifact rather than in a docstring only.

## Hyperparameters are declared, not tuned

ADR-V3-001's amendment sets the adoption gate in advance and Prime Directive 5 forbids
moving it after results are seen. The same logic applies to everything that could be
turned into a knob after a losing number: :data:`HIDDEN_DIM`, :data:`LEARNING_RATE`,
:data:`WEIGHT_DECAY` and :data:`EPOCHS` are fixed here, before the first run, and a losing
result is reported as a losing result rather than re-run at a different setting. There is
no early stopping and no epoch selection, because there is no third split to select on -
selecting on validation is what already makes Rung 5's own fitted tau selection-optimistic
(``score_rung5.fit_seed``'s docstring), and doing it twice for the challenger while the
baseline pays for it once would tilt the comparison the challenger's way.

The loss is plain unweighted binary cross-entropy on bag labels. No ``pos_weight``, for
the same reason ``rung2_lgbm`` carries no ``scale_pos_weight``: the decision layer consumes
the score *as* a probability, and the pooled score is only a probability while the
objective stays a proper scoring rule.

## Numerics and determinism

The within-bag softmax is max-shifted (the shift is ``detach``-ed, so it is a constant and
not a second gradient path) and the segmented reductions are ``scatter_reduce_`` /
``index_add_`` over a bag index that is **not** required to be sorted here - unlike
``rung5_mil.bag_offsets``, which needs contiguous runs for ``ufunc.reduceat``. Empty bags
receive :data:`~rakshak.models.rung5_mil.EMPTY_BAG_SCORE`, which is asserted to be ``0.0``
because ``index_add_`` into a zero-filled accumulator is what makes them free.

``torch`` is seeded through an explicit ``torch.Generator`` rather than a global, matching
the repo's "no module-level RNG" rule, and threads are pinned to 1 for scoring so the
charter §2 latency term is measured on one CPU core as it is written.

Prime Directive 3: this module never names a radioactive field. ``bag_y`` arrives as a
plain array assembled on the eval side.

torch is admitted for this rung and this rung only, by the 2026-09-02 AMENDMENT to
``docs/adr/ADR-V3-001-no-autograd.md``. No existing rung is rewritten onto it;
``rung5_mil`` is untouched and remains the baseline this module is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch

from rakshak.features.capsules import CAPSULE_VECTOR_COLUMNS
from rakshak.models.rung5_mil import EMPTY_BAG_SCORE, TrainedMIL

__all__ = [
    "EPOCHS",
    "HIDDEN_DIM",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "AttentionGate",
    "TrainedAttentionMIL",
    "gate_inputs",
    "pool_attention",
    "train_attention",
]

#: Ilse et al.'s ``L``. Declared, not tuned - see the module docstring.
HIDDEN_DIM: Final = 8

#: Full-batch Adam. Declared, not tuned.
LEARNING_RATE: Final = 1e-2
WEIGHT_DECAY: Final = 1e-4
EPOCHS: Final = 200

#: ``logit`` is undefined at the closed endpoints and LightGBM emits both.
_P_EPS: Final = 1e-6

# index_add_ into an accumulator pre-filled with EMPTY_BAG_SCORE is what gives empty bags
# the right score for free. If that constant ever stops being 0.0 the accumulator trick
# still holds, but the assertion below documents that the two are coupled.
assert EMPTY_BAG_SCORE == 0.0


@dataclass(frozen=True, slots=True)
class AttentionGate:
    """The three tensors of Ilse eq. 9, and the train-set standardisation they assume.

    Frozen and slotted like every other fitted object in ``models/``. The tensors carry
    ``requires_grad`` while :func:`train_attention` is fitting them and are detached for
    scoring, so a scored gate cannot silently accumulate a graph.
    """

    v: torch.Tensor
    u: torch.Tensor
    w: torch.Tensor
    mean: np.ndarray
    std: np.ndarray

    @property
    def n_parameters(self) -> int:
        """232 at :data:`HIDDEN_DIM` = 8. Reported on every artifact against the ~234
        trainable positive merchants that ADR-V3-001 records as the binding constraint."""
        return int(self.v.numel() + self.u.numel() + self.w.numel())

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Pre-softmax attention scores ``e_k``, one per instance."""
        return (torch.tanh(h @ self.v.T) * torch.sigmoid(h @ self.u.T)) @ self.w


def gate_inputs(
    x: np.ndarray, p: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h, mean, std)``: the standardised ``[capsule vector, logit(p)]`` gate input.

    ``mean``/``std`` are fitted here only when not supplied. Validation instances **must**
    pass the training statistics in; standardising validation by its own moments would let
    the gate see the validation distribution, which is a leak that no assertion downstream
    would catch because the resulting score is perfectly plausible.
    """
    feats = np.asarray(x, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[1] != len(CAPSULE_VECTOR_COLUMNS):
        raise ValueError(
            f"x must be (instances, {len(CAPSULE_VECTOR_COLUMNS)}) in CAPSULE_VECTOR_COLUMNS "
            f"order; got shape {feats.shape}"
        )
    prob = np.clip(np.asarray(p, dtype=np.float64), _P_EPS, 1.0 - _P_EPS)
    if prob.size != feats.shape[0]:
        raise ValueError(f"x has {feats.shape[0]} instances and p has {prob.size}")
    raw = np.column_stack([feats, np.log(prob / (1.0 - prob))])
    if mean is None or std is None:
        mean = raw.mean(axis=0)
        # A constant column has zero variance; dividing by 1 leaves it at 0 after centring,
        # which is the right answer (a constant carries no signal) rather than a nan.
        std = np.where(raw.std(axis=0) > 0.0, raw.std(axis=0), 1.0)
    return (raw - mean) / std, np.asarray(mean), np.asarray(std)


def pool_attention(
    e: torch.Tensor, p: torch.Tensor, bag_index: torch.Tensor, n_bags: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(bag_score, attention_weight)`` from instance gate logits and probabilities.

    The softmax is within-bag and max-shifted; the shift is detached so it contributes no
    gradient. Bags with no instances keep :data:`EMPTY_BAG_SCORE`, matching Rung 5 exactly
    - the two rungs must agree about what a zero-payer merchant-day is worth or the
    comparison is over different bag universes.
    """
    shift = torch.full((n_bags,), float("-inf"), dtype=e.dtype).scatter_reduce_(
        0, bag_index, e, reduce="amax", include_self=False
    )
    ex = torch.exp(e - shift.detach()[bag_index])
    denom = torch.zeros(n_bags, dtype=e.dtype).index_add_(0, bag_index, ex)
    a = ex / denom[bag_index]
    score = torch.full((n_bags,), float(EMPTY_BAG_SCORE), dtype=e.dtype).index_add_(
        0, bag_index, a * p
    )
    return score, a


@dataclass(frozen=True, slots=True)
class TrainedAttentionMIL:
    """A Rung 5 instance model with Rung 5b's learned pooling bolted on in place of LSE.

    ``instance`` is a :class:`~rakshak.models.rung5_mil.TrainedMIL` and is **not refitted**
    here - it is the object Rung 5 was scored through. ``predict`` keeps the
    ``explain.registry.Scorer`` shape (``predict(x, columns)``) for the same reason
    ``TrainedMIL.predict`` does: without ``bag_index`` every row is a singleton bag, and
    attention over one instance is ``a = 1``, so the fallback is the instance probability
    itself rather than a degraded score.
    """

    instance: TrainedMIL
    gate: AttentionGate
    n_train_bags: int
    n_train_positive_bags: int
    n_train_instances: int
    final_train_loss: float
    rung: str = "5b"

    @property
    def columns(self) -> tuple[str, ...]:
        return self.instance.columns

    def _weights(
        self, x: np.ndarray, columns: tuple[str, ...], bag_index: np.ndarray, n_bags: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.instance.instance_probabilities(x, columns)
        h, _, _ = gate_inputs(x, p, self.gate.mean, self.gate.std)
        with torch.no_grad():
            e = self.gate.logits(torch.as_tensor(h, dtype=torch.float64))
            score, a = pool_attention(
                e,
                torch.as_tensor(p, dtype=torch.float64),
                torch.as_tensor(np.asarray(bag_index), dtype=torch.int64),
                n_bags,
            )
        return score.numpy(), a.numpy(), p

    def predict(
        self,
        x: np.ndarray,
        columns: tuple[str, ...],
        *,
        bag_index: np.ndarray | None = None,
        n_bags: int | None = None,
    ) -> np.ndarray:
        if bag_index is None:
            return self.instance.instance_probabilities(x, columns)
        idx = np.asarray(bag_index)
        total = int(idx.max()) + 1 if n_bags is None and idx.size else (n_bags or 0)
        return self._weights(x, columns, idx, total)[0]

    def attention(
        self, x: np.ndarray, columns: tuple[str, ...], bag_index: np.ndarray, n_bags: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(attention_weight, instance_probability)`` per capsule.

        This is the acceptance criterion the ticket calls "the payoff for the added
        complexity": it answers *which payer* moved the bag, which fixed pooling cannot -
        LSE's implicit weights are a function of ``p`` alone, so they rank the same way the
        instance model already does and say nothing new.
        """
        _, a, p = self._weights(x, columns, np.asarray(bag_index), n_bags)
        return a, p

    def summary(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "pooling": "gated_attention",
            "hidden_dim": HIDDEN_DIM,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "n_attention_parameters": self.gate.n_parameters,
            "n_train_bags": self.n_train_bags,
            "n_train_positive_bags": self.n_train_positive_bags,
            "n_train_instances": self.n_train_instances,
            "final_train_loss": self.final_train_loss,
            "instance_model": self.instance.summary(),
        }


def train_attention(
    instance: TrainedMIL,
    x: np.ndarray,
    bag_index: np.ndarray,
    bag_y: np.ndarray,
    columns: tuple[str, ...],
    *,
    seed: int,
    epochs: int = EPOCHS,
    hidden_dim: int = HIDDEN_DIM,
    echo: Any = None,
) -> TrainedAttentionMIL:
    """Fit the gate on **training bags only**, against a frozen Rung 5 instance model.

    Same guarantee as ``rung5_mil.train`` and ``rung2_lgbm.train``: no validation data
    enters. Unlike Rung 5, nothing is selected on validation afterwards either - there is
    no ``fit_tau`` analogue, because the gate is fitted rather than chosen off a grid.

    ``x`` rows must correspond to ``bag_index`` rows; ordering is irrelevant here (the
    reductions are scatters, not ``reduceat``), but the caller is expected to hand over the
    same grouped matrix Rung 5 uses so the two rungs pool the same instances.
    """
    labels = np.asarray(bag_y)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("bag_y must be 0/1 bag labels, one per bag")
    n_bags = int(labels.size)
    idx = np.asarray(bag_index)
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_bags):
        raise ValueError(f"bag_index values must lie in [0, {n_bags})")

    p_np = instance.instance_probabilities(x, columns)
    h_np, mean, std = gate_inputs(x, p_np)

    gen = torch.Generator().manual_seed(seed)
    n_in = h_np.shape[1]
    # Xavier-uniform by hand rather than nn.init, so the draw is explicitly generator-fed
    # and the fit is reproducible without touching torch's global RNG.
    def _draw(rows: int, cols: int) -> torch.Tensor:
        bound = float(np.sqrt(6.0 / (rows + cols)))
        t = torch.empty(rows, cols, dtype=torch.float64).uniform_(-bound, bound, generator=gen)
        return t.requires_grad_(True)

    v = _draw(hidden_dim, n_in)
    u = _draw(hidden_dim, n_in)
    w = _draw(hidden_dim, 1).squeeze(1).detach().requires_grad_(True)

    h = torch.as_tensor(h_np, dtype=torch.float64)
    p = torch.as_tensor(p_np, dtype=torch.float64)
    bags = torch.as_tensor(idx, dtype=torch.int64)
    y = torch.as_tensor(labels.astype(np.float64), dtype=torch.float64)

    opt = torch.optim.Adam([v, u, w], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_value = float("nan")
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        e = (torch.tanh(h @ v.T) * torch.sigmoid(h @ u.T)) @ w
        score, _ = pool_attention(e, p, bags, n_bags)
        # clamp, not eps-add: the pooled score is a convex combination of clipped instance
        # probabilities and cannot reach 0 or 1, but an empty bag sits exactly at 0.0.
        s = score.clamp(_P_EPS, 1.0 - _P_EPS)
        loss = -(y * torch.log(s) + (1.0 - y) * torch.log1p(-s)).mean()
        # torch ships `Tensor.backward` untyped, so --strict flags the call, not the code.
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_value = float(loss.detach())
        if echo is not None and (epoch % 25 == 0 or epoch == epochs - 1):
            echo(f"    epoch {epoch:4d}  bce={loss_value:.6f}")

    gate = AttentionGate(
        v=v.detach(), u=u.detach(), w=w.detach(), mean=mean, std=std
    )
    return TrainedAttentionMIL(
        instance=instance,
        gate=gate,
        n_train_bags=n_bags,
        n_train_positive_bags=int((labels == 1).sum()),
        n_train_instances=int(np.asarray(x).shape[0]),
        final_train_loss=loss_value,
    )
