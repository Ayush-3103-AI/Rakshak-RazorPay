"""Rung 5 - multiple-instance learning by pooling over (payer, day) capsules (T-0120).

Rungs 0-4 score a **merchant-day vector**: one row per merchant per day, every payer
already summed away. Rung 5 scores the **bag** the aggregate came from. The instances are
the (payer, day) capsules from ``features.capsules``; an instance-level LightGBM scores
each capsule; a pooling function turns the bag of instance probabilities into the one
number the decision layer consumes.

ADR-V3-001 holds: no torch, so this is pooling, not attention. Attention would need a
learned query and a backward pass; pooling has one scalar and a closed form.

---

## The pooling, and why LSE is the family and noisy-OR is a comparator

Two forms were on the ticket and it left the choice open:

    noisy-OR :  s(bag) = 1 - prod_i (1 - p_i)
    LSE      :  s(bag) = (1/tau) * log( (1/n) * sum_i exp(tau * p_i) )

**LSE is the pooling; noisy-OR is scored beside it and reported, not assumed away.**
The reason is that only LSE is a *family*. ``tau -> 0`` is mean-pooling, which is what the
hand-built T1 register has effectively been assuming all along; ``tau -> inf`` is
max-pooling, the classic multiple-instance assumption that a bag is positive if any
instance is. One fitted scalar therefore places this problem on the
any-instance-to-all-instances axis by measurement rather than by assertion, and **the
fitted tau is itself the result** (:func:`fit_tau` returns the table, and
:meth:`TrainedMIL.summary` puts it in the results row).

Noisy-OR has no such knob, and it has a specific failure this dataset will provoke: it is
monotone in *bag size*. Two hundred payers at p=0.02 pool to 0.98 and three payers at
p=0.02 pool to 0.06, with identical per-payer evidence. Merchant-day bag sizes here span
orders of magnitude and the typologies move transaction volume, so a noisy-OR bag score
is partly a transaction counter wearing a probability's clothes. That is a reason to
expect it to lose, not a reason to skip it - it is on the grid in :func:`fit_tau` and if
it wins, it wins.

Pooling is over **probabilities**, not logits. The decision layer consumes the score *as*
a probability to compute expected cost (see ``rung2_lgbm``'s note on why there is no
``scale_pos_weight``), and LSE over ``p in [0, 1]`` returns a value in
``[min p_i, max p_i]``, so the bag score stays inside the unit interval and stays
monotone in every instance. Pooling logits would make ``tau -> 0`` the mean *logit*,
which is not the quantity the register has been assuming.

## Numerical stability

Both forms overflow or underflow if written the way they are printed above, and the
failure is silent - ``nan`` propagates into a PR-AUC and comes out as a number.

- LSE is max-shifted per bag, so ``exp`` never sees a positive argument. Written the way
  it is printed, ``tau = 1e6`` makes ``exp(tau * p)`` overflow to ``inf`` and the ratio
  ``inf/inf`` becomes ``nan``.
- **The max shift alone is not enough, and the first version of this module was wrong
  because of it.** The textbook form ``max + log(sum(exp(v - max)))`` is then divided by
  ``tau``, and as ``tau -> 0`` that divides a difference between two numbers of order
  ``log(n)`` whose true difference is of order ``tau`` - so at ``tau = 1e-8`` every digit
  is gone. It returned ``0.0`` for a bag whose mean was ``2.5e-12``; the unit test caught
  it, which is the entire reason the test is written against the extreme case rather than
  against a comfortable one. :func:`_segment_lse_mean` uses
  ``max + log1p(mean(expm1(v - max)))`` instead, which never forms the ``log(n)`` at all
  and is exact at both ends.
- The ``tau -> 0`` and ``tau -> inf`` limits are additionally **exact branches**:
  ``tau == 0.0`` *is* the arithmetic mean and ``tau == inf`` *is* the maximum. The grid
  contains both endpoints, so the two limits the acceptance criteria name are identities
  in this code, and the interior is asserted to converge to them separately - a
  special-cased endpoint that agrees with nothing near it is a special case, not a limit.
- Noisy-OR is computed as ``-expm1(sum_i log1p(-p_i))``. The printed form
  ``1 - prod(1 - p_i)`` underflows to exactly ``0.0`` for a bag of small probabilities -
  five instances at ``p = 1e-300`` pool to ``0.0`` instead of ``5e-300``, and a bag that
  should be ranked above an all-zero bag is ranked equal to it. ``log1p``/``expm1`` keep
  full relative precision at both ends. ``p = 1.0`` gives ``log1p(-1) = -inf``, which sums
  to ``-inf`` and returns exactly ``1.0``; that is correct, so the divide warning is
  silenced rather than the input clipped.

``tests/unit/test_rung5_mil.py`` asserts each of these against the naive form as a
negative control, because "it is stable" is a claim about arithmetic and arithmetic can
be run.

## The empty bag

A merchant-day with no payers is a real occurrence, not an error. **It scores
``EMPTY_BAG_SCORE = 0.0``**, which is the empty-product identity of noisy-OR
(``1 - prod over nothing = 1 - 1 = 0``) and is used for LSE too so the two poolings stay
comparable on the same bags. Semantically: no payer attempted anything, so there is no
payer-level evidence of anything.

The honest caveat is that a merchant *going silent* can itself be a bust-out signal, and
this rung is structurally blind to it - a zero score for a zero-payer day is the correct
answer to the question this rung asks and the wrong answer to the question a volume
feature asks. Silence is a T1 register feature (``v_gmv_z`` collapsing), it is already in
Rungs 2-4, and it is not the bag's job.

## Prime Directive 4 - the servability question, which this rung inherits and does not fix

``features/capsules.py`` records that it is **not** bounded. Eleven of its thirteen vector
columns are functions of one (merchant, day, payer) group and are incremental trivially;
``payer_is_new`` and ``device_shared_payers`` need per-merchant and per-device sets that
grow with distinct payers and devices. T-122 already cut exactly those two quantities from
the T2 register (``g_payer_hhi``, ``g_device_reuse_rate``) as too large for NFR-04's 4 KB.
T-0119 resolved that capsules are not register features and are not measured against
NFR-04, and explicitly handed the servability question to this ticket.

**The pooling does not make it worse, and it does not make it better.** Pooling is
``O(instances)`` time and ``O(bags)`` space in one pass over rows already ordered by bag;
it holds no per-payer state between days and adds no set that grows. What Rung 5 changes
is the *volume*: Rungs 2-4 score one row per merchant-day, Rung 5 scores one row per
(merchant, day, payer), so the per-merchant-day scoring cost is multiplied by the mean bag
size and the NFR-02 10 ms budget is spent per bag rather than per row. That is a real cost
and it is measurable, but it is not the unbounded part.

The unbounded part is upstream and is unchanged: two of the thirteen instance features
cannot be served from a bounded state object today. The upgrade path in the accessor's
docstring (per-merchant HLL for payer novelty, bounded LRU of hot devices for reuse)
**changes the numbers**, so adopting it is a decision to take with a re-run, not a patch.
Stated, not smuggled: as of this ticket, **Rung 5 is not servable under NFR-04**, and a
Rung 5 that wins on PR-AUC wins subject to that, which Prime Directive 5's "AND meets the
compute NFRs" clause makes an adoption question and not a footnote.

Prime Directive 3: this module never names a radioactive field. It receives ``bag_y`` as a
plain array, assembled on the eval side.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from rakshak.eval.metrics import pr_auc
from rakshak.features.capsules import CAPSULE_VECTOR_COLUMNS
from rakshak.models.rung2_lgbm import DEFAULT_PARAMS, HParams, TrainedRung
from rakshak.models.rung2_lgbm import train as _train_lgbm

__all__ = [
    "DEFAULT_TAU_GRID",
    "EMPTY_BAG_SCORE",
    "MAX_INSTANCE_WEIGHT",
    "MIN_INSTANCE_WEIGHT",
    "TRAIN_TAU",
    "Pooling",
    "TrainedMIL",
    "bag_offsets",
    "feature_columns",
    "fit_tau",
    "pool",
    "train",
]

Pooling = Literal["lse", "noisy_or"]

#: A merchant-day with no payers. The noisy-OR empty-product identity; see the module
#: docstring. Documented rather than raised, because a zero-payer day is not an error.
EMPTY_BAG_SCORE: Final = 0.0

#: Selected on validation by :func:`fit_tau`. Contains both exact endpoints, so
#: "``tau -> 0`` is mean-pooling" and "``tau -> inf`` is max-pooling" are reachable
#: identities rather than limits the grid only approaches.
DEFAULT_TAU_GRID: Final[tuple[float, ...]] = (
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    100.0,
    float("inf"),
)

#: Sharpness of the pass-2 instance responsibilities. Declared, not tuned: pass 2 has to
#: pick *some* tau before validation has spoken, and picking it from the validation result
#: would make the fitted tau a function of a fit that already used it.
TRAIN_TAU: Final = 5.0

#: Floor and cap on a pass-2 instance weight. The floor exists because a bag whose top
#: instance was wrong in pass 1 must not have its other instances silenced permanently;
#: the cap exists for the reason Rung 4's does - one instance carrying the weight of
#: twenty others is an asymmetry, one carrying the weight of five hundred is a model of
#: five hundred rows.
MIN_INSTANCE_WEIGHT: Final = 0.05
MAX_INSTANCE_WEIGHT: Final = 20.0


def feature_columns() -> tuple[str, ...]:
    """The instance feature contract: the capsule vector, in ``CAPSULE_SCHEMA`` order.

    The first four capsule columns are keys (merchant, date, payer, last event time) and
    are not features. Taken from ``features.capsules`` rather than re-listed, because a
    column order that is written down twice is a column order that drifts (09-interfaces
    §9).
    """
    return CAPSULE_VECTOR_COLUMNS


def bag_offsets(
    bag_index: np.ndarray, n_bags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(starts, labels, counts)`` for the contiguous runs of ``bag_index``.

    ``bag_index[i]`` is the ordinal of the bag instance ``i`` belongs to, and instances
    **must be grouped**: ``bag_index`` non-decreasing. That is not a convenience - it is
    what ``capsules_as_of`` already returns, since it sorts by
    ``(merchant_id, event_date, payer_id)``, and it is what lets every segmented reduction
    here be a ``ufunc.reduceat`` rather than a scattered ``ufunc.at``.

    It is validated rather than assumed. Silently pooling an unsorted bag index would mix
    two merchants into one bag and produce a plausible number, which is the failure mode
    this repo exists to avoid.

    Bags in ``range(n_bags)`` that never appear are **empty bags**; they have no entry in
    ``labels`` and :func:`pool` gives them :data:`EMPTY_BAG_SCORE`.
    """
    idx = np.asarray(bag_index)
    if idx.ndim != 1:
        raise ValueError(f"bag_index must be 1-d; got shape {idx.shape}")
    if n_bags < 0:
        raise ValueError(f"n_bags must be >= 0; got {n_bags}")
    empty = np.zeros(0, dtype=np.intp)
    if idx.size == 0:
        return empty, empty, empty
    if not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(f"bag_index must be an integer array; got dtype {idx.dtype}")
    if int(idx.min()) < 0 or int(idx.max()) >= n_bags:
        raise ValueError(
            f"bag_index values must lie in [0, {n_bags}); got "
            f"[{int(idx.min())}, {int(idx.max())}]"
        )
    breaks = np.diff(idx)
    if bool(np.any(breaks < 0)):
        raise ValueError(
            "bag_index must be non-decreasing so each bag is one contiguous run of rows. "
            "capsules_as_of already returns rows sorted by (merchant_id, event_date, "
            "payer_id); if this fires, the instance matrix was reordered after it was "
            "built, and pooling it would silently merge two bags into one."
        )
    starts = np.concatenate(([0], np.flatnonzero(breaks) + 1)).astype(np.intp)
    counts = np.diff(np.append(starts, idx.size)).astype(np.intp)
    return starts, idx[starts].astype(np.intp), counts


def _segment_max(v: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-segment maximum, broadcast back to instance length."""
    return np.repeat(np.maximum.reduceat(v, starts), counts)


def _segment_lse_mean(
    v: np.ndarray, starts: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """``log( (1/n) sum_i exp(v_i) )`` per contiguous segment, in the ``expm1``/``log1p``
    form rather than the textbook max-shifted one.

    The textbook form is ``max(v) + log(sum(exp(v - max(v))))``, and it is stable at large
    ``v`` and **wrong at small ``v``** for what this rung needs. Rung 5 divides the result
    by ``tau`` afterwards, so as ``tau -> 0`` the quantity being divided is a difference of
    two numbers of order ``log(n)`` whose difference is of order ``tau``. Double precision
    keeps about sixteen digits of ``1.386``, so a ``tau`` of 1e-8 leaves nothing: the first
    version of this function returned ``0.0`` for a bag whose mean was ``2.5e-12``, and the
    unit test caught it.

    Writing the shifted sum as ``1 + (1/n) sum_i expm1(v_i - max)`` fixes it, because
    ``expm1`` is exact for small arguments and ``log1p`` is exact for small ones: the
    ``log(n)`` that used to be added and then subtracted is never formed at all. The
    max-shift is kept, so ``exp`` still only ever sees non-positive arguments and a
    ``tau`` of 1e6 underflows harmlessly instead of overflowing to ``inf``.

    The mean element (``expm1(0) = 0``) is always in the sum, so the argument to ``log1p``
    is strictly greater than ``-1`` and the result is always finite.
    """
    seg_max: np.ndarray = np.maximum.reduceat(v, starts)
    excess = np.expm1(v - np.repeat(seg_max, counts))
    mean_excess = np.add.reduceat(excess, starts) / counts
    return np.asarray(seg_max + np.log1p(mean_excess), dtype=np.float64)


def pool(
    p: np.ndarray,
    bag_index: np.ndarray,
    n_bags: int,
    *,
    tau: float = 1.0,
    kind: Pooling = "lse",
) -> np.ndarray:
    """Pool instance probabilities into one score per bag. Returns ``(n_bags,)``.

    ``tau == 0.0`` is exactly the arithmetic mean and ``tau == inf`` is exactly the
    maximum; in between it is the stable LSE. ``kind="noisy_or"`` ignores ``tau``, because
    noisy-OR has no such parameter - that is the reason it is a comparator and not the
    family (module docstring).

    Empty bags get :data:`EMPTY_BAG_SCORE`.
    """
    prob = np.asarray(p, dtype=np.float64)
    if prob.ndim != 1:
        raise ValueError(f"p must be 1-d; got shape {prob.shape}")
    if prob.size != np.asarray(bag_index).size:
        raise ValueError(
            f"p has {prob.size} instances and bag_index has {np.asarray(bag_index).size}"
        )
    if prob.size and (float(prob.min()) < 0.0 or float(prob.max()) > 1.0):
        raise ValueError(
            "pooling is over probabilities, not logits: both forms assume p in [0, 1] and "
            f"noisy-OR is undefined outside it. Got [{prob.min()}, {prob.max()}]."
        )
    if tau < 0.0:
        raise ValueError(
            f"tau must be >= 0; got {tau}. Negative tau is min-pooling, which is outside "
            "the mean-to-max family this rung claims to fit."
        )

    out = np.full(n_bags, EMPTY_BAG_SCORE, dtype=np.float64)
    starts, labels, counts = bag_offsets(bag_index, n_bags)
    if starts.size == 0:
        return out

    if kind == "noisy_or":
        # p == 1.0 gives log1p(-1) = -inf, which sums to -inf and expm1's to exactly 1.0.
        # That is the right answer, so the warning is silenced and the input is not
        # clipped: clipping would move a certainty to an almost-certainty for cosmetics.
        with np.errstate(divide="ignore"):
            log_q = np.log1p(-prob)
        out[labels] = -np.expm1(np.add.reduceat(log_q, starts))
    elif kind == "lse":
        if tau == 0.0:
            out[labels] = np.add.reduceat(prob, starts) / counts
        elif np.isinf(tau):
            out[labels] = np.maximum.reduceat(prob, starts)
        else:
            out[labels] = _segment_lse_mean(tau * prob, starts, counts) / tau
    else:
        raise ValueError(f"unknown pooling {kind!r}; expected 'lse' or 'noisy_or'")
    return out


def _responsibilities(
    p: np.ndarray,
    bag_index: np.ndarray,
    bag_y: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    """Pass-2 instance weights: how much of its bag's positive label each instance owns.

    This is the gradient of the pooled loss with respect to the instance scores, up to a
    per-bag constant: for LSE pooling, ``d s(bag) / d p_i`` is the softmax of
    ``tau * p`` within the bag. So the two limits stay coherent through *training* and not
    only through scoring - at ``tau = 0`` every instance is equally responsible and pass 2
    reproduces pass 1 exactly, and at ``tau = inf`` one witness per bag carries the label,
    which is classic MI witness selection.

    Negative bags keep weight 1. Every instance in a negative bag is genuinely negative
    under the standard MI assumption, so there is nothing to reweight and reweighting
    would only shrink the negative class.
    """
    weight = np.ones(p.size, dtype=np.float64)
    if tau == 0.0:
        return weight

    rep_counts = np.repeat(counts, counts).astype(np.float64)
    if np.isinf(tau):
        top = (p >= _segment_max(p, starts, counts)).astype(np.float64)
        # Ties share the bag equally rather than the first row taking it, so the weights
        # do not depend on row order within a bag.
        share = top * rep_counts / np.repeat(np.add.reduceat(top, starts), counts)
    else:
        # Softmax within the bag, max-shifted: exp() sees only non-positive arguments, so
        # a large tau underflows the losers to 0 instead of overflowing the winner to inf.
        v = tau * p
        weights = np.exp(v - _segment_max(v, starts, counts))
        share = weights * rep_counts / np.repeat(np.add.reduceat(weights, starts), counts)

    # ``share`` has mean 1 within each bag, and so does the floored form, so pass 2 does
    # not quietly change the effective positive count that pass 1 fitted against.
    scaled = np.clip(
        MIN_INSTANCE_WEIGHT + (1.0 - MIN_INSTANCE_WEIGHT) * share,
        MIN_INSTANCE_WEIGHT,
        MAX_INSTANCE_WEIGHT,
    )
    in_positive_bag = bag_y[bag_index] == 1
    weight[in_positive_bag] = scaled[in_positive_bag]
    return weight


@dataclass(frozen=True, slots=True)
class TrainedMIL:
    """A fitted instance model plus the pooling it is scored through.

    Satisfies ``explain.registry.Scorer`` - ``predict(x, columns)`` - so it plugs into the
    existing scoring contract with no new seam (SPEC #51 §"seams", item 1). ``bag_index``
    is keyword-only and optional precisely so that signature holds: without it every row
    is its own singleton bag, and pooling a singleton is the **exact identity** for both
    forms (``lse`` of one value is that value for any tau; ``1 - (1 - p)`` is ``p``), so
    the fallback is a real answer rather than a degraded one.
    """

    instance: TrainedRung
    pooling: Pooling
    tau: float
    n_train_bags: int
    n_train_positive_bags: int
    passes: int
    rung: int = 5

    @property
    def columns(self) -> tuple[str, ...]:
        return self.instance.columns

    @property
    def params(self) -> HParams:
        return self.instance.params

    def instance_probabilities(self, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
        """One probability per capsule. The pre-pooling quantity, exposed because the
        instance ranking is the only place a "which payer" answer can come from."""
        return self.instance.predict(x, columns)

    def predict(
        self,
        x: np.ndarray,
        columns: tuple[str, ...],
        *,
        bag_index: np.ndarray | None = None,
        n_bags: int | None = None,
    ) -> np.ndarray:
        """Bag scores. ``(n_bags,)`` with ``bag_index``, else one score per row.

        ``n_bags`` must be given whenever the panel has trailing merchant-days with no
        payers, since those bags appear nowhere in ``bag_index`` and cannot be inferred
        from it. Defaulting to ``max(bag_index) + 1`` and silently dropping them would
        misalign the score vector against the label vector by an amount nobody would see.
        """
        p = self.instance_probabilities(x, columns)
        if bag_index is None:
            return p
        idx = np.asarray(bag_index)
        total = int(idx.max()) + 1 if n_bags is None and idx.size else (n_bags or 0)
        return pool(p, idx, total, tau=self.tau, kind=self.pooling)

    def with_pooling(self, pooling: Pooling, tau: float) -> TrainedMIL:
        """The same fitted instance model, scored through a different pooling."""
        return replace(self, pooling=pooling, tau=tau)

    def save(self, path: Path) -> Path:
        return self.instance.save(path)

    def size_mb(self, path: Path) -> float:
        return self.instance.size_mb(path)

    def summary(self) -> dict[str, Any]:
        """The results-table row. ``tau`` is in it because the fitted tau is a *result*
        (ticket #54): it says where this population sits on the
        any-instance-to-all-instances axis, and a tau near 0 is the finding that MIL adds
        nothing over the mean the T1 register was already taking."""
        return {
            "rung": self.rung,
            "pooling": self.pooling,
            "tau": self.tau,
            "passes": self.passes,
            "n_train_bags": self.n_train_bags,
            "n_train_positive_bags": self.n_train_positive_bags,
            "n_train_instances": self.instance.n_train_rows,
            "n_instance_columns": len(self.columns),
            "train_seconds": self.instance.train_seconds,
        }


def train(
    x: np.ndarray,
    bag_index: np.ndarray,
    bag_y: np.ndarray,
    columns: tuple[str, ...],
    *,
    pooling: Pooling = "lse",
    tau: float = TRAIN_TAU,
    passes: int = 2,
    params: HParams = DEFAULT_PARAMS,
    merchant_id: np.ndarray | None = None,
) -> TrainedMIL:
    """Fit the instance model on training bags only. **No validation data enters here.**

    Same guarantee, and for the same reason, as :func:`rakshak.models.rung2_lgbm.train`.
    ``tau`` is *selected* afterwards by :func:`fit_tau` on the validation split; the
    ``tau`` argument here only sets the sharpness of the pass-2 responsibilities.

    Two passes, which is the ticket's maximum and not a budget that was cut short:

    1. **Bag-label propagation.** Every instance in a positive bag is labelled positive.
       This is the standard MI initialisation and it is wrong on purpose - most payers of
       a laundering merchant are ordinary payers - which is exactly what pass 2 corrects.
    2. **Pooled-loss reweighting.** Refit with each positive-bag instance weighted by its
       responsibility under the pooling (:func:`_responsibilities`). The labels do not
       move; the weights do. Reweighting rather than relabelling keeps the objective a
       proper scoring rule, so the instance output is still a probability and the pooled
       output is still in [0, 1] - the same argument that keeps ``scale_pos_weight`` out
       of Rung 2.

    ``x`` rows must be **grouped by bag** and ``bag_index`` non-decreasing; see
    :func:`bag_offsets`. ``bag_y`` is one label per bag, so ``len(bag_y)`` is ``n_bags``
    and trailing empty bags are represented without needing to appear in ``bag_index``.
    """
    if passes < 1:
        raise ValueError(f"passes must be >= 1; got {passes}")
    features = np.asarray(x, dtype=np.float64)
    labels = np.asarray(bag_y)
    if features.ndim != 2:
        raise ValueError(f"x must be 2-d (instances x features); got shape {features.shape}")
    if features.shape[0] != np.asarray(bag_index).size:
        raise ValueError(
            f"x has {features.shape[0]} instances and bag_index has "
            f"{np.asarray(bag_index).size}"
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("bag_y must be 0/1 bag labels, one per bag")

    n_bags = int(labels.size)
    starts, _, counts = bag_offsets(bag_index, n_bags)
    idx = np.asarray(bag_index)
    y_instance = labels[idx].astype(np.int8)

    model = _train_lgbm(
        features, y_instance, columns, rung=5, params=params, merchant_id=merchant_id
    )
    for _ in range(passes - 1):
        weight = _responsibilities(
            model.predict(features, columns), idx, labels, starts, counts, tau=tau
        )
        model = _train_lgbm(
            features,
            y_instance,
            columns,
            rung=5,
            params=params,
            merchant_id=merchant_id,
            weight=weight,
        )

    return TrainedMIL(
        instance=model,
        pooling=pooling,
        tau=tau,
        n_train_bags=n_bags,
        n_train_positive_bags=int((labels == 1).sum()),
        passes=passes,
    )


def fit_tau(
    model: TrainedMIL,
    x: np.ndarray,
    bag_index: np.ndarray,
    bag_y: np.ndarray,
    *,
    grid: tuple[float, ...] = DEFAULT_TAU_GRID,
    include_noisy_or: bool = True,
) -> tuple[TrainedMIL, list[dict[str, Any]]]:
    """Select the pooling on **validation** bags. Returns the retuned model and the table.

    The table is returned, not logged, because the ticket makes the fitted tau a reported
    result and the losing rows are half of what makes the winning row mean anything - a
    tau of 0.5 chosen over a flat grid says something a tau of 0.5 alone does not.

    Ties go to the earliest row, and the grid is ascending with noisy-OR appended last, so
    a tie prefers the *smaller* tau and prefers LSE over noisy-OR. Both preferences are
    towards the simpler claim: if mean-pooling ties max-pooling, the honest report is that
    the pooling did not matter.

    ``ponytail: tau is selected over the pooling of one fitted instance model, not by
    refitting the instance model at every tau on the grid.`` Nine refits is nine times the
    ticket's declared two-pass budget, and the instance model's job - rank capsules - is
    only weakly a function of the pooling sharpness. Upgrade path if it matters: refit per
    tau, which is worth doing only if the selected tau lands on a grid **edge**, since
    that is the case where the fitted instance model and the chosen pooling actually
    disagree about what the bag is.
    """
    labels = np.asarray(bag_y)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("bag_y must be 0/1 bag labels, one per bag")
    n_bags = int(labels.size)
    p = model.instance_probabilities(x, model.columns)

    rows: list[dict[str, Any]] = [
        {
            "pooling": "lse",
            "tau": float(tau),
            "pr_auc": pr_auc(labels, pool(p, bag_index, n_bags, tau=tau, kind="lse")),
        }
        for tau in grid
    ]
    if include_noisy_or:
        rows.append(
            {
                "pooling": "noisy_or",
                "tau": None,
                "pr_auc": pr_auc(labels, pool(p, bag_index, n_bags, kind="noisy_or")),
            }
        )

    scored = [row for row in rows if not np.isnan(row["pr_auc"])]
    if not scored:
        raise ValueError(
            "every pooling scored nan, which pr_auc returns when a split holds one class "
            "only. tau cannot be fitted against a validation split with no positive bags; "
            "this is the delayed-label regime (FR-020) showing up, not a bug in the grid."
        )
    best = max(scored, key=lambda row: float(row["pr_auc"]))
    tuned = model.with_pooling(
        "noisy_or" if best["pooling"] == "noisy_or" else "lse",
        float("nan") if best["tau"] is None else float(best["tau"]),
    )
    return tuned, rows
