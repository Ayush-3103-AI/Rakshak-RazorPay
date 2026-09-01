"""Rung 6 — Mondrian conformal risk control over the three-action decision (T-0121).

Rungs 0-4 hand a score to the capacity-constrained selector in :mod:`rakshak.eval.capacity`
and take whatever HOLDs come out. Nobody has ever *bounded* what fraction of those HOLDs
land on merchants that were not drifting. A HOLD freezes settlements, so that fraction is
the number a risk team is actually accountable for, and this rung's whole claim is a
distribution-free upper bound on it.

**The claim, exactly.** For a declared ``alpha`` and every Mondrian stratum ``g``:

    P(action == HOLD | y == 0, stratum == g)  <=  alpha

which is precisely what :func:`rakshak.eval.metrics.false_hold_coverage` measures, per
stratum, with the stratum's negative merchant-days as the denominator. The metric was
declared in ``docs/PRE-REGISTRATION-CYCLE3-2026-08-31.md`` §2 and implemented in T-0118
*before* this module existed; nothing here redefines it, and nothing here softens the
strict ``realised > alpha`` comparison it uses. §5 of that pre-registration commits to a
violation being **reported**, so this module supplies a threshold and stops. It does not
clamp, shrink, or widen anything to make a violation go away.

**The construction is split conformal on a scalar score, in-house.** ``CLAUDE.md``
§Tech Stack permits ``crepes`` or ``mapie`` for this rung; neither is in ``pyproject.toml``
and the ticket says to take split conformal in-house rather than bend a library for what is
one order statistic. Within stratum ``g``, using the calibration negatives' scores
``s_1..s_n`` as nonconformity scores (higher = more anomalous):

    k_g = ceil((n_g + 1) * (1 - alpha))      t_g = s_(k_g)      HOLD permitted iff s > t_g

For a new negative exchangeable with the calibration negatives of ``g``,
``P(s > t_g) = (n_g + 1 - k_g) / (n_g + 1) <= alpha``. If ``k_g > n_g`` — the stratum has
too few calibration negatives to certify an alpha that small — ``t_g = +inf`` and the
stratum simply never HOLDs. That is the honest answer, not a fallback to a pooled
threshold: pooling is exactly the marginal guarantee Mondrian exists to refuse.

**Why the bound survives the capacity layer.** The realised HOLD event is a strict subset
of ``{s > t_g}``: a row must additionally be selected in the day's top-K and pass
:class:`~rakshak.eval.capacity.ActionPolicy`. So the realised rate is bounded by the
conformal rate *a fortiori*, and the wrapper only ever softens HOLD -> REVIEW. It never
promotes, the per-day non-PASS count is untouched, and ``alerts_per_day <= K`` continues to
hold by construction rather than by assertion.

**Strata.** The taxonomy is the cohort key the residual layer already uses —
``(mcc_group, gmv_decile, vintage_bucket)`` with the same 30-member backoff chain to
``mcc_group`` and then ``global`` (:func:`rakshak.features.cohort.assign_cohorts`). Reused
rather than redefined, so a coverage row and a residual are talking about the same set of
merchants. Backed-off cells are ordinary strata here and get their own threshold and their
own coverage row.

**Prime Directive 3.** Nothing here imports or names a ``ground_truth`` field. The cohort
key is built from onboarding facts; calibration reads the 0/1 merchant-day label, which is
the same supervised label every rung trains on, and it reads it *before* any decision is
made. :class:`~rakshak.eval.capacity.DecisionRequest` still carries no labels.

**Prime Directive 5.** Coverage is what this rung *claims*; it is not an adoption margin.
Adoption still requires >=10% relative PR-AUC or >=3 days median TTD over the rung below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rakshak.eval.capacity import DecisionPolicy, DecisionRequest
from rakshak.features.cohort import CohortAssignment
from rakshak.schemas import Action, Split

__all__ = [
    "ConformalHold",
    "MondrianCalibration",
    "calibrate",
    "split_validation",
    "strata_of",
]


def strata_of(merchant_id: np.ndarray, cohorts: CohortAssignment) -> np.ndarray:
    """Mondrian stratum per merchant-day: the merchant's cohort label, backoff level and all."""
    return np.asarray([cohorts.label[m] for m in np.asarray(merchant_id).tolist()])


def split_validation(
    merchant_id: np.ndarray, rng: np.random.Generator, *, fraction: float = 0.5
) -> np.ndarray:
    """Carve a calibration fold out of the validation split, **by merchant**.

    A conformal guarantee calibrated on data it also scores is not a guarantee, so the fold
    has to be held out. It is carved by merchant rather than by day for two reasons. Days
    240-299 are one window and splitting it in time makes the calibration and scoring folds
    differ by drift, which is the one thing conformal prediction has no defence against.
    And a merchant's own days are strongly dependent — the same merchant either side of a
    day-split would leak its level across the fold.

    The cost is stated rather than hidden: the exchangeability being assumed is between
    *merchants* within a stratum, while the metric denominates in merchant-days. Days
    cluster within merchants, so realised coverage has more variance than a binomial on the
    day count would suggest, and a stratum carried by a handful of merchants should be read
    off ``CoverageRow.n_negatives`` with that in mind.

    Returns a boolean mask, ``True`` for the calibration fold.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1; got {fraction!r}")
    merchants = np.unique(np.asarray(merchant_id))
    n_cal = max(1, min(merchants.size - 1, round(fraction * merchants.size)))
    chosen = set(rng.permutation(merchants)[:n_cal].tolist())
    return np.asarray([m in chosen for m in np.asarray(merchant_id).tolist()], dtype=bool)


@dataclass(frozen=True, slots=True)
class MondrianCalibration:
    """One conformal score threshold per Mondrian stratum, at nominal ``alpha``.

    ``threshold[g]`` is the ``ceil((n+1)(1-alpha))``-th smallest calibration-negative score
    in ``g``; ``+inf`` means the stratum had too few calibration negatives to certify this
    alpha and therefore may never HOLD. ``n_calibration[g]`` is kept because a threshold
    from 40 negatives and one from 40,000 are not the same object, and the reader is
    entitled to see which one they are looking at.
    """

    alpha: float
    threshold: dict[str, float]
    n_calibration: dict[str, int]

    def bound(self, stratum: str) -> float:
        """Exact finite-sample exceedance bound ``(n + 1 - k) / (n + 1)`` for ``stratum``.

        Always ``<= alpha``, and usually a little below it — the guarantee is discrete, so
        the achievable level steps in units of ``1 / (n + 1)``. A stratum that can never
        HOLD, seen or unseen, bounds at 0.0.
        """
        n = self.n_calibration.get(stratum, 0)
        if not n or math.isinf(self.threshold.get(stratum, math.inf)):
            return 0.0
        return (n + 1 - math.ceil((n + 1) * (1.0 - self.alpha))) / (n + 1)

    def permits(self, score: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        """Boolean mask: may this row HOLD at all, given its stratum's threshold?

        Strictly greater than the threshold, and unseen strata get ``+inf``. Both are the
        conservative direction — under ties and under novelty respectively — which is the
        only direction a risk control is allowed to round.
        """
        labels = [str(s) for s in np.asarray(stratum).tolist()]
        cut = np.asarray([self.threshold.get(g, math.inf) for g in labels], dtype=np.float64)
        return np.asarray(np.asarray(score, dtype=np.float64) > cut)


def calibrate(
    score: np.ndarray,
    y: np.ndarray,
    stratum: np.ndarray,
    alpha: float,
    *,
    split: Split = "val",
) -> MondrianCalibration:
    """Fit per-stratum conformal thresholds on the **negatives** of the validation split.

    Negatives only, because the quantity being bounded is ``P(HOLD | y == 0)``: the
    conditioning event *is* the calibration population, and including fraud days would
    calibrate a mixture nobody is claiming anything about.

    ``split`` is refused for anything but ``"val"``. This is deliberately stricter than
    :func:`rakshak.eval.lock.require_unlocked_or_refuse`, which lets ``RAKSHAK_UNLOCK=1``
    open the test split for the single scoring run at the end: a threshold fitted on days
    300-364 would make the coverage number a statement about data the rung was tuned on,
    and no environment variable makes that a guarantee. Days 0-239 are refused too — the
    models are fitted there, so their negatives' scores are in-sample and the quantiles
    taken from them are optimistic.
    """
    if split != "val":
        raise ValueError(
            f"conformal calibration reads the validation split only; got {split!r}. Train "
            "scores are in-sample and their quantiles are optimistic; the test split is "
            "opened exactly once, at the end, after every rung is final (Prime Directive 1)."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha is a nominal error rate in (0,1); got {alpha!r}")
    if not len(score) == len(y) == len(stratum):
        raise ValueError("score, y and stratum must be the same length")

    labels = np.asarray([str(s) for s in np.asarray(stratum).tolist()])
    negative = np.asarray(y) == 0
    scores = np.asarray(score, dtype=np.float64)

    threshold: dict[str, float] = {}
    n_calibration: dict[str, int] = {}
    for value in sorted(set(labels.tolist())):
        held = np.sort(scores[(labels == value) & negative])
        n = held.size
        n_calibration[value] = n
        # ceil((n+1)(1-alpha)) can be nudged up an ulp by float error. That direction is
        # the safe one — a larger k is a higher threshold and fewer HOLDs — so it is left
        # alone rather than papered over with a rounding tolerance.
        k = math.ceil((n + 1) * (1.0 - alpha))
        threshold[value] = math.inf if k > n else float(held[k - 1])
    return MondrianCalibration(alpha=alpha, threshold=threshold, n_calibration=n_calibration)


@dataclass(frozen=True, slots=True)
class ConformalHold:
    """A :class:`~rakshak.eval.capacity.DecisionPolicy` that softens uncertified HOLDs.

    Wraps another policy — normally ``DEFAULT_DECISION`` — and rewrites HOLD to REVIEW on
    any row whose score does not clear its stratum's conformal threshold. PASS and REVIEW
    rows come back untouched, so the per-day non-PASS count is identical to the inner
    policy's and the capacity budget K is preserved exactly. The seam's rule is that a
    wrapper may only ever soften; this one softens by exactly one step, on exactly the
    action whose error rate it claims to bound.

    ``stratum`` is a field rather than a ``decide`` argument because
    :class:`~rakshak.eval.capacity.DecisionRequest` is frozen inside ``eval_module_sha256``
    and carries no merchant identity. It must be aligned row-for-row with the request the
    policy will be handed; ``decide`` refuses a length mismatch rather than broadcasting one
    merchant's stratum onto another's row.
    """

    inner: DecisionPolicy
    calibration: MondrianCalibration
    stratum: np.ndarray

    @property
    def name(self) -> str:
        return f"crc({self.inner.name}, alpha={self.calibration.alpha:g})"

    def decide(self, request: DecisionRequest) -> np.ndarray:
        action = np.asarray(self.inner.decide(request))
        if action.size != np.asarray(self.stratum).size:
            raise ValueError(
                f"stratum has {np.asarray(self.stratum).size} rows but the request has "
                f"{action.size}. A stratum array that is not aligned row-for-row with the "
                "request would certify merchants against other merchants' thresholds."
            )
        uncertified = (action == Action.HOLD) & ~self.calibration.permits(
            request.score, self.stratum
        )
        return np.asarray(np.where(uncertified, Action.REVIEW, action))
