"""Rung 9 — Page/CUSUM on the within-day cross-sectional rank of the incumbent score.

Named and specified in ``docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md`` §4.1 and in
``project-context/13a-survey-quickest-change-detection.md`` §4, both committed before this
file existed. It is the one new *scoring* rung of cycle 4, and it is declared **cuttable
first** in the pre-registration's cut order: the cycle's results do not depend on it.

**Why this and not a better classifier.** Every rung below optimises a classification score
and inherits latency as a side effect. This one's objective *is* detection delay: the Page
recursion is exactly optimal for Lorden's worst-case delay criterion at every false-alarm
level (Moustakides, *Ann. Statist.* 14(4):1379-1387, 1986). Cycle 3 could not have rewarded
that — `detection_rate_d7` was 0.000 for an oracle too (``LIMITATIONS.md`` §8.7a) — which is
why a latency method had to wait for the geometry fix to be worth building.

**Three properties the rank transform buys, none of which cost anything.**

1. *Distribution-free.* Under the no-change null a within-day rank is uniform whatever the
   count distribution is. The measured Fano factor of 12.25-12.37 would wreck any Poisson or
   Gaussian likelihood ratio; it is simply irrelevant here.
2. *A strictly stronger confounder guard than the cohort residual.* A festival spike, a
   gateway outage or a fee change is a monotone shock applied to the whole panel on one day.
   A rank is invariant to **any** monotone shock, where the residual subtraction only cancels
   an additive one.
3. *The multi-stream aggregation rule is already built.* The literature's answer to "which of
   N streams changed under one global false-alarm budget" is top-r thresholding of local
   CUSUM statistics (Mei, *Biometrika* 97(2):419-433, 2010). The capacity layer already **is**
   top-r at K, so no threshold is ever calibrated and no ARL is ever tuned.

**The cap is load-bearing, not a safety rail.** Without ``c_max``, a merchant that drifted 200
days ago pins the top-K forever and the alert set becomes static — which is precisely the
`volume_rank` pathology (week-over-week alert Jaccard 1.000) this rung exists to defeat. The
pre-registered anti-degeneracy gate would then catch it, but it should not have to.

State cost is one ``float64`` per merchant — **8 bytes** against a 4 KB budget — and the work
is one ``argsort`` per cohort per day plus O(1) arithmetic per merchant. No dependency beyond
what is pinned, no autograd, no GPU, and nothing inside ``eval_module_sha256`` is touched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from rakshak.features.cohort import assign_cohorts
from rakshak.schemas import MerchantProfile

__all__ = [
    "C_MAX",
    "K_REFERENCE",
    "NORMAL_SCORE_CLIP",
    "RankCusum",
    "accumulator",
    "cohort_labels",
    "cross_sectional_normal_scores",
    "page_recursion",
    "run_length",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

#: Reference value: half the standardised shift to detect fastest. 0.25 targets a 0.5-sigma
#: shift. Survey §4 names it the only knob that matters and permits a grid of
#: {0.10, 0.25, 0.50} **on the training fold only**. The default is the pre-registered one.
K_REFERENCE = 0.25

#: Accumulator ceiling. See the module docstring — this is what stops the rung degenerating
#: into a static watchlist, which is the failure mode it was chosen to avoid.
C_MAX = 20.0

#: A single day must not dominate the accumulator. At |4| the normal score is a 1-in-31,574
#: event under the null, so clipping costs nothing real and bounds one day's contribution.
NORMAL_SCORE_CLIP = 4.0


def cross_sectional_normal_scores(
    score: F64, day: I64, cohort: npt.NDArray[np.str_]
) -> F64:
    """Per row, the normal score of its within-day, within-cohort rank of ``score``.

    ``u = (rank - 0.5) / n`` is uniform on (0,1) under no change, so ``Phi^-1(u)`` has mean 0
    and variance ~1 whatever the underlying distribution is — the property the whole method
    rests on.

    Ties are given their average rank. That matters more here than it usually does: a
    gateway outage drives a whole cohort's score to the same floor on the same day, and
    breaking those ties arbitrarily would manufacture a spread of normal scores out of an
    event in which nothing distinguished one merchant from another.

    A cohort of one on a given day yields ``u = 0.5`` and a normal score of exactly 0, which
    is the correct answer: a merchant compared only against itself provides no evidence of
    change, and contributing 0 leaves the accumulator to decay by ``k``.
    """
    score = np.asarray(score, dtype=np.float64)
    day = np.asarray(day, dtype=np.int64)
    cohort = np.asarray(cohort)
    if not (score.shape == day.shape == cohort.shape):
        raise ValueError(
            f"score {score.shape}, day {day.shape} and cohort {cohort.shape} must be "
            "aligned row-for-row; a mismatch ranks one merchant against another day's peers."
        )

    out = np.zeros_like(score)
    # One pass per (day, cohort) block. Grouping by a structured sort rather than a dict of
    # lists keeps this O(n log n) and allocation-light over a multi-million-row panel.
    order = np.lexsort((cohort, day))
    keys = list(zip(day[order], cohort[order], strict=True))
    if not keys:
        return out
    boundaries = [0]
    for i in range(1, len(keys)):
        if keys[i] != keys[i - 1]:
            boundaries.append(i)
    boundaries.append(len(keys))

    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
        idx = order[lo:hi]
        n = idx.size
        vals = score[idx]
        # Average ranks for ties, 1-based.
        tmp = np.argsort(vals, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[tmp] = np.arange(1, n + 1, dtype=np.float64)
        uniq, inverse, counts = np.unique(vals, return_inverse=True, return_counts=True)
        if uniq.size != n:  # ties present
            sums = np.zeros(uniq.size, dtype=np.float64)
            np.add.at(sums, inverse, ranks)
            ranks = (sums / counts)[inverse]
        out[idx] = np.clip(
            norm.ppf((ranks - 0.5) / n), -NORMAL_SCORE_CLIP, NORMAL_SCORE_CLIP
        )
    return out


def page_recursion(
    x: F64,
    day: I64,
    merchant: npt.NDArray[np.str_],
    *,
    k: float = K_REFERENCE,
    c_max: float = C_MAX,
) -> F64:
    """``C[m,t] = min(c_max, max(0, C[m,t-1] + x[m,t] - k))``, per merchant, in day order.

    Returns the accumulator aligned with the input rows. Rows are sorted internally by
    ``(merchant, day)`` so the caller's row order is irrelevant — a recursion that silently
    depended on input ordering would be a bug that only appears when the panel is reshuffled,
    which is exactly the kind that survives a test suite.
    """
    x = np.asarray(x, dtype=np.float64)
    day = np.asarray(day, dtype=np.int64)
    merchant = np.asarray(merchant)
    if not (x.shape == day.shape == merchant.shape):
        raise ValueError("x, day and merchant must be aligned row-for-row")
    if k < 0.0:
        raise ValueError(f"reference value k must be >= 0; got {k!r}")
    if c_max <= 0.0:
        raise ValueError(
            f"c_max must be > 0; got {c_max!r}. Removing the cap lets a merchant that "
            "drifted long ago pin the top-K forever, which is the static-watchlist failure "
            "this rung exists to avoid."
        )

    out = np.zeros_like(x)
    order = np.lexsort((day, merchant))
    m_sorted = merchant[order]
    running = 0.0
    for pos, i in enumerate(order):
        if pos == 0 or m_sorted[pos] != m_sorted[pos - 1]:
            running = 0.0  # a new merchant starts at 0, after its frozen warmup
        running = min(c_max, max(0.0, running + x[i] - k))
        out[i] = running
    return out


@dataclass(frozen=True, slots=True)
class RankCusum:
    """The fitted blend that turns ``(incumbent score, accumulator)`` into a probability.

    ``Decision.score`` must be a calibrated probability in [0,1] and a raw CUSUM statistic is
    not one. Three parameters against ~234 trainable positives is inside the label budget
    with room to spare, and keeping ``logit(s)`` in the blend preserves the *level*
    information the savings metric monetises alongside the *change* information that is the
    point of the rung.

    Fitted on the training fold only. It never sees a validation label, and the CUSUM itself
    consumes no labels at all.
    """

    model: LogisticRegression
    k: float
    c_max: float

    @staticmethod
    def _design(incumbent: F64, accumulator: F64) -> F64:
        # Clipped so logit() is finite at a score of exactly 0 or 1, which LightGBM does emit.
        p = np.clip(np.asarray(incumbent, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(p / (1.0 - p)), np.asarray(accumulator)])

    @classmethod
    def fit(
        cls,
        *,
        incumbent: F64,
        accumulator: F64,
        y: npt.NDArray[np.int_],
        k: float = K_REFERENCE,
        c_max: float = C_MAX,
    ) -> RankCusum:
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(cls._design(incumbent, accumulator), np.asarray(y).astype(int))
        return cls(model=model, k=k, c_max=c_max)

    def predict(self, incumbent: F64, accumulator: F64) -> F64:
        proba = self.model.predict_proba(self._design(incumbent, accumulator))[:, 1]
        return np.asarray(proba, dtype=np.float64)


def run_length(accumulator: F64, day: I64, merchant: npt.NDArray[np.str_]) -> I64:
    """Consecutive days since the accumulator was last exactly 0, per merchant.

    A reason code no static ranking can produce: "this merchant has been drifting for 11
    consecutive days" is a statement about time, and `volume_rank` — whose week-over-week
    alert Jaccard is 1.000 — has no time in it at all.
    """
    acc = np.asarray(accumulator, dtype=np.float64)
    order = np.lexsort((np.asarray(day), np.asarray(merchant)))
    m_sorted = np.asarray(merchant)[order]
    out = np.zeros(acc.size, dtype=np.int64)
    run = 0
    for pos, i in enumerate(order):
        if pos == 0 or m_sorted[pos] != m_sorted[pos - 1]:
            run = 0
        run = 0 if acc[i] <= 0.0 else run + 1
        out[i] = run
    return out


def cohort_labels(
    profiles: Mapping[str, MerchantProfile], merchant_id: npt.NDArray[np.str_]
) -> npt.NDArray[np.str_]:
    """The cohort each row is ranked inside, via the real ``assign_cohorts``.

    Reuses the existing ``(mcc_group, gmv_decile, vintage_bucket)`` key and its 30-member
    backoff chain rather than defining a second notion of "peer group". A rung that ranked
    against a cohort nothing else in the project uses would be measuring its own definition.
    """
    assignment = assign_cohorts(profiles)
    label = assignment.label
    return np.array([label.get(m, "global") for m in merchant_id])


def accumulator(
    incumbent: F64,
    day: I64,
    merchant: npt.NDArray[np.str_],
    cohort: npt.NDArray[np.str_],
    *,
    k: float = K_REFERENCE,
    c_max: float = C_MAX,
) -> F64:
    """Steps 1-3 in one call: cross-sectional normal scores, then the Page recursion.

    **The accumulator cold-starts at 0 on the first day each merchant appears**, and on the
    validation split that is day 240 for every merchant, because the panel carries a row
    only where the merchant's fold matches the day's split (``dataset`` builds it from
    ``fold == day_split``). That is the right setting rather than a limitation: every
    validation merchant is on equal footing, and a merchant whose drift began before the
    window opened has no measurable in-window detection delay anyway — which is exactly the
    point ``LIMITATIONS.md`` §8.7a makes about cycle 3.
    """
    return page_recursion(
        cross_sectional_normal_scores(incumbent, day, cohort),
        day,
        merchant,
        k=k,
        c_max=c_max,
    )
