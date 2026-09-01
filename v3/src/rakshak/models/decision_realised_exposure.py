"""Price the decision on the exposure a merchant *realised*, not the one it declared.

**This carries no rung number, deliberately.** ``configs/rung_roster.yaml`` already assigns
Rung 8 to ``tpp_hawkes_nb``, and more to the point
``docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md`` §4.2 registers this as a controlled
decision-policy A/B run over the *whole* ladder rather than as a competing rung. Numbering
it would have put two different things under one heading, which is how a results table ends
up comparing two quantities that share a column name. See the dated note at the foot of the
pre-registration recording the rename.

Pre-registered in ``docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md`` §4.2, before this file
existed. The evidence is ``LIMITATIONS.md`` §8.3a and it is worth restating, because this
module is four lines of arithmetic and the reason it exists is the whole of its value.

**This is not a model. It is a correction to what the decision layer is handed.**

``eval/capacity.py::select_actions`` already ranks merchant-days by expected rupees —
``benefit = 0.8 · p · exposure_inr − 250`` in the REVIEW branch. The cost-sensitive
literature's headline prescription, "rank by expected value rather than by probability",
was implemented before cycle 3 was scored. What was wrong was the second factor:

===========================  ====================================================
``exposure_inr`` as supplied  ``p_declared_monthly_gmv`` (``cli.py``), the monthly
                              GMV the merchant **declared at onboarding**
``volume_rank`` ranks on      **observed** captured GMV, an event-stream quantity
``true_loss_amount_inr`` is   ``loss_fraction`` × post-onset **realised** captured GMV
===========================  ====================================================

The generator corrupts the declaration deliberately — ``declaration_error_sigma: 0.55``,
and the config comment says why: the gap between declared and actual *is* the signal
``v_declared_ratio`` exists to read. Measured over the 294 fraud merchants in the cycle-3
ground truth, against realised loss:

============================  ==================  =============================
estimator                     Spearman ρ vs loss  share of loss in the top K=15
============================  ==================  =============================
``declared_monthly_gmv``      +0.533              20.51%
observed pre-window GMV       +0.929              37.83%
perfect foresight             1.000               46.18%
============================  ==================  =============================

So ``volume_rank`` was never a dumb floor beating clever models. It is a ρ = 0.93 exposure
estimator, competing against rungs whose excellent ``p`` was being multiplied by a ρ = 0.53
one. Rung 3 beats it on precision@K (0.869 vs 0.571) *and* recall@K (0.315 vs 0.195) at the
same K and still loses 27% on savings, which no ranking-quality hypothesis can produce.

**Nothing new is computed.** ``v_declared_ratio`` is
``trailing-30d GMV ÷ declared_monthly_gmv`` (``features/tier1.py::DeclaredRatio``), so

    v_declared_ratio × declared_monthly_gmv  ==  trailing-30d realised captured GMV

identically. Both factors are already in the register, already point-in-time, already past
the leakage gate. No new feature, no new dependency, no training, no label, and no edit to
anything inside ``eval_module_sha256``.

**On the seam's "may only ever soften" rule.** The conformal wrapper softens actions the
inner policy has already chosen, because promoting one after selection could breach K. This
wrapper does not touch actions at all: it substitutes an *input* and lets the unmodified
inner policy perform the entire selection, including its own top-K enforcement. K is
therefore preserved by construction rather than by care, and
``tests/unit/test_rung8_exposure.py`` asserts it at the boundary rather than trusting the
argument.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rakshak.eval.capacity import DecisionPolicy, DecisionRequest

__all__ = ["RealisedExposure", "realised_exposure_inr"]

F64 = npt.NDArray[np.float64]


def realised_exposure_inr(declared_monthly_gmv: F64, v_declared_ratio: F64) -> F64:
    """Trailing-30d realised captured GMV, from two already-registered features.

    ``v_declared_ratio`` is defined as ``trailing-30d GMV ÷ declared_monthly_gmv``, so the
    product recovers the numerator exactly. The reconstruction is used rather than a new
    feature because a new feature would need its own dual runner, its own parity proof and
    its own state budget, to arrive at a number the panel already carries.

    ``DeclaredRatio.read`` returns ``0.0`` when ``declared_gmv <= 0``, which would otherwise
    reconstruct an exposure of zero for a merchant that may be transacting perfectly well —
    and an exposure of zero is a merchant the decision layer can never afford to alert on,
    whatever its score. Those rows fall back to the declared figure, which is the estimator
    cycle 3 used throughout, so the fallback is never worse than the incumbent.
    """
    declared = np.asarray(declared_monthly_gmv, dtype=np.float64)
    ratio = np.asarray(v_declared_ratio, dtype=np.float64)
    if declared.shape != ratio.shape:
        raise ValueError(
            f"declared_monthly_gmv has shape {declared.shape} but v_declared_ratio has "
            f"{ratio.shape}; they must be aligned row-for-row or one merchant's trailing "
            "GMV is priced against another's declaration."
        )
    realised = declared * ratio
    usable = np.isfinite(realised) & (realised > 0.0)
    return np.asarray(np.where(usable, realised, declared), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RealisedExposure:
    """Forward to ``inner`` with ``exposure_inr`` replaced by realised trailing GMV.

    ``exposure`` is a field rather than a ``decide`` argument for the same reason the
    conformal wrapper's ``stratum`` is: :class:`~rakshak.eval.capacity.DecisionRequest` is
    frozen inside ``eval_module_sha256`` and cannot grow a parameter. It must be aligned
    row-for-row with the request; ``decide`` refuses a length mismatch rather than
    broadcasting one merchant's exposure onto another's row.
    """

    inner: DecisionPolicy
    exposure: F64

    @property
    def name(self) -> str:
        return f"realised_exposure({self.inner.name})"

    def decide(self, request: DecisionRequest) -> np.ndarray:
        exposure = np.asarray(self.exposure, dtype=np.float64)
        if exposure.size != np.asarray(request.score).size:
            raise ValueError(
                f"exposure has {exposure.size} rows but the request has "
                f"{np.asarray(request.score).size}. A misaligned exposure vector would "
                "price each merchant's decision on another merchant's trailing GMV, and "
                "the resulting savings would measure the misalignment."
            )
        if np.any(exposure < 0.0):
            raise ValueError(
                "exposure_inr must be non-negative; a negative exposure inverts the sign "
                "of the expected-loss term and makes REVIEW look profitable on merchants "
                "with nothing at stake."
            )
        return np.asarray(
            self.inner.decide(dataclasses.replace(request, exposure_inr=exposure))
        )
