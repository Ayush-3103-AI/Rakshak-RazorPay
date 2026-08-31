"""Risk typologies R1-R9: who turns, when, and how their behaviour bends after onset.

08-generator-v2-spec.md §3. Fraud is never one undifferentiated class in this project —
per-typology recall is a required output precisely so that a rung which wins on the
average while missing R2 entirely cannot hide behind that average.

The module produces three things and nothing else:

- **who** — a sparse assignment of typologies to merchants at the configured prevalence;
- **when** — ``drift_onset_at`` as a simulation day, and a ramp length;
- **how much** — a ``(n_merchants, n_days)`` ramp *progress* in [0,1] and the intensity
  multiplier built from it, plus per-merchant mark deltas the engine scales by progress.

Progress is the single mechanism. Every mark delta is ``base + delta * progress``, so a
typology cannot accidentally jump: R6 jumps because its ramp is one day long, and R2
crawls because its ramp is sixty to ninety. That is the difference between them, stated
once, in the config, rather than as two code paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rakshak.generator.config import TypologyParams
from rakshak.schemas import TypologyId

__all__ = [
    "NO_TYPOLOGY",
    "TypologyAssignment",
    "assign_typologies",
    "intensity_multiplier",
    "per_merchant_field",
    "ramp_progress",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B1 = npt.NDArray[np.bool_]

#: Sentinel in the per-merchant typology index array. 09-interfaces.md forbids sentinels
#: in the *contract* dataclasses (nullable is explicit ``X | None``); this is an internal
#: numpy index array, where a parallel mask array would be a second thing to keep in
#: sync. The engine converts it to ``None`` at the ``GroundTruth`` boundary.
NO_TYPOLOGY = -1


@dataclass(frozen=True, slots=True)
class TypologyAssignment:
    """Per-merchant fraud assignment. Arrays are all length ``n_merchants``.

    ``typology_index`` is ``NO_TYPOLOGY`` for the ~98.5% of merchants that never turn;
    ``onset_day`` and ``ramp_days`` are meaningless (and unused) for those.
    """

    typology_index: I64
    onset_day: I64
    ramp_days: I64

    @property
    def is_fraud(self) -> B1:
        return np.asarray(self.typology_index >= 0)

    @property
    def n_fraud(self) -> int:
        return int(self.is_fraud.sum())


def assign_typologies(
    rng: np.random.Generator,
    n_merchants: int,
    prevalence: float,
    typologies: dict[TypologyId, TypologyParams],
) -> TypologyAssignment:
    """Choose which merchants turn, into what, and when.

    The positive count is ``round(n * prevalence)`` exactly rather than a binomial draw.
    At 1.47% of 10,000 that is 147 merchants; a binomial would put the realised
    prevalence somewhere in 1.2%-1.7% run to run, and since prevalence is the number v1
    got wrong, it is the last thing that should be left to sampling noise.

    ``prevalence == 0`` is legal and is exactly what gate G5 runs.
    """
    order = list(TypologyId)
    typology_index = np.full(n_merchants, NO_TYPOLOGY, dtype=np.int64)
    onset_day = np.full(n_merchants, NO_TYPOLOGY, dtype=np.int64)
    ramp_days = np.ones(n_merchants, dtype=np.int64)

    n_fraud = int(round(n_merchants * prevalence))
    if n_fraud == 0:
        return TypologyAssignment(typology_index, onset_day, ramp_days)

    chosen = rng.choice(n_merchants, size=n_fraud, replace=False)
    mix = np.array([typologies[t].mix for t in order], dtype=np.float64)
    kinds = rng.choice(len(order), size=n_fraud, p=mix / mix.sum())
    typology_index[chosen] = kinds

    lo_onset = np.array([typologies[t].onset_day_min for t in order])
    hi_onset = np.array([typologies[t].onset_day_max for t in order])
    lo_ramp = np.array([typologies[t].ramp_days_min for t in order])
    hi_ramp = np.array([typologies[t].ramp_days_max for t in order])

    onset_day[chosen] = rng.integers(lo_onset[kinds], hi_onset[kinds] + 1)
    ramp_days[chosen] = rng.integers(lo_ramp[kinds], hi_ramp[kinds] + 1)
    return TypologyAssignment(typology_index, onset_day, ramp_days)


def ramp_progress(assignment: TypologyAssignment, n_days: int) -> F64:
    """``(n_merchants, n_days)`` ramp progress in [0,1]; 0 everywhere before onset.

    This is the one clock every typology effect runs on. A mark delta applied without
    multiplying by progress would step at onset regardless of the ramp length, which
    would quietly turn R2 — the slow-ramp typology, the one v1 failed on and the reason
    this generator exists — into a second R6.
    """
    days = np.arange(n_days, dtype=np.float64)[None, :]
    onset = assignment.onset_day[:, None].astype(np.float64)
    ramp = np.maximum(assignment.ramp_days[:, None].astype(np.float64), 1.0)
    progress = np.clip((days - onset) / ramp, 0.0, 1.0)
    return np.where(assignment.is_fraud[:, None], progress, 0.0)


def intensity_multiplier(
    assignment: TypologyAssignment,
    typologies: dict[TypologyId, TypologyParams],
    progress: F64,
) -> F64:
    """``(n_merchants, n_days)`` multiplier on the merchant's base daily intensity.

    ``1 + (multiple - 1) * progress ** convexity`` — the exponent is the entire
    separation between R1 and L3. L3 ramps linearly over the whole horizon (convexity is
    not a parameter it has); R1 ramps with convexity > 1 over two to three weeks. On
    ``v_gmv_z`` they are the same merchant. On the second difference they are not.

    ``vanish_after_ramp`` then collapses R1 to ``vanish_intensity`` once the ramp
    completes: the *bust* half of bust-out, without which R1 is only a growth spurt.
    """
    order = list(TypologyId)
    idx = assignment.typology_index
    fraud = assignment.is_fraud

    multiple = per_merchant_field(assignment, typologies, "intensity_multiple", default=1.0)
    convexity = per_merchant_field(assignment, typologies, "convexity", default=1.0)
    vanish_level = per_merchant_field(assignment, typologies, "vanish_intensity", default=1.0)
    vanishes = np.array(
        [typologies[t].vanish_after_ramp for t in order] + [False], dtype=bool
    )[np.where(fraud, idx, len(order))]

    mult = 1.0 + (multiple[:, None] - 1.0) * progress ** convexity[:, None]
    done = (progress >= 1.0) & vanishes[:, None] & fraud[:, None]
    return np.asarray(np.where(done, vanish_level[:, None], mult), dtype=np.float64)


def per_merchant_field(
    assignment: TypologyAssignment,
    typologies: dict[TypologyId, TypologyParams],
    name: str,
    *,
    default: float,
) -> F64:
    """Gather one ``TypologyParams`` field into a ``(n_merchants,)`` array.

    Non-fraud merchants get ``default`` — 0.0 for an additive delta, 1.0 for a
    multiplicative one. Written once and gathered by name rather than as seventeen
    hand-written lookups, because seventeen lookups is seventeen chances to gather
    ``fail_rate_add`` into the array the engine reads as ``cnp_share_add`` and never
    notice: both are small positive floats and every downstream number stays plausible.
    """
    order = list(TypologyId)
    table = np.array(
        [float(getattr(typologies[t], name)) for t in order] + [default], dtype=np.float64
    )
    idx = np.where(assignment.is_fraud, assignment.typology_index, len(order))
    return np.asarray(table[idx], dtype=np.float64)
