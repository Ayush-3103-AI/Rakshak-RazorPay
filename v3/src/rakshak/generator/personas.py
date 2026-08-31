"""Legitimate merchant behaviour, L1-L8. 08-generator-v2-spec.md §2.

Every merchant gets exactly one persona, including the fraud merchants: a fraud merchant
is a legitimate merchant that turned, so it behaves as its persona until
``drift_onset_at`` and carries typology deltas on top after it.

**The hard-negative annotations in the spec table are the whole point of this module.**
L3 must look like R1 on volume and differ only in convexity; L5 must look scripted; L8
must break naive refund features. A generator whose negatives are easy produces a model
whose false-positive cost is fictional, and v1's 20%-prevalence headline is what that
looks like when it reaches a results table.

Nothing here draws marks for a single transaction. The engine works on whole
``(n_merchants, n_days)`` arrays, so these functions return matrices; a per-transaction
loop over 10k x 180 merchant-days would dominate the runtime for no gain in fidelity.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rakshak.generator.config import PersonaParams
from rakshak.schemas import PersonaId

__all__ = [
    "SHAPE_BUILDERS",
    "daily_shape",
    "sample_persona_ids",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def sample_persona_ids(
    rng: np.random.Generator, n_merchants: int, personas: dict[PersonaId, PersonaParams]
) -> I64:
    """Assign a persona index per merchant, in ``PersonaId`` declaration order.

    Returns indices rather than enums so the engine can use them to gather rows out of
    per-persona parameter arrays. ``PersonaId`` is an ordered enum, so index *is* the
    identity — there is no separate lookup table to fall out of sync.
    """
    order = list(PersonaId)
    shares = np.array([personas[p].share for p in order], dtype=np.float64)
    return np.asarray(
        rng.choice(len(order), size=n_merchants, p=shares / shares.sum()), dtype=np.int64
    )


# ─────────────────────────────────────────────────────────────────────────────
# Daily shape: the persona's multiplicative modulation of its base intensity
# ─────────────────────────────────────────────────────────────────────────────


def _flat(
    rng: np.random.Generator, params: PersonaParams, n: int, n_days: int
) -> F64:
    """L1, L5. No structure — the modulation is entirely day-of-week and noise, which the
    engine applies. Returning ones rather than "nearly ones" keeps the easy negative
    genuinely easy, which is what makes it the *bulk* of the population."""
    return np.ones((n, n_days), dtype=np.float64)


def _seasonal(
    rng: np.random.Generator, params: PersonaParams, n: int, n_days: int
) -> F64:
    """L2, L8. Sale windows: large spikes, no fraud.

    **This is the hard negative for ``v_gmv_z``.** The windows are placed per merchant,
    not platform-wide — a platform-wide sale is a P1 confounder and lives in a different
    layer. If every L2 spiked on the same day the cohort residual would erase them for
    free, which is precisely the wrong lesson for gate G5 to teach.
    """
    shape = np.ones((n, n_days), dtype=np.float64)
    width = max(1, int(round(params.shape_span * n_days)))
    days = np.arange(n_days)
    for _ in range(params.shape_count):
        starts = rng.integers(0, max(1, n_days - width), size=n)[:, None]
        inside = (days[None, :] >= starts) & (days[None, :] < starts + width)
        # Half-sine so the spike ramps rather than teleports; a step here would make the
        # spike trivially separable from an organic ramp for the wrong reason.
        phase = np.clip((days[None, :] - starts) / width, 0.0, 1.0)
        shape += inside * (params.shape_strength - 1.0) * np.sin(np.pi * phase)
    return shape


def _growth(
    rng: np.random.Generator, params: PersonaParams, n: int, n_days: int
) -> F64:
    """L3. Sustained upward ramp — **the hardest negative in the population.**

    The ramp is *linear*, and that is the single design decision that makes L3 work. R1
    ramps convexly over 14-21 days; L3 ramps linearly over the whole horizon. On
    ``v_gmv_z`` and ``v_txn_count_z`` they are indistinguishable, and the only thing that
    separates them is the second difference (``v_gmv_accel``). T-112 asserts exactly that
    asymmetry: a linear fit explains L3's cohort GMV and fails on R1's.

    Per-merchant variation is in the *endpoint*, not the curvature, so the cohort mean
    stays linear.
    """
    end = params.shape_strength * np.exp(
        rng.normal(0.0, params.shape_jitter, size=n)
    )
    t = np.linspace(0.0, 1.0, n_days)[None, :]
    return np.asarray(1.0 + (end[:, None] - 1.0) * t, dtype=np.float64)


def _lumpy(
    rng: np.random.Generator, params: PersonaParams, n: int, n_days: int
) -> F64:
    """L4. Few transactions, very high ticket, irregular — breaks count-based features
    and earns a high ``t_p95_median_ratio`` without any fraud."""
    burst = rng.random((n, n_days)) < params.shape_span
    return np.where(burst, params.shape_strength, 1.0).astype(np.float64)


def _dormant(
    rng: np.random.Generator, params: PersonaParams, n: int, n_days: int
) -> F64:
    """L7. A 30-60 day gap and then a genuine resumption.

    **Hard negative for ``v_dormant_burst``.** The revival is a real business coming
    back, and it must look, on that feature alone, exactly like a dormant shell waking up
    to bust out.
    """
    shape = np.ones((n, n_days), dtype=np.float64)
    gap = max(1, int(round(params.shape_span * n_days)))
    starts = rng.integers(0, max(1, n_days - gap), size=n)[:, None]
    days = np.arange(n_days)[None, :]
    inside = (days >= starts) & (days < starts + gap)
    return np.where(inside, params.shape_strength, shape)


#: Named in ``configs/scenario_v2.yaml`` as ``personas.<id>.shape``. ``config.py``
#: validates the name against these keys at load time, so a typo fails on the config file
#: rather than silently selecting "flat" and quietly deleting a hard negative.
SHAPE_BUILDERS = {
    "flat": _flat,
    "seasonal": _seasonal,
    "growth": _growth,
    "lumpy": _lumpy,
    "dormant": _dormant,
}


def daily_shape(
    rng: np.random.Generator,
    params: PersonaParams,
    n_merchants: int,
    n_days: int,
) -> F64:
    """The ``(n_merchants, n_days)`` multiplicative daily factor for one persona."""
    return SHAPE_BUILDERS[params.shape](rng, params, n_merchants, n_days)
