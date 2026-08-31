"""Cohort assignment, empirical-Bayes shrinkage, and the cohort-residual layer.

**This is the sprint's central hypothesis.** Charter K-1 fires here, and T-142 is the
ticket that either validates v2 or kills it.

The named open problem in post-onboarding merchant monitoring is separating a fraudster's
gradual adversarial drift from the platform's sudden natural drift. Both look identical in
a per-merchant z-score: a merchant whose auth-failure rate jumps 3σ is either being used to
test stolen cards or is behind a gateway that fell over. The difference is not in the
merchant — it is in whether everybody *like* the merchant moved at the same time.

So every drift feature gets a companion:

    r_f(m, t) = z_f(m, t) - median over cohort(m) excluding m itself of z_f(., t)

When a festival lifts GMV platform-wide, every `z_gmv` rises together and every residual
stays near zero. When one merchant ramps alone, its residual explodes. The confounder layer
P1-P6 exists to test exactly this and gate G5 is its verdict.

Two implementation points that are load-bearing rather than decorative:

**Leave-one-out is a rank arithmetic trick, not N recomputations.** Sort the cohort once
per epoch — O(N log N) — then read each member's excluding-self median straight off the
sorted array by index. An O(N²) recompute at 10,000 merchants x 180 epochs is 1.8e10
median calls and the feature would be cut for compute rather than for merit, which would
be the wrong reason to lose the hypothesis.

**Excluding self is not a nicety.** In a cohort of 30 where one merchant is the drifter,
including it in the median it is measured against pulls the reference toward the anomaly
and shrinks exactly the signal the layer exists to expose. In a cohort of 3, it destroys it.

Prime Directive 3: nothing here may name a field in ``schemas.RADIOACTIVE_FIELDS``, and the
cohort key is built from ``MerchantProfile`` — onboarding facts — and nothing else. A cohort
that knew a merchant's persona would make every number downstream meaningless.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from rakshak.features.registry import ORDER, REGISTRY
from rakshak.schemas import MerchantProfile

__all__ = [
    "EB_PRIOR_WEIGHT",
    "GMV_DECILES",
    "MIN_COHORT_MEMBERS",
    "VINTAGE_BUCKET_EDGES",
    "CohortAssignment",
    "assign_cohorts",
    "eb_shrink",
    "loo_median",
    "residual_features",
    "residual_matrix",
    "residuals",
    "shrunk_z",
]

#: FR-013. Below this many members a cohort statistic is noise, and a residual against
#: noise is worse than no residual: it adds variance to a clean z. Back off instead.
MIN_COHORT_MEMBERS = 30

#: `(mcc_group, gmv_decile, vintage_bucket)` — the first term is what a merchant *does*,
#: the second is its scale, and the third is its maturity. All three come from the
#: onboarding profile, so a merchant's cohort is fixed at approval and cannot drift with
#: the behaviour the layer is trying to measure. A cohort keyed on anything observed after
#: onboarding would move with the fraud.
GMV_DECILES = 10

#: Months of business age at onboarding: under six months, under two years, under five,
#: and everything older. Bucketed rather than continuous because the key has to be an exact
#: match for a cohort to have members at all.
VINTAGE_BUCKET_EDGES: tuple[int, ...] = (6, 24, 60)

#: The pseudo-count in the empirical-Bayes shrinkage: how many days of the cohort's prior a
#: merchant's own history has to outweigh before its own baseline dominates. 30 makes the
#: crossover one warmup window, which is the point at which the baseline freezes anyway.
EB_PRIOR_WEIGHT = 30.0

CohortLevel = Literal["full", "mcc_group", "global"]


@dataclass(frozen=True, slots=True)
class CohortAssignment:
    """Which cohort each merchant is measured against, and how far it had to back off.

    ``level`` is kept per merchant rather than thrown away because it is the diagnostic
    that explains a residual that did nothing: a merchant sitting in the global cohort is
    being compared against the whole platform, and its residual is a weaker instrument than
    one computed inside a 200-member `(grocery, decile 4, 2-5y)` bucket. Reporting the
    distribution of levels is how the ablation in T-142 stays honest about which merchants
    the hypothesis was actually tested on.
    """

    label: dict[str, str]
    level: dict[str, CohortLevel]
    members: dict[str, tuple[str, ...]]

    def size(self, merchant_id: str) -> int:
        return len(self.members[self.label[merchant_id]])

    def level_counts(self) -> dict[CohortLevel, int]:
        counts: dict[CohortLevel, int] = {"full": 0, "mcc_group": 0, "global": 0}
        for lvl in self.level.values():
            counts[lvl] += 1
        return counts


def _vintage_bucket(months: int) -> int:
    return int(np.searchsorted(VINTAGE_BUCKET_EDGES, months, side="right"))


def _gmv_decile(gmv: float, cuts: np.ndarray) -> int:
    return int(np.searchsorted(cuts, gmv, side="right"))


def assign_cohorts(
    profiles: Mapping[str, MerchantProfile],
    *,
    min_members: int = MIN_COHORT_MEMBERS,
) -> CohortAssignment:
    """Assign every merchant to `(mcc_group, gmv_decile, vintage_bucket)`, backing off.

    The chain is full key -> `mcc_group` -> global, taking the first level with at least
    ``min_members``. Deciles are cut on the population's declared GMV, which is an
    onboarding fact, so the cut points do not move as merchants trade.
    """
    ids = sorted(profiles)
    if not ids:
        return CohortAssignment({}, {}, {})

    gmv = np.array([profiles[m].declared_monthly_gmv for m in ids], dtype=np.float64)
    cuts = np.quantile(gmv, np.arange(1, GMV_DECILES) / GMV_DECILES)

    full: dict[str, str] = {}
    coarse: dict[str, str] = {}
    for i, m in enumerate(ids):
        p = profiles[m]
        coarse[m] = f"mcc={p.mcc_group}"
        full[m] = (
            f"mcc={p.mcc_group}|gmv={_gmv_decile(gmv[i], cuts)}"
            f"|vint={_vintage_bucket(p.vintage_months)}"
        )

    def group(keys: Mapping[str, str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for m in ids:
            out.setdefault(keys[m], []).append(m)
        return out

    full_groups = group(full)
    coarse_groups = group(coarse)

    label: dict[str, str] = {}
    level: dict[str, CohortLevel] = {}
    for m in ids:
        if len(full_groups[full[m]]) >= min_members:
            label[m], level[m] = full[m], "full"
        elif len(coarse_groups[coarse[m]]) >= min_members:
            label[m], level[m] = coarse[m], "mcc_group"
        else:
            label[m], level[m] = "global", "global"

    grouped: dict[str, list[str]] = {}
    for m in ids:
        grouped.setdefault(label[m], []).append(m)
    return CohortAssignment(
        label=label,
        level=level,
        members={k: tuple(v) for k, v in grouped.items()},
    )


def loo_median(values: np.ndarray) -> np.ndarray:
    """Median of ``values`` excluding each element in turn. One sort, then index arithmetic.

    For a cohort of n, removing the element at sorted position ``r`` leaves an array of
    m = n-1 whose reduced index ``j`` is the original index ``j`` when ``j < r`` and
    ``j+1`` otherwise. Both median positions are known in closed form from m's parity, so
    every member's leave-one-out median is two gathers off one sorted array — O(n log n)
    for the cohort, not O(n²).

    A cohort of one has nothing to compare against and returns 0.0, which makes its
    residual equal to its raw z. That is the right degenerate behaviour: with no peers,
    the platform-drift correction has no information and must not invent any.
    """
    n = values.size
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)

    m = n - 1
    if m % 2 == 1:
        j = (m - 1) // 2
        return np.asarray(ordered[np.where(rank > j, j, j + 1)], dtype=np.float64)
    j1, j2 = m // 2 - 1, m // 2
    lo = ordered[np.where(rank > j1, j1, j1 + 1)]
    hi = ordered[np.where(rank > j2, j2, j2 + 1)]
    return np.asarray(0.5 * (lo + hi), dtype=np.float64)


def residuals(
    assignment: CohortAssignment,
    values: Mapping[str, float],
) -> dict[str, float]:
    """One feature's leave-one-out cohort residual, for every merchant in ``values``.

    Merchants absent from ``values`` are dropped from their cohort for this epoch rather
    than defaulted to 0.0 — a merchant that produced no value did not observe a platform
    event either, and stuffing it in at zero drags the cohort median toward zero on exactly
    the days a confounder is lifting everyone.
    """
    out: dict[str, float] = {}
    for _, members in assignment.members.items():
        present = [m for m in members if m in values]
        if not present:
            continue
        arr = np.array([values[m] for m in present], dtype=np.float64)
        med = loo_median(arr)
        for i, m in enumerate(present):
            out[m] = float(arr[i] - med[i])
    return out


def residual_matrix(
    assignment: CohortAssignment,
    merchants: Sequence[str],
    z: np.ndarray,
) -> np.ndarray:
    """Residualise a whole epoch at once: ``z`` is (merchants x features), row-aligned.

    This is the shape the model layer wants — one pass per epoch over the population, all
    flagged features together — and it is where the "O(1) per merchant after an O(N log N)
    sort once per day" cost claim in the register is actually paid.
    """
    if z.ndim != 2 or z.shape[0] != len(merchants):
        raise ValueError(
            f"z must be (len(merchants), n_features); got {z.shape} for "
            f"{len(merchants)} merchants"
        )
    index = {m: i for i, m in enumerate(merchants)}
    out = np.array(z, dtype=np.float64, copy=True)
    for _, members in assignment.members.items():
        rows = np.array([index[m] for m in members if m in index], dtype=np.int64)
        if rows.size == 0:
            continue
        block = z[rows, :]
        for col in range(z.shape[1]):
            out[rows, col] = block[:, col] - loo_median(block[:, col])
    return out


def residual_features() -> tuple[str, ...]:
    """The registered features carrying ``has_cohort_residual``, in ``registry.ORDER``.

    Column order for the residual block is the base features' order, so a model's residual
    columns line up with the columns they were derived from. 09-interfaces.md §9 again:
    order is contract.
    """
    return tuple(n for n in ORDER if REGISTRY[n].has_cohort_residual)


# ─────────────────────────────────────────────────────────────────────────────
# Empirical-Bayes shrinkage — the cold-start half of FR-011
# ─────────────────────────────────────────────────────────────────────────────


def eb_shrink(
    n: float,
    sample: float,
    prior: float,
    *,
    prior_weight: float = EB_PRIOR_WEIGHT,
) -> float:
    """Shrink a merchant's own statistic toward its cohort's, weighted by evidence.

    ``(n * sample + k * prior) / (n + k)``. At n=0 the merchant is its cohort; at n >> k it
    is itself. This is the carried-forward v1 ADR and it is the answer to the question the
    frozen baseline raises: a baseline that freezes after warmup is worthless for the first
    warmup window, and a merchant is at its most opaque in exactly the weeks a bust-out is
    being set up.
    """
    if prior_weight < 0.0:
        raise ValueError(
            f"prior_weight is a pseudo-count and cannot be negative; got {prior_weight}"
        )
    total = n + prior_weight
    if total <= 0.0:
        return prior
    return (n * sample + prior_weight * prior) / total


def shrunk_z(
    x: float,
    n: float,
    sample_mean: float,
    sample_std: float,
    prior_mean: float,
    prior_std: float,
    *,
    prior_weight: float = EB_PRIOR_WEIGHT,
    floor: float = 1e-9,
) -> float:
    """A z against an EB-shrunk baseline. Both moments shrink, not only the mean.

    Shrinking the mean and keeping a cold merchant's own (tiny, noisy, possibly zero)
    standard deviation would produce enormous z-scores for the shrinkage's own residual
    error — a false-positive generator dressed as a cold-start fix.
    """
    mean = eb_shrink(n, sample_mean, prior_mean, prior_weight=prior_weight)
    std = eb_shrink(n, sample_std, prior_std, prior_weight=prior_weight)
    return (x - mean) / max(std, floor)
