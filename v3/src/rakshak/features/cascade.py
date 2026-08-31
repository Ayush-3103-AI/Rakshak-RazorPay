"""The three-stage inference cascade (CLAUDE.md §Architecture).

| Stage | Runs on | Features | Budget |
|---|---|---|---|
| 0 — screen | every merchant, every day | T1 only | <= 0.5 ms |
| 1 — score | top 10% from stage 0 | T1 + T2 + cohort | <= 10 ms |
| 2 — explain | non-``PASS`` decisions only | ``pred_contrib`` reason codes | <= 50 ms |

**The cascade is what buys NFR-03**, and it is worth being explicit about why, because the
arithmetic is the entire argument. A full daily sweep of 10,000 merchants has a 30-second
budget. Reading all 28 features for every merchant costs ~0.24 ms each, which is 2.4 s —
so a single-stage design fits too, on this machine, today. What it does not do is *stay*
fitting: the T2 divergences are 75% of that cost for 4 of the 28 columns, and the fifth
column of that kind, or the tenth, is what turns a comfortable sweep into a missed window.
The cascade pays the T2 cost for a tenth of the population, so the budget grows with the
number of *interesting* merchants rather than with the register.

Nothing here imports a model. Stage 0's screen is a scale-free statistic over the T1
z-scores, stage 1 emits the feature matrix a rung scores, and stage 2 is a row selection
handed to whatever produced the decisions — the routing is the feature layer's business
and the explaining is the model layer's. That split is also what keeps this module inside
Prime Directive 3's quarantine.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from rakshak.features import cohort, registry
from rakshak.features.spec import FeatureSpec
from rakshak.features.state import MerchantState
from rakshak.schemas import Action, FeatureVector, Tier

__all__ = [
    "SCREEN_FRACTION",
    "STAGE0_BUDGET_MS",
    "STAGE1_BUDGET_MS",
    "STAGE2_BUDGET_MS",
    "SWEEP_BUDGET_S",
    "SWEEP_POPULATION",
    "Cascade",
    "SweepResult",
    "stage2_rows",
]

#: NFR-01. Stage-0 screen, p99 per merchant-epoch, one core.
STAGE0_BUDGET_MS = 0.5
#: NFR-02. Stage-1 full scoring, p99 per merchant-epoch, one core.
STAGE1_BUDGET_MS = 10.0
#: Stage-2 explain, p99 per non-PASS decision. From the architecture table; it has no NFR
#: number of its own because it is bounded by capacity K rather than by population.
STAGE2_BUDGET_MS = 50.0
#: NFR-03, and the population it is quoted for.
SWEEP_BUDGET_S = 30.0
SWEEP_POPULATION = 10_000

#: The fraction of the population stage 0 promotes to stage 1. 10% of 10,000 is 1,000
#: merchants against an analyst capacity K of 50 — a 20x headroom over what can actually be
#: actioned, which is the right side to err on: a merchant dropped at stage 0 is invisible
#: to every rung above it, and no metric downstream can recover it.
SCREEN_FRACTION = 0.10


@dataclass(frozen=True, slots=True)
class SweepResult:
    """One epoch of the cascade over a population."""

    as_of: datetime
    merchants: tuple[str, ...]
    #: Stage-0 screen statistic, one per merchant, in ``merchants`` order.
    screen: np.ndarray
    #: Row indices into ``merchants`` that stage 0 promoted, ranked best-first.
    promoted: np.ndarray
    #: ``(len(promoted) x len(registry.ORDER))`` feature matrix for the promoted rows only.
    features: np.ndarray
    #: Cohort residuals for the promoted rows, or None when no assignment was supplied.
    residuals: np.ndarray | None
    #: Wall-clock seconds for the whole sweep. Compared against ``SWEEP_BUDGET_S``.
    seconds: float

    def stage_reached(self) -> np.ndarray:
        out = np.zeros(len(self.merchants), dtype=np.int8)
        out[self.promoted] = 1
        return out

    def vectors(self) -> list[FeatureVector]:
        """The promoted merchants as ``FeatureVector``s, ``stage_reached=1``.

        Only the promoted ones: a merchant screened out at stage 0 has no T2 columns and
        therefore no vector a rung could score, which is the whole point of the cascade.
        """
        return [
            FeatureVector(
                merchant_id=self.merchants[row],
                as_of=self.as_of.date(),
                values=np.asarray(self.features[i], dtype=np.float64),
                feature_schema_version=1,
                computed_by="online",
                stage_reached=1,
            )
            for i, row in enumerate(self.promoted)
        ]


@dataclass(frozen=True, slots=True)
class Cascade:
    """The staged feature readers, resolved from the registry once.

    Instances are stateless and shared, exactly like ``FeatureSpec`` instances — everything
    mutable is in the ``MerchantState`` objects handed to each call.
    """

    t1: tuple[FeatureSpec, ...]
    t2: tuple[FeatureSpec, ...]
    #: Positions within ``t1`` that the screen statistic reads. See ``screen``.
    screen_index: np.ndarray
    screen_fraction: float

    @classmethod
    def from_registry(cls, *, screen_fraction: float = SCREEN_FRACTION) -> Cascade:
        t1 = registry.of_tier(Tier.T1)
        t2 = registry.of_tier(Tier.T2)
        if tuple(s.name for s in (*t1, *t2)) != registry.ORDER:
            # Concatenating the two stages has to reproduce registry.ORDER exactly, because
            # ORDER is the column order every model was trained on (09-interfaces.md §9) and
            # a mismatch fails silently. Today the tier modules are imported T1-then-T2 so
            # this holds; a T3 module inserted in the middle would break it, loudly, here.
            raise ValueError(
                "registry.ORDER is not T1 followed by T2, so a stage-0 vector concatenated "
                "with a stage-1 vector no longer reproduces the trained column order. "
                "Either restore the import order in features/__init__.py or teach the "
                "cascade an explicit permutation — do not let this pass."
            )
        # The screen reads the z-scored T1 columns, which are the ones the cohort layer
        # flags: they are the only T1 columns on a shared scale, so a max over them is
        # comparable across merchants. The bounded-share columns cannot exceed 1.0 and so
        # can never be the maximum next to a z, which makes including them harmless and
        # excluding them arbitrary.
        index = np.array(
            [i for i, s in enumerate(t1) if s.has_cohort_residual], dtype=np.intp
        )
        return cls(t1=t1, t2=t2, screen_index=index, screen_fraction=screen_fraction)

    # ── stage 0 ──────────────────────────────────────────────────────────────

    def stage0(self, state: MerchantState, as_of: datetime) -> np.ndarray:
        """T1 only, for one merchant. NFR-01: p99 <= 0.5 ms."""
        return np.fromiter(
            (spec.value(spec.state_of(state), as_of) for spec in self.t1),
            dtype=np.float64,
            count=len(self.t1),
        )

    def screen(self, t1_values: np.ndarray) -> float:
        """The stage-0 statistic: the largest absolute z the merchant is showing today.

        ponytail: a max-|z| screen, not a model. It needs no trained artifact, which is
        what keeps stage 0 servable on every merchant every day and keeps this module out
        of ``models/``. A deployment would screen with Rung 1's rules instead — pass their
        scores to ``promote`` directly; nothing here requires this particular statistic.
        """
        if self.screen_index.size == 0:
            return 0.0
        return float(np.max(np.abs(t1_values[self.screen_index])))

    def promote(self, screen: np.ndarray) -> np.ndarray:
        """Row indices of the top ``screen_fraction``, ranked best-first.

        Ties are broken by ``argsort``'s stable order rather than by expanding the cut,
        because the promoted set is a fixed compute budget and a tie that widens it is a
        tie that blows it.
        """
        n = screen.size
        k = min(n, max(1, int(round(n * self.screen_fraction)))) if n else 0
        if k == 0:
            return np.empty(0, dtype=np.intp)
        top = np.argpartition(-screen, k - 1)[:k]
        return np.asarray(top[np.argsort(-screen[top], kind="stable")], dtype=np.intp)

    # ── stage 1 ──────────────────────────────────────────────────────────────

    def stage1(
        self, state: MerchantState, as_of: datetime, *, t1_values: np.ndarray | None = None
    ) -> np.ndarray:
        """The full ``registry.ORDER`` vector for one merchant. NFR-02: p99 <= 10 ms.

        ``t1_values`` is the stage-0 vector when the caller still has it. Re-reading T1 is
        correct but wasteful, and in the sweep it is exactly the work the cascade exists to
        avoid doing twice.
        """
        head = self.stage0(state, as_of) if t1_values is None else t1_values
        tail = np.fromiter(
            (spec.value(spec.state_of(state), as_of) for spec in self.t2),
            dtype=np.float64,
            count=len(self.t2),
        )
        return np.concatenate((head, tail))

    # ── the sweep ────────────────────────────────────────────────────────────

    def sweep(
        self,
        states: Sequence[MerchantState],
        as_of: datetime,
        *,
        assignment: cohort.CohortAssignment | None = None,
        screen_fn: Callable[[np.ndarray], float] | None = None,
    ) -> SweepResult:
        """One epoch: stage 0 over everyone, stage 1 over the promoted tenth. NFR-03.

        ``screen_fn`` replaces the default max-|z| statistic; it is handed the stage-0
        vector and returns one number, larger meaning more interesting.
        """
        started = time.perf_counter_ns()
        score = self.screen if screen_fn is None else screen_fn
        merchants = tuple(s.merchant_id for s in states)

        n_t1 = len(self.t1)
        heads = np.empty((len(states), n_t1), dtype=np.float64)
        screen = np.empty(len(states), dtype=np.float64)
        for i, state in enumerate(states):
            heads[i] = self.stage0(state, as_of)
            screen[i] = score(heads[i])

        promoted = self.promote(screen)
        features = np.empty((promoted.size, n_t1 + len(self.t2)), dtype=np.float64)
        for i, row in enumerate(promoted):
            features[i] = self.stage1(states[row], as_of, t1_values=heads[row])

        residuals = None
        if assignment is not None and promoted.size:
            resid_names = cohort.residual_features()
            cols = np.array([registry.ORDER.index(n) for n in resid_names], dtype=np.intp)
            # Residualised across the promoted set only. The leave-one-out median is a
            # property of whoever is in the cohort at the time, and stage 1 has only
            # promoted merchants in front of it — so a confounder that lifts the whole
            # platform is only partly visible here, which understates the correction.
            # Residualising the full population would mean paying stage 1 for all of it,
            # which is the cost the cascade exists to avoid. Named, not hidden.
            residuals = cohort.residual_matrix(
                assignment, [merchants[r] for r in promoted], features[:, cols]
            )

        return SweepResult(
            as_of=as_of,
            merchants=merchants,
            screen=screen,
            promoted=promoted,
            features=features,
            residuals=residuals,
            seconds=(time.perf_counter_ns() - started) / 1e9,
        )


def stage2_rows(actions: Sequence[Action]) -> np.ndarray:
    """Row indices stage 2 explains: the non-``PASS`` decisions and nothing else.

    ``Decision.__post_init__`` requires exactly three reason codes on every non-``PASS``
    action and none on a ``PASS``, so this selection and that validator are two halves of
    FR-014. Computing ``pred_contrib`` for the whole population instead would cost roughly
    the population over K — two hundred times — more than it can ever be read.
    """
    return np.array([i for i, a in enumerate(actions) if a is not Action.PASS], dtype=np.intp)
