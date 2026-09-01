"""The metric suite: PR-AUC, savings + four mandatory floors, TTD, P@K, ECE, stability,
per-typology recall (10-eval-harness-spec.md §2; FR-021, FR-022, FR-024).

Two decisions are made here once, and everything downstream inherits them.

**The unit of evaluation is the merchant-day, not the merchant.** Rakshak emits an action
for every cleared merchant every epoch, so that is what must be scored. The positive class
is ``(merchant is fraud) AND (day >= drift_onset_day)``: a fraud merchant's days *before*
its drift onset are legitimate days, and an alert on one of them is a false positive, not
an early catch. Scoring at merchant level instead would silently reward a model that
alerts on day 0 for a merchant that turns bad on day 150, and it would make PR-AUC
inconsistent with TTD.

**Floors and rungs are compared as rankers with the decision layer held fixed.**
``savings_of_ranking`` is the one function both go through, so the difference between a
rung and the ``volume_rank`` floor is the ranking and nothing else. That is what makes a
FLOOR-FAIL attributable.

Every stochastic function takes ``rng: np.random.Generator``. There is no module-level RNG
here and no bare ``np.random.*`` call.

Intra-package note: ``RungOutput`` and ``Truth`` below are *not* cross-package boundary
types — they are the eval package talking to itself, and ``schemas.py`` is frozen. Nothing
outside ``src/rakshak/eval/`` constructs them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from rakshak.schemas import Action, EvalResult, Split, TypologyId

__all__ = [
    "ALL_FLOORS",
    "CostParams",
    "CoverageRow",
    "Floors",
    "OnsetLocalisation",
    "PerfBudget",
    "RescaledKS",
    "RungOutput",
    "Truth",
    "alert_jaccard_wow",
    "alerts_per_day",
    "build_eval_result",
    "detection_rates",
    "expected_calibration_error",
    "false_hold_coverage",
    "floors_at_capacity",
    "onset_localisation_error",
    "precision_recall_at_k",
    "pr_auc",
    "recall_by_typology",
    "roc_auc",
    "savings_of_actions",
    "savings_of_ranking",
    "time_to_detection",
    "top_k_by_day",
    "tpp_rescaled_ks",
]

#: The four floors that appear on EVERY savings row (FR-021). v1 discovered `random`
#: winning on savings at 20% prevalence *by accident*; naming them here makes that
#: discovery automatic and unavoidable rather than lucky.
ALL_FLOORS: Final = ("all_pass", "all_hold", "random_at_k", "volume_rank")


@dataclass(frozen=True, slots=True)
class CostParams:
    """Instance-dependent costs (10-eval-harness-spec.md §2; 08-generator-v2-spec.md §8).

    These three rupee values are **swept parameters, not constants**. v1 measured the
    asymmetry at 47.5 / 13.1 / 61,368 against a literature band of 400-600 — three orders
    of magnitude of spread, which says the ratio cannot be assumed, only measured per
    deployment. See :func:`rakshak.eval.capacity.sweep_cost_asymmetry`.

    ``p_catch`` is a **spec gap**: the cost matrix in 10-eval-harness-spec.md §2 uses it
    (``REVIEW & fraud -> review_cost + (1-p_catch) * loss``) but the config block in
    08-generator-v2-spec.md §8 does not define it. 0.80 is carried here as an explicit,
    named default so the number is visible rather than buried; it belongs in
    ``configs/scenario_v2.yaml`` under ``costs:`` when Lane A writes that file.
    """

    review_cost_inr: float = 250.0
    false_hold_cost_inr: float = 8000.0
    fraud_loss_multiplier: float = 1.0
    p_catch: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_catch <= 1.0:
            raise ValueError(f"p_catch is a probability; got {self.p_catch!r}")
        for name in ("review_cost_inr", "false_hold_cost_inr", "fraud_loss_multiplier"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0; got {getattr(self, name)!r}")

    @property
    def asymmetry(self) -> float:
        """``false_hold_cost / review_cost`` — the ratio the sweep varies."""
        return self.false_hold_cost_inr / self.review_cost_inr


@dataclass(frozen=True, slots=True)
class PerfBudget:
    """Measured compute facts, supplied by the perf suite (NFR-01..05).

    They are fields on ``EvalResult`` because charter §2 puts latency *inside* the success
    metric: a rung that wins on PR-AUC and cannot be served is not a win. The harness does
    not measure them, so it refuses to invent them — the caller passes what it measured.
    """

    p99_latency_ms: float
    state_bytes_p99: float
    model_size_mb: float


@dataclass(frozen=True, slots=True)
class RungOutput:
    """One row per merchant-day: what the rung scored and what it decided."""

    merchant_id: np.ndarray
    day: np.ndarray
    score: np.ndarray
    action: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.merchant_id)
        if not all(len(a) == n for a in (self.day, self.score, self.action)):
            raise ValueError("RungOutput columns must be the same length")
        if n and not (0.0 <= float(np.min(self.score)) and float(np.max(self.score)) <= 1.0):
            raise ValueError("score is a calibrated probability in [0,1] (it feeds the cost layer)")

    @property
    def alerted(self) -> np.ndarray:
        mask: np.ndarray = self.action != Action.PASS
        return mask

    @property
    def n_days(self) -> int:
        return int(np.unique(self.day).size)


@dataclass(frozen=True, slots=True)
class Truth:
    """One row per merchant. Read only through ``eval``; radioactive to features/models.

    ``onset_day`` is ``nan`` for a merchant with no typology, matching the GroundTruth
    invariant that a typology and a drift onset travel together.
    """

    merchant_id: np.ndarray
    label: np.ndarray
    is_censored: np.ndarray
    loss_inr: np.ndarray
    onset_day: np.ndarray
    typology: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.merchant_id)
        fields = (self.label, self.is_censored, self.loss_inr, self.onset_day, self.typology)
        if not all(len(a) == n for a in (*fields, self.volume)):
            raise ValueError("Truth columns must be the same length")
        if len(np.unique(self.merchant_id)) != n:
            raise ValueError("Truth is one row per merchant; merchant_id must be unique")

    @property
    def prevalence(self) -> float:
        """Merchant-level prevalence among **uncensored** merchants (FR-021).

        Censored merchants are excluded from the denominator, not counted as negatives:
        counting them as negatives deflates the rate, and dropping them without saying so
        is exactly the v1 failure this field exists to prevent.
        """
        keep = ~self.is_censored
        n = int(keep.sum())
        return float(self.label[keep].sum()) / n if n else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Joining truth onto merchant-days
# ─────────────────────────────────────────────────────────────────────────────


def _index_of(output: RungOutput, truth: Truth) -> np.ndarray:
    """Row index into ``truth`` for every merchant-day in ``output``."""
    order = np.argsort(truth.merchant_id)
    pos = np.searchsorted(truth.merchant_id[order], output.merchant_id)
    if pos.size and (pos >= len(order)).any():
        raise KeyError("a scored merchant is absent from Truth")
    idx = order[np.clip(pos, 0, len(order) - 1)]
    if idx.size and (truth.merchant_id[idx] != output.merchant_id).any():
        missing = set(output.merchant_id[truth.merchant_id[idx] != output.merchant_id])
        raise KeyError(f"scored merchants absent from Truth: {sorted(missing)[:5]}")
    return idx


def day_labels(output: RungOutput, truth: Truth) -> tuple[np.ndarray, np.ndarray]:
    """``(y, keep)`` per merchant-day.

    ``y`` is 1 iff the merchant is fraud **and** the day is at or after its drift onset.
    ``keep`` drops label-censored merchants, which have no resolved outcome to score
    against — they are excluded here and counted by ``splits.label_coverage``.
    """
    idx = _index_of(output, truth)
    onset = truth.onset_day[idx]
    is_fraud = truth.label[idx] == 1
    after_onset = np.where(np.isnan(onset), False, output.day >= np.nan_to_num(onset, nan=np.inf))
    return (is_fraud & after_onset).astype(np.int8), ~truth.is_censored[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Ranking metrics
# ─────────────────────────────────────────────────────────────────────────────


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision. ``nan`` if the split contains only one class — reporting 0.0 or
    1.0 there would be a fabricated number, and ``nan`` is loud."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with **equal-mass** bins (10-eval-harness-spec.md §2).

    Equal-mass, not equal-width: at 1.5% prevalence almost every score sits in the bottom
    equal-width bin, and an equal-width ECE would report the calibration of one bin and
    call it the model's. The cost-aware decision layer consumes probabilities, so an
    uncalibrated score makes the whole cost calculation meaningless.
    """
    if y.size == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, min(n_bins, y.size)):
        if chunk.size == 0:
            continue
        total += chunk.size * abs(float(p[chunk].mean()) - float(y[chunk].mean()))
    return total / y.size


# ─────────────────────────────────────────────────────────────────────────────
# Capacity primitive — one implementation, used by floors, rungs and the oracle
# ─────────────────────────────────────────────────────────────────────────────


def top_k_by_day(score: np.ndarray, day: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the ``k`` highest-scoring rows within each day.

    Ties break by row order, deterministically — a random tiebreak would make
    ``alert_jaccard_wow`` measure the tiebreak rather than the model.
    """
    if k < 0:
        raise ValueError(f"capacity K must be >= 0; got {k!r}")
    mask = np.zeros(score.shape, dtype=bool)
    if score.size == 0 or k == 0:
        return mask
    order = np.lexsort((np.arange(score.size), -score, day))
    sorted_day = day[order]
    starts = np.flatnonzero(np.r_[True, sorted_day[1:] != sorted_day[:-1]])
    group_start = np.repeat(starts, np.diff(np.r_[starts, sorted_day.size]))
    rank_in_day = np.arange(sorted_day.size) - group_start
    mask[order[rank_in_day < k]] = True
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Cost and savings (Bahnsen-style, instance-dependent)
# ─────────────────────────────────────────────────────────────────────────────


def row_cost(
    action: np.ndarray, y: np.ndarray, loss: np.ndarray, params: CostParams
) -> np.ndarray:
    """The cost matrix from 10-eval-harness-spec.md §2, elementwise."""
    scaled = loss * params.fraud_loss_multiplier
    fraud = y == 1
    cost = np.zeros(action.shape, dtype=np.float64)

    is_pass = action == Action.PASS
    is_review = action == Action.REVIEW
    is_hold = action == Action.HOLD
    if not np.all(is_pass | is_review | is_hold):
        raise ValueError("every row must carry a PASS/REVIEW/HOLD action")

    cost[is_pass & fraud] = scaled[is_pass & fraud]
    cost[is_review & fraud] = (
        params.review_cost_inr + (1.0 - params.p_catch) * scaled[is_review & fraud]
    )
    cost[is_review & ~fraud] = params.review_cost_inr
    cost[is_hold & fraud] = params.review_cost_inr
    cost[is_hold & ~fraud] = params.false_hold_cost_inr + params.review_cost_inr
    return cost


def cost_of_all_pass(y: np.ndarray, loss: np.ndarray, params: CostParams) -> float:
    """The denominator of savings: doing nothing at all."""
    return float((loss * params.fraud_loss_multiplier)[y == 1].sum())


def savings_of_actions(
    action: np.ndarray, y: np.ndarray, loss: np.ndarray, params: CostParams
) -> float:
    """``1 - total_cost / cost_of_all_pass``. Zero means "no better than doing nothing"."""
    baseline = cost_of_all_pass(y, loss, params)
    if baseline <= 0.0:
        # No fraud in the window: there is nothing to save, and every intervention is
        # pure cost. nan rather than a fabricated 0.0 or -inf.
        return float("nan")
    return 1.0 - float(row_cost(action, y, loss, params).sum()) / baseline


def savings_of_ranking(
    score: np.ndarray,
    day: np.ndarray,
    y: np.ndarray,
    loss: np.ndarray,
    k: int,
    params: CostParams,
    *,
    action: Action = Action.REVIEW,
) -> float:
    """Savings of a *ranker* under capacity K, decision layer held fixed.

    Both the rung and the ranking floors go through this, so the only thing that differs
    between them is the score vector. That is what makes a FLOOR-FAIL attributable to the
    ranking rather than to two different decision policies.
    """
    selected = top_k_by_day(score, day, k)
    actions = np.where(selected, action, Action.PASS)
    return savings_of_actions(actions, y, loss, params)


@dataclass(frozen=True, slots=True)
class Floors:
    """The four mandatory floors. Present on every savings row, without exception."""

    all_pass: float
    all_hold: float
    random_at_k: float
    volume_rank: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.all_pass, self.all_hold, self.random_at_k, self.volume_rank)

    def failed_by(self, savings: float) -> list[str]:
        """Names of the floors ``savings`` does not beat. Non-empty means **FLOOR-FAIL**.

        A ``nan`` savings fails everything: an unmeasurable result is not a passing one.
        """
        if math.isnan(savings):
            return list(ALL_FLOORS)
        return [
            name
            for name, floor in zip(ALL_FLOORS, self.as_tuple(), strict=True)
            if not math.isnan(floor) and savings <= floor
        ]


def floors_at_capacity(
    day: np.ndarray,
    volume: np.ndarray,
    y: np.ndarray,
    loss: np.ndarray,
    k: int,
    params: CostParams,
    rng: np.random.Generator,
) -> Floors:
    """All four floors, scored on exactly the rows the rung was scored on.

    Every argument is the same length and already filtered to the scored rows — a floor
    computed over a different row set than the rung it is compared against is worse than
    no floor at all.
    """
    n = y.size
    if not all(a.size == n for a in (day, volume, loss)):
        raise ValueError("floors must be scored on exactly the rung's rows; lengths differ")
    return Floors(
        all_pass=savings_of_actions(np.full(n, Action.PASS, dtype=object), y, loss, params),
        all_hold=savings_of_actions(np.full(n, Action.HOLD, dtype=object), y, loss, params),
        random_at_k=savings_of_ranking(rng.random(n), day, y, loss, k, params),
        volume_rank=savings_of_ranking(volume, day, y, loss, k, params),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Operational metrics under capacity K
# ─────────────────────────────────────────────────────────────────────────────


def alerts_per_day(output: RungOutput) -> float:
    days = output.n_days
    return float(output.alerted.sum()) / days if days else 0.0


def precision_recall_at_k(
    output: RungOutput, y: np.ndarray, keep: np.ndarray
) -> tuple[float, float]:
    """Precision and recall of the alert set actually emitted under K."""
    alerted = output.alerted & keep
    positives = (y == 1) & keep
    n_alerts = int(alerted.sum())
    n_pos = int(positives.sum())
    hits = int((alerted & positives).sum())
    precision = hits / n_alerts if n_alerts else float("nan")
    recall = hits / n_pos if n_pos else float("nan")
    return precision, recall


# ─────────────────────────────────────────────────────────────────────────────
# Time to detection (FR-022) — with censoring, not around it
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Detection:
    """Per-positive-merchant detection outcome.

    ``ttd`` is ``inf`` for a merchant never detected inside the window — right-censored,
    not missing. ``observable`` is how many days of follow-up that merchant actually had
    after its onset, which is what makes a d30 rate honest at the end of the window: a
    merchant whose drift starts on day 178 cannot be "not detected by day 30", it simply
    was not observed that long, and counting it as a miss would understate the model.
    """

    merchant_id: np.ndarray
    ttd: np.ndarray
    observable: np.ndarray
    typology: np.ndarray


def time_to_detection(output: RungOutput, truth: Truth) -> Detection:
    """TTD = (first epoch with action in {REVIEW, HOLD} at/after onset) - drift_onset.

    An alert *before* onset is not an early detection, it is a false positive that happens
    to be on a merchant that later goes bad. Counting it would make TTD negative and the
    model look prescient.
    """
    last_day = int(output.day.max()) if output.day.size else 0
    idx = _index_of(output, truth)
    onset_row = truth.onset_day[idx]
    positive_row = (truth.label[idx] == 1) & ~truth.is_censored[idx] & ~np.isnan(onset_row)

    merchant_ids, inverse = np.unique(output.merchant_id[positive_row], return_inverse=True)
    onset = np.zeros(merchant_ids.size)
    onset[inverse] = onset_row[positive_row]
    typology = np.empty(merchant_ids.size, dtype=object)
    typology[inverse] = truth.typology[idx][positive_row]

    ttd = np.full(merchant_ids.size, np.inf)
    hit = positive_row & output.alerted & (output.day >= np.nan_to_num(onset_row, nan=np.inf))
    if hit.any():
        pos = np.searchsorted(merchant_ids, output.merchant_id[hit])
        np.minimum.at(ttd, pos, output.day[hit].astype(np.float64) - onset[pos])
    return Detection(
        merchant_id=merchant_ids,
        ttd=ttd,
        observable=last_day - onset,
        typology=typology,
    )


def median_ttd(detection: Detection) -> float:
    """Median TTD over uncensored positives, **including** never-detected as ``inf``.

    Dropping the never-detected and taking a mean of the rest is the standard way to
    report a flattering latency number; if more than half of positives are never caught
    this returns ``inf``, which is the truth and is impossible to misread.
    """
    if detection.ttd.size == 0:
        return float("nan")
    return float(np.median(detection.ttd))


def detection_rates(
    detection: Detection, horizons: tuple[int, ...] = (7, 14, 30)
) -> dict[int, float]:
    """Fraction detected within each horizon, over merchants observed that long.

    A merchant with fewer than ``h`` days of follow-up after onset is dropped from the
    ``h`` denominator rather than counted as a miss. That is administrative right-
    censoring, and it is the difference between a d30 rate and a d30 rate that is wrong at
    the end of every window.
    """
    out = {}
    for h in horizons:
        eligible = detection.observable >= h
        n = int(eligible.sum())
        out[h] = float((detection.ttd[eligible] <= h).sum()) / n if n else float("nan")
    return out


def recall_by_typology(detection: Detection, horizon: int = 30) -> dict[TypologyId, float]:
    """Recall broken out by R1-R9 (required output, not a nice-to-have).

    A single aggregate lets easy R1 hide hard R2 and R7 — that is the v1 slow-ramp
    failure. A typology with no members in the split gets ``nan``, never 0.0: "we caught
    none of the zero R8 merchants" is a different statement from "we missed every R8".
    """
    out: dict[TypologyId, float] = {}
    for typ in TypologyId:
        members = detection.typology == typ.value
        n = int(members.sum())
        eligible = members & (detection.observable >= horizon)
        n_eligible = int(eligible.sum())
        if n == 0 or n_eligible == 0:
            out[typ] = float("nan")
        else:
            out[typ] = float((detection.ttd[eligible] <= horizon).sum()) / n_eligible
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stability (FR-024, NFR-09)
# ─────────────────────────────────────────────────────────────────────────────


def alert_jaccard_wow(output: RungOutput, truth: Truth, *, week_len: int = 7) -> float:
    """Week-over-week Jaccard of the alert set, **restricted to non-drifting merchants**.

    Restricted because a churning alert list on merchants that are genuinely changing is
    correct behaviour; churn on merchants that are not is noise. An alert list that
    reshuffles weekly is unusable by an ops team no matter how good its PR-AUC, and this
    is the number that makes that visible. Target >= 0.60 (NFR-09).
    """
    idx = _index_of(output, truth)
    stable = np.isnan(truth.onset_day[idx])
    alerted = output.alerted & stable
    if not alerted.any():
        return float("nan")

    week = output.day // week_len
    sets = {
        int(w): set(output.merchant_id[alerted & (week == w)].tolist())
        for w in np.unique(week[alerted])
    }
    weeks = sorted(sets)
    scores = []
    for a, b in zip(weeks, weeks[1:], strict=False):
        if b != a + 1:
            continue  # non-adjacent weeks are not a week-over-week comparison
        union = sets[a] | sets[b]
        if union:
            scores.append(len(sets[a] & sets[b]) / len(union))
    return float(np.mean(scores)) if scores else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# The row
# ─────────────────────────────────────────────────────────────────────────────


def build_eval_result(
    *,
    rung: int,
    split: Split,
    output: RungOutput,
    truth: Truth,
    k: int,
    params: CostParams,
    rng: np.random.Generator,
    perf: PerfBudget,
    oracle_savings: float,
    eval_lock_sha: str,
    open_count: int,
    git_sha: str,
    cost_scenario: str = "base",
) -> EvalResult:
    """Every field of ``EvalResult``, from one rung's merchant-day output.

    Nothing here is optional and nothing defaults to a flattering value. ``prevalence`` is
    computed, never passed in (FR-021): v1's headline was computed at 20% prevalence
    against a real rate near 1.5% and reported without saying so, and the surest fix is
    that the number cannot be supplied by whoever writes the report.
    """
    if k < 0:
        raise ValueError(f"capacity K must be >= 0; got {k!r}")
    y, keep = day_labels(output, truth)
    idx = _index_of(output, truth)
    loss = truth.loss_inr[idx]

    apd = alerts_per_day(output)
    if apd > k + 1e-9:
        raise ValueError(
            f"alerts_per_day={apd:.4f} exceeds capacity K={k}. The capacity constraint is "
            "the binding operational fact; a metric computed above it is decoration "
            "(10-eval-harness-spec.md §4)."
        )

    floors = floors_at_capacity(
        output.day[keep], truth.volume[idx][keep], y[keep], loss[keep], k, params, rng
    )
    savings = savings_of_actions(output.action[keep], y[keep], loss[keep], params)
    detection = time_to_detection(output, truth)
    rates = detection_rates(detection)
    precision, recall = precision_recall_at_k(output, y, keep)

    # 10-eval-harness-spec.md §3: "this assertion has caught real bugs and should run on
    # every eval". Imported here rather than at module scope because oracle.py imports this
    # module; a deferred import is cheaper than splitting the cost matrix out to break it.
    from rakshak.eval.oracle import assert_no_leakage, gap_to_oracle

    assert_no_leakage(savings, oracle_savings, label=f"rung {rung} on {split}")
    gap = gap_to_oracle(savings, oracle_savings)

    return EvalResult(
        rung=rung,
        split=split,
        prevalence=truth.prevalence,
        pr_auc=pr_auc(y[keep], output.score[keep]),
        roc_auc=roc_auc(y[keep], output.score[keep]),
        ece=expected_calibration_error(y[keep], output.score[keep]),
        savings=savings,
        savings_floor_random=floors.random_at_k,
        savings_floor_all_pass=floors.all_pass,
        savings_floor_all_hold=floors.all_hold,
        savings_floor_volume_rank=floors.volume_rank,
        precision_at_k=precision,
        recall_at_k=recall,
        alerts_per_day=apd,
        ttd_median_days=median_ttd(detection),
        detection_rate_d7=rates[7],
        detection_rate_d14=rates[14],
        detection_rate_d30=rates[30],
        gap_to_oracle=gap,
        alert_jaccard_wow=alert_jaccard_wow(output, truth),
        recall_by_typology=recall_by_typology(detection),
        p99_latency_ms=perf.p99_latency_ms,
        state_bytes_p99=perf.state_bytes_p99,
        model_size_mb=perf.model_size_mb,
        eval_lock_sha=eval_lock_sha,
        open_count=open_count,
        git_sha=git_sha,
        cost_scenario=cost_scenario,
        floor_fail=floors.failed_by(savings),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cycle-3 additions (docs/PRE-REGISTRATION-CYCLE3-2026-08-31.md §2)
#
# Three metric names, declared in that document and implemented here with NO rung
# attached, so EVAL-LOCK-CYCLE3.json can hash this module *after* they exist rather
# than hard-failing on the very commit that adds them (pre-registration §4).
#
# Nothing above this line is redefined, removed or rescored. These three are inert for
# Rungs 0-4, which stay judged on the cycle-2 lock exactly as before (§3).
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CoverageRow:
    """Realised false-HOLD rate in one Mondrian stratum, against nominal ``alpha``.

    ``realised`` is ``P(HOLD | not a fraud-day)`` *within the stratum*: the denominator is
    the stratum's negative merchant-days, not all of its rows. That is the quantity a
    conformal risk controller claims to bound, so it is the quantity measured.

    ``violated`` is a plain ``realised > alpha``. There is deliberately no finite-sample
    slack, no shrinkage toward alpha and no clamp: pre-registration §5 commits to
    *reporting* a coverage violation, and a tolerance band wide enough to absorb one is
    indistinguishable from suppressing it. A stratum too small to say anything carries its
    own ``n_negatives`` so the reader can discount it themselves.
    """

    stratum: str
    n_negatives: int
    n_false_hold: int
    realised: float
    alpha: float
    violated: bool


def false_hold_coverage(
    action: np.ndarray,
    y: np.ndarray,
    stratum: np.ndarray,
    alpha: float,
) -> list[CoverageRow]:
    """Per-Mondrian-stratum realised false-HOLD rate against nominal ``alpha`` (Rung 6).

    A *false HOLD* is ``action == Action.HOLD`` on a merchant-day whose label is 0 — a
    settlement freeze on a merchant that was not drifting. Conformal risk control's claim
    is that this rate respects ``alpha`` in **every** stratum of the Mondrian taxonomy,
    not merely on average: a marginal guarantee that holds overall while one merchant
    category is frozen at three times alpha is the failure stratification exists to
    expose, so there is no aggregate row here to hide behind.

    Args:
        action: ``Action`` per merchant-day — the same array as ``RungOutput.action``.
        y: 0/1 merchant-day label from :func:`day_labels`, already filtered by ``keep``.
            Censored merchants have no resolved outcome and cannot be scored here.
        stratum: the Mondrian taxonomy value per merchant-day; compared as ``str``.
        alpha: nominal false-HOLD rate in [0, 1], e.g. 0.05.

    Returns:
        One :class:`CoverageRow` per stratum, sorted by stratum name. A stratum with no
        negative days gets ``realised = nan`` and ``violated = False`` — there is no rate
        to violate on a zero denominator, and reporting 0.0 there would be a fabricated
        number of the kind :func:`pr_auc` also refuses to invent.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha is a nominal error rate in [0,1]; got {alpha!r}")
    if not len(action) == len(y) == len(stratum):
        raise ValueError("action, y and stratum must be the same length")

    labels = np.asarray([str(s) for s in np.asarray(stratum).tolist()])
    is_negative = np.asarray(y) == 0
    is_false_hold = is_negative & (np.asarray(action) == Action.HOLD)

    rows: list[CoverageRow] = []
    for value in sorted(set(labels.tolist())):
        member = labels == value
        n_neg = int((member & is_negative).sum())
        n_fh = int((member & is_false_hold).sum())
        realised = n_fh / n_neg if n_neg else float("nan")
        rows.append(
            CoverageRow(
                stratum=value,
                n_negatives=n_neg,
                n_false_hold=n_fh,
                realised=realised,
                alpha=alpha,
                violated=bool(n_neg) and realised > alpha,
            )
        )
    return rows


@dataclass(frozen=True, slots=True)
class OnsetLocalisation:
    """The signed onset-localisation error distribution (Rung 7).

    ``error_days`` is ``estimated - true``, so a **negative** error is an estimate that is
    too early and a positive one is late. The sign is kept because the two mistakes are
    not interchangeable operationally: early attributes clean days to a drift, late writes
    off days of real exposure as normal.

    Median and IQR rather than mean and sd, because a handful of merchants where the
    change-point search lands on the wrong side of the series drags a mean anywhere, and
    the resulting number then describes those merchants rather than the method.

    ``n_unlocalised`` are merchants that genuinely drifted but for which no change-point
    was declared. They are counted, never silently dropped and never imputed: dropping
    them lets a method that only fires on the easy half report a flattering IQR.
    """

    error_days: np.ndarray
    median: float
    q25: float
    q75: float
    iqr: float
    n: int
    n_unlocalised: int


def onset_localisation_error(
    estimated_onset_day: np.ndarray, true_onset_day: np.ndarray
) -> OnsetLocalisation:
    """Signed days between an estimated change-point and ``drift_onset_at`` (Rung 7).

    Both arrays are one row per merchant and index-aligned. ``nan`` in ``true_onset_day``
    means the merchant never drifted, so there is nothing to localise and the row is
    dropped entirely — scoring a change-point against an onset that does not exist
    measures a false positive, which is :func:`precision_recall_at_k`'s job, not this
    one. ``nan`` in ``estimated_onset_day`` on a merchant that *did* drift means the
    estimator declined to localise: dropped from the distribution, counted in
    ``n_unlocalised``.

    Quartiles use numpy's default linear interpolation; ``iqr = q75 - q25``.
    """
    est = np.asarray(estimated_onset_day, dtype=np.float64)
    true = np.asarray(true_onset_day, dtype=np.float64)
    if est.shape != true.shape:
        raise ValueError("estimated_onset_day and true_onset_day must be the same length")

    drifted = ~np.isnan(true)
    localised = drifted & ~np.isnan(est)
    errors = est[localised] - true[localised]
    n_unlocalised = int(drifted.sum() - localised.sum())

    if errors.size == 0:
        nan = float("nan")
        return OnsetLocalisation(
            error_days=errors,
            median=nan,
            q25=nan,
            q75=nan,
            iqr=nan,
            n=0,
            n_unlocalised=n_unlocalised,
        )
    q25, median, q75 = (float(q) for q in np.percentile(errors, (25.0, 50.0, 75.0)))
    return OnsetLocalisation(
        error_days=errors,
        median=median,
        q25=q25,
        q75=q75,
        iqr=q75 - q25,
        n=int(errors.size),
        n_unlocalised=n_unlocalised,
    )


@dataclass(frozen=True, slots=True)
class RescaledKS:
    """Time-rescaling goodness-of-fit for a temporal point process (Rung 8)."""

    statistic: float
    p_value: float
    n: int

    def rejects_at(self, level: float = 0.05) -> bool:
        """``True`` when the fit is rejected, i.e. the intensity is misspecified."""
        return self.p_value < level


def tpp_rescaled_ks(compensator_increments: np.ndarray) -> RescaledKS:
    """KS statistic and p-value on time-rescaled inter-arrival times (Rung 8).

    By the time-rescaling theorem, if a point process's conditional intensity
    ``lambda(t)`` is correct then the compensator increments between consecutive events,
    ``Lambda_k = integral over (t_{k-1}, t_k] of lambda(u) du``, are i.i.d.
    Exponential(1), so ``u_k = 1 - exp(-Lambda_k)`` is i.i.d. Uniform(0, 1). This is a
    one-sample KS test of that uniformity.

    Taking the *increments* rather than event times plus an intensity callable is
    deliberate: the metric then holds no opinion about the rung's parametric form, which
    is what lets it be declared before Rung 8 exists and applied unchanged to whatever
    intensity that rung ends up with.

    A small statistic with a large p-value is a fit that has **not been rejected**, never
    a fit that has been confirmed. Pre-registration §5: if this rejects, Rung 8's
    intensity is misspecified and is reported as such.

    Args:
        compensator_increments: ``Lambda_k >= 0``, one per inter-arrival interval.

    Returns:
        :class:`RescaledKS`. An empty input gives ``nan`` for both, loudly, rather than a
        p-value of 1.0 that would read as a passing fit computed on nothing.
    """
    lam = np.asarray(compensator_increments, dtype=np.float64)
    if lam.size and bool((lam < 0).any()):
        raise ValueError(
            "compensator increments integrate a non-negative intensity and cannot be "
            "negative; a negative one means the intensity or the integration is wrong"
        )
    if lam.size == 0:
        return RescaledKS(statistic=float("nan"), p_value=float("nan"), n=0)

    from scipy import stats  # local: only Rung 8 pays the scipy.stats import

    result = stats.kstest(1.0 - np.exp(-lam), "uniform")
    return RescaledKS(
        statistic=float(result.statistic), p_value=float(result.pvalue), n=int(lam.size)
    )
