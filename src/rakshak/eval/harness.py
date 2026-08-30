"""Eval harness entry point — `make eval`. Writes `results/summary.md`.

The harness is deliberately dumb: it loads a split, asks every registered model
for a per-merchant suspicion score, spends the same review budget on each, and
tabulates. All the judgement lives in `splits.py`, `metrics.py` and `oracle.py`.

**Registering a model (T-0006 and later).** A scorer is::

    def score_x(split: Split, rng: np.random.Generator) -> pd.Series | pd.DataFrame

indexed by `split.merchant_ids`. Return a Series of scores, or a DataFrame with
a `score` column and an optional `flag_day` column (first day the model raised
a flag, NaN if never) — `flag_day` is what makes median detection lag
computable, so time-resolved models should return it. Then::

    MODEL_REGISTRY["rules"] = score_rules

one line per baseline. Names in `EXPECTED_MODELS` that are not in the registry
are reported in the summary as absent rather than silently omitted.

**Determinism (NFR-003).** `results/summary.md` contains no wall-clock time, no
host detail and no dict iteration that depends on anything but insertion order.
Each model gets its own RNG seeded from `(seed, crc32(name))` so that adding a
model does not perturb the others. Wall-clock timing goes to stdout only.

**The test window is not touched here.** The harness evaluates on `validate`
(06-requirements.md §3: all thresholds chosen on the validation window).
T-0011/T-0013 pass `unlock_test=` to render the final verdict.
"""

from __future__ import annotations

import sys
import time
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.cli import base_parser, seed_everything
from rakshak.config import (
    FRAUD_MERCHANT_RATE,
    GENERATOR_START_DATE,
    N_MERCHANTS,
    RESULTS_DIR,
    REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS,
    TAU_REVIEW_HOURS,
)
from rakshak.decision import policy
from rakshak.decision.cost import fp_cost_per_100_of_fraud_loss
from rakshak.eval import metrics
from rakshak.eval.figures import render_sensitivity_figure
from rakshak.eval.oracle import (
    OracleResult,
    perfect_hindsight_oracle,
    review_knapsack_oracle,
    review_slots,
)
from rakshak.eval.splits import (
    BAD_STATES,
    SPLIT_DAY_BOUNDS,
    Split,
    load_split,
    split_summary,
)
from rakshak.models.gbdt import score_gbdt
from rakshak.models.hmm_score import score_hmm
from rakshak.models.rules import score_rules

Scorer = Callable[[Split, np.random.Generator], "pd.Series | pd.DataFrame"]

EVAL_SPLIT: str = "validate"
"""Everything before T-0011 tunes and reports here. See module docstring."""


def score_random(split: Split, rng: np.random.Generator) -> pd.Series:
    """The absolute floor (06-requirements.md §3): uniform random suspicion."""
    return pd.Series(
        rng.random(split.n_merchants), index=pd.Index(split.merchant_ids, name="merchant_id")
    )


MODEL_REGISTRY: dict[str, Scorer] = {
    "random": score_random,
    "rules": score_rules,
    "gbdt": score_gbdt,
    "hmm": score_hmm,
}
"""name -> scorer. Add one line per baseline; see the module docstring."""

EXPECTED_MODELS: tuple[tuple[str, str], ...] = (
    ("random", "T-0005 — absolute floor"),
    ("rules", "T-0006 — static rule engine, the floor that must be beaten"),
    ("gbdt", "T-0006 — LightGBM on windowed aggregates, no HMM"),
    ("bocpd", "T-0006 — changepoint baseline"),
    ("hmm", "T-0006b — the proposal: HMM filtered posterior"),
)
"""The models the frozen eval requires a row for. Anything here but not in
MODEL_REGISTRY is printed as ABSENT, never silently dropped."""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _model_rng(seed: int, name: str) -> np.random.Generator:
    """Per-model RNG so registry membership never perturbs another model."""
    return np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])


def _normalise(output: pd.Series | pd.DataFrame, split: Split) -> pd.DataFrame:
    """Coerce a scorer's return value to a `score` / `flag_day` frame."""
    index = pd.Index(split.merchant_ids, name="merchant_id")
    if isinstance(output, pd.Series):
        frame = output.rename("score").to_frame()
    else:
        frame = output.copy()
    if "score" not in frame.columns:
        raise ValueError("scorer must return a Series or a DataFrame with a 'score' column")
    if "flag_day" not in frame.columns:
        frame["flag_day"] = np.nan
    return frame.reindex(index)[["score", "flag_day"]]


def evaluate_model(name: str, split: Split, seed: int, k: int) -> dict[str, object]:
    """Score one model on `split` and return its summary row.

    Actions come from `decision.policy.bmr_policy` (T-0007b) — Bayes Minimum Risk
    over {PASS, REVIEW, HOLD} under the analyst-hour budget. This replaced
    `budget_policy`, the top-K placeholder that spent the whole budget on the
    highest scores, never held anyone, and therefore penalised a
    well-covering-but-badly-calibrated model twice.

    **The scores are used as posteriors without recalibration.** T-0008
    (empirical-Bayes shrinkage) was cut in the 2026-08-28 re-plan, so BMR
    consumes each model's raw score clipped to [0, 1]. That is a real limitation
    of every savings number this harness produces and it belongs in the README:
    a miscalibrated score moves the argmin, not just the ranking.

    Args:
        name: A key of `MODEL_REGISTRY`.
        split: The split to score on.
        seed: Global seed (NFR-003).
        k: Review budget in merchants; the analyst-hour budget is `k * tau`.

    Returns:
        The summary row, plus a `posterior` entry holding the clipped scores so
        the FR-020 sweep can re-score without re-fitting.
    """
    frame = _normalise(MODEL_REGISTRY[name](split, _model_rng(seed, name)), split)
    y = split.labels.to_numpy(dtype=float)
    scores = frame["score"].to_numpy(dtype=float)
    posterior = np.clip(scores, 0.0, 1.0)
    params = policy.CostParams(
        loss_inr=split.loss_inr.to_numpy(dtype=float),
        value_inr=split.value_inr.to_numpy(dtype=float),
    )
    result = policy.bmr_policy(posterior, params, capacity_hours=k * TAU_REVIEW_HOURS)
    lag, flagged_fraction, _ = metrics.detection_lag_days(
        frame["flag_day"], split.transition_day, split.labels
    )
    return {
        "model": name,
        "savings": metrics.savings_score(
            y, result.actions, split.loss_inr, split.value_inr
        ),
        "pr_auc": metrics.pr_auc(y, scores),
        "precision_at_k": metrics.precision_at_k(y, scores, k),
        "brier": metrics.brier_score(y, posterior),
        "lag_days": lag,
        "flagged_fraction": flagged_fraction,
        "n_reviewed": result.n_reviewed,
        "n_held": result.n_held,
        "hours_used": result.hours_used,
        "binding_constraint": result.binding_constraint,
        "unconstrained_n_reviewed": result.unconstrained_n_reviewed,
        "posterior": posterior,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _f(value: float, places: int = 4) -> str:
    """Fixed-width float formatting. NaN renders as 'n/a' so it never varies."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def _inr(value: float) -> str:
    return f"{value:,.0f}"


def render_summary(
    split: Split,
    rows: list[dict[str, object]],
    oracles: list[OracleResult],
    seed: int,
    k: int,
    capacity_hours: float,
) -> str:
    """Build `results/summary.md`. Byte-identical for a fixed seed (NFR-003)."""
    n_bad = int(split.labels.sum())
    binding = any(o.capacity_binding for o in oracles)
    knapsack = oracles[0]

    lines: list[str] = []
    add = lines.append

    add("# Rakshak — evaluation summary")
    add("")
    add(
        "> **Sequence-layer metrics are measured on synthetic merchant streams with injected "
        "typologies; the generator is in this repo.** The decision layer is additionally "
        "validated on BAF (Feedzai, NeurIPS 2022), a public benchmark derived from real bank "
        "data."
    )
    add("")
    add(
        "The BAF half of that sentence is backed by `results/baf_validation.md` (T-0012), on "
        "BAF's own native temporal split. **It validates the decision layer only** — BAF is "
        "account-opening applications with no sequences, so the HMM cannot run there and does "
        "not. Nothing below is measured on BAF; everything below is the synthetic split."
    )
    add("")
    add(
        "**And it validates it in a different cost regime from this one.** BAF's only monetary "
        "column is a proposed credit limit of 190-2000 in its own units, against absolute INR "
        "support and review costs, so its asymmetry runs 5,497-519,634 and **never reaches the "
        "47.5 reported below**. On BAF the correct policy is to hold almost nobody, and the "
        "decision layer correctly does. What is validated there is that BMR takes the right "
        "action when false positives dominate — not the review-versus-hold trade-off at this "
        "split's asymmetry."
    )
    add("")

    add("## Provenance")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(f"| Produced by | `python -m rakshak.eval.harness --seed {seed}` |")
    add(f"| Seed | {seed} |")
    add(f"| Split reported | `{EVAL_SPLIT}` (days {split.start_day}-{split.end_day - 1}) |")
    add(f"| Data horizon | day 0 = {GENERATOR_START_DATE}, {N_MERCHANTS} merchants |")
    add(
        f"| Review budget K | {k} merchants ({capacity_hours:.2f} h / "
        f"{TAU_REVIEW_HOURS:.3f} h per review) |"
    )
    add(
        f"| Capacity rule | {REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS:.1f} analyst-hours per "
        f"1000 merchants under watch, scaled to this split's {split.n_merchants} merchants "
        "(ADR-0008) |"
    )
    add(f"| Bad states | {', '.join(sorted(BAD_STATES))} |")
    add("")
    add(
        "The test window (days 210-269) is **not** touched by this run. "
        "06-requirements.md §3 reserves it for T-0011/T-0013."
    )
    add("")

    add("## Prevalence — read every precision number against this")
    add("")
    add(
        f"- **{n_bad} of {split.n_merchants} merchants in this split are truly bad "
        f"({split.prevalence:.1%}).**"
    )
    add(
        f"- The generator's `FRAUD_MERCHANT_RATE = {FRAUD_MERCHANT_RATE:.2f}` is far above "
        "real-world merchant-fraud prevalence. It was chosen so each of the five typologies "
        "has enough merchants for a per-class metric, not because it is realistic."
    )
    add(
        "- A precision of P here corresponds to a lift of "
        f"P / {split.prevalence:.2f} over random selection. Quote the lift, not the "
        "precision, when comparing against a real base rate."
    )
    add("")

    add("## Splits — merchant counts (NFR-002: disjoint merchant IDs, enforced in code)")
    add("")
    table = split_summary()
    header = list(SPLIT_DAY_BOUNDS)
    add("| typology | " + " | ".join(header) + " |")
    add("|---|" + "---|" * len(header))
    for typology, row in table.iterrows():
        add(f"| {typology} | " + " | ".join(str(int(row[c])) for c in header) + " |")
    add("")
    for name, (start, end) in SPLIT_DAY_BOUNDS.items():
        add(f"- `{name}`: days {start}-{end - 1} ({(end - start) // 30} month(s))")
    add("")

    add("## Ceilings — perfect foresight")
    add("")
    add("| ceiling | reviewed | held | hours used | loss averted (INR) | savings |")
    add("|---|---|---|---|---|---|")
    for o in oracles:
        add(
            f"| {o.name} | {o.n_reviewed} | {o.n_held} | {o.hours_used:.2f} | "
            f"{_inr(o.loss_averted_inr)} | {_f(o.savings)} |"
        )
    add("")
    if not binding:
        add(
            f"> **The review budget is not binding on this split.** K = {k} review slots "
            f"against {split.n_merchants} merchants ({n_bad} of them bad), so the capacity "
            "constraint that motivates the whole decision layer does nothing here. "
            "`precision@K` degenerates towards prevalence and gap-to-oracle is not a "
            "capacity story. That is a configuration problem, not a result — ADR-0008."
        )
    else:
        add(
            f"> **The review budget binds.** K = {k} review slots "
            f"({capacity_hours:.2f} analyst-hours at {TAU_REVIEW_HOURS:.3f} h per review) "
            f"against {split.n_merchants} merchants, {n_bad} of them truly bad. Even with "
            f"perfect foresight the knapsack oracle can only reach {knapsack.n_reviewed} of "
            f"them, leaving {n_bad - knapsack.n_reviewed} bad merchants unreviewed for want "
            f"of analyst hours. K covers {k / split.n_merchants:.0%} of the book against a "
            f"{split.prevalence:.0%} prevalence, so `precision@{k}` has real headroom and "
            "the baselines can separate on it. Capacity is now expressed per 1000 "
            "merchants (ADR-0008, T-0003b); before that it was an absolute 40 h = 597 "
            "slots and bound nothing."
        )
    add("")

    add("## Cost-matrix cross-check (07-math.md §5, as amended by T-0017)")
    add("")
    add(
        "> Indian payments commentary estimates INR 400-600 lost to falsely declined "
        "legitimate orders for every INR 100 saved by preventing fraud."
    )
    add("")
    ratio, fp_cost_total, fraud_loss_total = fp_cost_per_100_of_fraud_loss(
        split.labels.to_numpy(dtype=bool), split.loss_inr, split.value_inr
    )
    add("| quantity | value |")
    add("|---|---|")
    add(f"| Total false-positive cost, all healthy merchants held (INR) | {_inr(fp_cost_total)} |")
    add(f"| Total fraud loss, all bad merchants passed (INR) | {_inr(fraud_loss_total)} |")
    add(f"| INR of FP cost per INR 100 of fraud loss | {_f(ratio, 1)} |")
    add("| 07-math.md §5 cross-check (commentary, not a gate) | 400 - 600 |")
    add(f"| Divergence from the band | {_f(ratio, 1)} vs 400-600 — reported, not closed |")
    add("")
    add(
        "**T-0017 demoted this row from a gate to a cross-check, and T-0007a corrected "
        "the two definitions underneath it.** `V_m` is now expected *lifetime* gross "
        "margin (`g * v_m * l_m`, with `g` the platform's own ~10 bps of TPV, not the "
        "merchant-facing 2% MDR — a price is not a margin), and `L_m` is *realised* loss "
        "(`r_cb * (1 + phi) * G_bad`), not gross turnover during a bad state. Turnover is "
        "not loss: a bust-out processing INR 10,00,000 with INR 50,000 charged back cost "
        "the acquirer INR 50,000 plus fees. The previous definitions were wrong by roughly "
        "15x on `L_m` and 3x net on `V_m`, in opposite directions — they partly cancelled, "
        "which is why no sanity check on this ratio alone could ever have found them. "
        "Every constant carries a citation or an explicit ASSUMPTION tag and a range in "
        "`config.py`. **The divergence from 400-600 is stated, not tuned away**: the "
        "commentary band measures declined baskets at checkout, this ratio measures held "
        "settlements costing the platform its own margin. They were never the same "
        "asymmetry, and closing the gap by choosing constants that land in the band is "
        "the practice T-0016 forbids for the generator."
    )
    add("")

    add("## Models")
    add("")
    add(
        "All rows share the same analyst-hour budget. Actions come from the "
        "cost-optimal three-action policy in `decision/policy.py` (T-0007b): Bayes "
        "Minimum Risk over {PASS, REVIEW, HOLD} under the cost matrix, then the "
        "capacity constraint. It replaced `harness.budget_policy`, the top-K "
        "placeholder that never held anyone."
    )
    add("")
    add(
        "**No verdict is rendered here (T-0006 is plumbing).** These are the baseline "
        "rows only; the comparison that decides anything happens at T-0011 on the test "
        "window, with the sequence layer present."
    )
    add("")
    add(
        "**`gbdt` caveat.** LightGBM early-stops its iteration count on this same "
        "`validate` split, as 06-requirements.md §3 directs (\"all hyperparameters and "
        "thresholds chosen on the validation window\"). Its row here is therefore mildly "
        "optimistic while the harness reports validate; it is clean at T-0011, where the "
        "reported window is `test` and validate is only the early-stopping set. `rules` "
        "has no fitted quantity at all and `random` has none either, so neither carries "
        "this caveat."
    )
    add("")
    add(
        "**`hmm` is the proposal.** The row comes from `models/hmm_score.py` (T-0006b): "
        "a per-merchant belief over four latent states, fitted on the training split "
        "alone with T-0004b's shipping configuration, scored by the **forward-only "
        "filtered posterior** so that neither `score` nor `flag_day` uses information "
        "from after the window it reports on. A truncation test proves this, and carries "
        "a negative control that runs the same assertion against the smoothed posterior "
        "and requires it to fail, so the proof cannot be vacuous."
    )
    add("")
    add(
        "**`savings` became readable at T-0007a** and these are the first cost numbers "
        "in the project worth reading. Two caveats bind them. First, this is the "
        "`validate` split — the `test` window is reserved (06-requirements.md §3) and "
        "**no verdict is rendered here**; K2 renders at T-0011. Second, T-0007b's BMR "
        "policy consumes each model's raw score as a posterior — **T-0008's "
        "empirical-Bayes calibration was cut in the 2026-08-28 re-plan and no "
        "recalibration happens anywhere in this repo.** A miscalibrated score moves "
        "the argmin, not merely the ranking, so `savings` and `Brier` are coupled here "
        "in a way they would not be in a calibrated system. Read PR-AUC, precision@K, "
        "Brier and median lag alongside `savings`, not instead of it. The sensitivity "
        "of every `savings` figure to the cost asymmetry is in `results/sensitivity.md` "
        "(FR-020)."
    )
    add("")
    add(
        "**The median-lag column was wrong until T-0011 and is now corrected.** It "
        "previously read -1.0 for `gbdt` and `hmm`, which looked like detection before "
        "onset. It was neither leakage nor early warning: those two scorers attributed a "
        "flag to the *start* day of the 7-day window that raised it, while `rules` has "
        "always reported the last day of its own trailing evidence — so the column "
        "compared two conventions without saying so. A window straddling onset holds up "
        "to 6 post-onset days, which is exactly the earliness the old number invented. "
        "Both window-based scorers now attribute the flag to the day their evidence was "
        "complete (`models/gbdt.py`, `models/hmm_score.first_flag_day`), and `rules` was "
        "left alone because shifting it would double-count. `results/lag_probe.md` shows "
        "both numbers side by side, clears the leakage question with a merchant-clustered "
        "permutation test, and records that the correction reverses the reading: `gbdt` "
        "is *later* than `rules` on this split, not earlier. Read the column against the "
        "quantisation too — `validate` admits only four distinct flag days."
    )
    add("")
    add(
        "| model | savings | gap to knapsack oracle | gap to hindsight oracle | PR-AUC | "
        f"precision@{k} | Brier | median lag (days) | flagged frac | reviewed | held | "
        "hours | capacity binds |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        savings = float(row["savings"])
        add(
            f"| {row['model']} | {_f(savings)} | "
            f"{_f(metrics.gap_to_oracle(savings, oracles[0].savings))} | "
            f"{_f(metrics.gap_to_oracle(savings, oracles[1].savings))} | "
            f"{_f(float(row['pr_auc']))} | {_f(float(row['precision_at_k']))} | "
            f"{_f(float(row['brier']))} | {_f(float(row['lag_days']), 1)} | "
            f"{_f(float(row['flagged_fraction']), 2)} | {row['n_reviewed']} | "
            f"{row['n_held']} | {float(row['hours_used']):.2f} | "
            f"{row['binding_constraint']} "
            f"(wanted {row['unconstrained_n_reviewed']}) |"
        )
    add("")
    add(
        "**`gap to knapsack oracle` is now a category error and is printed only "
        "because deleting it would hide the reason.** The review-knapsack ceiling is "
        "the best *review-only, <= K* allocation; T-0007b's policy may HOLD, and "
        "nothing bounds a holding policy by a review-only ceiling. T-0007a wrote this "
        "down before it bit (`tests/test_cost.py` header: *\"nothing forces it above "
        "hold-everything\"*). Read `gap to hindsight oracle` instead, and read "
        "`results/sensitivity.md` for the asymmetry below which the knapsack ceiling "
        "stops clearing hold-everything altogether."
    )
    add("")
    add(
        "`capacity binds` reports FR-017's binding constraint per model, with the "
        "number of reviews unconstrained BMR *wanted* beside it. A model whose "
        "unconstrained demand is below K is not being limited by analyst hours at all, "
        "and its row must not be read as a capacity result."
    )
    add("")
    add(
        "### The `random` row is the most important number in this table"
    )
    add("")
    random_savings = next(
        (float(r["savings"]) for r in rows if r["model"] == "random"), float("nan")
    )
    rules_savings = next(
        (float(r["savings"]) for r in rows if r["model"] == "rules"), float("nan")
    )
    random_pr_auc = next(
        (float(r["pr_auc"]) for r in rows if r["model"] == "random"), float("nan")
    )
    add(
        f"**Under the BMR policy, `random` scores {_f(random_savings)} savings against "
        f"`rules`' {_f(rules_savings)} — a gap of "
        f"{_f(abs(rules_savings - random_savings))} — while ranking at PR-AUC "
        f"{_f(random_pr_auc)}, i.e. at this split's prevalence.** Nothing about the "
        "model produced that; the cost matrix did. "
        "A uniform random score still lands most merchants on the correct side of a "
        "merchant-specific threshold when `c_fp` is small relative to `L_m`, so most "
        "of the savings on this split is attributable to the decision layer's cost "
        "arithmetic rather than to detection. This is 07-math.md §6's AP-06 guard "
        "arriving as a measurement rather than as a warning: **the savings score is "
        "manipulable through the cost matrix and must never be quoted without PR-AUC "
        "beside it.** Any headline of the form \"Rakshak saves X%\" that does not "
        "subtract the random floor is not a claim about the model. T-0011 must report "
        "savings *relative to the `random` row*, not in absolute terms."
    )
    add("")
    add(
        "This also changed the ordering. Under T-0006's top-K placeholder the HMM sat "
        "below both baselines on savings; under BMR it sits above them. STATE.md "
        "predicted the mechanism before the policy existed — a well-covering but "
        "badly-calibrated model was penalised twice by a rank-only policy. **That is "
        "an explanation, not a verdict.** The verdict is T-0011's, on the test window, "
        "and the `random` row above says how much of any margin is the cost matrix."
    )
    add("")

    absent = [(n, why) for n, why in EXPECTED_MODELS if n not in MODEL_REGISTRY]
    add("### Models absent from this run")
    add("")
    if not absent:
        add("None — every model the frozen eval requires produced a row.")
    else:
        add("| model | status |")
        add("|---|---|")
        for name, why in absent:
            add(f"| {name} | **ABSENT** — {why} |")
        add("")
        add(
            "These rows are missing, not zero. No headline claim can be made from this "
            "run until they land."
        )
    add("")

    add("## Metrics deliberately not reported")
    add("")
    add(
        "**ROC-AUC and raw accuracy are prohibited as headline metrics** "
        "(06-requirements.md §3) and are not implemented in `rakshak.eval.metrics`. "
        f"At {split.prevalence:.0%} prevalence ROC-AUC flatters every model and "
        "\"predict healthy\" beats most models on accuracy."
    )
    add("")
    add(
        "Median detection lag reads `n/a` for any model that does not return a "
        "`flag_day`; a single per-merchant score has no time at which it fired."
    )
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(seed: int, results_dir: Path = RESULTS_DIR) -> Path:
    """Run the evaluation, write `summary.md` and `sensitivity.md`.

    Returns the path to `summary.md`. Both artifacts come from one scoring pass:
    the FR-020 sweep re-uses the posteriors `evaluate_model` already computed, so
    adding it costs no model fit and leaves NFR-004's 15-minute budget alone.
    """
    split = load_split(EVAL_SPLIT)
    # ADR-0008: capacity scales with the population being scored, not with an absolute
    # figure that happens to exceed every split we own.
    capacity_hours = REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS * split.n_merchants / 1000.0
    k = min(review_slots(capacity_hours), split.n_merchants)
    y = split.labels.to_numpy(dtype=float)

    oracles = [
        review_knapsack_oracle(
            y, split.loss_inr, split.value_inr, capacity_hours=capacity_hours
        ),
        perfect_hindsight_oracle(y, split.loss_inr, split.value_inr),
    ]
    rows = [evaluate_model(name, split, seed, k) for name in MODEL_REGISTRY]
    posteriors = {str(r["model"]): r.pop("posterior") for r in rows}
    params = policy.CostParams(
        loss_inr=split.loss_inr.to_numpy(dtype=float),
        value_inr=split.value_inr.to_numpy(dtype=float),
    )

    # T-0007a's invariant, scoped to the action class each ceiling bounds (T-0007b).
    # Fires before anything is written. See `policy.assert_ceilings_dominate`: the
    # scored policy now HOLDs, and the review-knapsack ceiling never bounded a
    # holding policy — T-0007a's own test header said so.
    policy.assert_ceilings_dominate(
        y,
        params,
        {o.name: o.savings for o in oracles},
        {str(r["model"]): float(r["savings"]) for r in rows},
        seed=seed,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "summary.md"
    path.write_text(
        render_summary(split, rows, oracles, seed, k, capacity_hours), encoding="utf-8",
        newline="\n"
    )

    frame = policy.sweep_cost_asymmetry(y, posteriors, params, capacity_hours, seed=seed)
    (results_dir / "sensitivity.md").write_text(
        policy.render_sensitivity(frame, y, params, capacity_hours, seed),
        encoding="utf-8",
        newline="\n",
    )
    # The CSV is what the figure is drawn from, so --figures-only can redraw
    # without refitting a model and the figure can never disagree with the table.
    (results_dir / "sensitivity.csv").write_text(
        frame.to_csv(index=False), encoding="utf-8", newline=""
    )
    render_sensitivity_figure(frame, results_dir / "figures" / "sensitivity.png")
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the full evaluation. Returns a process exit code."""
    parser = base_parser("Run the Rakshak evaluation harness.")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures from existing results without re-running models.",
    )
    args = parser.parse_args(argv)
    seed_everything(args.seed)

    if args.figures_only:
        # FR-020 figure. Redrawn from the committed sweep CSV; no model is refit.
        csv_path = RESULTS_DIR / "sensitivity.csv"
        if not csv_path.exists():
            print(f"rakshak: {csv_path} not found - run `make eval` first.")
            return 1
        out = render_sensitivity_figure(
            pd.read_csv(csv_path), RESULTS_DIR / "figures" / "sensitivity.png"
        )
        print(f"rakshak: wrote {out}")
        return 0

    started = time.perf_counter()
    path = run(args.seed)
    elapsed = time.perf_counter() - started

    absent = [n for n, _ in EXPECTED_MODELS if n not in MODEL_REGISTRY]
    print(f"rakshak: wrote {path} (seed={args.seed}) in {elapsed:.1f}s")
    print(f"rakshak: models run: {', '.join(MODEL_REGISTRY)}")
    if absent:
        print(f"rakshak: models ABSENT: {', '.join(absent)}")
    print(
        "rakshak: savings is READABLE as of T-0007a (corrected L_m and V_m; the "
        "oracle-dominance invariant passed as a precondition to this run). This is the "
        "validate split and no verdict is rendered here - K2 renders at T-0011 on test."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
