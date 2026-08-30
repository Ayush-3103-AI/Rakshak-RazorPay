"""K2's verdict, rendered once on the test window (T-0011).

`results/summary.md` is the `validate` window and says, everywhere, that it renders
no verdict. This module is where the verdict is rendered: it opens the **test**
window — the only place besides T-0013 permitted to (`06-requirements.md` §3,
`eval.splits._TEST_UNLOCK_TICKETS`) — scores the same registry the harness scores,
runs FR-020's cost-asymmetry sweep there, and writes `results/verdict.md`.

**It re-implements nothing.** Scoring is `harness.evaluate_model`, the ceilings are
`eval.oracle`, the policy and the sweep are `decision.policy`, the figure is
`eval.figures`. The only thing this module owns is the *statement* of the result:
`00-charter.md` §2's >=20% relative margin over `rules` at the cited central cost
asymmetry, and `00-charter.md` §3's kill criterion K2 written out in plain words,
whichever way it comes out.

**Three rules bind the prose below and are worth stating here.**

1. `CLAUDE.md` non-negotiable 1: *"If a baseline beats the HMM, report that the
   baseline beat the HMM."* Nothing here is tuned, narrowed or re-run to improve a
   number. The sweep ships every point it measured.
2. `07-math.md` §6's AP-06 guard, which `results/summary.md` turned from a warning
   into a measurement: the savings score is manipulable through the cost matrix, so
   **every savings figure in this file is printed against the `random` floor and
   with PR-AUC beside it.**
3. NFR-003: the file contains no wall-clock time and no host detail, and is
   byte-identical for a fixed seed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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

# `_f` and `_inr` are the harness's number formatters. Imported rather than copied so
# that a cell in `verdict.md` and the same cell in `summary.md` cannot drift apart —
# `decision.policy.run` reaches into the harness the same way.
from rakshak.eval.harness import (
    EXPECTED_MODELS,
    MODEL_REGISTRY,
    _f,
    _inr,
    evaluate_model,
)
from rakshak.eval.oracle import (
    OracleResult,
    perfect_hindsight_oracle,
    review_knapsack_oracle,
    review_slots,
)
from rakshak.eval.splits import BAD_STATES, Split, load_split

VERDICT_SPLIT: str = "test"
"""The held-out window. Touched here, once, and by T-0013's README run."""

UNLOCK_TICKET: str = "T-0011"
"""The ticket ID `eval.splits.load_split` requires before it opens `test`."""

MARGIN_BAR: float = 0.20
"""`00-charter.md` §2 / NFR-001: >=20% relative savings improvement over `rules`."""

FLOOR_MODEL: str = "random"
REFERENCE_MODEL: str = "rules"
PROPOSAL_MODEL: str = "hmm"

K2_VERDICT_PREFIX: str = "K2 VERDICT:"
"""The rendered document carries exactly one line starting with this. Pinned by a
test so the verdict can never quietly disappear from the file."""

__all__ = [
    "K2_VERDICT_PREFIX",
    "MARGIN_BAR",
    "UNLOCK_TICKET",
    "VERDICT_SPLIT",
    "K2Verdict",
    "assess_k2",
    "main",
    "render_verdict",
    "run",
]


# ---------------------------------------------------------------------------
# The verdict, as a computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class K2Verdict:
    """`00-charter.md` §2's claim, evaluated. All margins are dimensionless.

    Attributes:
        margin_abs: `hmm` savings minus `rules` savings at the central asymmetry.
        margin_rel: The same margin divided by `abs(rules savings)` — the quantity
            NFR-001's >=20% bar is stated in. NaN if `rules` sits at zero.
        holds_at_central: Whether `margin_rel` clears `MARGIN_BAR` at the cited
            central asymmetry, which is what the charter claim is *about*.
        holds_everywhere: Whether it clears the bar at every swept asymmetry.
        fails_everywhere: Whether it clears the bar at no swept asymmetry.
        boundary_fails_at: Highest swept asymmetry at which the >=20% claim does
            NOT hold, NaN if it holds at every point. Units: INR of false-positive
            cost per INR 100 of realised fraud loss.
        boundary_holds_at: Lowest swept asymmetry at which it does hold, NaN if it
            never does. Same units. The boundary lies between the two.
        contiguous_above: True when the claim holds at every point at or above
            `boundary_holds_at`. False means the crossing is not a single
            boundary and the range must be read point by point.
        zero_fails_at: Highest swept asymmetry at which `margin_abs` is not
            positive, NaN if it is positive everywhere. Same units.
        zero_holds_at: Lowest swept asymmetry at which `margin_abs` is positive.
        word: "PASS", "CONDITIONAL PASS" or "FAIL".
    """

    margin_abs: float
    margin_rel: float
    holds_at_central: bool
    holds_everywhere: bool
    fails_everywhere: bool
    boundary_fails_at: float
    boundary_holds_at: float
    contiguous_above: bool
    zero_fails_at: float
    zero_holds_at: float
    word: str


def _bracket(asymmetry: np.ndarray, holds: np.ndarray) -> tuple[float, float, bool]:
    """Bracket the crossing of a boolean condition over a sorted asymmetry grid.

    Args:
        asymmetry: Swept asymmetries, ascending. Units: INR FP cost per INR 100 loss.
        holds: Whether the condition holds at each point.

    Returns:
        `(highest_point_where_it_fails, lowest_point_where_it_holds, contiguous)`.
        Either bound is NaN when the condition is uniform over the grid.
        `contiguous` is True when the condition holds at every point at or above
        the lowest holding point — i.e. when a single boundary describes the sweep.
    """
    if not holds.any():
        return float(asymmetry.max()), float("nan"), False
    if holds.all():
        return float("nan"), float(asymmetry.min()), True
    first = int(np.argmax(holds))
    below = asymmetry[:first]
    return (
        float(below.max()) if below.size else float("nan"),
        float(asymmetry[first]),
        bool(holds[first:].all()),
    )


def assess_k2(rows: list[dict[str, object]], sweep: pd.DataFrame) -> K2Verdict:
    """Evaluate `00-charter.md` §2 at the central asymmetry and across the sweep.

    Args:
        rows: Summary rows from `harness.evaluate_model`, at the shipping (central)
            cost primitives.
        sweep: The FR-020 sweep frame from `policy.sweep_cost_asymmetry`.

    Returns:
        A populated `K2Verdict`. No tuning knob and no tolerance — the bar is
        `MARGIN_BAR` exactly, as pre-registered on 2026-08-28 (T-0017).
    """
    savings = {str(r["model"]): float(r["savings"]) for r in rows}
    margin_abs = savings[PROPOSAL_MODEL] - savings[REFERENCE_MODEL]
    reference = savings[REFERENCE_MODEL]
    margin_rel = margin_abs / abs(reference) if abs(reference) > 1e-6 else float("nan")

    points = sweep.drop_duplicates("asymmetry").sort_values("asymmetry")
    asymmetry = points["asymmetry"].to_numpy(dtype=float)
    rel = points["margin_rel"].to_numpy(dtype=float)
    absolute = points["margin_abs"].to_numpy(dtype=float)
    # NaN margin_rel means the `rules` denominator was ~0 at that point; treat it as
    # "the bar is not demonstrated here" rather than silently as a pass.
    holds = np.nan_to_num(rel, nan=-np.inf) >= MARGIN_BAR
    fails_at, holds_at, contiguous = _bracket(asymmetry, holds)
    zero_fails_at, zero_holds_at, _ = _bracket(asymmetry, absolute > 0.0)

    holds_at_central = bool(np.isfinite(margin_rel) and margin_rel >= MARGIN_BAR)
    word = (
        "FAIL"
        if not holds_at_central
        else "PASS"
        if bool(holds.all())
        else "CONDITIONAL PASS"
    )
    return K2Verdict(
        margin_abs=margin_abs,
        margin_rel=margin_rel,
        holds_at_central=holds_at_central,
        holds_everywhere=bool(holds.all()),
        fails_everywhere=not bool(holds.any()),
        boundary_fails_at=fails_at,
        boundary_holds_at=holds_at,
        contiguous_above=contiguous,
        zero_fails_at=zero_fails_at,
        zero_holds_at=zero_holds_at,
        word=word,
    )


# ---------------------------------------------------------------------------
# FR-019 — the operational vocabulary
# ---------------------------------------------------------------------------


def operational_row(
    row: dict[str, object], baseline_cost_inr: float, floor_savings: float, n_merchants: int
) -> dict[str, float]:
    """Restate one model row in operational units (FR-019).

    Every INR figure comes from the savings score's own denominator — the Bahnsen
    `Cost_l = min(all-PASS, all-HOLD)` from `eval.metrics.baseline_cost` — so this
    is a re-expression of the number already in the table, not a second cost path:
    `savings = (Cost_l - Cost(f)) / Cost_l`, therefore `Cost_l - Cost(f) =
    savings * Cost_l`.

    Args:
        row: A `harness.evaluate_model` row.
        baseline_cost_inr: Cost_l on this split. Units: INR.
        floor_savings: The `random` model's savings on this split. Dimensionless.
        n_merchants: Merchants in the split, for the per-1000 rate.

    Returns:
        Dict with `inr_saved_vs_baseline`, `inr_saved_vs_random` (both INR),
        `analyst_hours` (hours) and `held_per_1000` (merchants per 1000 merchants).
    """
    savings = float(row["savings"])
    return {
        "inr_saved_vs_baseline": savings * baseline_cost_inr,
        "inr_saved_vs_random": (savings - floor_savings) * baseline_cost_inr,
        "analyst_hours": float(row["hours_used"]),
        "held_per_1000": 1000.0 * float(row["n_held"]) / n_merchants,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_verdict(
    split: Split,
    rows: list[dict[str, object]],
    oracles: list[OracleResult],
    sweep: pd.DataFrame,
    params: policy.CostParams,
    verdict: K2Verdict,
    seed: int,
    k: int,
    capacity_hours: float,
) -> str:
    """Build `results/verdict.md`. Byte-identical for a fixed seed (NFR-003).

    Args:
        split: The `test` split, already unlocked.
        rows: `harness.evaluate_model` rows at the central cost primitives.
        oracles: `[review_knapsack_oracle, perfect_hindsight_oracle]`, in that order.
        sweep: The FR-020 sweep frame, the same one written to
            `results/sensitivity_test.csv` and drawn as the figure.
        params: Central cost primitives for this split.
        verdict: The K2 assessment.
        seed: Global seed.
        k: Review budget in merchants.
        capacity_hours: B. Units: hours.

    Returns:
        The document, as text.
    """
    y = split.labels.to_numpy(dtype=float)
    n_bad = int(split.labels.sum())
    by_model = {str(r["model"]): r for r in rows}
    savings_of = {name: float(r["savings"]) for name, r in by_model.items()}
    pr_auc_of = {name: float(r["pr_auc"]) for name, r in by_model.items()}
    floor = savings_of[FLOOR_MODEL]
    baseline_cost_inr = metrics.baseline_cost(y, split.loss_inr, split.value_inr)
    low, central, high = policy.asymmetry_range(y, params)
    ratio, fp_cost_total, fraud_loss_total = fp_cost_per_100_of_fraud_loss(
        y.astype(bool), split.loss_inr, split.value_inr
    )
    points = sweep.drop_duplicates("asymmetry").sort_values("asymmetry")
    models = list(dict.fromkeys(sweep["model"]))

    lines: list[str] = []
    add = lines.append

    add("# Rakshak — K2's verdict on the held-out test window (T-0011)")
    add("")
    add(
        "> **Sequence-layer metrics are measured on synthetic merchant streams with "
        "injected typologies; the generator is in this repo.** The decision layer is "
        "additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark "
        "derived from real bank data."
    )
    add("")
    add(
        "The BAF half of that sentence is backed by `results/baf_validation.md` (T-0012), "
        "on BAF's own native temporal split. **It validates the decision layer only** — "
        "BAF is account-opening applications with no sequences, so the HMM cannot run "
        "there and does not. Every number in this file is the synthetic split."
    )
    add("")
    add("## This is the test window, and it is touched here")
    add("")
    add(
        f"`06-requirements.md` §3 reserves days {split.start_day}-{split.end_day - 1} for "
        "the tickets that render the final result: *\"test set touched — exactly once, at "
        "the end\"*. That reservation is enforced in code — `eval.splits.load_split` "
        f"refuses `test` without an unlock ticket — and this run passes "
        f"`unlock_test=\"{UNLOCK_TICKET}\"`. Every threshold, hyperparameter and "
        "configuration decision in this repo was made on `train` and `validate` before "
        "this file existed, and **nothing was changed after reading it**."
    )
    add("")
    add(
        "**One row is cleaner here than it was in `results/summary.md`.** LightGBM "
        "early-stops its iteration count on `validate`, which is the split `summary.md` "
        "reports on — so its row there is mildly optimistic and says so. Here the "
        "reported window is `test` and `validate` is only the early-stopping set, so "
        "`gbdt`'s row carries no such caveat. `rules` and `random` fit nothing and never "
        "carried it. The `hmm` is fitted on `train` alone (T-0006b)."
    )
    add("")

    add("## Provenance")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(f"| Produced by | `python -m rakshak.eval.verdict --seed {seed}` |")
    add(f"| Seed | {seed} |")
    add(
        f"| Split reported | `{VERDICT_SPLIT}` (days {split.start_day}-{split.end_day - 1}), "
        f"unlocked with ticket {UNLOCK_TICKET} |"
    )
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
    add(
        f"| Prevalence | {n_bad} of {split.n_merchants} merchants truly bad "
        f"({split.prevalence:.1%}); generator `FRAUD_MERCHANT_RATE` = "
        f"{FRAUD_MERCHANT_RATE:.2f} |"
    )
    add(f"| Cited central asymmetry | {central:.1f} INR FP cost per INR 100 of loss |")
    add(f"| Swept asymmetry range | {low:.1f} - {high:.1f} (derived, not chosen) |")
    add("")
    add(
        f"Read every precision-like number against the prevalence: a precision of P is a "
        f"lift of P / {split.prevalence:.2f} over random selection at this base rate, and "
        "this base rate is far above a real merchant book's."
    )
    add("")
    add(
        f"**The cited central asymmetry reads {central:.1f} here against 47.5 on "
        "`validate`, and that is a property of the population rather than a constant.** "
        "`L_m` is a *stock* — realised loss accumulated over every bad-state transaction "
        "in the merchant's history before the window ends — while T-0007a deliberately "
        "made `V_m` a *rate-derived* figure (monthly volume x margin x lifetime) so it "
        "would stop growing with how many days a split had loaded. The test window loads "
        "60 more days than `validate`, so the denominator grows and the ratio falls. "
        "Nothing was tuned; the number is simply not comparable across splits and is "
        "recorded here so nobody reads the move as a result. The sweep below spans "
        f"{low:.1f} - {high:.1f} and contains `validate`'s 47.5 comfortably, so the "
        "verdict does not turn on which of the two is quoted."
    )
    add("")

    add("## Ceilings — perfect foresight")
    add("")
    add("| ceiling | reviewed | held | hours used | loss averted (INR) | savings |")
    add("|---|---|---|---|---|---|")
    for oracle in oracles:
        add(
            f"| {oracle.name} | {oracle.n_reviewed} | {oracle.n_held} | "
            f"{oracle.hours_used:.2f} | {_inr(oracle.loss_averted_inr)} | "
            f"{_f(oracle.savings)} |"
        )
    add("")

    add(f"## The models on `{VERDICT_SPLIT}`")
    add("")
    add(
        "All rows share the same analyst-hour budget. Actions come from the three-action "
        "Bayes-Minimum-Risk policy in `decision/policy.py` under the capacity constraint. "
        "**`savings` is never the whole story on this split — read it beside PR-AUC and "
        "beside the `random` row, for the reason measured in the next section but one.**"
    )
    add("")
    add(
        f"| model | savings | savings - `{FLOOR_MODEL}` | gap to hindsight oracle | PR-AUC | "
        f"precision@{k} | Brier | median lag (days) | flagged frac | reviewed | held | "
        "hours | capacity binds |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        savings = float(row["savings"])
        add(
            f"| {row['model']} | {_f(savings)} | {_f(savings - floor)} | "
            f"{_f(metrics.gap_to_oracle(savings, oracles[1].savings))} | "
            f"{_f(float(row['pr_auc']))} | {_f(float(row['precision_at_k']))} | "
            f"{_f(float(row['brier']))} | {_f(float(row['lag_days']), 1)} | "
            f"{_f(float(row['flagged_fraction']), 2)} | {row['n_reviewed']} | "
            f"{row['n_held']} | {float(row['hours_used']):.2f} | "
            f"{row['binding_constraint']} (wanted {row['unconstrained_n_reviewed']}) |"
        )
    add("")
    slack = [
        str(r["model"]) for r in rows if str(r["binding_constraint"]) != "capacity"
    ]
    if slack:
        add(
            "**The analyst-hour budget does not bind for "
            + ", ".join(f"`{name}`" for name in slack)
            + " on this window** — unconstrained BMR asked for fewer reviews than K "
            "there, so those cells are not a capacity result and must not be read as one "
            "(FR-017). "
            "The binding constraint is reported per model rather than inferred precisely "
            "so that a run where the budget did nothing and a run where it forced a "
            "downgrade do not look alike."
        )
        add("")
    add(
        f"**Where `{PROPOSAL_MODEL}` stands relative to `{REFERENCE_MODEL}` and `gbdt` "
        "differs by metric, and the disagreement is the finding.** On this window "
        f"`{PROPOSAL_MODEL}` ranks at PR-AUC {_f(pr_auc_of[PROPOSAL_MODEL])} against "
        f"`gbdt`'s {_f(pr_auc_of.get('gbdt', float('nan')))} and `{REFERENCE_MODEL}`'s "
        f"{_f(pr_auc_of[REFERENCE_MODEL])}, at Brier "
        f"{_f(float(by_model[PROPOSAL_MODEL]['brier']))} against `gbdt`'s "
        f"{_f(float(by_model['gbdt']['brier']))}. Its advantages are "
        "coverage — it flags "
        f"{_f(float(by_model[PROPOSAL_MODEL]['flagged_fraction']), 2)} of truly-bad "
        "merchants — and a savings column that a badly-calibrated-but-well-covering "
        "model is *rewarded* for under a cost-optimal policy in a way a rank-only policy "
        "would punish. **That is an explanation of the savings ordering, not a "
        "vindication of the model.** `CLAUDE.md` non-negotiable 1 applies without "
        "softening: on ranking and calibration, LightGBM beat the HMM here, exactly as "
        "it did on `validate`."
    )
    add("")
    add(
        "**`gap to knapsack oracle` is deliberately not a column here.** "
        "`results/summary.md` carries the reason and it carries forward unchanged: the "
        "review-knapsack ceiling is the best *review-only, <= K* allocation, this policy "
        "may HOLD, and nothing bounds a holding policy by a review-only ceiling. Quoting "
        "a gap against it would be a category error. The ceiling itself is still printed "
        "above and at every swept point below, including where it goes negative."
    )
    add("")
    lags = {
        str(r["model"]): float(r["lag_days"])
        for r in rows
        if not np.isnan(float(r["lag_days"]))
    }
    add(
        "**The median-lag column was corrected at T-0011 and every model now uses one "
        "convention.** A flag is attributed to the day its evidence was complete — for "
        "`gbdt` and `hmm` the *last* day of the 7-day window that raised it, which is what "
        "`rules` had always reported. Before the correction those two attributed the flag "
        "to the window's *start* day, crediting them with up to 6 days of earliness they "
        "never had; on `validate` that produced the -1.0 median that read as detection "
        "before onset. Here the medians read "
        + ", ".join(f"`{name}` {value:+.1f} d" for name, value in lags.items())
        + " over a 60-day window. **The correction reverses the reading of this column** — "
        "the window-based models are *later* than `rules`, not earlier. "
        "`results/lag_probe.md` reports both conventions side by side and clears the "
        "leakage question separately, with a merchant-clustered permutation test over "
        "pre-onset windows (largest effect 0.159, p = 0.310 on this split). The HMM's "
        "`flag_day` is independently provably forward-only — truncation test with a "
        "negative control — so this was never a leakage question. **No claim of the form "
        "\"Rakshak detects N days before the fraud starts\" is available to this repo.**"
    )
    add("")

    add("## K2's verdict")
    add("")
    add(
        "> `00-charter.md` §3, kill criterion **K2**: *\"Rakshak does not beat the static "
        "rule engine on savings by Tue 1 Sep EOD → do not tune to win. Report the negative "
        "result, pivot the narrative to explainability and the cost frontier, and say so "
        "on camera.\"*"
    )
    add("")
    add(
        "> `00-charter.md` §2, as amended by T-0017 on 2026-08-28 **before any swept number "
        "existed**: *\"Rakshak beats a static velocity/refund-ratio rule engine by >=20% "
        "relative on the Bahnsen savings score at the cited central cost asymmetry, at "
        "equal analyst-hour budget, on a temporally-and-group-split held-out set of unseen "
        "merchants — with the relative improvement reported across the full plausible "
        "asymmetry range and the boundary at which the claim fails stated explicitly.\"*"
    )
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(
        f"| `{PROPOSAL_MODEL}` savings at the central asymmetry | "
        f"{_f(savings_of[PROPOSAL_MODEL])} |"
    )
    add(
        f"| `{REFERENCE_MODEL}` savings at the central asymmetry | "
        f"{_f(savings_of[REFERENCE_MODEL])} |"
    )
    add(f"| absolute margin | {_f(verdict.margin_abs)} |")
    add(f"| **relative margin** | **{_f(verdict.margin_rel * 100.0, 1)}%** |")
    add(f"| bar (NFR-001, pre-registered) | {MARGIN_BAR:.0%} |")
    add(
        f"| `{PROPOSAL_MODEL}` PR-AUC / `{REFERENCE_MODEL}` PR-AUC | "
        f"{_f(pr_auc_of[PROPOSAL_MODEL])} / {_f(pr_auc_of[REFERENCE_MODEL])} |"
    )
    add("")
    if verdict.word == "FAIL":
        headline = (
            f"{K2_VERDICT_PREFIX} **FAIL.** At the cited central asymmetry of "
            f"{central:.1f}, `{PROPOSAL_MODEL}` improves on `{REFERENCE_MODEL}` by "
            f"{_f(verdict.margin_rel * 100.0, 1)}% relative, against a pre-registered bar "
            f"of {MARGIN_BAR:.0%}. **The claim in `00-charter.md` §2 does not hold on the "
            "test window.** K2 fires: nothing is tuned to close the gap, the negative "
            "result is the result, and the narrative moves to explainability and the cost "
            "frontier."
        )
    elif verdict.word == "CONDITIONAL PASS":
        headline = (
            f"{K2_VERDICT_PREFIX} **CONDITIONAL PASS.** At the cited central asymmetry of "
            f"{central:.1f}, `{PROPOSAL_MODEL}` improves on `{REFERENCE_MODEL}` by "
            f"{_f(verdict.margin_rel * 100.0, 1)}% relative, clearing the pre-registered "
            f"{MARGIN_BAR:.0%} bar. **It does not clear it across the whole plausible "
            "asymmetry range**, and the boundary is stated below. The claim is "
            "conditional, it was pre-registered as conditional, and quoting the "
            "favourable point without the range would misrepresent it."
        )
    else:
        headline = (
            f"{K2_VERDICT_PREFIX} **PASS.** At the cited central asymmetry of "
            f"{central:.1f}, `{PROPOSAL_MODEL}` improves on `{REFERENCE_MODEL}` by "
            f"{_f(verdict.margin_rel * 100.0, 1)}% relative, and it clears the "
            f"{MARGIN_BAR:.0%} bar at **every** swept asymmetry — a materially stronger "
            "result than winning at one point, because the claim then depends on no "
            "hand-picked cost number."
        )
    add(headline)
    add("")
    beaten_by_floor = [
        name for name, value in savings_of.items() if name != FLOOR_MODEL and value < floor
    ]
    add(
        "**Whatever that line says, read the next section before quoting it.** The "
        "savings score on this split is dominated by the cost matrix rather than by "
        "detection, and the margin above is a difference of two numbers that a uniform "
        "random score very nearly reaches on its own."
    )
    add("")
    if beaten_by_floor:
        add(
            f"Concretely, and this is the second finding of the run: **`{FLOOR_MODEL}` "
            f"scores {_f(floor)} on this window and beats "
            + ", ".join(f"`{name}`" for name in beaten_by_floor)
            + " on savings.** `"
            + PROPOSAL_MODEL
            + f"` is {_f(savings_of[PROPOSAL_MODEL] - floor)} against the floor — "
            "negative. So the K2 margin above is a comparison between two models that "
            "**both sit below a uniform random score on the primary metric**, and no "
            "reading of it supports a savings claim about the model. `00-charter.md` §2 "
            "is stated against `rules` and is answered against `rules`; the floor "
            "comparison is reported because it is the one that says what the number "
            "means."
        )
        add("")

    add(f"## Savings relative to the `{FLOOR_MODEL}` floor — read this before any savings number")
    add("")
    add(
        f"| model | savings | savings - `{FLOOR_MODEL}` | PR-AUC | what the PR-AUC says |"
    )
    add("|---|---|---|---|---|")
    for name in models:
        if name not in savings_of:
            continue
        note = (
            "ranks at this split's prevalence, i.e. not at all"
            if abs(pr_auc_of[name] - split.prevalence) < 0.05
            else "ranks above prevalence"
        )
        add(
            f"| {name} | {_f(savings_of[name])} | {_f(savings_of[name] - floor)} | "
            f"{_f(pr_auc_of[name])} | {note} |"
        )
    add("")
    add(
        f"**A uniform random score posts {_f(floor)} savings on this split while ranking "
        f"at PR-AUC {_f(pr_auc_of[FLOOR_MODEL])} — at the prevalence, i.e. with no "
        "discriminating power whatsoever.** Nothing about any model produced that; the "
        "cost matrix did. When `c_fp` is small relative to `L_m`, a random score still "
        "lands most merchants on the correct side of a merchant-specific threshold. This "
        "is `07-math.md` §6's AP-06 guard arriving as a measurement rather than a "
        "warning, and it was first measured on `validate` at T-0007b "
        f"(`random` +0.6929 against `rules`' +0.6980). **Any headline of the form "
        "\"Rakshak saves X%\" that does not subtract this floor is not a claim about the "
        f"model.** On the test window the whole spread between the best and worst model "
        f"is {_f(max(savings_of.values()) - min(savings_of.values()))} of savings, "
        f"against a floor level of {_f(floor)}."
    )
    add("")
    if beaten_by_floor:
        best = max(
            (name for name in savings_of if name != FLOOR_MODEL),
            key=lambda name: savings_of[name],
        )
        add(
            f"**On this window the floor does not merely come close — it wins.** "
            f"`{FLOOR_MODEL}` posts the highest savings of any row, "
            f"{_f(floor)} against the best model's {_f(savings_of[best])} (`{best}`), "
            f"while ranking at PR-AUC {_f(pr_auc_of[FLOOR_MODEL])} against `{best}`'s "
            f"{_f(pr_auc_of[best])}. On `validate` at T-0007b the floor sat 0.0051 below "
            "`rules`; here it is above everything. The mechanism is the same one AP-06 "
            "names and it is now unambiguous: **at this prevalence and this cost "
            "asymmetry the savings score is close to insensitive to whether the score "
            "ranks merchants at all.** The honest consequence is that `savings` cannot "
            "carry a headline on this split, in either direction — not for the HMM, and "
            "not against it. PR-AUC, precision@K and the held-per-1000 rate can."
        )
        add("")
    add("### The other half of that finding, from BAF")
    add("")
    add(
        "`results/baf_validation.md` (T-0012) ran the same decision layer on BAF's own "
        "temporal split at a realistic **1.47%** prevalence. There, `random` scores "
        "**-28.2169** — catastrophically negative, not within a whisker of the domain "
        "floor. **That points at this generator's `FRAUD_MERCHANT_RATE = "
        f"{FRAUD_MERCHANT_RATE:.2f}`, not at the savings metric.** At 20% prevalence a "
        "random policy hits enough true positives to look competent; at 1.5% it cannot. "
        "So both halves stand and both must be said together: the AP-06 warning is real "
        "and savings must never be quoted without PR-AUC beside it, *and* the severity of "
        "the `random` floor on this synthetic split is substantially an artefact of a "
        "prevalence the generator inflated on purpose for per-typology sample size. This "
        "is the strongest single piece of evidence in the repo about what the 20% rate "
        "costs."
    )
    add("")

    add("## FR-019 — every headline number in two vocabularies")
    add("")
    add(
        "The ML metrics above, restated in the units a risk-operations reader budgets in. "
        "**The INR column is not a second cost path**: `savings = (Cost_l - Cost(f)) / "
        "Cost_l` by definition (`eval/metrics.py`, `decision/cost.py`), so INR saved is "
        f"exactly `savings * Cost_l` with `Cost_l = {_inr(baseline_cost_inr)}` INR, the "
        "Bahnsen denominator on this split — the cheaper of all-PASS and all-HOLD."
    )
    add("")
    add(
        f"| model | PR-AUC | precision@{k} | INR saved vs Cost_l | INR saved vs "
        f"`{FLOOR_MODEL}` | analyst-hours consumed | merchants held per 1000 |"
    )
    add("|---|---|---|---|---|---|---|")
    for row in rows:
        ops = operational_row(row, baseline_cost_inr, floor, split.n_merchants)
        add(
            f"| {row['model']} | {_f(float(row['pr_auc']))} | "
            f"{_f(float(row['precision_at_k']))} | "
            f"{_inr(ops['inr_saved_vs_baseline'])} | {_inr(ops['inr_saved_vs_random'])} | "
            f"{ops['analyst_hours']:.2f} | {ops['held_per_1000']:.0f} |"
        )
    add("")
    add(
        "`INR saved vs Cost_l` is the operational reading of the savings column and "
        "inherits its whole AP-06 caveat: most of it is the cost matrix. **`INR saved vs "
        f"`{FLOOR_MODEL}`` is the part attributable to the model** — the only one of the "
        "two that is a claim about detection. "
        + (
            "**On this window it is negative for every model: no model here saves money "
            "relative to scoring merchants at random.** "
            if len(beaten_by_floor) == len(savings_of) - 1
            else "Read it, not the column beside it. "
        )
        + "Analyst-hours are the FR-017 budget actually consumed; "
        "merchants held per 1000 is the honest-merchant cost the panel's second question "
        "asks about, and it is a rate so it transfers to a real book of any size."
    )
    add("")

    add("## FR-020 — the cost-asymmetry sweep, run on `test`")
    add("")
    add("![Cost-asymmetry sensitivity on the test window](figures/sensitivity_test.png)")
    add("")
    add(
        "Drawn by `rakshak.eval.figures` from `results/sensitivity_test.csv`, which is the "
        "same frame that produced every table below — **the figure computes nothing of its "
        "own and cannot disagree with the tables.** `results/sensitivity.md` carries the "
        "full FR-020 commentary on the `validate` window (how the range is derived, why "
        "the review-only ceiling stops being a ceiling at low asymmetry, and the "
        "parameterisation caveat); it is not repeated here. What is below is the test "
        "window and the boundary."
    )
    add("")
    add("### (a) Relative improvement over `rules` at every swept point")
    add("")
    add(
        "| asymmetry | " + " | ".join(models) + " | margin abs | margin rel | "
        f">= {MARGIN_BAR:.0%}? |"
    )
    add("|---|" + "---|" * (len(models) + 3))
    for asymmetry, block in sweep.groupby("asymmetry", sort=True):
        by_name = dict(zip(block["model"], block["savings"], strict=True))
        cells = " | ".join(f"{by_name.get(m, float('nan')):+.4f}" for m in models)
        margin_abs = float(block["margin_abs"].iloc[0])
        margin_rel = float(block["margin_rel"].iloc[0])
        rel = "n/a" if np.isnan(margin_rel) else f"{margin_rel:+.1%}"
        clears = "**yes**" if (margin_rel >= MARGIN_BAR) else "no"
        add(f"| {asymmetry:.1f} | {cells} | {margin_abs:+.4f} | {rel} | {clears} |")
    add("")
    add(
        f"**Read the `{FLOOR_MODEL}` column before any other.** It is a uniform random "
        "score. Any margin quoted off this table must be quoted against it, not against "
        "zero. `margin rel` reads `n/a` where the `rules` denominator sits within 1e-6 of "
        "zero — a relative margin over a near-zero denominator is not a number worth "
        "printing, and the absolute margin beside it is always defined."
    )
    add("")
    add("### (b) The boundary asymmetry, stated as a number")
    add("")
    if verdict.holds_everywhere:
        add(
            f"**The >=20% claim holds at every swept asymmetry, {low:.1f} through "
            f"{high:.1f}.** There is no boundary to state inside the plausible range. "
            "That is materially stronger than winning at one point: the claim depends on "
            "no hand-picked cost number anywhere in the range the cited primitives admit."
        )
    elif verdict.fails_everywhere:
        add(
            f"**The >=20% claim holds at no swept asymmetry between {low:.1f} and "
            f"{high:.1f}.** There is no boundary above which it starts holding inside the "
            "plausible range: it fails throughout. Reported as measured."
        )
    else:
        add(
            f"**The >=20% claim stops holding between asymmetry "
            f"{verdict.boundary_fails_at:.1f} and {verdict.boundary_holds_at:.1f}.** It "
            f"holds at and above {verdict.boundary_holds_at:.1f} and fails at and below "
            f"{verdict.boundary_fails_at:.1f}, against a cited central value of "
            f"{central:.1f} and a plausible range of {low:.1f} - {high:.1f}. **That "
            "boundary is the deliverable, it goes in the README and the video, and it is "
            "stated rather than narrowed away.**"
        )
        if not verdict.contiguous_above:
            add("")
            add(
                "**The crossing is not a single boundary.** The claim does not hold at "
                "every point above the first point at which it holds, so the table must "
                "be read point by point rather than as one threshold. Reported because it "
                "is what the sweep measured."
            )
    add("")
    if np.isfinite(verdict.zero_holds_at) and np.isfinite(verdict.zero_fails_at):
        add(
            f"The weaker question — does `{PROPOSAL_MODEL}` beat `{REFERENCE_MODEL}` at "
            f"all — crosses between asymmetry {verdict.zero_fails_at:.1f} and "
            f"{verdict.zero_holds_at:.1f} on this split. On `validate`, T-0007b measured "
            "that crossing between **18.5 and 36.2**."
        )
    elif np.isfinite(verdict.zero_holds_at):
        add(
            f"The weaker question — does `{PROPOSAL_MODEL}` beat `{REFERENCE_MODEL}` at "
            "all — is answered yes at every swept point on this split. On `validate`, "
            "T-0007b measured that crossing between **18.5 and 36.2**, so the test window "
            "is more favourable to the proposal than the validation window was."
        )
    else:
        add(
            f"The weaker question — does `{PROPOSAL_MODEL}` beat `{REFERENCE_MODEL}` at "
            "all — is answered **no at every swept point** on this split. On `validate`, "
            "T-0007b measured the crossing between **18.5 and 36.2**, so the test window "
            "is less favourable to the proposal than the validation window was."
        )
    add("")
    add(
        "**Caveat, carried forward from T-0007b and disclosed there rather than found "
        "later.** `asymmetry_range` reaches its corners by rescaling `value_inr` *and* "
        "`loss_inr` together with six primitives; the sweep reproduces each asymmetry by "
        "moving the false-positive branch alone (`fp_cost_scale`), which isolates the "
        "asymmetry instead of confounding it with the absolute size of fraud loss. The "
        "two routes agree on the **ratio** and not on the whole cost matrix — "
        "`cost_review_inr` is an analyst wage and rescales with neither. The model "
        "*ordering* at a point is unaffected, since every model at a point faces the "
        "identical matrix, but **the crossing asymmetry is specific to this "
        "parameterisation** and would move under the other one."
    )
    add("")
    add("### (c) The FP-cost-per-100 ratio the cited primitives produce")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| Total FP cost, all healthy merchants held (INR) | {_inr(fp_cost_total)} |")
    add(f"| Total fraud loss, all bad merchants passed (INR) | {_inr(fraud_loss_total)} |")
    add(f"| INR of FP cost per INR 100 of fraud loss | {_f(ratio, 1)} |")
    add("| 07-math.md §5 commentary band (cross-check, **not** a gate) | 400 - 600 |")
    add(f"| Divergence | {_f(ratio, 1)} vs 400-600 — **stated, not closed** |")
    add("")
    add(
        f"**The divergence is roughly {400.0 / ratio:.0f}x at the low end of the band and "
        f"is not closed.** The commentary band measures *falsely declined baskets at "
        "checkout*, where the denied item is the full basket value; this ratio measures "
        "*held merchant settlements*, where the cost is the platform's own ~10 bps margin "
        "over the merchant's remaining lifetime and the fraud side is realised chargebacks "
        "rather than an abandoned cart. They were never the same asymmetry. T-0017 demoted "
        "the band from a gate to a reported cross-check precisely so that no primitive "
        "would be moved to reach it, and none was. The swept range runs "
        f"{low:.1f} - {high:.1f}, so the highest-asymmetry rows above are the closest this "
        "repo gets to what the band would imply if it applied — an illustration, not a "
        "second operating point."
    )
    add("")
    add("### (d) The change in optimal thresholds over the sweep")
    add("")
    add(
        "| asymmetry | p* median (at risk, L_m > 0) | p* median (all merchants) | "
        "knapsack ceiling | knapsack >= hold-everything |"
    )
    add("|---|---|---|---|---|")
    for _, row in points.iterrows():
        add(
            f"| {float(row['asymmetry']):.1f} | "
            f"{float(row['hold_threshold_median_at_risk']):.4f} | "
            f"{float(row['hold_threshold_median']):.4f} | "
            f"{float(row['knapsack_ceiling']):+.4f} | "
            f"{'yes' if bool(row['knapsack_clears_hold_everything']) else '**no**'} |"
        )
    add("")
    n_zero_loss = int((np.asarray(params.loss_inr, dtype=float) <= 0.0).sum())
    at_risk_first = float(points["hold_threshold_median_at_risk"].iloc[0])
    at_risk_last = float(points["hold_threshold_median_at_risk"].iloc[-1])
    add(
        "p* = c_fp(m) / (L_m + c_fp(m) - rho L_m) is Elkan (2001)'s cost-matrix-derived "
        "threshold, and it is per merchant — that example-dependence is the whole argument "
        "for the decision layer. **The all-merchant median is pinned at 1.0000 at every "
        f"asymmetry and that is not a result**: {n_zero_loss} of these "
        f"{split.n_merchants} merchants never transact in a bad state, so L_m = 0 and p* "
        "collapses to c_fp / c_fp = 1 exactly — \"never hold this merchant\", correct and "
        "uninformative, and precisely the opposite of Elkan's point. The at-risk median is "
        f"the column that moves, from {at_risk_first:.4f} at asymmetry {low:.1f} to "
        f"{at_risk_last:.4f} at {high:.1f}. Both ship, so neither can be quoted without "
        "the other. T-0007b found the degeneracy on `validate`; it reproduces here."
    )
    add("")

    add("## What this does not establish")
    add("")
    add(
        "1. **Whether the win — where there is one — comes from sequence modelling or "
        "from the HMM specifically is left OPEN.** T-0010's BOCPD changepoint baseline "
        "was cut in the 2026-08-28 re-plan, so **no sequence-aware baseline other than "
        "the HMM was measured anywhere in this repo.** `rules` and `gbdt` are both "
        "point-in-time over windowed aggregates. The comparison that would answer the "
        "question does not exist and is not approximated by anything here. It is not "
        "reported as zero and not omitted — it is open."
    )
    add(
        "2. **No calibration happens anywhere in this repo.** T-0008 (empirical-Bayes "
        "shrinkage) was cut in the same re-plan, so Bayes Minimum Risk consumes each "
        "model's raw score, clipped to [0, 1], as if it were a calibrated posterior. "
        "Under a rank-only policy miscalibration would only cost a model its Brier gap; "
        "**under BMR it moves the argmin, not merely the ranking.** Every savings number "
        "above inherits that, and it is why `savings` and `Brier` are coupled here in a "
        "way they would not be in a calibrated system."
    )
    add(
        "3. **The perfect-hindsight oracle dominates by construction and proves "
        "nothing.** It is a per-merchant argmin over the whole action set with the label "
        "known, so it is above every policy under any cost matrix. It is printed as an "
        "upper bound for gap-to-oracle, not as a validation that anything works."
    )
    add(
        "4. **The review-knapsack ceiling clears hold-everything on this split only "
        "because loss is concentrated.** It is review-only and capacity-bound, so nothing "
        "forces it above a policy that may HOLD — T-0007a wrote that down in "
        "`tests/test_cost.py`'s header before it bit, and the sweep column above shows "
        "exactly where it stops clearing. On a flat population with identical constants "
        "it scored -0.092 and the invariant fired. The honest framing is *\"the "
        "constrained ceiling clears hold-everything on this split because loss is "
        "concentrated\"*, never *\"the oracle beat everything\"*."
    )
    add(
        "5. **Everything above is measured on a generator this repo wrote**, at a 20% "
        "merchant fraud rate chosen for per-typology sample size rather than realism. "
        "`results/calibration_gap.md` (T-0015) measures the divergence between the "
        "generator's marginals and a real transaction dataset instead of merely admitting "
        "it: 5 of 8 ratio-scale marginals diverge by >=1.9x, and one of them "
        "(`daily_count_fano_factor`) is structural rather than parametric and closable by "
        "no choice of constant."
    )
    add("")

    absent = [(name, why) for name, why in EXPECTED_MODELS if name not in MODEL_REGISTRY]
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
            "These rows are missing, not zero. The verdict above is rendered over the "
            "models that ran, and the absent ones are named so no reader has to infer "
            "them."
        )
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(seed: int, results_dir: Path = RESULTS_DIR) -> Path:
    """Score `test`, sweep it, and write the verdict. Returns `verdict.md`'s path.

    The capacity rule is the harness's, unchanged: `REVIEW_CAPACITY_HOURS_PER_1000_
    MERCHANTS` scaled to the split's own population (ADR-0008). The
    oracle-dominance invariant runs as a precondition, before anything is written,
    exactly as `harness.run` does it.

    Args:
        seed: Global seed (NFR-003).
        results_dir: Where `verdict.md`, `sensitivity_test.csv` and the figure go.

    Returns:
        Path to `results/verdict.md`.
    """
    split = load_split(VERDICT_SPLIT, unlock_test=UNLOCK_TICKET)
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

    # Precondition, not a postcondition: if a policy beats a ceiling that genuinely
    # bounds it, no file is written and no verdict is rendered.
    policy.assert_ceilings_dominate(
        y,
        params,
        {o.name: o.savings for o in oracles},
        {str(r["model"]): float(r["savings"]) for r in rows},
        seed=seed,
    )

    frame = policy.sweep_cost_asymmetry(y, posteriors, params, capacity_hours, seed=seed)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "sensitivity_test.csv"
    csv_path.write_text(frame.to_csv(index=False), encoding="utf-8", newline="")
    # Read back rather than pass the in-memory frame: the figure is then provably a
    # rendering of the committed CSV and can compute nothing the tables do not show.
    committed = pd.read_csv(csv_path)
    render_sensitivity_figure(committed, results_dir / "figures" / "sensitivity_test.png")

    verdict = assess_k2(rows, committed)
    path = results_dir / "verdict.md"
    path.write_text(
        render_verdict(
            split, rows, oracles, committed, params, verdict, seed, k, capacity_hours
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """Render K2's verdict on the test window. Returns a process exit code."""
    args = base_parser(
        "Render K2's verdict on the held-out test window (T-0011)."
    ).parse_args(argv)
    seed_everything(args.seed)
    path = run(args.seed)
    print(f"rakshak: wrote {path} (seed={args.seed})")
    print(
        "rakshak: the TEST window was opened here, once, under ticket "
        f"{UNLOCK_TICKET} - 06-requirements.md section 3."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
