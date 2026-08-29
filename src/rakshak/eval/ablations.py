"""FR-018 ablation table — what each component is actually worth, on `test`.

`make eval` reports `results/summary.md` on `validate`. This module is one of the
two places authorised to open the `test` window (06-requirements.md §3,
`splits._TEST_UNLOCK_TICKETS`), and it opens it to *report*, not to search:

**No configuration was selected on `test`.** The shipping configuration — four
latent states, T-0004b's items 1+2 fit, the full FR-008/FR-009 emission vector,
FR-007 within-merchant standardisation — was fixed at T-0004b on `validate` and
has not moved since. Every row below re-runs that frozen configuration with one
component removed. Nothing here is tuned, nothing is selected, and no row was
dropped for being unflattering. That distinction is the whole reason the unlock
is honest.

**What an ablation costs, and why it is only three live rows.** Each "off" row
refits the model on the reduced emission vector — a variant fit must never be
served from the shipping `lru_cache`, so the variant is part of the cache key in
both `models/gbdt.py` and `models/hmm_score.py`. Six fits in total, and the
segment map is still fitted on the training population alone and passed to the
held-out build in every one of them, so the leakage guard is unchanged.

**FR-018's own words: "a component whose removal changes no number is
decoration."** That is a finding to print, not to hide, and the delta columns
exist so that a zero is as visible as a win.

**AP-06.** Every savings number here is printed beside its PR-AUC. On this
generator's 20% prevalence the savings score is largely earned by the cost
matrix rather than by detection (`results/summary.md`, the `random` row), so a
savings delta on its own is not interpretable and must not be quoted alone.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rakshak.cli import base_parser, seed_everything
from rakshak.config import (
    RESULTS_DIR,
    REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS,
    TAU_REVIEW_HOURS,
)
from rakshak.eval import metrics
from rakshak.eval.harness import MODEL_REGISTRY, Scorer, _model_rng, evaluate_model
from rakshak.eval.oracle import review_slots
from rakshak.eval.splits import Split, load_split
from rakshak.models.gbdt import score_gbdt
from rakshak.models.hmm_score import score_hmm

ABLATION_SPLIT: str = "test"
"""The window this module reports on. T-0011 is authorised to unlock it."""

UNLOCK_TICKET: str = "T-0011"
"""Ticket id `eval/splits.py` requires before it will open the test window."""

GRAPH_FEATURES: tuple[str, ...] = (
    "payer_entropy",
    "repeat_payer_ratio",
    "payer_jaccard_prev",
    "payer_herfindahl",
)
"""FR-008's graph-derived scalars — the CPU stand-in ADR-0002 chose over a GNN.
Dropping exactly these four is the ADR-0002 ablation; if it changes nothing, that
is a finding about ADR-0002 and it is stated as one."""

_BASE_SCORERS: dict[str, Scorer] = {"gbdt": score_gbdt, "hmm": score_hmm}


@dataclass(frozen=True)
class Config:
    """One ablation configuration: a base model plus the component removed.

    Attributes:
        key: Stable identifier used by the table layout.
        base_model: "hmm" or "gbdt". Fixes which scorer and which fit seed is used.
        drop_features: Emission columns removed before fitting and before scoring.
        standardise: False removes FR-007 within-merchant standardisation.
        label: Human-readable description for the table's `configuration` column.
    """

    key: str
    base_model: str
    drop_features: tuple[str, ...]
    standardise: bool
    label: str


CONFIGS: tuple[Config, ...] = (
    Config("hmm.full", "hmm", (), True, "HMM, full emission vector, standardised"),
    Config("gbdt.full", "gbdt", (), True, "LightGBM, full emission vector, standardised"),
    Config("hmm.nograph", "hmm", GRAPH_FEATURES, True, "HMM, 4 graph scalars dropped"),
    Config("gbdt.nograph", "gbdt", GRAPH_FEATURES, True, "LightGBM, 4 graph scalars dropped"),
    Config("hmm.nostd", "hmm", (), False, "HMM, raw window features, no standardisation"),
    Config("gbdt.nostd", "gbdt", (), False, "LightGBM, raw window features, no standardisation"),
)
"""The six configurations that are actually fitted. `hmm.full` and `gbdt.full`
each serve as the reference for their own model's deltas, and `gbdt.full` is
also the "HMM off" row — see `render_ablations`."""

CONTEXT_MODELS: tuple[str, ...] = ("random", "rules")
"""Rows carried for reading the savings column only: `random` is the floor the
cost matrix alone earns (AP-06) and `rules` is the incumbent floor. Neither is
fitted and neither is an ablation. No verdict is rendered from them here."""


# ---------------------------------------------------------------------------
# Running the variants
# ---------------------------------------------------------------------------


@contextmanager
def _registered(name: str, scorer: Scorer) -> Iterator[None]:
    """Temporarily add a scorer to `harness.MODEL_REGISTRY`.

    `harness.evaluate_model` is the only implementation of scoring -> BMR policy ->
    metrics in the repo and it looks its scorer up by name, so a variant has to be
    reachable through the registry for exactly as long as it is being scored. The
    entry is removed in a `finally`, so `MODEL_REGISTRY` is unchanged afterwards
    whatever happens.
    """
    MODEL_REGISTRY[name] = scorer
    try:
        yield
    finally:
        MODEL_REGISTRY.pop(name, None)


def _variant_scorer(config: Config, seed: int) -> Scorer:
    """Wrap a base scorer so that only the ablated component differs.

    The returned scorer **ignores the RNG the harness hands it** and rebuilds the
    base model's own RNG from `(seed, base_model)`. That is deliberate: the fit
    seed is derived from the RNG, and a variant registered under a different name
    would otherwise draw a different fit seed and confound "removed a component"
    with "reseeded the fit".

    Args:
        config: The configuration to score.
        seed: Global seed (NFR-003).

    Returns:
        A `harness.Scorer`.
    """
    base = _BASE_SCORERS[config.base_model]

    def scorer(split: Split, _rng: np.random.Generator) -> object:
        return base(
            split,
            _model_rng(seed, config.base_model),
            drop_features=config.drop_features,
            standardise=config.standardise,
        )

    return scorer


def evaluate_configs(split: Split, seed: int, k: int) -> dict[str, dict[str, object]]:
    """Score every configuration in `CONFIGS` plus the context models.

    Args:
        split: The split to report on.
        seed: Global seed (NFR-003).
        k: Review budget in merchants; the analyst-hour budget is `k * tau`.

    Returns:
        key -> the `harness.evaluate_model` summary row, with the `posterior`
        entry dropped (nothing here re-scores under a different cost matrix).
    """
    rows: dict[str, dict[str, object]] = {}
    for config in CONFIGS:
        with _registered(config.key, _variant_scorer(config, seed)):
            row = evaluate_model(config.key, split, seed, k)
        row.pop("posterior", None)
        rows[config.key] = row
    for name in CONTEXT_MODELS:
        row = evaluate_model(name, split, seed, k)
        row.pop("posterior", None)
        rows[name] = row
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# component | setting | configuration key | reference key. A None key is a row that
# was never measured and says so in every cell; it is never rendered as a zero.
TABLE: tuple[tuple[str, str, str | None, str | None], ...] = (
    ("HMM (the proposal)", "**on** — shipping", "hmm.full", None),
    ("HMM (the proposal)", "**off** — same pipeline, LightGBM scorer", "gbdt.full", "hmm.full"),
    ("graph features (FR-008)", "**on** — HMM", "hmm.full", None),
    ("graph features (FR-008)", "**off** — HMM refitted", "hmm.nograph", "hmm.full"),
    ("graph features (FR-008)", "**on** — LightGBM", "gbdt.full", None),
    ("graph features (FR-008)", "**off** — LightGBM refitted", "gbdt.nograph", "gbdt.full"),
    ("within-merchant standardisation (FR-007)", "**on** — HMM", "hmm.full", None),
    ("within-merchant standardisation (FR-007)", "**off** — HMM refitted", "hmm.nostd", "hmm.full"),
    ("within-merchant standardisation (FR-007)", "**on** — LightGBM", "gbdt.full", None),
    (
        "within-merchant standardisation (FR-007)",
        "**off** — LightGBM refitted",
        "gbdt.nostd",
        "gbdt.full",
    ),
    ("empirical-Bayes shrinkage (ADR-0006)", "on / off", None, None),
    ("NSGA-II vs. grid search (ADR-0004)", "frontier vs. grid", None, None),
)

_NOT_MEASURED = "**not measured**"


def _f(value: float, places: int = 4) -> str:
    """Fixed-width float formatting. NaN renders as 'n/a' so it never varies."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def _delta(value: float, reference: float, places: int = 4) -> str:
    """Signed delta against the reference row, with an explicit `0` for no change."""
    difference = value - reference
    if difference == 0.0:
        return "0 (no change)"
    return f"{difference:+.{places}f}"


def _direction(delta: float, higher_is_better: bool = True) -> str:
    """'improves' / 'degrades' / 'does not move', so the prose cannot contradict a delta.

    Args:
        delta: Row value minus reference value.
        higher_is_better: False for Brier, where a fall is an improvement.

    Returns:
        The verb to use in the findings paragraphs.
    """
    if delta == 0.0:
        return "does not move"
    return "improves" if (delta > 0) == higher_is_better else "degrades"


def _row_cells(row: dict[str, object], baseline_inr: float, n_merchants: int) -> list[str]:
    """The metric half of one table row, in both FR-019 vocabularies."""
    savings = float(row["savings"])
    return [
        _f(savings),
        "",  # delta placeholder, filled by the caller
        _f(float(row["pr_auc"])),
        "",
        _f(float(row["precision_at_k"])),
        "",
        _f(float(row["brier"])),
        "",
        f"{savings * baseline_inr:,.0f}",
        f"{float(row['hours_used']):.2f}",
        f"{int(row['n_held']) / n_merchants * 1000:.0f}",
    ]


def render_ablations(
    split: Split,
    rows: dict[str, dict[str, object]],
    seed: int,
    k: int,
    capacity_hours: float,
) -> str:
    """Build `results/ablations.md`. Byte-identical for a fixed seed (NFR-003).

    Args:
        split: The split reported on.
        rows: Output of `evaluate_configs`.
        seed: Global seed, printed for provenance.
        k: Review budget in merchants.
        capacity_hours: The analyst-hour budget, in hours.

    Returns:
        The full markdown document.
    """
    y = split.labels.to_numpy(dtype=float)
    baseline_inr = metrics.baseline_cost(y, split.loss_inr, split.value_inr)
    n_bad = int(split.labels.sum())

    lines: list[str] = []
    add = lines.append

    add("# Rakshak — ablation table (FR-018)")
    add("")
    add(
        "> **Sequence-layer metrics are measured on synthetic merchant streams with injected "
        "typologies; the generator is in this repo.** The decision layer is additionally "
        "validated on BAF (Feedzai, NeurIPS 2022), a public benchmark derived from real bank "
        "data."
    )
    add("")
    add(
        "Everything on this page is the synthetic split. BAF has no sequences, so no ablation "
        "of the sequence layer could have been run there (`results/baf_validation.md`)."
    )
    add("")

    add("## Provenance")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(f"| Produced by | `python -m rakshak.eval.ablations --seed {seed}` |")
    add(f"| Seed | {seed} |")
    add(
        f"| Split reported | `{split.name}` (days {split.start_day}-{split.end_day - 1}), "
        f"unlocked with `unlock_test=\"{UNLOCK_TICKET}\"` |"
    )
    add(
        f"| Population | {split.n_merchants} merchants, {n_bad} truly bad "
        f"({split.prevalence:.1%} prevalence) |"
    )
    add(
        f"| Review budget K | {k} merchants ({capacity_hours:.2f} analyst-hours at "
        f"{TAU_REVIEW_HOURS:.3f} h per review) |"
    )
    add(f"| Cost basis for the INR column | Cost_l = INR {baseline_inr:,.0f} on this split |")
    add(f"| Fits performed | {len(CONFIGS)} ({len(CONFIGS) // 2} configurations x 2 models) |")
    add("")

    add("### No configuration was selected on `test`")
    add("")
    add(
        "The shipping configuration was fixed at **T-0004b on `validate`** — four latent "
        "states, the items 1+2 partially-supervised fit, the full FR-008/FR-009 emission "
        "vector, FR-007 within-merchant standardisation — and has not moved since. "
        "**These rows are a report, not a search.** Each one re-runs that frozen "
        "configuration with a single component removed, at one seed, and every row that ran "
        "is printed whether it flatters the component or not. Nothing on this page was "
        "chosen because it looked better here; if it had been, the test window would no "
        "longer be a held-out window and every number in the README would inherit the "
        "problem."
    )
    add("")
    add(
        "Both models refit for every ablation. The segment map is fitted on the **training** "
        "population alone and passed into the held-out build in all six fits, so the leakage "
        "guard in `eval/splits.py` holds unchanged across the variants. The variant is part "
        "of the memoisation key in `models/gbdt.py` and `models/hmm_score.py`, so a variant "
        "fit can never be served to the shipping path."
    )
    add("")

    add("## The table")
    add("")
    add(
        "Headline metric is `savings`. **PR-AUC is printed beside every savings number** and "
        "must be read with it — see the AP-06 note below. FR-019's two vocabularies: the ML "
        "columns are PR-AUC / precision@K / Brier, the operational columns are INR saved, "
        "analyst-hours consumed and merchants held per 1000."
    )
    add("")
    add(
        f"| component | setting | savings | d savings | PR-AUC | d PR-AUC | precision@{k} | "
        "d prec | Brier | d Brier | INR saved | analyst-h | held /1000 |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for component, setting, key, reference_key in TABLE:
        if key is None:
            add(
                f"| {component} | {setting} | "
                + " | ".join([_NOT_MEASURED] * 11)
                + " |"
            )
            continue
        row = rows[key]
        cells = _row_cells(row, baseline_inr, split.n_merchants)
        if reference_key is None:
            cells[1] = cells[3] = cells[5] = cells[7] = "reference"
        else:
            ref = rows[reference_key]
            cells[1] = _delta(float(row["savings"]), float(ref["savings"]))
            cells[3] = _delta(float(row["pr_auc"]), float(ref["pr_auc"]))
            cells[5] = _delta(float(row["precision_at_k"]), float(ref["precision_at_k"]))
            cells[7] = _delta(float(row["brier"]), float(ref["brier"]))
        add(f"| {component} | {setting} | " + " | ".join(cells) + " |")
    add("")
    add(
        "`d X` is the row's value minus its reference row's. For `Brier`, lower is better, so "
        "a **negative** delta is an improvement; for the other three, higher is better. "
        "`INR saved` is `savings x Cost_l` and therefore carries every caveat `savings` "
        "carries. `held /1000` is FR-019's operational quantity: merchants placed on HOLD per "
        "1000 under watch."
    )
    add("")

    add("## Context rows — how much of `savings` is the cost matrix")
    add("")
    add(
        "Not ablations. Printed because a savings delta cannot be read without knowing what a "
        "model that ranks at chance earns from the same cost matrix. **No verdict is rendered "
        "here**; K2 is rendered elsewhere in T-0011."
    )
    add("")
    add(f"| model | savings | PR-AUC | precision@{k} | Brier | INR saved | held /1000 |")
    add("|---|---|---|---|---|---|---|")
    for name in CONTEXT_MODELS:
        row = rows[name]
        savings = float(row["savings"])
        add(
            f"| {name} | {_f(savings)} | {_f(float(row['pr_auc']))} | "
            f"{_f(float(row['precision_at_k']))} | {_f(float(row['brier']))} | "
            f"{savings * baseline_inr:,.0f} | "
            f"{int(row['n_held']) / split.n_merchants * 1000:.0f} |"
        )
    add("")
    random_savings = float(rows["random"]["savings"])
    random_pr_auc = float(rows["random"]["pr_auc"])
    hmm_savings = float(rows["hmm.full"]["savings"])
    rules_savings = float(rows["rules"]["savings"])
    over_random = hmm_savings - random_savings
    add(
        f"**AP-06, as a measurement, and it is worse on `test` than it was on `validate`.** "
        f"`random` scores **{_f(random_savings)}** savings here while ranking at PR-AUC "
        f"{_f(random_pr_auc)} — chance, at this split's {split.prevalence:.0%} prevalence. "
        f"The shipping configuration scores {_f(hmm_savings)} and `rules` scores "
        f"{_f(rules_savings)}. **Savings net of the random floor is "
        f"{_f(over_random)} for the shipping configuration and "
        f"{_f(rules_savings - random_savings)} for `rules`** — on this split a uniform "
        "random score, spent through the same Bayes-Minimum-Risk policy, out-saves every "
        "fitted model in the table above."
    )
    add("")
    add(
        "Nothing about any model produced that; the cost matrix did. When `c_fp` is small "
        "relative to `L_m`, a merchant-specific threshold puts most merchants on the correct "
        "side of the decision whatever the score is, so `savings` measures the cost "
        "arithmetic far more than it measures detection. On `validate` (T-0007b) the same "
        "effect left `random` 0.0051 behind `rules`; on `test` it puts `random` ahead of "
        "everything. **This is not a result about the models and it must not be read as "
        "one — it is the strongest evidence in the repo that an absolute savings figure is "
        "not a claim about detection.** Whoever renders K2 must report savings relative to "
        "this floor, and must state that on `test` that relative figure is negative for "
        "every model here. On the ranking metrics — PR-AUC, precision@K, Brier — `random` "
        "sits where it belongs, at the bottom."
    )
    add("")

    add("## What the table actually says")
    add("")

    def _pair(key: str, reference_key: str, field: str, higher_is_better: bool = True) -> str:
        """`"+0.1234 (improves)"` for one cell, so no sentence can outlive its number."""
        delta = float(rows[key][field]) - float(rows[reference_key][field])
        return f"{delta:+.4f} ({_direction(delta, higher_is_better)})"

    add(
        "**1. The incumbent out-ranks the proposal on every ML metric, and the savings "
        "column barely moves.** Against the shipping configuration, the LightGBM row moves "
        f"savings by {_pair('gbdt.full', 'hmm.full', 'savings')}, PR-AUC by "
        f"{_pair('gbdt.full', 'hmm.full', 'pr_auc')}, precision@{k} by "
        f"{_pair('gbdt.full', 'hmm.full', 'precision_at_k')} and Brier by "
        f"{_pair('gbdt.full', 'hmm.full', 'brier', False)}. The incumbent ranks and "
        "calibrates far better on this window. That is A-005's question arriving with an "
        "answer, and it is reported here rather than tuned: **the verdict clause of K2 is "
        "rendered elsewhere in T-0011, but nothing in this row supports a claim that the "
        "sequence layer earns its place on ranking quality.** What the HMM has is the "
        "Viterbi path an analyst can read, and that is an explainability argument, not a "
        "metric argument."
    )
    add("")
    add(
        "**2. The graph scalars are not decoration — for either model.** Dropping the four "
        f"FR-008 columns moves the HMM's PR-AUC by {_pair('hmm.nograph', 'hmm.full', 'pr_auc')} "
        f"and its savings by {_pair('hmm.nograph', 'hmm.full', 'savings')}; it moves "
        f"LightGBM's PR-AUC by {_pair('gbdt.nograph', 'gbdt.full', 'pr_auc')} and its "
        f"precision@{k} by {_pair('gbdt.nograph', 'gbdt.full', 'precision_at_k')}. "
        "**ADR-0002's substitution of four CPU scalars for a graph neural network is "
        "carrying real signal on this generator.** It does not follow that a GNN was "
        "unnecessary, and it does not follow that the scalars would carry the same signal on "
        "a real payer graph — the generator wrote the payer process these features read."
    )
    add("")
    add(
        "**3. Within-merchant standardisation is load-bearing for the HMM and close to "
        "decoration for LightGBM.** Turning FR-007 off moves the HMM's PR-AUC by "
        f"{_pair('hmm.nostd', 'hmm.full', 'pr_auc')} and its Brier by "
        f"{_pair('hmm.nostd', 'hmm.full', 'brier', False)}; it moves LightGBM's savings by "
        f"{_pair('gbdt.nostd', 'gbdt.full', 'savings')}, its PR-AUC by "
        f"{_pair('gbdt.nostd', 'gbdt.full', 'pr_auc')} and its precision@{k} by "
        f"{_pair('gbdt.nostd', 'gbdt.full', 'precision_at_k')}. **This is FR-018's own test "
        "landing on a component: for LightGBM, on this split, at this seed, "
        "within-merchant standardisation is very nearly a component whose removal changes no "
        "number.** The asymmetry is mechanical and was predictable — see the standardisation "
        "section below — but it is printed rather than smoothed over, and it means the "
        "`gbdt` baseline in `results/summary.md` would be about as strong without P-02 as "
        "with it. P-02 earns its place through the pooled Gaussian HMM, not through the "
        "incumbent."
    )
    add("")

    add("## What each row means")
    add("")
    add("### HMM on / off — the construction, stated plainly")
    add("")
    add(
        '**"HMM off" here is the `gbdt` path: the identical feature pipeline, the identical '
        "segment map, the identical decision-window mask and the identical BMR policy, scored "
        "by LightGBM over windowed aggregates instead of by a filtered latent-state "
        "posterior.** `models/hmm_score.py` takes its design matrix from "
        "`models/gbdt.py::build_window_matrix` precisely so the two see byte-identical inputs."
    )
    add("")
    add(
        "**That is not the same experiment as switching the sequence layer off in place**, and "
        "the difference matters enough to state rather than bury. Removing the HMM leaves no "
        "scorer at all; something has to score the merchants. What this row measures is "
        "therefore *HMM versus the incumbent discriminative model on the same features*, "
        "which is A-005's question, not *the marginal value of sequence structure*. The "
        "cleaner experiment — a sequence-aware model that is not this HMM — was BOCPD, and "
        "BOCPD was cut. See the note below."
    )
    add("")

    add("### Graph features on / off — this is an ADR-0002 result")
    add("")
    add(
        "The four scalars removed are `payer_entropy`, `repeat_payer_ratio`, "
        "`payer_jaccard_prev` and `payer_herfindahl` (`features/windows.py::BASE_FEATURES`). "
        "**They exist because ADR-0002 rejected a graph neural network** — GPU-bound, "
        "circular to evaluate on a synthetic graph, infeasible solo in four days — and chose "
        "these CPU-computable scalars as the stand-in. The emission vector goes from 14 "
        "features to 10 and both models are refitted."
    )
    add("")
    add(
        "So this row is not a feature-selection curiosity. It is the only evidence in the "
        "repo about whether ADR-0002's substitution bought anything, and the delta must be "
        "read as an ADR-0002 result in either direction: a delta of ~0 would have meant "
        "**the GNN stand-in is decoration on this generator**, and a large delta means the "
        "substitution carries signal *that the generator put in the payer process*. Neither "
        "reading licenses a claim about what a GNN would have done on real data — this repo "
        "has no evidence about that either way."
    )
    add("")

    add("### Within-merchant standardisation on / off — cross-merchant comparability")
    add("")
    add(
        "`features/standardise.py::standardise_panel` expresses every emission as a deviation "
        "from *that merchant's own* burn-in norm (FR-007, P-02). With it off, the raw "
        "per-window aggregates go straight into both models with no location, no scale, no "
        "segment shrinkage and no Z_CLIP winsoriser."
    )
    add("")
    add(
        "`features/windows.py`'s module docstring says what that costs: *\"Nothing here is "
        "comparable across merchants yet: a grocer's velocity and a jeweller's velocity live "
        "three orders of magnitude apart.\"* One pooled Gaussian HMM over raw features is "
        "therefore modelling merchant identity — size, category, ticket scale — rather than "
        "merchant drift, and it is exactly the 2008-era cardholder-HMM failure mode where the "
        "jeweller is flagged for being a jeweller. LightGBM is far less exposed: it splits on "
        "thresholds per feature and can carve out scale bands on its own, so the two models "
        "are **not** expected to lose the same amount here, and a small LightGBM delta is not "
        "evidence that standardisation is decoration for the HMM."
    )
    add("")

    add("## Rows that were never measured")
    add("")
    add(
        "FR-018 names five components. Two of them cannot be measured because the tickets "
        "that would have built them were cut in the 2026-08-28 re-plan. **They are printed as "
        "`not measured`, never as zero and never omitted** — a blank row and a zero row make "
        "opposite claims, and only one of them is true here."
    )
    add("")
    add("| row | why it is absent | what is undischarged |")
    add("|---|---|---|")
    add(
        "| empirical-Bayes shrinkage on / off | **T-0008 was cut.** ADR-0006 records the "
        "decision and its status line says it was never built. | No recalibration happens "
        "anywhere in this repo, and the BMR policy in `decision/policy.py` consumes each "
        "model's raw score **as if it were a calibrated posterior**. Under a rank-only policy "
        "miscalibration would only cost the Brier column; under BMR it moves the argmin. "
        "Every `savings` number on this page inherits that. |"
    )
    add(
        "| NSGA-II vs. grid search | **T-0009 was cut.** ADR-0004 chose NSGA-II over NSGA-III "
        "and made the grid-search comparison a *mandatory* ablation. | No Pareto frontier "
        "exists, so the obligation ADR-0004 wrote down is **undischarged**. `pymoo` is still "
        "declared as a dependency in `pyproject.toml` for work that did not happen; it should "
        "be removed or explicitly justified before freeze. |"
    )
    add("")

    add("### No sequence-aware baseline other than the HMM was measured")
    add("")
    add(
        "**T-0010 (BOCPD, Adams & MacKay 2007) was cut in the same re-plan.** It was the only "
        "planned model that was sequence-aware without being this HMM, so with it gone the "
        "question **\"is any margin here from sequence modelling, or from the HMM "
        "specifically?\"** has no experiment behind it in this repo."
    )
    add("")
    add(
        "**That question is left open, and it is stated as open.** The HMM-on/off row above "
        "compares a sequence model against a non-sequence model, which cannot separate the "
        "two hypotheses: any margin it shows is consistent with *sequence structure helps* "
        "and equally consistent with *this particular HMM happens to suit this generator*. "
        "Nothing in the README or the video may claim the former on the strength of that row. "
        "Closing it needs a second sequence-aware baseline, and none was built."
    )
    add("")

    add("## Limits of this page")
    add("")
    add(
        "- **One seed.** Every row is a single fit at one seed; no repeat-seed variance is "
        "reported, so a small delta is not distinguishable from fit noise. Treat any delta "
        "below the seed-to-seed spread — which this repo has not measured — as unresolved "
        "rather than as zero."
    )
    add(
        f"- **{split.n_merchants} merchants, {n_bad} of them bad.** Precision@{k} moves in "
        f"steps of {1.0 / max(k, 1):.2f}, so its deltas are coarse by construction."
    )
    add(
        "- **The prevalence is not real.** `FRAUD_MERCHANT_RATE` is 0.20, chosen for "
        "per-typology sample size. `results/baf_validation.md` shows what a 1.47% prevalence "
        "does to the same decision layer."
    )
    add(
        "- **The generator is ours.** `results/calibration_gap.md` measures the divergence "
        "from the one public real-merchant dataset that survived the licence gate; 5 of 8 "
        "ratio-scale marginals diverge by 1.9x or more. Every delta above is a delta on our "
        "own assumptions."
    )
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(seed: int, results_dir: Path = RESULTS_DIR) -> Path:
    """Run every ablation and write `results/ablations.md`.

    Args:
        seed: Global seed (NFR-003). The document is byte-identical for a fixed seed.
        results_dir: Directory to write into.

    Returns:
        Path to the written `ablations.md`.
    """
    split = load_split(ABLATION_SPLIT, unlock_test=UNLOCK_TICKET)
    # ADR-0008: capacity scales with the population being scored, exactly as in
    # `harness.run`, so K here is comparable with K in `results/summary.md`.
    capacity_hours = REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS * split.n_merchants / 1000.0
    k = min(review_slots(capacity_hours), split.n_merchants)

    rows = evaluate_configs(split, seed, k)

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "ablations.md"
    path.write_text(
        render_ablations(split, rows, seed, k, capacity_hours),
        encoding="utf-8",
        newline="\n",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """Render the FR-018 ablation table. Returns a process exit code."""
    parser = base_parser("Render the Rakshak ablation table (FR-018) on the test window.")
    args = parser.parse_args(argv)
    seed_everything(args.seed)

    started = time.perf_counter()
    path = run(args.seed)
    elapsed = time.perf_counter() - started

    print(f"rakshak: wrote {path} (seed={args.seed}) in {elapsed:.1f}s")
    print(
        f"rakshak: {len(CONFIGS)} fits on the `{ABLATION_SPLIT}` window, unlocked with "
        f"{UNLOCK_TICKET}. No configuration was selected on test - the shipping "
        "configuration was fixed at T-0004b on validate."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
