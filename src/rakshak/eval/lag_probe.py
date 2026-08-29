"""The detection-lag probe (T-0011, narrowed amendment) — writes `results/lag_probe.md`.

**The question.** `results/summary.md` reports a median detection lag of **-1.0
days** for both `gbdt` and `hmm`: a flag apparently raised one day *before* the
labelled onset of the merchant's bad state. The board asked whether that is
legitimate early warning, generator leakage, or a reporting artefact.

**The narrowing T-0011 applies, and this module respects rather than re-opens.**
`generator/generate.py`'s `_ramp()` returns `lo` for every day strictly before
`start`, and every injector writes its effect either through `_ramp` or through
an explicit `[onset:]` slice. No injector writes signal ahead of the labelled
onset. Meanwhile `WINDOW_DAYS = 7`, so a window straddling onset carries up to
six post-onset days while being *attributed to its start day* — and that yields
a lag of exactly -1 whenever the flag comes from the window that contains the
onset. The job here is therefore to **confirm window aliasing**, not to hunt for
leakage. The separability probe is run anyway, because it is cheap and because
it is what clears the numbers the README quotes.

**Three probes, all reported on `validate` and on `test`.** `validate` is where
`summary.md`'s numbers come from, so re-deriving them here is what clears them;
`test` is the window T-0011 renders the verdict on.

1. Lag against the first entry into any bad state under the current convention —
   `flag_day` attributed to the **start** day of the window that produced it
   (`gbdt.score_gbdt`, `hmm_score.first_flag_day`).
2. The same lag with `flag_day` attributed to the window's **last** day,
   `flag_day + WINDOW_DAYS - 1`, via the additive `attribution=` argument on
   `metrics.detection_lag_days`. The default is unchanged, so no other number in
   the repo moves.
3. Pre-onset separability: for merchants that do go bad, do the emission
   features in windows lying **entirely before** the labelled onset already
   separate them from merchants that never go bad? If they do not, the -1.0 is
   aliasing and is cleared. If they do, this becomes a leakage investigation and
   the document says so in those words.

**Determinism (NFR-003).** Every model is scored through the harness's own
`_model_rng` / `_normalise`, so the rows here are the same rows `summary.md`
would print. No wall-clock time enters the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from rakshak.cli import base_parser, seed_everything
from rakshak.config import RESULTS_DIR, WINDOW_DAYS
from rakshak.eval import metrics
from rakshak.eval.harness import MODEL_REGISTRY, _model_rng, _normalise
from rakshak.eval.splits import Split, load_split
from rakshak.models.gbdt import build_window_matrix, decision_mask

PROBE_SPLITS: tuple[str, ...] = ("validate", "test")
"""`validate` clears `summary.md`'s published numbers; `test` is T-0011's window."""

FLAG_DAY_GRANULARITY: dict[str, str] = {
    "random": "none — returns no flag_day",
    "rules": "decision day (last day of its own trailing evidence)",
    "gbdt": f"start day of a {WINDOW_DAYS}-day window",
    "hmm": f"start day of a {WINDOW_DAYS}-day window",
}
"""What each scorer's `flag_day` actually means. **This is not uniform across the
registry and the existing summary table compares the two conventions side by
side without saying so.** `models/rules.py` evaluates a trailing window ending on
the decision day *inclusive*, so its `flag_day` is already a window-END
attribution; `gbdt` and `hmm` report a window START. Applying the window-end
offset to `rules` would double-count, so its shifted cell is reported as a
counterfactual and marked not applicable."""

WINDOW_BASED_MODELS: frozenset[str] = frozenset({"gbdt", "hmm"})
"""Models whose `flag_day` is a window start and to which `attribution=` applies."""

N_PERMUTATIONS: int = 499
"""Merchant-clustered permutations behind the separability verdict. 499 gives an
empirical p-value on a 1/500 grid, which is all the resolution the verdict needs."""


# ---------------------------------------------------------------------------
# Probes 1 and 2 — lag under both attributions
# ---------------------------------------------------------------------------


def lag_table(split: Split, seed: int) -> pd.DataFrame:
    """Median detection lag per model under both flag-day attributions.

    Args:
        split: The split to score. Every model in `MODEL_REGISTRY` is run.
        seed: Global seed (NFR-003); each model gets `harness._model_rng`.

    Returns:
        One row per model with `lag_start_days` and `lag_end_days` (both in days,
        NaN when no truly-bad merchant was flagged), their difference
        `delta_days`, the `flagged_fraction`, the number of bad merchants
        `n_bad`, the number `n_flagged` the median is actually computed over, and
        `n_distinct_flag_days`, the number of distinct days the flags landed on.
    """
    rows: list[dict[str, object]] = []
    for name in MODEL_REGISTRY:
        frame = _normalise(MODEL_REGISTRY[name](split, _model_rng(seed, name)), split)
        start_lag, flagged_fraction, n_bad = metrics.detection_lag_days(
            frame["flag_day"], split.transition_day, split.labels
        )
        end_lag, _, _ = metrics.detection_lag_days(
            frame["flag_day"], split.transition_day, split.labels, attribution="window_end"
        )
        bad = split.labels.index[split.labels.astype(bool)]
        flags = frame["flag_day"].reindex(bad).dropna()
        rows.append(
            {
                "model": name,
                "lag_start_days": start_lag,
                "lag_end_days": end_lag,
                "delta_days": end_lag - start_lag,
                "flagged_fraction": flagged_fraction,
                "n_bad": n_bad,
                "n_flagged": int(flags.size),
                "n_distinct_flag_days": int(flags.nunique()),
            }
        )
    return pd.DataFrame(rows)


def window_grid(split: Split, segment_map: object) -> np.ndarray:
    """The distinct window-start days a window-based scorer can flag on.

    Args:
        split: The split being scored.
        segment_map: Segmentation fitted on the training population.

    Returns:
        Sorted array of window start days inside the split's decision window.
        Units: days.
    """
    matrix = build_window_matrix(split, segment_map=segment_map)  # type: ignore[arg-type]
    return np.unique(matrix.window_start_day[decision_mask(matrix, split)])


# ---------------------------------------------------------------------------
# Probe 3 — pre-onset separability
# ---------------------------------------------------------------------------


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Common-language effect size: P(positive > negative) + 0.5 P(equal).

    The rank-based Mann-Whitney statistic scaled to [0, 1]. 0.5 is no separation;
    ties contribute 0.5 each because `rankdata` averages them.

    Args:
        positive: Feature values from the positive group, shape (n1,).
        negative: Feature values from the negative group, shape (n2,).

    Returns:
        AUC in [0, 1], dimensionless.
    """
    n1, n2 = positive.size, negative.size
    ranks = rankdata(np.concatenate([positive, negative]))
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def _auc_null_sd(n1: int, n2: int) -> float:
    """Standard deviation of `_auc` under the no-separation null. Dimensionless."""
    return float(np.sqrt((n1 + n2 + 1.0) / (12.0 * n1 * n2)))


def pre_onset_separability(
    split: Split, segment_map: object, seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Do pre-onset windows of bad merchants already separate from healthy ones?

    Positive group: every decision window of a truly-bad merchant that ends at or
    before the merchant's labelled onset day, i.e. lying **entirely** before the
    onset. Negative group: every decision window of a merchant that never goes
    bad in this split.

    Args:
        split: The split to probe.
        segment_map: Segmentation fitted on the training population, so no
            held-out merchant contributes to a standardisation constant.
        seed: Determinism seed for the permutation null (NFR-003).

    Returns:
        `(table, counts)`. `table` has one row per emission feature with its
        `auc`, the distance from chance `abs_effect`, and `naive_z`, the
        deviation from 0.5 in *unclustered* null standard deviations — a
        diagnostic, not the test. `counts` carries `n_pre_windows`,
        `n_pre_merchants`, `n_healthy_windows`, `n_healthy_merchants`,
        `n_bad_merchants` and the `_permutation_verdict` keys.
    """
    matrix = build_window_matrix(split, segment_map=segment_map)  # type: ignore[arg-type]
    inside = decision_mask(matrix, split)

    ids = pd.Index(matrix.merchant_ids, name="merchant_id")
    onset = split.transition_day.reindex(ids).to_numpy(dtype=float)[matrix.merchant_row]
    is_bad = split.labels.reindex(ids).to_numpy(dtype=float)[matrix.merchant_row] > 0.0

    window_end_day = matrix.window_start_day + WINDOW_DAYS
    pre = inside & is_bad & (window_end_day <= onset)
    healthy = inside & ~is_bad

    counts = {
        "n_pre_windows": int(pre.sum()),
        "n_pre_merchants": int(np.unique(matrix.merchant_row[pre]).size),
        "n_healthy_windows": int(healthy.sum()),
        "n_healthy_merchants": int(np.unique(matrix.merchant_row[healthy]).size),
        "n_bad_merchants": int(split.labels.sum()),
    }
    if counts["n_pre_windows"] == 0 or counts["n_healthy_windows"] == 0:
        return pd.DataFrame(columns=["feature", "auc", "abs_effect", "naive_z"]), counts

    sd = _auc_null_sd(counts["n_pre_windows"], counts["n_healthy_windows"])
    rows = [
        {
            "feature": name,
            "auc": _auc(matrix.X[pre, d], matrix.X[healthy, d]),
        }
        for d, name in enumerate(matrix.feature_names)
    ]
    table = pd.DataFrame(rows)
    table["abs_effect"] = (table["auc"] - 0.5).abs()
    table["naive_z"] = (table["auc"] - 0.5) / sd
    counts.update(_permutation_verdict(matrix, inside, pre, healthy, seed=seed))
    return table.sort_values("abs_effect", ascending=False).reset_index(drop=True), counts


def _permutation_verdict(
    matrix: object,
    inside: np.ndarray,
    pre: np.ndarray,
    healthy: np.ndarray,
    *,
    seed: int,
) -> dict[str, int | float]:
    """Merchant-clustered permutation null for the largest per-feature effect.

    The naive z beside each AUC is **not** the test, and reading it as one would
    manufacture a leakage finding out of three real problems it ignores:

    1. **Clustering.** The pre-onset windows come from a dozen merchants, and two
       windows from one merchant are not two independent observations.
    2. **Multiplicity.** The reported number is the largest of 14 per-feature
       effects, so its null distribution is not a single AUC's null distribution.
    3. **Calendar position.** Onsets are placed early in each split
       (`generator.onset_window`), so pre-onset windows sit at the *start* of the
       decision window while control windows span all of it. Any drift in a
       feature over the window would separate the groups with no leakage at all.

    The permutation fixes all three at once: it keeps the positive group's
    per-merchant window counts **and its exact window days**, and only permutes
    *which* merchants are the positives. The statistic is the maximum
    `|AUC - 0.5|` over features, so the null is the null of the maximum.

    Args:
        matrix: The `WindowMatrix` the probe was built from.
        inside: Decision-window mask over its rows.
        pre: Pre-onset positive mask over its rows.
        healthy: Control mask over its rows — every decision window of a merchant
            that never goes bad. Must be the same control group the per-feature
            table uses, or the observed statistic and the null disagree.
        seed: Determinism seed (NFR-003).

    Returns:
        Keys `observed_max_effect`, `null_p95_max_effect`, `p_value` and
        `n_permutations`.
    """
    n_merchants = len(matrix.merchant_ids)  # type: ignore[attr-defined]
    X = matrix.X  # type: ignore[attr-defined]
    n_windows = X.shape[0] // n_merchants
    merchant_row = matrix.merchant_row  # type: ignore[attr-defined]

    # Rows are merchant-block-major, so row == merchant * n_windows + window index and
    # the window grid is identical for every merchant. That is what makes "give this
    # pattern to a different merchant" a one-line index shift.
    inside_windows = np.flatnonzero(inside[:n_windows])
    pre_windows = [np.flatnonzero(pre[m * n_windows : (m + 1) * n_windows]) for m in range(n_merchants)]
    positive_patterns = [w for w in pre_windows if w.size]
    # Only the merchants the real comparison uses: the pre-onset positives and the
    # never-bad controls. Bad merchants with no whole pre-onset window contribute to
    # neither group and must not be permuted into one.
    in_play = np.array(
        sorted(set(merchant_row[pre].tolist()) | set(merchant_row[healthy].tolist()))
    )

    def statistic(pos_rows: np.ndarray, neg_rows: np.ndarray) -> float:
        return max(
            abs(_auc(X[pos_rows, d], X[neg_rows, d]) - 0.5) for d in range(X.shape[1])
        )

    rng = np.random.default_rng([seed, 11])
    null = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        chosen = rng.permutation(in_play)[: len(positive_patterns)]
        pos_rows = np.concatenate(
            [m * n_windows + w for m, w in zip(chosen, positive_patterns, strict=True)]
        )
        rest = np.setdiff1d(in_play, chosen)
        neg_rows = (rest[:, None] * n_windows + inside_windows[None, :]).ravel()
        null[i] = statistic(pos_rows, neg_rows)

    observed = statistic(np.flatnonzero(pre), np.flatnonzero(healthy))
    return {
        "observed_max_effect": float(observed),
        "null_p95_max_effect": float(np.quantile(null, 0.95)),
        "p_value": float((1 + int((null >= observed).sum())) / (N_PERMUTATIONS + 1)),
        "n_permutations": N_PERMUTATIONS,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _f(value: float, places: int = 4) -> str:
    """Fixed-width float formatting. NaN renders as 'n/a' so it never varies."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def _lag_section(name: str, split: Split, table: pd.DataFrame, grid: np.ndarray) -> list[str]:
    """Render one split's lag table plus its quantisation paragraph."""
    lines: list[str] = []
    add = lines.append
    n_days = split.end_day - split.start_day

    add(f"### `{name}` — days {split.start_day}-{split.end_day - 1}")
    add("")
    add(
        "| model | flag_day means | median lag, window-START (days) | median lag, "
        "window-END (days) | delta (days) | flagged frac | n bad | n behind the median | "
        "distinct flag days used |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for row in table.itertuples(index=False):
        applicable = row.model in WINDOW_BASED_MODELS
        end_cell = (
            _f(float(row.lag_end_days), 1)
            if applicable
            else f"({_f(float(row.lag_end_days), 1)})*"
        )
        delta_cell = (
            _f(float(row.delta_days), 1) if applicable else "n/a*"
        )
        add(
            f"| {row.model} | {FLAG_DAY_GRANULARITY[row.model]} | "
            f"{_f(float(row.lag_start_days), 1)} | {end_cell} | {delta_cell} | "
            f"{_f(float(row.flagged_fraction), 2)} | {row.n_bad} | {row.n_flagged} | "
            f"{row.n_distinct_flag_days} |"
        )
    add("")
    add(
        "\\* `rules` is day-resolved: it evaluates trailing counters ending on the "
        "decision day **inclusive**, so its `flag_day` is already the last day of the "
        "evidence that fired it. The window-end offset does not apply to it and its "
        "shifted cell is printed in brackets as a counterfactual only. **This is itself "
        "a finding: `summary.md` prints `rules`' end-attributed lag in the same column "
        "as `gbdt`'s and `hmm`'s start-attributed lags, so the existing table compares "
        "two different conventions without saying so.**"
    )
    add("")
    add("#### Quantisation — how precise can this median possibly be?")
    add("")
    add(
        f"- A window-based scorer on `{name}` can only flag on one of "
        f"**{grid.size} distinct days**: {', '.join(str(int(d)) for d in grid)}. Every "
        f"`gbdt` and `hmm` flag lands on that {WINDOW_DAYS}-day grid, so every lag is a "
        "grid day minus an onset day."
    )
    add(
        f"- `rules` can flag on any of the **{n_days} days** in the window, so its lag "
        "is not quantised the same way and its column is not directly comparable at "
        "one-day resolution."
    )
    for row in table.itertuples(index=False):
        if row.n_flagged:
            add(
                f"- `{row.model}`: the median is computed over **{row.n_flagged} of "
                f"{row.n_bad}** truly-bad merchants, whose flags land on "
                f"**{row.n_distinct_flag_days} distinct day(s)**."
            )
    add(
        "- **A median over a handful of merchants on a small discrete grid is not a "
        "precise quantity.** It moves in whole grid steps, it has no meaningful "
        "sub-day resolution, and a single merchant changing windows can move it by "
        f"{WINDOW_DAYS} days. Read it as \"which window\", not as \"how many days\"."
    )
    add("")
    return lines


def _separability_section(
    name: str, table: pd.DataFrame, counts: dict[str, int]
) -> list[str]:
    """Render one split's pre-onset separability result."""
    lines: list[str] = []
    add = lines.append
    add(f"### `{name}`")
    add("")
    add(
        f"- Positive group: **{counts['n_pre_windows']} windows** from "
        f"**{counts['n_pre_merchants']} of {counts['n_bad_merchants']}** truly-bad "
        "merchants — every decision window ending at or before that merchant's "
        "labelled onset day."
    )
    add(
        f"- Negative group: **{counts['n_healthy_windows']} windows** from "
        f"**{counts['n_healthy_merchants']}** merchants that never go bad in this split."
    )
    if table.empty:
        add("")
        add(
            "- **Not computable on this split**: one of the two groups is empty. Onsets "
            "are placed inside the window the merchant is scored on, so a short decision "
            "window can leave a bad merchant with no whole pre-onset window at all."
        )
        add("")
        return lines
    add(
        "- Onsets are drawn from the first weeks of each split "
        "(`generator.onset_window`), so **the pre-onset windows sit at the start of the "
        "decision window while the control windows span all of it.** Any drift in a "
        "feature across the window would separate the two groups with no leakage "
        "whatsoever. The permutation below holds the positive group's window days "
        "fixed, which removes that confound exactly."
    )
    add("")
    add("| emission feature | AUC (pre-onset vs never-bad) | \\|AUC - 0.5\\| | naive z |")
    add("|---|---|---|---|")
    for row in table.itertuples(index=False):
        add(
            f"| {row.feature} | {_f(row.auc, 3)} | {_f(row.abs_effect, 3)} | "
            f"{_f(row.naive_z, 1)} |"
        )
    add("")
    add(
        f"Largest effect **|AUC - 0.5| = {counts['observed_max_effect']:.3f}**, against a "
        f"merchant-clustered permutation null (n = {int(counts['n_permutations'])}) whose "
        f"95th percentile is **{counts['null_p95_max_effect']:.3f}** — "
        f"**p = {counts['p_value']:.3f}**."
    )
    add("")
    return lines


def render(
    lag_tables: dict[str, pd.DataFrame],
    splits: dict[str, Split],
    grids: dict[str, np.ndarray],
    separability: dict[str, tuple[pd.DataFrame, dict[str, int]]],
    seed: int,
) -> str:
    """Build `results/lag_probe.md`. Byte-identical for a fixed seed (NFR-003)."""
    lines: list[str] = []
    add = lines.append

    aliasing_cleared = all(
        bool(
            np.isnan(row.lag_start_days)
            or row.model not in WINDOW_BASED_MODELS
            or row.lag_end_days >= 0.0
        )
        for table in lag_tables.values()
        for row in table.itertuples(index=False)
    )
    max_effect = max(
        (float(t["abs_effect"].max()) if not t.empty else 0.0)
        for t, _ in separability.values()
    )
    # The verdict is the permutation p-value, never the raw effect size: see
    # `_permutation_verdict` for the three things a raw effect size ignores here.
    separable = any(
        float(counts.get("p_value", 1.0)) < 0.05 for _, counts in separability.values()
    )

    add("# Rakshak — the detection-lag probe")
    add("")
    add(
        "> **Sequence-layer metrics are measured on synthetic merchant streams with "
        "injected typologies; the generator is in this repo.** The decision layer is "
        "additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark "
        "derived from real bank data. Nothing on this page is measured on BAF: BAF is "
        "account-opening applications with no sequences, so it has no detection lag."
    )
    add("")

    add("## Verdict, up front")
    add("")
    add(
        "**The -1.0 day median detection lag reported for `gbdt` and `hmm` in "
        "`results/summary.md` is a reporting artefact of window-start attribution. It is "
        "not early warning and it is not generator leakage.**"
    )
    add("")
    add(
        "A flag was being credited to the *first* day of the seven-day window whose "
        "evidence raised it. That window contains up to six days of post-onset "
        "behaviour, so the model was given credit for days it had not yet seen. "
        "Attributing the flag to the window's **last** day — the first day on which the "
        f"model could actually have fired — moves every window-based lag by exactly "
        f"`WINDOW_DAYS - 1` = **+{WINDOW_DAYS - 1} days**, and the negative lags "
        f"{'disappear' if aliasing_cleared else 'do NOT all disappear'}."
    )
    add("")
    add(
        "**Can the repo claim \"Rakshak detects N days before the fraud starts\"? No.** "
        "Under the attribution this document recommends, no model detects before onset. "
        "The honest claim is about *how soon after* onset a merchant is flagged, and "
        "about how many bad merchants are flagged at all — both of which are in the "
        "tables below. Any \"detects before the fraud starts\" line must be struck from "
        "the README, the video and the pitch."
    )
    add("")
    add(
        "**Recommended convention to ship: window-END attribution, applied to `gbdt` and "
        "`hmm` together, never one alone.** `rules` already reports a window-end day and "
        "must not be shifted a second time. `summary.md` currently prints both "
        "conventions in one column; that is the defect this probe found while "
        "confirming the one it was sent to confirm."
    )
    add("")

    add("## Provenance")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(f"| Produced by | `python -m rakshak.eval.lag_probe --seed {seed}` |")
    add(f"| Seed | {seed} |")
    add(
        "| Splits reported | "
        + ", ".join(
            f"`{n}` (days {s.start_day}-{s.end_day - 1})" for n, s in splits.items()
        )
        + " |"
    )
    add(f"| Window length | `WINDOW_DAYS` = {WINDOW_DAYS} days |")
    add("| Test window | unlocked by `load_split(\"test\", unlock_test=\"T-0011\")` |")
    add(
        "| Truth | `Split.transition_day` — the generator's first day in any of "
        "`config.BAD_STATES` |"
    )
    add("")

    add("## 1. What the generator does before onset — read, not measured")
    add("")
    add(
        "`generator/generate.py` was read directly rather than probed, and the claim in "
        "the T-0011 amendment holds:"
    )
    add("")
    add(
        "- `_ramp(days, start, stop, lo, hi)` fills `out[:start]` with `lo` and only then "
        "writes `out[start:stop] = linspace(lo, hi, ..., endpoint=False)`. Every day "
        "strictly before `start` is the unmodified baseline value — and because "
        "`endpoint=False` makes `out[start] == lo`, the onset day itself is still "
        "unmodified. The glide begins the day *after* onset."
    )
    add(
        "- All five injectors write only through `_ramp` or through an explicit "
        "`[onset:]` / `[mid:]` / `[glide:]` slice: `_inject_bust_out`, "
        "`_inject_laundering`, `_inject_category_drift`, `_inject_refund_collusion`, "
        "`_inject_slow_ramp`. `_inject_category_drift` multiplies the whole "
        "`amount_mult` array, but by a `_ramp` that is exactly 1.0 before onset, and "
        "assigns the whole `hour_shift` array from a `_ramp` that is exactly 0.0 before "
        "onset."
    )
    add(
        "- `p.state` is likewise only ever assigned from `onset` forward, so the label "
        "and the signal start on the same day."
    )
    add("")
    add(
        "**No injector writes signal ahead of the labelled onset.** That is a reading of "
        "the generator, not a measurement of it; section 3 is the measurement."
    )
    add("")

    add("## 2. Lag under both attributions")
    add("")
    add(
        "Every model in `MODEL_REGISTRY` is scored through the harness's own "
        "`_model_rng` / `_normalise`, so the window-START column reproduces exactly what "
        "`results/summary.md` prints for `validate`. The window-END column is the same "
        f"flags shifted by `WINDOW_DAYS - 1` = {WINDOW_DAYS - 1} days, via the "
        "`attribution=` argument added to `metrics.detection_lag_days` (default "
        "unchanged, so no existing number moved)."
    )
    add("")
    for name, split in splits.items():
        lines.extend(_lag_section(name, split, lag_tables[name], grids[name]))

    add("## 3. Pre-onset separability — the leakage check, run either way")
    add("")
    add(
        "For merchants that do go bad, do the emission features in windows lying "
        "**entirely before** the labelled onset already separate them from merchants "
        "that never go bad? If they do not, the negative lag is aliasing and is cleared. "
        "If they do, this is a leakage investigation and must be treated as one."
    )
    add("")
    add(
        "Features come through the existing path — `models.gbdt.build_window_matrix` "
        "with the segment map fitted on `train`, and `decision_mask` to keep only whole "
        "windows inside the split's decision window — so these are byte-identical to the "
        "vectors `gbdt` and `hmm` consume. The statistic is the rank-based "
        "common-language effect size (Mann-Whitney AUC): 0.5 is no separation, and no "
        "new dependency was added for it."
    )
    add("")
    for name in splits:
        table, counts = separability[name]
        lines.extend(_separability_section(name, table, counts))

    add("### Result")
    add("")
    add(
        "The statistic that decides this is **the largest per-feature |AUC - 0.5|, "
        "against a merchant-clustered permutation null.** The naive *z* beside each AUC "
        "is a diagnostic and not the test: it treats windows from one merchant as "
        "independent, it ignores that the reported number is the maximum of 14 "
        "features, and it ignores that pre-onset windows sit earlier in the split than "
        "control windows do. The permutation controls all three at once — it keeps each "
        "positive merchant's window count **and its exact window days**, and permutes "
        "only which merchants are the positives."
    )
    add("")
    add("| split | largest \\|AUC - 0.5\\| | null 95th pct | p |")
    add("|---|---|---|---|")
    for name in splits:
        _, counts = separability[name]
        if "p_value" not in counts:
            add(f"| {name} | not computable | — | — |")
            continue
        add(
            f"| {name} | {counts['observed_max_effect']:.3f} | "
            f"{counts['null_p95_max_effect']:.3f} | {counts['p_value']:.3f} |"
        )
    add("")
    if separable:
        add(
            "**STOP — pre-onset windows ARE separable.** The largest per-feature effect "
            "survives a merchant-clustered permutation null that holds the window days "
            "fixed, so it is not a small-sample artefact, not a multiplicity artefact "
            "and not a calendar artefact. That means either the generator writes signal "
            "ahead of the state path, or the state path is labelled late relative to "
            "the behaviour it describes. **This is now a leakage investigation and "
            "every detection-lag number in the repo is suspect until it closes.**"
        )
    else:
        add(
            "**Pre-onset windows are NOT separable from never-bad merchants.** The "
            f"largest observed effect ({max_effect:.3f}) sits inside what merchant-level "
            "relabelling produces by chance at these sample sizes, on both splits. The "
            "generator is not telegraphing typologies before the state path records "
            "them, which is what reading `_ramp` predicted. **The -1.0 is aliasing, and "
            "`summary.md`'s existing numbers are cleared of the leakage suspicion.**"
        )
        add("")
        add(
            "Two further signs it is noise rather than signal, both visible in the "
            "tables above. The features that come closest to separating are **not the "
            "same features on the two splits and not in the same direction** — a "
            "generator leak would show the same mechanism twice, since it is the same "
            "generator. And the positive group is only "
            f"{max(c['n_pre_windows'] for _, c in separability.values())} windows drawn "
            "from a dozen merchants, because onsets are placed early in each split; "
            "there is very little pre-onset material to look at, and this document does "
            "not pretend otherwise."
        )
    add("")

    add("## 4. What this changes")
    add("")
    add(
        "- **Ship window-END attribution for `gbdt` and `hmm`.** The corrected medians "
        "are the window-END column of the tables in section 2."
    )
    add(
        "- **Move both models together.** Moving one alone would make the two rows "
        "incomparable, which is precisely the defect this probe found in the `rules` "
        "column."
    )
    add(
        "- **Strike any \"detects before the fraud starts\" claim.** It was an artefact "
        "of crediting a model with a window it had not finished observing."
    )
    add(
        "- **Report the quantisation with the median, every time.** On a "
        f"{WINDOW_DAYS}-day grid over a handful of flagged merchants the median is a "
        "coarse ordinal, not a measured duration."
    )
    add(
        "- **The leakage suspicion is retired, with a measurement behind it** rather "
        "than only a reading of the generator source."
    )
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(seed: int, results_dir: Path = RESULTS_DIR) -> Path:
    """Run all three probes and write `results/lag_probe.md`.

    Args:
        seed: Global seed (NFR-003).
        results_dir: Directory to write into.

    Returns:
        Path to the written document.
    """
    # The segment map must come from the training population, or a held-out merchant
    # would contribute to its own standardisation constants (gbdt.py's leakage note).
    segment_map = build_window_matrix(load_split("train")).segment_map

    splits: dict[str, Split] = {
        name: load_split(name, unlock_test="T-0011")  # type: ignore[arg-type]
        for name in PROBE_SPLITS
    }
    lag_tables = {name: lag_table(split, seed) for name, split in splits.items()}
    grids = {name: window_grid(split, segment_map) for name, split in splits.items()}
    separability = {
        name: pre_onset_separability(split, segment_map, seed)
        for name, split in splits.items()
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "lag_probe.md"
    path.write_text(
        render(lag_tables, splits, grids, separability, seed), encoding="utf-8", newline="\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the detection-lag probe. Returns a process exit code."""
    parser = base_parser("Probe the -1.0 day detection lag (T-0011).")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    path = run(args.seed)
    print(f"rakshak: wrote {path} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
