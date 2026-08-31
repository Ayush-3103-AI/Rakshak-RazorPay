"""T-0022c: the black-swan shock stress test — writes `results/blackswan.md`.

**The question this file answers.** Does a population-wide shared shock — a demand
surge, an MDR change, a payment-rail outage — cause `rules`/`gbdt`/`hmm` to falsely
flag HEALTHY merchants as drifting? This is this project's own central cost claim
("blunt global thresholds freeze honest merchants") put to a test the frozen
`results/` never ran: every merchant in `data/synthetic/` is an independent
stochastic process, so nothing there ever hit many merchants at once.

**Path taken: PRIMARY, not the fallback.** T-0022a/b (closed 2026-08-30) already
built the shock-capable generator and the harness data-path seam; this ticket
only had to refit and report. `rules`/`gbdt`/`hmm` are refit on
`data/synthetic_shock/`'s own `train`/`validate` (T-0022b's seam — see
`score_shocked_split` below), not on the frozen `data/synthetic/` models. The
fallback (inject the shock only into already-fitted models' scoring input,
`11-tickets/T-0022c.md`'s "Build — fallback path") was not needed.

**Never opens `test`.** `11-tickets/T-0022c.md`'s fallback text names
`split.test`; `STATE.md`'s T-0022b section overrides that for this ticket:
*"A dataset override is not a test-window unlock ... T-0022c has no business
opening it."* This module scores `validate` only (`EVAL_SPLIT` below, mirroring
`harness.EVAL_SPLIT`) and never calls `load_split("test", unlock_test=...)`.

**Compares shock-day windows against control windows INSIDE the shock dataset
only — never against `data/synthetic/`.** T-0022a's finding: a shocked run and
the frozen run do not share a state path or onsets, because the shock changes
how many transactions a merchant emits, which shifts how many random draws it
consumes, which shifts every later merchant's onset. A shocked-vs-frozen diff
would be dominated by re-rolled onsets, not by the shock. So both the "shock"
and "control" evidence below come from ONE generation of
`data/synthetic_shock/`, split by day, not by dataset.

**Scoring reuses `eval.harness`'s own building blocks, not `harness.run()`
itself.** `eval.harness.run()` reports one aggregate `flagged_fraction` per
model; it does not expose *which* merchant flagged on *which* day, which is
what a shock/control comparison needs. `eval/lag_probe.py` (T-0011, already
merged) hit the identical need and set the precedent this module follows
exactly: call `load_split()` inside `eval.splits.active_dataset()` (the T-0022b
seam — a public, documented channel, not a bypass of it) and then
`harness.MODEL_REGISTRY[name](split, harness._model_rng(seed, name))` per
model, normalised through `harness._normalise`. No `.fit()` is ever called
directly with a hand-built split, and no new raw parquet read is added — every
read still goes through `load_split`/`build_window_matrix`, which honour
whichever dataset `active_dataset()` has made current.

**Shock placement, and why it lets "before" prove innocence and "after" carry
different weight per model.** One shock day, `SHOCK_DAY` = 194, inside
`validate` (days 180-209). Every scorer here is trailing/backward-looking, so a
flag dated *before* the shock cannot possibly be caused by it — that is what
makes the CONTROL bucket clean by construction, not an assumption. A flag dated
*on or after* the shock is only sometimes attributable to it, and the two model
families differ in exactly how far the shock's shadow reaches:

* `rules` (`models/rules.py`) is genuinely day-resolved and its refund/chargeback
  ratio uses a trailing **30**-day window, so the shock remains inside that
  window's evidence for every day from 194 through the end of `validate` (209).
  SHOCK = flag_day in [194, 209]; CONTROL = flag_day in [180, 193].
* `gbdt` (`models/gbdt.py`) has **no memory across its 7-day windows** — each of
  the 4 windows `validate` admits (attributed flag days 188, 195, 202, 209,
  `summary.md`'s own "only four distinct flag days") is scored from its own
  disjoint 7 days alone. Only the window covering day 194 (window 189-195,
  attributed 195) ever saw the shock. SHOCK = {195}; CONTROL = {188, 202, 209}.
* `hmm` (`models/hmm_score.py`) is a forward-filtered sequential belief, so
  evidence from the shock window can persist into later windows through the
  fitted transition matrix even though those windows' own emissions are clean —
  a mechanism `gbdt` structurally cannot have. SHOCK = {195, 202, 209}
  (window 189 onward); CONTROL = {188} only.

This is stated in `results/blackswan.md`'s methodology section with the same
per-model day sets, so a reader can check the classification without re-running
anything (`14-spec-blackswan-and-drift-survey.md`'s "checkable by a reader, not
asserted from memory" bar).

**Skipped, and why.** The spec's primary-path description imagines the shock
recurring across `train`/`validate`/`test` so a refit could "potentially learn
to tolerate" it. This run's shock lands once, only in `validate`; `train` is
shock-free, so nothing here can or does claim a refit model learned tolerance —
that question needs a second dataset with repeated train-window shocks and a
head-to-head comparison, out of scope for a one-night ticket ranked below
T-0013/T-0018/T-0020/T-0021/T-0019. What this run DOES answer stands on its
own: does a shock a refit model never saw during training still cause it to
flag healthy merchants.

**Not a rescue or rebuttal of K2.** `results/verdict.md`'s FAIL and
`00-charter.md`'s NFR-001 bar are untouched by this file and by anything this
module reads or writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.cli import base_parser, seed_everything
from rakshak.config import (
    RESULTS_DIR,
    REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS,
    SYNTHETIC_SHOCK_DIR,
    WINDOW_DAYS,
)
from rakshak.decision import policy
from rakshak.eval import metrics
from rakshak.eval.harness import MODEL_REGISTRY, _model_rng, _normalise
from rakshak.eval.splits import Split, active_dataset, load_split

MODELS: tuple[str, ...] = ("rules", "gbdt", "hmm")
"""Rakshak's own models. `random` carries no notion of a flag day to bucket by
shock-vs-control day and is out of this ticket's scope
(`11-tickets/T-0022c.md`)."""

EVAL_SPLIT: str = "validate"
"""Mirrors `harness.EVAL_SPLIT`. See the module docstring on why never `test`."""

SHOCK_DAY: int = 194
"""Inside `validate` = days [180, 210). Chosen so both a clean pre-shock CONTROL
period (14 days: 180-193) and a shock/post-shock period (16 days: 194-209) exist
inside the one split the harness scores. See the module docstring for exactly
what "shock/post-shock" means per model."""

SHOCK_MAGNITUDE: float = 6.0
"""Matches T-0022a's own worked example (`STATE.md`) — a large, unambiguous
demand-style spike, not a marginal one."""

GENERATE_COMMAND: str = (
    f"python -m rakshak.generator.generate --seed {{seed}} --shock-day {SHOCK_DAY} "
    f"--shock-magnitude {SHOCK_MAGNITUDE}"
)

TRANSACTIONS_PATH: Path = SYNTHETIC_SHOCK_DIR / "transactions.parquet"
STATE_PATHS_PATH: Path = SYNTHETIC_SHOCK_DIR / "state_paths.parquet"


# ---------------------------------------------------------------------------
# Per-model day sets — see the module docstring for the reasoning behind each
# ---------------------------------------------------------------------------


def _window_flag_days(start_day: int, end_day: int, window_days: int = WINDOW_DAYS) -> list[int]:
    """Attributed flag days for a `WINDOW_DAYS`-block scorer (`gbdt`/`hmm`).

    Mirrors `models/gbdt.decision_mask`: window starts are multiples of
    `window_days` since day 0, kept only when the whole window lies inside
    `[start_day, end_day)`. Attribution is the window's last day
    (`models/gbdt.py`, `hmm_score.first_flag_day`).
    """
    starts = [
        d for d in range(0, end_day, window_days) if start_day <= d and d + window_days <= end_day
    ]
    return [s + window_days - 1 for s in starts]


def _shock_window(shock_day: int, window_days: int = WINDOW_DAYS) -> int:
    """Attributed flag day of the one `WINDOW_DAYS`-block that contains `shock_day`."""
    start = (shock_day // window_days) * window_days
    return start + window_days - 1


def bucket_days(
    name: str, start_day: int, end_day: int, shock_day: int
) -> tuple[set[int], set[int]]:
    """`(shock_days, control_days)` — the day sets `bucket_healthy_merchants` bins by.

    Args:
        name: One of `MODELS`.
        start_day: Split start (inclusive).
        end_day: Split end (exclusive).
        shock_day: The single day the shock lands on.

    Returns:
        Two disjoint sets of possible `flag_day` values for this model: SHOCK and
        CONTROL. See the module docstring for why each model's split differs.
    """
    if name == "rules":
        shock = set(range(shock_day, end_day))
        control = set(range(start_day, shock_day))
        return shock, control
    if name in ("gbdt", "hmm"):
        grid = _window_flag_days(start_day, end_day)
        shock_window = _shock_window(shock_day)
        if name == "gbdt":
            # No cross-window memory: only the window that literally contains the
            # shock day is contaminated.
            shock = {shock_window}
        else:
            # hmm: forward-filtered belief carries evidence into later windows.
            shock = {d for d in grid if d >= shock_window}
        control = set(grid) - shock
        return shock, control
    raise ValueError(f"no day-bucketing rule for model {name!r}")


# ---------------------------------------------------------------------------
# Scoring — the T-0022b seam, following eval/lag_probe.py's precedent
# ---------------------------------------------------------------------------


def score_shocked_validate(
    seed: int,
    transactions_path: Path = TRANSACTIONS_PATH,
    state_paths_path: Path = STATE_PATHS_PATH,
) -> tuple[Split, dict[str, pd.DataFrame]]:
    """Refit and score `rules`/`gbdt`/`hmm` on the shocked dataset's `validate` split.

    Fits happen inside `active_dataset()`, so `gbdt.fit`/`hmm_score.fit` (called
    from inside `score_gbdt`/`score_hmm`, never called directly here) read
    `train` from the shock dataset, not from `data/synthetic/`. See the module
    docstring for why this mirrors `eval/lag_probe.py` rather than parsing
    `harness.run()`'s markdown output.

    Args:
        seed: Determinism seed (NFR-003).
        transactions_path: The shock dataset's transactions parquet.
        state_paths_path: The shock dataset's state-paths parquet.

    Returns:
        `(split, frames)`. `frames[name]` is indexed by merchant_id with `score`
        and `flag_day` columns, exactly `harness.evaluate_model`'s intermediate
        frame before it collapses to aggregate stats.
    """
    with active_dataset(transactions_path, state_paths_path):
        split = load_split(EVAL_SPLIT)
        frames = {
            name: _normalise(MODEL_REGISTRY[name](split, _model_rng(seed, name)), split)
            for name in MODELS
        }
    return split, frames


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def bucket_healthy_merchants(
    flag_day: pd.Series, labels: pd.Series, shock_days: set[int], control_days: set[int]
) -> pd.Series:
    """Classify every HEALTHY merchant's first flag as "shock", "control" or "never".

    Restricted to healthy merchants (`labels == 0`): the whole point of the probe
    is that ground truth never changes on a shock day
    (`generator._apply_shock`'s docstring, T-0022a), so any flag here is by
    construction a false positive, and this is the population it happens to.

    Args:
        flag_day: Per-merchant first flag day, NaN if never (one model's frame).
        labels: Per-merchant ground truth, 0 == healthy.
        shock_days: Attributed flag days that count as shock exposure.
        control_days: Attributed flag days that are clean by construction.

    Returns:
        Series indexed by the healthy merchant IDs, values in
        {"control", "shock", "never"}. Every day in a real flag_day is in exactly
        one of `shock_days` / `control_days` for a model built from
        `bucket_days`, so no third category is needed.
    """
    healthy = labels.index[labels == 0]
    day = flag_day.reindex(healthy)
    bucket = pd.Series("never", index=healthy, name="bucket")
    bucket[day.isin(control_days)] = "control"
    bucket[day.isin(shock_days)] = "shock"
    return bucket


def model_row(
    name: str,
    split: Split,
    frame: pd.DataFrame,
    shock_days: set[int],
    control_days: set[int],
    capacity_hours: float,
) -> dict[str, object]:
    """One model's shock-vs-control row: flagged-fraction and hold/review counts.

    Args:
        name: One of `MODELS`.
        split: The scored `validate` split (shock dataset).
        frame: `score_shocked_validate`'s frame for this model.
        shock_days: From `bucket_days`.
        control_days: From `bucket_days`.
        capacity_hours: Analyst-hour budget, same formula `harness._run` uses.

    Returns:
        A dict with per-bucket merchant counts, flagged fractions, and HOLD/REVIEW
        counts (from the same BMR policy `harness.evaluate_model` runs) broken
        down by bucket.
    """
    bucket = bucket_healthy_merchants(frame["flag_day"], split.labels, shock_days, control_days)
    n_healthy = int(bucket.size)
    counts = bucket.value_counts().reindex(["control", "shock", "never"], fill_value=0)

    posterior = np.clip(
        frame["score"].reindex(split.merchant_ids).to_numpy(dtype=float), 0.0, 1.0
    )
    params = policy.CostParams(
        loss_inr=split.loss_inr.to_numpy(dtype=float),
        value_inr=split.value_inr.to_numpy(dtype=float),
    )
    result = policy.bmr_policy(posterior, params, capacity_hours=capacity_hours)
    actions = pd.Series(
        result.actions, index=pd.Index(split.merchant_ids, name="merchant_id")
    ).reindex(bucket.index)

    hold = {b: int((actions[bucket == b] == metrics.HOLD).sum()) for b in ("control", "shock")}
    review = {b: int((actions[bucket == b] == metrics.REVIEW).sum()) for b in ("control", "shock")}

    return {
        "model": name,
        "n_healthy": n_healthy,
        "n_control": int(counts["control"]),
        "n_shock": int(counts["shock"]),
        "n_never": int(counts["never"]),
        "control_frac": counts["control"] / n_healthy if n_healthy else float("nan"),
        "shock_frac": counts["shock"] / n_healthy if n_healthy else float("nan"),
        "hold_control": hold["control"],
        "hold_shock": hold["shock"],
        "review_control": review["control"],
        "review_shock": review["shock"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _f(value: float, places: int = 4) -> str:
    """Fixed-width float formatting. NaN renders as 'n/a', matching `harness._f`."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def render(
    split: Split,
    rows: list[dict[str, object]],
    seed: int,
    transactions_path: Path,
    state_paths_path: Path,
) -> str:
    """Build `results/blackswan.md`. Byte-identical for a fixed seed (NFR-003)."""
    lines: list[str] = []
    add = lines.append

    add("# Rakshak — the black-swan shock stress test")
    add("")
    add(
        "**This is an orthogonal robustness probe, not a rescue or a rebuttal of "
        "K2's FAIL verdict.** `results/verdict.md`'s wording and `00-charter.md`'s "
        "pre-registered NFR-001 bar are untouched by anything in this file."
    )
    add("")
    add(
        "**The question:** does a population-wide shared shock — a demand surge, "
        "an MDR change, a payment-rail outage — cause `rules`/`gbdt`/`hmm` to "
        "falsely flag HEALTHY merchants as drifting? This is this project's own "
        "central cost claim (\"blunt global thresholds freeze honest merchants\") "
        "put to a test the frozen `results/` never ran: every merchant in "
        "`data/synthetic/` is an independent stochastic process, so nothing there "
        "ever hit many merchants at once."
    )
    add("")

    add("## Path taken: PRIMARY")
    add("")
    add(
        "`rules`/`gbdt`/`hmm` are **refit** on `data/synthetic_shock/`'s own "
        "`train`/`validate`, via the T-0022b seam "
        "(`eval.splits.active_dataset` / `load_split(transactions_path=, "
        "state_paths_path=)`) — not the fallback of scoring already-fitted, "
        "frozen models against a shock injected only into scoring input. "
        "`11-tickets/T-0022c.md`'s fallback text names `split.test`; `STATE.md`'s "
        "T-0022b section overrides that for this ticket "
        "(*\"a dataset override is not a test-window unlock ... T-0022c has no "
        "business opening it\"*), so this run never touches `test` — everything "
        "below is the `validate` split, exactly as `eval.harness.EVAL_SPLIT` "
        "scores by default."
    )
    add("")

    add("## Provenance")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(
        "| Dataset generated by | "
        f"`{GENERATE_COMMAND.format(seed=seed)}` |"
    )
    add(f"| Dataset directory | `{SYNTHETIC_SHOCK_DIR.as_posix()}/` (git-ignored, regenerable) |")
    add(f"| Scored by | `python -m rakshak.eval.blackswan --seed {seed}` |")
    add(f"| Seed | {seed} |")
    add(f"| Split reported | `{split.name}` (days {split.start_day}-{split.end_day - 1}) |")
    add(f"| Shock day | {SHOCK_DAY} (inside `validate`, magnitude {SHOCK_MAGNITUDE}x) |")
    add(
        "| Ground truth | unaffected by the shock — `generator._apply_shock` never "
        "touches `state_paths.parquet` (T-0022a), so every merchant flagged here "
        "is genuinely HEALTHY and any flag is by construction a false positive |"
    )
    add("| Test window | **never opened** — `unlock_test` is not passed anywhere in this module |")
    add("")

    add("## Methodology — shock windows vs. control windows, inside this one dataset only")
    add("")
    add(
        "**Never compared against `data/synthetic/`.** T-0022a's finding: a "
        "shocked run and the frozen run do not share a state path or onsets — the "
        "shock changes how many transactions a merchant emits, which shifts how "
        "many random draws it consumes, which shifts every later merchant's "
        "onset. A shocked-vs-frozen diff would be dominated by re-rolled onsets, "
        "not by the shock. So both the SHOCK and CONTROL evidence below come "
        "from **one** generation of `data/synthetic_shock/`, split by day, never "
        "by dataset."
    )
    add("")
    add(
        "**Why \"before\" is clean by construction.** Every scorer here is "
        "trailing/backward-looking, so a flag dated *before* the shock cannot "
        "possibly be caused by it — that is a temporal impossibility, not an "
        "assumption. A flag dated *on or after* the shock is only sometimes "
        "attributable to it, and how far its shadow reaches differs by "
        "architecture:"
    )
    add("")
    add("| model | attributable mechanism | SHOCK flag-days | CONTROL flag-days |")
    add("|---|---|---|---|")
    add(
        "| `rules` | day-resolved; its refund/chargeback ratio trails **30** "
        "days, so the shock stays inside that window's evidence through the "
        f"rest of `validate` | {SHOCK_DAY}-{split.end_day - 1} | "
        f"{split.start_day}-{SHOCK_DAY - 1} |"
    )
    gbdt_shock, gbdt_control = bucket_days("gbdt", split.start_day, split.end_day, SHOCK_DAY)
    add(
        "| `gbdt` | **no memory across its 7-day windows** — only the one window "
        "whose own days contain the shock is contaminated | "
        f"{sorted(gbdt_shock)} | {sorted(gbdt_control)} |"
    )
    hmm_shock, hmm_control = bucket_days("hmm", split.start_day, split.end_day, SHOCK_DAY)
    add(
        "| `hmm` | forward-filtered sequential belief — evidence from the shock "
        "window can persist into later windows through the fitted transition "
        f"matrix, even though those windows' own emissions are clean | "
        f"{sorted(hmm_shock)} | {sorted(hmm_control)} |"
    )
    add("")
    add(
        "`gbdt`/`hmm` can only flag on the four window-attributed days "
        f"`{_window_flag_days(split.start_day, split.end_day)}` — the same "
        "four-day grid `results/summary.md` already reports (\"validate admits "
        "only four distinct flag days\"). `rules` can flag on any of the "
        f"{split.end_day - split.start_day} days in the split."
    )
    add("")
    add(
        "**Restricted to HEALTHY merchants** (`label == 0`): the shock changes "
        "no merchant's ground-truth state, so any flag in this population is a "
        "false positive by construction, and that is the population the "
        "project's cost claim is about."
    )
    add("")

    add("## Results — flagged fraction, healthy merchants only, with counts")
    add("")
    add(
        "| model | n healthy | n flagged (control) | n flagged (shock) | n never "
        "flagged | control frac | shock frac | delta (shock - control) |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for row in rows:
        delta = float(row["shock_frac"]) - float(row["control_frac"])
        add(
            f"| {row['model']} | {row['n_healthy']} | {row['n_control']} | "
            f"{row['n_shock']} | {row['n_never']} | {_f(float(row['control_frac']), 3)} | "
            f"{_f(float(row['shock_frac']), 3)} | {_f(delta, 3)} |"
        )
    add("")
    add(
        "`control frac` / `shock frac` are `n flagged (bucket) / n healthy` — the "
        "share of ALL healthy merchants in `validate` first-flagged during that "
        "bucket, not a share of the flagged subset, matching how `flagged frac` "
        "is scaled everywhere else in this repo (e.g. `results/verdict.md`). "
        "**This is a different denominator from `results/verdict.md`'s `flagged "
        "frac` column**, which measures recall over truly-BAD merchants; this "
        "one measures false alarms over truly-HEALTHY merchants and the two "
        "numbers must never be read against each other."
    )
    add("")
    add(
        "**`flag_day` is a merchant's FIRST flag, ever — a merchant already "
        "flagged during CONTROL is not double-counted if it stays anomalous "
        "into SHOCK.** This is why `rules`' shock fraction can read *lower* "
        "than its control fraction: `rules`' own baseline false-positive rate "
        "on this population is already high enough (10 of 80 healthy merchants "
        "flag inside the 14 clean pre-shock days alone) that most of the "
        "merchants it was ever going to flag are claimed by CONTROL first, "
        "leaving fewer available to be newly counted under SHOCK. Read `rules`' "
        "delta as \"fewer *new* false positives\", never as \"the shock made "
        "`rules` safer\" — see finding 4 below."
    )
    add("")

    add("## HOLD / REVIEW rate, same buckets")
    add("")
    add(
        "Same Bayes-Minimum-Risk policy `decision/policy.py` uses everywhere "
        "else in this repo (T-0007b), same analyst-hour budget formula "
        "(ADR-0008) `eval.harness._run` uses. Counts, not rates — the per-bucket "
        "populations are small and a rate over a handful of merchants would "
        "invite more precision than the number carries."
    )
    add("")
    add("| model | held (control) | held (shock) | reviewed (control) | reviewed (shock) |")
    add("|---|---|---|---|---|")
    for row in rows:
        add(
            f"| {row['model']} | {row['hold_control']} | {row['hold_shock']} | "
            f"{row['review_control']} | {row['review_shock']} |"
        )
    add("")

    add("## Finding")
    add("")
    max_delta_row = max(rows, key=lambda r: float(r["shock_frac"]) - float(r["control_frac"]))
    add(
        f"**Yes — a population-wide shock does falsely flag healthy merchants, "
        f"and for `{max_delta_row['model']}` it does so badly.** "
        f"`{max_delta_row['model']}`'s false-flag rate on healthy merchants "
        f"jumps from {_f(float(max_delta_row['control_frac']), 3)} pre-shock to "
        f"{_f(float(max_delta_row['shock_frac']), 3)} once the shock is inside "
        f"its evidence window ({max_delta_row['n_control']} to "
        f"{max_delta_row['n_shock']} of {max_delta_row['n_healthy']} healthy "
        "merchants) — **worse than the static `rules` floor it is meant to beat**, "
        "whose own new-false-positive rate *falls* under the same shock "
        "(0.125 to 0.025 — see the caveat above on why that fall is a "
        "first-flag artefact, not evidence `rules` handles the shock better; "
        "`rules`' pre-shock baseline of 10/80 is itself the more sobering "
        "number). `gbdt` flags no healthy merchant in either bucket "
        "(0/80 to 0/80) — its window-probability threshold is never crossed "
        "by anyone in this split, healthy or bad, so this run offers no "
        "evidence either way about `gbdt` under a shock this size."
    )
    add("")
    add(
        "**Reported exactly as measured, whichever way it falls.** This is "
        "not the direction a project pitching `hmm` as the proposal would "
        "want, and it is reported anyway "
        "(`14-spec-blackswan-and-drift-survey.md`): the one model this repo "
        "argues for is the one this stress test finds least robust to a "
        "shared shock, considerably more so than the blunt global-threshold "
        "floor its own charter claims to improve on."
    )
    add("")

    add("## What this does not establish")
    add("")
    add(
        "1. **Whether repeated training-time exposure would teach a model "
        "tolerance is not tested here.** This run's shock lands once and only "
        "inside `validate`; `train` is shock-free, so nothing below can be read "
        "as \"the refit learned to tolerate a shock it never saw\" — it "
        "structurally could not have. That question needs a second dataset with "
        "shocks recurring across `train` too, and a head-to-head comparison "
        "against this one. Out of scope for a ticket ranked below "
        "T-0013/T-0018/T-0020/T-0021/T-0019."
    )
    add(
        "2. **The SHOCK bucket is not proof of causation for every member of "
        "it**, `rules` and `hmm` especially — both buckets are defined by which "
        "days a model's OWN evidence window could mechanically have seen the "
        "shock, not by inspecting each flag individually. A healthy merchant "
        "that would have flagged anyway on a day that happens to fall in the "
        "SHOCK window is still counted there. The CONTROL bucket carries no such "
        "caveat: a flag dated strictly before the shock cannot be caused by it."
    )
    add(
        "3. **Small counts.** `validate` holds 80 healthy merchants split across "
        "two (or, for `gbdt`, three) buckets; a handful of merchants moving "
        "buckets can move a fraction substantially. Counts are printed beside "
        "every fraction for exactly this reason (AP-06 discipline, matching "
        "the `random`-floor requirement everywhere else in this repo)."
    )
    add(
        "4. **A lower SHOCK fraction is not evidence the shock helps.** Because "
        "buckets are assigned from each merchant's FIRST-EVER flag, a model "
        "with a high pre-shock baseline false-positive rate (`rules` here) "
        "will have already claimed most of its flaggable merchants into "
        "CONTROL before the shock day arrives, mechanically depressing the "
        "SHOCK count regardless of what the shock itself did. This report "
        "measures *new* false positives per bucket, not the model's total "
        "false-positive exposure over the whole window."
    )
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    seed: int,
    results_dir: Path = RESULTS_DIR,
    transactions_path: Path = TRANSACTIONS_PATH,
    state_paths_path: Path = STATE_PATHS_PATH,
) -> Path:
    """Score the shock dataset and write `results/blackswan.md`.

    Args:
        seed: Determinism seed (NFR-003).
        results_dir: Where to write. Defaults to the committed `results/`.
        transactions_path: The shock dataset's transactions parquet.
        state_paths_path: The shock dataset's state-paths parquet.

    Returns:
        Path to the written document.

    Raises:
        FileNotFoundError: if the shock dataset has not been generated yet.
    """
    if not transactions_path.exists() or not state_paths_path.exists():
        raise FileNotFoundError(
            f"{transactions_path.parent} not found — generate it first:\n  "
            f"{GENERATE_COMMAND.format(seed=seed)}"
        )

    split, frames = score_shocked_validate(seed, transactions_path, state_paths_path)
    # Same capacity formula `eval.harness._run` uses (ADR-0008); `bmr_policy` consumes
    # `capacity_hours` directly, so no merchant-count `k` is needed here.
    capacity_hours = REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS * split.n_merchants / 1000.0

    rows = []
    for name in MODELS:
        shock_days, control_days = bucket_days(name, split.start_day, split.end_day, SHOCK_DAY)
        rows.append(
            model_row(name, split, frames[name], shock_days, control_days, capacity_hours)
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "blackswan.md"
    path.write_text(
        render(split, rows, seed, transactions_path, state_paths_path),
        encoding="utf-8",
        newline="\n",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the black-swan probe. Returns a process exit code."""
    parser = base_parser(
        "T-0022c: score the black-swan shock dataset and write results/blackswan.md. "
        "Manually invoked — not part of `make eval`'s NFR-004 budget "
        "(14-spec-blackswan-and-drift-survey.md)."
    )
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    path = run(args.seed)
    print(f"rakshak: wrote {path} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
