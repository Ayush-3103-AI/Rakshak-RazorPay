"""Per-typology recall — where the models fail, broken out by fraud pattern (T-0013).

`CLAUDE.md` non-negotiable 1 is explicit: *"Report where the model fails. The
slow-ramp evader typology exists specifically so that we can report degraded
recall on it. Do not tune it away. Do not hide it."* Until this module landed the
repo had **no per-typology number anywhere** — `results/verdict.md` reports one
pooled recall over all twenty bad merchants, `results/ablations.md` reports
component deltas, and `SLOW_RAMP` appears in `results/summary.md` only as a row in
a merchant-count table. The typology built to be reported on was not being
reported on.

**This module measures, it does not select.** Every configuration was frozen at
T-0004b on `validate` and the verdict was rendered at T-0011. Nothing here changes
a model, a threshold or a constant; it re-partitions the same actions
`harness.evaluate_model` already chose, so no number in `verdict.md` can move and
none of these rows can feed back into a decision.

**The sample sizes are tiny and the document says so on every row.** The test
split holds 20 bad merchants over 5 typologies — **4 each**. A recall over 4
merchants is quantised to {0, .25, .5, .75, 1} and its 95% Wilson interval spans
most of the unit interval at every one of those points. These rows are honest
about direction and near-useless about magnitude, which is a statement worth
publishing rather than a reason to omit the table: a submission that reports
"SLOW_RAMP recall 0.25" without the interval is claiming precision it does not
have, and one that reports nothing is hiding the failure the charter promised to
surface.

**Two definitions of "caught", both reported.** They answer different questions
and they disagree:

* **acted on** — the policy chose REVIEW or HOLD. This is the operational
  definition, it is what consumes analyst hours, it is subject to the capacity
  constraint, and it is defined for every model including `random`.
* **flagged** — the model's own `flag_day` fired, i.e. its score crossed
  `FLAG_THRESHOLD`. Defined only for the time-resolved models, and unconstrained
  by capacity.

Reporting only the first would hide a model that detects a typology but is
outbid for review slots; reporting only the second would credit a model for
detections the analyst never sees.

Run it::

    python -m rakshak.eval.typology --seed 42

which writes `results/typology_recall.md`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.cli import base_parser, seed_everything
from rakshak.config import RESULTS_DIR, REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS, SEED
from rakshak.decision.policy import HOLD, REVIEW
from rakshak.eval.harness import MODEL_REGISTRY, _f, evaluate_model
from rakshak.eval.oracle import review_slots
from rakshak.eval.splits import Split, active_state_paths_path, load_split

UNLOCK_TICKET: str = "T-0013"
"""`eval.splits.load_split` requires this to open the test window."""

TYPOLOGY_SPLIT: str = "test"
"""The window the verdict reports on, so these rows decompose that verdict."""

NO_TYPOLOGY: str = "NONE"
"""The generator's label for a merchant with no injected typology."""

ADVERSARIAL: str = "SLOW_RAMP"
"""FR-005's deliberately-hard typology. Its row is the one the charter promised."""

TYPOLOGY_DESCRIPTIONS: dict[str, str] = {
    "BUST_OUT": "legitimate history, then a hard volume ramp, then the account vanishes",
    "LAUNDERING_ENDPOINT": "normal tickets, abnormal payer graph — many payers, no repeats",
    "CATEGORY_DRIFT": "a silent shift of ticket size and time-of-day profile",
    "REFUND_COLLUSION": "merchant and a small payer set extract value through refunds",
    "SLOW_RAMP": (
        "**ADVERSARIAL (FR-005)** — a monotone, changepoint-free drift built to defeat "
        "exactly the changepoint logic this project is made of"
    ),
}

__all__ = [
    "ADVERSARIAL",
    "TYPOLOGY_SPLIT",
    "UNLOCK_TICKET",
    "merchant_typologies",
    "recall_table",
    "render_typology_recall",
    "run",
    "wilson_interval",
]


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because every cell here has
    `trials` of about 4: the normal interval is degenerate at 0 and 1 successes
    (it returns zero width, which would be a lie in exactly the cells that matter
    most) and routinely runs outside [0, 1]. Wilson is closed-form, needs no
    dependency, and stays inside the unit interval at every count.

    Args:
        successes: Number of successes.
        trials: Number of trials. Zero returns the full interval.
        z: Normal quantile; 1.96 is 95%.

    Returns:
        `(low, high)`, both dimensionless in [0, 1].
    """
    if trials <= 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def merchant_typologies(split: Split) -> pd.Series:
    """Injected typology per merchant in `split`, indexed by merchant_id.

    Reads the **active** state-paths parquet (`eval.splits.active_state_paths_path`)
    rather than the config constant, so a run under `active_dataset` sees the
    dataset it is scoring. Reaching for `STATE_PATHS_PARQUET` here is exactly the
    defect `tests/test_dataset_seam.py::test_override_reaches_every_reader` exists
    to catch (T-0022b).

    Args:
        split: The split whose merchants to label.

    Returns:
        String Series indexed by merchant_id; `NO_TYPOLOGY` for healthy merchants.
    """
    state_paths = pd.read_parquet(active_state_paths_path())
    per_merchant = (
        state_paths.groupby("merchant_id", observed=True)["typology"].first().astype(str)
    )
    return per_merchant.reindex(pd.Index(split.merchant_ids, name="merchant_id")).fillna(
        NO_TYPOLOGY
    )


def recall_table(split: Split, seed: int, k: int) -> pd.DataFrame:
    """Per-model, per-typology recall on `split`.

    Args:
        split: The split to report on.
        seed: Global seed (NFR-003).
        k: Review budget in merchants.

    Returns:
        Long-form frame: one row per (model, typology) with `n`, `n_acted_on`,
        `n_flagged` and the two recalls. Recalls are dimensionless in [0, 1].
    """
    typologies = merchant_typologies(split)
    index = pd.Index(split.merchant_ids, name="merchant_id")
    bad = split.labels.reindex(index).astype(bool)

    records: list[dict[str, object]] = []
    for name in MODEL_REGISTRY:
        row = evaluate_model(name, split, seed, k)
        actions = pd.Series(np.asarray(row["actions"]), index=index)
        acted_on = actions.isin((REVIEW, HOLD))
        # `flag_day` is re-derived from the same scorer call rather than stored,
        # so it cannot drift from the actions above.
        flagged = _flag_series(name, split, seed).notna()

        for typology in sorted(set(typologies[bad])):
            members = index[bad & (typologies == typology)]
            records.append(
                {
                    "model": name,
                    "typology": typology,
                    "n": len(members),
                    "n_acted_on": int(acted_on.reindex(members).sum()),
                    "n_flagged": int(flagged.reindex(members).fillna(False).sum()),
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame["recall_acted_on"] = frame["n_acted_on"] / frame["n"]
    frame["recall_flagged"] = frame["n_flagged"] / frame["n"]
    return frame


def _flag_series(name: str, split: Split, seed: int) -> pd.Series:
    """`flag_day` per merchant for one model, all-NaN when it reports none."""
    from rakshak.eval.harness import _model_rng, _normalise

    frame = _normalise(MODEL_REGISTRY[name](split, _model_rng(seed, name)), split)
    return frame["flag_day"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_typology_recall(frame: pd.DataFrame, split: Split, seed: int, k: int) -> str:
    """Build `results/typology_recall.md`. Byte-identical for a fixed seed."""
    lines: list[str] = []
    add = lines.append
    models = list(MODEL_REGISTRY)
    typologies = sorted(frame["typology"].unique())
    per_typology_n = frame.groupby("typology")["n"].first()

    add("# Rakshak — recall by fraud typology (FR-005, CLAUDE.md non-negotiable 1)")
    add("")
    add(
        "> **Sequence-layer metrics are measured on synthetic merchant streams with "
        "injected typologies; the generator is in this repo.** The decision layer is "
        "additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark "
        "derived from real bank data."
    )
    add("")
    add(
        "BAF has no typologies and no sequences, so nothing on this page could have been "
        "measured there. Every number here is the synthetic split."
    )
    add("")

    add("## Provenance")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(f"| Produced by | `python -m rakshak.eval.typology --seed {seed}` |")
    add(f"| Seed | {seed} |")
    add(
        f"| Split reported | `{split.name}` (days {split.start_day}-{split.end_day - 1}), "
        f"unlocked with ticket {UNLOCK_TICKET} |"
    )
    add(f"| Population | {split.n_merchants} merchants, {int(split.labels.sum())} truly bad |")
    add(f"| Review budget K | {k} merchants |")
    add("")
    add(
        "**Nothing here selects anything.** Every configuration was frozen at T-0004b on "
        "`validate`; this module re-partitions the actions `harness.evaluate_model` "
        "already chose at T-0011. No number in `results/verdict.md` can move because of "
        "this file, and no row here can feed back into a decision."
    )
    add("")

    add("## Read the sample size before the recall")
    add("")
    counts = ", ".join(f"`{t}` n={int(per_typology_n[t])}" for t in typologies)
    add(f"Truly-bad merchants per typology on this split: {counts}.")
    add("")
    add(
        "**Every cell below is a proportion over about four merchants.** Recall is "
        "therefore quantised to {0, 0.25, 0.50, 0.75, 1.00} and the 95% Wilson interval "
        "spans most of the unit interval at every one of those points. These rows carry "
        "information about *direction* and almost none about *magnitude*. They are "
        "published with their intervals rather than omitted, because the alternative to "
        "an honestly-underpowered table is not a better table — it is a submission that "
        "quietly never reports the typology it promised to fail on."
    )
    add("")
    add(
        "The intervals are Wilson score intervals, not normal approximations: at 0 and 4 "
        "successes out of 4 the normal interval has zero width, which would be a lie in "
        "exactly the cells that matter most."
    )
    add("")

    add("## Recall by typology — `acted on` (policy chose REVIEW or HOLD)")
    add("")
    add(
        "The operational definition: the merchant reached an analyst or had settlement "
        "held. Subject to FR-017's capacity constraint, and defined for every model."
    )
    add("")
    add("| typology | n | " + " | ".join(f"`{m}`" for m in models) + " |")
    add("|---|---|" + "---|" * len(models))
    for typology in typologies:
        cells = []
        for model in models:
            row = frame[(frame["model"] == model) & (frame["typology"] == typology)].iloc[0]
            low, high = wilson_interval(int(row["n_acted_on"]), int(row["n"]))
            cells.append(
                f"{_f(float(row['recall_acted_on']), 2)} "
                f"({int(row['n_acted_on'])}/{int(row['n'])}, "
                f"95% CI {_f(low, 2)}-{_f(high, 2)})"
            )
        marker = " **(adversarial)**" if typology == ADVERSARIAL else ""
        add(
            f"| `{typology}`{marker} | {int(per_typology_n[typology])} | "
            + " | ".join(cells)
            + " |"
        )
    add("")

    add("## Recall by typology — `flagged` (the model's own score crossed its threshold)")
    add("")
    add(
        "Unconstrained by capacity. `random` returns no `flag_day` — a single "
        "per-merchant score has no time at which it fired — so its column is `n/a` "
        "rather than zero."
    )
    add("")
    add("| typology | n | " + " | ".join(f"`{m}`" for m in models) + " |")
    add("|---|---|" + "---|" * len(models))
    for typology in typologies:
        cells = []
        for model in models:
            row = frame[(frame["model"] == model) & (frame["typology"] == typology)].iloc[0]
            if model == "random":
                cells.append("n/a")
                continue
            low, high = wilson_interval(int(row["n_flagged"]), int(row["n"]))
            cells.append(
                f"{_f(float(row['recall_flagged']), 2)} "
                f"({int(row['n_flagged'])}/{int(row['n'])}, "
                f"95% CI {_f(low, 2)}-{_f(high, 2)})"
            )
        marker = " **(adversarial)**" if typology == ADVERSARIAL else ""
        add(
            f"| `{typology}`{marker} | {int(per_typology_n[typology])} | "
            + " | ".join(cells)
            + " |"
        )
    add("")

    add("## The typologies")
    add("")
    add("| typology | what the generator injects |")
    add("|---|---|")
    for typology in typologies:
        add(f"| `{typology}` | {TYPOLOGY_DESCRIPTIONS.get(typology, 'see the generator')} |")
    add("")

    add(f"## `{ADVERSARIAL}` — the row this table exists for")
    add("")
    add(
        f"`{ADVERSARIAL}` is FR-005's adversarial typology: a monotone, changepoint-free "
        "drift, built deliberately to defeat changepoint and state-transition logic. "
        "**It exists so that this project has somewhere honest to fail**, and "
        "`CLAUDE.md` forbids tuning it away."
    )
    add("")
    adversarial = frame[frame["typology"] == ADVERSARIAL]
    if adversarial.empty:
        add(f"**No `{ADVERSARIAL}` merchant is present on this split.** Nothing to report.")
    else:
        pooled = {
            str(r["model"]): (
                float(r["recall_acted_on"]),
                float(
                    frame[
                        (frame["model"] == r["model"]) & (frame["typology"] != ADVERSARIAL)
                    ]["n_acted_on"].sum()
                )
                / max(
                    1,
                    int(
                        frame[
                            (frame["model"] == r["model"])
                            & (frame["typology"] != ADVERSARIAL)
                        ]["n"].sum()
                    ),
                ),
            )
            for _, r in adversarial.iterrows()
        }
        add("| model | recall on `SLOW_RAMP` | recall on the other four typologies | delta |")
        add("|---|---|---|---|")
        for model in models:
            slow, others = pooled[model]
            add(
                f"| `{model}` | {_f(slow, 2)} | {_f(others, 2)} | "
                f"{_f(slow - others, 2)} |"
            )
        add("")
        add(
            "**A delta at or above zero here is not evidence that the adversarial "
            "typology was solved.** With four merchants in the `SLOW_RAMP` cell and "
            "sixteen in the comparison cell, this difference is not separable from "
            "sampling noise at any conventional level, and the intervals in the tables "
            "above show it directly. The row is published in whichever direction it "
            "falls; it is not evidence in either."
        )
    add("")

    add("## What this table does not establish")
    add("")
    add(
        "1. **Nothing about magnitude.** Four merchants per cell. See the interval on "
        "every number."
    )
    add(
        "2. **Nothing about real fraud.** These are the generator's own five caricatures, "
        "injected by this repo and detected by this repo. `results/calibration_gap.md` "
        "measures how far the generator's marginals sit from a real transaction stream; "
        "its typology *dynamics* are calibrated against nothing, because no public "
        "merchant-sequence dataset with merchant-level risk labels exists "
        "(`06-requirements.md:28`, ADR-0007)."
    )
    add(
        "3. **Nothing about the pooled verdict.** K2 was rendered at T-0011 on pooled "
        "savings and PR-AUC and is unchanged by this decomposition. See "
        "`results/verdict.md`."
    )
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(seed: int = SEED, results_dir: Path = RESULTS_DIR) -> Path:
    """Write `results/typology_recall.md`. Returns its path."""
    split = load_split(TYPOLOGY_SPLIT, unlock_test=UNLOCK_TICKET)
    capacity_hours = REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS * split.n_merchants / 1000.0
    k = min(review_slots(capacity_hours), split.n_merchants)
    frame = recall_table(split, seed, k)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "typology_recall.md"
    path.write_text(
        render_typology_recall(frame, split, seed, k), encoding="utf-8", newline="\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """Write the per-typology recall table. Returns a process exit code."""
    parser = base_parser("Break detection recall down by injected fraud typology.")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    path = run(args.seed)
    print(f"rakshak: wrote {path} (seed={args.seed})")
    print(
        "rakshak: every cell is a proportion over ~4 merchants - read the Wilson "
        "intervals, not the point estimates"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
