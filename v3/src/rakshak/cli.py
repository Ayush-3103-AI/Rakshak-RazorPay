"""The typer CLI. Every ``make`` target calls through here rather than into a module.

One entry point per pipeline stage, so that the Makefile stays a list of names and the
argument handling lives in exactly one place.

**This file is the guard on the one-way door.** Lane C built ``require_unlocked_or_refuse``
and ``verify_lock`` and tested both, but could not wire them - ``cli.py`` was another
lane's file. Every path that scores anything now goes through ``_guard(split)``, which
calls both, in that order, before a single row is read. Nothing else guards the test
split: the environment variable is checked in exactly one place, ``lock.py``, and it is
reached from exactly one place, here.

``require_unlocked_or_refuse`` refuses on anything but the literal string ``"1"``.
``"true"``, ``"yes"`` and ``"TRUE"`` all refuse, deliberately - a guard that accepts
several spellings is a guard someone eventually trips over by accident.

This file also sits on the eval side of the Prime Directive 3 quarantine, and that is on
purpose. ``src/rakshak/models/`` may not name a ground-truth field, so the ``Truth`` object
every metric needs is assembled *here*, and the rungs are handed plain arrays.
"""

from __future__ import annotations

import dataclasses
import json
import time as _time
from datetime import UTC, datetime, time, timedelta
from hashlib import blake2b
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
import typer

from rakshak.eval import capacity
from rakshak.eval.capacity import DEFAULT_POLICY, ActionPolicy
from rakshak.eval.lock import (
    load_lock,
    read_open_count,
    record_open,
    require_unlocked_or_refuse,
    resolve_authoritative,
    verify_lock,
)
from rakshak.eval.metrics import CostParams, PerfBudget, RungOutput, Truth, build_eval_result
from rakshak.eval.metrics import day_labels as _day_labels
from rakshak.eval.oracle import oracle_savings
from rakshak.eval.splits import (
    DEFAULT_BOUNDARIES,
    SplitBoundaries,
    available_labels,
    label_coverage,
)
from rakshak.generator.config import ScenarioConfig, load_scenario
from rakshak.generator.engine import generate
from rakshak.models import (
    dataset,
    rung0_floors,
    rung1_rules,
    rung2_lgbm,
    rung3_cohort,
    rung4_cost,
    rung5_mil,
    rung8_realised_exposure,
    rung9_rank_cusum,
)
from rakshak.schemas import MerchantProfile, Split

app = typer.Typer(add_completion=False, help="Rakshak v2 — merchant risk sentinel.")

#: The repository root, derived from this file rather than from the working directory —
#: EVAL-LOCK.json hashes paths relative to it, so `make eval` run from a subdirectory must
#: still verify the same tree.
ROOT = Path(__file__).resolve().parents[2]

#: Where trained boosters and scored results land. Under `data/`, which is gitignored and
#: regenerable from seed + config; nothing here is an input to anything.
MODEL_DIR = Path("data/v2/models")
RESULT_DIR = Path("data/v2/eval")

#: Background on why more than one lock exists: `EVAL-LOCK.json` pins `split_boundaries` to
#: the 180-day geometry, so the T-0101-corrected 20,000 x 365-day window
#: (docs/RE-FREEZE-2026-08-31.md Amendment 4) needed its own lock. Cycle 1 stays committed,
#: untouched, `open_count: 0` intact. `eval_module_sha256` was byte-identical across those
#: two because the harness did not change, only the window and the population it is pointed
#: at — which is exactly why the staleness below went unnoticed for two cycles.
def _lock_path(root: Path = ROOT) -> Path:
    """The lock that governs right now, resolved from the supersession chain.

    This was a module constant pinned to a filename — first ``EVAL-LOCK.json``, then
    ``EVAL-LOCK-CYCLE2.json`` when cycle 2 superseded it. Both edits were correct on the
    day and both went stale on the next freeze, silently: a pinned name keeps loading a
    superseded lock and nothing complains, because a superseded lock is still a valid file.

    That matters most at ``record_open`` below, which is the one-way door. Opening the test
    split against a pinned name would increment the counter on a lock that no longer
    governs, leaving the authoritative lock reading ``open_count: 0`` after the split had
    been opened — the single fact this project most needs to be true.
    """
    return resolve_authoritative(root)

#: Default scenario manifest. Named once; commands below default to it.
DEFAULT_CONFIG = Path("configs/scenario_v2.yaml")


def _boundaries(config: Path, *, root: Path = ROOT) -> SplitBoundaries:
    """The split geometry, READ from ``scenario.splits`` (T-0101, GitHub #34).

    Day boundaries and the merchant fold are independent facts about this geometry: day
    spans are 65.75% / 16.44% / 17.81% of the 365-day horizon, the merchant fold is the
    declared 60% / 15% / 25%. Both are named fields on ``ScenarioConfig.splits``
    (``generator/config.py::SplitsConfig``). ``eval/splits.py`` is NOT edited:
    ``SplitBoundaries`` is used only through its existing constructor (arbitrary day
    tuples were always legal), and the merchant fold is assigned by
    ``_merchant_fold_t0101`` below — a sibling function, not an edit to
    ``eval.splits.merchant_fold``.

    Cross-checked against the lock on every call, so a scenario file edited after the
    freeze cannot quietly move the scored days.
    """
    scenario = load_scenario(config)
    s = scenario.splits
    boundaries = SplitBoundaries(
        origin=DEFAULT_BOUNDARIES.origin,
        train=(0, s.train_end_day),
        val=(s.train_end_day + 1, s.val_end_day),
        test=(s.val_end_day + 1, s.test_end_day),
    )
    locked = load_lock(root, lock_path=_lock_path(root))["split_boundaries"]
    derived = {"train": list(boundaries.train), "val": list(boundaries.val),
               "test": list(boundaries.test)}
    if derived != locked:
        raise typer.BadParameter(
            f"{config} implies split boundaries {derived}, but "
            f"{_lock_path(root).name} froze {locked}. "
            "The window and the splitter are the same fact stated twice and they disagree; "
            "results scored across that gap are not comparable to anything."
        )
    return boundaries


#: T-0101 (GitHub #34): the merchant fold, INDEPENDENT of the day-span proportions
#: ``eval.splits.merchant_fold()`` derives its shares from. A NEW, own-salted function —
#: a sibling of the locked one, not an edit to it, so ``eval_module_sha256`` stays
#: byte-identical.
_T0101_FOLD_SALT = b"rakshak-t0101-merchant-fold"


def _merchant_fold_t0101(merchant_id: str, shares: tuple[float, float, float]) -> Split:
    """Deterministic merchant -> fold at the declared (train, val, test) shares.

    Identical algorithm to ``eval.splits.merchant_fold()`` — hash the id to a stable
    uniform variate, walk the cumulative shares — with a different salt and a
    caller-supplied ratio, because the day-span ratio and the merchant-fold ratio are
    independent facts about this geometry.
    """
    digest = blake2b(merchant_id.encode(), key=_T0101_FOLD_SALT, digest_size=8).digest()
    u = int.from_bytes(digest, "big") / 2**64
    cumulative = 0.0
    names: tuple[Split, ...] = ("train", "val", "test")
    for name, share in zip(names, shares, strict=True):
        cumulative += share
        if u < cumulative:
            return name
    return "test"


def _fold_shares(config: Path) -> tuple[float, float, float]:
    s = load_scenario(config).splits
    return (s.merchant_fold_train, s.merchant_fold_val, s.merchant_fold_test)


@app.callback()
def main() -> None:
    """Rakshak v2. Run ``rakshak <command> --help`` for a stage."""


@app.command()
def gen(
    # B008 is suppressed on the two Path defaults only: `typer.Option(Path(...))` is
    # typer's documented way to give an option a path default, and the inner Path() is
    # what ruff sees. Rewriting it as a module-level singleton would obscure the CLI
    # signature to satisfy a rule aimed at mutable defaults, which Path is not.
    config: Path = typer.Option(  # noqa: B008
        Path("configs/scenario_v2.yaml"), "--config", "-c", help="Scenario manifest."
    ),
    seed: int = typer.Option(42, "--seed", help="Overrides the seed in the manifest."),
    out: Path = typer.Option(  # noqa: B008
        Path("data/v2"), "--out", "-o", help="Output directory."
    ),
    merchants: int | None = typer.Option(
        None, "--merchants", help="Override population.n_merchants (smoke runs, gates)."
    ),
    days: int | None = typer.Option(None, "--days", help="Override population.n_days."),
    prevalence: float | None = typer.Option(
        None,
        "--prevalence",
        help="Override population.prevalence. 0.0 is the gate-G5 confounder-null run.",
    ),
    confounders: bool = typer.Option(
        True, "--confounders/--no-confounders", help="Toggle the P1-P6 platform layer."
    ),
) -> None:
    """Generate the v2 dataset from a scenario manifest.

    Deterministic in ``--seed``: two runs at the same seed produce byte-identical tables,
    and gate G3 asserts exactly that.
    """
    scenario = _apply_overrides(
        load_scenario(config),
        merchants=merchants,
        days=days,
        prevalence=prevalence,
        confounders=confounders,
    )
    data = generate(scenario, np.random.default_rng(seed))
    paths = data.write(out)

    summary = {
        "seed": seed,
        "config": str(config),
        "n_merchants": scenario.population.n_merchants,
        "n_days": scenario.population.n_days,
        "prevalence": scenario.population.prevalence,
        "confounders_enabled": scenario.confounders.enabled,
        "analyst_capacity_k": scenario.analyst_capacity,
        "rows": {name: data.row_counts[name] for name in paths},
        "content_sha256": data.sha256(),
    }
    (Path(out) / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


def _apply_overrides(
    scenario: ScenarioConfig,
    *,
    merchants: int | None,
    days: int | None,
    prevalence: float | None,
    confounders: bool,
) -> ScenarioConfig:
    """CLI overrides, applied to the loaded manifest.

    Overrides exist for the gates and for smoke runs, not as a second configuration
    surface: ``run_summary.json`` records every one of them next to the content hash, so
    a dataset can never be mistaken for the manifest's default population.
    """
    population = scenario.population
    if merchants is not None:
        population = dataclasses.replace(population, n_merchants=merchants)
    if days is not None:
        population = dataclasses.replace(population, n_days=days)
    if prevalence is not None:
        population = dataclasses.replace(population, prevalence=prevalence)
    return dataclasses.replace(
        scenario,
        population=population,
        confounders=dataclasses.replace(scenario.confounders, enabled=confounders),
    )


# ─────────────────────────────────────────────────────────────────────────────
# The guard. Both halves, before any scoring path, no exceptions.
# ─────────────────────────────────────────────────────────────────────────────


def _guard(split: Split, *, root: Path = ROOT) -> list[str]:
    """``require_unlocked_or_refuse`` **and** ``verify_lock``, in that order.

    Order matters. The split guard is the cheap, loud one and it should fire before the
    hash check has a chance to fail for an unrelated reason — a run that is not permitted
    to happen at all should not first spend time telling you the generator moved.

    Returns the *unenforced* drift as human-readable lines. Drift is reported on every run
    and never silently: only ``eval_module_sha256`` is a hard fail (LIMITATIONS.md §4), and
    a recorded hash that no longer matches is provenance the reader is entitled to see.
    """
    require_unlocked_or_refuse(split)
    return [
        f"{d.key}: recorded {d.expected[:12]}… now {d.actual[:12]}…"
        for d in verify_lock(root, lock_path=_lock_path(root))
    ]


def _epoch_end(day: int, boundaries: SplitBoundaries) -> datetime:
    return datetime.combine(boundaries.origin + timedelta(days=day), time.max, tzinfo=UTC)


#: An as_of past every ``label_available_at`` the generator can emit. Used for **one**
#: thing: recovering which merchants are label-censored, so the metric suite can exclude
#: them (10-eval-harness-spec.md §1). Censoring is an eval-side fact — within the 180-day
#: window a censored merchant is indistinguishable from a pending one, so it cannot be
#: recovered at an in-window as_of. It is never used to obtain a training label; those come
#: from ``available_labels(train_as_of)`` and nothing else.
_CENSORING_AS_OF = datetime(2099, 1, 1, tzinfo=UTC)


def _label_path(root: Path) -> Path:
    """The label table for a dataset root, named through ``splits`` rather than spelled
    out here — there is one door and this file is not it."""
    from rakshak.eval.splits import DEFAULT_LABEL_PATH

    return root / DEFAULT_LABEL_PATH.name


def _observed_volume(
    root: Path, merchants: list[str], cutoff_day: int, boundaries: SplitBoundaries
) -> np.ndarray:
    """Captured, non-refunded GMV per merchant up to ``cutoff_day``. The volume the
    ``volume_rank`` floor ranks on, and ``Truth.volume``.

    Cut at the day *before* the scored window opens, so the dumbest heuristic is still a
    point-in-time one and cannot rank merchants on volume it has not seen yet.
    """
    frame = (
        pl.scan_parquet(root / "transactions.parquet")
        .filter(
            (pl.col("event_date") <= _epoch_end(max(cutoff_day, 0), boundaries).date())
            & (pl.col("status") == "captured")
            & ~pl.col("is_refund")
        )
        .group_by("merchant_id")
        .agg(pl.col("amount_inr").sum().alias("volume"))
        .collect()
    )
    lookup = dict(zip(frame["merchant_id"], frame["volume"], strict=True))
    return np.array([float(lookup.get(m, 0.0)) for m in merchants], dtype=np.float64)


def _build_truth(
    root: Path, merchants: list[str], cutoff_day: int, boundaries: SplitBoundaries
) -> Truth:
    """One row per scored merchant, from the generator's truth table.

    Assembled here rather than under ``models/`` because it names quarantined fields, and
    Prime Directive 3's AST gate covers ``models/``. The rungs never see this object; they
    are handed a score vector's worth of arrays and nothing else.
    """
    truth_frame = (
        pl.read_parquet(root / "ground_truth.parquet")
        .filter(pl.col("merchant_id").is_in(merchants))
        .sort("merchant_id")
    )
    by_id = {row["merchant_id"]: row for row in truth_frame.iter_rows(named=True)}
    missing = [m for m in merchants if m not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} scored merchants are absent from the truth table")

    # Narrowed to the merchants being scored, so this future-dated read cannot see a row
    # for a merchant in another split even incidentally.
    censored_ids = set(
        available_labels(_CENSORING_AS_OF, _label_path(root), include_censored=True)
        .filter(pl.col("is_censored") & pl.col("merchant_id").is_in(merchants))
        .collect()["merchant_id"]
        .to_list()
    )

    origin = boundaries.origin
    onset = np.array(
        [
            (by_id[m]["drift_onset_at"].date() - origin).days
            if by_id[m]["drift_onset_at"] is not None
            else np.nan
            for m in merchants
        ],
        dtype=np.float64,
    )
    typology = np.array(
        [by_id[m]["risk_typology_id"] for m in merchants], dtype=object
    )

    # The loss is AMORTISED OVER THE DAYS IT ACCRUES, and this is a decision, not a
    # detail. The harness's unit of evaluation is the merchant-day, and `row_cost` charges
    # `loss` on every merchant-day a fraud is left to PASS. A merchant-level total handed
    # in unamortised is therefore charged once per day - a merchant that turns on day 40
    # of 180 is billed 140 times its own loss - which inflates the all-PASS denominator
    # until `all_hold` looks profitable and every capacity-constrained rung FLOOR-FAILs
    # against it for an accounting reason rather than a modelling one. Dividing by the
    # days from onset to the end of the horizon makes the loss summed over any window the
    # loss actually accrued in that window, which is what the cost matrix is charging for.
    # Made here rather than in eval/, which is frozen and correct as written.
    horizon = float(boundaries.n_days)
    accrual_days = np.maximum(horizon - np.nan_to_num(onset, nan=horizon - 1.0), 1.0)
    return Truth(
        merchant_id=np.array(merchants, dtype=object),
        label=np.array([int(t is not None) for t in typology], dtype=np.int8),
        is_censored=np.array([m in censored_ids for m in merchants], dtype=bool),
        loss_inr=np.array(
            [float(by_id[m]["true_loss_amount_inr"]) for m in merchants], dtype=np.float64
        )
        / accrual_days,
        onset_day=onset,
        typology=typology,
        volume=_observed_volume(root, merchants, cutoff_day, boundaries),
    )


def _cost_params(config: Path) -> CostParams:
    """The cost matrix, from ``configs/scenario_v2.yaml`` where the numbers exist.

    ``p_catch`` is **not in the manifest** and the §2 cost matrix needs it. It stands in as
    ``CostParams``' own default of 0.80, which is a code default impersonating config —
    reported as a carry-forward rather than patched around, because ``configs/`` is Lane
    A's file. The exact block it wants is in docs/logbook/T-140.md.
    """
    import yaml

    costs = yaml.safe_load(config.read_text(encoding="utf-8")).get("costs", {})
    return CostParams(
        review_cost_inr=float(costs["review_cost_inr"]),
        false_hold_cost_inr=float(costs["false_hold_cost_inr"]),
        fraud_loss_multiplier=float(costs["fraud_loss_multiplier"]),
        **({"p_catch": float(costs["p_catch"])} if "p_catch" in costs else {}),
    )


def _action_policy(config: Path) -> ActionPolicy:
    """The HOLD thresholds. Config if the manifest has a ``decision:`` block, defaults
    otherwise — the second half of the same carry-forward as ``p_catch``."""
    import yaml

    decision = yaml.safe_load(config.read_text(encoding="utf-8")).get("decision") or {}
    if not decision:
        return DEFAULT_POLICY
    return ActionPolicy(
        hold_score_threshold=float(decision["hold_score_threshold"]),
        hold_expected_loss_floor_inr=float(decision["hold_expected_loss_floor_inr"]),
    )


def _capacity(n_merchants: int, root: Path = ROOT) -> int:
    """K for the population actually scored, from the ratio EVAL-LOCK records.

    The lock stores ``capacity_k: 50`` alongside ``capacity_per_n_merchants: 10000``
    precisely because it is a rate, not a constant: a split holds a fold of the population,
    and holding K at 50 over a sixth of the merchants would hand that split six times the
    analyst budget the system is claimed to run under.
    """
    lock = load_lock(root, lock_path=_lock_path(root))
    per = int(lock["capacity_per_n_merchants"])
    return max(1, round(int(lock["capacity_k"]) * n_merchants / per))


def _training_labels(root: Path, merchants: np.ndarray, as_of: datetime) -> np.ndarray:
    """Per-row target: 1 iff this merchant has a **resolved, available** positive label.

    Everything else is 0, including merchants whose dispute has not landed yet. That is
    positive-unlabelled by construction and it is the real operating condition — the
    system cannot tell "clean" from "not disputed yet", and a harness that could would be
    measuring a system that cannot exist (10-eval-harness-spec.md §1).

    The label is broadcast to every one of the merchant's rows in the window. The drift
    onset is *not* known at training time, so the days before a merchant turned are
    labelled positive too. That is real label noise and it is not corrected here: any
    correction would need a lead-time constant nobody has measured, and inventing one to
    improve the number is the kind of tuning this project exists to avoid.
    """
    positives = set(
        available_labels(as_of, _label_path(root))
        .filter(pl.col("label") == 1)
        .collect()["merchant_id"]
        .to_list()
    )
    return np.fromiter((m in positives for m in merchants), dtype=np.int8, count=len(merchants))


def _model_path(rung: int, seed: int) -> Path:
    return MODEL_DIR / f"rung{rung}_seed{seed}.txt"


def _sidecar_path(rung: int, seed: int) -> Path:
    return MODEL_DIR / f"rung{rung}_seed{seed}.json"


def _columns_for(rung: int) -> tuple[str, ...]:
    if rung == 2:
        return dataset.base_columns()
    if rung in (3, 4):
        return rung3_cohort.feature_columns()
    if rung == 5:
        # NOT a merchant-day vector. Rung 5's rows are (merchant, day, payer) capsules and
        # its columns are the capsule vector, so a Rung 5 "row" and a Rung 2 "row" are
        # different objects entirely — see rung5_mil's module docstring. The two share this
        # function only because both need a column contract that travels with the model.
        return rung5_mil.feature_columns()
    if rung == 9:
        # Rung 9 is a wrapper over Rung 3 and consumes no columns of its own; it scores
        # through Rung 3's contract. Reported as Rung 3's width so the results table does
        # not claim a feature count that no model ever saw.
        return rung3_cohort.feature_columns()
    raise ValueError(f"rung {rung} is not a trained rung; 2, 3, 4, 5 and 9 are")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def features(
    seed: int = typer.Option(42, "--seed", help="Recorded for provenance; the panel is "
                             "deterministic given the dataset."),
    root: Path = typer.Option(Path("data/v2"), "--root", help="Generated dataset."),  # noqa: B008
    out: Path = typer.Option(  # noqa: B008
        dataset.DEFAULT_PANEL, "--out", "-o", help="Panel destination."
    ),
    config: Path = typer.Option(  # noqa: B008
        DEFAULT_CONFIG, "--config", "-c", help="Scenario manifest. The split geometry is "
        "derived from it, so the panel is labelled with the same boundaries eval scores on."
    ),
    last_day: int | None = typer.Option(
        None, "--last-day", help="Final epoch to materialise. Defaults to the last "
        "VALIDATION day; the test split cannot be materialised from here at all."
    ),
    workers: int = typer.Option(
        1, "--workers", min=1, help="Processes to replay merchant chunks on. 1 is the "
        "serial walk and the default; the panel is identical at any worker count "
        "(tests/unit/test_dataset_parallel.py), only faster."
    ),
) -> None:
    """Materialise the (merchant-day x feature) panel every rung trains and scores on."""
    shares = _fold_shares(config)
    summary = dataset.materialise(
        root,
        out,
        boundaries=_boundaries(config),
        fold_fn=lambda m: _merchant_fold_t0101(m, shares),
        last_day=last_day,
        workers=workers,
        echo=typer.echo,
    )
    summary["seed"] = seed
    (out.parent / "features_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    typer.echo(json.dumps(summary, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Rung 5. Its bulk lives in ``score_rung5`` — a merchant subsample, a chunked capsule
# materialiser and a RAM floor, none of which the other rungs need — and these two
# functions are the seam that makes it reachable as ``train --rung 5`` / ``eval --rung 5``
# rather than as a script somebody has to know about. The import is deferred because
# ``score_rung5`` imports this module's helpers; a module-level import would be a cycle.
# ─────────────────────────────────────────────────────────────────────────────


def _train_rung5(*, seed: int, root: Path, config: Path) -> None:
    """Fit the instance model on train bags, select the pooling on val bags, persist both."""
    from rakshak import score_rung5

    sample, capsules, boundaries, panel = score_rung5.prepare(root, config, echo=typer.echo)
    tuned, tau_table = score_rung5.fit_seed(
        seed,
        root=root,
        boundaries=boundaries,
        panel=panel,
        capsules=capsules,
        train_merchants=sample["train"],
        val_merchants=sample["val"],
    )
    path = tuned.save(_model_path(5, seed))
    as_of = _epoch_end(boundaries.train[1], boundaries)
    coverage = label_coverage(as_of, _label_path(root))
    summary = {
        "rung": 5,
        "seed": seed,
        "train_as_of_day": boundaries.train[1],
        "columns": list(tuned.columns),
        "n_columns": len(tuned.columns),
        "n_train_rows": tuned.instance.n_train_rows,
        "n_train_positive_rows": tuned.instance.n_train_positive_rows,
        "n_train_positive_merchants": tuned.instance.n_train_positive_merchants,
        "train_seconds": tuned.instance.train_seconds,
        "model_size_mb": round(tuned.size_mb(path), 4),
        "hparams": dataclasses.asdict(tuned.params),
        # The fitted state. Persisted deliberately, and reconstructed deliberately — see
        # _SIDECAR_RECONSTRUCTED. `tau` is NaN when noisy-OR wins, which is not a missing
        # value: noisy-OR has no tau, and that is the point of it being the comparator.
        "pooling": tuned.pooling,
        "tau": tuned.tau,
        "passes": tuned.passes,
        "n_train_bags": tuned.n_train_bags,
        "n_train_positive_bags": tuned.n_train_positive_bags,
        "tau_selection_table": tau_table,
        "subsample": {
            "n_train_merchants": len(sample["train"]),
            "n_val_merchants": len(sample["val"]),
        },
        "label_coverage_at_train_boundary": {
            "n_merchants": coverage.n_merchants,
            "n_available": coverage.n_available,
            "n_positive": coverage.n_positive,
            "n_censored": coverage.n_censored,
            "n_pending": coverage.n_pending,
        },
    }
    _sidecar_path(5, seed).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


def _eval_rung5(*, seed: int, root: Path, config: Path, panel_path: Path) -> None:
    """Score the persisted Rung 5 through the ``_load_trained`` round-trip.

    Deliberately reloads from disk rather than keeping the model from ``train`` in memory:
    the round-trip is the thing that would have silently lost the fitted pooling, so the
    scoring path is the one place it must be exercised for real.
    """
    from rakshak import score_rung5

    sample, capsules, boundaries, panel = score_rung5.prepare(root, config, echo=typer.echo)
    tuned = _load_trained(5, seed)
    if not isinstance(tuned, rung5_mil.TrainedMIL):
        raise typer.BadParameter(f"rung 5 seed {seed} did not reload as a TrainedMIL")
    payload = score_rung5.score_seed(
        tuned,
        seed,
        root=root,
        config=config,
        boundaries=boundaries,
        panel=panel,
        capsules=capsules,
        val_merchants=sample["val"],
        train_merchants=sample["train"],
        tau_table=json.loads(_sidecar_path(5, seed).read_text(encoding="utf-8"))[
            "tau_selection_table"
        ],
        booster_path=_model_path(5, seed),
    )
    results = panel_path.parent / RESULT_DIR.name
    results.mkdir(parents=True, exist_ok=True)
    (results / f"rung5_mil_val_seed{seed}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def train(
    rung: int = typer.Option(2, "--rung", help="2, 3, 4 or 5."),
    seed: int = typer.Option(42, "--seed", help="Threaded into LightGBM; identical "
                             "across rungs so the Rung-3 delta is single-variable."),
    root: Path = typer.Option(Path("data/v2"), "--root", help="Generated dataset."),  # noqa: B008
    panel: Path = typer.Option(  # noqa: B008
        dataset.DEFAULT_PANEL, "--panel", help="Materialised feature panel."
    ),
    config: Path = typer.Option(  # noqa: B008
        Path("configs/scenario_v2.yaml"), "--config", "-c", help="Scenario manifest."
    ),
) -> None:
    """Train one rung on the TRAIN split only, with labels available at the train boundary.

    Guarded exactly like ``eval``: the split guard and the lock hash both run before a row
    is read. Training is a scoring path in every sense that matters — it reads features and
    labels — so it gets the same door.
    """
    drift = _guard("train", root=ROOT)
    for line in drift:
        typer.echo(f"[lock drift, unenforced] {line}")

    if rung == 5:
        _train_rung5(seed=seed, root=root, config=config)
        return

    full = dataset.load_panel(panel)
    rows = full.select("train")
    columns = _columns_for(rung)
    x = rows.with_columns(columns).x
    boundaries = _boundaries(config)
    as_of = _epoch_end(boundaries.train[1], boundaries)
    y = _training_labels(root, rows.merchant_id, as_of)
    coverage = label_coverage(as_of, _label_path(root))

    # One HParams instance, one seed, for every rung. Rung 3's delta is attributable to
    # the residual columns only if literally nothing else moved (FR-031), and "nothing
    # else" includes the four separate seeds LightGBM reads.
    hparams = rung2_lgbm.DEFAULT_PARAMS.with_seed(seed)
    if rung == 2:
        model = rung2_lgbm.train(x, y, columns, params=hparams, merchant_id=rows.merchant_id)
    elif rung == 3:
        model = rung3_cohort.train(
            x,
            y,
            columns,
            rung2_columns=dataset.base_columns(),
            params=hparams,
            merchant_id=rows.merchant_id,
        )
    else:
        params = _cost_params(config)
        model = rung4_cost.train(
            x,
            y,
            columns,
            exposure_inr=rows.column("p_declared_monthly_gmv"),
            review_cost_inr=params.review_cost_inr,
            params=hparams,
            merchant_id=rows.merchant_id,
        )

    path = model.save(_model_path(rung, seed))
    summary = {
        "rung": rung,
        "seed": seed,
        "train_as_of_day": boundaries.train[1],
        "columns": list(columns),
        "n_columns": len(columns),
        "n_train_rows": model.n_train_rows,
        "n_train_positive_rows": model.n_train_positive_rows,
        "n_train_positive_merchants": model.n_train_positive_merchants,
        "train_seconds": model.train_seconds,
        "model_size_mb": round(model.size_mb(path), 4),
        "hparams": dataclasses.asdict(model.params),
        "label_coverage_at_train_boundary": {
            "n_merchants": coverage.n_merchants,
            "n_available": coverage.n_available,
            "n_positive": coverage.n_positive,
            "n_censored": coverage.n_censored,
            "n_pending": coverage.n_pending,
        },
    }
    _sidecar_path(rung, seed).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


#: Everything ``_load_trained`` below actually reads back out of a sidecar. A saved rung is
#: reconstructed from these keys and nothing else, so a key written at train time and absent
#: here is state that is silently dropped on reload.
_SIDECAR_RECONSTRUCTED: Final = frozenset(
    {
        "columns",
        "hparams",
        "n_train_rows",
        "n_train_positive_rows",
        "n_train_positive_merchants",
        "train_seconds",
        # Rung 5's fitted state (T-0120). These five are the reason the refusal below
        # exists at all, and they are handled here rather than exempted: `pooling` and
        # `tau` are FITTED on validation by `rung5_mil.fit_tau` and change the score, so a
        # reload that dropped them would return a rung scoring under the default pooling
        # while every label still said Rung 5. `passes`, `n_train_bags` and
        # `n_train_positive_bags` are constructor arguments of `TrainedMIL` and are
        # rebuilt from here for the same reason `n_train_rows` is.
        "pooling",
        "tau",
        "passes",
        "n_train_bags",
        "n_train_positive_bags",
    }
)

#: Written for provenance and deliberately not reconstructed — they describe the run or are
#: derivable from what is, so rebuilding the model without them loses nothing. Kept as an
#: explicit list rather than an "ignore the rest" rule: the whole point is that a NEW key
#: nobody has classified stops the reload instead of vanishing from it.
_SIDECAR_PROVENANCE_ONLY: Final = frozenset(
    {
        "rung",
        "seed",
        "train_as_of_day",
        "n_columns",
        "model_size_mb",
        "label_coverage_at_train_boundary",
        # Rung 5. `tau_selection_table` is the whole validation grid `fit_tau` scored; the
        # winning row is already reconstructed as `pooling`/`tau` above, so the table is a
        # *report* — ticket #54 makes the fitted tau a result, and the losing rows are what
        # make the winning one mean anything. `subsample` is the merchant sample this rung
        # was fitted and scored on, and it rides on every Rung 5 artefact so no reader can
        # mistake the row for a full-population one.
        "tau_selection_table",
        "subsample",
    }
)


def _load_trained(rung: int, seed: int) -> rung2_lgbm.TrainedRung | rung5_mil.TrainedMIL:
    """Rebuild a trained rung from its booster file and sidecar.

    **Refuses a sidecar carrying state it cannot rebuild.** Rung 5 (T-0120) fits a
    ``pooling`` and a ``tau`` at train time; ``TrainedRung`` has nowhere to put them, so a
    reload would return a rung that scores with the *default* pooling while every label,
    filename and log line still called it Rung 5. That is a wrong number that looks
    entirely ordinary — the failure mode this project is built to refuse — and it would not
    raise anywhere, because a bare ``TrainedRung`` is perfectly valid.

    So the check is on the sidecar, not on the rung number: any future rung that saves
    fitted state hits it too, without anyone remembering to come back here.

    **T-0120 extended this rather than bypassing it.** Rung 5's ``pooling`` and ``tau`` are
    now in ``_SIDECAR_RECONSTRUCTED`` and are rebuilt into a :class:`rung5_mil.TrainedMIL`
    below; the refusal is untouched and still fires on any key nobody has classified. The
    two lists and this function moved together, which is exactly what the docstring above
    asked the next person to do.
    """
    import lightgbm as lgb

    path = _model_path(rung, seed)
    sidecar = json.loads(_sidecar_path(rung, seed).read_text(encoding="utf-8"))
    dropped = set(sidecar) - _SIDECAR_RECONSTRUCTED - _SIDECAR_PROVENANCE_ONLY
    if dropped:
        raise typer.BadParameter(
            f"rung {rung} seed {seed}: the sidecar carries {sorted(dropped)}, which "
            f"_load_trained does not reconstruct. Reloading would return a rung missing "
            f"that fitted state and score with defaults under the right name. Extend "
            f"_load_trained and _SIDECAR_RECONSTRUCTED together, or do not persist it."
        )
    booster = rung2_lgbm.TrainedRung(
        rung=rung,
        booster=lgb.Booster(model_file=str(path)),
        columns=tuple(sidecar["columns"]),
        params=rung2_lgbm.HParams(**sidecar["hparams"]),
        n_train_rows=sidecar["n_train_rows"],
        n_train_positive_rows=sidecar["n_train_positive_rows"],
        n_train_positive_merchants=sidecar["n_train_positive_merchants"],
        train_seconds=sidecar["train_seconds"],
    )
    if rung != 5:
        return booster
    missing = sorted({"pooling", "tau", "passes", "n_train_bags", "n_train_positive_bags"}
                     - set(sidecar))
    if missing:
        raise typer.BadParameter(
            f"rung 5 seed {seed}: the sidecar is missing {missing}. Rung 5 is an instance "
            "model PLUS a fitted pooling; without those keys the booster on disk is only "
            "half of it, and scoring it under the default pooling would produce a wrong "
            "number wearing the right name. Re-run `train --rung 5`."
        )
    return rung5_mil.TrainedMIL(
        instance=booster,
        pooling=sidecar["pooling"],
        tau=float(sidecar["tau"]),
        n_train_bags=int(sidecar["n_train_bags"]),
        n_train_positive_bags=int(sidecar["n_train_positive_bags"]),
        passes=int(sidecar["passes"]),
    )


def _score_rung9(
    *,
    full: Any,
    rows: Any,
    seed: int,
    root: Path,
    boundaries: SplitBoundaries,
) -> tuple[np.ndarray, rung9_rank_cusum.RankCusum, np.ndarray]:
    """Rung 9's score: the incumbent's probability, blended with its rank-CUSUM accumulator.

    **Rung 9 is a wrapper over Rung 3, not a new model**, so it loads Rung 3's artefact
    rather than training anything of its own. The only fitted object is the three-parameter
    logistic blend that turns ``(logit incumbent, accumulator)`` back into the calibrated
    probability the decision layer contracts for, and **it is fitted on TRAIN rows only** —
    the panel carries a row only where the merchant's fold matches the day's split, so the
    train rows here belong to train-fold merchants on train days and share neither a
    merchant nor a day with the split being scored.

    The CUSUM itself consumes no labels at all. Three parameters against the train fold's
    positives is inside the label budget with room to spare, which is the whole reason a
    method that needs no labels was weighted above one that does.
    """
    incumbent = _load_trained(3, seed)
    profiles = {
        r["merchant_id"]: MerchantProfile(**r)
        for r in pl.read_parquet(root / "profiles.parquet").to_dicts()
    }

    def channel(panel: Any) -> tuple[np.ndarray, np.ndarray]:
        base = incumbent.predict(panel.x, panel.columns)
        acc = rung9_rank_cusum.accumulator(
            base,
            panel.day,
            panel.merchant_id,
            rung9_rank_cusum.cohort_labels(profiles, panel.merchant_id),
        )
        return base, acc

    train = full.select("train")
    if train.x.shape[0] == 0:
        raise typer.BadParameter(
            "the panel holds no train rows, so Rung 9's blend cannot be fitted. Fitting it "
            "on the split being scored would be self-evaluation."
        )
    base_tr, acc_tr = channel(train)
    y_tr = _training_labels(
        root,
        np.array(sorted(set(train.merchant_id.tolist()))),
        _epoch_end(boundaries.train[1], boundaries),
    )
    by_merchant = dict(
        zip(sorted(set(train.merchant_id.tolist())), y_tr, strict=True)
    )
    y_rows = np.array([by_merchant[m] for m in train.merchant_id], dtype=int)
    blend = rung9_rank_cusum.RankCusum.fit(
        incumbent=base_tr, accumulator=acc_tr, y=y_rows
    )

    base, acc = channel(rows)
    return blend.predict(base, acc), blend, acc


@app.command("eval")
def score_split(
    rung: int = typer.Option(
        2, "--rung", help="0, 1, 2, 3, 4, 5, 6 or 9. Rung 9 wraps Rung 3."
    ),
    seed: int = typer.Option(42, "--seed", help="Threaded into the random_at_k floor."),
    split: str = typer.Option("val", "--split", help="val (default) or test. "
                              "test REFUSES without RAKSHAK_UNLOCK=1."),
    root: Path = typer.Option(Path("data/v2"), "--root", help="Generated dataset."),  # noqa: B008
    panel: Path = typer.Option(  # noqa: B008
        dataset.DEFAULT_PANEL, "--panel", help="Materialised feature panel."
    ),
    config: Path = typer.Option(  # noqa: B008
        Path("configs/scenario_v2.yaml"), "--config", "-c", help="Scenario manifest."
    ),
    floor: str = typer.Option("", "--floor", help="With --rung 0: which floor to score."),
    alphas: str = typer.Option(
        "0.05,0.10", "--alphas", help="With --rung 6: comma-separated nominal false-HOLD "
        "rates. One EvalResult row is written per alpha."
    ),
    base_rung: int = typer.Option(
        2, "--base-rung", help="With --rung 6: which rung's decisions the conformal "
        "wrapper softens. Rung 6 is a wrapper, not a scorer — it has no score of its own."
    ),
    exposure_arm: str = typer.Option(
        "declared", "--exposure",
        help="Cycle-4 A/B (PRE-REGISTRATION-CYCLE4 §4.2). 'declared' is cycle 3's wiring "
        "and the default, so nothing changes silently; 'realised' prices the decision on "
        "trailing-30d captured GMV via the Rung 8 wrapper."
    ),
) -> None:
    """Score one rung and write a complete ``EvalResult`` row.

    Refuses the test split unless ``RAKSHAK_UNLOCK=1`` and refuses any split at all if the
    frozen eval modules have changed. Both checks happen before the panel is opened.

    **Rungs 5 and 6 are not shaped like Rungs 0-4 and are dispatched rather than folded in.**
    Rung 5 scores bags of (merchant, day, payer) capsules over a merchant subsample, so its
    row universe is not this panel's. Rung 6 is a *decision-policy wrapper*: it emits no
    score at all, it softens another rung's HOLDs, and it writes one row per alpha rather
    than one row. Forcing either into the branch below would mean a ``--rung`` that means
    three different things, which is how a results table ends up comparing two quantities
    that share a column heading.
    """
    if split not in ("train", "val", "test"):
        raise typer.BadParameter(f"split is train/val/test; got {split!r}")
    if exposure_arm not in ("declared", "realised"):
        raise typer.BadParameter(
            f"--exposure is 'declared' or 'realised'; got {exposure_arm!r}"
        )
    drift = _guard(split, root=ROOT)  # type: ignore[arg-type]
    for line in drift:
        typer.echo(f"[lock drift, unenforced] {line}")

    if rung in (5, 6):
        if split != "val":
            raise typer.BadParameter(
                f"rung {rung} scores the validation split only; got {split!r}. Rung 6 "
                "calibrates on validation by contract (rung6_conformal.calibrate refuses "
                "anything else) and Rung 5's pooling is selected there; the test split "
                "opens exactly once, in T-0116, and not from here."
            )
        if rung == 5:
            _eval_rung5(seed=seed, root=root, config=config, panel_path=panel)
        else:
            from rakshak import score_rung6

            for path, row in score_rung6.score_val(
                base_rung=base_rung,
                seeds=(seed,),
                alphas=tuple(float(a) for a in alphas.split(",")),
                root=root,
                panel=panel,
                config=config,
            ):
                typer.echo(
                    json.dumps(
                        {
                            "wrote": str(path),
                            "alpha": row["alpha"],
                            "savings": row["savings"],
                            "n_hold_base_rung": row["n_hold_base_rung"],
                            "n_softened": row["n_softened_hold_to_review"],
                        }
                    )
                )
        return

    rng = np.random.default_rng(seed)
    params = _cost_params(config)
    policy = _action_policy(config)
    full = dataset.load_panel(panel)
    rows = full.select(split)  # type: ignore[arg-type]
    if rows.x.shape[0] == 0:
        raise typer.BadParameter(f"the panel holds no {split!r} rows")

    merchants = sorted(set(rows.merchant_id.tolist()))
    boundaries = _boundaries(config)
    span = getattr(boundaries, split)
    truth = _build_truth(root, merchants, cutoff_day=span[0] - 1, boundaries=boundaries)
    k = _capacity(len(merchants))
    exposure = rows.column("p_declared_monthly_gmv")

    latency_ms = float("nan")
    model_size_mb = float("nan")
    label = f"rung{rung}"
    if rung == 0:
        label = floor or "volume_rank"
        if label not in rung0_floors.ROW_FLOORS:
            raise typer.BadParameter(
                f"--floor must be one of {rung0_floors.ROW_FLOORS}; got {label!r}. "
                "all_hold alerts on the whole population, so alerts_per_day is the "
                "population size and the harness refuses to compute metrics above "
                "capacity K. Its savings is on every row as savings_floor_all_hold."
            )
        if label == "all_pass":
            score = np.zeros(rows.x.shape[0], dtype=np.float64)
            action = rung0_floors.all_pass_actions(rows.x.shape[0])
        else:
            volume_by_id = dict(zip(truth.merchant_id, truth.volume, strict=True))
            score = (
                rung0_floors.random_scores(rows.x.shape[0], rng)
                if label == "random_at_k"
                else rung0_floors.volume_scores(
                    np.array([volume_by_id[m] for m in rows.merchant_id])
                )
            )
            action = rung0_floors.floor_actions(score, rows.day, k)
    else:
        if rung == 1:
            score = rung1_rules.score(rows.x, rows.columns)
        elif rung == 9:
            score, blend, acc = _score_rung9(
                full=full, rows=rows, seed=seed, root=root, boundaries=boundaries
            )
            latency_ms = float("nan")
        else:
            model = _load_trained(rung, seed)
            started = _time.perf_counter()
            score = model.predict(rows.x, rows.columns)
            latency_ms = (_time.perf_counter() - started) / rows.x.shape[0] * 1000.0
            model_size_mb = round(model.size_mb(_model_path(rung, seed)), 4)
        # Cycle-4 arm B. `declared` forwards to select_actions exactly as before —
        # test_rung8_exposure.py asserts the wrapper is a byte-identical no-op when the
        # two exposure vectors agree, which is what makes the A/B a controlled comparison
        # rather than two things changing at once.
        decision = capacity.DEFAULT_DECISION
        if exposure_arm == "realised":
            exposure = rung8_realised_exposure.realised_exposure_inr(
                exposure, rows.column("v_declared_ratio")
            )
            decision = rung8_realised_exposure.RealisedExposure(
                inner=capacity.DEFAULT_DECISION, exposure=exposure
            )
            label = f"{label}_realised_exposure"
        action = decision.decide(
            capacity.DecisionRequest(
                score=score,
                day=rows.day,
                exposure_inr=exposure,
                k=k,
                params=params,
                hold_policy=policy,
            )
        )

    output = RungOutput(
        merchant_id=rows.merchant_id, day=rows.day, score=score, action=action
    )
    y, keep = _day_labels(output, truth)
    order = np.argsort(truth.merchant_id)
    idx = order[np.searchsorted(truth.merchant_id[order], output.merchant_id)]
    ceiling = oracle_savings(
        output.day[keep], y[keep], truth.loss_inr[idx][keep], k, params
    )

    summary_path = panel.parent / "features_summary.json"
    state_bytes = (
        float(json.loads(summary_path.read_text(encoding="utf-8"))["state_bytes_p99"])
        if summary_path.exists()
        else float("nan")
    )
    result = build_eval_result(
        rung=rung,
        split=split,  # type: ignore[arg-type]
        output=output,
        truth=truth,
        k=k,
        params=params,
        rng=rng,
        perf=PerfBudget(
            p99_latency_ms=latency_ms,
            state_bytes_p99=state_bytes,
            model_size_mb=model_size_mb,
        ),
        oracle_savings=ceiling,
        eval_lock_sha=load_lock(ROOT, lock_path=_lock_path(ROOT))["eval_module_sha256"],
        open_count=read_open_count(ROOT, lock_path=_lock_path(ROOT)),
        git_sha=str(load_lock(ROOT, lock_path=_lock_path(ROOT))["frozen_at_git_sha"]),
    )

    if rung == 9:
        payload_extra_rung9 = {
            "incumbent_rung": 3,
            "cusum_k": rung9_rank_cusum.K_REFERENCE,
            "cusum_c_max": rung9_rank_cusum.C_MAX,
            "blend_coef_logit_incumbent": float(blend.model.coef_[0][0]),
            "blend_coef_accumulator": float(blend.model.coef_[0][1]),
            "blend_intercept": float(blend.model.intercept_[0]),
            "accumulator_mean": float(np.mean(acc)),
            "accumulator_p99": float(np.quantile(acc, 0.99)),
            "accumulator_frac_at_cap": float(
                np.mean(acc >= rung9_rank_cusum.C_MAX - 1e-9)
            ),
        }
    else:
        payload_extra_rung9 = {}

    if split == "test":
        record_open(ROOT, [rung], lock_path=_lock_path(ROOT))
        typer.echo(
            f"opened the test split. {_lock_path(ROOT).name} open_count is now "
            f"{read_open_count(ROOT, lock_path=_lock_path(ROOT))} — COMMIT IT."
        )

    payload: dict[str, Any] = dataclasses.asdict(result)
    payload["recall_by_typology"] = {
        str(key): value for key, value in result.recall_by_typology.items()
    }
    payload |= {
        **payload_extra_rung9,
        "label": label,
        "exposure_arm": exposure_arm,
        "decision_policy": decision.name if rung != 0 else "floor_actions(review_only)",
        "capacity_k": k,
        "n_merchants_scored": len(merchants),
        "n_rows_scored": int(rows.x.shape[0]),
        "n_rows_kept": int(keep.sum()),
        "n_censored_dropped": int((~keep).sum()),
        # T-0101 (GitHub #34): both denominators, on every row, at MERCHANT grain (the
        # two above are row/merchant-day grain). `truth` is one row per merchant scored
        # in this split; `is_censored` is exactly the set whose label had not resolved
        # by the cutoff this Truth was built at (`_build_truth` above). Landed here as
        # sidecar diagnostics rather than as new `EvalResult` fields, so `schemas.py`
        # stays untouched (docs/RE-FREEZE-2026-08-31.md Amendment 4).
        "n_labelled_merchants": int((~truth.is_censored).sum()),
        "n_censored_merchants": int(truth.is_censored.sum()),
        "oracle_savings": ceiling,
        "beats_all_floors": result.beats_all_floors,
        "n_features": len(rows.columns) if rung in (0, 1) else len(_columns_for(rung)),
        "split_boundaries": {
            "train": list(boundaries.train),
            "val": list(boundaries.val),
            "test": list(boundaries.test),
        },
    }
    results = panel.parent / RESULT_DIR.name
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{label}_{split}_seed{seed}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def explain(
    rung: int = typer.Option(7, "--rung", help="7 — the HSMM onset explainer."),
    seeds: str = typer.Option(
        "1", "--seeds", help="Comma-separated EM INITIALISATION seeds. Not the lock's five "
        "model seeds: Rung 7 is not judged on the PR-AUC/TTD adoption margin."
    ),
    fit_pool_size: int = typer.Option(500, "--fit-pool-size"),
    n_iter: int = typer.Option(15, "--n-iter", help="EM iteration cap."),
    n_states: int = typer.Option(4, "--n-states"),
    root: Path = typer.Option(Path("data/v2"), "--root", help="Generated dataset."),  # noqa: B008
    config: Path = typer.Option(  # noqa: B008
        DEFAULT_CONFIG, "--config", "-c", help="Scenario manifest."
    ),
) -> None:
    """Run the Stage-2 explainer and write its explanation-quality artifact.

    **A separate stage from ``eval`` on purpose.** ``eval`` writes ``EvalResult`` rows into
    ``data/v2/eval/``, which ``artifacts/build.py`` globs into ``ladder.json``; an explainer
    routed through it would acquire a ladder row with an empty PR-AUC column and read as a
    rung that scored badly rather than as one that does not score at all. The artifact goes
    to ``data/v2/explanation_quality/`` and the metric is ``onset_localisation_error``,
    which ``EVAL-LOCK-CYCLE3.json`` declared for exactly this.

    The split guard still runs, on ``"val"``. Rung 7 reads days 0-299 and nothing later.
    """
    if rung != 7:
        raise typer.BadParameter(
            f"rung {rung} is not an explainer; 7 is. Rungs 0-6 are scored through `eval`."
        )
    drift = _guard("val", root=ROOT)
    for line in drift:
        typer.echo(f"[lock drift, unenforced] {line}")

    from rakshak import score_rung7

    score_rung7.measure(
        seeds=tuple(int(s) for s in seeds.split(",")),
        fit_pool_size=fit_pool_size,
        n_iter=n_iter,
        n_states=n_states,
        root=root,
        config=config,
        echo=typer.echo,
    )


if __name__ == "__main__":  # pragma: no cover - `python -m rakshak.cli`
    app()
