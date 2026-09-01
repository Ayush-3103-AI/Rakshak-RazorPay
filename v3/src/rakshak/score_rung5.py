"""T-0121: Rung 5 (MIL) scoring entry point.

**On the eval side of the Prime Directive 3 quarantine**, deliberately: this module reads
``ground_truth.parquet`` (via ``rakshak.cli``'s own helpers) to build training labels, the
merchant-day ``Truth`` and the stratified subsample below. It is a sibling of ``cli.py``,
not a member of ``src/rakshak/features/`` or ``src/rakshak/models/`` — G4's AST scan
(``tests/gates/test_g4_no_leakage.py``) only walks those two packages, so this file is
outside the quarantine boundary by construction, the same way ``cli.py`` itself is.

``cli.py``'s private ``_boundaries`` / ``_build_truth`` / ``_training_labels`` /
``_cost_params`` / ``_action_policy`` / ``_capacity`` / ``_epoch_end`` / ``_label_path`` are
imported and reused rather than re-implemented, so Rung 5 is scored under exactly the same
split geometry, cost matrix, decision policy and ``Truth`` construction as Rungs 0-4 — the
ladder compares rungs on the same axes only if nothing about how a number is built quietly
forks. The bulk lives here rather than in ``cli.py`` because Rung 5's data acquisition is
its own several-hundred-line problem (a merchant subsample, a chunked resumable capsule
materialiser, a RAM floor) that has nothing to do with the other rungs; ``cli.py`` calls
:func:`fit_seed` and :func:`score_seed` from its ``train``/``eval`` stages.

## Why this exists and not a straight run over all 20,000 merchants

``features/capsules.py`` (T-0119) is explicit that ``capsules_as_of`` is **not bounded**: it
rescans the whole visible prefix on every call, and T-0120's own module docstring already
concluded Rung 5 is not servable under NFR-04 as implemented. A cost probe
(``docs/LOGBOOK.md`` T-0121) benchmarked single-merchant calls at 50 and 200 merchants
(multiple in-run checkpoints each, not one early sample — a single early sample was wrong
by ~5x elsewhere in this project today) and measured ~4.1-5.4 merchants/second, stable
across both tiers. Extrapolated to 20,000 merchants over the 300 days Rung 5 actually needs
(train 0-239 + val 240-299, captured in one ``as_of``-bounded scan per merchant since
``capsules_as_of`` returns every earlier day too): **~62-81 minutes**, over the ~60-minute
go/no-go line, and the box's free RAM was independently observed to fluctuate between 0.5
and 5.5 GB *during that same probe* with other lanes running — a second, independent reason
not to commit to the full population run blind.

**Decision: NO-GO on the full population. Scored on a stratified subsample instead**
(:func:`sample_merchants`), clearly labelled as a subsample result everywhere it is
reported, never merged with a full-population claim.
"""

from __future__ import annotations

import ctypes
import dataclasses
import gc
import json
import time as _time
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from rakshak.cli import (
    ROOT as REPO_ROOT,
)
from rakshak.cli import (
    _action_policy,
    _boundaries,
    _build_truth,
    _capacity,
    _cost_params,
    _epoch_end,
    _training_labels,
)
from rakshak.eval.capacity import select_actions
from rakshak.eval.lock import load_lock, read_open_count, resolve_authoritative
from rakshak.eval.metrics import (
    PerfBudget,
    RungOutput,
    build_eval_result,
    day_labels,
    pr_auc,
    roc_auc,
)
from rakshak.eval.oracle import oracle_savings
from rakshak.eval.splits import SplitBoundaries
from rakshak.features.capsules import CAPSULE_VECTOR_COLUMNS, capsules_as_of
from rakshak.models import dataset
from rakshak.models.rung2_lgbm import DEFAULT_PARAMS
from rakshak.models.rung5_mil import feature_columns, fit_tau
from rakshak.models.rung5_mil import train as train_mil
from rakshak.schemas import CAPSULE_SCHEMA, Action
from rakshak.store import EventStore

__all__ = [
    "SEEDS",
    "capsule_scan",
    "TARGET_N_MERCHANTS",
    "build_bags",
    "materialise_capsules",
    "sample_merchants",
    "score_seed",
]

CONFIG: Final = Path("configs/scenario_v2.yaml")

#: The generated dataset. ``cli.ROOT`` is the REPOSITORY root (it resolves EVAL-LOCK*.json);
#: every cli command carries the data root as a separate ``--root``. Both are needed here,
#: and conflating them reads ``ground_truth.parquet`` from the wrong directory.
DATA_ROOT: Final = Path("data/v2")
SUBSAMPLE_DIR: Final = Path("data/v2/rung5_subsample")
RESULT_DIR: Final = Path("data/v2/eval")

#: cli.py's score_split scores Rungs 0-4 at these same five seeds.
SEEDS: Final[tuple[int, ...]] = (42, 43, 44, 45, 46)

#: The merchant subsample is chosen ONCE (below), independent of the training seed — only
#: LightGBM's seed varies across SEEDS, so all five runs score the same bags.
SAMPLE_SEED: Final = 42

#: BELOW the ticket's suggested 2,000-4,000 band, and that is a measurement, not a
#: shortcut. A 250-merchant trial run produced **1,895,459 capsule rows — 7,582 per
#: merchant**, 3.4x the volume estimated from the cost probe's row counts. At 3,000
#: merchants that is ~22.7M instance rows, a ~2.0 GB float64 design matrix before
#: LightGBM copies it, on a 16 GB box where free RAM was measured at 0.5-5.9 GB with
#: other lanes running. The same trial also watched the scan rate collapse from
#: ~2 merchants/s to ~0.02 merchants/s once free RAM reached 0.5 GB and the machine
#: began paging. RAM is the binding resource; the subsample is sized to it and the
#: shortfall against the suggested band is reported rather than hidden.
TARGET_N_MERCHANTS: Final = 800

#: Abort the materialiser rather than risk the box — other lanes are concurrently running.
MIN_FREE_GB: Final = 0.8


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _free_gb() -> float:
    """Free physical RAM, stdlib-only (no psutil pinned in this env).

    ponytail: Windows-only (``ctypes.windll``); add psutil if this ever needs to run
    cross-platform.
    """
    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return float(stat.ullAvailPhys) / 1e9


# ─────────────────────────────────────────────────────────────────────────────
# Stratified merchant subsample
# ─────────────────────────────────────────────────────────────────────────────


def sample_merchants(
    root: Path, panel_path: Path, *, n_target: int = TARGET_N_MERCHANTS, seed: int = SAMPLE_SEED
) -> dict[str, list[str]]:
    """``{"train": [...], "val": [...]}`` merchant ids, stratified by typology.

    Population prevalence is 1.47% (294/20,000 fraud merchants total); a proportional
    draw at ``n_target`` would leave ~29-59 fraud merchants split across 9 typologies, too
    few to fit or tau-select against. So **every** fraud merchant already assigned to the
    train or val fold (from the materialised panel's own ``split`` column, i.e. exactly the
    T-0101 fold the rest of the ladder uses) is kept, and the rest of ``n_target`` is filled
    with a random draw of legit merchants at the population's train:val ratio. This
    deliberately **oversamples the positive class** relative to the population rate — stated
    here, and visible downstream in ``EvalResult.prevalence``, which reports the
    subsample's own (higher) rate honestly rather than a population figure this subsample
    was not drawn to match.
    """
    gt = pl.read_parquet(root / "ground_truth.parquet").select("merchant_id", "risk_typology_id")
    panel_split = pl.read_parquet(panel_path, columns=["merchant_id", "split"]).unique()
    # inner join drops test-fold merchants automatically: they never appear in the panel,
    # which was materialised only through the last validation day (models/dataset.py).
    joined = gt.join(panel_split, on="merchant_id", how="inner")

    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {}
    for split_name in ("train", "val"):
        sub = joined.filter(pl.col("split") == split_name)
        fraud = sub.filter(pl.col("risk_typology_id").is_not_null())["merchant_id"].to_list()
        legit = sub.filter(pl.col("risk_typology_id").is_null())["merchant_id"].to_list()
        share = sub.height / joined.height
        target = max(len(fraud), round(n_target * share))
        n_legit = max(0, target - len(fraud))
        chosen_legit = rng.choice(
            np.array(legit, dtype=object), size=min(n_legit, len(legit)), replace=False
        ).tolist()
        out[split_name] = sorted(fraud + chosen_legit)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bulk capsule materialiser — batch-friendly, NOT the point-in-time-safe online shape
# ─────────────────────────────────────────────────────────────────────────────


def materialise_capsules(
    store: EventStore,
    merchants: list[str],
    as_of: Any,
    out_dir: Path,
    *,
    echo: Any = None,
    min_free_gb: float = MIN_FREE_GB,
    chunk: int = 100,
) -> Path:
    """One ``capsules_as_of`` call per merchant, written out a chunk at a time.

    This is the "does not need to be point-in-time-safe" bulk shape the ticket sanctions:
    ``as_of`` is fixed once at the end of the validation window, so a single call per
    merchant returns every earlier day's capsule rows too (train 0-239 AND val 240-299 in
    one scan). A day-X row built this way may use evidence from days after X but at or
    before ``as_of`` — ``features/capsules.py`` records that only ``payer_is_new`` is
    lookahead-safe for free and ``device_shared_payers`` is not. That is acceptable here
    and nowhere else: this is a training/eval input, not a served feature.

    **Chunked and resumable.** Each chunk of ``chunk`` merchants is written to its own
    ``chunk_NNNN.parquet``; a chunk whose file already exists is skipped. A ``MemoryError``
    abort therefore costs one chunk, not the whole run, and the caller can simply re-invoke.
    Merchants are sorted before chunking, so the chunk boundaries are deterministic and a
    resumed run reproduces the same partition.

    Raises ``MemoryError`` rather than continuing below ``min_free_gb`` — the trial run
    watched throughput fall ~100x once this box started paging, so backing off is faster
    than pushing on, quite apart from not taking the other lanes down.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(merchants)
    chunks = [ordered[i : i + chunk] for i in range(0, len(ordered), chunk)]
    t0 = _time.perf_counter()
    done = 0
    for c, ids in enumerate(chunks):
        path = out_dir / f"chunk_{c:04d}.parquet"
        done += len(ids)
        if path.exists():
            continue
        frames = [capsules_as_of(store, m, as_of=as_of) for m in ids]
        frame = pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema=CAPSULE_SCHEMA)
        # Written to a temp name and renamed, so an interrupted write never leaves a
        # truncated chunk that a resumed run would happily skip as "already done".
        tmp = path.with_suffix(".partial")
        frame.write_parquet(tmp)
        tmp.replace(path)
        del frames, frame
        gc.collect()
        free = _free_gb()
        if echo is not None:
            echo(
                f"  capsules {done}/{len(ordered)} merchants, "
                f"{_time.perf_counter() - t0:.0f}s, free RAM {free:.2f} GB"
            )
        if free < min_free_gb:
            raise MemoryError(
                f"free RAM {free:.2f} GB < floor {min_free_gb} GB after chunk {c} "
                f"({done}/{len(ordered)} merchants). Chunks already written are kept; "
                "re-invoke to resume."
            )
    return out_dir


def capsule_scan(out_dir: Path) -> pl.LazyFrame:
    """Lazy scan over the materialised chunks. Row order is irrelevant: ``build_bags``
    sorts by bag ordinal after the join, which is what ``bag_offsets`` actually requires."""
    return pl.scan_parquet(out_dir / "chunk_*.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# Bags: one per (merchant, day) already scored by the rest of the ladder for that split
# ─────────────────────────────────────────────────────────────────────────────


def build_bags(
    panel: dataset.Panel,
    split: str,
    merchants: list[str],
    capsules: pl.LazyFrame,
    boundaries: SplitBoundaries,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(x, bag_index, bag_merchant_id, bag_day, exposure_inr)`` for one split.

    The bag universe is the materialised panel's own dense ``(merchant, day)`` grid for
    these merchants — the exact row set Rungs 0-4 are scored on for this split — so an
    (merchant, day) with zero capsule rows is a real, present, **empty bag**
    (``EMPTY_BAG_SCORE``, per ``rung5_mil``'s module docstring), not a missing one.
    """
    day_lo, day_hi = getattr(boundaries, split)
    sub = panel.select(split)  # type: ignore[arg-type]
    mask = np.isin(sub.merchant_id, np.array(merchants, dtype=object))
    sub = sub.rows(mask)

    order = np.lexsort((sub.day, sub.merchant_id))
    bag_merchant = sub.merchant_id[order]
    bag_day = sub.day[order]
    exposure = sub.column("p_declared_monthly_gmv")[order]
    n_bags = bag_merchant.size
    bag_table = pl.DataFrame(
        {
            "merchant_id": pl.Series("merchant_id", bag_merchant.tolist(), dtype=pl.String),
            "day": pl.Series("day", bag_day.astype(np.int64)),
            "bag_ordinal": pl.Series("bag_ordinal", np.arange(n_bags, dtype=np.int64)),
        }
    )

    lo_date = boundaries.origin + timedelta(days=int(day_lo))
    hi_date = boundaries.origin + timedelta(days=int(day_hi))
    cap = (
        capsules.select("merchant_id", "event_date", "payer_id", *CAPSULE_VECTOR_COLUMNS)
        .filter(
            pl.col("merchant_id").is_in(merchants)
            & (pl.col("event_date") >= lo_date)
            & (pl.col("event_date") <= hi_date)
        )
        .with_columns(
            (pl.col("event_date") - pl.lit(boundaries.origin)).dt.total_days().alias("day")
        )
        .join(bag_table.lazy(), on=["merchant_id", "day"], how="inner")
        .sort("bag_ordinal")
        .collect()
    )

    x = cap.select(CAPSULE_VECTOR_COLUMNS).to_numpy().astype(np.float64)
    bag_index = cap["bag_ordinal"].to_numpy().astype(np.intp)
    return x, bag_index, bag_merchant, bag_day, exposure


def bar_on_the_same_bags(
    panel: dataset.Panel,
    val_merchants: list[str],
    seed: int,
    y: np.ndarray,
    keep: np.ndarray,
) -> dict[str, Any]:
    """Rung 2 — the declared bar — scored on **exactly the bags Rung 5 was scored on**.

    Without this the Rung 5 row is uninterpretable and dangerously so. Its subsample
    oversamples the positive class by design, so its prevalence is ~25% against the
    population's 1.47%, and **PR-AUC rises with prevalence**: a PR-AUC of 0.72 at 25%
    prevalence is a far weaker result than 0.83 at 1.4%, while looking like a near miss.
    Anyone comparing the two numbers straight off the ladder would draw the opposite
    conclusion from the true one.

    So the comparator is computed here, on the same merchants, the same days, the same
    labels and the same censoring mask, and carried inside the Rung 5 artifact. That makes
    the relative-PR-AUC delta against Prime Directive 5's >=10% margin a single-variable
    comparison — bag pooling versus the merchant-day aggregate — which is the comparison
    ticket #54 actually asks for.

    Uses the same ``seed`` as Rung 5, so LightGBM's four seeds match too.
    """
    from rakshak.cli import _load_trained

    sub = panel.select("val")
    mask = np.isin(sub.merchant_id, np.array(val_merchants, dtype=object))
    sub = sub.rows(mask)
    order = np.lexsort((sub.day, sub.merchant_id))
    columns = dataset.base_columns()
    model = _load_trained(2, seed)
    score = model.predict(sub.with_columns(columns).x[order], columns)
    return {
        "rung": 2,
        "seed": seed,
        "pr_auc": pr_auc(y[keep], score[keep]),
        "roc_auc": roc_auc(y[keep], score[keep]),
        "note": "Rung 2 on the identical bags/labels/censoring as this Rung 5 row. The "
        "ONLY admissible PR-AUC comparison for it — the ladder's own rung2 row is scored "
        "on all 3,036 validation merchants at ~1.5% prevalence and is not comparable.",
    }


def _drop_censored(
    x: np.ndarray, bag_index: np.ndarray, y: np.ndarray, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict a (x, bag_index, per-bag y) triple to ``keep`` bags, renumbering ordinals.

    Mirrors what ``build_eval_result`` does internally (``y[keep]``, ``score[keep]``) for
    the bag-pooling functions in ``rung5_mil``, which need dense ``bag_index in [0, n_bags)``
    rather than a boolean mask alongside it.
    """
    keep_ids = np.flatnonzero(keep)
    remap = np.full(keep.size, -1, dtype=np.intp)
    remap[keep_ids] = np.arange(keep_ids.size, dtype=np.intp)
    inst_keep = keep[bag_index]
    return x[inst_keep], remap[bag_index[inst_keep]], y[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Two halves, because `cli.py` has two stages
#
# `fit_seed` is `rakshak.cli train --rung 5`; `score_seed` is `rakshak.cli eval --rung 5`.
# They are separate functions rather than one because they run in separate processes, and
# because a Rung 5 that could only ever be trained-and-scored in one breath would never
# exercise the `_load_trained` round-trip that carries its fitted pooling — which is the
# exact failure `cli.py::_load_trained`'s own docstring was written to refuse.
# ─────────────────────────────────────────────────────────────────────────────


def fit_seed(
    seed: int,
    *,
    root: Path,
    boundaries: SplitBoundaries,
    panel: dataset.Panel,
    capsules: pl.LazyFrame,
    train_merchants: list[str],
    val_merchants: list[str],
) -> tuple[Any, list[dict[str, Any]]]:
    """Train the instance model on TRAIN bags, then select the pooling on VAL bags.

    Returns ``(TrainedMIL, tau_selection_table)``.

    **The tau selection is on validation and that makes Rung 5's reported PR-AUC
    selection-optimistic in a way Rungs 0-4's are not.** ``rung5_mil.fit_tau`` scores a
    ten-row grid on the validation bags and keeps the best, and the number this rung is
    then judged on is computed on those same bags. It is done here anyway because it is
    what the rung *is* — the fitted tau is the result ticket #54 asks for, and there is no
    third split to select on without opening the test split. The mitigation is disclosure,
    not a correction: the whole grid is returned, persisted and reported, so the gap
    between the selected row and the median row is visible rather than asserted.
    """
    columns = feature_columns()
    x_tr, bag_idx_tr, bm_tr, _bd_tr, _exp_tr = build_bags(
        panel, "train", train_merchants, capsules, boundaries
    )
    train_as_of = _epoch_end(boundaries.train[1], boundaries)
    bag_y_tr = _training_labels(root, bm_tr, train_as_of)

    hparams = DEFAULT_PARAMS.with_seed(seed)
    # merchant_id is per-INSTANCE inside rung2_lgbm.train (it masks it with the instance
    # positive mask to count distinct positive merchants), while bm_tr is per-BAG. Passing
    # the bag-grain array raises; passing it expanded is the count the sidecar means.
    model = train_mil(
        x_tr, bag_idx_tr, bag_y_tr, columns, params=hparams, merchant_id=bm_tr[bag_idx_tr]
    )

    x_val, bag_idx_val, bm_val, bd_val, _exp_val = build_bags(
        panel, "val", val_merchants, capsules, boundaries
    )
    truth = _build_truth(
        root, val_merchants, cutoff_day=boundaries.val[0] - 1, boundaries=boundaries
    )
    placeholder = RungOutput(
        merchant_id=bm_val,
        day=bd_val,
        score=np.zeros(bm_val.size),
        action=np.full(bm_val.size, Action.PASS, dtype=object),
    )
    y_val, keep_val = day_labels(placeholder, truth)
    x_val_kept, bag_idx_val_kept, y_val_kept = _drop_censored(x_val, bag_idx_val, y_val, keep_val)
    return fit_tau(model, x_val_kept, bag_idx_val_kept, y_val_kept)


def score_seed(
    tuned: Any,
    seed: int,
    *,
    root: Path,
    config: Path,
    boundaries: SplitBoundaries,
    panel: dataset.Panel,
    capsules: pl.LazyFrame,
    val_merchants: list[str],
    tau_table: list[dict[str, Any]],
    train_merchants: list[str],
    booster_path: Path,
) -> dict[str, Any]:
    """Score a fitted :class:`TrainedMIL` on the validation bags of the subsample."""
    columns = feature_columns()
    x_val, bag_idx_val, bm_val, bd_val, exposure_val = build_bags(
        panel, "val", val_merchants, capsules, boundaries
    )
    truth = _build_truth(
        root, val_merchants, cutoff_day=boundaries.val[0] - 1, boundaries=boundaries
    )

    scored_at = _time.perf_counter()
    score_val = tuned.predict(x_val, columns, bag_index=bag_idx_val, n_bags=bm_val.size)
    # NFR-02 is a per-merchant-day budget, and a bag IS a merchant-day, so the divisor is
    # n_bags — not n_instances. Rung 5 spends the 10 ms on a whole bag of payers, which is
    # exactly the cost multiplier rung5_mil's docstring names.
    latency_ms = (_time.perf_counter() - scored_at) / max(bm_val.size, 1) * 1000.0
    model_size_mb = round(tuned.size_mb(booster_path), 4)
    # The register's measured state, from the panel run — 9,716 B against NFR-04's 4,096 B.
    # Carried onto this row for the same reason cli.py carries it: a rung is adopted on
    # PR-AUC *and* the compute NFRs, and this row must not read as if the budget were met.
    summary_path = dataset.DEFAULT_PANEL.parent / "features_summary.json"
    state_bytes = (
        float(json.loads(summary_path.read_text(encoding="utf-8"))["state_bytes_p99"])
        if summary_path.exists()
        else float("nan")
    )
    params = _cost_params(config)
    policy = _action_policy(config)
    k = _capacity(len(val_merchants), REPO_ROOT)
    action_val = select_actions(score_val, bd_val, exposure_val, k, params, policy)
    output = RungOutput(merchant_id=bm_val, day=bd_val, score=score_val, action=action_val)

    y_all, keep_all = day_labels(output, truth)
    n_holds = int((action_val == Action.HOLD).sum())
    n_non_pass = int((action_val != Action.PASS).sum())
    order = np.argsort(truth.merchant_id)
    idx = order[np.searchsorted(truth.merchant_id[order], output.merchant_id)]
    ceiling = oracle_savings(
        output.day[keep_all], y_all[keep_all], truth.loss_inr[idx][keep_all], k, params
    )

    lock_path = resolve_authoritative(REPO_ROOT)
    result = build_eval_result(
        rung=5,
        split="val",
        output=output,
        truth=truth,
        k=k,
        params=params,
        rng=np.random.default_rng(seed),
        perf=PerfBudget(
            p99_latency_ms=latency_ms,
            state_bytes_p99=state_bytes,
            model_size_mb=model_size_mb,
        ),
        oracle_savings=ceiling,
        eval_lock_sha=load_lock(REPO_ROOT, lock_path=lock_path)["eval_module_sha256"],
        open_count=read_open_count(REPO_ROOT, lock_path=lock_path),
        git_sha=str(load_lock(REPO_ROOT, lock_path=lock_path)["frozen_at_git_sha"]),
    )

    payload: dict[str, Any] = dataclasses.asdict(result)
    payload["recall_by_typology"] = {str(k_): v for k_, v in result.recall_by_typology.items()}
    payload |= {
        "label": "rung5_mil",
        "capacity_k": k,
        "n_merchants_scored": len(val_merchants),
        "n_rows_scored": int(bm_val.size),
        "n_rows_kept": int(keep_all.sum()),
        "n_censored_dropped": int((~keep_all).sum()),
        "n_labelled_merchants": int((~truth.is_censored).sum()),
        "n_censored_merchants": int(truth.is_censored.sum()),
        "oracle_savings": ceiling,
        "beats_all_floors": result.beats_all_floors,
        "n_features": len(columns),
        "split_boundaries": {
            "train": list(boundaries.train),
            "val": list(boundaries.val),
            "test": list(boundaries.test),
        },
        "n_hold_actions": n_holds,
        "n_non_pass_actions": n_non_pass,
        "subsample": {
            "n_train_merchants": len(train_merchants),
            "n_val_merchants": len(val_merchants),
            "note": "SUBSAMPLE result, not full population. Rungs 0-4 are scored on all "
            "3,036 validation-fold merchants; this row is a typology-stratified subsample "
            "that deliberately oversamples the positive class, so its prevalence, its "
            "capacity K and therefore every capacity-dependent metric (savings, "
            "precision@K, recall@K, alerts_per_day, gap_to_oracle) are NOT comparable "
            "row-for-row with a full-population rung. PR-AUC is prevalence-sensitive too. "
            "See docs/LOGBOOK.md T-0120 and this module's docstring for the cost probe "
            "that forced the subsample.",
        },
        "bar_on_the_same_bags": bar_on_the_same_bags(
            panel, val_merchants, seed, y_all, keep_all
        ),
        "nfr04_conditionality": (
            "T-0120 (rung5_mil.py docstring): payer_is_new and device_shared_payers need "
            "unbounded per-merchant/per-device state (the same two quantities T-122 cut "
            "from the T2 register). This rung is CONDITIONALLY adopted at best even if it "
            "clears the PR-AUC margin: it is not servable under NFR-04 as implemented."
        ),
        "tau_selection_table": tau_table,
    }
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration — the two entry points `cli.py` calls
# ─────────────────────────────────────────────────────────────────────────────


def prepare(
    root: Path = DATA_ROOT,
    config: Path = CONFIG,
    *,
    n_target: int = TARGET_N_MERCHANTS,
    echo: Any = print,
) -> tuple[dict[str, list[str]], pl.LazyFrame, SplitBoundaries, dataset.Panel]:
    """Choose the subsample, materialise its capsules, and open everything Rung 5 needs.

    Idempotent: :func:`materialise_capsules` skips chunks that already exist, so calling
    this from ``train`` and again from ``eval`` costs one directory listing the second
    time, not a second scan. The subsample manifest is written on the first call and
    **read back** on later ones, so ``train`` and ``eval`` cannot silently disagree about
    which merchants Rung 5 is about — a re-draw between the two stages would train on one
    population and score on another and nothing would raise.
    """
    boundaries = _boundaries(config, root=REPO_ROOT)
    panel = dataset.load_panel(dataset.DEFAULT_PANEL)

    SUBSAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SUBSAMPLE_DIR / "manifest.json"
    if manifest_path.exists():
        sample = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        sample = sample_merchants(root, dataset.DEFAULT_PANEL, n_target=n_target)
        manifest_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    train_merchants, val_merchants = sample["train"], sample["val"]
    echo(
        f"subsample: {len(train_merchants)} train merchants, {len(val_merchants)} val "
        f"merchants ({len(train_merchants) + len(val_merchants)} total)"
    )

    capsule_dir = SUBSAMPLE_DIR / "capsules"
    as_of = _epoch_end(boundaries.val[1], boundaries)
    all_merchants = sorted(set(train_merchants) | set(val_merchants))
    with EventStore(root) as store:
        materialise_capsules(store, all_merchants, as_of, capsule_dir, echo=echo)
    capsules = capsule_scan(capsule_dir)
    return sample, capsules, boundaries, panel


def main(root: Path = DATA_ROOT, config: Path = CONFIG) -> None:
    """Every declared seed, end to end. ``cli.py`` drives one seed at a time instead."""
    sample, capsules, boundaries, panel = prepare(root, config)
    n_rows = int(capsules.select(pl.len()).collect().item())
    print(f"materialised {n_rows:,} capsule rows")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        started = _time.perf_counter()
        tuned, tau_table = fit_seed(
            seed,
            root=root,
            boundaries=boundaries,
            panel=panel,
            capsules=capsules,
            train_merchants=sample["train"],
            val_merchants=sample["val"],
        )
        booster_path = tuned.save(SUBSAMPLE_DIR / f"rung5_seed{seed}.txt")
        payload = score_seed(
            tuned,
            seed,
            root=root,
            config=config,
            boundaries=boundaries,
            panel=panel,
            capsules=capsules,
            val_merchants=sample["val"],
            train_merchants=sample["train"],
            tau_table=tau_table,
            booster_path=booster_path,
        )
        (RESULT_DIR / f"rung5_mil_val_seed{seed}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"seed {seed}: pooling={tuned.pooling} tau={tuned.tau} "
            f"pr_auc={payload['pr_auc']:.4f} savings={payload['savings']:.4f} "
            f"floor_fail={payload['floor_fail']} beats_all_floors={payload['beats_all_floors']} "
            f"({_time.perf_counter() - started:.1f}s)"
        )


if __name__ == "__main__":
    main()
