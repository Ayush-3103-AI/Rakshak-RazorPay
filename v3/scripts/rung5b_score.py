"""T-0131 / GitHub #65: score Rung 5b (gated-attention MIL) against Rung 5's fitted-tau LSE.

Run: ``uv run python scripts/rung5b_score.py``

## What this script is careful about, and why

The adoption gate declared in advance by the 2026-09-02 AMENDMENT to
``docs/adr/ADR-V3-001-no-autograd.md`` is a **relative PR-AUC margin against one specific
baseline number**: Rung 5's tau = 5.0 LSE pooling, pooled over seeds 42-46 on validation.
A margin is only a margin if both sides are measured on the same bags, the same labels,
the same censoring mask and the same instance model. So this script does not read Rung 5's
stored PR-AUC and compare against it - it **recomputes** the baseline inside the same
process, from the same fitted instance model, and then asserts that the recomputed tau
grid is bit-identical to the ``tau_selection_table`` already committed in
``data/v2/eval/rung5_mil_val_seed{seed}.json``. If that assertion ever fails, the
comparison is not single-variable and the run stops rather than reporting a number.

Everything upstream of the pooling is imported from ``rakshak.score_rung5`` rather than
re-implemented: the same subsample manifest, the same materialised capsules, the same
``build_bags``, the same ``_build_truth`` / ``_training_labels`` / ``day_labels``
censoring. The only fork is the twenty lines of ``score_rung5.fit_seed`` that this script
has to repeat because that function returns the tuned model and discards the training bags
the attention gate needs. The tau-table assertion is what proves the repeat did not drift.

## What is deliberately NOT computed

No savings, no precision@K, no oracle gap, no test split. The gate is PR-AUC and latency,
and Rung 5's own artifact already records that its subsample's capacity-dependent metrics
are not comparable row-for-row with a full-population rung. Computing them here would add
numbers nobody is allowed to use.

``RAKSHAK_UNLOCK`` is never set and the test split is never touched. ``open_count`` stays 0.
"""

from __future__ import annotations

import json
import time as _time
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
import torch

from rakshak.cli import _build_truth, _epoch_end, _training_labels
from rakshak.eval.metrics import RungOutput, day_labels, pr_auc
from rakshak.features.capsules import CAPSULE_VECTOR_COLUMNS
from rakshak.models.rung2_lgbm import DEFAULT_PARAMS, TrainedRung
from rakshak.models.rung5_mil import (
    DEFAULT_TAU_GRID,
    TrainedMIL,
    feature_columns,
    fit_tau,
)
from rakshak.models.rung5_mil import train as train_mil
from rakshak.models.rung5b_attention import (
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    WEIGHT_DECAY,
    TrainedAttentionMIL,
    train_attention,
)
from rakshak.schemas import Action
from rakshak.score_rung5 import (
    DATA_ROOT,
    RESULT_DIR,
    SEEDS,
    SUBSAMPLE_DIR,
    _drop_censored,
    build_bags,
    prepare,
)

#: The ADR amendment's gate, in one place. Not adjustable after results are seen
#: (Prime Directive 5); read, never written, by everything below.
BASELINE_TAU: Final = 5.0
REQUIRED_RELATIVE_MARGIN: Final = 0.10
LATENCY_BUDGET_MS: Final = 10.0

#: The binding constraint ADR-V3-001 records and its amendment waives rather than
#: discharges. Carried onto every artifact beside the attention parameter count.
TRAINABLE_POSITIVE_MERCHANTS: Final = 234

#: NOT ``data/v2/eval``. ``artifacts/build.py::read_result_rows`` globs that directory
#: into ``artifacts/ladder.json`` and refuses anything that is not a valid ``EvalResult``,
#: so a Rung 5b diagnostic file there breaks ``tests/unit/test_artifacts_contract.py`` —
#: and if it did parse it would silently *become* a ladder row. Rung 5b emits no
#: ``EvalResult``: its gate is PR-AUC and latency, and the capacity-dependent metrics an
#: ``EvalResult`` carries are not comparable across this subsample anyway (see the module
#: docstring's "What is deliberately NOT computed"). ``score_rung7.EXPLANATION_DIR`` is
#: the same decision for the same reason.
OUT_DIR: Final = DATA_ROOT / "rung5b_attention"

#: Read-only, and only to recompute the baseline: the committed Rung 5 rows live here.
BASELINE_DIR: Final = RESULT_DIR

#: One CPU core, because that is what charter §2's latency term says.
LATENCY_THREADS: Final = 1

#: Per-bag latency is timed one bag at a time over this many bags, drawn with a fixed
#: seed. All 9,600 would be ~30x slower for a p99 that does not move.
LATENCY_SAMPLE: Final = 1500


#: The 2-pass instance LightGBM over ~2.3M capsule instances is the expensive half of a
#: seed and it is a pure function of (seed, train bags). Cached so a re-run of this script
#: costs the attention fit and nothing else. Deleting the directory is always safe.
CACHE_DIR: Final = SUBSAMPLE_DIR / "rung5b_cache"

_T0: float = _time.perf_counter()


def _say(message: str) -> None:
    """Flushed, wall-clocked. An unflushed print into a redirected stdout is invisible for
    tens of minutes, which is how the first attempt at this run became undiagnosable."""
    print(f"[{_time.perf_counter() - _T0:7.1f}s] {message}", flush=True)


def _stage(name: str, fn: Any) -> Any:
    started = _time.perf_counter()
    _say(f"  ... {name}")
    out = fn()
    _say(f"  done {name} ({_time.perf_counter() - started:.1f}s)")
    return out


def _cached_instance_model(
    seed: int,
    x: np.ndarray,
    bag_index: np.ndarray,
    bag_y: np.ndarray,
    columns: tuple[str, ...],
    merchant_id: np.ndarray,
) -> Any:
    """Rung 5's fitted instance model for this seed, from cache when it is on disk.

    The cache holds only the booster and the reporting counters that ``TrainedRung``
    carries; the pooling is NOT cached, because the whole point of this script is that the
    pooling is the variable. ``fit_tau`` is re-run against the committed grid on every
    invocation, cache or no cache, so a stale booster is caught rather than trusted.
    """
    import lightgbm as lgb

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    booster_path = CACHE_DIR / f"instance_seed{seed}.txt"
    sidecar_path = CACHE_DIR / f"instance_seed{seed}.json"
    if booster_path.exists() and sidecar_path.exists():
        side = json.loads(sidecar_path.read_text(encoding="utf-8"))
        instance = TrainedRung(
            rung=5,
            booster=lgb.Booster(model_file=str(booster_path)),
            columns=tuple(side["columns"]),
            params=DEFAULT_PARAMS.with_seed(seed),
            n_train_rows=side["n_train_rows"],
            n_train_positive_rows=side["n_train_positive_rows"],
            n_train_positive_merchants=side["n_train_positive_merchants"],
            train_seconds=side["train_seconds"],
        )
        return TrainedMIL(
            instance=instance,
            pooling=side["pooling"],
            tau=float(side["tau"]),
            n_train_bags=int(side["n_train_bags"]),
            n_train_positive_bags=int(side["n_train_positive_bags"]),
            passes=int(side["passes"]),
        )

    model = train_mil(
        x, bag_index, bag_y, columns, params=DEFAULT_PARAMS.with_seed(seed),
        merchant_id=merchant_id,
    )
    model.save(booster_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "columns": list(model.columns),
                "n_train_rows": model.instance.n_train_rows,
                "n_train_positive_rows": model.instance.n_train_positive_rows,
                "n_train_positive_merchants": model.instance.n_train_positive_merchants,
                "train_seconds": model.instance.train_seconds,
                "pooling": model.pooling,
                "tau": model.tau,
                "passes": model.passes,
                "n_train_bags": model.n_train_bags,
                "n_train_positive_bags": model.n_train_positive_bags,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


def _bags_once(
    *,
    root: Path,
    boundaries: Any,
    panel: Any,
    capsules: pl.LazyFrame,
    train_merchants: list[str],
    val_merchants: list[str],
) -> dict[str, Any]:
    """The train and validation bags, built ONCE for all five seeds.

    ``score_rung5.fit_seed`` rebuilds these inside its per-seed call, which is correct
    there because ``cli.py`` runs one seed per process. Here all five seeds run in one
    process, and **the bags do not depend on the seed** - only LightGBM's ``seed`` does
    (``score_rung5.SAMPLE_SEED``'s own comment says the subsample is chosen once,
    independent of the training seed, so all five runs score the same bags). Rebuilding
    them per seed cost ~20 minutes a side for a bit-identical array. Hoisting is a pure
    speedup: the arrays handed to each seed are the same objects the per-seed version
    would have rebuilt, and the ``fit_tau`` assertion against the committed grid proves it
    on every seed.
    """
    columns = feature_columns()
    x_tr, bag_idx_tr, bm_tr, _bd_tr, _exp_tr = _stage(
        "build TRAIN bags (once, all seeds)",
        lambda: build_bags(panel, "train", train_merchants, capsules, boundaries),
    )
    train_as_of = _epoch_end(boundaries.train[1], boundaries)
    bag_y_tr = _training_labels(root, bm_tr, train_as_of)

    x_val, bag_idx_val, bm_val, bd_val, _exp_val = _stage(
        "build VAL bags (once, all seeds)",
        lambda: build_bags(panel, "val", val_merchants, capsules, boundaries),
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
    x_val_kept, bag_idx_val_kept, y_val_kept = _drop_censored(
        x_val, bag_idx_val, y_val, keep_val
    )
    _say(
        f"  bags: train {bm_tr.size} bags / {x_tr.shape[0]} instances, "
        f"val {int(keep_val.sum())} kept bags / {x_val_kept.shape[0]} instances, "
        f"{int((y_val_kept == 1).sum())} positive val bags"
    )
    return {
        "columns": columns,
        "x_tr": x_tr,
        "bag_idx_tr": bag_idx_tr,
        "bag_y_tr": bag_y_tr,
        "merchant_tr": bm_tr[bag_idx_tr],
        "n_train_bags": int(bm_tr.size),
        "x_val": x_val_kept,
        "bag_idx_val": bag_idx_val_kept,
        "bag_y_val": y_val_kept,
        "bag_merchant_val": bm_val[keep_val],
        "bag_day_val": bd_val[keep_val],
    }


def _fit_instance(seed: int, bags: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Rung 5's instance model for one seed plus its validation pooling grid.

    Same operations, same order as ``score_rung5.fit_seed``; the tau table is asserted
    against the committed one by the caller, which is what proves the split did not drift.
    """
    model = _stage(
        "instance model (2-pass LightGBM, cached)",
        lambda: _cached_instance_model(
            seed,
            bags["x_tr"],
            bags["bag_idx_tr"],
            bags["bag_y_tr"],
            bags["columns"],
            bags["merchant_tr"],
        ),
    )
    _tuned, tau_table = _stage(
        "fit_tau grid on validation bags",
        lambda: fit_tau(model, bags["x_val"], bags["bag_idx_val"], bags["bag_y_val"]),
    )
    return model, tau_table


def _assert_baseline_is_the_committed_one(seed: int, tau_table: list[dict[str, Any]]) -> float:
    """Return the tau = 5.0 PR-AUC, having proved it is the number Rung 5 already reported.

    The whole adoption gate is a ratio against this figure. If this run's grid differs from
    the committed one by so much as a float, the two sides of the ratio were not built the
    same way and the margin means nothing - so this raises rather than warns.
    """
    committed_path = BASELINE_DIR / f"rung5_mil_val_seed{seed}.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))["tau_selection_table"]
    here = {(r["pooling"], r["tau"]): r["pr_auc"] for r in tau_table}
    for row in committed:
        key = (row["pooling"], row["tau"])
        if key not in here or not np.isclose(here[key], row["pr_auc"], rtol=0, atol=1e-12):
            raise AssertionError(
                f"seed {seed}: recomputed pooling grid does not reproduce {committed_path}. "
                f"{key}: committed {row['pr_auc']!r}, here {here.get(key)!r}. The Rung 5b "
                "comparison is only single-variable if the baseline is rebuilt identically; "
                "refusing to report a margin against a baseline that moved."
            )
    if float(committed[0]["tau"]) != 0.0 or BASELINE_TAU not in DEFAULT_TAU_GRID:
        raise AssertionError("the committed grid is not the grid this baseline was fitted on")
    return float(here[("lse", BASELINE_TAU)])


def _latency(
    tuned: TrainedAttentionMIL,
    x: np.ndarray,
    bag_index: np.ndarray,
    columns: tuple[str, ...],
    n_bags: int,
) -> dict[str, float]:
    """Two readings, both reported, because they answer different questions.

    - ``amortised_ms_per_bag`` is the whole validation split scored in one call divided by
      the bag count. It is what ``score_rung5.score_seed`` writes into Rung 5's
      ``p99_latency_ms`` field, so it is the only figure directly comparable to the 0.10-0.18
      ms already on the ladder for Rung 5. It is a mean, not a p99, despite the field name.
    - ``p99_ms_per_bag`` scores one bag at a time and takes the 99th percentile. That is
      what charter §2 actually asks for - a per-merchant budget - and it includes the
      per-call overhead that the amortised figure hides. It is the number the gate is read
      against.
    """
    torch.set_num_threads(LATENCY_THREADS)
    started = _time.perf_counter()
    tuned.predict(x, columns, bag_index=bag_index, n_bags=n_bags)
    amortised = (_time.perf_counter() - started) / max(n_bags, 1) * 1000.0

    order = np.argsort(bag_index, kind="stable")
    x_sorted, bag_sorted = x[order], bag_index[order]
    starts = np.searchsorted(bag_sorted, np.arange(n_bags), side="left")
    ends = np.searchsorted(bag_sorted, np.arange(n_bags), side="right")
    rng = np.random.default_rng(0)
    sample = rng.choice(n_bags, size=min(LATENCY_SAMPLE, n_bags), replace=False)
    per_bag: list[float] = []
    for b in sample:
        rows = x_sorted[starts[b] : ends[b]]
        one_bag = np.zeros(rows.shape[0], dtype=np.intp)
        t0 = _time.perf_counter()
        if rows.shape[0]:
            tuned.predict(rows, columns, bag_index=one_bag, n_bags=1)
        per_bag.append((_time.perf_counter() - t0) * 1000.0)
    return {
        "amortised_ms_per_bag": amortised,
        "p99_ms_per_bag": float(np.percentile(per_bag, 99)),
        "median_ms_per_bag": float(np.median(per_bag)),
        "n_bags_timed": len(per_bag),
    }


def _replay(
    tuned: TrainedAttentionMIL,
    capsules: pl.LazyFrame,
    boundaries: Any,
    merchant: str,
    day: int,
    columns: tuple[str, ...],
    top: int = 10,
) -> dict[str, Any]:
    """Per-capsule attention weights for one replayed merchant-day.

    GitHub #65's third acceptance criterion, and the stated payoff for the added
    complexity. It is produced whether or not the adoption gate is met, because "we cannot
    say which payer" is exactly the thing fixed pooling could not do.

    The capsule rows are re-queried here with ``payer_id`` retained (``build_bags`` drops
    it - it is a key, not a feature) and the feature matrix is asserted equal to the one
    the score was computed from, so the payer labels cannot silently belong to other rows.
    """
    date = boundaries.origin + timedelta(days=int(day))
    cap = (
        capsules.select("merchant_id", "event_date", "payer_id", *CAPSULE_VECTOR_COLUMNS)
        .filter((pl.col("merchant_id") == merchant) & (pl.col("event_date") == date))
        .collect()
    )
    x = cap.select(CAPSULE_VECTOR_COLUMNS).to_numpy().astype(np.float64)
    n = x.shape[0]
    bag = np.zeros(n, dtype=np.intp)
    weights, probs = tuned.attention(x, columns, bag, 1)
    score = float(tuned.predict(x, columns, bag_index=bag, n_bags=1)[0])
    order = np.argsort(-weights)[:top]
    return {
        "merchant_id": merchant,
        "day": int(day),
        "event_date": str(date),
        "n_capsules_in_bag": n,
        "bag_score": score,
        "uniform_weight_would_be": 1.0 / max(n, 1),
        "attention_entropy_nats": float(-(weights * np.log(weights + 1e-300)).sum()),
        "max_entropy_nats": float(np.log(max(n, 1))),
        "top_capsules": [
            {
                "payer_id": cap["payer_id"][int(i)],
                "attention_weight": float(weights[i]),
                "instance_probability": float(probs[i]),
                "share_of_bag_score": float(weights[i] * probs[i] / score) if score else None,
            }
            for i in order
        ],
    }


def _pick_replay_merchant(
    bag_merchant: np.ndarray, bag_day: np.ndarray, bag_y: np.ndarray, bag_index: np.ndarray
) -> tuple[str, int]:
    """The positive validation bag with the most capsules - the one where "which payer"
    is a question worth asking. Deterministic: ties break on (merchant_id, day)."""
    sizes = np.bincount(bag_index, minlength=bag_y.size)
    positive = np.flatnonzero((bag_y == 1) & (sizes > 1))
    if positive.size == 0:
        raise ValueError("no non-singleton positive validation bag to replay")
    best = positive[np.lexsort((bag_day[positive], bag_merchant[positive], -sizes[positive]))[0]]
    return str(bag_merchant[best]), int(bag_day[best])


def main(root: Path = DATA_ROOT) -> None:
    sample, capsules, boundaries, panel = prepare(root, echo=_say)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fit = _bags_once(
        root=root,
        boundaries=boundaries,
        panel=panel,
        capsules=capsules,
        train_merchants=sample["train"],
        val_merchants=sample["val"],
    )

    rows: list[dict[str, Any]] = []
    replay: dict[str, Any] | None = None
    for seed in SEEDS:
        started = _time.perf_counter()
        _say(f"=== seed {seed} ===")
        model, tau_table = _fit_instance(seed, fit)
        baseline = _assert_baseline_is_the_committed_one(seed, tau_table)
        _say(f"  baseline tau={BASELINE_TAU} pr_auc={baseline:.6f} (reproduces committed)")

        tuned = train_attention(
            model,
            fit["x_tr"],
            fit["bag_idx_tr"],
            fit["bag_y_tr"],
            fit["columns"],
            seed=seed,
            echo=_say,
        )
        y_val = fit["bag_y_val"]
        n_val_bags = int(y_val.size)
        score = tuned.predict(
            fit["x_val"], fit["columns"], bag_index=fit["bag_idx_val"], n_bags=n_val_bags
        )
        attention_pr = pr_auc(y_val, score)
        latency = _latency(
            tuned, fit["x_val"], fit["bag_idx_val"], fit["columns"], n_val_bags
        )
        margin = (attention_pr - baseline) / baseline

        summary = tuned.summary()
        row = {
            "rung": "5b",
            "seed": seed,
            "split": "val",
            "label": "rung5b_gated_attention",
            "pr_auc": attention_pr,
            "baseline_rung5_lse_tau5_pr_auc": baseline,
            "relative_margin": margin,
            "required_relative_margin": REQUIRED_RELATIVE_MARGIN,
            "n_val_bags": n_val_bags,
            "n_val_positive_bags": int((y_val == 1).sum()),
            "n_attention_parameters": summary["n_attention_parameters"],
            "trainable_positive_merchants": TRAINABLE_POSITIVE_MERCHANTS,
            "parameters_per_positive_merchant": (
                summary["n_attention_parameters"] / TRAINABLE_POSITIVE_MERCHANTS
            ),
            "n_train_positive_bags": summary["n_train_positive_bags"],
            "n_train_bags": summary["n_train_bags"],
            "n_train_instances": summary["n_train_instances"],
            "final_train_loss": summary["final_train_loss"],
            "hyperparameters": {
                "hidden_dim": HIDDEN_DIM,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "declared": "fixed before the first run; not re-tuned after seeing a result",
            },
            "latency": latency,
            "latency_budget_ms": LATENCY_BUDGET_MS,
            "tau_selection_table": tau_table,
            "open_count": 0,
            "note": (
                "SUBSAMPLE result on the same 800-merchant stratified subsample and the "
                "same bags as Rung 5 (data/v2/eval/rung5_mil_val_seed*.json). Comparable "
                "to that row and to nothing else on the ladder; PR-AUC is prevalence-"
                "sensitive and this subsample oversamples the positive class by design."
            ),
        }
        rows.append(row)
        (OUT_DIR / f"rung5b_attention_val_seed{seed}.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"  attention pr_auc={attention_pr:.6f}  margin={margin:+.4%}  "
            f"p99={latency['p99_ms_per_bag']:.3f} ms  "
            f"({_time.perf_counter() - started:.1f}s)",
            flush=True,
        )

        if replay is None:
            merchant, day = _pick_replay_merchant(
                fit["bag_merchant_val"], fit["bag_day_val"], y_val, fit["bag_idx_val"]
            )
            replay = _replay(tuned, capsules, boundaries, merchant, day, fit["columns"])
            replay["seed"] = seed

    attention = np.array([r["pr_auc"] for r in rows])
    base = np.array([r["baseline_rung5_lse_tau5_pr_auc"] for r in rows])
    pooled_margin = float((attention.mean() - base.mean()) / base.mean())
    # "Not adopted if the pooled margin is inside the per-seed spread" (ADR amendment).
    # The spread is the full range of the per-seed relative margins.
    per_seed = np.array([r["relative_margin"] for r in rows])
    spread = float(per_seed.max() - per_seed.min())
    inside_spread = bool(abs(pooled_margin) <= spread)
    p99 = float(max(r["latency"]["p99_ms_per_bag"] for r in rows))
    adopted = bool(
        pooled_margin >= REQUIRED_RELATIVE_MARGIN
        and not inside_spread
        and p99 <= LATENCY_BUDGET_MS
    )

    verdict = {
        "rung": "5b",
        "gate": (
            "ADR-V3-001 AMENDMENT 2026-09-02: adopted only if Rung 5b beats Rung 5's "
            "fitted-tau=5.0 LSE pooling by >= 10% relative PR-AUC on validation, pooled "
            "over seeds 42-46, with the pooled margin OUTSIDE the per-seed spread, and "
            "p99 <= 10 ms per merchant on one CPU core. Not adopted on a tie."
        ),
        "seeds": list(SEEDS),
        "pooled_rung5b_pr_auc": float(attention.mean()),
        "pooled_rung5_lse_tau5_pr_auc": float(base.mean()),
        "pooled_relative_margin": pooled_margin,
        "per_seed_relative_margin": {str(r["seed"]): r["relative_margin"] for r in rows},
        "per_seed_relative_margin_spread": spread,
        "pooled_margin_inside_per_seed_spread": inside_spread,
        "per_seed_rung5b_pr_auc": {str(r["seed"]): r["pr_auc"] for r in rows},
        "per_seed_rung5_pr_auc": {
            str(r["seed"]): r["baseline_rung5_lse_tau5_pr_auc"] for r in rows
        },
        "rung5b_pr_auc_seed_spread": float(attention.max() - attention.min()),
        "worst_seed_p99_ms_per_bag": p99,
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "latency_holds": bool(p99 <= LATENCY_BUDGET_MS),
        "n_attention_parameters": rows[0]["n_attention_parameters"],
        "trainable_positive_merchants": TRAINABLE_POSITIVE_MERCHANTS,
        "adopted": adopted,
        "replay": replay,
        "open_count": 0,
    }
    (OUT_DIR / "rung5b_attention_verdict.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(f"pooled Rung 5b PR-AUC      {verdict['pooled_rung5b_pr_auc']:.6f}")
    print(f"pooled Rung 5 (tau=5.0)    {verdict['pooled_rung5_lse_tau5_pr_auc']:.6f}")
    print(f"pooled relative margin     {pooled_margin:+.4%}  (gate: >= +10.00%)")
    print(f"per-seed margin spread     {spread:.4%}  inside_spread={inside_spread}")
    print(f"worst-seed p99             {p99:.3f} ms/bag  (budget {LATENCY_BUDGET_MS} ms)")
    print(f"attention parameters       {verdict['n_attention_parameters']} against "
          f"{TRAINABLE_POSITIVE_MERCHANTS} trainable positive merchants")
    print(f"VERDICT                    {'ADOPTED' if adopted else 'NOT ADOPTED'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
