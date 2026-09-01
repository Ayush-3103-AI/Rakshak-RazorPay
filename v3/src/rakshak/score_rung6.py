"""Rung 6 scoring entry point (T-0121 scoring lane, GitHub #55) — sibling to ``cli.py``.

Scores :class:`~rakshak.models.rung6_conformal.ConformalHold` wrapping a base rung's
decisions on the validation split, at one or more ``alpha``, and writes one
``EvalResult``-shaped row per alpha to ``data/v2/eval/`` — the same shape
``cli.py::score_split`` produces, so the ladder can compare rungs on identical axes.

**Why this file and not the obvious home.** Not
``src/rakshak/models/rung6_conformal.py``: Prime Directive 3's AST gate covers
``src/rakshak/models/``, and this script must open ``ground_truth.parquet`` (via
``cli.py``'s own ``_build_truth``) to assemble ``y`` for ``false_hold_coverage`` — exactly
the pattern ``cli.py::score_split`` uses, reused here rather than re-derived. Re-deriving
it would duplicate ~80 lines of censoring and loss-accrual logic whose whole point (per
``cli.py``'s own docstring) is to live in exactly one place.

**Which base rung, and why.** ``--base-rung 4`` is the flag's default and **Rung 2 is what
was actually run**, because Rung 4 has no trained artifact on the v3 panel: LIMITATIONS.md
§8.5 cut it on the Lane D measurement ("loses on every seed"), so nothing retrained it here
and there is no ``rung4_seed42.txt`` to wrap. Rung 2 is the incumbent and the declared bar
(CLAUDE.md §Architecture), it is trained at all five locked seeds on this panel, and Rung 6
is a *wrapper*: what it bounds is the false-HOLD rate of whatever it wraps, so the base rung
changes the numbers but not the claim. The results rows carry ``base_rung`` so which one was
wrapped is never in doubt.

**The calibration/evaluation split.** Rung 6 calibrates on validation only
(``rung6_conformal.calibrate`` refuses anything else), and there is no third split
available to evaluate on afterwards — days 300-364 stay shut (Prime Directive 1; the test
split opens exactly once, in T-0116). So the *validation* merchants are carved in two, by merchant
(:func:`~rakshak.models.rung6_conformal.split_validation`, for the same reason its own
docstring gives: a day-split would leak drift across the fold, and a merchant's own days
are too dependent to split across it). Half calibrates the per-stratum thresholds; the
other half is scored — both the standard metric set and ``false_hold_coverage``. Scoring on
the merchants that calibrated the threshold would not be a guarantee, it would be a
tautology. The cost, stated rather than hidden: Rung 6's numbers below are measured on
half of validation's merchants, not all of it, so they are smaller-sample than the Rung 4
row ``cli.py`` produces on the full split. ``n_eval_merchants`` is in every payload so a
reader can weigh that.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import typer

from rakshak.cli import (
    MODEL_DIR,
    RESULT_DIR,
    ROOT,
    _action_policy,
    _boundaries,
    _build_truth,
    _capacity,
    _columns_for,
    _cost_params,
    _load_trained,
    _lock_path,
)
from rakshak.eval.capacity import DEFAULT_DECISION, ActionPolicy, DecisionRequest
from rakshak.eval.lock import (
    load_lock,
    read_open_count,
    require_unlocked_or_refuse,
    verify_lock,
)
from rakshak.eval.metrics import (
    CostParams,
    PerfBudget,
    RungOutput,
    Truth,
    build_eval_result,
    day_labels,
    false_hold_coverage,
)
from rakshak.eval.oracle import oracle_savings
from rakshak.eval.splits import SplitBoundaries
from rakshak.features.cohort import assign_cohorts
from rakshak.models import dataset
from rakshak.models.dataset import Panel
from rakshak.models.rung5_mil import TrainedMIL
from rakshak.models.rung6_conformal import ConformalHold, calibrate, split_validation, strata_of
from rakshak.schemas import Action, MerchantProfile

app = typer.Typer(add_completion=False, help="Rung 6 — Mondrian conformal scoring lane.")

DEFAULT_ALPHAS: tuple[float, ...] = (0.05, 0.10)

#: `cli.py::_load_trained` now rebuilds Rung 5's fitted pooling too (T-0120), so the
#: obstacle to wrapping Rung 5 is no longer the loader — it is the SHAPE. Rung 5 does not
#: score the merchant-day panel: its rows are (merchant, day, payer) capsules and its
#: output is one pooled score per bag, produced by `score_rung5.build_bags` over a
#: merchant subsample that is not this module's validation population. Wrapping it would
#: mean aligning two different row universes, which is a ticket, not a branch. Refused
#: loudly rather than approximated.
_TRAINED_BOOSTER_RUNGS = (2, 4)


def _base_score(base_rung: int, seed: int, rows: Panel) -> tuple[np.ndarray, float, float]:
    """The wrapped rung's probability score over ``rows``, plus perf facts.

    Mirrors ``cli.py::score_split``'s rung>=2 branch exactly: predict, time it, report the
    booster's on-disk size. Raises ``FileNotFoundError`` (from ``_load_trained``) if the
    base rung has not been trained yet — the caller is expected to check for that rather
    than let it surface as a stack trace.
    """
    if base_rung not in _TRAINED_BOOSTER_RUNGS:
        raise NotImplementedError(
            f"_base_score handles the merchant-day-panel rungs {_TRAINED_BOOSTER_RUNGS}; "
            f"got {base_rung!r}. Rung 5 scores bags of capsules, not panel rows — see the "
            "note on _TRAINED_BOOSTER_RUNGS above."
        )
    model = _load_trained(base_rung, seed)
    assert not isinstance(model, TrainedMIL)  # noqa: S101 - narrowed by the guard above
    started = _time.perf_counter()
    score = model.predict(rows.x, rows.columns)
    latency_ms = (_time.perf_counter() - started) / rows.x.shape[0] * 1000.0
    model_size_mb = round(model.size_mb(MODEL_DIR / f"rung{base_rung}_seed{seed}.txt"), 4)
    return score, latency_ms, model_size_mb


def _state_bytes_p99(panel: Path) -> float:
    summary_path = panel.parent / "features_summary.json"
    if not summary_path.exists():
        return float("nan")
    return float(json.loads(summary_path.read_text(encoding="utf-8"))["state_bytes_p99"])


# ─────────────────────────────────────────────────────────────────────────────
# The core. Everything above this point is data acquisition (real model, real panel,
# real ground truth); everything in `run` is arithmetic over arrays it is handed. That
# split is what lets this function be exercised against a synthetic stand-in score vector
# before any base-rung model exists — see tests/unit/test_score_rung6.py.
# ─────────────────────────────────────────────────────────────────────────────


def run(
    *,
    score: np.ndarray,
    rows: Panel,
    truth_full: Truth,
    profiles: dict[str, MerchantProfile],
    params: CostParams,
    policy: ActionPolicy,
    alphas: tuple[float, ...],
    cal_fraction: float,
    cal_rng: np.random.Generator,
    metric_seed: int,
    perf: PerfBudget,
    base_rung: int,
    repo_root: Path,
    boundaries: SplitBoundaries,
    eval_lock_sha: str,
    open_count: int,
    git_sha: str,
) -> dict[float, dict[str, Any]]:
    """One ``EvalResult``-shaped payload per alpha, wrapping ``score`` with conformal HOLD.

    ``rows`` must already be the validation split (``Panel.select("val")``) and ``score``
    must be row-aligned with it. ``truth_full`` must cover every merchant in ``rows``; the
    eval-fold ``Truth`` handed to ``build_eval_result`` (so ``EvalResult.prevalence`` is
    scoped to the merchants actually scored, not all of validation) is sliced out of it in
    memory rather than re-read from disk — a second ``ground_truth.parquet`` scan for data
    already in hand would just be I/O, and slicing is what keeps this function free of any
    disk access at all, which is what lets a test drive it with a synthetic ``Truth``.
    """
    cohorts = assign_cohorts(profiles)
    stratum = strata_of(rows.merchant_id, cohorts)
    exposure = rows.column("p_declared_monthly_gmv")

    # A throwaway RungOutput to run day_labels once; `action` is unused by day_labels but
    # RungOutput's own __post_init__ requires a same-length, valid Action array.
    placeholder = np.full(rows.x.shape[0], Action.PASS, dtype=object)
    dummy = RungOutput(merchant_id=rows.merchant_id, day=rows.day, score=score, action=placeholder)
    y_full, keep_full = day_labels(dummy, truth_full)

    cal_mask = split_validation(rows.merchant_id, cal_rng, fraction=cal_fraction)
    eval_mask = ~cal_mask
    cal_merchants = sorted(set(rows.merchant_id[cal_mask].tolist()))
    eval_merchants = sorted(set(rows.merchant_id[eval_mask].tolist()))
    truth_mask = np.isin(truth_full.merchant_id, np.array(eval_merchants, dtype=object))
    truth_eval = Truth(
        merchant_id=truth_full.merchant_id[truth_mask],
        label=truth_full.label[truth_mask],
        is_censored=truth_full.is_censored[truth_mask],
        loss_inr=truth_full.loss_inr[truth_mask],
        onset_day=truth_full.onset_day[truth_mask],
        typology=truth_full.typology[truth_mask],
        volume=truth_full.volume[truth_mask],
    )
    # NOTE: repo root, not the dataset root -- the lock that carries the capacity ratio
    # lives beside the source tree, while `root` below is data/v2. Conflating the two is
    # exactly the FileNotFoundError this line raised the first time it ran.
    k_eval = _capacity(len(eval_merchants), root=repo_root)

    # The unwrapped base-rung decision on the same rows, computed once: it is what the
    # wrapper is softening, and its HOLD count is the number that says whether a coverage
    # table of zeros means "conformal held the rate down" or "the base rung never HOLDs at
    # all, so there was nothing to bound". Those two read identically in the coverage rows
    # and are completely different findings, so both counts are reported.
    base_request = DecisionRequest(
        score=score[eval_mask],
        day=rows.day[eval_mask],
        exposure_inr=exposure[eval_mask],
        k=k_eval,
        params=params,
        hold_policy=policy,
    )
    base_action = np.asarray(DEFAULT_DECISION.decide(base_request))
    n_hold_base = int((base_action == Action.HOLD).sum())

    order = np.argsort(truth_eval.merchant_id)
    eval_merchant_arr = rows.merchant_id[eval_mask]
    idx = order[np.searchsorted(truth_eval.merchant_id[order], eval_merchant_arr)]
    y_eval = y_full[eval_mask]
    keep_eval = keep_full[eval_mask]
    ceiling = oracle_savings(
        rows.day[eval_mask][keep_eval],
        y_eval[keep_eval],
        truth_eval.loss_inr[idx][keep_eval],
        k_eval,
        params,
    )

    # The SAME metric row for the unwrapped base rung on the SAME fold. Without it the only
    # available comparison is against the base rung's own published row, which is scored on
    # all of validation at twice this capacity K -- so any difference reads as the wrapper's
    # doing when most of it is the fold. This makes the wrapper's actual delta visible.
    base_result = build_eval_result(
        rung=base_rung,
        split="val",
        output=RungOutput(
            merchant_id=eval_merchant_arr,
            day=rows.day[eval_mask],
            score=score[eval_mask],
            action=base_action,
        ),
        truth=truth_eval,
        k=k_eval,
        params=params,
        # A FRESH generator per row, from the same seed, so the random_at_k floor is
        # identical for the base row and for every alpha. Threading one advancing
        # Generator through them made that floor differ between alphas on nothing but
        # rng position -- noise presented as a difference.
        rng=np.random.default_rng(metric_seed),
        perf=perf,
        oracle_savings=ceiling,
        eval_lock_sha=eval_lock_sha,
        open_count=open_count,
        git_sha=git_sha,
    )
    base_same_fold = {
        "savings": base_result.savings,
        "pr_auc": base_result.pr_auc,
        "precision_at_k": base_result.precision_at_k,
        "recall_at_k": base_result.recall_at_k,
        "alerts_per_day": base_result.alerts_per_day,
        "ttd_median_days": base_result.ttd_median_days,
        "floor_fail": base_result.floor_fail,
        "n_hold": n_hold_base,
    }

    results: dict[float, dict[str, Any]] = {}
    for alpha in alphas:
        cal = calibrate(score[cal_mask], y_full[cal_mask], stratum[cal_mask], alpha, split="val")

        wrapper = ConformalHold(DEFAULT_DECISION, cal, stratum[eval_mask])
        action = wrapper.decide(base_request)

        output = RungOutput(
            merchant_id=rows.merchant_id[eval_mask],
            day=rows.day[eval_mask],
            score=score[eval_mask],
            action=action,
        )
        result = build_eval_result(
            rung=6,
            split="val",
            output=output,
            truth=truth_eval,
            k=k_eval,
            params=params,
            rng=np.random.default_rng(metric_seed),
            perf=perf,
            oracle_savings=ceiling,
            eval_lock_sha=eval_lock_sha,
            open_count=open_count,
            git_sha=git_sha,
        )

        coverage = false_hold_coverage(
            action[keep_eval], y_eval[keep_eval], stratum[eval_mask][keep_eval], alpha
        )
        coverage_rows = [
            {
                **dataclasses.asdict(row),
                "bound": cal.bound(row.stratum),
                # The order statistic itself. With this in the file a reader can see
                # directly whether alpha moved the gate (it does) and whether the base
                # rung's HOLDs were anywhere near it (they are not).
                "threshold": cal.threshold.get(row.stratum, math.inf),
                "n_calibration": cal.n_calibration.get(row.stratum, 0),
                # Headroom under nominal alpha, not just the boolean — a violated=True row
                # at a tiny negative margin next to n_calibration in the thousands is the
                # metric working as designed (see rung6_conformal.py's own module docstring
                # and T-0121's logbook entry: a valid predictor trips `violated` about half
                # the time near the boundary, by construction).
                "margin": alpha - row.realised,
            }
            for row in coverage
        ]

        payload: dict[str, Any] = dataclasses.asdict(result)
        payload["recall_by_typology"] = {
            str(key): value for key, value in result.recall_by_typology.items()
        }
        payload |= {
            "label": f"rung6_crc_base{base_rung}_alpha{alpha:g}",
            "wrapper_name": wrapper.name,
            "base_rung": base_rung,
            "alpha": alpha,
            "cal_fraction": cal_fraction,
            "n_cal_merchants": len(cal_merchants),
            "n_eval_merchants": len(eval_merchants),
            "capacity_k": k_eval,
            "n_merchants_scored": len(eval_merchants),
            "n_rows_scored": int(output.merchant_id.size),
            "n_rows_kept": int(keep_eval.sum()),
            "n_censored_dropped": int((~keep_eval).sum()),
            "n_labelled_merchants": int((~truth_eval.is_censored).sum()),
            "n_censored_merchants": int(truth_eval.is_censored.sum()),
            "oracle_savings": ceiling,
            "beats_all_floors": result.beats_all_floors,
            "n_features": len(_columns_for(base_rung)),
            "n_hold_base_rung": n_hold_base,
            "n_hold_after_conformal": int((np.asarray(action) == Action.HOLD).sum()),
            "n_softened_hold_to_review": int(
                ((base_action == Action.HOLD) & (np.asarray(action) == Action.REVIEW)).sum()
            ),
            "n_review": int((np.asarray(action) == Action.REVIEW).sum()),
            "split_boundaries": {
                "train": list(boundaries.train),
                "val": list(boundaries.val),
                "test": list(boundaries.test),
            },
            # Named for the metric it is, not "coverage": this is exactly what
            # rakshak.eval.metrics.false_hold_coverage returned, per Mondrian stratum,
            # with bound/threshold/n_calibration/margin added alongside each row.
            "false_hold_coverage": coverage_rows,
            "n_strata": len(coverage_rows),
            "n_strata_infinite_threshold": sum(
                1 for r in coverage_rows if math.isinf(float(r["threshold"]))
            ),
            "base_rung_same_fold": base_same_fold,
        }
        results[alpha] = payload
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — real data only. `run()` above is what a test drives with a synthetic
# score vector; this function is the thing that opens the real panel and the real models.
# ─────────────────────────────────────────────────────────────────────────────


def score_val(
    *,
    base_rung: int = 2,
    seeds: tuple[int, ...] = (42,),
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    cal_fraction: float = 0.5,
    cal_seed: int = 0,
    root: Path = Path("data/v2"),
    panel: Path = dataset.DEFAULT_PANEL,
    config: Path = Path("configs/scenario_v2.yaml"),
) -> list[tuple[Path, dict[str, Any]]]:
    """Score ConformalHold-wrapped Rung ``base_rung`` on validation, at each alpha, and
    write one ``EvalResult``-shaped row per alpha to ``data/v2/eval/``.

    Split is hard-coded to validation — Rung 6 calibrates on validation by contract and
    there is no third split available under the lock. ``require_unlocked_or_refuse`` is
    still called, on the literal string ``"val"``, so a future edit that adds a ``split``
    argument here cannot bypass the guard by accident.

    A plain function with plain defaults, because ``cli.py``'s ``eval --rung 6`` calls it
    directly: a typer command's defaults are ``OptionInfo`` sentinels, so calling one as a
    function passes those sentinels where numbers are expected and fails somewhere far away
    from the cause. The ``score`` command below is the thin string-parsing shell over it.
    """
    alpha_values = tuple(alphas)
    if not alpha_values:
        raise typer.BadParameter("--alphas must name at least one value")

    require_unlocked_or_refuse("val")
    for d in verify_lock(ROOT, lock_path=_lock_path(ROOT)):
        typer.echo(
            f"[lock drift, unenforced] {d.key}: recorded {d.expected[:12]}… now {d.actual[:12]}…"
        )

    params = _cost_params(config)
    policy = _action_policy(config)
    full = dataset.load_panel(panel)
    rows = full.select("val")
    if rows.x.shape[0] == 0:
        raise typer.BadParameter("the panel holds no val rows")

    boundaries = _boundaries(config)
    val_merchants = sorted(set(rows.merchant_id.tolist()))
    truth_full = _build_truth(
        root, val_merchants, cutoff_day=boundaries.val[0] - 1, boundaries=boundaries
    )
    profiles = dataset.load_profiles(root)

    lock = load_lock(ROOT, lock_path=_lock_path(ROOT))
    results_dir = panel.parent / RESULT_DIR.name
    results_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[Path, dict[str, Any]]] = []
    # `cal_seed` is deliberately NOT threaded off `--seed`: every seed is calibrated on the
    # same by-merchant carve, so seed-to-seed movement in these rows is the base rung's
    # variance and nothing else.
    for one_seed in seeds:
        score_arr, latency_ms, model_size_mb = _base_score(base_rung, one_seed, rows)
        perf = PerfBudget(
            p99_latency_ms=latency_ms,
            state_bytes_p99=_state_bytes_p99(panel),
            model_size_mb=model_size_mb,
        )
        results = run(
            score=score_arr,
            rows=rows,
            truth_full=truth_full,
            profiles=profiles,
            params=params,
            policy=policy,
            alphas=alpha_values,
            cal_fraction=cal_fraction,
            cal_rng=np.random.default_rng(cal_seed),
            metric_seed=one_seed,
            perf=perf,
            base_rung=base_rung,
            repo_root=ROOT,
            boundaries=boundaries,
            eval_lock_sha=str(lock["eval_module_sha256"]),
            open_count=read_open_count(ROOT, lock_path=_lock_path(ROOT)),
            git_sha=str(lock["frozen_at_git_sha"]),
        )
        for alpha, payload in results.items():
            path = (
                results_dir
                / f"rung6_crc_base{base_rung}_alpha{alpha:g}_val_seed{one_seed}.json"
            )
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            written.append((path, payload))
    return written


@app.command()
def score(
    base_rung: int = typer.Option(
        2, "--base-rung", help="2 or 4 — which rung's decisions Rung 6 wraps. Rung 4 has "
        "no trained artifact on the v3 panel (LIMITATIONS.md §8.5 cut it), so 2 is the "
        "default and the incumbent bar."
    ),
    seed: str = typer.Option(
        "42",
        "--seed",
        help="One seed, or several comma-separated. Several load the 264 MB panel once.",
    ),
    alphas: str = typer.Option(
        ",".join(str(a) for a in DEFAULT_ALPHAS),
        "--alphas",
        help="Comma-separated nominal false-HOLD rates.",
    ),
    cal_fraction: float = typer.Option(
        0.5, "--cal-fraction", help="Share of val merchants used to calibrate."
    ),
    cal_seed: int = typer.Option(
        0, "--cal-seed", help="RNG seed for the by-merchant calibration carve."
    ),
    root: Path = typer.Option(Path("data/v2"), "--root"),  # noqa: B008
    panel: Path = typer.Option(dataset.DEFAULT_PANEL, "--panel"),  # noqa: B008
    config: Path = typer.Option(Path("configs/scenario_v2.yaml"), "--config", "-c"),  # noqa: B008
) -> None:
    """String-parsing shell over :func:`score_val`. ``rakshak.cli eval --rung 6`` is the
    other caller and goes straight to the function."""
    for path, payload in score_val(
        base_rung=base_rung,
        seeds=tuple(int(s) for s in seed.split(",")),
        alphas=tuple(float(a) for a in alphas.split(",")),
        cal_fraction=cal_fraction,
        cal_seed=cal_seed,
        root=root,
        panel=panel,
        config=config,
    ):
        typer.echo(
            json.dumps(
                {
                    "wrote": str(path),
                    "alpha": payload["alpha"],
                    "savings": payload["savings"],
                    "n_hold_base_rung": payload["n_hold_base_rung"],
                    "n_softened": payload["n_softened_hold_to_review"],
                }
            )
        )


if __name__ == "__main__":
    app()
