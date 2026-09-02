"""Rung 7b's runner — onset localisation, state agreement, the segmented narrative (#58).

Rung 7a (``rakshak.score_rung7``) fits the HSMM and scores *the first structural break* in
the decoded segmentation against ``drift_onset_at``. #58 asks a narrower and harder question:
the signed error of the inferred **HEALTHY -> RAMP** transition, over **alerted true-positive
merchants only**, against a trivial baseline that guesses "onset = the day we alerted".

Four things this file has to do that 7a did not, and where each lives:

1. **Name the decoded states without looking at a label.** ``explain/segmentation.py``:
   HEALTHY is the modal decoded state across the fit pool, the rest are named by NB mean
   relative to it. No reference partition is consulted, so the estimate is an estimate.
2. **Reconstruct the reference partition.** ``ground_truth.parquet`` records
   ``drift_onset_at`` but **not** ``ramp_days``, and without the ramp length there is no
   RAMP/EXFIL boundary and therefore no four-state reference to agree with. So
   :func:`_replay_assignment` re-runs the generator's own RNG stream up to
   ``assign_typologies`` and recovers it — and then **verifies** the recovery by asserting
   that every replayed onset equals the committed one, 588/588, before using any of it.
   A replay that silently drifted would produce a reference partition that looks fine and
   is wrong, which is the exact failure mode this project exists to refuse.
3. **Find the alerted true positives.** That needs the decision layer, so it re-scores the
   base rung on the validation panel exactly as ``cli.py::score_split`` does — same panel,
   same trained booster, same ``capacity.DEFAULT_DECISION``, same K.
4. **Render the timeline beside the reason codes.** The ``pred_contrib`` codes come from the
   same booster's ``reason_codes``; the timeline comes from the registered
   ``SegmentedTimelineExplainer``. Both go in the artifact, one after the other, because
   #58's criterion is "alongside ... not in place of them".

**This is a script and not ``src/rakshak/score_rung7b.py`` because T-0124's file allowlist
says so** — two other agents are editing this tree. The consequence is that
``rakshak.cli explain`` cannot reach it; ``uv run python scripts/rung7b_score.py`` is the
only door, and promoting it into the package next to ``score_rung7`` is a one-move follow-up.

**Prime Directive 1.** Days 300-364 are never read. The panel is selected on ``"val"``, the
counts are bounded at ``VAL_END_DAY``, and nothing here sets ``RAKSHAK_UNLOCK``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # `uv run python scripts/...` without an install
    sys.path.insert(0, str(ROOT / "src"))

from rakshak.eval import capacity  # noqa: E402
from rakshak.eval.lock import load_lock, read_open_count, resolve_authoritative  # noqa: E402
from rakshak.eval.metrics import OnsetLocalisation, onset_localisation_error  # noqa: E402
from rakshak.explain.registry import ExplanationRequest, register, registered  # noqa: E402
from rakshak.explain.segmentation import (  # noqa: E402
    SegmentedTimelineExplainer,
    modal_state,
    name_states,
    onset_from_healthy,
    segments,
    state_agreement,
)
from rakshak.models import dataset  # noqa: E402
from rakshak.models.rung7_hsmm import MAX_DURATION, STATE_NAMES, fit, seed_model  # noqa: E402
from rakshak.schemas import Action  # noqa: E402
from rakshak.score_rung7 import (  # noqa: E402
    EXPLANATION_DIR,
    ORIGIN,
    VAL_END_DAY,
    _check_boundaries,
    _daily_counts,
    _ground_truth_folds,
)

#: The generator's own fit-pool RNG seed, copied from ``score_rung7.measure`` so 7a and 7b
#: fit on the identical pool and any difference between them is the estimator, not the data.
POOL_SEED: Final = 20260901


# ─────────────────────────────────────────────────────────────────────────────
# The reference partition: HEALTHY / RAMP / EXFIL / BURNT per merchant-day
# ─────────────────────────────────────────────────────────────────────────────


def _replay_assignment(config: Path, root: Path) -> pl.DataFrame:
    """Recover ``(onset_day, ramp_days, typology)`` per merchant, and verify the recovery.

    ``ramp_days`` is drawn inside ``generator.typologies.assign_typologies`` and never
    persisted — ``ground_truth.parquet`` keeps ``drift_onset_at`` and the typology id and
    nothing else — so the only way to a RAMP/EXFIL boundary is to replay the generator's
    single threaded RNG up to that call. ``engine.generate`` consumes a fixed prefix of
    draws before it (personas, MCC group, MCC, onboarding offset, declared GMV), so the
    replay reproduces them in order rather than guessing.

    The verification is the point: every replayed onset must equal the committed
    ``drift_onset_at``. If a config value, the seed or the draw order ever moves, this
    raises instead of handing back a plausible and wrong reference partition.
    """
    from rakshak.generator import engine
    from rakshak.generator import personas as personas_mod
    from rakshak.generator.config import load_scenario
    from rakshak.generator.typologies import assign_typologies

    cfg = load_scenario(config)
    rng = np.random.default_rng(cfg.seed)
    n = cfg.population.n_merchants

    persona_idx = personas_mod.sample_persona_ids(rng, n, cfg.personas)
    for field_name in (
        "base_daily_txns",
        "amount_mu",
        "amount_sigma",
        "cnp_share",
        "fail_rate",
        "refund_rate",
        "refund_latency_hours",
        "new_payer_rate",
        "payer_pool",
        "payout_period_days",
        "payout_drawdown",
    ):
        engine._persona_field(cfg, persona_idx, field_name)
    declarable = sorted(g for g in cfg.mcc_groups if g != cfg.mcc_drift_group)
    group_idx = rng.integers(0, len(declarable), size=n)
    mcc_group = np.array(declarable, dtype=object)[group_idx]
    [
        cfg.mcc_groups[g][int(u * len(cfg.mcc_groups[g]))]
        for g, u in zip(mcc_group, rng.random(n), strict=True)
    ]
    rng.integers(1, cfg.population.onboarding_spread_days + 1, size=n).astype(np.int64)
    rng.normal(0.0, cfg.population.declaration_error_sigma, n)
    assignment = assign_typologies(rng, n, cfg.population.prevalence, cfg.typologies)

    order = list(cfg.typologies)
    vanishes = np.array(
        [cfg.typologies[t].vanish_after_ramp for t in order] + [False], dtype=bool
    )
    index = np.where(assignment.is_fraud, assignment.typology_index, len(order))
    replayed = pl.DataFrame(
        {
            "merchant_id": [f"M{i:06d}" for i in range(n)],
            "replay_onset_day": assignment.onset_day.astype(np.int64),
            "ramp_days": assignment.ramp_days.astype(np.int64),
            "vanishes": vanishes[index],
        }
    )

    committed = pl.read_parquet(root / "ground_truth.parquet").select(
        "merchant_id",
        pl.when(pl.col("drift_onset_at").is_not_null())
        .then((pl.col("drift_onset_at").dt.date() - ORIGIN).dt.total_days())
        .otherwise(None)
        .cast(pl.Int64)
        .alias("onset_day"),
    )
    joined = committed.join(replayed, on="merchant_id", how="inner")
    drifted = joined.filter(pl.col("onset_day").is_not_null())
    agree = int((drifted["onset_day"] == drifted["replay_onset_day"]).sum())
    if joined.height != committed.height or agree != drifted.height:
        raise SystemExit(
            f"generator replay does not reproduce the committed ground truth: {agree} of "
            f"{drifted.height} onsets agree over {joined.height}/{committed.height} joined "
            f"merchants. ramp_days cannot be trusted, so the four-state reference partition "
            f"cannot be built. Check configs/scenario_v2.yaml's seed and the draw order in "
            f"generator/engine.py::generate before this file's line ordering."
        )
    return joined.drop("replay_onset_day")


def _reference_path(
    onset: int, ramp: int, vanishes: bool, n_days: int, state_names: tuple[str, ...]
) -> np.ndarray:
    """The generator's own phase schedule for one drifted merchant, as state names.

    ``HEALTHY`` before onset; ``RAMP`` while ``typologies.ramp_progress`` is in ``[0, 1)``;
    then ``EXFIL``, or ``BURNT`` for a ``vanish_after_ramp`` typology — which is exactly
    where ``typologies.intensity_multiplier`` switches to ``vanish_intensity``. The four
    names are ``rung7_hsmm.STATE_NAMES`` and the four phases are the generator's; that they
    line up is not a coincidence, it is why K=4 was chosen.
    """
    healthy, ramping, exfil, burnt = state_names
    days = np.arange(n_days)
    terminal = burnt if vanishes else exfil
    return np.where(
        days < onset, healthy, np.where(days < onset + max(ramp, 1), ramping, terminal)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alerted true positives: the population #58 restricts the metric to
# ─────────────────────────────────────────────────────────────────────────────


def _first_alert_day(
    *, base_rung: int, seed: int, panel: Path, config: Path, echo: Any
) -> tuple[dict[str, int], Any, Any]:
    """``merchant_id -> first VAL day this merchant was not PASSed``, plus the rows and model.

    Re-scores rather than reads, because no committed artefact carries a per-merchant
    decision — ``EvalResult`` is aggregate, which is the same gap that made Rung 9's
    McNemar gate uncomputable (roster note, cycle 4). The wiring is copied from
    ``cli.py::score_split``: same panel, same booster, same ``DEFAULT_DECISION``, same K,
    declared-exposure arm. A different decision layer here would mean "alerted" meant
    something the ladder never reported.

    The panel and the booster come back with the alerts because :func:`_reason_codes`
    needs both and the panel is 6.15M rows — loading it twice for three reason codes is
    the kind of thing that makes a runner too slow to re-run.
    """
    from rakshak.cli import _action_policy, _capacity, _cost_params, _load_trained

    rows = dataset.load_panel(panel).select("val")
    if rows.x.shape[0] == 0:
        raise SystemExit(f"{panel} holds no val rows")
    model = _load_trained(base_rung, seed)
    score = model.predict(rows.x, rows.columns)
    k = _capacity(len({*rows.merchant_id.tolist()}))
    action = capacity.DEFAULT_DECISION.decide(
        capacity.DecisionRequest(
            score=score,
            day=rows.day,
            exposure_inr=rows.column("p_declared_monthly_gmv"),
            k=k,
            params=_cost_params(config),
            hold_policy=_action_policy(config),
        )
    )
    alerted = np.asarray(action) != Action.PASS
    echo(f"base rung {base_rung} seed {seed}: K={k}, {int(alerted.sum())} non-PASS rows")
    first: dict[str, int] = {}
    for merchant, day in zip(
        rows.merchant_id[alerted].tolist(), rows.day[alerted].tolist(), strict=True
    ):
        if merchant not in first or day < first[merchant]:
            first[merchant] = int(day)
    return first, rows, model


def _reason_codes(*, rows: Any, model: Any, merchant_id: str, day: int) -> list[str]:
    """The merchant-day's existing ``pred_contrib`` reason codes, unchanged.

    Fetched so the artifact can show the timeline **beside** them. Rung 1 has no booster,
    so a non-LightGBM base rung degrades to an honest empty list rather than a fabricated
    one.
    """
    hit = np.flatnonzero((rows.merchant_id == merchant_id) & (rows.day == day))
    codes = getattr(model, "reason_codes", None)
    if hit.size == 0 or codes is None:
        return []
    return list(codes(rows.x, rows.columns, hit[:1])[0])


# ─────────────────────────────────────────────────────────────────────────────


def _summary(onset: OnsetLocalisation) -> dict[str, Any]:
    absolute = np.abs(onset.error_days)
    return {
        "median": onset.median,
        "q25": onset.q25,
        "q75": onset.q75,
        "iqr": onset.iqr,
        "median_abs": float(np.median(absolute)) if absolute.size else float("nan"),
        "n": onset.n,
        "n_unlocalised": onset.n_unlocalised,
        "error_days": onset.error_days.tolist(),
    }


def measure(
    *,
    seed: int = 1,
    base_rung: int = 4,
    base_seed: int = 42,
    fit_pool_size: int = 500,
    n_iter: int = 15,
    n_states: int = 4,
    root: Path = ROOT / "data" / "v2",
    panel: Path = ROOT / "data" / "v2" / "features.parquet",
    config: Path = ROOT / "configs" / "scenario_v2.yaml",
    echo: Any = print,
) -> Path:
    """Fit, decode, localise, compare against the trivial baseline, and write the artifact.

    The comparison rule is fixed **here, before any number is read**: Rung 7b beats the
    trivial "onset = alert day" baseline only if its median absolute error is strictly
    smaller. Median absolute rather than signed median because a signed median near zero
    can be two large errors of opposite sign cancelling, and because the baseline is
    structurally one-signed (an alert cannot precede the data that triggers it, so its
    error is almost always late) — comparing signed medians would flatter whichever method
    happens to straddle zero.
    """
    _check_boundaries(config)
    folds = _ground_truth_folds(root)
    reference = _replay_assignment(config, root)
    echo(f"generator replay verified against {reference.height} committed ground-truth rows")

    train_onset = folds.filter((pl.col("fold") == "train") & pl.col("onset_day").is_not_null())
    train_other = folds.filter((pl.col("fold") == "train") & pl.col("onset_day").is_null())
    val_onset = folds.filter((pl.col("fold") == "val") & pl.col("onset_day").is_not_null())

    pool_rng = np.random.default_rng(POOL_SEED)
    n_extra = max(0, fit_pool_size - train_onset.height)
    extra_ids = pool_rng.choice(
        train_other["merchant_id"].to_numpy(), size=min(n_extra, train_other.height), replace=False
    ).tolist()
    fit_pool_ids = train_onset["merchant_id"].to_list() + extra_ids

    alerts, val_rows, base_model = _first_alert_day(
        base_rung=base_rung, seed=base_seed, panel=panel, config=config, echo=echo
    )
    evaluable = val_onset.filter(
        pl.col("merchant_id").is_in(list(alerts)) & (pl.col("onset_day") <= float(VAL_END_DAY))
    ).sort("merchant_id")
    val_ids = evaluable["merchant_id"].to_list()
    true_onset = evaluable["onset_day"].to_numpy()
    echo(
        f"val fold: {val_onset.height} merchants with a known onset; "
        f"{len(val_ids)} of them alerted by rung {base_rung} AND onsetting by day "
        f"{VAL_END_DAY} — that is the evaluable population"
    )
    if not val_ids:
        raise SystemExit(
            "no alerted true-positive merchant has an in-window onset, so "
            "onset_localisation_error has an empty denominator. That is a reportable "
            "result, not a crash, but it is not one this runner can write an artifact for."
        )

    counts, materialise_seconds = _daily_counts(root, fit_pool_ids + val_ids)
    echo(f"materialisation: {materialise_seconds:.2f}s, days 0-{VAL_END_DAY}")
    fit_seqs = [counts[i] for i in range(len(fit_pool_ids))]
    val_seqs = {m: counts[len(fit_pool_ids) + j] for j, m in enumerate(val_ids)}

    init = seed_model(
        fit_seqs, n_states=n_states, rng=np.random.default_rng(seed), max_duration=MAX_DURATION
    )
    started = time.perf_counter()
    em = fit(fit_seqs, init, n_iter=n_iter, tol=1e-6)
    fit_seconds = time.perf_counter() - started
    echo(
        f"EM: {em.n_iter} iterations "
        f"({'converged' if em.converged else 'hit the n_iter cap'}), "
        f"loglik {em.log_likelihoods[0]:.1f} -> {em.log_likelihoods[-1]:.1f}, {fit_seconds:.1f}s"
    )

    started = time.perf_counter()
    fit_paths = [em.model.decode(s) for s in fit_seqs]
    healthy = modal_state(fit_paths, n_states)
    names = name_states(em.model.nb_mean, healthy, STATE_NAMES)
    val_paths = {m: em.model.decode(val_seqs[m]) for m in val_ids}
    decode_seconds = time.perf_counter() - started
    echo(f"decode: {decode_seconds:.1f}s; HEALTHY = state {healthy}; state names {names}")

    # The estimator #58 names is the HEALTHY -> RAMP transition specifically. The relaxed
    # HEALTHY -> anything variant is reported beside it, not instead of it: if the two
    # disagree the model has not found an ordered escalation, and that is worth knowing
    # whichever way the comparison against the baseline lands.
    ramp = names.index(STATE_NAMES[1]) if STATE_NAMES[1] in names else None
    estimated = np.array([onset_from_healthy(val_paths[m], healthy, ramp) for m in val_ids])
    rung7b = onset_localisation_error(estimated, true_onset)
    relaxed = onset_localisation_error(
        np.array([onset_from_healthy(val_paths[m], healthy) for m in val_ids]), true_onset
    )
    regimes = [len(segments(val_paths[m])) for m in val_ids]
    baseline = onset_localisation_error(
        np.array([float(alerts[m]) for m in val_ids]), true_onset
    )

    # State agreement, over the same evaluable merchants and the same day window.
    lookup = {
        row["merchant_id"]: row
        for row in reference.filter(pl.col("merchant_id").is_in(val_ids)).to_dicts()
    }
    predicted_states: list[np.ndarray] = []
    reference_states: list[np.ndarray] = []
    for merchant in val_ids:
        row = lookup[merchant]
        decoded = np.asarray(val_paths[merchant])
        predicted_states.append(np.array([names[int(s)] for s in decoded]))
        reference_states.append(
            _reference_path(
                int(row["onset_day"]),
                int(row["ramp_days"]),
                bool(row["vanishes"]),
                int(decoded.size),
                STATE_NAMES,
            )
        )
    agreement = state_agreement(
        np.concatenate(predicted_states), np.concatenate(reference_states)
    )

    explainer = SegmentedTimelineExplainer(
        paths=val_paths, names=names, mean_dwell=em.model.mean_dwell, healthy=healthy
    )
    if explainer.name not in registered():
        register(explainer)

    replayed = val_ids[0]
    replay_day = alerts[replayed]
    narrative = {
        "merchant_id": replayed,
        "day": replay_day,
        "action": Action.HOLD.name,
        "pred_contrib_reason_codes": _reason_codes(
            rows=val_rows, model=base_model, merchant_id=replayed, day=replay_day
        ),
        "segmented_timeline": explainer.explain(
            ExplanationRequest(
                merchant_id=replayed,
                day=replay_day,
                x=np.zeros(0),
                columns=(),
                score=float("nan"),
                action=Action.HOLD,
            )
        ),
        "true_drift_onset_day": int(lookup[replayed]["onset_day"]),
        "note": (
            "The reason codes are the base rung's existing pred_contrib output, unchanged. "
            "The timeline is printed BESIDE them, not instead of them (#58)."
        ),
    }

    beats_baseline = bool(
        rung7b.n > 0
        and np.median(np.abs(rung7b.error_days)) < np.median(np.abs(baseline.error_days))
    )
    lock_path = resolve_authoritative(ROOT)
    lock = load_lock(ROOT, lock_path=lock_path)
    payload = {
        "artifact": "explanation_quality",
        "rung": "7b",
        "role": "explainer",
        "label": "rung7b_onset_localisation",
        "split": "val",
        "adopted": False,
        "not_a_ladder_row": (
            "Rung 7b is the narrative half of an EXPLAINER. It has no PR-AUC, no savings and "
            "no capacity K, and makes no claim on any of them. Written here and NOT to "
            "data/v2/eval/, which artifacts/build.py::read_result_rows globs into ladder.json."
        ),
        "em_seed": seed,
        "base_rung": base_rung,
        "base_seed": base_seed,
        "population": (
            "alerted true positives: VAL-fold merchants with a non-null drift_onset_at at or "
            "before day 299 that the base rung did not PASS on at least one validation day"
        ),
        "n_evaluable": len(val_ids),
        "n_val_fold_with_onset": val_onset.height,
        "onset_estimator": "first HEALTHY -> RAMP transition in the Viterbi path (#58)",
        "healthy_state_index": healthy,
        "ramp_state_index": ramp,
        "state_naming_rule": (
            "unsupervised: HEALTHY = modal decoded state over the fit pool; the rest named by "
            "fitted NB mean relative to it (above, ascending = RAMP then EXFIL; below = "
            "BURNT). NOT an optimal assignment to the reference partition."
        ),
        "state_names_by_index": list(names),
        "onset_localisation_error": _summary(rung7b),
        "onset_localisation_error_relaxed": {
            "estimator": "first HEALTHY -> ANY non-HEALTHY transition; reported beside the "
            "primary, never instead of it",
            **_summary(relaxed),
        },
        "trivial_baseline": {
            "estimator": "onset = the first validation day the base rung alerted",
            **_summary(baseline),
        },
        "decoded_regimes_per_merchant": {
            "note": (
                "the mechanism behind the number above: how many constant-state runs the "
                "Viterbi path breaks a 300-day sequence into. A segmentation that flips "
                "dozens of times has a first HEALTHY-exit near day 0 for every merchant, "
                "which is an onset estimate that cannot be wrong late and is almost always "
                "wrong early."
            ),
            "median": float(np.median(regimes)),
            "min": int(np.min(regimes)),
            "max": int(np.max(regimes)),
        },
        "comparison": {
            "rule": (
                "declared before the numbers were read: 7b beats the baseline only if its "
                "MEDIAN ABSOLUTE error is strictly smaller"
            ),
            "beats_trivial_baseline": beats_baseline,
        },
        "sign_convention": "estimated - true; negative is EARLY, positive is LATE",
        "state_recovery": {
            "headline": "ami",
            "ami": agreement.ami,
            "ari": agreement.ari,
            "ari_note": (
                "printed beside AMI, never instead of it. Romano, Vinh, Bailey & Verspoor "
                "(JMLR 17, 2016): ARI's null model assumes balanced clusters and this "
                "reference partition is not balanced — see reference_state_support."
            ),
            "recall_by_state": agreement.recall,
            "macro_recall": agreement.macro_recall,
            "reference_state_support": agreement.support,
            "n_merchant_days": agreement.n,
        },
        "reference_partition_source": (
            "generator RNG replay up to typologies.assign_typologies, VERIFIED by asserting "
            "every replayed onset equals the committed drift_onset_at. ramp_days is not "
            "persisted in ground_truth.parquet and there is no other route to a RAMP/EXFIL "
            "boundary."
        ),
        "segmented_narrative": narrative,
        "fit": {
            "pool_size": len(fit_pool_ids),
            "n_known_onset_in_pool": train_onset.height,
            "n_states": n_states,
            "max_duration": MAX_DURATION,
            "n_iter_run": em.n_iter,
            "n_iter_cap": n_iter,
            "converged": em.converged,
            "log_likelihood_start": em.log_likelihoods[0],
            "log_likelihood_end": em.log_likelihoods[-1],
            "fit_seconds": fit_seconds,
        },
        "decode_seconds": decode_seconds,
        "materialise_seconds": materialise_seconds,
        "window_days": [0, VAL_END_DAY],
        "open_count": read_open_count(ROOT, lock_path=lock_path),
        "eval_lock_sha": str(lock["eval_module_sha256"]),
        "git_sha": str(lock["frozen_at_git_sha"]),
        "nfr_caveat": (
            "The 50 ms Stage-2 latency budget is NOT certified. decode_seconds is per-batch "
            "wall clock on a contended box, not a p99 per merchant on a quiet one."
        ),
    }
    EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)
    path: Path = EXPLANATION_DIR / f"rung7b_onset_localisation_val_emseed{seed}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    echo(
        f"onset_localisation_error (rung 7b, HEALTHY->RAMP): median={rung7b.median} "
        f"iqr={rung7b.iqr} n={rung7b.n} n_unlocalised={rung7b.n_unlocalised}"
    )
    echo(
        f"onset_localisation_error (rung 7b, HEALTHY->any): median={relaxed.median} "
        f"iqr={relaxed.iqr} n={relaxed.n} n_unlocalised={relaxed.n_unlocalised}"
    )
    echo(
        f"decoded regimes per merchant: median={np.median(regimes)} "
        f"min={np.min(regimes)} max={np.max(regimes)}"
    )
    echo(
        f"onset_localisation_error (trivial): median={baseline.median} iqr={baseline.iqr} "
        f"n={baseline.n} n_unlocalised={baseline.n_unlocalised}"
    )
    echo(f"beats_trivial_baseline: {beats_baseline}")
    echo(
        f"state recovery: AMI={agreement.ami:.4f} (headline)  ARI={agreement.ari:.4f} "
        f"(beside it)  macro-recall={agreement.macro_recall:.4f}"
    )
    for state, value in agreement.recall.items():
        echo(f"  recall[{state}] = {value:.4f}  (support {agreement.support[state]})")
    echo(f"wrote {path}")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Rung 7b: onset localisation (#58)")
    parser.add_argument("--seed", type=int, default=1, help="EM initialisation seed.")
    parser.add_argument("--base-rung", type=int, default=4, help="Whose alerts define 'alerted'.")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--fit-pool-size", type=int, default=500)
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--n-states", type=int, default=4)
    args = parser.parse_args()
    measure(
        seed=args.seed,
        base_rung=args.base_rung,
        base_seed=args.base_seed,
        fit_pool_size=args.fit_pool_size,
        n_iter=args.n_iter,
        n_states=args.n_states,
    )


if __name__ == "__main__":
    main()
