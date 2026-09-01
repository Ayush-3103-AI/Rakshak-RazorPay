"""Rung 7 as an **explainer**: onset localisation, and the artifact that judges it.

``models/rung7_hsmm.py`` (T-0123, #57) is an inference core. It had never been run on real
data and never scored against ``onset_localisation_error``, the metric
``docs/PRE-REGISTRATION-CYCLE3-2026-08-31.md`` §2 declared and
``EVAL-LOCK-CYCLE3.json`` sealed for it. This module closes that gap and is the whole of
Rung 7's presence in the pipeline:

1. :class:`HsmmOnsetExplainer` — the registered explainer. It satisfies
   ``explain.registry.Explainer`` and, critically, **does not** satisfy ``Scorer``: it has
   no ``predict``, so ``register`` accepts it and the scoring path cannot reach it.
2. :func:`measure` — fits a pooled HSMM on TRAIN-fold sequences, decodes a change-point per
   VAL-fold merchant with a known onset, and writes an **explanation-quality artifact**.

**Why this is not a ladder row, structurally and not by convention.**
``artifacts/build.py::read_result_rows`` globs ``data/v2/eval/*.json`` and turns every file
it finds into a row of ``ladder.json``. A Rung 7 file in that directory therefore *becomes*
a ladder row no matter what it says inside it — it would sit in the results table beside
Rungs 0-6 with a blank PR-AUC column, and a reader would reasonably conclude Rung 7 scored
badly rather than that it was never a scorer. So the artifact is written to
:data:`EXPLANATION_DIR` instead. #51 is explicit that Rung 7 runs "at Stage 2 of the cascade
only — on non-PASS decisions, never in the scoring path, never scored on PR-AUC", and the
directory is what enforces it.

**Why this lives in ``explain/`` and not ``models/`` or ``scripts/``.**
``models/rung7_hsmm.py`` may not import ``rakshak.eval`` or name a radioactive field
(Prime Directive 3, enforced by ``tests/gates/test_g4_no_leakage.py`` over
``features/`` and ``models/``, and by
``test_rung7_hsmm.py::test_no_scoring_rung_imports_the_hsmm`` over ``models/*.py``). This
module needs ``onset_localisation_error`` **and** ``drift_onset_at``, so it must sit outside
that quarantine — exactly where ``cli.py`` sits, and for exactly the same reason. It is in
``explain/`` rather than in ``scripts/`` so that ``rakshak.cli explain`` can import it: an
explainer nobody can reach from the CLI is an explainer nobody runs.

**What the estimator actually is, stated plainly.** The fit is unsupervised and univariate.
Nothing tells it which of the K decoded states is "healthy" in :data:`STATE_NAMES` terms, so
a semantic estimator ("the first day in the EXFIL state") is not available without the
narrative layer T-0124 owns. The estimator here is **the first structural break in the
decoded segmentation** — the first day the Viterbi path leaves its day-0 state. That is what
"when did drift begin" reduces to for an unlabelled onset, and it is a weaker claim than the
one Rung 7 will eventually make.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from hashlib import blake2b
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from rakshak.eval.lock import load_lock, read_open_count, resolve_authoritative
from rakshak.eval.metrics import onset_localisation_error
from rakshak.explain.registry import ExplanationRequest, register, registered
from rakshak.models.rung7_hsmm import MAX_DURATION, STATE_NAMES, HsmmNb, fit, seed_model

__all__ = [
    "EXPLANATION_DIR",
    "HsmmOnsetExplainer",
    "first_change_point",
    "measure",
]

ROOT: Final = Path(__file__).resolve().parents[3]
DATA_ROOT: Final = ROOT / "data" / "v2"

#: NOT ``data/v2/eval``. See the module docstring: that directory is the ladder's input glob,
#: and Rung 7 has no ladder row by design.
EXPLANATION_DIR: Final = DATA_ROOT / "explanation_quality"

#: The split geometry this repo has locked (``configs/scenario_v2.yaml::splits``,
#: cross-checked against ``EVAL-LOCK-CYCLE3.json`` by ``cli.py::_boundaries``). Verified at
#: run time by :func:`_check_boundaries` rather than trusted, so a moved boundary fails
#: loudly instead of silently scoring the wrong window.
ORIGIN: Final = date(2026, 1, 1)
TRAIN_END_DAY: Final = 239
VAL_END_DAY: Final = 299

#: ``cli.py::_merchant_fold_t0101``'s salt and shares. Same fold, so Rung 7's TRAIN/VAL
#: merchants are the same merchants every other rung's are.
_FOLD_SALT: Final = b"rakshak-t0101-merchant-fold"
_FOLD_SHARES: Final = (0.60, 0.15, 0.25)
_FOLD_NAMES: Final = ("train", "val", "test")


def _merchant_fold(merchant_id: str) -> str:
    digest = blake2b(merchant_id.encode(), key=_FOLD_SALT, digest_size=8).digest()
    u = int.from_bytes(digest, "big") / 2**64
    cumulative = 0.0
    for name, share in zip(_FOLD_NAMES, _FOLD_SHARES, strict=True):
        cumulative += share
        if u < cumulative:
            return name
    return _FOLD_NAMES[-1]


def first_change_point(model: HsmmNb, obs: np.ndarray) -> float:
    """The day the Viterbi path first leaves its day-0 state, or ``nan`` if it never does.

    ``nan`` is *declining to localise*, which
    :func:`~rakshak.eval.metrics.onset_localisation_error` counts as ``n_unlocalised``
    rather than dropping. That distinction is the reason this returns ``nan`` instead of
    falling back to a guess: a method that only fires on the easy half and reports a
    flattering IQR is the failure the metric was written to expose.
    """
    path = model.decode(obs)
    boundaries = np.flatnonzero(np.diff(path))
    return float(boundaries[0] + 1) if boundaries.size else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# The registered explainer
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class HsmmOnsetExplainer:
    """Stage-2 narrative for one non-PASS merchant-day: *when* this merchant changed.

    **Deliberately has no ``predict``.** ``explain.registry.register`` refuses anything
    satisfying ``Scorer``, and that refusal is the whole point of the register: an HSMM
    fitted on, and decoded over, the subset another rung already promoted would produce a
    PR-AUC computed on a population selected by a different model. This class is what the
    register was built to accept, and ``rung7_hsmm.HsmmNb`` — which it holds rather than
    inherits from — is what the register was built to keep out.

    ``sequences`` maps merchant id to its daily count vector.
    :class:`~rakshak.explain.registry.ExplanationRequest` carries the decision, not the
    merchant's history, so the history is held here. A merchant with no sequence gets an
    honest "cannot say" rather than a fabricated day: at Stage 2 the alternative to an
    explanation is silence, never a guess.
    """

    model: HsmmNb
    sequences: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "hsmm_onset"

    def explain(self, request: ExplanationRequest) -> str:
        obs = self.sequences.get(request.merchant_id)
        if obs is None or request.day < 1:
            return (
                f"No daily-count history is loaded for {request.merchant_id}, so this "
                f"explainer cannot say when its behaviour changed. The "
                f"{request.action.name} decision stands on the score alone."
            )
        path = self.model.decode(np.asarray(obs)[: request.day + 1])
        breaks = np.flatnonzero(np.diff(path))
        if not breaks.size:
            return (
                f"{request.merchant_id} shows one continuous behavioural regime through "
                f"day {request.day}: the duration model finds no change-point, so the "
                f"{request.action.name} rests on level, not on a change."
            )
        onset = int(breaks[-1]) + 1
        state = STATE_NAMES[int(path[-1])] if int(path[-1]) < len(STATE_NAMES) else "state "
        return (
            f"{request.merchant_id} entered its current transaction-volume regime "
            f"({state}) on day {onset}, {request.day - onset} day(s) before this "
            f"{request.action.name}. The regime before it had held for "
            f"{onset - (int(breaks[-2]) + 1) if breaks.size > 1 else onset} day(s)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# The measurement
# ─────────────────────────────────────────────────────────────────────────────


def _check_boundaries(config_path: Path) -> None:
    """Refuse to run if the window above has drifted from the scenario manifest."""
    import yaml

    splits = yaml.safe_load(config_path.read_text(encoding="utf-8"))["splits"]
    if (splits["train_end_day"], splits["val_end_day"]) != (TRAIN_END_DAY, VAL_END_DAY):
        raise SystemExit(
            f"{config_path} splits {splits} no longer match the (train_end_day, "
            f"val_end_day)=({TRAIN_END_DAY}, {VAL_END_DAY}) this module pins — update the "
            "constants before trusting the result."
        )


def _ground_truth_folds(root: Path) -> pl.DataFrame:
    """``merchant_id``, ``fold``, ``onset_day`` (null if the merchant never drifted).

    Reads ``ground_truth.parquet`` legitimately, from the eval side, exactly as
    ``cli.py::_build_truth`` does. ``models/rung7_hsmm.py`` never sees any of it.
    """
    gt = pl.read_parquet(root / "ground_truth.parquet")
    return gt.select(
        "merchant_id",
        pl.col("merchant_id").map_elements(_merchant_fold, return_dtype=pl.String).alias("fold"),
        pl.when(pl.col("drift_onset_at").is_not_null())
        .then((pl.col("drift_onset_at").dt.date() - ORIGIN).dt.total_days())
        .otherwise(None)
        .cast(pl.Float64)
        .alias("onset_day"),
    )


def _daily_counts(root: Path, merchant_ids: list[str]) -> tuple[np.ndarray, float]:
    """``(len(merchant_ids), VAL_END_DAY+1)`` daily transaction counts, and the seconds taken.

    One channel: total transaction events per merchant-day (any status, refunds included) —
    the cheapest univariate "something changed" signal. A 2-3 channel extension (distinct
    payers, failure count) is real future work and is not done here.

    Bounded at ``VAL_END_DAY``. Days 300-364 are never read (Prime Directive 1).
    """
    started = time.perf_counter()
    end_date = ORIGIN + timedelta(days=VAL_END_DAY)
    long = (
        pl.scan_parquet(root / "transactions.parquet")
        .filter(pl.col("merchant_id").is_in(merchant_ids))
        .filter((pl.col("event_date") >= ORIGIN) & (pl.col("event_date") <= end_date))
        .group_by("merchant_id", "event_date")
        .agg(pl.len().alias("n"))
        .collect()
    )
    elapsed = time.perf_counter() - started

    index = {m: i for i, m in enumerate(merchant_ids)}
    rows = np.fromiter((index[m] for m in long["merchant_id"]), dtype=np.int64, count=long.height)
    cols = (
        (long["event_date"].to_numpy() - np.datetime64(ORIGIN))
        .astype("timedelta64[D]")
        .astype(np.int64)
    )
    matrix = np.zeros((len(merchant_ids), VAL_END_DAY + 1), dtype=np.int64)
    matrix[rows, cols] = long["n"].to_numpy()
    return matrix, elapsed


def measure(
    *,
    seeds: tuple[int, ...] = (1,),
    fit_pool_size: int = 500,
    n_iter: int = 15,
    n_states: int = 4,
    root: Path = DATA_ROOT,
    config: Path = ROOT / "configs" / "scenario_v2.yaml",
    echo: Any = print,
) -> list[Path]:
    """Fit, decode, score against ``onset_localisation_error``, and write the artifact.

    ``seeds`` are **EM initialisation** seeds, not the five model seeds
    ``EVAL-LOCK-CYCLE3.json`` declares. Those five govern rungs judged on PR-AUC under the
    declared adoption margin; Rung 7 is judged on onset localisation and is not on that
    ladder, so borrowing their numbers here would imply a comparison that does not exist.
    The artifact says so in ``seed_meaning``.
    """
    _check_boundaries(config)
    folds = _ground_truth_folds(root)
    train_onset = folds.filter((pl.col("fold") == "train") & pl.col("onset_day").is_not_null())
    train_other = folds.filter((pl.col("fold") == "train") & pl.col("onset_day").is_null())
    val_onset = folds.filter((pl.col("fold") == "val") & pl.col("onset_day").is_not_null())

    # Fixed across seeds: the same fit pool every time, so only the EM initialisation varies.
    pool_rng = np.random.default_rng(20260901)
    n_extra = max(0, fit_pool_size - train_onset.height)
    extra_ids = pool_rng.choice(
        train_other["merchant_id"].to_numpy(), size=min(n_extra, train_other.height), replace=False
    ).tolist()
    fit_pool_ids = train_onset["merchant_id"].to_list() + extra_ids
    val_ids = val_onset["merchant_id"].to_list()
    val_true_onset = val_onset["onset_day"].to_numpy()

    echo(
        f"fit pool: {len(fit_pool_ids)} TRAIN-fold merchants "
        f"({train_onset.height} with a known onset + {len(extra_ids)} random others)"
    )
    echo(f"scoring:  {len(val_ids)} VAL-fold merchants with a known drift_onset_at")

    counts, materialise_seconds = _daily_counts(root, fit_pool_ids + val_ids)
    echo(f"materialisation: {materialise_seconds:.2f}s, days 0-{VAL_END_DAY}")

    fit_seqs = [counts[i] for i in range(len(fit_pool_ids))]
    val_seqs = {m: counts[len(fit_pool_ids) + j] for j, m in enumerate(val_ids)}

    EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = resolve_authoritative(ROOT)
    lock = load_lock(ROOT, lock_path=lock_path)
    prevalence = float(val_onset.height) / float(folds.filter(pl.col("fold") == "val").height)

    written: list[Path] = []
    for seed in seeds:
        init = seed_model(
            fit_seqs, n_states=n_states, rng=np.random.default_rng(seed), max_duration=MAX_DURATION
        )
        started = time.perf_counter()
        em = fit(fit_seqs, init, n_iter=n_iter, tol=1e-6)
        fit_seconds = time.perf_counter() - started
        echo(
            f"[seed {seed}] EM: {em.n_iter} iterations "
            f"({'converged' if em.converged else 'hit the n_iter cap'}), "
            f"loglik {em.log_likelihoods[0]:.1f} -> {em.log_likelihoods[-1]:.1f}, "
            f"{fit_seconds:.1f}s"
        )

        started = time.perf_counter()
        estimated = np.array([first_change_point(em.model, val_seqs[m]) for m in val_ids])
        decode_seconds = time.perf_counter() - started
        onset = onset_localisation_error(estimated, val_true_onset)
        echo(
            f"[seed {seed}] onset_localisation_error: median={onset.median} "
            f"iqr={onset.iqr} n={onset.n} n_unlocalised={onset.n_unlocalised}"
        )

        # Registered, not merely defined. `register` refuses anything with a `predict`, so
        # this call is the runtime proof that Rung 7 cannot reach the scoring path — the
        # assertion tests/unit/test_rung7_hsmm.py makes structurally, made again on the
        # real fitted object. Registration happens HERE rather than at import because an
        # explainer without a fitted model can explain nothing; the guard is for the
        # second call in one process, since a name may be registered only once.
        explainer = HsmmOnsetExplainer(model=em.model, sequences=val_seqs)
        if explainer.name not in registered():
            register(explainer)

        payload = {
            "artifact": "explanation_quality",
            "rung": 7,
            "role": "explainer",
            "label": "rung7_hsmm_onset",
            "split": "val",
            "explainer_name": explainer.name,
            "adopted": False,
            "not_a_ladder_row": (
                "Rung 7 is an EXPLAINER, not a scoring rung. It has no PR-AUC, no savings, "
                "no precision@K and no capacity K, and it makes no claim on any of them — "
                "not a bad score, no score. It is therefore written here and NOT to "
                "data/v2/eval/, which artifacts/build.py::read_result_rows globs into "
                "ladder.json. Its declared metric is onset_localisation_error "
                "(PRE-REGISTRATION-CYCLE3 §2), and that is the only number below."
            ),
            "seed": seed,
            "seed_meaning": (
                "EM initialisation seed, NOT one of EVAL-LOCK-CYCLE3.json's five declared "
                "model seeds (42-46). Those govern rungs judged on the declared PR-AUC/TTD "
                "adoption margin; Rung 7 is not one of them."
            ),
            "val_merchant_drift_prevalence": prevalence,
            "channel": "daily_transaction_count (all statuses, C=1)",
            "onset_estimator": "first Viterbi segmentation boundary after day 0",
            "onset_localisation_error": {
                "median": onset.median,
                "q25": onset.q25,
                "q75": onset.q75,
                "iqr": onset.iqr,
                "n": onset.n,
                "n_unlocalised": onset.n_unlocalised,
                "sign_convention": "estimated - true; negative is EARLY, positive is LATE",
                "error_days": onset.error_days.tolist(),
            },
            "fit": {
                "fold": "train (merchant-level, T-0101 60/15/25 salt)",
                "pool_size": len(fit_pool_ids),
                "n_known_onset_in_pool": train_onset.height,
                "n_states": n_states,
                "state_names": list(STATE_NAMES),
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
            "example_explanations": [
                explainer.explain(_example_request(m, VAL_END_DAY)) for m in val_ids[:3]
            ],
            "nfr_caveat": (
                "The 50 ms Stage-2 latency budget is NOT certified. rung7_hsmm.py's own "
                "docstring records that its timings were taken on a contended box; decode "
                "time above is per-batch wall clock, not a p99 per merchant on a quiet box. "
                "Prime Directive 5 needs the NFR as well as the metric."
            ),
        }
        path = EXPLANATION_DIR / f"rung7_hsmm_onset_val_emseed{seed}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        echo(f"[seed {seed}] wrote {path}")
        written.append(path)
    return written


def _example_request(merchant_id: str, day: int) -> ExplanationRequest:
    """A Stage-2 request for the sample explanations in the artifact.

    ``action`` is HOLD because Stage 2 runs on non-PASS decisions only — an explanation
    attached to a PASS is an explanation nobody asked for.
    """
    from rakshak.schemas import Action

    return ExplanationRequest(
        merchant_id=merchant_id,
        day=day,
        x=np.zeros(0),
        columns=(),
        score=float("nan"),
        action=Action.HOLD,
    )


def main() -> None:
    """``python -m rakshak.explain.hsmm_onset``. ``rakshak.cli explain`` is the other door."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--fit-pool-size", type=int, default=500)
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--n-states", type=int, default=4)
    args = parser.parse_args()
    measure(
        seeds=tuple(args.seeds),
        fit_pool_size=args.fit_pool_size,
        n_iter=args.n_iter,
        n_states=args.n_states,
    )


if __name__ == "__main__":
    main()
