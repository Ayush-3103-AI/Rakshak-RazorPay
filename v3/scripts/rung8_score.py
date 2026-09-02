"""Rung 8's runner: the two circularity mitigations GitHub #59 makes mandatory, then score.

#59 states the circularity objection up front rather than leaving it to be discovered:
the generator produces NB/Hawkes arrivals, so fitting an NB/Hawkes intensity to them and
calling the misfit "anomaly" is well-specified-model-on-well-specified-data — the same
objection v1's ADR-0002 used to reject GNNs, and an *evaluation-validity* problem rather
than a compute one. The ticket therefore makes three things mandatory, and this script is
the first two of them:

1. **The null.** Measure the statistic at ``prevalence = 0`` **with confounders on** —
   literally ``tests/gates/gates_report.scenario(prevalence=0.0)``, the same configuration
   G5's ``null_data`` fixture builds, imported rather than re-declared so the two cannot
   drift apart. Every alert there is a false positive by construction. If the alert rate
   inside a confounder window exceeds nominal by more than G5's own +2pp headroom, Rung 8
   is detecting the *platform* and not the *merchant*, and it fails the bar every other
   rung must clear.
2. **BAF.** Check the test's size against the external anchor through
   ``eval.baf_adapter``. BAF is CC BY-NC-SA and is deliberately **not vendored**; when it
   is absent this records SKIP with the reason, exactly as the four skipped gates do. A
   skip is an unmet acceptance criterion and is reported as one, not as a pass.

The third is ``LIMITATIONS.md``'s Rung 8 section, written from what these two find.

Part 3 then scores the statistic on the **validation** fold of the real cycle-4 dataset.
Days 300-364 are never read: ``VAL_END_DAY`` bounds every scan, the test split stays shut
and ``open_count`` stays 0. No ``EVAL-LOCK`` module is imported, hashed or touched.

**This writes to ``data/v2/rung8_tpp/``, NOT ``data/v2/eval/``.** Rung 7's runner records
why in the same words: ``artifacts/build.py::read_result_rows`` globs the latter into
``ladder.json``, so a file there *becomes* a ladder row whatever it says inside. Rung 8 is a
hypothesis test with a null, not a calibrated probability; it has no PR-AUC, no savings and
no capacity K, and it makes no claim on any of them.

    uv run python scripts/rung8_score.py --part null
    uv run python scripts/rung8_score.py --part baf
    uv run python scripts/rung8_score.py --part val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
# gates_report is a named module beside tests/gates/conftest.py, not a package: pytest puts
# it on sys.path via rootdir collection and a loose script has to do the same. Importing it
# is the point -- re-declaring `scenario()` here would let this run and G5 disagree about
# what "the same configuration G5 uses" means, which is the one thing #59 asks for by name.
sys.path.insert(0, str(ROOT / "tests" / "gates"))

from gates_report import GATE_SEED, scenario  # noqa: E402

from rakshak.eval.baf_adapter import BAF_ENV_VAR, baf_path, load_baf  # noqa: E402
from rakshak.eval.metrics import tpp_rescaled_ks  # noqa: E402
from rakshak.generator.confounders import build_layer  # noqa: E402
from rakshak.generator.engine import generate  # noqa: E402
from rakshak.models.rung8_tpp import (  # noqa: E402
    MIN_EVENTS,
    HawkesNbFit,
    compensator_increments,
    fit,
    nb_dispersion,
)

OUT_DIR = ROOT / "data" / "v2" / "rung8_tpp"
DATA_ROOT = ROOT / "data" / "v2"
ORIGIN = date(2026, 1, 1)
START = datetime(2026, 1, 1, tzinfo=UTC)

#: Days 300-364 are the test split. Nothing below reads past this. Prime Directive 1.
VAL_START_DAY = 240
VAL_END_DAY = 299

#: The merchant's post-onboarding baseline window, in simulation days. 30 and not more:
#: ``population.onset_window_min_day`` is 30, so days 0-29 are the longest stretch that is
#: guaranteed drift-free for EVERY merchant in the population. A longer baseline would fit
#: some merchants' fits to the drift they are supposed to detect.
BASELINE_DAYS = 30

#: Trailing window each epoch's KS test is computed over. Seven days is the shortest window
#: that holds enough inter-arrivals for a KS test on a typical L1 merchant (6 txn/day) while
#: still being short enough that "when did it change" means something.
EPOCH_WINDOW_DAYS = 7

#: Below this many inter-arrivals in the window there is no statistic, and the epoch is
#: recorded as *unavailable* rather than as a quiet PASS. A missing measurement counted as a
#: non-alert would drag every rate below toward zero for a reason that is not the detector.
MIN_INCREMENTS = 10

#: G5's headroom, imported in spirit and restated because this is not a pytest module:
#: ``tests/gates/test_g5_confounder_null.py::EXCESS_ALLOWED``. Same bar, same units.
EXCESS_ALLOWED = 0.02


def _days_since_start(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        t=(pl.col("event_time") - START).dt.total_nanoseconds().cast(pl.Float64) / 86_400e9
    )


def _per_merchant_times(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """``merchant_id -> sorted arrival times in days``, refunds excluded.

    Refunds are excluded because they are not arrivals: a refund is emitted by the
    generator at a latency after its capture, so counting it would put a second, purely
    mechanical point process on top of the one being fitted.
    """
    grouped = (
        _days_since_start(frame.filter(~pl.col("is_refund")))
        .select("merchant_id", "t")
        .sort("merchant_id", "t")
        .group_by("merchant_id", maintain_order=True)
        .agg(pl.col("t"))
    )
    return {
        m: np.asarray(t, dtype=np.float64)
        for m, t in zip(grouped["merchant_id"], grouped["t"].to_list(), strict=True)
    }


def _fit_baseline(times: np.ndarray, start: int = 0) -> HawkesNbFit | None:
    """Fit on days ``[start, start + BASELINE_DAYS)``, or ``None`` if the window is thin.

    ``start`` is 0 everywhere except the one diagnostic in ``run_val``: days 0-29 are the
    post-onboarding baseline #59 specifies and the longest stretch guaranteed drift-free for
    every merchant. The diagnostic re-fits on a *recent* window instead, which attributes
    the test's realised size between two candidate causes rather than leaving it unattributed
    — see that call site. It is a diagnostic, not a second configuration to report as the
    result, and the headline number is always ``start = 0``.
    """
    window = times[(times >= start) & (times < start + BASELINE_DAYS)] - start
    if window.size < MIN_EVENTS:
        return None
    counts = np.bincount(window.astype(np.int64), minlength=BASELINE_DAYS)
    return fit(window, horizon_days=float(BASELINE_DAYS), daily_counts=counts)


def _epoch_p_values(
    times: np.ndarray, fitted: HawkesNbFit, first_day: int, last_day: int
) -> np.ndarray:
    """One KS p-value per day in ``[first_day, last_day]``; NaN where unavailable.

    The compensator is computed **once** over the whole scored stretch and then sliced by
    day, rather than re-integrated per epoch. Both give the same increments; only the second
    costs O(days x events).
    """
    scored = times[(times >= first_day - EPOCH_WINDOW_DAYS) & (times <= last_day + 1)]
    out = np.full(last_day - first_day + 1, np.nan)
    if scored.size < 2:
        return out
    increments = compensator_increments(scored, fitted)
    right = scored[1:]  # each increment is attributed to the event that closes it
    for i, day in enumerate(range(first_day, last_day + 1)):
        lo, hi = np.searchsorted(right, [day + 1.0 - EPOCH_WINDOW_DAYS, day + 1.0])
        if hi - lo < MIN_INCREMENTS:
            continue
        out[i] = tpp_rescaled_ks(increments[lo:hi]).p_value
    return out


def _alert_rate(p_values: np.ndarray, threshold: float, days: np.ndarray) -> float:
    window = p_values[:, days]
    valid = np.isfinite(window)
    if not valid.any():
        return float("nan")
    return float((window[valid] < threshold).mean())


def run_null(n_merchants: int, echo: Any) -> dict[str, Any]:
    """Mitigation 1 — the ``prevalence = 0`` null with confounders on. G5's own bar."""
    config = scenario(prevalence=0.0, n_merchants=n_merchants)
    n_days = config.population.n_days
    nominal = config.capacity.analyst_reviews_per_day / config.capacity.per_n_merchants
    layer = build_layer(config, np.zeros(4, dtype=np.int64), np.full(4, 6.0), np.full(4, 0.5))
    windows = layer.windows

    started = time.perf_counter()
    # GATE_SEED + 1 is conftest.py::null_data's seed. Same population G5 measures.
    data = generate(config, np.random.default_rng(GATE_SEED + 1))
    assert data.ground_truth["risk_typology_id"].null_count() == data.ground_truth.height, (
        "the null run is not null: some merchant carries a risk typology"
    )
    times_by_merchant = _per_merchant_times(data.transactions)
    echo(f"generated {len(times_by_merchant)} merchants x {n_days} days "
         f"in {time.perf_counter() - started:.1f}s")

    first_day = BASELINE_DAYS + EPOCH_WINDOW_DAYS
    last_day = n_days - 1
    rows: list[np.ndarray] = []
    fanos: list[float] = []
    started = time.perf_counter()
    for times in times_by_merchant.values():
        fitted = _fit_baseline(times)
        if fitted is None:
            continue
        fanos.append(fitted.nb_fano)
        rows.append(_epoch_p_values(times, fitted, first_day, last_day))
    p_values = np.full((len(rows), n_days), np.nan)
    if rows:
        p_values[:, first_day : last_day + 1] = np.vstack(rows)
    echo(f"fitted and scored {len(rows)} merchants in {time.perf_counter() - started:.1f}s")

    busy = np.zeros(n_days, dtype=bool)
    for window in windows:
        busy[window.start_day : window.end_day] = True
    quiet = np.flatnonzero(~busy & (np.arange(n_days) >= first_day))

    # Calibrate on the quiet stretch only. A threshold fitted to the whole series has
    # already absorbed the confounder spikes into its own definition of normal --
    # test_g5_confounder_null.py::calibrate makes the same argument for the same reason.
    quiet_p = p_values[:, quiet]
    quiet_p = quiet_p[np.isfinite(quiet_p)]
    threshold = float(np.quantile(quiet_p, nominal))
    quiet_rate = _alert_rate(p_values, threshold, quiet)

    excesses: list[dict[str, Any]] = []
    worst = 0.0
    for window in windows:
        days = np.arange(window.start_day, window.end_day)
        days = days[days >= first_day]
        if days.size == 0:
            continue
        rate = _alert_rate(p_values, threshold, days)
        excess = rate - nominal
        worst = max(worst, excess)
        excesses.append(
            {
                "confounder": window.confounder.value,
                "start_day": window.start_day,
                "end_day": window.end_day,
                "feature": window.feature,
                "days_measured": int(days.size),
                "alert_rate": rate,
                "excess_pp": excess * 100.0,
                "verdict": "GREEN" if excess <= EXCESS_ALLOWED else "RED",
            }
        )
        echo(f"  {window.confounder.value:<28} alert {rate:.4f} vs nominal {nominal:.4f} "
             f"({excess * 100:+.2f}pp) {'GREEN' if excess <= EXCESS_ALLOWED else 'RED'}")

    verdict = "GREEN" if worst <= EXCESS_ALLOWED else "RED"
    echo(f"NULL RUN {verdict}: worst window excess {worst * 100:+.2f}pp, "
         f"quiet-day rate {quiet_rate:.4f} against nominal {nominal:.4f}")
    return {
        "mitigation": "1 — null distribution at prevalence=0 with confounders on",
        "configuration": "tests/gates/gates_report.scenario(prevalence=0.0), seed GATE_SEED+1",
        "n_merchants_generated": config.population.n_merchants,
        "n_merchants_fitted": len(rows),
        "n_days": n_days,
        "baseline_days": BASELINE_DAYS,
        "epoch_window_days": EPOCH_WINDOW_DAYS,
        "nominal_alert_rate": nominal,
        "p_value_threshold": threshold,
        "quiet_day_alert_rate": quiet_rate,
        "excess_allowed_pp": EXCESS_ALLOWED * 100.0,
        "median_baseline_nb_fano": float(np.median(fanos)) if fanos else float("nan"),
        "window_excess": excesses,
        "worst_excess_pp": worst * 100.0,
        "verdict": verdict,
    }


def run_baf(echo: Any) -> dict[str, Any]:
    """Mitigation 2 — test-size calibration against BAF through the existing adapter.

    BAF has no timestamps and no per-entity event sequences (``baf_adapter``'s own
    docstring: "One row per application. No amount, no timestamp, no payer, no merchant, no
    sequences"), so the strongest honest check available is on the *count* analogues: fit
    the NB layer to a BAF count column and compare its dispersion against the generator's.
    That is a calibration of the background model, not of the point process, and it is
    reported as the narrower thing it is.

    None of it runs when BAF is absent, which is the shipped state: SKIP, with the reason.
    """
    path = baf_path()
    if path is None:
        echo(f"BAF NOT VENDORED — set {BAF_ENV_VAR} to enable. Criterion NOT met.")
        return {
            "mitigation": "2 — test-size calibration against BAF",
            "verdict": "SKIP",
            "criterion_met": False,
            "reason": (
                "BAF is licensed CC BY-NC-SA 4.0 and is deliberately not vendored "
                "(eval/baf_adapter.py: 'BAF is not vendored and must not be'). Four gates "
                f"already record SKIP for the same reason. Set {BAF_ENV_VAR} to a baf.zip "
                "or an extracted Base.csv/Base.parquet to run this. A SKIP is an UNMET "
                "acceptance criterion and is reported as one, never as a pass."
            ),
        }
    frame = load_baf(["zip_count_4w", "bank_branch_count_8w"])
    assert frame is not None
    columns = {
        column: nb_dispersion(frame[column].to_numpy().astype(np.float64))
        for column in frame.columns
    }
    echo(f"BAF at {path}: " + ", ".join(f"{c} Fano={f:.2f}" for c, (_, f) in columns.items()))
    return {
        "mitigation": "2 — test-size calibration against BAF",
        "verdict": "MEASURED",
        "criterion_met": True,
        "baf_path": str(path),
        "count_dispersion": {c: {"r": r, "fano": f} for c, (r, f) in columns.items()},
        "scope_caveat": (
            "BAF has no event timestamps, so this calibrates the NB background's dispersion "
            "only. The time-rescaling test's size cannot be measured on BAF at all."
        ),
    }


def _val_merchants(sample: int, rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """VAL-fold merchants: every one with a drift onset, plus a random sample of the rest.

    ``_merchant_fold`` is ``rakshak.score_rung7``'s, so Rung 8's VAL merchants are the same
    merchants every other rung's are. ``drift_onset_at`` is read HERE, on the eval side,
    exactly as ``cli.py::_build_truth`` and ``score_rung7`` do — never inside ``models/``.
    """
    from rakshak.score_rung7 import _merchant_fold

    truth = pl.read_parquet(DATA_ROOT / "ground_truth.parquet").select(
        "merchant_id",
        pl.when(pl.col("drift_onset_at").is_not_null())
        .then((pl.col("drift_onset_at").dt.date() - ORIGIN).dt.total_days())
        .otherwise(None)
        .cast(pl.Float64)
        .alias("onset_day"),
    )
    fold = truth["merchant_id"].map_elements(_merchant_fold, return_dtype=pl.String)
    val = truth.filter(fold == "val")
    drifted = val.filter(pl.col("onset_day").is_not_null() & (pl.col("onset_day") <= VAL_END_DAY))
    clean = val.filter(pl.col("onset_day").is_null())
    take = min(max(sample - drifted.height, 0), clean.height)
    picked = rng.choice(clean["merchant_id"].to_numpy(), size=take, replace=False).tolist()
    ids = drifted["merchant_id"].to_list() + picked
    onsets = np.concatenate([drifted["onset_day"].to_numpy(), np.full(take, np.nan)])
    return ids, onsets


def _val_statistics(
    ids: list[str],
    onsets: np.ndarray,
    times_by_merchant: dict[str, np.ndarray],
    baseline_start: int,
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    p_values: list[float] = []
    labels: list[int] = []
    fanos: list[float] = []
    n_skipped = 0
    for merchant, onset in zip(ids, onsets, strict=True):
        times = times_by_merchant.get(merchant)
        if times is None:
            n_skipped += 1
            continue
        fitted = _fit_baseline(times, baseline_start)
        window = times[(times >= VAL_START_DAY) & (times <= VAL_END_DAY + 1)]
        if fitted is None or window.size < MIN_INCREMENTS + 1:
            n_skipped += 1
            continue
        fanos.append(fitted.nb_fano)
        p_values.append(tpp_rescaled_ks(compensator_increments(window, fitted)).p_value)
        labels.append(int(np.isfinite(onset)))
    return np.asarray(p_values), np.asarray(labels), fanos, n_skipped


def _summarise(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    return {
        "rejection_rate_non_drifted_at_0.05": (
            float((p[y == 0] < 0.05).mean()) if (y == 0).any() else float("nan")
        ),
        "rejection_rate_drifted_at_0.05": (
            float((p[y == 1] < 0.05).mean()) if (y == 1).any() else float("nan")
        ),
        "roc_auc_neg_log10_p": (
            float(roc_auc_score(y, -np.log10(np.maximum(p, 1e-300))))
            if 0 < int(y.sum()) < y.size
            else float("nan")
        ),
    }


def run_val(sample: int, echo: Any) -> dict[str, Any]:
    """Part 3 — the statistic on the real cycle-4 dataset, VALIDATION fold only."""
    rng = np.random.default_rng(20260902)
    ids, onsets = _val_merchants(sample, rng)
    echo(f"VAL merchants: {int(np.isfinite(onsets).sum())} drifted + "
         f"{int(np.isnan(onsets).sum())} clean = {len(ids)}")

    started = time.perf_counter()
    frame = (
        pl.scan_parquet(DATA_ROOT / "transactions.parquet")
        .filter(pl.col("merchant_id").is_in(ids))
        .filter(pl.col("event_date") <= ORIGIN + timedelta(days=VAL_END_DAY))
        .select("merchant_id", "event_time", "is_refund")
        .collect()
    )
    echo(f"materialised {frame.height} rows in {time.perf_counter() - started:.1f}s "
         f"(days 0-{VAL_END_DAY}; the test split was not read)")
    times_by_merchant = _per_merchant_times(frame)

    p, y, fanos, n_skipped = _val_statistics(ids, onsets, times_by_merchant, 0)
    headline = _summarise(p, y)
    echo(f"scored {p.size} merchants ({int(y.sum())} drifted), {n_skipped} skipped")
    for key, value in headline.items():
        echo(f"  {key:<38} {value:.4f}")

    # The diagnostic. The headline fit is 210 days older than the window it scores, so a
    # bad realised size has two candidate causes that cannot be told apart from it: the NB
    # day-to-day multiplier the intensity structurally cannot hold, and 210 days of ordinary
    # persona non-stationarity (an L3 growth ramp, an L2 sale window, the day-of-week
    # factors, P6's macro sinusoid). Re-fitting on days 210-239 removes the second and
    # leaves the first. It is NOT a rescue and is not the reported result: on a drifted
    # merchant the recent window can already contain the drift, so its power is optimistic
    # in a way the headline's is not.
    recent_start = VAL_START_DAY - BASELINE_DAYS
    p_recent, y_recent, _, _ = _val_statistics(ids, onsets, times_by_merchant, recent_start)
    recent = _summarise(p_recent, y_recent)
    echo(f"diagnostic, baseline re-fit on days {recent_start}-{VAL_START_DAY - 1}:")
    for key, value in recent.items():
        echo(f"  {key:<38} {value:.4f}")

    return {
        "part": "3 - validation-fold measurement on the cycle-4 dataset",
        "split": "val",
        "window_days": [VAL_START_DAY, VAL_END_DAY],
        "test_split_read": False,
        "n_scored": int(p.size),
        "n_drifted": int(y.sum()),
        "n_skipped": n_skipped,
        "baseline_days": BASELINE_DAYS,
        "median_baseline_nb_fano": float(np.median(fanos)) if fanos else float("nan"),
        **headline,
        "diagnostic_recent_baseline": {
            "baseline_window_days": [recent_start, VAL_START_DAY - 1],
            "purpose": (
                "Attribution only. Separates 210 days of ordinary non-stationarity from the "
                "NB day-to-day multiplier the conditional intensity cannot represent. Its "
                "power is optimistic - a recent baseline can already contain the drift."
            ),
            **recent,
        },
        "size_note": (
            "A calibrated test rejects 5% of non-drifted merchants at level 0.05. Whatever "
            "the number above is, it is the test's realised size on this population, and "
            "the gap between it and 0.05 is model misspecification, not fraud."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("null", "baf", "val", "all"), default="all")
    parser.add_argument("--merchants", type=int, default=1_200, help="null-run population")
    parser.add_argument("--sample", type=int, default=600, help="val-fold merchants scored")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact": "rung8_tpp",
        "rung": 8,
        "label": "tpp_hawkes_nb",
        "adopted": False,
        "not_a_ladder_row": (
            "Rung 8 is a goodness-of-fit HYPOTHESIS TEST, not a calibrated probability. It "
            "has no PR-AUC, no savings, no precision@K and no capacity K, and makes no claim "
            "on any of them. Written here and NOT to data/v2/eval/, which "
            "artifacts/build.py::read_result_rows globs into ladder.json."
        ),
        "model": "lambda(t) = mu*s(t) + sum alpha*beta*exp(-beta(t-t_i)); L-BFGS-B, analytic jac",
        "dependencies_added": "none — scipy was already pinned. No autograd, no GPU.",
        "acceptance_criterion_1": (
            "Measured in tests/unit/test_rung8.py, not here: a correctly specified fit is "
            "not rejected by the rescaling test. Run `uv run pytest tests/unit/test_rung8.py`."
        ),
    }
    if args.part in ("null", "all"):
        payload["null_run"] = run_null(args.merchants, print)
    if args.part in ("baf", "all"):
        payload["baf"] = run_baf(print)
    if args.part in ("val", "all"):
        payload["validation"] = run_val(args.sample, print)

    path = OUT_DIR / f"rung8_tpp_{args.part}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
