"""Rung 8b's runner: the same three mitigations Rung 8 ran, with the neural intensity.

GitHub #66 is a *comparison* ticket, not a win ticket. It asks whether replacing Rung 8's
parametric Hawkes/NB intensity with a neural one fixes the circularity finding in
``LIMITATIONS.md`` §12 — and it predicts, in its own text, that it will not: *"a flexible
neural intensity fits the generator's own process even more exactly than a parametric one
does, which makes the circularity objection in #125 worse, not better."* So every number
here is produced by running **the parametric rung's own measurement code** with one thing
swapped, and every number has the parametric result printed beside it.

That is why this file imports from ``scripts/rung8_score.py`` rather than restating it::

    scenario / GATE_SEED       the null population, via rung8_score's own gates_report import
    BASELINE_DAYS, EPOCH_WINDOW_DAYS, MIN_INCREMENTS, EXCESS_ALLOWED, VAL_* the constants
    _per_merchant_times        refund filtering and the day-scale conversion
    _val_merchants             the VAL fold, so both rungs score the identical merchants
    _alert_rate / _summarise   the alert and rejection-rate arithmetic
    run_baf                    mitigation 2, unchanged -- see below

#59 asked for ``scenario()`` to be imported rather than re-declared "so the two cannot drift
apart". The same argument applies with more force to a rung whose entire deliverable is a
comparison: a helper copied here and edited later would make the two columns incomparable
while both still ran green. What is **not** imported is the four functions that bind
``rung8_tpp.fit`` at import time; those are restated below with the neural fit substituted
and nothing else changed, and the threshold-calibration block in ``run_null`` is deliberately
line-for-line identical to the parametric one.

The four measurements, in the order the ticket asks for them:

1. ``--part sim``  — acceptance criterion 1 and #66's criterion 4 in one run: fit **both**
   rungs to the same correctly-specified simulated Hawkes and report both KS statistics.
   "Demonstrably improve goodness-of-fit calibration, not just model capacity" is a
   comparison, so it is measured as one.
2. ``--part null`` — mitigation 1. ``prevalence = 0`` with confounders on, G5's own bar.
   The parametric rung failed this at **+6.61pp** against a +2pp allowance.
3. ``--part baf``  — mitigation 2. **Structurally unavailable to any implementation**: BAF
   is CC BY-NC-SA and deliberately not vendored, which is why ``make gates`` reports 4
   skips. Recorded as UNMET, exactly as Rung 8 recorded it. Not substituted, not faked.
4. ``--part val``  — the empirical test size on the real cycle-4 **validation** fold. The
   parametric rung rejects **83.65%** of merchants that never drifted at a nominal 0.05.

Days 300-364 are never read: ``VAL_END_DAY`` bounds every scan, the test split stays shut
and ``open_count`` stays 0. No ``EVAL-LOCK`` module is imported, hashed or touched.

**This writes to ``data/v2/rung8b_neural/``, NOT ``data/v2/eval/``.** Rung 8's runner records
why in the same words: ``artifacts/build.py::read_result_rows`` globs the latter into
``ladder.json``, so a file there *becomes* a ladder row whatever it says inside. Rung 8b is a
hypothesis test with a null, not a calibrated probability.

    uv run python scripts/rung8b_score.py --part sim
    uv run python scripts/rung8b_score.py --part null
    uv run python scripts/rung8b_score.py --part baf
    uv run python scripts/rung8b_score.py --part val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
# scripts/ is sys.path[0] only when this file is run directly, and tests/gates is never on it;
# both are named here so `uv run python scripts/rung8b_score.py` and an import of this module
# resolve the same way. See the module docstring for why these two are imported at all.
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "gates"))

# The same two lines rung8_score.py uses, so `scenario` and `GATE_SEED` are the SAME module
# objects G5's conftest holds rather than a second copy of the configuration. #59 asked for
# them to be imported rather than re-declared; a comparison rung has the same need twice over.
from gates_report import GATE_SEED, scenario  # noqa: E402
from rung8_score import (  # noqa: E402
    BASELINE_DAYS,
    DATA_ROOT,
    EPOCH_WINDOW_DAYS,
    EXCESS_ALLOWED,
    MIN_INCREMENTS,
    ORIGIN,
    VAL_END_DAY,
    VAL_START_DAY,
    _alert_rate,
    _per_merchant_times,
    _summarise,
    _val_merchants,
    run_baf,
)

from rakshak.eval.metrics import tpp_rescaled_ks  # noqa: E402
from rakshak.generator.confounders import build_layer  # noqa: E402
from rakshak.generator.engine import generate  # noqa: E402
from rakshak.models import rung8_tpp  # noqa: E402
from rakshak.models.rung8b_neural import (  # noqa: E402
    EPOCHS,
    LEARNING_RATE,
    N_PARAMETERS,
    WEIGHT_DECAY,
    NeuralIntensityFit,
    compensator_increments,
    fit,
)

OUT_DIR = ROOT / "data" / "v2" / "rung8b_neural"

#: The parametric rung's published results, quoted here so every line of output carries its
#: own baseline. Sources: LIMITATIONS.md §12 and data/v2/rung8_tpp/*.json.
PARAMETRIC = {
    "sim_ks_statistic": 0.0106,
    "sim_ks_p_value": 0.799,
    "null_worst_excess_pp": 6.61,
    "null_threshold": 1.085054589895893e-92,
    "val_rejection_rate_non_drifted_at_0.05": 0.8365019011406845,
    "val_rejection_rate_drifted_at_0.05": 0.9166666666666666,
    "val_roc_auc_neg_log10_p": 0.8013624841571608,
    "n_parameters": 3,
}


#: Epoch budget for the fits in this run. Set by ``--epochs``, whose ONLY legitimate use is
#: the convergence diagnostic described at ``run_sim`` — the headline for every mitigation is
#: the budget declared in ``rung8b_neural.EPOCHS`` before any of them ran. An artefact written
#: at a non-default value carries ``epochs`` in its payload and must be read as a diagnostic.
_EPOCHS = EPOCHS


def _fit_baseline(times: np.ndarray, start: int = 0) -> NeuralIntensityFit | None:
    """``rung8_score._fit_baseline`` with the neural fit substituted. Same window, same rule."""
    window = times[(times >= start) & (times < start + BASELINE_DAYS)] - start
    if window.size < rung8_tpp.MIN_EVENTS:
        return None
    counts = np.bincount(window.astype(np.int64), minlength=BASELINE_DAYS)
    return fit(window, horizon_days=float(BASELINE_DAYS), daily_counts=counts, epochs=_EPOCHS)


def _epoch_p_values(
    times: np.ndarray, fitted: NeuralIntensityFit, first_day: int, last_day: int
) -> np.ndarray:
    """One KS p-value per day in ``[first_day, last_day]``; NaN where unavailable.

    Identical to ``rung8_score._epoch_p_values`` including the one-pass compensator and the
    attribution of each increment to the event that closes it. Only the ``fitted`` type
    differs, and both types answer the same ``compensator_increments`` contract.
    """
    scored = times[(times >= first_day - EPOCH_WINDOW_DAYS) & (times <= last_day + 1)]
    out = np.full(last_day - first_day + 1, np.nan)
    if scored.size < 2:
        return out
    increments = compensator_increments(scored, fitted)
    right = scored[1:]
    for i, day in enumerate(range(first_day, last_day + 1)):
        lo, hi = np.searchsorted(right, [day + 1.0 - EPOCH_WINDOW_DAYS, day + 1.0])
        if hi - lo < MIN_INCREMENTS:
            continue
        out[i] = tpp_rescaled_ks(increments[lo:hi]).p_value
    return out


def run_sim(echo: Any) -> dict[str, Any]:
    """Criterion 1 and #66's criterion 4: both rungs, one simulated process, two KS numbers.

    The process is ``generator.arrivals.hawkes_overlay``'s own branching construction at the
    manifest's kernel — the same simulator, seed and horizon
    ``tests/unit/test_rung8.py`` uses, so the parametric column here reproduces the published
    KS 0.0106 / p 0.799 rather than approximating it.

    **A correctly-specified parametric fit is the ceiling on this data, not the floor.** The
    Hawkes model *is* the generating process; the neural intensity can at best match it and
    is being asked to spend 209 parameters where 3 were exactly right. Beating it here would
    be evidence of overfitting the realisation, not of better calibration, which is why the
    ticket asks for calibration on the *validation* fold as well.
    """
    sys.path.insert(0, str(ROOT / "tests" / "unit"))
    from test_rung8 import FLAT, simulate

    rng = np.random.default_rng(20260902)
    horizon = 120.0
    times = simulate(rng, mu=20.0, alpha=0.30, horizon=horizon)
    echo(f"simulated Hawkes: {times.size} events over {horizon:.0f} days "
         f"(mu 20/day, alpha 0.30, beta 480/day)")

    started = time.perf_counter()
    parametric = rung8_tpp.fit(times, horizon_days=horizon, shape=FLAT)
    ks_p = tpp_rescaled_ks(rung8_tpp.compensator_increments(times, parametric))
    echo(f"  parametric (3 params, {time.perf_counter() - started:.1f}s): "
         f"KS={ks_p.statistic:.4f} p={ks_p.p_value:.4f} n={ks_p.n}")

    started = time.perf_counter()
    neural = fit(times, horizon_days=horizon)
    ks_n = tpp_rescaled_ks(compensator_increments(times, neural))
    echo(f"  neural ({N_PARAMETERS} params, {time.perf_counter() - started:.1f}s): "
         f"KS={ks_n.statistic:.4f} p={ks_n.p_value:.4f} n={ks_n.n}")

    # Convergence diagnostic, and it is a diagnostic rather than a second configuration to
    # report. If the neural fit loses on KS, the first objection is "you under-trained it",
    # so the same fit is re-run at 3x the declared epoch budget and the number is written
    # down. The HEADLINE stays at the declared EPOCHS -- Prime Directive 5 forbids re-choosing
    # a setting after seeing a result, and this exists to bound the objection, not to answer
    # it in the rung's favour.
    long_run = fit(times, horizon_days=horizon, epochs=EPOCHS * 3)
    ks_long = tpp_rescaled_ks(compensator_increments(times, long_run))
    echo(f"  diagnostic, {EPOCHS * 3} epochs: KS={ks_long.statistic:.4f} p={ks_long.p_value:.4f}")

    better = ks_n.statistic < ks_p.statistic
    echo(f"CRITERION 1 {'PASS' if not ks_n.rejects_at(0.05) else 'FAIL'}; "
         f"KS calibration vs parametric: {'better' if better else 'worse'}")
    return {
        "criterion": "1 - a correctly specified fit is not rejected; and #66 criterion 4",
        "process": "generator.arrivals.hawkes_overlay, mu 20/day, alpha 0.30, beta 480/day",
        "seed": 20260902,
        "horizon_days": horizon,
        "n_events": int(times.size),
        "parametric": {
            "n_parameters": 3,
            "ks_statistic": ks_p.statistic,
            "ks_p_value": ks_p.p_value,
            "rejected_at_0.05": ks_p.rejects_at(0.05),
            "recovered": {"mu": parametric.mu, "alpha": parametric.alpha, "beta": parametric.beta},
        },
        "neural": {
            "n_parameters": N_PARAMETERS,
            "ks_statistic": ks_n.statistic,
            "ks_p_value": ks_n.p_value,
            "rejected_at_0.05": ks_n.rejects_at(0.05),
            "final_loglik": neural.loglik,
        },
        "diagnostic_longer_training": {
            "epochs": EPOCHS * 3,
            "purpose": (
                "Bounds the 'it was under-trained' objection to the headline. NOT the "
                "reported configuration: the epoch budget was declared before any "
                "mitigation ran and is not re-chosen after seeing a result."
            ),
            "ks_statistic": ks_long.statistic,
            "ks_p_value": ks_long.p_value,
            "final_loglik": long_run.loglik,
        },
        "criterion_1_met": not ks_n.rejects_at(0.05),
        "ks_better_than_parametric": better,
        "ceiling_note": (
            "The parametric model IS the generating process here, so its KS is a ceiling "
            "and not a floor. A lower neural KS on the fitted realisation is evidence of "
            "capacity spent on that realisation, not of better calibration; the calibration "
            "claim is settled by --part null and --part val, not here."
        ),
    }


def run_null(n_merchants: int, echo: Any) -> dict[str, Any]:
    """Mitigation 1 — the ``prevalence = 0`` null with confounders on. G5's own bar.

    The population, the seed, the baseline window, the epoch window, the quiet-day threshold
    calibration and the +2pp allowance are all the parametric rung's. Only the intensity
    changed.
    """
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
    for i, times in enumerate(times_by_merchant.values()):
        fitted = _fit_baseline(times)
        if fitted is None:
            continue
        fanos.append(fitted.nb_fano)
        rows.append(_epoch_p_values(times, fitted, first_day, last_day))
        if i % 100 == 0:
            echo(f"  fitted {len(rows)} of {i + 1} seen, "
                 f"{time.perf_counter() - started:.0f}s elapsed")
    p_values = np.full((len(rows), n_days), np.nan)
    if rows:
        p_values[:, first_day : last_day + 1] = np.vstack(rows)
    echo(f"fitted and scored {len(rows)} merchants in {time.perf_counter() - started:.1f}s")

    busy = np.zeros(n_days, dtype=bool)
    for window in windows:
        busy[window.start_day : window.end_day] = True
    quiet = np.flatnonzero(~busy & (np.arange(n_days) >= first_day))

    # Calibrate on the quiet stretch only -- rung8_score.run_null's block, unchanged. A
    # threshold fitted to the whole series has already absorbed the confounder spikes into
    # its own definition of normal.
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
    echo(f"NULL RUN {verdict}: worst window excess {worst * 100:+.2f}pp "
         f"(parametric: {PARAMETRIC['null_worst_excess_pp']:+.2f}pp), "
         f"quiet-day rate {quiet_rate:.4f} against nominal {nominal:.4f}")
    echo(f"threshold to hold nominal on a fraud-free population: p < {threshold:.3e} "
         f"(parametric: {PARAMETRIC['null_threshold']:.3e})")
    return {
        "mitigation": "1 — null distribution at prevalence=0 with confounders on",
        "configuration": "tests/gates/gates_report.scenario(prevalence=0.0), seed GATE_SEED+1",
        "imported_from": "scripts/rung8_score.py — same scenario, constants and arithmetic",
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
        "parametric_worst_excess_pp": PARAMETRIC["null_worst_excess_pp"],
        "parametric_p_value_threshold": PARAMETRIC["null_threshold"],
    }


def _val_statistics(
    ids: list[str],
    onsets: np.ndarray,
    times_by_merchant: dict[str, np.ndarray],
    baseline_start: int,
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    """``rung8_score._val_statistics`` with the neural fit substituted."""
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


def run_val(sample: int, echo: Any) -> dict[str, Any]:
    """The statistic on the real cycle-4 dataset, VALIDATION fold only.

    ``_val_merchants`` is imported, so the merchants, the sampling seed and the fold
    assignment are the parametric rung's. The number to beat is its realised size: it
    rejects 0.8365 of merchants that never drifted at a nominal 0.05.
    """
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

    started = time.perf_counter()
    p, y, fanos, n_skipped = _val_statistics(ids, onsets, times_by_merchant, 0)
    headline = _summarise(p, y)
    echo(f"scored {p.size} merchants ({int(y.sum())} drifted), {n_skipped} skipped, "
         f"{time.perf_counter() - started:.0f}s")
    for key, value in headline.items():
        echo(f"  {key:<38} {value:.4f}  (parametric {PARAMETRIC['val_' + key]:.4f})")

    # The same attribution diagnostic Rung 8 ran, for the same reason and with the same
    # caveat: on a drifted merchant a recent baseline can already contain the drift, so its
    # power is optimistic. It is not the reported result; the headline is always start = 0.
    recent_start = VAL_START_DAY - BASELINE_DAYS
    p_recent, y_recent, _, _ = _val_statistics(ids, onsets, times_by_merchant, recent_start)
    recent = _summarise(p_recent, y_recent)
    echo(f"diagnostic, baseline re-fit on days {recent_start}-{VAL_START_DAY - 1}:")
    for key, value in recent.items():
        echo(f"  {key:<38} {value:.4f}")

    return {
        "part": "validation-fold measurement on the cycle-4 dataset",
        "split": "val",
        "window_days": [VAL_START_DAY, VAL_END_DAY],
        "test_split_read": False,
        "n_scored": int(p.size),
        "n_drifted": int(y.sum()),
        "n_skipped": n_skipped,
        "baseline_days": BASELINE_DAYS,
        "median_baseline_nb_fano": float(np.median(fanos)) if fanos else float("nan"),
        **headline,
        "parametric": {k[4:]: v for k, v in PARAMETRIC.items() if k.startswith("val_")},
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
    parser.add_argument("--part", choices=("sim", "null", "baf", "val", "all"), default="all")
    parser.add_argument("--merchants", type=int, default=1_200, help="null-run population")
    parser.add_argument("--sample", type=int, default=600, help="val-fold merchants scored")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="DIAGNOSTIC ONLY. The headline budget is the declared EPOCHS; see run_sim.",
    )
    args = parser.parse_args()
    global _EPOCHS  # noqa: PLW0603
    _EPOCHS = int(args.epochs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact": "rung8b_neural",
        "rung": 8,
        "label": "tpp_neural_intensity",
        "adopted": False,
        "baseline": "rung8_tpp (T-0125, GitHub #59) — LIMITATIONS.md §12",
        "not_a_ladder_row": (
            "Rung 8b is a goodness-of-fit HYPOTHESIS TEST, not a calibrated probability. It "
            "has no PR-AUC, no savings, no precision@K and no capacity K, and makes no claim "
            "on any of them. Written here and NOT to data/v2/eval/, which "
            "artifacts/build.py::read_result_rows globs into ladder.json."
        ),
        "model": (
            "Lambda(tau|h) = Phi(tau,h) - Phi(0,h), a monotone cumulative-hazard network "
            "(Omi et al., NeurIPS 2019) over a six-timescale closed-form excitation memory; "
            "lambda = dLambda/dtau by torch.autograd. Adam, CPU float64."
        ),
        "dependencies_added": (
            "torch==2.14.0+cpu, admitted for this rung by the 2026-09-02 AMENDMENT to "
            "docs/adr/ADR-V3-001-no-autograd.md. CPU-only; torch.cuda is never touched."
        ),
        "optimiser": {
            "epochs": _EPOCHS,
            "declared_epochs": EPOCHS,
            "is_diagnostic_run": _EPOCHS != EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "declared_before_any_mitigation_ran": True,
        },
        "n_parameters": N_PARAMETERS,
        "n_parameters_parametric": PARAMETRIC["n_parameters"],
        "adr_trainable_positives": 234,
        "adoption_gate": (
            "ADR-V3-001 §AMENDMENT: adopted only if ALL THREE of T-0125's mitigations pass "
            "with the neural intensity AND its goodness-of-fit calibration is demonstrably "
            "better on the same time-rescaling KS framing. Mitigation 2 is structurally "
            "unavailable to any implementation (BAF is not vendored), so the gate is "
            "UNREACHABLE BY CONSTRUCTION. That is reported, not worked around."
        ),
    }
    if args.part in ("sim", "all"):
        payload["simulated_recovery"] = run_sim(print)
    if args.part in ("null", "all"):
        payload["null_run"] = run_null(args.merchants, print)
    if args.part in ("baf", "all"):
        # Mitigation 2, verbatim from the parametric runner. BAF is not vendored, so this
        # records SKIP with the reason and the enabling environment variable. A SKIP is an
        # UNMET acceptance criterion and is reported as one, never as a pass, and no
        # substitute anchor is invented to fill the hole.
        payload["baf"] = run_baf(print)
    if args.part in ("val", "all"):
        payload["validation"] = run_val(args.sample, print)

    suffix = "" if _EPOCHS == EPOCHS else f"_epochs{_EPOCHS}"
    path = OUT_DIR / f"rung8b_neural_{args.part}{suffix}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
