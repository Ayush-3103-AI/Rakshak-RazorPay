"""G5 — the confounder null. Does the detector alert when the *platform* moves?

08-generator-v2-spec.md §7 calls this "the gate worth building the demo around", and it
is the one that separates a merchant sentinel from a drift detector with a fraud
detector's job title.

The run is at ``prevalence = 0`` with confounders on, so **every alert is by construction
a false positive**. GREEN when the alert rate inside each of the six confounder windows
stays within the nominal FPR + 2 percentage points.

Two detectors are measured side by side, because the comparison is the actual v2
hypothesis (charter K-1):

- **raw** — z of daily transaction count against the merchant's own trailing baseline,
  which is what ``v_txn_count_z`` is;
- **cohort-residual** — the same z minus the population median of that z on the same day,
  which is what the cohort-residual layer is *for*. When the whole platform moves, the
  residual should stay near zero.

**This gate records rather than asserts, and that is deliberate.** The ticket's Done-when
requires G3 and G4 to be green and requires G1, G2 and G5 to *report their statistic even
if red*. A RED here is charter K-4 firing — a reported negative finding about the
system's central claim — not a build error, and turning it into a failing test would make
``make gates`` unable to run at exactly the moment its output matters most. What is
asserted is that the measurement itself is sound: the threshold really was calibrated to
the nominal rate on quiet days, and there really is no fraud in the population.

Re-run in T-151 against Rungs 2 and 3 to produce the headline figure (10-eval-harness-
spec.md §7): alert rate over time, confounder windows shaded, raw line spiking inside
every band and residual line flat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from gates_report import GATE_DAYS, START, daily_counts, record, scenario

from rakshak.generator.confounders import ConfounderWindow, build_layer
from rakshak.generator.engine import GeneratedData

#: Percentage-point headroom above the nominal FPR (spec §7, G5).
EXCESS_ALLOWED = 0.02
#: Where the artefact generator looks for this gate's numbers: it is
#: ``rakshak.artifacts.build.DEFAULT_G5_PATH`` and nothing else is read. Until this file
#: exists the dashboard's lead figure is emitted as MISSING rather than fabricated, so the
#: gate owns the dump exactly as it owns the measurement.
G5_SERIES_PATH = Path(__file__).resolve().parents[2] / "data" / "v2" / "gates" / "g5_series.json"
#: The one observable both detectors read: ``trailing_z`` is a z of the daily transaction
#: count. A confounder window that moves *this* is an adversarial test of the null; a
#: window that moves a feature neither detector can see is a control, and an excess inside
#: it would be coincidence rather than the confounder. Which is which is a property of the
#: run -- of which detector was measured against which confounder -- which is why ``role``
#: travels in the artefact instead of being hardcoded in a dashboard component (#62).
DETECTOR_FEATURE = "txn_count"
#: Trailing window the baseline is estimated over. Matches the 28-day window the volume
#: features in 07-feature-register.md use (v_fano_trailing, g_payer_hhi).
BASELINE_DAYS = 28


def trailing_z(counts: np.ndarray, window: int = BASELINE_DAYS) -> np.ndarray:
    """z of each day against the merchant's own trailing ``window`` days, lagged by one.

    Point-in-time by construction: day ``d``'s baseline uses days ``[d-window, d-1]`` and
    never day ``d`` itself. A baseline that included today would shrink every real
    excursion toward zero and would make this gate pass for the wrong reason.
    """
    n, days = counts.shape
    padded = np.concatenate([np.zeros((n, 1)), np.cumsum(counts, axis=1)], axis=1)
    padded_sq = np.concatenate([np.zeros((n, 1)), np.cumsum(counts**2, axis=1)], axis=1)
    z = np.full((n, days), np.nan)
    for d in range(window + 1, days):
        lo, hi = d - window, d
        k = hi - lo
        total = padded[:, hi] - padded[:, lo]
        total_sq = padded_sq[:, hi] - padded_sq[:, lo]
        mean = total / k
        var = np.maximum(total_sq / k - mean**2, 0.0)
        sd = np.sqrt(var * k / max(k - 1, 1))
        z[:, d] = (counts[:, d] - mean) / np.maximum(sd, 1e-9)
    return z


def cohort_residual(z: np.ndarray) -> np.ndarray:
    """z minus the population median of z on the same day.

    The global backoff of the cohort defined in CLAUDE.md — ``(mcc_group, gmv_decile,
    vintage_bucket)`` backing off to ``mcc_group`` then global. The real cohort assignment
    is Lane B's (``features/cohort.py``), so this gate uses the global level, which is the
    *weakest* form of the idea. If even the global residual flattens the confounder
    windows, the finer cohort can only do better.
    """
    return z - np.nanmedian(z, axis=0, keepdims=True)


def alert_rate(z: np.ndarray, threshold: float, days: np.ndarray) -> float:
    window = z[:, days]
    valid = np.isfinite(window)
    if not valid.any():
        return float("nan")
    return float((window[valid] > threshold).mean())


def calibrate(z: np.ndarray, quiet_days: np.ndarray, nominal: float) -> float:
    """The z that produces exactly ``nominal`` alert rate on days with no platform event.

    Calibrating on the quiet stretch rather than on the whole series is the only way the
    gate means anything: a threshold fitted to the whole series has already absorbed the
    confounder spikes into its own definition of normal.
    """
    values = z[:, quiet_days]
    values = values[np.isfinite(values)]
    return float(np.quantile(values, 1.0 - nominal))


def window_role(window: ConfounderWindow) -> str:
    """``"adversarial"`` if the window moves the observable the detector reads."""
    return "adversarial" if window.feature == DETECTOR_FEATURE else "control"


def day_rate(z: np.ndarray, threshold: float, day: int) -> float | None:
    """One day of the plotted line, or ``None`` where there is nothing to plot.

    The first ``BASELINE_DAYS`` days have no trailing baseline, so their z is undefined.
    ``null`` draws that as a gap; ``0.0`` would draw it as "the detector was quiet", which
    is a different and false claim. ``NaN`` is not an option -- the artefact contract
    serialises with ``allow_nan=False``.
    """
    rate = alert_rate(z, threshold, np.array([day]))
    return None if np.isnan(rate) else rate


def dump_series(
    prevalence: float,
    nominal: float,
    windows: list[ConfounderWindow],
    series: list[dict[str, Any]],
) -> None:
    """Write the figure's data to ``DEFAULT_G5_PATH`` in the shape ``build_g5`` requires.

    ``start_day``/``end_day`` are the generator's own **half-open** ``[start, end)`` range,
    unmodified: ``ConfounderWindow`` says "gate G5 shades exactly these", and re-deriving
    the band here would let the shading and the measurement drift apart.
    """
    G5_SERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "prevalence": prevalence,
        "nominal_alert_rate": nominal,
        "excess_allowed_pp": EXCESS_ALLOWED * 100.0,
        "n_days": GATE_DAYS,
        "windows": [
            {
                "confounder": window.confounder.value,
                "start_day": window.start_day,
                "end_day": window.end_day,
                "feature": window.feature,
                "role": window_role(window),
            }
            for window in windows
        ],
        "series": series,
    }
    G5_SERIES_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_g5_confounder_null(null_data: GeneratedData) -> None:
    config = scenario(prevalence=0.0)
    layer = build_layer(
        config, np.zeros(4, dtype=np.int64), np.full(4, 6.0), np.full(4, 0.5)
    )
    windows = layer.windows

    # Nominal FPR is the analyst-capacity rate: K reviews per day per N merchants. A
    # metric that ignores K is decoration (CLAUDE.md), and that includes this one.
    nominal = config.capacity.analyst_reviews_per_day / config.capacity.per_n_merchants

    assert null_data.ground_truth["risk_typology_id"].null_count() == null_data.ground_truth.height
    counts = daily_counts(null_data)

    busy = np.zeros(GATE_DAYS, dtype=bool)
    for window in windows:
        busy[window.start_day : window.end_day] = True
    quiet = np.flatnonzero(~busy & (np.arange(GATE_DAYS) > BASELINE_DAYS))
    assert quiet.size > 40, "not enough quiet days to calibrate a threshold against"

    raw = trailing_z(counts, BASELINE_DAYS)
    residual = cohort_residual(raw)

    series: list[dict[str, Any]] = []
    for name, z in (("raw", raw), ("cohort-residual", residual)):
        threshold = calibrate(z, quiet, nominal)
        baseline_rate = alert_rate(z, threshold, quiet)
        worst = 0.0
        excesses: list[dict[str, Any]] = []
        for window in windows:
            days = np.arange(window.start_day, window.end_day)
            days = days[days > BASELINE_DAYS]
            if days.size == 0:
                continue
            rate = alert_rate(z, threshold, days)
            excess = rate - nominal
            worst = max(worst, excess)
            excesses.append(
                {
                    "confounder": window.confounder.value,
                    "start_day": window.start_day,
                    "end_day": window.end_day,
                    "role": window_role(window),
                    "days_measured": int(days.size),
                    "alert_rate": rate,
                    "excess_pp": excess * 100.0,
                }
            )
            record(
                f"G5 {name} {window.confounder.value}",
                "GREEN" if excess <= EXCESS_ALLOWED else "RED",
                f"alert rate {rate:.4f} vs nominal {nominal:.4f} "
                f"(excess {excess * 100:+.2f}pp, allowed +{EXCESS_ALLOWED * 100:.0f}pp)",
                f"days {window.start_day}-{window.end_day}, feature {window.feature}",
            )
        record(
            f"G5 {name} SUMMARY",
            "GREEN" if worst <= EXCESS_ALLOWED else "RED",
            f"worst window excess {worst * 100:+.2f}pp; quiet-day rate {baseline_rate:.4f}",
            "prevalence=0, so every alert here is a false positive by construction",
        )
        # Every number in the dump is the one just recorded to the terminal rather than a
        # second computation of it: a dump that quietly disagreed with the reported gate
        # would be worse than no dump at all.
        series.append(
            {
                "detector": name,
                # A freshly generated zero-prevalence population belongs to no split.
                "split": "NULL_RUN",
                "threshold": threshold,
                "quiet_day_rate": baseline_rate,
                "alert_rate_by_day": [day_rate(z, threshold, d) for d in range(GATE_DAYS)],
                "window_excess": excesses,
                "verdict": "GREEN" if worst <= EXCESS_ALLOWED else "RED",
            }
        )
        # The measurement is sound: the threshold really does sit at the nominal rate on
        # the quiet stretch. Without this, a RED verdict could just be a mis-calibration.
        assert abs(baseline_rate - nominal) < 0.005, (
            f"{name} threshold did not calibrate: quiet-day rate {baseline_rate:.4f} vs "
            f"nominal {nominal:.4f}"
        )

    dump_series(config.population.prevalence, nominal, windows, series)


def test_g5_windows_cover_all_six_confounders(null_data: GeneratedData) -> None:
    """A null test that silently examined four of the six events would pass for a reason
    that has nothing to do with the detector."""
    config = scenario(prevalence=0.0)
    layer = build_layer(config, np.zeros(4, dtype=np.int64), np.full(4, 6.0), np.full(4, 0.5))
    assert len({w.confounder for w in layer.windows}) == 6
    # And the run really is confounded: volume moves on the platform, not per merchant.
    daily_total = (
        null_data.transactions.with_columns(
            day=(pl.col("event_time") - START).dt.total_days().cast(pl.Int64)
        )
        .group_by("day")
        .len()
        .sort("day")["len"]
        .to_numpy()
    )
    assert daily_total.max() / np.median(daily_total) > 1.3
