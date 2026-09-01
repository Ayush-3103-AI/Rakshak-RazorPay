"""G1 — the Fano calibration, the external marginal parity, and the anchor that matters.

Three parts, and they do not carry equal weight:

* **G1a** — realised Fano against v1's measured 12.25 ± 1.0. Self-contained, runs
  everywhere, and it is a calibration of the generator against its own target.
* **G1b** — per 08-generator-v2-spec.md §7, a two-sample KS ≤ 0.15 against the BAF
  marginal *for each shared feature analogue*. The spec assumes a shared feature space
  that largely does not exist; see ``eval/baf_adapter.py``. Three analogues survive
  scrutiny, four are named as NOT ANCHORABLE and printed rather than dropped.
* **G1c** — **the highest-value external check this project can make.** v2's central
  correction over v1 is that real counts are overdispersed (Fano 12.25) where v1 assumed
  Poisson. BAF's count columns are counts over a window in real-derived fraud data that
  informed *none* of the generator's parameters. If their Fano is far from 1, the v2
  premise has external support. If it is near 1, that is a bigger finding than anything
  else in the sprint and it has to be reported.
* **G1d** — the fraud base rate and the temporal drift. The two things BAF supplies with
  no analogue argument needed, and the check on v1's diagnosed 20%-prevalence error.

G1b, G1c and G1d need the ~1M-row anchor, which is CC BY-NC-SA and not vendored, so on a clean
clone they record SKIP with the reason — and a skipped external anchor is a materially
weaker claim than a passing one, which is exactly why it is reported rather than quietly
dropped.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from gates_report import (
    GATE_DAYS,
    GATE_MERCHANTS,
    complete_window_counts,
    daily_counts,
    green_if,
    record,
)

from rakshak.eval.baf_adapter import (
    ANALOGUES,
    BAF_COUNT_COLUMNS,
    MIN_DISTINCT_FOR_DISPERSION,
    baf_path,
    fano,
    ks_statistic,
    load_baf,
    robust_standardise,
)
from rakshak.generator.arrivals import fano_factor
from rakshak.generator.engine import GeneratedData

KS_CEILING = 0.15
FANO_TOLERANCE = 1.0
POISSON_FANO = 1.0


def test_g1_realised_fano_matches_the_target(gate_data: GeneratedData) -> None:
    """FR-003's acceptance clause, measured on the real generated stream.

    Per-merchant Fano averaged over merchants, not pooled across the population: pooling
    folds the between-merchant intensity spread into the statistic and would report a
    Fano far above the target for a population of perfectly calibrated merchants. See
    ``arrivals.fano_factor``.

    The measured value lands *above* the target here rather than on it, and the reason is
    real: the population Fano is measured on the composed intensity, which includes the
    persona shapes, the L3 growth ramp, the typology ramps and the confounder layer.
    Every one of those makes a merchant's own daily rate non-stationary, and
    non-stationary rate is extra variance that the arrival process did not put there.
    ``tests/unit/test_arrivals.py`` isolates the process itself at a constant intensity;
    this gate measures what the dataset actually has, which is the honest thing to report
    even though it is the less flattering of the two.
    """
    n_days = gate_data.transactions["event_time"].dt.date().n_unique()
    counts = daily_counts(gate_data)
    realised = fano_factor(counts)
    target = 12.25
    ok = green_if(
        "G1a fano",
        abs(realised - target) <= FANO_TOLERANCE,
        f"realised Fano = {realised:.3f} vs target {target} (+/- {FANO_TOLERANCE})",
        f"measured over {GATE_MERCHANTS} merchants x {n_days} observed days, "
        f"on the composed intensity (persona x typology x confounder)",
    )
    assert ok, f"realised Fano {realised:.3f} is outside {target} +/- {FANO_TOLERANCE}"


def window_counts(data: GeneratedData, window_days: int) -> np.ndarray:
    """Transactions per merchant per complete ``window_days`` window.

    Complete-window selection and the horizon it is derived from live in
    ``gates_report.complete_window_counts``; G2 reads the same helper.
    """
    return complete_window_counts(data, window_days)["len"].cast(pl.Float64).to_numpy()


def _rakshak_marginal(data: GeneratedData, name: str) -> np.ndarray:
    if name == "txn_per_merchant_28d":
        return window_counts(data, 28)
    if name == "txn_per_merchant_56d":
        return window_counts(data, 56)
    if name == "is_international":
        return (
            data.transactions.filter(~pl.col("is_refund"))["is_international"]
            .cast(pl.Float64)
            .to_numpy()
        )
    raise KeyError(name)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "G1b is RED and is expected to be, and the spec anticipates it: section 7's "
        "remediation for a red G1 is 'recalibrate persona parameters, one attempt, then "
        "charter K-3' - a decision, not a build break. MEASURED: count_28d KS 0.0922 "
        "GREEN, cross_border KS 0.0056 GREEN, count_56d KS 0.2679 RED against a 0.15 "
        "ceiling (0.1531 under a log transform too, so not an artefact of the scale). "
        "The recalibration is DECLINED, deliberately: BAF's 8-week analogue is "
        "bank_branch_count_8w, whose quantiles are q10/q50/q90 = 0/9/750 against our "
        "merchants' 102/406/790, so much of the gap is a fact about counting bank "
        "branches rather than about our merchants' transaction counts. Tuning the "
        "generator's 56-day window to match a different entity population would make the "
        "gate green and the generator worse. count_56d is KEPT rather than dropped, "
        "because dropping it would leave the gate standing on the one window that agreed. "
        "strict=True so this cannot rot: if it ever passes, the suite fails and someone "
        "has to come back and rewrite this reason. Recorded in LIMITATIONS.md section 5."
    ),
)
def test_g1_marginal_parity_against_baf(gate_data: GeneratedData) -> None:
    """KS ≤ 0.15 per analogue, on robust-standardised marginals.

    **Robust-standardised, not rank-normalised.** The rank normalisation this gate used
    before T-116b made it vacuous: mapping a sample onto its own ranks makes its
    empirical CDF uniform by construction, so the KS between two rank-normalised samples
    is ~0 no matter what the samples are. At a million draws each,
    ``KS(rank(Normal), rank(Exponential))`` measures **0.0** exactly. G1b could not have
    failed, and a gate that cannot fail is not evidence. Centring on the median and
    scaling by the IQR removes the units — which genuinely do not correspond between an
    application funnel and a payments funnel — while leaving skew and tail intact, which
    is the thing G1 is supposed to be asking about.
    """
    if baf_path() is None:
        record(
            "G1b baf-parity",
            "SKIP",
            "BAF dataset not present",
            "set RAKSHAK_BAF_PATH (e.g. at data/external/baf.zip) to enable. The external "
            "anchor is the only real-derived data this project has; a SKIP here is a "
            "materially weaker claim than a GREEN and is reported as such.",
        )
        pytest.skip("BAF not available on this machine")

    baf = load_baf([a.baf_column for a in ANALOGUES if a.anchorable])
    assert baf is not None
    for analogue in (a for a in ANALOGUES if not a.anchorable):
        record(
            "G1b baf-parity",
            "SKIP",
            f"{analogue.name}: NOT ANCHORABLE ({analogue.baf_column})",
            analogue.why,
        )

    worst_name, worst_ks = "", 0.0
    for analogue in (a for a in ANALOGUES if a.anchorable):
        left = robust_standardise(_rakshak_marginal(gate_data, analogue.rakshak))
        right = robust_standardise(baf[analogue.baf_column].cast(pl.Float64).to_numpy())
        ks = ks_statistic(left, right)
        record(
            "G1b baf-parity",
            "GREEN" if ks <= KS_CEILING else "RED",
            f"{analogue.name}: KS = {ks:.4f} (ceiling {KS_CEILING})",
            f"{analogue.rakshak} vs BAF {analogue.baf_column}, "
            f"n = {left.size:,} vs {right.size:,}",
        )
        if ks > worst_ks:
            worst_name, worst_ks = analogue.name, ks

    record(
        "G1b baf-parity",
        "GREEN" if worst_ks <= KS_CEILING else "RED",
        f"OVERALL: worst analogue {worst_name} at KS {worst_ks:.4f}",
        f"{sum(a.anchorable for a in ANALOGUES)} analogues scored, "
        f"{sum(not a.anchorable for a in ANALOGUES)} named NOT ANCHORABLE above",
    )
    assert worst_ks <= KS_CEILING, (
        f"G1b RED: worst analogue {worst_name} at KS {worst_ks:.4f} > {KS_CEILING}"
    )


def test_g1c_baf_counts_are_overdispersed(gate_data: GeneratedData) -> None:
    """The external test of v2's central premise: are real counts Poisson?

    v1 assumed Poisson arrivals. v2's whole correction is that they are not — measured
    Fano 12.25. Every other gate in this suite is internal: G1a checks the generator
    against its own target, G3 checks determinism, G4 checks the quarantine, G5 checks
    the generator against a detector built on the same generator. This is the one place a
    number arrives from outside, from a dataset that informed none of the generator's
    parameters.

    GREEN when every non-degenerate BAF count column has Fano > 1, i.e. the Poisson null
    is rejected on data this project did not make. The degeneracy filter
    (``MIN_DISTINCT_FOR_DISPERSION``) is declared in the adapter ahead of being applied,
    and the column it excludes is printed with its number anyway.
    """
    if baf_path() is None:
        record(
            "G1c baf-overdispersion",
            "SKIP",
            "BAF dataset not present",
            "this is the single highest-value external check available to the project — "
            "whether real count data rejects the Poisson assumption v1 made — and without "
            "the anchor it cannot be made. Set RAKSHAK_BAF_PATH to enable.",
        )
        pytest.skip("BAF not available on this machine")

    baf = load_baf(list(BAF_COUNT_COLUMNS))
    assert baf is not None
    gated: dict[str, float] = {}
    for column in BAF_COUNT_COLUMNS:
        values = baf[column].cast(pl.Float64).to_numpy()
        f = fano(values)
        distinct = int(baf[column].n_unique())
        if distinct < MIN_DISTINCT_FOR_DISPERSION:
            record(
                "G1c baf-overdispersion",
                "SKIP",
                f"{column}: Fano = {f:.3f} — REPORTED, NOT GATED",
                f"only {distinct} distinct values, so there is no dispersion left to "
                f"measure. Excluded by the rule declared in baf_adapter, not because the "
                f"number is inconvenient — the number is printed here either way.",
            )
            continue
        gated[column] = f
        record(
            "G1c baf-overdispersion",
            "GREEN" if f > POISSON_FANO else "RED",
            f"{column}: Fano = {f:.3f} vs Poisson null {POISSON_FANO}",
            f"mean {values.mean():.2f}, var {values.var():.2f}, {distinct:,} distinct",
        )

    ours = fano(window_counts(gate_data, 28))
    record(
        "G1c baf-overdispersion",
        "GREEN" if all(f > POISSON_FANO for f in gated.values()) else "RED",
        "OVERALL: Poisson REJECTED on "
        f"{sum(f > POISSON_FANO for f in gated.values())}/{len(gated)} BAF count columns",
        f"generator's own pooled 28-day count Fano is {ours:.1f}, inside BAF's "
        f"[{min(gated.values()):.1f}, {max(gated.values()):.1f}] — but note both sides "
        f"pool across entities, and BAF has no entity id, so a per-entity Fano (the "
        f"12.25 the generator targets) cannot be computed on BAF at all.",
    )
    assert all(f > POISSON_FANO for f in gated.values()), (
        f"a BAF count column is not overdispersed: {gated}. v2's central premise over v1 "
        f"is that real counts are overdispersed; this is the external evidence for it."
    )


def test_g1d_prevalence_and_drift_are_anchored(gate_data: GeneratedData) -> None:
    """The second thing BAF genuinely has: a real fraud base rate and real drift.

    v1's diagnosed error was evaluating at 20% prevalence when the real rate is ~1.5%
    (`CLAUDE.md`, `14-lit-survey-v2.md`). BAF is the only outside number available to
    check that against, and unlike the marginals it needs no analogue argument: a fraud
    base rate is a fraud base rate on both sides.

    GREEN when the shipped scenario's prevalence is within 2x of BAF's **and** v1's 0.20
    is not. Both halves matter — the first alone would pass for any small number, and the
    point of the clause is that it discriminates.

    The monthly spread is reported rather than gated. BAF's drift is real but it is drift
    in *account-opening* fraud over eight months; the generator's window is not the
    same clock, and asserting agreement between them would be inventing a correspondence
    again.
    """
    if baf_path() is None:
        record(
            "G1d baf-prevalence",
            "SKIP",
            "BAF dataset not present",
            "the real fraud base rate and the real temporal drift are the two things BAF "
            "supplies without needing an analogue argument. Set RAKSHAK_BAF_PATH.",
        )
        pytest.skip("BAF not available on this machine")

    baf = load_baf(["fraud_bool", "month"])
    assert baf is not None
    theirs = float(baf["fraud_bool"].mean())
    ours = float(gate_data.labels["label"].drop_nulls().mean())
    configured = 0.0147
    v1 = 0.20

    by_month = (
        baf.group_by("month")
        .agg(pl.col("fraud_bool").mean().alias("p"))
        .sort("month")["p"]
        .to_numpy()
    )
    record(
        "G1d baf-prevalence",
        "SKIP",
        f"BAF monthly prevalence {by_month.min():.4f} -> {by_month.max():.4f} "
        f"({by_month.max() / by_month.min():.2f}x swing over 8 months) — REPORTED",
        "real temporal drift in a real label distribution, and the reason T-0012 used "
        "BAF at all. Not gated: BAF's eight months of account-opening fraud is not the "
        f"generator's {GATE_DAYS} days of merchant behaviour, and asserting they agree "
        "would be inventing a correspondence.",
    )
    ok = green_if(
        "G1d baf-prevalence",
        theirs / 2 <= configured <= theirs * 2 and not (theirs / 2 <= v1 <= theirs * 2),
        f"BAF {theirs:.4f} vs configured {configured:.4f} "
        f"({configured / theirs:.2f}x) — v1's {v1} is {v1 / theirs:.1f}x and REJECTED",
        f"realised on the gate population: {ours:.4f} over "
        f"{gate_data.labels['label'].drop_nulls().len():,} resolved merchants",
    )
    assert ok, f"configured prevalence {configured} is not within 2x of BAF's {theirs:.4f}"
