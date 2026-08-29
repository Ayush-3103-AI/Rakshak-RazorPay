"""Empirical calibration profile and the profile-vs-generator gap diff (T-0015).

`build_profile` measures a fixed set of marginals on **one merchant's transaction
stream** in the canonical schema (`timestamp`, `amount`, `payer_id`, `is_refund`). The
same function runs over real data and over the synthetic generator's own output, so the
two sides of `results/calibration_gap.md` are computed identically and are comparable by
construction rather than by hand-transcription.

Deliberately measured per merchant, not pooled. Pooling the generator's 500 merchants
would fold a 60-30,000 INR cross-merchant AOV spread into `amount_log_sd` and make it
incomparable with a single real retailer's within-merchant dispersion.

WHAT THIS CANNOT INFORM
-----------------------
`06-requirements.md:28` and ADR-0007: no public merchant-sequence dataset with
merchant-level risk labels exists. Everything in `CANNOT_INFORM` therefore stays exactly
as hand-chosen as it was before this module existed, and the README must say so.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rakshak.cli import base_parser
from rakshak.config import RESULTS_DIR, SEED, STATE_PATHS_PARQUET, TRANSACTIONS_PARQUET
from rakshak.data.download import EXTERNAL_DIR

CANONICAL_COLUMNS: tuple[str, ...] = ("timestamp", "amount", "payer_id", "is_refund")
"""Schema `build_profile` consumes. `amount` is a positive magnitude; `is_refund` is the sign."""

CANNOT_INFORM: tuple[str, ...] = (
    "Latent risk states and their transition structure — no public dataset labels a "
    "merchant's risk state through time. The HMM's 4-state path is entirely synthetic.",
    "Merchant-level fraud labels and prevalence — public fraud data is transaction-level "
    "or application-level. FRAUD_MERCHANT_RATE = 0.20 remains a hand-chosen evaluation "
    "convenience, far above any real prevalence.",
    "Chargeback rate — no public transaction dataset carries a chargeback flag. "
    "Segment.chargeback_rate (0.0004-0.0040) is unchanged and uncalibrated.",
    "Cross-merchant payer graph — the generator scopes payers to one merchant by design "
    "(ADR-0002), and no public dataset supplies a real merchant x payer bipartite graph.",
    "MCC / category structure and absolute ticket size — the one real stream available is "
    "a single UK gift-ware retailer in GBP. Level comparisons across currency and category "
    "are meaningless; only unit-free shape statistics are compared.",
    "Typology dynamics — bust-out, laundering, category drift, refund collusion and the "
    "adversarial slow ramp are all invented here and calibrated against nothing.",
)
"""Stated in `results/calibration_gap.md` and required in the README beside any number."""

GENERATOR_CONSTANTS: dict[str, str] = {
    "amount_log_sd": "generator.SEGMENTS[*].amount_sigma, hand-chosen 0.35-0.90",
    "amount_p50": "generator.SEGMENTS[*].aov_lo..aov_hi, log-uniform 60-30,000 INR",
    "amount_p90_over_p50": "implied by amount_sigma: exp(1.2816 * sigma) = 1.57-3.16",
    "hour_of_day_mean": "generator.SEGMENTS[*].peak_hour, hand-chosen 11.5-22.0",
    "hour_of_day_sd_hours": "rng.normal(0.0, 2.6) in generate._emit_merchant",
    "weekday_volume_factor": "generator._WEEKDAY_FACTOR, hand-chosen Mon..Sun",
    "refund_rate": "generator.SEGMENTS[*].refund_rate, hand-chosen 0.008-0.070",
    "new_payer_frac": "generator.SEGMENTS[*].unique_payer_frac, hand-chosen 0.12-0.92",
    "top_decile_payer_share": "emergent from generator._REPEAT_CONCENTRATION = 2.5",
    "txns_per_active_day_mean": (
        "generator.SEGMENTS[*].volume_lo..volume_hi / 30, log-uniform 5-800 txn/month"
    ),
    "daily_count_fano_factor": "rng.poisson(lam) in generate._emit_merchant: 1.0 by construction",
}
"""Which hand-chosen generator value each marginal is the empirical counterpart of."""

SURVEY_MARKDOWN: str = """## Dataset survey and licence gate

Six candidates evaluated on 2026-08-29 — the five `11-tickets/T-0015.md` names, plus
Online Retail II. **Every licence below was read at the source URL given, not assumed.**

| dataset | what it actually contains | granularity | licence (verified at source) | access | verdict |
|---|---|---|---|---|---|
| **Online Retail II** (UCI 502) | Real invoices of ONE UK B2B gift-ware wholesaler, Dec 2009 - Dec 2011. 1,067,371 line items -> 48,374 invoices. | line item -> invoice; has timestamp, amount, customer id, cancellation flag | **CC BY 4.0** — `archive.ics.uci.edu/dataset/502/online+retail+ii` states "Creative Commons Attribution 4.0 International". Permits commercial use, redistribution and vendoring, with attribution. | direct HTTPS, no account | **SELECTED** |
| **BAF** (Feedzai, NeurIPS 2022) | 6 variants x ~1M bank **account-opening applications**, CTGAN-synthesised from an anonymised real bank dataset. No amounts, no timestamps, no payer ids. | application | **CC BY-NC-SA 4.0** — Kaggle dataset page `sgpjesus/bank-account-fraud-dataset-neurips-2022`. Note the trap: the GitHub repo's `LICENSE` is **Apache-2.0 and covers the code, not the data.** NC + SA is fine for evaluation inside git-ignored `data/`; **not vendorable** into an MIT repo. | Kaggle account + API token | **NOT DOWNLOADED** — no credentials on this machine. Wrong granularity for the profile in any case; retained in `SOURCES` for T-0012. |
| **IEEE-CIS Fraud Detection** (Vesta) | ~590k card transactions; `TransactionAmt`, `TransactionDT` offset, `ProductCD`, obfuscated card/address/email fields. No merchant id. | transaction | **Competition rules**, acceptance required per entrant; redistribution not permitted. The rules page sits behind the login wall and **could not be read at source** — recorded as unverified rather than assumed. | Kaggle account + rules acceptance | **REJECTED** — access gate, unverifiable licence, no merchant axis |
| **ULB credit-card fraud** (mlg-ulb) | 284,807 European card transactions over **2 days**; 28 PCA components + `Time` (s) + `Amount` (EUR). | transaction, pooled across an issuer's whole card base | **ODbL** for the database, **DbCL v1.0** for the contents — Kaggle page structured metadata, `opendatacommons.org/licenses/dbcl/1.0/`. Share-alike attaches to any derived database published. | Kaggle account | **REJECTED** — no merchant axis, no payer id, no refund/chargeback flag; a 2-day window cannot inform weekday seasonality or per-merchant velocity |
| **PaySim** (ealaxi) | 6.35M **simulated** mobile-money transfers, 744 hourly steps. | agent-level simulation | **CC BY-SA 4.0** — Kaggle dataset page `ealaxi/paysim1`. | Kaggle account | **REJECTED on circularity, which outranks the licence** — it is a simulator's output. Calibrating our generator on another generator's hand-chosen parameters launders our assumptions through someone else's. |
| **Sparkov** (kartik2112) | 1.85M **simulated** card transactions with `merchant`, `category`, `amt`, `trans_date_trans_time` — superficially the best schema match of the six. | transaction simulation | **CC0: Public Domain** — Kaggle dataset page `kartik2112/fraud-detection`. The most permissive licence in the table. | Kaggle account | **REJECTED** — same circularity as PaySim. A CC0 licence does not make a simulation real. |

**The licence gate was not the binding constraint, and that is the finding.** No candidate
was dropped because of its licence. Two were dropped on **access** (Kaggle credentials),
two on **syntheticity** (they are simulations), one on **granularity** (applications, not
transactions), and one on **structure** (no merchant axis, 2-day window). The single
dataset that survived is real, permissively licensed, and describes exactly one shop.

"""
"""The survey table, rendered into `results/calibration_gap.md`. Licences verified 2026-08-29."""

LEVEL_ONLY: frozenset[str] = frozenset({"amount_p50"})
"""Marginals whose *level* is not comparable across currency and merchant category."""

ABSOLUTE_DIFF: frozenset[str] = frozenset({"hour_of_day_mean"})
"""Marginals on an interval scale, where a ratio is meaningless. A clock hour has no
true zero, so 12.75h vs 15.60h is a 2.85-hour shift, not a factor of 0.82."""


def _figure(
    value: float | list[float],
    unit: str,
    dataset: str,
    column: str,
    n: int,
    note: str = "",
) -> dict[str, Any]:
    """Wrap one measured number with the provenance the ticket requires on every figure."""
    return {
        "value": value,
        "unit": unit,
        "dataset": dataset,
        "column": column,
        "n": int(n),
        "note": note,
    }


def build_profile(df: pd.DataFrame, dataset: str = "unknown") -> dict[str, Any]:
    """Measure the calibration marginals on one merchant's transaction stream.

    Args:
        df: Transactions in `CANONICAL_COLUMNS`. `timestamp` must be datetime-like,
            `amount` a positive magnitude in the source currency, `payer_id` a string
            (empty when unknown), `is_refund` boolean.
        dataset: Provenance label stamped onto every figure.

    Returns:
        `{"dataset", "n_transactions", "marginals"}` where each marginal is a dict of
        `{value, unit, dataset, column, n, note}`. Units are stated per figure; ratios
        and standard deviations of log-amounts are dimensionless and are the only
        amount statistics that survive a currency change.

    Raises:
        ValueError: If a canonical column is missing or the frame is empty.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    if df.empty:
        raise ValueError("cannot profile an empty transaction stream")

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["is_refund"] = frame["is_refund"].astype(bool)
    frame["payer_id"] = frame["payer_id"].fillna("").astype(str)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

    n_all = len(frame)
    sale = frame.loc[~frame["is_refund"] & (frame["amount"] > 0.0)]
    amounts = sale["amount"].to_numpy(dtype=float)
    # Hour statistics are linear, not circular. Safe for a daytime retail stream; the
    # generator's 22:00-peak segments do wrap, and the gap file says so.
    hours = frame["timestamp"].dt.hour.to_numpy(dtype=float)

    weekday_counts = np.zeros(7, dtype=float)
    for index, count in frame["timestamp"].dt.weekday.value_counts().items():
        weekday_counts[int(index)] = float(count)
    weekday_factor = weekday_counts / (n_all / 7.0)

    daily = frame.groupby(frame["timestamp"].dt.date).size().to_numpy(dtype=float)

    payers = frame.loc[frame["payer_id"] != "", "payer_id"]
    n_payer_rows = int(len(payers))
    if n_payer_rows:
        counts = payers.value_counts().to_numpy(dtype=float)
        n_payers = int(len(counts))
        top_k = max(1, math.ceil(0.10 * n_payers))
        new_payer_frac = float(n_payers) / float(n_payer_rows)
        top_decile_share = float(np.sort(counts)[::-1][:top_k].sum()) / float(n_payer_rows)
    else:
        n_payers, new_payer_frac, top_decile_share = 0, float("nan"), float("nan")

    p50 = float(np.percentile(amounts, 50.0))
    p90 = float(np.percentile(amounts, 90.0))

    marginals: dict[str, dict[str, Any]] = {
        "amount_log_sd": _figure(
            float(np.log(amounts).std()),
            "dimensionless (sd of ln amount)",
            dataset,
            "amount (non-refund rows)",
            len(amounts),
            "Currency-free. Directly comparable with the generator's lognormal sigma.",
        ),
        "amount_p50": _figure(
            p50,
            "source currency per transaction",
            dataset,
            "amount (non-refund rows)",
            len(amounts),
            "LEVEL ONLY — not comparable across currency or merchant category.",
        ),
        "amount_p90_over_p50": _figure(
            p90 / p50 if p50 else float("nan"),
            "dimensionless ratio",
            dataset,
            "amount (non-refund rows)",
            len(amounts),
            "Currency-free tail-shape statistic.",
        ),
        "hour_of_day_mean": _figure(
            float(hours.mean()),
            "hour of day (0-23, local clock)",
            dataset,
            "timestamp.hour",
            n_all,
            "Linear mean, not circular. Valid while the mass does not straddle midnight.",
        ),
        "hour_of_day_sd_hours": _figure(
            float(hours.std()),
            "hours",
            dataset,
            "timestamp.hour",
            n_all,
            "Population sd (ddof=0).",
        ),
        "weekday_volume_factor": _figure(
            [float(x) for x in weekday_factor],
            "dimensionless multiplier, Mon..Sun, mean 1 by construction",
            dataset,
            "timestamp.weekday",
            n_all,
            "Transaction counts per weekday over the whole stream, normalised to mean 1.",
        ),
        "refund_rate": _figure(
            float(frame["is_refund"].mean()),
            "fraction of transactions",
            dataset,
            "is_refund",
            n_all,
            "",
        ),
        "new_payer_frac": _figure(
            new_payer_frac,
            "fraction of transactions",
            dataset,
            "payer_id",
            n_payer_rows,
            "Distinct payers divided by transactions carrying a payer id.",
        ),
        "top_decile_payer_share": _figure(
            top_decile_share,
            "fraction of transactions",
            dataset,
            "payer_id",
            n_payer_rows,
            f"Share held by the busiest ceil(0.1 * {n_payers}) payers.",
        ),
        "txns_per_active_day_mean": _figure(
            float(daily.mean()),
            "transactions per active day",
            dataset,
            "timestamp.date",
            int(len(daily)),
            "Days with zero transactions are excluded; this is the velocity marginal.",
        ),
        "daily_count_fano_factor": _figure(
            float(daily.var() / daily.mean()) if daily.mean() else float("nan"),
            "dimensionless (variance / mean of daily counts)",
            dataset,
            "timestamp.date",
            int(len(daily)),
            "1.0 for a Poisson process. Above 1.0 means over-dispersed.",
        ),
    }
    return {"dataset": dataset, "n_transactions": n_all, "marginals": marginals}


def profile_population(
    df: pd.DataFrame, merchant_col: str = "merchant_id", dataset: str = "unknown"
) -> dict[str, Any]:
    """Profile every merchant separately and return the across-merchant median per marginal.

    Args:
        df: Transactions in `CANONICAL_COLUMNS` plus a merchant column.
        merchant_col: Column identifying the merchant.
        dataset: Provenance label.

    Returns:
        The same shape as `build_profile`, with each figure's `value` the median across
        merchants and `n` the number of merchants that contributed.
    """
    per_merchant: list[dict[str, Any]] = []
    for _, group in df.groupby(merchant_col, sort=True):
        if len(group) < 2:
            continue
        per_merchant.append(build_profile(group, dataset=dataset))
    if not per_merchant:
        raise ValueError("no merchant had enough transactions to profile")

    template = per_merchant[0]["marginals"]
    marginals: dict[str, dict[str, Any]] = {}
    for name, figure in template.items():
        stack = [p["marginals"][name]["value"] for p in per_merchant]
        if isinstance(figure["value"], list):
            value: float | list[float] = [
                float(x) for x in np.nanmedian(np.asarray(stack, dtype=float), axis=0)
            ]
        else:
            value = float(np.nanmedian(np.asarray(stack, dtype=float)))
        merged = dict(figure)
        merged["value"] = value
        merged["n"] = len(per_merchant)
        merged["note"] = ("Median across merchants. " + str(figure["note"])).strip()
        marginals[name] = merged
    return {
        "dataset": dataset,
        "n_transactions": int(len(df)),
        "n_merchants": len(per_merchant),
        "marginals": marginals,
    }


def diff_profile(empirical: dict[str, Any], generator: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare every marginal in `empirical` against the generator's realised value.

    Args:
        empirical: Output of `build_profile` on real data.
        generator: Output of `build_profile`/`profile_population` on generator output.

    Returns:
        One row per marginal, in the empirical profile's key order, each carrying the
        empirical value, the generator value, the hand-chosen constant that governs the
        generator side, and a divergence. `ratio` is empirical/generator for scalars and
        None for vector or non-comparable marginals; `max_abs_diff` is populated for
        vectors. Rows whose level is not comparable across currency/category are marked
        `comparable=False` and carry no divergence verdict.

    Raises:
        KeyError: If the generator profile is missing a marginal the empirical one has.
    """
    rows: list[dict[str, Any]] = []
    for name, emp in empirical["marginals"].items():
        gen = generator["marginals"][name]
        comparable = name not in LEVEL_ONLY
        row: dict[str, Any] = {
            "marginal": name,
            "unit": emp["unit"],
            "empirical": emp["value"],
            "empirical_source": f"{emp['dataset']}:{emp['column']}",
            "generator": gen["value"],
            "generator_constant": GENERATOR_CONSTANTS.get(name, "unmapped"),
            "comparable": comparable,
            "ratio": None,
            "abs_diff": None,
        }
        if isinstance(emp["value"], list):
            e = np.asarray(emp["value"], dtype=float)
            g = np.asarray(gen["value"], dtype=float)
            row["abs_diff"] = float(np.max(np.abs(e - g)))
        elif name in ABSOLUTE_DIFF:
            row["abs_diff"] = float(emp["value"]) - float(gen["value"])
        elif comparable and gen["value"]:
            row["ratio"] = float(emp["value"]) / float(gen["value"])
        rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    """Format a scalar or vector for the markdown table."""
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return "[" + ", ".join(f"{v:.2f}" for v in value) + "]"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_gap_markdown(
    rows: list[dict[str, Any]],
    empirical: dict[str, Any],
    generator: dict[str, Any],
    seed: int = SEED,
) -> str:
    """Render `results/calibration_gap.md` from a `diff_profile` result.

    Args:
        rows: Output of `diff_profile`.
        empirical: Profile of the real stream.
        generator: Profile of the synthetic population.
        seed: Seed stamped into the provenance line. The profiler itself draws no
            random numbers -- it reads a fixed manifest -- so the seed changes the
            stamp and nothing else. It is threaded through so the stamp can never
            disagree with the flag the run was actually invoked with.
    """
    lines: list[str] = [
        "# Calibration gap — empirical marginals vs the generator's hand-chosen values",
        "",
        f"**Generated by** `python -m rakshak.data.profile --seed {seed}`. "
        "**Ticket:** T-0015. **Consumes:** `results/calibration_profile.json`.",
        "",
        "> **Every Rakshak sequence-layer number is measured on synthetic merchant streams "
        "with injected typologies; the generator is in this repo.** This file measures how "
        "far that generator's marginals sit from marginals measured on a real transaction "
        "stream. It does not close the gap — it states it.",
        "",
        f"- **Empirical side:** `{empirical['dataset']}`, "
        f"{empirical['n_transactions']:,} transactions, "
        f"{empirical.get('n_merchants', 1)} merchant(s).",
        f"- **Generator side:** `{generator['dataset']}`, "
        f"{generator['n_transactions']:,} transactions, "
        f"{generator.get('n_merchants', 1)} merchant(s), **healthy merchants only** "
        "(typology == NONE), so no injected typology contaminates the comparison.",
        "- Both sides are measured **per merchant** by the same function "
        "(`rakshak.data.profile.build_profile`); the generator column is the median "
        "across merchants. Nothing here is hand-transcribed.",
        "",
        SURVEY_MARKDOWN,
        "## Per-marginal diff",
        "",
        "| marginal | unit | empirical | generator (realised) | generator's hand-chosen value "
        "| divergence |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if not row["comparable"]:
            divergence = "**not comparable** (currency + category)"
        elif row["abs_diff"] is not None:
            divergence = f"abs diff {row['abs_diff']:+.2f}"
        elif row["ratio"] is None:
            divergence = "n/a"
        else:
            divergence = f"x{row['ratio']:.2f}"
        lines.append(
            f"| `{row['marginal']}` | {row['unit']} | {_fmt(row['empirical'])} "
            f"| {_fmt(row['generator'])} | {row['generator_constant']} | {divergence} |"
        )

    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    wide = [r for r in ratios if r >= 1.9 or r <= 1.0 / 1.9]
    very_wide = [r for r in ratios if r >= 5.0 or r <= 0.2]
    worst = max((max(r, 1.0 / r) for r in ratios), default=float("nan"))

    fano = next(r["empirical"] for r in rows if r["marginal"] == "daily_count_fano_factor")
    lines += [
        "",
        "`divergence` is empirical / generator. `x1.00` is agreement; `x0.50` means the "
        "generator is twice the empirical value. Clock hours use an absolute difference "
        "because a ratio of interval-scale values is meaningless.",
        "",
        "## How to read this table — three caveats, none of which shrink the gap",
        "",
        "1. **n = 1 merchant on the empirical side.** Online Retail II is a single UK "
        "business-to-business gift-ware wholesaler trading in GBP. It is not a sample of "
        "merchants; it is one merchant. Marginals that vary with business model — daily "
        "volume, weekday shape, refund rate — measure *this shop* as much as they measure "
        "reality. **This weakens the profile as a recalibration target, not the gap as a "
        "finding.**",
        "2. **Category mismatch.** None of the generator's eight MCCs resembles a B2B "
        "wholesaler. The empirical weekday profile has a hard zero on Saturday (the "
        "business is closed) and peaks on Thursday; the generator peaks at the weekend, "
        "which is a consumer pattern. The `weekday_volume_factor` row is therefore a "
        "*shape mismatch between two different businesses*, not evidence that "
        "`_WEEKDAY_FACTOR` is wrong for retail.",
        "3. **`refund_rate` is not a like-for-like definition.** The source flags order "
        "*cancellations* (invoice ids prefixed `C`), which include stock-outs and "
        "data-entry reversals; the generator models customer-initiated refunds only. The "
        f"{next(r['ratio'] for r in rows if r['marginal'] == 'refund_rate'):.2f}x is an "
        "**upper bound** on the true divergence. The measured figure is left unadjusted — "
        "adjusting it would be exactly the tuning this ticket forbids.",
        "",
        "## The decision this file gates",
        "",
        "`11-tickets/T-0015.md` (amendment, 2026-08-28) makes T-0016 conditional on this "
        "diff: *divergence small* → recalibrate the generator; *divergence large* → "
        "publish this file as the stated limitation and **T-0016 does not run**.",
        "",
        "**Measured:** of the "
        f"{len(ratios)} ratio-scale marginals, **{len(wide)} diverge by 1.9x or more** and "
        f"{len(very_wide)} by 5x or more; the widest is **x{worst:.1f}**.",
        "",
        "**One divergence is structural, not parametric.** `daily_count_fano_factor` is "
        "1.0 for the generator *by construction* — it draws daily counts from "
        "`rng.poisson`, and a Poisson process has variance equal to its mean. The real "
        f"stream reads {fano:.1f}. "
        "**No choice of Poisson rate can produce over-dispersion.** Closing that marginal "
        "means replacing the emission process (negative binomial, or a latent intensity), "
        "which invalidates the K1 analysis, the 0.404 oracle ceiling and every baseline "
        "row measured to date. That is not a parameter swap.",
        "",
        "**Conclusion: the *divergence large* branch.** The gap ships documented and "
        "T-0016 is not run. Recalibrating a 500-merchant Indian-payments generator against "
        "the marginals of one UK wholesaler would substitute a *measured* limitation for "
        "an *unmeasurable* one — a generator calibrated to a distribution nobody can "
        "characterise, which `11-tickets/T-0016.md` itself forbids.",
        "",
        "## What this profile CANNOT inform",
        "",
        "`06-requirements.md:28` and ADR-0007: **no public merchant-sequence dataset with "
        "merchant-level risk labels exists.** Public data supplies marginals and rates only. "
        "The following remain **entirely synthetic and entirely uncalibrated**:",
        "",
    ]
    lines += [f"{i}. {text}" for i, text in enumerate(CANNOT_INFORM, start=1)]
    lines += [
        "",
        "This list is not a caveat appended to a result. It is the majority of what the "
        "generator asserts. The marginals above are its surface statistics; its sequence "
        "structure and its labels — the things the HMM is actually scored on — are "
        "calibrated against nothing and cannot be.",
        "",
    ]
    return "\n".join(lines)


def _load_generator_frame(healthy_only: bool = True) -> pd.DataFrame:
    """Load synthetic transactions in the canonical schema, optionally healthy merchants only.

    Args:
        healthy_only: Keep only merchants whose typology is NONE, so no injected
            typology contaminates the baseline comparison.

    Returns:
        DataFrame with `CANONICAL_COLUMNS` plus `merchant_id`.

    Raises:
        FileNotFoundError: If the generator has not been run.
    """
    if not TRANSACTIONS_PARQUET.exists():
        raise FileNotFoundError(
            f"{TRANSACTIONS_PARQUET} not found — run "
            "`python -m rakshak.generator.generate --seed 42` first."
        )
    txns = pd.read_parquet(TRANSACTIONS_PARQUET)
    if healthy_only and STATE_PATHS_PARQUET.exists():
        paths = pd.read_parquet(STATE_PATHS_PARQUET)
        healthy = set(paths.loc[paths["typology"] == "NONE", "merchant_id"].unique())
        txns = txns[txns["merchant_id"].isin(healthy)]
    txns = txns.copy()
    txns["payer_id"] = txns["payer_id"].astype(str)
    return txns[["merchant_id", *CANONICAL_COLUMNS]]


def main() -> None:
    """CLI: write `results/calibration_profile.json` and `results/calibration_gap.md`."""
    parser = base_parser("Build the empirical calibration profile and the generator gap diff.")
    parser.add_argument(
        "--empirical",
        type=Path,
        default=EXTERNAL_DIR / "online_retail_ii.parquet",
        help="Normalised real-data parquet written by rakshak.data.download.",
    )
    args = parser.parse_args()

    manifest_path = EXTERNAL_DIR / "online_retail_ii.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    empirical = build_profile(pd.read_parquet(args.empirical), dataset="online_retail_ii")
    generator = profile_population(_load_generator_frame(), dataset="rakshak_synthetic_healthy")
    rows = diff_profile(empirical, generator)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "ticket": "T-0015",
        "produced_by": f"python -m rakshak.data.profile --seed {args.seed}",
        "sources": {
            manifest["name"]: {
                key: manifest[key]
                for key in ("source_url", "source_page", "licence", "licence_url", "sha256")
            }
        },
        "cannot_inform": list(CANNOT_INFORM),
        "empirical": empirical,
        "generator": generator,
        "diff": rows,
    }
    profile_path = RESULTS_DIR / "calibration_profile.json"
    profile_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gap_path = RESULTS_DIR / "calibration_gap.md"
    gap_path.write_text(
        render_gap_markdown(rows, empirical, generator, args.seed), encoding="utf-8"
    )
    print(f"wrote {profile_path}")
    print(f"wrote {gap_path}")


if __name__ == "__main__":
    main()
