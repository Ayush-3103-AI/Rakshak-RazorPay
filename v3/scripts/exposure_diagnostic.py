"""Reproduce the measurement behind ``LIMITATIONS.md`` §8.3a and PRE-REGISTRATION-CYCLE4 §1.2.

§8.3a asserts that `volume_rank` beats every rung on savings because of the *exposure*
estimator the decision layer is handed, not because of anything about detection. That claim
rests on two numbers — a rank correlation and a loss-capture share — and a graded artefact
should not cite numbers a reader cannot regenerate.

    uv run python scripts/exposure_diagnostic.py                        # cycle-3 data
    uv run python scripts/exposure_diagnostic.py --root data/v2         # cycle-4 data

**Defaults to the cycle-3 dataset**, preserved at ``data/_v2_cycle3_immutable`` when cycle 4
regenerated ``data/v2``, because that is the data the published §8.3a numbers were measured
on and the cycle-3 ladder is tagged ``cycle3-ladder-immutable``.

Ground truth is the diagnostic *target* here, never a model input. This is eval-side
analysis of the same shape as the rest of ``LIMITATIONS.md``; nothing in ``src/rakshak/``
reads what this script reads.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ORIGIN = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
CYCLE3 = Path("data/_v2_cycle3_immutable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=CYCLE3, help="Generated dataset.")
    ap.add_argument("--cutoff-day", type=int, default=240,
                    help="The day the scored window opens; volume is read strictly before "
                         "it, so the estimator stays point-in-time.")
    ap.add_argument("--k", type=int, nargs="*", default=[15, 50, 100, 200])
    args = ap.parse_args()

    if not (args.root / "ground_truth.parquet").exists():
        print(f"no dataset at {args.root}")
        return 1

    gt = pl.read_parquet(args.root / "ground_truth.parquet")
    prof = pl.read_parquet(args.root / "profiles.parquet")
    cutoff = ORIGIN + dt.timedelta(days=args.cutoff_day)

    # The same quantity `volume_rank` ranks on: captured, non-refunded GMV before the window.
    observed = (
        pl.scan_parquet(args.root / "transactions.parquet")
        .filter(
            (pl.col("event_time") < cutoff)
            & (pl.col("status") == "captured")
            & ~pl.col("is_refund")
        )
        .group_by("merchant_id")
        .agg(pl.col("amount_inr").sum().alias("observed_gmv"))
        .collect(engine="streaming")
    )

    df = (
        gt.join(prof.select("merchant_id", "declared_monthly_gmv"), on="merchant_id")
        .join(observed, on="merchant_id", how="left")
        .with_columns(pl.col("observed_gmv").fill_null(0.0))
    )
    fraud = df.filter(
        pl.col("drift_onset_at").is_not_null() & (pl.col("true_loss_amount_inr") > 1.0)
    )

    loss = fraud["true_loss_amount_inr"].to_numpy()
    declared = fraud["declared_monthly_gmv"].to_numpy()
    obs = fraud["observed_gmv"].to_numpy()

    print(f"dataset {args.root}   merchants {df.height:,}   "
          f"fraud merchants with a real loss {fraud.height}")
    print(f"volume read strictly before day {args.cutoff_day}\n")

    print("how well does each exposure estimator RANK the realised loss?")
    print(f"  {'estimator':<46}{'spearman':>10}{'pearson(log)':>15}")
    for name, x in (
        ("declared_monthly_gmv  (the decision layer's)", declared),
        ("observed pre-window GMV  (volume_rank's)", obs),
    ):
        print(f"  {name:<46}{stats.spearmanr(x, loss).statistic:>+10.4f}"
              f"{np.corrcoef(np.log1p(x), np.log1p(loss))[0, 1]:>+15.4f}")

    allm = df.filter(pl.col("observed_gmv") > 0)
    d_all, o_all = (allm["declared_monthly_gmv"].to_numpy(),
                    allm["observed_gmv"].to_numpy())
    print(f"\n  spearman(declared, observed) over all {allm.height:,} merchants = "
          f"{stats.spearmanr(d_all, o_all).statistic:+.4f}")
    print(f"  sd of log(observed / declared) = {np.std(np.log(o_all / d_all)):.4f}"
          f"   (config declaration_error_sigma = 0.55)")

    print("\nshare of TOTAL realised fraud loss sitting in the top K under each ranking")
    print(f"  {'K':>6}{'declared':>12}{'observed':>12}{'oracle':>12}")
    total = loss.sum()
    for k in args.k:
        row = f"  {k:>6}"
        for x in (declared, obs, loss):
            row += f"{loss[np.argsort(-x)[:k]].sum() / total:>11.2%} "
        print(row)
    print("\nThe decision layer ranks on 0.8 * p * exposure - 250, so the exposure column is "
          "\na multiplicative factor in every alert it chooses. See LIMITATIONS.md 8.3a.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
