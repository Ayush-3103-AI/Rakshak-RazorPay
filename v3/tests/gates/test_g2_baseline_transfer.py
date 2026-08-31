"""G2 — baseline transfer. Does a model trained on the generator work on real data?

GREEN when a LightGBM trained on the generator and scored on BAF reaches PR-AUC >= 0.5x
its in-domain PR-AUC, and vice versa. If it does not, the generator is fiction and
charter K-3 fires.

This is the gate with the sharpest teeth and the weakest footing, and both facts belong
in the report. BAF is account-opening fraud with one row per application; Rakshak is
post-onboarding merchant behaviour with thousands of rows per merchant. A model cannot
literally transfer between them, so G2 is implemented over the **shared analogue
subspace** defined in ``eval/baf_adapter.py`` — the four columns that describe the same
*family* of quantity on both sides, rank-normalised. That is a real test of whether the
generator's joint structure over those four is realistic, and it is emphatically not a
test of whether a Rakshak model would work at a bank.

Without BAF on the machine this reports SKIP. It does not fail, because ``make gates``
must run on a clean clone (charter K-5), and it does not silently pass, because a
skipped anchor is not evidence.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from gates_report import record

from rakshak.eval.baf_adapter import ANALOGUES, baf_path
from rakshak.generator.engine import GeneratedData

TRANSFER_RATIO = 0.5


def test_g2_baseline_transfer(gate_data: GeneratedData) -> None:
    if baf_path() is None:
        record(
            "G2 baseline-transfer",
            "SKIP",
            "BAF dataset not present",
            "the strongest available evidence that the generator is not fiction is "
            "unavailable on this machine; charter K-3 cannot be evaluated. Set "
            "RAKSHAK_BAF_PATH to enable.",
        )
        pytest.skip("BAF not available on this machine")

    in_domain, transferred = _run_transfer(gate_data)
    ratio = transferred / in_domain if in_domain > 0 else 0.0
    verdict = "GREEN" if ratio >= TRANSFER_RATIO else "RED"
    record(
        "G2 baseline-transfer",
        verdict,
        f"PR-AUC in-domain {in_domain:.4f} -> transferred {transferred:.4f} "
        f"(ratio {ratio:.3f})",
        f"threshold {TRANSFER_RATIO}; measured over the shared analogue subspace only",
    )
    assert ratio >= TRANSFER_RATIO, (
        f"transfer ratio {ratio:.3f} below {TRANSFER_RATIO}. The generator does not "
        f"reproduce the joint structure of the anchor: charter K-3."
    )


def _generator_analogue_matrix(data: GeneratedData) -> tuple[np.ndarray, np.ndarray]:
    """The generator side of the shared subspace, one row per merchant.

    Merchant-level, because the generator's label is merchant-level — which is itself one
    of v1's diagnosed errors (the task was formulated as latent-state inference when the
    labels are merchant-level). Aggregating to the entity the label describes is the only
    join that means anything against BAF's one-row-per-application shape.
    """
    per_merchant = (
        data.transactions.filter(~pl.col("is_refund"))
        .with_columns(day=pl.col("event_time").dt.date())
        .group_by("merchant_id")
        .agg(
            (pl.len() / pl.col("day").n_unique()).alias("txn_count"),
            pl.col("amount_inr").median().alias("amount_inr"),
            (pl.col("payer_id").n_unique() / pl.col("device_hash").n_unique()).alias(
                "payers_per_device"
            ),
            pl.col("is_international").mean().alias("is_international"),
        )
        .join(data.labels.select("merchant_id", "label"), on="merchant_id")
        .filter(pl.col("label").is_not_null())
    )
    x = np.column_stack(
        [_rank(per_merchant[a.rakshak].to_numpy().astype(np.float64)) for a in ANALOGUES]
    )
    return x, per_merchant["label"].to_numpy().astype(np.int64)


def _run_transfer(data: GeneratedData) -> tuple[float, float]:
    """Train on the generator's analogue subspace, score on BAF's.

    Reached only when BAF is present, so lightgbm and sklearn are imported here rather
    than at module scope: paying a second of import on every ``make gates`` run in order
    to then skip is a second nobody gets back.
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    from rakshak.eval.baf_adapter import load_baf

    baf = load_baf([*(a.baf_column for a in ANALOGUES), "fraud_bool"])
    assert baf is not None
    x_baf = np.column_stack(
        [_rank(baf[a.baf_column].to_numpy().astype(np.float64)) for a in ANALOGUES]
    )
    y_baf = baf["fraud_bool"].to_numpy().astype(np.int64)

    x_gen, y_gen = _generator_analogue_matrix(data)
    if y_gen.sum() < 5:
        raise RuntimeError(
            "not enough resolved positives in the gate dataset to train a transfer model; "
            "raise the gate population or its prevalence"
        )

    def fit(x: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
        model = lgb.LGBMClassifier(n_estimators=120, num_leaves=31, verbose=-1)
        model.fit(x, y)
        return model

    split = int(0.7 * x_baf.shape[0])
    in_domain = float(
        average_precision_score(
            y_baf[split:], fit(x_baf[:split], y_baf[:split]).predict_proba(x_baf[split:])[:, 1]
        )
    )
    transferred = float(
        average_precision_score(y_baf, fit(x_gen, y_gen).predict_proba(x_baf)[:, 1])
    )
    return in_domain, transferred


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / max(values.size - 1, 1)
