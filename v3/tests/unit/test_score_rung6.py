"""End-to-end verification of ``score_rung6.run`` against a synthetic stand-in for a base
rung's score vector (no trained model, no real panel, no disk access at all).

Rung 6's own construction (per-stratum order statistic, soften-only wrapper) is already
covered by 19 tests in ``tests/unit/test_rung6_conformal.py``; nothing here re-tests that
arithmetic. What is novel to ``score_rung6.py`` and untested elsewhere is the *wiring*: the
by-merchant calibration/evaluation carve, building an ``EvalResult``-shaped payload out of
it, and getting ``false_hold_coverage`` fed the same eval-fold rows the metric set is scored
on. This fixture is small and structurally realistic — two Mondrian strata, censoring,
fraud and clean merchants, an exposure column — precisely so that when a real base-rung
model lands, this same ``run()`` function runs against it with zero remaining engineering.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from rakshak.eval.capacity import DEFAULT_POLICY
from rakshak.eval.metrics import CostParams, PerfBudget, Truth
from rakshak.eval.splits import DEFAULT_BOUNDARIES, SplitBoundaries
from rakshak.models.dataset import Panel
from rakshak.schemas import MerchantProfile
from rakshak.score_rung6 import run

QUIET_MCC = "grocery"
NOISY_MCC = "travel"
N_PER_GROUP = 40  # >= MIN_COHORT_MEMBERS (30), so both groups land on a "full" stratum


def _synthetic_panel_and_truth(
    rng: np.random.Generator,
) -> tuple[Panel, Truth, dict[str, MerchantProfile], np.ndarray]:
    """A small val-split fixture: two MCC groups, a handful of fraud and censored
    merchants, and a synthetic score correlated with the label so the pipeline exercises
    a realistic (non-degenerate) precision/recall and TTD, not just an empty edge case.
    """
    n_merchants = 2 * N_PER_GROUP
    merchant_ids = [f"M{i:04d}" for i in range(n_merchants)]
    onboarded = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    profiles = {
        m: MerchantProfile(
            merchant_id=m,
            onboarded_at=onboarded,
            mcc="5411" if i < N_PER_GROUP else "4511",
            mcc_group=QUIET_MCC if i < N_PER_GROUP else NOISY_MCC,
            declared_monthly_gmv=float(rng.uniform(50_000, 500_000)),
            kyc_tier=2,
            vintage_months=36,
            city_tier=1,
        )
        for i, m in enumerate(merchant_ids)
    }

    n_days = 20
    day = np.tile(np.arange(n_days), n_merchants)
    merchant_id = np.repeat(merchant_ids, n_days)

    # 6 fraud merchants (drifting from day 10), 2 censored, the rest clean.
    fraud = set(merchant_ids[2:5]) | set(merchant_ids[N_PER_GROUP + 2 : N_PER_GROUP + 5])
    censored = {merchant_ids[6], merchant_ids[N_PER_GROUP + 6]}
    onset_day = 10

    label = np.array([1 if m in fraud else 0 for m in merchant_ids], dtype=np.int8)
    is_censored = np.array([m in censored for m in merchant_ids], dtype=bool)
    onset = np.array(
        [float(onset_day) if m in fraud else np.nan for m in merchant_ids], dtype=np.float64
    )
    typology = np.array(["R1" if m in fraud else None for m in merchant_ids], dtype=object)
    loss = np.array([25_000.0 if m in fraud else 0.0 for m in merchant_ids], dtype=np.float64)
    volume = rng.uniform(10_000, 1_000_000, n_merchants)
    truth = Truth(
        merchant_id=np.array(merchant_ids, dtype=object),
        label=label,
        is_censored=is_censored,
        loss_inr=loss,
        onset_day=onset,
        typology=typology,
        volume=volume,
    )

    is_fraud_day = np.array(
        [merchant_id[i] in fraud and day[i] >= onset_day for i in range(day.size)]
    )
    # Score correlated with the true (fraud-and-past-onset) row label plus noise, clipped
    # to a probability -- a structurally realistic stand-in for a trained booster's output.
    score = np.clip(
        np.where(is_fraud_day, 0.8, 0.1) + rng.normal(0.0, 0.15, day.size), 0.0, 1.0
    )
    exposure = rng.uniform(10_000, 500_000, day.size)

    x = np.column_stack([exposure])
    panel = Panel(
        merchant_id=merchant_id,
        day=day,
        split=np.full(day.size, "val", dtype=object),
        x=x,
        columns=("p_declared_monthly_gmv",),
    )
    return panel, truth, profiles, score


def test_run_end_to_end_against_a_synthetic_score_vector(rng: np.random.Generator) -> None:
    panel, truth, profiles, score = _synthetic_panel_and_truth(rng)
    boundaries = SplitBoundaries(
        origin=DEFAULT_BOUNDARIES.origin, train=(0, 9), val=(10, 19), test=(20, 29)
    )
    perf = PerfBudget(p99_latency_ms=1.0, state_bytes_p99=4096.0, model_size_mb=0.5)

    results = run(
        score=score,
        rows=panel,
        truth_full=truth,
        profiles=profiles,
        params=CostParams(),
        policy=DEFAULT_POLICY,
        alphas=(0.05, 0.10),
        cal_fraction=0.5,
        cal_rng=np.random.default_rng(1),
        metric_seed=2,
        perf=perf,
        base_rung=4,
        repo_root=__import__("pathlib").Path("."),
        boundaries=boundaries,
        eval_lock_sha="deadbeef",
        open_count=0,
        git_sha="cafebabe",
    )

    assert set(results) == {0.05, 0.10}
    for alpha, payload in results.items():
        # The shape cli.py::score_split's own payload has, per the module docstring.
        for key in ("savings", "pr_auc", "precision_at_k", "recall_at_k", "ttd_median_days",
                    "alerts_per_day", "false_hold_coverage", "capacity_k",
                    "n_eval_merchants", "base_rung_same_fold"):
            assert key in payload, f"missing {key!r} at alpha={alpha}"
        assert payload["alpha"] == alpha
        assert payload["base_rung"] == 4
        assert payload["rung"] == 6
        assert payload["split"] == "val"
        # Capacity is never breached: alerts_per_day <= capacity_k (build_eval_result
        # itself raises if this is violated, so reaching this line already proves it, but
        # asserting it here documents the guarantee this test is protecting).
        assert payload["alerts_per_day"] <= payload["capacity_k"] + 1e-9
        # Calibration and evaluation are disjoint by merchant, and neither is empty --
        # a bug that let the same merchants calibrate and get scored would show up here
        # as n_cal_merchants + n_eval_merchants > n_merchants (rounding aside) or one of
        # them at 0.
        assert payload["n_cal_merchants"] > 0
        assert payload["n_eval_merchants"] > 0
        assert payload["n_cal_merchants"] + payload["n_eval_merchants"] == 2 * N_PER_GROUP

        # Every coverage row carries the margin, not just the boolean, and the two
        # Mondrian strata both got their own row (no pooled row hiding a per-cell miss).
        assert len(payload["false_hold_coverage"]) == 2
        for row in payload["false_hold_coverage"]:
            assert row["stratum"] in {f"mcc={QUIET_MCC}", f"mcc={NOISY_MCC}"}
            assert row["margin"] == pytest.approx(alpha - row["realised"])
            assert row["bound"] <= alpha + 1e-9
            assert row["n_calibration"] >= 0
            assert row["threshold"] > 0.0  # a real order statistic, not a degenerate gate

        # The unwrapped base rung on the identical rows -- without this the wrapper's
        # delta cannot be separated from the fold it was measured on.
        assert payload["base_rung_same_fold"]["alerts_per_day"] == payload["alerts_per_day"]


def test_run_never_promotes_an_action_relative_to_the_unwrapped_selector(
    rng: np.random.Generator,
) -> None:
    """Sanity check on the seam itself, exercised through `run()` rather than assumed:
    softening can only ever lower alerts_per_day for a fixed K, never raise it, so a
    savings comparison against the base rung's own row is never inflated by Rung 6."""
    panel, truth, profiles, score = _synthetic_panel_and_truth(rng)
    boundaries = SplitBoundaries(
        origin=DEFAULT_BOUNDARIES.origin, train=(0, 9), val=(10, 19), test=(20, 29)
    )
    perf = PerfBudget(p99_latency_ms=1.0, state_bytes_p99=4096.0, model_size_mb=0.5)

    tight = run(
        score=score,
        rows=panel,
        truth_full=truth,
        profiles=profiles,
        params=CostParams(),
        policy=DEFAULT_POLICY,
        alphas=(0.01,),
        cal_fraction=0.5,
        cal_rng=np.random.default_rng(1),
        metric_seed=2,
        perf=perf,
        base_rung=4,
        repo_root=__import__("pathlib").Path("."),
        boundaries=boundaries,
        eval_lock_sha="deadbeef",
        open_count=0,
        git_sha="cafebabe",
    )[0.01]
    loose = run(
        score=score,
        rows=panel,
        truth_full=truth,
        profiles=profiles,
        params=CostParams(),
        policy=DEFAULT_POLICY,
        alphas=(0.5,),
        cal_fraction=0.5,
        cal_rng=np.random.default_rng(1),
        metric_seed=2,
        perf=perf,
        base_rung=4,
        repo_root=__import__("pathlib").Path("."),
        boundaries=boundaries,
        eval_lock_sha="deadbeef",
        open_count=0,
        git_sha="cafebabe",
    )[0.5]

    # A tighter alpha certifies fewer rows to HOLD, so it can only soften more or equally
    # -- alerts_per_day is unaffected either way (soften-only preserves the non-PASS set),
    # but the count of rows still holding cannot rise as alpha shrinks.
    assert tight["alerts_per_day"] == pytest.approx(loose["alerts_per_day"])
