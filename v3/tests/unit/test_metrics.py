"""T-131 — the metric suite. Synthetic fixtures; the generator does not exist yet.

The centrepiece is ``test_random_beating_the_model_fires_floor_fail``: v1 discovered a
random ranker winning on savings at 20% prevalence by accident, weeks late. This asserts
the harness cannot fail to notice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rakshak.eval.metrics import (
    ALL_FLOORS,
    CostParams,
    Floors,
    PerfBudget,
    RungOutput,
    Truth,
    alert_jaccard_wow,
    alerts_per_day,
    build_eval_result,
    cost_of_all_pass,
    day_labels,
    detection_rates,
    expected_calibration_error,
    floors_at_capacity,
    median_ttd,
    pr_auc,
    precision_recall_at_k,
    recall_by_typology,
    row_cost,
    savings_of_actions,
    savings_of_ranking,
    time_to_detection,
    top_k_by_day,
)
from rakshak.schemas import Action, EvalResult, TypologyId

PARAMS = CostParams()
PERF = PerfBudget(p99_latency_ms=8.0, state_bytes_p99=3200.0, model_size_mb=1.4)


# ─────────────────────────── fixture builder ───────────────────────────


def make_world(
    n_merchants: int = 200,
    n_days: int = 60,
    n_fraud: int = 20,
    *,
    seed: int = 7,
) -> tuple[Truth, np.ndarray, np.ndarray]:
    """A merchant population with known fraud, plus the (merchant, day) grid."""
    rng = np.random.default_rng(seed)
    ids = np.array([f"M{i:05d}" for i in range(n_merchants)])
    label = np.zeros(n_merchants, dtype=np.int8)
    label[:n_fraud] = 1
    onset = np.full(n_merchants, np.nan)
    onset[:n_fraud] = rng.integers(2, max(3, n_days // 2), size=n_fraud)
    typology = np.array([""] * n_merchants, dtype=object)
    typology[:n_fraud] = [TypologyId(f"R{1 + i % 9}").value for i in range(n_fraud)]
    loss = np.zeros(n_merchants)
    loss[:n_fraud] = rng.uniform(20_000, 200_000, size=n_fraud)
    truth = Truth(
        merchant_id=ids,
        label=label,
        is_censored=np.zeros(n_merchants, dtype=bool),
        loss_inr=loss,
        onset_day=onset,
        typology=typology,
        volume=rng.uniform(1, 100, size=n_merchants),
    )
    grid_m = np.repeat(ids, n_days)
    grid_d = np.tile(np.arange(n_days), n_merchants)
    return truth, grid_m, grid_d


def output_from_scores(
    truth: Truth, grid_m: np.ndarray, grid_d: np.ndarray, score: np.ndarray, k: int
) -> RungOutput:
    """Alert on the top-K scores per day; everything else PASSes."""
    selected = top_k_by_day(score, grid_d, k)
    action = np.where(selected, Action.REVIEW, Action.PASS)
    return RungOutput(merchant_id=grid_m, day=grid_d, score=score, action=action)


def oracle_ish_score(truth: Truth, grid_m: np.ndarray, grid_d: np.ndarray) -> np.ndarray:
    """A near-perfect ranker: high on fraud merchants at/after onset."""
    lookup = {m: i for i, m in enumerate(truth.merchant_id)}
    idx = np.array([lookup[m] for m in grid_m])
    onset = truth.onset_day[idx]
    hot = (truth.label[idx] == 1) & ~np.isnan(onset) & (grid_d >= np.nan_to_num(onset, nan=np.inf))
    return np.where(hot, 0.95, 0.02)


# ─────────────────────────── capacity primitive ───────────────────────────


def test_top_k_by_day_selects_exactly_k_per_day() -> None:
    day = np.repeat(np.arange(5), 10)
    score = np.tile(np.linspace(0, 1, 10), 5)
    mask = top_k_by_day(score, day, 3)
    for d in range(5):
        assert mask[day == d].sum() == 3


def test_top_k_by_day_takes_the_highest_scores() -> None:
    day = np.zeros(6, dtype=int)
    score = np.array([0.1, 0.9, 0.5, 0.8, 0.2, 0.7])
    assert top_k_by_day(score, day, 2).tolist() == [False, True, False, True, False, False]


def test_top_k_never_exceeds_the_group_size() -> None:
    day = np.array([0, 0, 1])
    mask = top_k_by_day(np.array([0.1, 0.2, 0.3]), day, 10)
    assert mask.all()


def test_top_k_ties_break_deterministically() -> None:
    day = np.zeros(4, dtype=int)
    score = np.full(4, 0.5)
    a = top_k_by_day(score, day, 2)
    b = top_k_by_day(score, day, 2)
    assert a.tolist() == b.tolist() == [True, True, False, False]


def test_negative_capacity_is_refused() -> None:
    with pytest.raises(ValueError, match="K must be >= 0"):
        top_k_by_day(np.array([0.5]), np.array([0]), -1)


# ─────────────────────────── the cost matrix ───────────────────────────


def test_cost_matrix_matches_the_spec_row_for_row() -> None:
    loss = np.array([100_000.0] * 6)
    y = np.array([1, 0, 1, 0, 1, 0])
    action = np.array(
        [Action.PASS, Action.PASS, Action.REVIEW, Action.REVIEW, Action.HOLD, Action.HOLD],
        dtype=object,
    )
    got = row_cost(action, y, loss, PARAMS)
    assert got[0] == pytest.approx(100_000.0)  # PASS  & fraud -> full loss
    assert got[1] == pytest.approx(0.0)  # PASS  & good  -> nothing
    assert got[2] == pytest.approx(250.0 + 0.2 * 100_000.0)  # REVIEW& fraud
    assert got[3] == pytest.approx(250.0)  # REVIEW& good
    assert got[4] == pytest.approx(250.0)  # HOLD  & fraud
    assert got[5] == pytest.approx(8250.0)  # HOLD  & good


def test_an_unknown_action_is_refused_rather_than_costed_as_zero() -> None:
    with pytest.raises(ValueError, match="PASS/REVIEW/HOLD"):
        row_cost(np.array(["dither"], dtype=object), np.array([1]), np.array([1.0]), PARAMS)


def test_all_pass_savings_is_exactly_zero_by_construction() -> None:
    y = np.array([1, 0, 1])
    loss = np.array([1000.0, 0.0, 500.0])
    actions = np.full(3, Action.PASS, dtype=object)
    assert savings_of_actions(actions, y, loss, PARAMS) == pytest.approx(0.0)
    assert cost_of_all_pass(y, loss, PARAMS) == pytest.approx(1500.0)


def test_savings_is_nan_when_there_is_no_fraud_to_save() -> None:
    y = np.zeros(3, dtype=np.int8)
    actions = np.full(3, Action.PASS, dtype=object)
    assert math.isnan(savings_of_actions(actions, y, np.zeros(3), PARAMS))


def test_holding_everything_can_cost_more_than_doing_nothing() -> None:
    """all_hold is a real floor, not a formality: at low prevalence it goes negative."""
    y = np.array([1] + [0] * 99, dtype=np.int8)
    loss = np.array([50_000.0] + [0.0] * 99)
    savings = savings_of_actions(np.full(100, Action.HOLD, dtype=object), y, loss, PARAMS)
    assert savings < 0


# ─────────────────────────── FLOOR-FAIL: the v1 failure, automated ───────────────────────────


def test_floors_failed_by_names_every_floor_not_beaten() -> None:
    floors = Floors(all_pass=0.0, all_hold=-2.0, random_at_k=0.30, volume_rank=0.10)
    assert floors.failed_by(0.40) == []
    assert floors.failed_by(0.20) == ["random_at_k"]
    assert floors.failed_by(-0.5) == ["all_pass", "random_at_k", "volume_rank"]


def test_nan_savings_fails_every_floor() -> None:
    """An unmeasurable result is not a passing one."""
    floors = Floors(all_pass=0.0, all_hold=-2.0, random_at_k=0.3, volume_rank=0.1)
    assert floors.failed_by(float("nan")) == list(ALL_FLOORS)


def test_random_beating_the_model_fires_floor_fail() -> None:
    """The adversarial case T-131 names. A model that ranks *anti*-correlated with fraud
    must be flagged FLOOR-FAIL, and the flag must survive onto the EvalResult row."""
    truth, grid_m, grid_d = make_world(n_merchants=150, n_days=40, n_fraud=15, seed=11)
    good = oracle_ish_score(truth, grid_m, grid_d)
    # Deliberately inverted: the "model" is confidently wrong.
    bad_score = 1.0 - good
    k = 5
    output = output_from_scores(truth, grid_m, grid_d, bad_score, k)

    result = build_eval_result(
        rung=99,
        split="val",
        output=output,
        truth=truth,
        k=k,
        params=PARAMS,
        rng=np.random.default_rng(0),
        perf=PERF,
        oracle_savings=0.9,
        eval_lock_sha="0" * 16,
        open_count=0,
        git_sha="deadbeef",
    )

    assert result.floor_fail, "an anti-correlated ranker cleared every floor — the floors lie"
    assert not result.beats_all_floors
    assert result.savings <= max(
        result.savings_floor_random,
        result.savings_floor_all_pass,
        result.savings_floor_all_hold,
        result.savings_floor_volume_rank,
    )
    # And the diagnosis is legible: PR-AUC is bad too, but the FLOOR-FAIL is what stops it.
    assert result.pr_auc < 0.5


def test_a_good_model_clears_every_floor() -> None:
    """The control. If this also FLOOR-FAILs, the floors are broken, not the model."""
    truth, grid_m, grid_d = make_world(n_merchants=150, n_days=40, n_fraud=15, seed=11)
    score = oracle_ish_score(truth, grid_m, grid_d)
    k = 5
    output = output_from_scores(truth, grid_m, grid_d, score, k)
    result = build_eval_result(
        rung=2,
        split="val",
        output=output,
        truth=truth,
        k=k,
        params=PARAMS,
        rng=np.random.default_rng(0),
        perf=PERF,
        oracle_savings=0.95,
        eval_lock_sha="0" * 16,
        open_count=0,
        git_sha="deadbeef",
    )
    assert result.floor_fail == []
    assert result.beats_all_floors
    assert result.savings > 0


def test_every_savings_row_carries_all_four_floors() -> None:
    """FR-021. Not "when available" — always."""
    truth, grid_m, grid_d = make_world(n_days=30, n_fraud=10)
    y, keep = day_labels(
        RungOutput(grid_m, grid_d, np.zeros(grid_d.size), np.full(grid_d.size, Action.PASS)),
        truth,
    )
    lookup = {m: i for i, m in enumerate(truth.merchant_id)}
    idx = np.array([lookup[m] for m in grid_m])
    floors = floors_at_capacity(
        grid_d, truth.volume[idx], y, truth.loss_inr[idx], 5, PARAMS, np.random.default_rng(1)
    )
    assert len(floors.as_tuple()) == 4
    assert all(not math.isnan(f) for f in floors.as_tuple())


def test_floors_must_be_scored_on_the_rungs_own_rows() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        floors_at_capacity(
            np.arange(5),
            np.arange(4).astype(float),
            np.zeros(5, np.int8),
            np.zeros(5),
            2,
            PARAMS,
            np.random.default_rng(0),
        )


def test_savings_of_ranking_holds_the_decision_layer_fixed() -> None:
    """Same rows, same K, same action — only the score differs. That is the whole point."""
    day = np.repeat(np.arange(10), 20)
    rng = np.random.default_rng(3)
    y = (rng.random(200) < 0.05).astype(np.int8)
    loss = np.where(y == 1, 50_000.0, 0.0)
    perfect = y.astype(float)
    inverted = 1.0 - perfect
    assert savings_of_ranking(perfect, day, y, loss, 3, PARAMS) > savings_of_ranking(
        inverted, day, y, loss, 3, PARAMS
    )


# ─────────────────────────── ranking + calibration ───────────────────────────


def test_pr_auc_is_nan_on_a_single_class_split_not_a_fabricated_number() -> None:
    assert math.isnan(pr_auc(np.zeros(10, np.int8), np.random.default_rng(0).random(10)))


def test_ece_is_zero_for_a_perfectly_calibrated_score() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.random(20_000) < p).astype(np.int8)
    assert expected_calibration_error(y, p) < 0.02


def test_ece_is_large_for_a_confidently_wrong_score() -> None:
    y = np.zeros(1000, dtype=np.int8)
    p = np.full(1000, 0.9)
    assert expected_calibration_error(y, p) == pytest.approx(0.9, abs=1e-6)


def test_ece_uses_equal_mass_bins() -> None:
    """At 1.5% prevalence, equal-width bins would put nearly every score in bin 0 and
    report the calibration of one bin as the model's."""
    rng = np.random.default_rng(9)
    p = np.concatenate([rng.uniform(0.0, 0.01, 990), rng.uniform(0.9, 1.0, 10)])
    y = np.concatenate([np.zeros(990, np.int8), np.ones(10, np.int8)])
    # Equal-mass: the ten confident rows land in the top bin and are graded there.
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


# ─────────────────────────── TTD and censoring ───────────────────────────


def _ttd_world() -> tuple[Truth, RungOutput]:
    """Four merchants: caught fast, caught slow, never caught, and label-censored."""
    ids = np.array(["fast", "slow", "never", "censored"])
    truth = Truth(
        merchant_id=ids,
        label=np.array([1, 1, 1, 0], dtype=np.int8),
        is_censored=np.array([False, False, False, True]),
        loss_inr=np.array([10_000.0, 10_000.0, 10_000.0, 0.0]),
        onset_day=np.array([10.0, 10.0, 10.0, 10.0]),
        typology=np.array(["R1", "R2", "R3", "R4"], dtype=object),
        volume=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    days = np.arange(60)
    m = np.repeat(ids, 60)
    d = np.tile(days, 4)
    action = np.full(m.size, Action.PASS, dtype=object)
    action[(m == "fast") & (d == 13)] = Action.REVIEW
    action[(m == "slow") & (d == 35)] = Action.HOLD
    action[(m == "censored") & (d == 11)] = Action.REVIEW
    output = RungOutput(m, d, np.full(m.size, 0.5), action)
    return truth, output


def test_ttd_is_measured_from_drift_onset() -> None:
    truth, output = _ttd_world()
    det = time_to_detection(output, truth)
    got = dict(zip(det.merchant_id.tolist(), det.ttd.tolist(), strict=True))
    assert got["fast"] == 3.0
    assert got["slow"] == 25.0
    assert math.isinf(got["never"])


def test_label_censored_merchants_are_not_scored_for_ttd() -> None:
    truth, output = _ttd_world()
    det = time_to_detection(output, truth)
    assert "censored" not in det.merchant_id.tolist()


def test_median_ttd_does_not_silently_drop_the_never_detected() -> None:
    """The flattering bug: mean over the caught ones only. Here 1 of 3 is never caught,
    so the median is finite; make 2 of 3 never caught and it must go to inf."""
    truth, output = _ttd_world()
    assert median_ttd(time_to_detection(output, truth)) == 25.0

    action = output.action.copy()
    action[(output.merchant_id == "slow")] = Action.PASS
    worse = RungOutput(output.merchant_id, output.day, output.score, action)
    assert math.isinf(median_ttd(time_to_detection(worse, truth)))


def test_an_alert_before_onset_is_not_an_early_detection() -> None:
    truth, output = _ttd_world()
    action = output.action.copy()
    action[(output.merchant_id == "never") & (output.day == 2)] = Action.REVIEW
    early = RungOutput(output.merchant_id, output.day, output.score, action)
    det = time_to_detection(early, truth)
    got = dict(zip(det.merchant_id.tolist(), det.ttd.tolist(), strict=True))
    assert math.isinf(got["never"]), "a pre-onset alert was counted as a catch"
    assert (det.ttd >= 0).all(), "a negative TTD means an alert was credited before onset"


def test_detection_rates_drop_merchants_not_observed_long_enough() -> None:
    """Administrative right-censoring. A merchant whose onset is 5 days before the window
    ends has not "failed to be detected within 30 days" — it was not observed that long,
    and counting it as a miss understates every model equally and wrongly."""
    ids = np.array(["late"])
    truth = Truth(
        merchant_id=ids,
        label=np.array([1], np.int8),
        is_censored=np.array([False]),
        loss_inr=np.array([1000.0]),
        onset_day=np.array([55.0]),
        typology=np.array(["R1"], dtype=object),
        volume=np.array([1.0]),
    )
    d = np.arange(60)
    output = RungOutput(
        np.repeat(ids, 60), d, np.full(60, 0.5), np.full(60, Action.PASS, dtype=object)
    )
    rates = detection_rates(time_to_detection(output, truth))
    # Onset day 55, window ends day 59: four days of follow-up. Eligible for no horizon.
    assert math.isnan(rates[7])
    assert math.isnan(rates[30]), "a merchant with 4 days of follow-up was graded at d30"


def test_recall_by_typology_covers_all_nine_and_uses_nan_for_absent_ones() -> None:
    truth, output = _ttd_world()
    recall = recall_by_typology(time_to_detection(output, truth))
    assert set(recall) == set(TypologyId)
    assert recall[TypologyId.R1] == 1.0  # caught at d3
    assert recall[TypologyId.R3] == 0.0  # never caught, but observed long enough
    assert math.isnan(recall[TypologyId.R9]), "an absent typology was reported as 0.0 recall"


# ─────────────────────────── operational ───────────────────────────


def test_alerts_per_day_and_precision_recall_at_k() -> None:
    truth, grid_m, grid_d = make_world(n_merchants=100, n_days=20, n_fraud=10, seed=2)
    k = 4
    output = output_from_scores(
        truth, grid_m, grid_d, oracle_ish_score(truth, grid_m, grid_d), k
    )
    assert alerts_per_day(output) <= k
    y, keep = day_labels(output, truth)
    precision, recall = precision_recall_at_k(output, y, keep)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0


def test_alerts_per_day_above_k_is_a_hard_error_not_a_warning() -> None:
    truth, grid_m, grid_d = make_world(n_merchants=50, n_days=10, n_fraud=5)
    output = RungOutput(
        grid_m, grid_d, np.full(grid_d.size, 0.5), np.full(grid_d.size, Action.REVIEW, dtype=object)
    )
    with pytest.raises(ValueError, match="exceeds capacity"):
        build_eval_result(
            rung=1,
            split="val",
            output=output,
            truth=truth,
            k=5,
            params=PARAMS,
            rng=np.random.default_rng(0),
            perf=PERF,
            oracle_savings=0.5,
            eval_lock_sha="0" * 16,
            open_count=0,
            git_sha="x",
        )


def test_alert_jaccard_is_one_for_a_perfectly_stable_alert_list() -> None:
    ids = np.array([f"S{i}" for i in range(10)])
    truth = Truth(
        merchant_id=ids,
        label=np.zeros(10, np.int8),
        is_censored=np.zeros(10, bool),
        loss_inr=np.zeros(10),
        onset_day=np.full(10, np.nan),
        typology=np.array([""] * 10, dtype=object),
        volume=np.ones(10),
    )
    d = np.tile(np.arange(28), 10)
    m = np.repeat(ids, 28)
    action = np.where(np.isin(m, ["S0", "S1", "S2"]), Action.REVIEW, Action.PASS)
    assert alert_jaccard_wow(RungOutput(m, d, np.full(m.size, 0.5), action), truth) == 1.0


def test_alert_jaccard_is_near_zero_for_a_churning_alert_list() -> None:
    ids = np.array([f"S{i}" for i in range(28)])
    truth = Truth(
        merchant_id=ids,
        label=np.zeros(28, np.int8),
        is_censored=np.zeros(28, bool),
        loss_inr=np.zeros(28),
        onset_day=np.full(28, np.nan),
        typology=np.array([""] * 28, dtype=object),
        volume=np.ones(28),
    )
    d = np.tile(np.arange(28), 28)
    m = np.repeat(ids, 28)
    # each week alerts an entirely different merchant
    week = d // 7
    who = np.array([f"S{w}" for w in week])
    action = np.where(m == who, Action.REVIEW, Action.PASS)
    assert alert_jaccard_wow(RungOutput(m, d, np.full(m.size, 0.5), action), truth) == 0.0


def test_jaccard_ignores_drifting_merchants() -> None:
    """Churn on merchants that are genuinely changing is correct behaviour, not noise."""
    truth, output = _ttd_world()
    assert math.isnan(alert_jaccard_wow(output, truth))


# ─────────────────────────── the row itself ───────────────────────────


def test_every_eval_result_field_is_produced() -> None:
    """T-131's done-when: every field in the EvalResult schema is produced."""
    truth, grid_m, grid_d = make_world(n_merchants=120, n_days=45, n_fraud=12, seed=4)
    k = 4
    output = output_from_scores(
        truth, grid_m, grid_d, oracle_ish_score(truth, grid_m, grid_d), k
    )
    result = build_eval_result(
        rung=2,
        split="val",
        output=output,
        truth=truth,
        k=k,
        params=PARAMS,
        rng=np.random.default_rng(0),
        perf=PERF,
        oracle_savings=0.9,
        eval_lock_sha="a" * 16,
        open_count=0,
        git_sha="cafe1234",
    )
    for name in EvalResult.__dataclass_fields__:
        assert hasattr(result, name), name
        assert getattr(result, name) is not None, name
    assert set(result.recall_by_typology) == set(TypologyId)
    assert result.cost_scenario == "base"
    assert result.eval_lock_sha == "a" * 16
    assert result.open_count == 0


def test_prevalence_is_computed_not_supplied() -> None:
    """FR-021, enforced by construction: there is no prevalence argument to pass wrong."""
    truth, grid_m, grid_d = make_world(n_merchants=200, n_days=30, n_fraud=3, seed=6)
    output = output_from_scores(truth, grid_m, grid_d, oracle_ish_score(truth, grid_m, grid_d), 3)
    result = build_eval_result(
        rung=0,
        split="val",
        output=output,
        truth=truth,
        k=3,
        params=PARAMS,
        rng=np.random.default_rng(0),
        perf=PERF,
        oracle_savings=0.99,
        eval_lock_sha="b" * 16,
        open_count=0,
        git_sha="x",
    )
    assert result.prevalence == pytest.approx(3 / 200)


def test_censored_merchants_are_excluded_from_prevalence() -> None:
    truth, _, _ = make_world(n_merchants=100, n_days=10, n_fraud=10)
    censored = Truth(
        merchant_id=truth.merchant_id,
        label=truth.label,
        is_censored=np.arange(100) >= 80,
        loss_inr=truth.loss_inr,
        onset_day=truth.onset_day,
        typology=truth.typology,
        volume=truth.volume,
    )
    assert censored.prevalence == pytest.approx(10 / 80)


def test_gap_to_oracle_is_a_fraction_of_the_achievable() -> None:
    truth, grid_m, grid_d = make_world(n_merchants=120, n_days=40, n_fraud=12, seed=8)
    output = output_from_scores(truth, grid_m, grid_d, oracle_ish_score(truth, grid_m, grid_d), 4)
    result = build_eval_result(
        rung=2,
        split="val",
        output=output,
        truth=truth,
        k=4,
        params=PARAMS,
        rng=np.random.default_rng(0),
        perf=PERF,
        oracle_savings=1.0,
        eval_lock_sha="c" * 16,
        open_count=0,
        git_sha="x",
    )
    assert result.gap_to_oracle == pytest.approx(1.0 - result.savings)


def test_a_scored_merchant_missing_from_truth_is_an_error() -> None:
    truth, grid_m, grid_d = make_world(n_merchants=10, n_days=5, n_fraud=2)
    ghost_m = np.append(grid_m, "GHOST")
    ghost_d = np.append(grid_d, 0)
    output = RungOutput(
        ghost_m,
        ghost_d,
        np.full(ghost_m.size, 0.5),
        np.full(ghost_m.size, Action.PASS, dtype=object),
    )
    with pytest.raises(KeyError):
        day_labels(output, truth)


def test_build_eval_result_runs_the_leakage_assertion_on_every_eval() -> None:
    """10-eval-harness-spec.md §3 says this runs on every eval, so it is not optional and
    not the caller's job to remember."""
    from rakshak.eval.oracle import LeakageError

    truth, grid_m, grid_d = make_world(n_merchants=120, n_days=30, n_fraud=12, seed=13)
    output = output_from_scores(truth, grid_m, grid_d, oracle_ish_score(truth, grid_m, grid_d), 4)
    with pytest.raises(LeakageError, match="beats the perfect-foresight oracle"):
        build_eval_result(
            rung=3,
            split="val",
            output=output,
            truth=truth,
            k=4,
            params=PARAMS,
            rng=np.random.default_rng(0),
            perf=PERF,
            oracle_savings=-5.0,  # an impossible ceiling: any real savings beats it
            eval_lock_sha="d" * 16,
            open_count=0,
            git_sha="x",
        )
