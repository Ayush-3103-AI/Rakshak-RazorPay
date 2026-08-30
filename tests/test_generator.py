"""Tests for the synthetic generator (T-0003).

These assert properties of the DATA, not of any model. The separability test in particular is
a statement about the generator: if a typology were not separable by a single emission
statistic, no detector could be expected to find it and the ground truth would be meaningless.

The same test is what keeps SLOW_RAMP honest. It asserts that SLOW_RAMP is the *least*
separable of the five. If a future change makes SLOW_RAMP easy, this test fails — which is the
point (CLAUDE.md non-negotiable #1: do not tune the adversarial typology away).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from rakshak.cli import seed_everything
from rakshak.config import GENERATOR_START_DATE, SPLIT_DAY_BOUNDS
from rakshak.eval.splits import assign_merchant_groups
from rakshak.generator import (
    MIN_ONSET_DAY,
    MIN_POST_ONSET_DAYS,
    NO_TYPOLOGY,
    SEGMENTS,
    STATE_PATH_COLUMNS,
    STATES,
    TRANSACTION_COLUMNS,
    TYPOLOGIES,
    GeneratorConfig,
    generate,
    write_outputs,
)
from rakshak.generator.generate import _apply_shock, _baseline_profile, _inject_bust_out

CONFIG = GeneratorConfig(n_merchants=150, horizon_days=270, fraud_rate=0.5)
"""Test population. `fraud_rate` is deliberately far above the production default so each of
the five typologies gets ~15 merchants — enough for a stable rank statistic."""

PRE_FRACTION = 0.35
POST_FRACTION = 0.75
"""Comparison windows, as fractions of the horizon. Since T-0003b the onset schedule spans
days 63-235 (`generator.onset_window`), so the pre-window is clean for late-onset merchants
and partly contaminated for early-onset ones. That contamination is intentional: a slow
evader is supposed to poison the baseline it is measured against. The post-window catches a
test-window merchant mid-typology rather than after it has finished, which is exactly the
"still drifting at the horizon end" case the frozen split now has to score."""


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one population at the global seed, shared across tests."""
    return generate(CONFIG, seed_everything(42))


def _frame_hash(df: pd.DataFrame) -> int:
    """Order-sensitive content hash of a frame."""
    return int(pd.util.hash_pandas_object(df, index=True).sum())


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney rank AUC: P(a random positive scores above a random negative)."""
    ranks = rankdata(np.concatenate([positive, negative]))
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _suspicion_scores(transactions: pd.DataFrame) -> pd.DataFrame:
    """One hand-picked emission statistic per typology, higher meaning more suspicious.

    Each is a within-merchant comparison of a late window against an early one, which is the
    normalisation P-02 mandates. Units: BUST_OUT, CATEGORY_DRIFT and SLOW_RAMP are
    natural-log ratios, LAUNDERING_ENDPOINT a share in [0, 1], REFUND_COLLUSION a difference
    of rates in [0, 1].
    """
    days = CONFIG.horizon_days
    day = (transactions["timestamp"] - pd.Timestamp(GENERATOR_START_DATE)).dt.days
    pre_mask, post_mask = day < PRE_FRACTION * days, day >= POST_FRACTION * days
    pre_days, post_days = PRE_FRACTION * days, (1.0 - POST_FRACTION) * days

    merchants = sorted(transactions["merchant_id"].unique())
    pre = transactions[pre_mask].groupby("merchant_id")
    post = transactions[post_mask].groupby("merchant_id")

    def col(grouped: pd.core.groupby.DataFrameGroupBy, name: str, how: str) -> pd.Series:
        return getattr(grouped[name], how)().reindex(merchants)

    n_pre = pre.size().reindex(merchants).fillna(0.0)
    n_post = post.size().reindex(merchants).fillna(0.0)
    median_pre = col(pre, "amount", "median")
    gross_pre = col(pre, "amount", "sum").fillna(0.0)
    gross_post = col(post, "amount", "sum").fillna(0.0)

    known_payers = set(transactions.loc[pre_mask, "payer_id"])
    returning = (
        transactions[post_mask]
        .assign(_r=transactions.loc[post_mask, "payer_id"].isin(known_payers))
        .groupby("merchant_id")["_r"]
        .mean()
        .reindex(merchants)
        .fillna(1.0)
    )

    scores = {
        # The transaction rate moved hard, in either direction. Two-sided since T-0003b:
        # a merchant whose onset falls in the test window is still ramping UP at the horizon
        # end rather than already gone quiet, and the one-sided "went quiet" form scored
        # those four merchants backwards (AUC 0.862 vs 0.953 two-sided). The statistic, not
        # the threshold, was wrong.
        "BUST_OUT": np.abs(
            np.log((n_post / post_days + 1e-9) / (n_pre / pre_days + 1e-9))
        ),
        # No organic returns: almost no post-window payer was ever seen before.
        "LAUNDERING_ENDPOINT": -returning,
        # Ticket size moved, in either direction.
        "CATEGORY_DRIFT": np.abs(
            np.log(col(post, "amount", "median").fillna(median_pre) / median_pre)
        ),
        # Refund rate jumped.
        "REFUND_COLLUSION": col(post, "is_refund", "mean").fillna(0.0)
        - col(pre, "is_refund", "mean").fillna(0.0),
        # Gross processed per day grew — the only trace a slow evader leaves, and one that
        # organically growing healthy merchants leave too.
        "SLOW_RAMP": np.log((gross_post / post_days + 1.0) / (gross_pre / pre_days + 1.0)),
    }
    return pd.DataFrame(scores).fillna(0.0)


# ------------------------------------------------------------------------------------------
# FR-001 — schema, row count, determinism
# ------------------------------------------------------------------------------------------


def test_transaction_schema(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    transactions, _ = data
    assert tuple(transactions.columns) == TRANSACTION_COLUMNS
    assert transactions["merchant_id"].dtype == object
    assert transactions["payer_id"].dtype == object
    assert transactions["method"].dtype == object
    assert transactions["mcc"].dtype == object
    assert pd.api.types.is_datetime64_ns_dtype(transactions["timestamp"])
    assert transactions["amount"].dtype == np.float64
    assert transactions["is_refund"].dtype == np.bool_
    assert transactions["is_chargeback"].dtype == np.bool_

    assert len(transactions) > 10_000
    assert not transactions.isna().any().any()
    assert (transactions["amount"] > 0).all()
    assert not (transactions["is_refund"] & transactions["is_chargeback"]).any()
    assert transactions["method"].nunique() == 5


def test_state_path_schema_covers_every_merchant(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    transactions, state_paths = data
    assert tuple(state_paths.columns) == STATE_PATH_COLUMNS
    assert set(state_paths["state"]) <= set(STATES)
    assert set(state_paths["typology"]) == {NO_TYPOLOGY, *TYPOLOGIES}
    assert state_paths["merchant_id"].nunique() == CONFIG.n_merchants
    assert set(transactions["merchant_id"]) <= set(state_paths["merchant_id"])

    # Row-count assertion: the ground truth accounts for every transaction exactly once.
    assert int(state_paths["n_txns"].sum()) == len(transactions)


def test_determinism(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    transactions, state_paths = data
    same_txn, same_paths = generate(CONFIG, seed_everything(42))
    assert _frame_hash(same_txn) == _frame_hash(transactions)
    assert _frame_hash(same_paths) == _frame_hash(state_paths)

    other_txn, _ = generate(CONFIG, seed_everything(43))
    assert _frame_hash(other_txn) != _frame_hash(transactions)


def test_parquet_roundtrip(tmp_path, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    transactions, state_paths = data
    txn_path, path_path = write_outputs(transactions, state_paths, tmp_path)
    reloaded = pd.read_parquet(txn_path)
    assert tuple(reloaded.columns) == TRANSACTION_COLUMNS
    assert len(reloaded) == len(transactions)
    assert tuple(pd.read_parquet(path_path).columns) == STATE_PATH_COLUMNS


# ------------------------------------------------------------------------------------------
# FR-002 — population heterogeneity
# ------------------------------------------------------------------------------------------


def test_population_heterogeneity(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    transactions, _ = data
    per_merchant = transactions.groupby("merchant_id")
    aov = per_merchant["amount"].mean()
    monthly_volume = per_merchant.size() / (CONFIG.horizon_days / 30.0)

    assert aov.max() / aov.min() >= 100.0, f"AOV spread only {aov.max() / aov.min():.1f}x"
    ratio = monthly_volume.max() / monthly_volume.min()
    assert ratio >= 100.0, f"monthly-volume spread only {ratio:.1f}x"
    assert transactions["mcc"].nunique() >= 6


# ------------------------------------------------------------------------------------------
# FR-003 — transition times, not just labels
# ------------------------------------------------------------------------------------------


def test_transitions_carry_time_and_index(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    transactions, state_paths = data
    epoch = pd.Timestamp(GENERATOR_START_DATE)
    horizon_end = epoch + pd.Timedelta(days=CONFIG.horizon_days)
    streams = {mid: df for mid, df in transactions.groupby("merchant_id")}

    fraudulent = state_paths[state_paths["typology"] != NO_TYPOLOGY]
    assert fraudulent.groupby("merchant_id").size().min() >= 2, "a typology must change state"
    healthy = state_paths[state_paths["typology"] == NO_TYPOLOGY]
    assert (healthy["state"] == "HEALTHY").all()
    assert healthy.groupby("merchant_id").size().max() == 1

    for merchant_id, segments in state_paths.groupby("merchant_id"):
        segments = segments.sort_values("segment_index")
        assert segments["start_timestamp"].iloc[0] == epoch
        assert segments["end_timestamp"].iloc[-1] == horizon_end
        # Segments tile the horizon with no gap and no overlap.
        assert (
            segments["start_timestamp"].to_numpy()[1:]
            == segments["end_timestamp"].to_numpy()[:-1]
        ).all()
        assert (segments["end_day"] > segments["start_day"]).all()

        stream = streams.get(merchant_id)
        if stream is None:
            continue
        for row in segments.itertuples():
            if row.n_txns == 0:
                continue
            window = stream.iloc[row.start_txn_index : row.start_txn_index + row.n_txns]
            assert window["timestamp"].min() >= row.start_timestamp
            assert window["timestamp"].max() < row.end_timestamp


# ------------------------------------------------------------------------------------------
# FR-004 / FR-005 — separability, and the adversarial typology staying adversarial
# ------------------------------------------------------------------------------------------


def test_each_typology_is_separable(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    transactions, state_paths = data
    scores = _suspicion_scores(transactions)
    labels = state_paths.groupby("merchant_id")["typology"].first().reindex(scores.index)
    healthy = labels == NO_TYPOLOGY
    assert healthy.sum() >= 20

    auc = {t: _auc(scores.loc[labels == t, t].to_numpy(), scores.loc[healthy, t].to_numpy())
           for t in TYPOLOGIES}
    print("\nseparability AUC vs healthy: " + ", ".join(f"{k}={v:.3f}" for k, v in auc.items()))

    for typology in ("BUST_OUT", "LAUNDERING_ENDPOINT", "CATEGORY_DRIFT", "REFUND_COLLUSION"):
        assert auc[typology] >= 0.90, f"{typology} not separable: AUC={auc[typology]:.3f}"


def test_slow_ramp_stays_adversarial(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """FR-005. SLOW_RAMP must be detectable in principle but the hardest of the five.

    The lower bound stops it degenerating into pure noise (an undetectable typology would make
    the reported failure meaningless). The upper bound is the real assertion: if SLOW_RAMP ever
    becomes as easy as the others, the adversarial case has been tuned away.
    """
    transactions, state_paths = data
    scores = _suspicion_scores(transactions)
    labels = state_paths.groupby("merchant_id")["typology"].first().reindex(scores.index)
    healthy = labels == NO_TYPOLOGY

    auc = {t: _auc(scores.loc[labels == t, t].to_numpy(), scores.loc[healthy, t].to_numpy())
           for t in TYPOLOGIES}
    slow = auc["SLOW_RAMP"]
    print(f"\nSLOW_RAMP separability AUC = {slow:.3f} (others: "
          + ", ".join(f"{k}={v:.3f}" for k, v in auc.items() if k != "SLOW_RAMP") + ")")

    assert slow >= 0.60, f"SLOW_RAMP is noise, not a typology: AUC={slow:.3f}"
    assert slow < 0.90, f"SLOW_RAMP is no longer adversarial: AUC={slow:.3f}"
    assert slow == min(auc.values()), "SLOW_RAMP must be the hardest typology"


# ------------------------------------------------------------------------------------------
# T-0003b — the onset schedule must compose with the frozen evaluation split
# ------------------------------------------------------------------------------------------


def test_onset_falls_inside_every_split_window(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Every split must hold merchants of every typology transitioning INSIDE its window.

    Before T-0003b onsets spanned roughly days 67-187 only, so every bad merchant in the
    validate and test splits had already gone bad before its window opened: detection lag
    was undefined there and the drift-detection claim was being measured on the different,
    easier task of spotting an already-bad merchant. The split boundaries are frozen
    (06-requirements.md §3), so the generator is what moved.
    """
    _, state_paths = data
    groups = assign_merchant_groups(state_paths)
    bad = state_paths[
        (state_paths["typology"] != NO_TYPOLOGY) & (state_paths["state"] != "HEALTHY")
    ]
    onset = bad.groupby("merchant_id")["start_day"].min()
    typology = state_paths.groupby("merchant_id")["typology"].first().reindex(onset.index)
    split = groups.reindex(onset.index)

    assert onset.min() >= MIN_ONSET_DAY, (
        f"onset at day {onset.min()} lands inside the feature layer's burn-in "
        f"(< {MIN_ONSET_DAY} days), contaminating the baseline it is measured against"
    )
    assert onset.max() <= CONFIG.horizon_days - MIN_POST_ONSET_DAYS

    for name, (start, end) in SPLIT_DAY_BOUNDS.items():
        for typ in TYPOLOGIES:
            rows = (split == name) & (typology == typ)
            inside = ((onset >= start) & (onset < end) & rows).sum()
            assert inside >= 3, (
                f"split {name!r} holds only {inside} {typ} merchant(s) transitioning inside "
                f"days {start}-{end - 1}; detection lag is not measurable there"
            )
        # And no bad merchant transitions outside its own window, which would put it in the
        # already-bad regime the frozen split is not trying to measure.
        assert (((onset >= start) & (onset < end)) | ~(split == name))[split == name].all()


# ------------------------------------------------------------------------------------------
# T-0022a — the population-wide shock is emission-only and additive
# ------------------------------------------------------------------------------------------

SHOCK_DAY = 240
"""Shock lands inside the test window (days 210-269), where the black-swan report scores."""

SHOCK_MAGNITUDE = 6.0

SHOCK_CONFIG = GeneratorConfig(
    n_merchants=CONFIG.n_merchants,
    horizon_days=CONFIG.horizon_days,
    fraud_rate=CONFIG.fraud_rate,
    shock_days=(SHOCK_DAY,),
    shock_magnitude=SHOCK_MAGNITUDE,
)


@pytest.fixture(scope="module")
def shocked() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same population as `data`, plus one population-wide shock day."""
    return generate(SHOCK_CONFIG, seed_everything(42))


def test_shock_never_touches_the_state_path() -> None:
    """The invariant the whole stress test rests on, asserted directly on the profile.

    Ground truth must say nothing happened on a shock day. If it ever did, a model flagging a
    shocked merchant would be scored as a correct catch and `results/blackswan.md` would
    measure the opposite of what it claims to measure.
    """
    rng = seed_everything(7)
    profile = _baseline_profile(SEGMENTS[0], CONFIG.horizon_days, rng)
    _inject_bust_out(profile, SEGMENTS[0], CONFIG.horizon_days, 100, rng)

    states_before = profile.state.copy()
    volume_before = profile.volume_mult.copy()
    amount_before = profile.amount_mult.copy()

    _apply_shock(profile, CONFIG.horizon_days, (SHOCK_DAY,), SHOCK_MAGNITUDE)

    assert (profile.state == states_before).all(), "the shock leaked into ground truth"
    assert profile.volume_mult[SHOCK_DAY] == volume_before[SHOCK_DAY] * SHOCK_MAGNITUDE
    assert profile.amount_mult[SHOCK_DAY] == amount_before[SHOCK_DAY] * SHOCK_MAGNITUDE

    off = np.ones(CONFIG.horizon_days, dtype=bool)
    off[SHOCK_DAY] = False
    assert (profile.volume_mult[off] == volume_before[off]).all()
    assert (profile.amount_mult[off] == amount_before[off]).all()
    # A bust-out that has already vanished stays vanished — a dead merchant does not
    # benefit from a demand surge.
    assert profile.volume_mult[-1] == 0.0


def test_shock_day_outside_the_horizon_is_rejected() -> None:
    rng = seed_everything(7)
    profile = _baseline_profile(SEGMENTS[0], CONFIG.horizon_days, rng)
    with pytest.raises(ValueError, match="outside the horizon"):
        _apply_shock(profile, CONFIG.horizon_days, (CONFIG.horizon_days,), SHOCK_MAGNITUDE)


def test_unshocked_path_is_untouched(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The seam is inert on the default path — this is what protects `data/synthetic/`.

    A magnitude of 1.0 is exactly the identity in floating point, so if the shock code path
    were doing anything at all beyond multiplying two arrays, these hashes would diverge.
    Every committed number in this repo is measured on the frame this test pins.
    """
    transactions, state_paths = data
    inert = GeneratorConfig(
        n_merchants=CONFIG.n_merchants,
        horizon_days=CONFIG.horizon_days,
        fraud_rate=CONFIG.fraud_rate,
        shock_days=(SHOCK_DAY,),
        shock_magnitude=1.0,
    )
    same_txn, same_paths = generate(inert, seed_everything(42))
    assert _frame_hash(same_txn) == _frame_hash(transactions)
    assert _frame_hash(same_paths) == _frame_hash(state_paths)


def test_shock_determinism(shocked: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    transactions, state_paths = shocked
    same_txn, same_paths = generate(SHOCK_CONFIG, seed_everything(42))
    assert _frame_hash(same_txn) == _frame_hash(transactions)
    assert _frame_hash(same_paths) == _frame_hash(state_paths)


def test_shock_is_visible_in_transactions_only(
    data: tuple[pd.DataFrame, pd.DataFrame], shocked: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """The shock moves the observable stream hard, and moves the labels not at all.

    Note that the shocked and unshocked runs are NOT comparable row by row: changing the
    Poisson rate on one day changes how many variates each merchant consumes, so every later
    merchant's draws shift. That is why ground truth is checked structurally (healthy
    merchants stay healthy, with no segment boundary at the shock day) rather than by
    diffing the two `state_paths` frames.
    """
    _, unshocked_paths = data
    transactions, state_paths = shocked

    day = (transactions["timestamp"] - pd.Timestamp(GENERATOR_START_DATE)).dt.days
    per_day = transactions.groupby(day).size()
    neighbours = per_day.loc[[SHOCK_DAY - 2, SHOCK_DAY - 1, SHOCK_DAY + 1, SHOCK_DAY + 2]].mean()
    assert per_day.loc[SHOCK_DAY] / neighbours >= 3.0, "the shock is not visible in the stream"

    healthy = state_paths[state_paths["typology"] == NO_TYPOLOGY]
    assert (healthy["state"] == "HEALTHY").all()
    assert healthy.groupby("merchant_id").size().max() == 1, "a shock created a state boundary"
    # No merchant of any typology gained a transition on the shock day itself.
    assert SHOCK_DAY not in set(state_paths.loc[state_paths["segment_index"] > 0, "start_day"])
    # And the shocked population carries the same typology mix as the unshocked one.
    assert (
        state_paths.groupby("merchant_id")["typology"].first().value_counts().to_dict()
        == unshocked_paths.groupby("merchant_id")["typology"].first().value_counts().to_dict()
    )
