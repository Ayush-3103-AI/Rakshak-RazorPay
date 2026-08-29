"""T-0006 — the three baselines: static rules, LightGBM, random.

Two kinds of test here, deliberately separated:

* **Fixture tests** build a tiny hand-written `Split` whose right answer is known
  by construction, so a rule firing on the wrong day is a hard failure rather
  than a plausible-looking number.
* **Contract tests** run the real baselines against the real validate split and
  check the harness contract (index, range, determinism) plus the leakage guard:
  nothing in this ticket may open the test window.

No test here asserts that one baseline beats another. T-0006 is plumbing;
06-requirements.md §3 puts the comparison at T-0011.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rakshak.config import (
    GENERATOR_START_DATE,
    SEED,
    STATE_PATHS_PARQUET,
    TRANSACTIONS_PARQUET,
    WINDOW_DAYS,
)
from rakshak.eval.harness import MODEL_REGISTRY, _model_rng, _normalise, evaluate_model
from rakshak.eval.splits import Split, load_split
from rakshak.models import gbdt, rules

needs_data = pytest.mark.skipif(
    not (TRANSACTIONS_PARQUET.exists() and STATE_PATHS_PARQUET.exists()),
    reason="run `python -m rakshak.generator --seed 42` first",
)


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def _transactions(rows: list[tuple[str, int, bool, bool]]) -> pd.DataFrame:
    """Build a transaction frame from (merchant_id, day, is_refund, is_chargeback)."""
    origin = pd.Timestamp(GENERATOR_START_DATE)
    frame = pd.DataFrame(rows, columns=["merchant_id", "day", "is_refund", "is_chargeback"])
    frame["timestamp"] = origin + pd.to_timedelta(frame["day"], unit="D")
    frame["amount"] = 1000.0
    frame["payer_id"] = "P" + frame.index.astype(str)
    frame["method"] = "card"
    frame["mcc"] = "5411"
    return frame


def _split(
    transactions: pd.DataFrame, start_day: int, end_day: int, name: str = "fixture"
) -> Split:
    """Wrap a transaction frame in a Split with all-healthy ground truth."""
    merchant_ids = tuple(sorted(transactions["merchant_id"].unique()))
    index = pd.Index(merchant_ids, name="merchant_id")
    zeros = pd.Series(0.0, index=index)
    return Split(
        name=name,
        start_day=start_day,
        end_day=end_day,
        merchant_ids=merchant_ids,
        transactions=transactions.sort_values(["merchant_id", "timestamp"], kind="stable"),
        labels=pd.Series(0, index=index, name="label"),
        transition_day=pd.Series(np.nan, index=index, name="transition_day"),
        transition_timestamp=pd.Series(pd.NaT, index=index, name="transition_timestamp"),
        loss_inr=zeros.rename("loss_inr"),
        value_inr=zeros.rename("value_inr"),
    )


@pytest.fixture(scope="module")
def rule_split() -> Split:
    """Four merchants, one per rule plus a clean control. Window is days [100, 130).

    * `M_CLEAN` — 2 transactions every day, no refunds, no chargebacks. Fires nothing.
    * `M_SPIKE` — 1/day until day 109, then 20/day. The 7-day count crosses 3x the
      trailing-90-day expectation on day 110.
    * `M_REFUND` — 2/day, half of them refunds from day 71 (i.e. the whole 30-day
      ratio window at the decision window's first day), so it fires on day 100.
    * `M_CHARGEBACK` — 20/day with 1 chargeback/day (5%) from day 71, likewise
      firing on day 100.
    """
    rows: list[tuple[str, int, bool, bool]] = []
    for day in range(130):
        rows += [("M_CLEAN", day, False, False)] * 2
        rows += [("M_SPIKE", day, False, False)] * (20 if day >= 110 else 1)
        refunding = day >= 71
        rows += [("M_REFUND", day, False, False), ("M_REFUND", day, refunding, False)]
        charging_back = day >= 71
        rows += [("M_CHARGEBACK", day, False, False)] * 19
        rows += [("M_CHARGEBACK", day, False, charging_back)]
    return _split(_transactions(rows), start_day=100, end_day=130)


# ---------------------------------------------------------------------------
# rules.py — the floor (06-requirements.md §3)
# ---------------------------------------------------------------------------


def test_rule_thresholds_match_the_frozen_spec() -> None:
    """The floor is frozen text, not a tunable. Softening it is out of bounds."""
    assert rules.VELOCITY_MULTIPLE == 3.0
    assert rules.VELOCITY_WINDOW_DAYS == 7
    assert rules.VELOCITY_BASELINE_DAYS == 90
    assert rules.REFUND_RATIO_THRESHOLD == 0.15
    assert rules.CHARGEBACK_RATIO_THRESHOLD == 0.01


def test_clean_merchant_never_fires(rule_split: Split) -> None:
    out = rules.score_rules(rule_split, np.random.default_rng(0))
    assert np.isnan(out.loc["M_CLEAN", "flag_day"])
    severity, _ = rules.rule_severities(rule_split)
    clean = severity[rule_split.merchant_ids.index("M_CLEAN")]
    assert clean.max() < 1.0


@pytest.mark.parametrize(
    ("merchant", "expected_flag_day", "rule"),
    [
        ("M_SPIKE", 110.0, "velocity"),
        ("M_REFUND", 100.0, "refund_ratio"),
        ("M_CHARGEBACK", 100.0, "chargeback_ratio"),
    ],
)
def test_each_rule_fires_on_the_expected_day(
    rule_split: Split, merchant: str, expected_flag_day: float, rule: str
) -> None:
    out = rules.score_rules(rule_split, np.random.default_rng(0))
    assert out.loc[merchant, "flag_day"] == expected_flag_day

    severity, days = rules.rule_severities(rule_split)
    row = severity[rule_split.merchant_ids.index(merchant)]
    fired_on = row[days == expected_flag_day][0]
    assert fired_on[rules.RULE_NAMES.index(rule)] >= 1.0, "the wrong rule fired"


def test_scores_are_bounded_and_rank_flagged_above_clean(rule_split: Split) -> None:
    out = rules.score_rules(rule_split, np.random.default_rng(0))
    assert out["score"].between(0.0, 1.0).all()
    for merchant in ("M_SPIKE", "M_REFUND", "M_CHARGEBACK"):
        assert out.loc[merchant, "score"] > out.loc["M_CLEAN", "score"]


def test_rules_read_no_labels_and_no_rng(rule_split: Split) -> None:
    """The floor is unlearned: same output whatever the labels or the seed say."""
    first = rules.score_rules(rule_split, np.random.default_rng(0))
    flipped = _split(rule_split.transactions, 100, 130)
    object.__setattr__(
        flipped, "labels", pd.Series(1, index=first.index, name="label")
    )
    second = rules.score_rules(flipped, np.random.default_rng(999))
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# gbdt.py — LightGBM on windowed aggregates, no HMM
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gbdt_fixture() -> tuple[Split, pd.DataFrame]:
    """Eight merchants over 140 days; two of them enter FRAUD on day 91."""
    rows: list[tuple[str, int, bool, bool]] = []
    merchants = [f"M{i:02d}" for i in range(8)]
    bad = set(merchants[:2])
    for merchant in merchants:
        for day in range(140):
            n = 8 if (merchant in bad and day >= 91) else 2
            rows += [(merchant, day, False, merchant in bad and day >= 91)] * n
    state_paths = pd.DataFrame(
        [
            {"merchant_id": m, "state": "HEALTHY", "start_day": 0, "end_day": 91, "typology": "T"}
            for m in merchants
        ]
        + [
            {
                "merchant_id": m,
                "state": "FRAUD" if m in bad else "HEALTHY",
                "start_day": 91,
                "end_day": 140,
                "typology": "T",
            }
            for m in merchants
        ]
    )
    return _split(_transactions(rows), start_day=112, end_day=140), state_paths


def test_window_matrix_aligns_windows_to_absolute_days(
    gbdt_fixture: tuple[Split, pd.DataFrame],
) -> None:
    split, state_paths = gbdt_fixture
    matrix = gbdt.build_window_matrix(split, state_paths=state_paths)

    assert matrix.X.shape[0] == matrix.y.shape[0] == matrix.window_start_day.shape[0]
    assert matrix.X.shape[0] == len(matrix.merchant_ids) * (140 // WINDOW_DAYS)
    # Row block r covers merchant r's windows in ascending day order from day 0.
    first_block = matrix.window_start_day[matrix.merchant_row == 0]
    assert first_block[0] == 0
    assert np.array_equal(np.diff(first_block), np.full(first_block.size - 1, WINDOW_DAYS))


def test_decision_mask_excludes_windows_outside_the_split(
    gbdt_fixture: tuple[Split, pd.DataFrame],
) -> None:
    split, state_paths = gbdt_fixture
    matrix = gbdt.build_window_matrix(split, state_paths=state_paths)
    mask = gbdt.decision_mask(matrix, split)
    assert mask.any()
    assert matrix.window_start_day[mask].min() >= split.start_day
    assert (matrix.window_start_day[mask] + WINDOW_DAYS).max() <= split.end_day


def test_window_labels_track_the_generator_ground_truth(
    gbdt_fixture: tuple[Split, pd.DataFrame],
) -> None:
    split, state_paths = gbdt_fixture
    matrix = gbdt.build_window_matrix(split, state_paths=state_paths)
    bad_rows = matrix.y == 1
    assert bad_rows.sum() > 0
    # Only the two FRAUD merchants, and only from day 91 on.
    flagged_merchants = {matrix.merchant_ids[r] for r in matrix.merchant_row[bad_rows]}
    assert flagged_merchants == {"M00", "M01"}
    assert matrix.window_start_day[bad_rows].min() >= 91 - WINDOW_DAYS


@needs_data
def test_gbdt_training_never_opens_the_test_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The leakage guard for this ticket. `load_split("test")` would raise anyway;
    this pins the intent so a future edit cannot pass an unlock ticket instead."""
    opened: list[str] = []
    real = gbdt.load_split

    def spy(name: str, **kwargs: object) -> Split:
        opened.append(name)
        assert "unlock_test" not in kwargs, "a baseline may not unlock the test window"
        return real(name, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(gbdt, "load_split", spy)
    gbdt.fit(seed=SEED)
    assert set(opened) == {"train", "validate"}


# ---------------------------------------------------------------------------
# random — the absolute floor
# ---------------------------------------------------------------------------


def test_random_is_seeded_and_reproducible(rule_split: Split) -> None:
    score_random = MODEL_REGISTRY["random"]
    first = score_random(rule_split, _model_rng(SEED, "random"))
    second = score_random(rule_split, _model_rng(SEED, "random"))
    other = score_random(rule_split, _model_rng(SEED + 1, "random"))
    pd.testing.assert_series_equal(first, second)
    assert not first.equals(other)


# ---------------------------------------------------------------------------
# Harness contract — all three produce a row (T-0006 "Done when")
# ---------------------------------------------------------------------------


def test_all_three_baselines_are_registered() -> None:
    for name in ("random", "rules", "gbdt"):
        assert name in MODEL_REGISTRY


@needs_data
@pytest.mark.parametrize("name", ["random", "rules", "gbdt"])
def test_baseline_produces_a_valid_harness_row(name: str) -> None:
    split = load_split("validate")
    frame = _normalise(MODEL_REGISTRY[name](split, _model_rng(SEED, name)), split)
    assert list(frame.index) == list(split.merchant_ids)
    assert frame["score"].notna().all()
    assert frame["score"].between(0.0, 1.0).all()

    row = evaluate_model(name, split, seed=SEED, k=5)
    assert row["model"] == name
    assert 0.0 <= float(row["pr_auc"]) <= 1.0
    assert 0.0 <= float(row["precision_at_k"]) <= 1.0
    assert row["n_reviewed"] == 5


@needs_data
@pytest.mark.parametrize("name", ["random", "rules", "gbdt"])
def test_baseline_is_deterministic_at_a_fixed_seed(name: str) -> None:
    split = load_split("validate")
    first = _normalise(MODEL_REGISTRY[name](split, _model_rng(SEED, name)), split)
    second = _normalise(MODEL_REGISTRY[name](split, _model_rng(SEED, name)), split)
    pd.testing.assert_frame_equal(first, second)
