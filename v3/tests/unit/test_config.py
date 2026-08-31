"""T-110: the scenario file loads into typed dataclasses, and a bad one says which field.

The Done-when clause is three things: it loads; persona shares validate to 1.0 +/- 1e-9;
an invalid config raises with a message naming the field. The third is the one worth
testing hardest — a loader that raises `KeyError: 'share'` against a file with sixty
`share` keys has told you nothing at 2am.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from rakshak.cli import _merchant_fold_t0101
from rakshak.eval.splits import SplitBoundaries, split_of_day
from rakshak.generator.config import (
    SHARE_TOLERANCE,
    ConfigError,
    ScenarioConfig,
    load_scenario,
)
from rakshak.generator.labels import NO_TIME, emit_labels
from rakshak.generator.typologies import assign_typologies
from rakshak.schemas import Instrument, PersonaId, TypologyId

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenario_v2.yaml"
_NS_PER_DAY = 86_400_000_000_000

#: T-0101 (GitHub #34): the five seeds EVAL-LOCK-CYCLE2.json declares.
CYCLE2_SEEDS = (42, 43, 44, 45, 46)


@pytest.fixture(scope="module")
def raw() -> dict[str, Any]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def scenario() -> ScenarioConfig:
    return load_scenario(CONFIG_PATH)


def write(tmp_path: Path, data: dict[str, Any]) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# It loads, and it loads the numbers the charter confirmed
# ─────────────────────────────────────────────────────────────────────────────


def test_ships_config_loads(scenario: ScenarioConfig) -> None:
    assert isinstance(scenario, ScenarioConfig)
    assert set(scenario.personas) == set(PersonaId)
    assert set(scenario.typologies) == set(TypologyId)


def test_charter_section_10_parameters(scenario: ScenarioConfig) -> None:
    """Charter §10, settled 2026-08-31, population corrected by T-0101 (GitHub #34,
    docs/RE-FREEZE-2026-08-31.md Amendment 4). A silent drift in any of these invalidates
    every downstream number, so they are asserted rather than trusted."""
    assert scenario.population.n_merchants == 20_000
    assert scenario.population.n_days == 365
    assert scenario.population.onset_window_min_day == 30
    assert scenario.population.onset_window_max_day == 240
    assert scenario.population.prevalence == pytest.approx(0.0147)
    assert scenario.capacity.analyst_reviews_per_day == 50
    assert scenario.capacity.per_n_merchants == 10_000
    assert scenario.analyst_capacity == 100
    assert scenario.arrivals.target_fano == pytest.approx(12.25)


def test_the_horizon_leaves_room_for_the_label_pipeline(scenario: ScenarioConfig) -> None:
    """`n_days` and the onset window are not pinned to constants here for their own sake.

    Cycle 1 pinned 180 and that is the number that was wrong: a mean label lag of 103.5
    days against a train split ending on day 119 left 8 trainable positive merchants
    (LIMITATIONS.md §8.1). What is asserted instead is the property that violation had
    and that is checkable with arithmetic before any model runs: the train boundary must
    sit at least a mean label lag past the EARLIEST onset any typology can draw.
    """
    labels = scenario.labels
    lag = labels.fraud_to_dispute_mean_days + sum(labels.dispute_delay_days) / 2.0
    train_end = scenario.splits.train_end_day
    earliest_onset = min(t.onset_day_min for t in scenario.typologies.values())
    assert train_end - lag >= earliest_onset, (
        f"train ends on day {train_end}; the mean label lag is {lag:.1f} days, so a "
        f"merchant is trainable only if it turned by day {train_end - lag:.1f}. The "
        f"earliest onset any typology can draw is day {earliest_onset}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# T-0101 (GitHub #34) — the corrected-geometry guard
# ─────────────────────────────────────────────────────────────────────────────


def _boundaries_of(scenario: ScenarioConfig) -> SplitBoundaries:
    s = scenario.splits
    return SplitBoundaries(
        origin=datetime.fromisoformat(scenario.population.start_date).date(),
        train=(0, s.train_end_day),
        val=(s.train_end_day + 1, s.val_end_day),
        test=(s.val_end_day + 1, s.test_end_day),
    )


def _test_fold_mask(n_merchants: int, shares: tuple[float, float, float]) -> np.ndarray:
    """Which merchant indices land in the TEST fold, at T-0101's independent ratio.

    Merchant ids are ``"M"`` plus a zero-padded index (``engine._merchant_id_series``) —
    no randomness in them at all, so fold membership is identical for every seed. Uses
    the real ``cli._merchant_fold_t0101`` rather than a re-derivation of its algorithm.
    """
    ids = [f"M{i:06d}" for i in range(n_merchants)]
    return np.array([_merchant_fold_t0101(m, shares) == "test" for m in ids])


def _labelled_positives_in_test_fold(
    scenario: ScenarioConfig, test_mask: np.ndarray, seed: int
) -> int:
    """How many TEST-fold merchants have a resolved (non-censored) label==1, at ``seed``.

    Deliberately does NOT run the full transaction generator — that cost is already
    owned, once, by ``tests/perf/test_gen_budget.py`` under NFR-10. Which merchants turn
    fraudulent, when, and whether their label resolves inside the horizon is decided
    ENTIRELY by ``assign_typologies`` and ``emit_labels`` — real production code — and
    neither one reads a single transaction.

    Does not reproduce ``engine.generate``'s RNG stream bit-for-bit (the persona/MCC/GMV
    draws that precede ``assign_typologies`` in the real pipeline are skipped), so this
    is not a claim of matching the shipped dataset byte for byte — it is the narrower,
    checkable claim the acceptance criterion asks for: at each of the five locked seeds,
    this configuration puts at least 50 labelled positives in the test fold.
    """
    rng = np.random.default_rng(seed)
    n = scenario.population.n_merchants
    assignment = assign_typologies(rng, n, scenario.population.prevalence, scenario.typologies)
    start = datetime.fromisoformat(scenario.population.start_date).replace(tzinfo=UTC)
    start_ns = int(start.timestamp()) * 1_000_000_000
    end_ns = start_ns + scenario.population.n_days * _NS_PER_DAY
    onset_ns = np.where(
        assignment.is_fraud,
        start_ns + assignment.onset_day.astype(np.int64) * _NS_PER_DAY,
        NO_TIME,
    )
    draw = emit_labels(
        rng, scenario.labels, drift_onset_ns=onset_ns, sim_start_ns=start_ns, sim_end_ns=end_ns
    )
    return int(np.sum((draw.label == 1.0) & test_mask))


def test_test_split_has_enough_labelled_positives_per_seed(scenario: ScenarioConfig) -> None:
    """T-0101 (GitHub #34)'s guard: the corrected geometry — 20,000 merchants, 365 days,
    drift onsets confined to [30, 240], an independent 60/15/25 merchant fold — must put
    at least 50 labelled positives in the TEST split for EVERY one of the five locked
    seeds (EVAL-LOCK-CYCLE2.json), not in expectation across seeds.

    This is what makes it a guard rather than a decoration: pointed at the previous
    (drafted) 10,000 x 360-day geometry with a day-proportional (not independent)
    merchant fold, the same counting logic returns 16-29 labelled positives per seed,
    RED against the >=50 floor. Verified by hand while writing this test; not left in
    the suite as a second, permanent test.
    """
    shares = (
        scenario.splits.merchant_fold_train,
        scenario.splits.merchant_fold_val,
        scenario.splits.merchant_fold_test,
    )
    test_mask = _test_fold_mask(scenario.population.n_merchants, shares)
    assert int(test_mask.sum()) > 0, "no merchant hashed into the test fold at all"
    for seed in CYCLE2_SEEDS:
        n = _labelled_positives_in_test_fold(scenario, test_mask, seed)
        assert n >= 50, (
            f"seed {seed}: only {n} labelled positives landed in the test split "
            f"({int(test_mask.sum())} merchants in the test fold); T-0101 requires >= 50 "
            "per seed"
        )


def test_every_split_carries_at_least_one_discrete_confounder(scenario: ScenarioConfig) -> None:
    """Amendment 2's property, re-verified for T-0101's window rather than assumed to
    still hold after the horizon moved again (180 -> 365)."""
    boundaries = _boundaries_of(scenario)
    c = scenario.confounders
    discrete_days = [
        *c.P1_festival.days,
        *c.P2_outage.days,
        c.P3_fee_change.day,
        c.P4_new_method.day,
        c.P5_regulatory.day,
    ]
    covered = {split_of_day(d, boundaries) for d in discrete_days}
    assert covered >= {"train", "val", "test"}, (
        f"discrete confounder days {discrete_days} land in splits {covered}, missing "
        f"{({'train', 'val', 'test'} - covered)}"
    )


def test_analyst_capacity_scales_with_population(scenario: ScenarioConfig) -> None:
    """K is quoted per 10k merchants; a 2k-merchant smoke run must not get 50 reviews."""
    import dataclasses

    small = dataclasses.replace(
        scenario, population=dataclasses.replace(scenario.population, n_merchants=2_000)
    )
    assert small.analyst_capacity == 10


# ─────────────────────────────────────────────────────────────────────────────
# Shares
# ─────────────────────────────────────────────────────────────────────────────


def test_persona_shares_sum_to_one(scenario: ScenarioConfig) -> None:
    total = sum(p.share for p in scenario.personas.values())
    assert abs(total - 1.0) <= SHARE_TOLERANCE


def test_typology_mix_sums_to_one(scenario: ScenarioConfig) -> None:
    total = sum(t.mix for t in scenario.typologies.values())
    assert abs(total - 1.0) <= SHARE_TOLERANCE


def test_instrument_mixes_sum_to_one_and_name_real_instruments(
    scenario: ScenarioConfig,
) -> None:
    valid = {i.value for i in Instrument}
    for pid, persona in scenario.personas.items():
        assert abs(sum(persona.instrument_mix.values()) - 1.0) <= SHARE_TOLERANCE, pid
        assert set(persona.instrument_mix) <= valid, pid


def test_l3_share_not_shrunk_below_the_floor(scenario: ScenarioConfig) -> None:
    """08-generator-v2-spec.md §2: 'hard negative — do not shrink below 0.05'. If L3 is
    rare, v_gmv_accel never gets tested and the hardest negative is decorative."""
    assert scenario.personas[PersonaId.L3].share >= 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Invalid configs name the field
# ─────────────────────────────────────────────────────────────────────────────


def test_bad_persona_share_names_personas_and_share(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    bad = copy.deepcopy(raw)
    bad["personas"]["L1"]["share"] = 0.40
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    assert "personas" in str(exc.value)
    assert "share" in str(exc.value)


def test_missing_key_names_the_dotted_path(tmp_path: Path, raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    del bad["personas"]["L4"]["amount_sigma"]
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    message = str(exc.value)
    assert "personas.L4" in message
    assert "amount_sigma" in message


def test_unknown_key_is_rejected_by_name(tmp_path: Path, raw: dict[str, Any]) -> None:
    """A renamed parameter still set under its old name is the failure this catches."""
    bad = copy.deepcopy(raw)
    bad["arrivals"]["target_phano"] = 12.25
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    assert "arrivals" in str(exc.value)
    assert "target_phano" in str(exc.value)


def test_wrong_type_names_the_field(tmp_path: Path, raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["population"]["n_merchants"] = "ten thousand"
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    assert "population.n_merchants" in str(exc.value)


def test_missing_typology_is_rejected(tmp_path: Path, raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    del bad["typologies"]["R7"]
    bad["typologies"]["R1"]["mix"] += 0.07
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    assert "R7" in str(exc.value)


def test_unknown_persona_key_is_rejected(tmp_path: Path, raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["personas"]["L9"] = copy.deepcopy(bad["personas"]["L1"])
    bad["personas"]["L9"]["share"] = 0.0
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    assert "L9" in str(exc.value)


def test_bad_shape_name_lists_the_valid_shapes(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    bad = copy.deepcopy(raw)
    bad["personas"]["L2"]["shape"] = "seasonl"
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    assert "personas.L2.shape" in str(exc.value)
    assert "seasonal" in str(exc.value)


def test_hour_weights_must_be_24_long(tmp_path: Path, raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["personas"]["L1"]["hour_weights"] = [1.0] * 23
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    assert "hour_weights" in str(exc.value)


def test_explosive_hawkes_excitation_is_rejected(
    tmp_path: Path, raw: dict[str, Any]
) -> None:
    bad = copy.deepcopy(raw)
    bad["arrivals"]["hawkes_excitation"] = 1.4
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    assert "hawkes_excitation" in str(exc.value)


def test_under_dispersed_fano_is_rejected(tmp_path: Path, raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["arrivals"]["target_fano"] = 0.5
    with pytest.raises(ConfigError) as exc:
        load_scenario(write(tmp_path, bad))
    assert "target_fano" in str(exc.value)


def test_prevalence_zero_is_legal(tmp_path: Path, raw: dict[str, Any]) -> None:
    """Gate G5 runs at prevalence=0 with confounders on. It must load, not raise."""
    ok = copy.deepcopy(raw)
    ok["population"]["prevalence"] = 0.0
    assert load_scenario(write(tmp_path, ok)).population.prevalence == 0.0


def test_missing_file_and_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_scenario(tmp_path / "nope.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_scenario(empty)
