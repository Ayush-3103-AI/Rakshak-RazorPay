"""T-110: the scenario file loads into typed dataclasses, and a bad one says which field.

The Done-when clause is three things: it loads; persona shares validate to 1.0 +/- 1e-9;
an invalid config raises with a message naming the field. The third is the one worth
testing hardest — a loader that raises `KeyError: 'share'` against a file with sixty
`share` keys has told you nothing at 2am.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from rakshak.generator.config import (
    SHARE_TOLERANCE,
    ConfigError,
    ScenarioConfig,
    load_scenario,
)
from rakshak.schemas import Instrument, PersonaId, TypologyId

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenario_v2.yaml"


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
    """Charter §10, settled 2026-08-31. A silent drift in any of these invalidates
    every downstream number, so they are asserted rather than trusted."""
    assert scenario.population.n_merchants == 10_000
    assert scenario.population.n_days == 180
    assert scenario.population.prevalence == pytest.approx(0.0147)
    assert scenario.capacity.analyst_reviews_per_day == 50
    assert scenario.capacity.per_n_merchants == 10_000
    assert scenario.analyst_capacity == 50
    assert scenario.arrivals.target_fano == pytest.approx(12.25)


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
