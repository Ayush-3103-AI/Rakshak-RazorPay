"""T-0015 — calibration profiler, manifest builder and profile-vs-generator diff.

Every expected value below is derived independently of the implementation: by hand from
the 50-row fixture in `tests/fixtures/online_retail_ii_sample.csv`, or in closed form.
No test recomputes a figure the way `profile.py` computes it.

The fixture is constructed so that every marginal has an exact answer:

* 50 rows, 45 non-refund + 5 refund   -> refund_rate = 5/50 = 0.10
* non-refund amounts: 15 x 1.00, 15 x 10.00, 15 x 100.00
  -> ln-amounts are {0, ln10, 2 ln10} in equal thirds, so the population sd is
     ln(10) * sqrt(2/3); the median is 10.00 and p90/p50 = 10.0
* hours: 10 x 12h, 30 x 14h, 10 x 16h -> mean 14, population var 80/50 = 1.6
* rows per weekday Mon..Sun: 5,5,5,5,10,10,10 -> mean 50/7, factors 0.7 and 1.4;
  population variance 42.857.../7, so the Fano factor is exactly 6/7
* 10 distinct payers over 50 rows -> new_payer_frac = 0.20; the top decile is
  ceil(0.1 * 10) = 1 payer holding 14 rows -> 14/50 = 0.28

There is no live-network test here. The real download runs once, manually, and its
evidence is the committed manifest under `data/external/`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from rakshak.data.download import file_manifest
from rakshak.data.profile import CANNOT_INFORM, build_profile, diff_profile

FIXTURE = Path(__file__).parent / "fixtures" / "online_retail_ii_sample.csv"

# Computed with `sha256sum tests/fixtures/online_retail_ii_sample.csv`, not with hashlib.
FIXTURE_SHA256 = "73f0af44454385a978643f52db367f2ba143f406296acd44bcee472fa543975f"


def test_file_manifest_records_sha256_and_row_count() -> None:
    """The manifest is the provenance artifact: hash and row count must be exact."""
    manifest = file_manifest(
        name="fixture",
        path=FIXTURE,
        source_url="https://example.invalid/fixture.csv",
        licence="test fixture, hand-authored",
        licence_url="https://example.invalid/licence",
        row_count=50,
    )
    assert manifest["sha256"] == FIXTURE_SHA256
    assert manifest["row_count"] == 50
    assert manifest["size_bytes"] == 1882
    assert manifest["licence"] == "test fixture, hand-authored"
    assert manifest["retrieved_utc"].endswith("Z")


def test_fixture_loads_with_the_expected_shape() -> None:
    df = pd.read_csv(FIXTURE, parse_dates=["timestamp"])
    assert len(df) == 50
    assert list(df.columns) == ["timestamp", "amount", "payer_id", "is_refund"]
    assert math.isclose(float(df["is_refund"].sum()), 5.0)


# ---------------------------------------------------------------------------
# build_profile — one merchant stream in, marginals out
# ---------------------------------------------------------------------------


def _profile() -> dict:
    df = pd.read_csv(FIXTURE, parse_dates=["timestamp"])
    return build_profile(df, dataset="fixture")


def _value(key: str) -> float:
    return _profile()["marginals"][key]["value"]


def test_amount_marginals_match_the_closed_form() -> None:
    """ln-amounts are {0, ln10, 2 ln10} in equal thirds over the 45 non-refund rows."""
    assert _value("amount_log_sd") == pytest.approx(math.log(10.0) * math.sqrt(2.0 / 3.0))
    assert _value("amount_p50") == pytest.approx(10.0)
    assert _value("amount_p90_over_p50") == pytest.approx(10.0)


def test_time_of_day_marginals_match_the_closed_form() -> None:
    """Hours are 10 x 12h, 30 x 14h, 10 x 16h: mean 14, population variance 80/50."""
    assert _value("hour_of_day_mean") == pytest.approx(14.0)
    assert _value("hour_of_day_sd_hours") == pytest.approx(math.sqrt(1.6))


def test_weekday_factor_is_counts_normalised_to_mean_one() -> None:
    """Rows per weekday Mon..Sun are 5,5,5,5,10,10,10 against a mean of 50/7."""
    factors = _profile()["marginals"]["weekday_volume_factor"]["value"]
    assert factors == pytest.approx([0.7, 0.7, 0.7, 0.7, 1.4, 1.4, 1.4])


def test_rate_and_payer_marginals_match_the_hand_count() -> None:
    assert _value("refund_rate") == pytest.approx(5.0 / 50.0)
    assert _value("new_payer_frac") == pytest.approx(10.0 / 50.0)
    # Top decile of 10 payers is 1 payer, and P01 holds 14 of the 50 rows.
    assert _value("top_decile_payer_share") == pytest.approx(14.0 / 50.0)


def test_volume_marginals_match_the_hand_count() -> None:
    """Daily counts 5,5,5,5,10,10,10: mean 50/7, population variance 300/49, Fano 6/7."""
    assert _value("txns_per_active_day_mean") == pytest.approx(50.0 / 7.0)
    assert _value("daily_count_fano_factor") == pytest.approx(6.0 / 7.0)


def test_every_marginal_carries_dataset_and_column_provenance() -> None:
    marginals = _profile()["marginals"]
    assert marginals, "profile is empty"
    for name, figure in marginals.items():
        assert figure["dataset"] == "fixture", name
        assert figure["column"], name
        assert figure["unit"], name
        assert figure["n"] > 0, name


# ---------------------------------------------------------------------------
# Determinism and the gap diff
# ---------------------------------------------------------------------------


def test_profile_json_is_byte_identical_across_runs() -> None:
    """`Done when`: re-running the profiler on the same input reproduces the JSON exactly."""
    first = json.dumps(_profile(), indent=2, sort_keys=True).encode("utf-8")
    second = json.dumps(_profile(), indent=2, sort_keys=True).encode("utf-8")
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_diff_places_every_marginal_beside_a_named_generator_value() -> None:
    """`Done when`: every marginal in the profile appears in the diff with the generator's value."""
    empirical = _profile()
    generator = build_profile(
        pd.read_csv(FIXTURE, parse_dates=["timestamp"]), dataset="rakshak_synthetic"
    )
    rows = diff_profile(empirical, generator)

    assert [r["marginal"] for r in rows] == list(empirical["marginals"])
    for row in rows:
        assert row["generator"] is not None, row["marginal"]
        assert row["generator_constant"] != "unmapped", row["marginal"]
        assert row["empirical_source"].startswith("fixture:"), row["marginal"]
    # Diffing a profile against itself must show no divergence for the comparable scalars.
    for row in rows:
        if row["ratio"] is not None:
            assert row["ratio"] == pytest.approx(1.0)
        if row["abs_diff"] is not None:
            assert row["abs_diff"] == pytest.approx(0.0)


def test_level_only_marginals_are_flagged_not_comparable() -> None:
    """Absolute ticket size cannot cross a currency and a merchant category. Say so, in data."""
    rows = {r["marginal"]: r for r in diff_profile(_profile(), _profile())}
    assert rows["amount_p50"]["comparable"] is False
    assert rows["amount_p50"]["ratio"] is None
    assert rows["amount_log_sd"]["comparable"] is True


def test_cannot_inform_names_labels_and_sequence_structure() -> None:
    """`Done when`: the diff states what the profile cannot inform."""
    joined = " ".join(CANNOT_INFORM).lower()
    assert "latent risk state" in joined
    assert "merchant-level fraud label" in joined
    assert "chargeback" in joined
