"""T-101: every table and enum in 09-interfaces.md exists, and its invariants bite.

These tests are cheap and they look pedantic. They are here because `schemas.py` freezes
at the end of Block 1 and four parallel lanes then build against it without re-reading it.
A contract nobody re-reads needs a test that fails when someone edits it anyway.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, date, datetime

import numpy as np
import pytest

from rakshak import SCHEMA_VERSION
from rakshak.schemas import (
    CARD_INSTRUMENTS,
    GROUND_TRUTH_SCHEMA,
    LABEL_SCHEMA,
    PAYOUT_SCHEMA,
    PROFILE_SCHEMA,
    RADIOACTIVE_FIELDS,
    TRANSACTION_SCHEMA,
    Action,
    ConfounderId,
    Decision,
    EvalResult,
    FeatureVector,
    GroundTruth,
    Instrument,
    Label,
    LabelSource,
    MerchantProfile,
    Payout,
    PersonaId,
    Tier,
    Transaction,
    TxnStatus,
    TypologyId,
)

T0 = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def a_transaction(**overrides: object) -> Transaction:
    base: dict[str, object] = {
        "event_id": "E1",
        "merchant_id": "M1",
        "payer_id": "P1",
        "event_time": T0,
        "event_date": T0.date(),
        "amount_inr": 1500.0,
        "instrument": Instrument.UPI,
        "is_cnp": True,
        "is_international": False,
        "bin_hash": None,
        "device_hash": "d" * 16,
        "ip_hash": "a" * 16,
        "status": TxnStatus.CAPTURED,
        "decline_code": None,
        "mcc": "5411",
        "is_refund": False,
        "refund_of": None,
    }
    return Transaction(**(base | overrides))  # type: ignore[arg-type]


# ── §6 enums ──────────────────────────────────────────────────────────────────


def test_enum_cardinalities_match_the_spec() -> None:
    # If one of these changes, a generator lane and a metrics lane disagree about how many
    # typologies exist, and per-typology recall silently reports on a subset.
    assert len(PersonaId) == 8
    assert len(TypologyId) == 9
    assert len(ConfounderId) == 6
    assert len(Instrument) == 7
    assert [t.value for t in Tier] == [1, 2, 3]
    assert {a.value for a in Action} == {"pass", "review", "hold"}
    assert {s.value for s in LabelSource} == {"chargeback", "manual_review", "none"}


def test_enums_are_string_valued_so_parquet_round_trips() -> None:
    # Parquet has no enum type; every enum column is written as a string and read back
    # through the StrEnum constructor. A non-string value here breaks that round trip.
    for member in (*Instrument, *TxnStatus, *Action, *LabelSource, *PersonaId, *TypologyId):
        assert isinstance(member.value, str)


# ── §1 Transaction ────────────────────────────────────────────────────────────


def test_transaction_accepts_a_well_formed_row() -> None:
    txn = a_transaction()
    assert txn.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize(
    ("overrides", "expect"),
    [
        ({"amount_inr": 0.0}, "amount_inr"),
        ({"amount_inr": -10.0}, "amount_inr"),
        ({"event_time": datetime(2026, 3, 1, 12, 0)}, "tz-aware"),
        ({"event_date": date(2026, 3, 2)}, "event_date"),
        ({"decline_code": "insufficient_funds"}, "decline_code"),
        ({"status": TxnStatus.FAILED}, "decline_code"),
        ({"refund_of": "E0"}, "refund_of"),
        ({"is_refund": True}, "refund_of"),
        ({"instrument": Instrument.CARD_CREDIT}, "bin_hash"),
        ({"bin_hash": "b" * 16}, "bin_hash"),
    ],
)
def test_transaction_rejects_malformed_rows(overrides: dict[str, object], expect: str) -> None:
    with pytest.raises(ValueError, match=expect):
        a_transaction(**overrides)


def test_refund_amount_is_positive_and_the_sign_lives_in_is_refund() -> None:
    # 09-interfaces.md §1. A negative amount on a refund would make every sum-based
    # feature accidentally correct and every count-based one accidentally wrong.
    refund = a_transaction(event_id="E2", amount_inr=1500.0, is_refund=True, refund_of="E1")
    assert refund.amount_inr > 0
    assert refund.is_refund


def test_transaction_carries_no_label_shaped_field() -> None:
    # FR-006. The event stream is the only table the feature layer reads; a chargeback,
    # dispute, persona, typology or onset column here is leakage by construction.
    names = {f.name for f in fields(Transaction)}
    forbidden = {"chargeback", "is_fraud", "label", "dispute", *RADIOACTIVE_FIELDS}
    assert not (names & forbidden), f"Transaction carries label-shaped fields: {names & forbidden}"


def test_card_instruments_is_the_set_that_carries_a_bin() -> None:
    assert Instrument.UPI not in CARD_INSTRUMENTS
    assert Instrument.NETBANKING not in CARD_INSTRUMENTS
    assert Instrument.WALLET not in CARD_INSTRUMENTS
    assert CARD_INSTRUMENTS <= set(Instrument)


# ── §2 MerchantProfile ────────────────────────────────────────────────────────


def a_profile(**overrides: object) -> MerchantProfile:
    base: dict[str, object] = {
        "merchant_id": "M1",
        "onboarded_at": T0,
        "mcc": "5411",
        "mcc_group": "grocery",
        "declared_monthly_gmv": 500_000.0,
        "kyc_tier": 2,
        "vintage_months": 18,
        "city_tier": 1,
    }
    return MerchantProfile(**(base | overrides))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "expect"),
    [
        ({"declared_monthly_gmv": 0.0}, "denominator"),
        ({"kyc_tier": 0}, "kyc_tier"),
        ({"kyc_tier": 4}, "kyc_tier"),
        ({"city_tier": 9}, "city_tier"),
        ({"onboarded_at": datetime(2026, 3, 1)}, "tz-aware"),
    ],
)
def test_profile_rejects_malformed_rows(overrides: dict[str, object], expect: str) -> None:
    with pytest.raises(ValueError, match=expect):
        a_profile(**overrides)


# ── §3 Payout ─────────────────────────────────────────────────────────────────


def test_payout_rejects_settlement_before_request() -> None:
    with pytest.raises(ValueError, match="precedes requested_at"):
        Payout(
            payout_id="PO1",
            merchant_id="M1",
            requested_at=T0,
            settled_at=datetime(2026, 2, 1, tzinfo=UTC),
            amount_inr=1000.0,
            balance_before_inr=5000.0,
            is_accelerated=False,
        )


def test_payout_allows_a_pending_settlement() -> None:
    assert Payout(
        payout_id="PO1",
        merchant_id="M1",
        requested_at=T0,
        settled_at=None,
        amount_inr=1000.0,
        balance_before_inr=5000.0,
        is_accelerated=True,
    ).settled_at is None


# ── §4 Label ──────────────────────────────────────────────────────────────────


def test_label_available_at_must_be_strictly_after_the_event() -> None:
    # The invariant that makes v2 honest. Equality is rejected too: a label that is usable
    # the instant the dispute occurs is exactly the instant-label assumption v1 made.
    with pytest.raises(ValueError, match="strictly after"):
        Label(
            merchant_id="M1",
            label=1,
            label_event_at=T0,
            label_available_at=T0,
            label_source=LabelSource.CHARGEBACK,
            is_censored=False,
        )


def test_censored_merchants_carry_no_resolved_label() -> None:
    with pytest.raises(ValueError, match="censored"):
        Label(
            merchant_id="M1",
            label=0,
            label_event_at=None,
            label_available_at=None,
            label_source=LabelSource.NONE,
            is_censored=True,
        )


def test_a_censored_label_row_is_well_formed_with_all_nulls() -> None:
    row = Label(
        merchant_id="M1",
        label=None,
        label_event_at=None,
        label_available_at=None,
        label_source=LabelSource.NONE,
        is_censored=True,
    )
    assert row.label is None


# ── §5 GroundTruth ────────────────────────────────────────────────────────────


def test_typology_and_onset_travel_together() -> None:
    with pytest.raises(ValueError, match="typology iff"):
        GroundTruth(
            merchant_id="M1",
            persona_id=PersonaId.L1,
            risk_typology_id=TypologyId.R2,
            drift_onset_at=None,
            true_loss_amount_inr=50_000.0,
            is_unreported=False,
        )


def test_every_fraud_merchant_carries_a_positive_loss() -> None:
    # true_loss_amount_inr is the weight in the oracle knapsack. A zero-weight fraud row
    # makes the oracle ceiling too low, which makes every rung look better than it is.
    with pytest.raises(ValueError, match="positive true_loss_amount_inr"):
        GroundTruth(
            merchant_id="M1",
            persona_id=PersonaId.L1,
            risk_typology_id=TypologyId.R2,
            drift_onset_at=T0,
            true_loss_amount_inr=0.0,
            is_unreported=False,
        )


def test_a_clean_merchant_causes_no_loss() -> None:
    with pytest.raises(ValueError, match="no typology causes no loss"):
        GroundTruth(
            merchant_id="M1",
            persona_id=PersonaId.L1,
            risk_typology_id=None,
            drift_onset_at=None,
            true_loss_amount_inr=1.0,
            is_unreported=False,
        )


def test_radioactive_field_names_match_the_ground_truth_dataclass() -> None:
    # The AST scan in tests/gates/ greps for RADIOACTIVE_FIELDS. If GroundTruth grows a
    # field and this set does not, the quarantine develops a hole nobody notices.
    gt_fields = {f.name for f in fields(GroundTruth)} - {"merchant_id", "schema_version"}
    assert gt_fields <= RADIOACTIVE_FIELDS


# ── §9/§10/§11 model-facing shapes ────────────────────────────────────────────


def test_feature_vector_insists_on_float64() -> None:
    with pytest.raises(ValueError, match="float64"):
        FeatureVector(
            merchant_id="M1",
            as_of=T0.date(),
            values=np.zeros(3, dtype=np.float32),
            feature_schema_version=1,
            computed_by="offline",
            stage_reached=0,
        )


@pytest.mark.parametrize(
    ("action", "codes", "ok"),
    [
        (Action.PASS, [], True),
        (Action.PASS, ["a", "b", "c"], False),
        (Action.REVIEW, ["a", "b", "c"], True),
        (Action.REVIEW, ["a", "b"], False),
        (Action.HOLD, ["a", "b", "c"], True),
        (Action.HOLD, [], False),
    ],
)
def test_non_pass_decisions_must_be_explainable(
    action: Action, codes: list[str], ok: bool
) -> None:
    # FR-014: an unexplained HOLD is not a shippable decision, it is an outage the
    # merchant cannot appeal.
    make = lambda: Decision(  # noqa: E731
        merchant_id="M1",
        as_of=T0.date(),
        score=0.9,
        action=action,
        reason_codes=codes,
        model_version="abc1234:rung2",
        eval_run_id="run-1",
    )
    if ok:
        assert make().action is action
    else:
        with pytest.raises(ValueError, match="reason codes"):
            make()


def an_eval_result(**overrides: object) -> EvalResult:
    base: dict[str, object] = {
        "rung": 2,
        "split": "val",
        "prevalence": 0.015,
        "pr_auc": 0.4,
        "roc_auc": 0.9,
        "ece": 0.02,
        "savings": 100.0,
        "savings_floor_random": 10.0,
        "savings_floor_all_pass": 0.0,
        "savings_floor_all_hold": -50.0,
        "savings_floor_volume_rank": 40.0,
        "precision_at_k": 0.3,
        "recall_at_k": 0.5,
        "alerts_per_day": 50.0,
        "ttd_median_days": 6.0,
        "detection_rate_d7": 0.5,
        "detection_rate_d14": 0.7,
        "detection_rate_d30": 0.8,
        "gap_to_oracle": 0.4,
        "alert_jaccard_wow": 0.6,
        "recall_by_typology": {TypologyId.R1: 0.6},
        "p99_latency_ms": 8.0,
        "state_bytes_p99": 3000.0,
        "model_size_mb": 4.0,
        "eval_lock_sha": "0" * 16,
        "open_count": 0,
        "git_sha": "abc1234",
    }
    return EvalResult(**(base | overrides))  # type: ignore[arg-type]


def test_prevalence_is_never_optional() -> None:
    # FR-021. v1 reported a headline computed at 20% prevalence against a real rate near
    # 1.5% and did not say so. The schema now refuses to hold a row that omits it.
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="prevalence"):
            an_eval_result(prevalence=bad)


def test_floor_comparison_uses_the_worst_floor_not_the_average() -> None:
    assert an_eval_result(savings=100.0).beats_all_floors
    assert not an_eval_result(savings=40.0).beats_all_floors  # ties volume_rank, does not beat
    assert not an_eval_result(savings=39.0).beats_all_floors


# ── polars schemas ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("dataclass_", "schema"),
    [
        (Transaction, TRANSACTION_SCHEMA),
        (MerchantProfile, PROFILE_SCHEMA),
        (Payout, PAYOUT_SCHEMA),
        (Label, LABEL_SCHEMA),
        (GroundTruth, GROUND_TRUTH_SCHEMA),
    ],
)
def test_polars_schema_columns_match_the_dataclass_fields(
    dataclass_: type, schema: dict[str, object]
) -> None:
    # The dataclass is what code passes around; the polars schema is what hits disk. They
    # drift the moment someone adds a field to one and not the other, and the failure mode
    # is a parquet file that reads back with a missing column at 3am.
    assert {f.name for f in fields(dataclass_)} == set(schema)
