"""The single source of truth for every shape that crosses a module boundary.

This file *is* ``project-context/09-interfaces.md``, expressed as code. No module
exchanges an ad-hoc dict with another module: if two modules trade data, the shape is
here. That is what makes the parallel lanes possible — Lane A can build the generator
against these dataclasses while Lane B builds the feature layer, and neither lane has to
read the other's code.

**This file is frozen at the end of Block 1.** After that, a change here is a DESCEND: it
means work in flight is being built against a moving contract. Stop, change the interface
deliberately, and restart the affected tickets. Do not add a field "quickly".

Conventions, applied without exception (09-interfaces.md §Conventions):

- Timestamps are UTC, tz-aware, nanosecond. No naive datetime crosses a boundary.
- Money is float64 in whole rupees. Not Decimal, not paise-integers.
- IDs are ``str``. Hashes are 16-hex-char ``str``.
- Nullable is explicit ``X | None`` — never a sentinel like ``-1`` or ``""``.
- Every persisted table carries ``schema_version: int``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import IntEnum, StrEnum
from typing import Literal

import numpy as np
import polars as pl

from rakshak import SCHEMA_VERSION

__all__ = [
    "CARD_INSTRUMENTS",
    "GROUND_TRUTH_SCHEMA",
    "HASH_LEN",
    "LABEL_SCHEMA",
    "PAYOUT_SCHEMA",
    "PROFILE_SCHEMA",
    "RADIOACTIVE_FIELDS",
    "SCHEMA_VERSION",
    "TIMESTAMP",
    "TRANSACTION_SCHEMA",
    "Action",
    "ConfounderId",
    "Decision",
    "EvalResult",
    "FeatureVector",
    "GroundTruth",
    "Instrument",
    "Label",
    "LabelSource",
    "MerchantProfile",
    "Payout",
    "PersonaId",
    "Split",
    "Tier",
    "Transaction",
    "TxnStatus",
    "TypologyId",
]

# 09-interfaces.md §Conventions: hashes are 16 hex chars. Long enough that a collision
# across 10k merchants x 180 days is not a thing that happens, short enough to eyeball in
# a failing test.
HASH_LEN = 16

#: The tz-aware nanosecond timestamp type every table uses. Defined once so that no table
#: can quietly be written at microsecond precision or on a naive clock.
TIMESTAMP = pl.Datetime("ns", "UTC")

Split = Literal["train", "val", "test"]


# ─────────────────────────────────────────────────────────────────────────────
# §6  Enums
# ─────────────────────────────────────────────────────────────────────────────


class Instrument(StrEnum):
    CARD_CREDIT = "card_credit"
    CARD_DEBIT = "card_debit"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMI = "emi"
    INTL_CARD = "intl_card"


#: Instruments that carry an issuer BIN. ``bin_hash`` is non-None iff the instrument is
#: one of these — asserted in ``Transaction.__post_init__`` so that a generator bug cannot
#: ship a card transaction with no BIN and have the issuer-entropy feature silently read
#: zero for it.
CARD_INSTRUMENTS: frozenset[Instrument] = frozenset(
    {
        Instrument.CARD_CREDIT,
        Instrument.CARD_DEBIT,
        Instrument.EMI,
        Instrument.INTL_CARD,
    }
)


class TxnStatus(StrEnum):
    CAPTURED = "captured"
    FAILED = "failed"
    PENDING = "pending"


class Action(StrEnum):
    PASS = "pass"
    REVIEW = "review"
    HOLD = "hold"


class LabelSource(StrEnum):
    CHARGEBACK = "chargeback"
    MANUAL_REVIEW = "manual_review"
    NONE = "none"


class PersonaId(StrEnum):
    """L1-L8, the legitimate merchant behaviours. Ground truth: ablation slicing only."""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    L6 = "L6"
    L7 = "L7"
    L8 = "L8"


class TypologyId(StrEnum):
    """R1-R9, the named fraud patterns.

    Fraud is never one undifferentiated class here. Per-typology recall is a reported
    metric precisely so that a rung which wins on the average while missing R2 entirely
    cannot hide behind that average.
    """

    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"
    R5 = "R5"
    R6 = "R6"
    R7 = "R7"
    R8 = "R8"
    R9 = "R9"


class ConfounderId(StrEnum):
    """P1-P6, platform-wide events that move everyone's features with no fraud present.

    Gate G5 asserts the detector does not alert on these.
    """

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
    P6 = "P6"


class Tier(IntEnum):
    """Inference-cascade tier.

    T1 runs on every merchant every day, T2 only on the top 10% from stage 0, and T3 is
    graph-scoped and out of scope for this sprint.
    """

    T1 = 1
    T2 = 2
    T3 = 3


# ─────────────────────────────────────────────────────────────────────────────
# §1  Transaction — the event stream
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Transaction:
    """The only table the feature layer may read.

    Not present, and must never be added (FR-006): anything about chargebacks, disputes,
    personas, typologies, or drift onset. Chargebacks are the label, and a label on the
    event stream is the leak that makes every downstream number meaningless.
    """

    event_id: str
    merchant_id: str
    payer_id: str
    event_time: datetime
    event_date: date
    amount_inr: float
    instrument: Instrument
    is_cnp: bool
    is_international: bool
    bin_hash: str | None
    device_hash: str | None
    ip_hash: str | None
    status: TxnStatus
    decline_code: str | None
    mcc: str
    is_refund: bool
    refund_of: str | None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # These are invariants of the *contract*, not of any one generator. Checking them
        # at construction means a malformed row fails on the line that made it, rather
        # than forty minutes downstream inside a feature reading a None it was promised.
        if self.amount_inr <= 0:
            raise ValueError(
                "amount_inr must be > 0 including refunds (the sign is carried by "
                f"is_refund); got {self.amount_inr!r} for event_id={self.event_id!r}"
            )
        if self.event_time.tzinfo is None:
            raise ValueError(f"event_time must be tz-aware UTC; got naive {self.event_time!r}")
        if self.event_date != self.event_time.date():
            raise ValueError(
                f"event_date {self.event_date!r} is not the date of event_time "
                f"{self.event_time!r} — it is a derived partition key, not a free field"
            )
        if (self.decline_code is not None) != (self.status is TxnStatus.FAILED):
            raise ValueError(
                "decline_code is non-None iff status is FAILED; got "
                f"status={self.status!r} decline_code={self.decline_code!r}"
            )
        if (self.refund_of is not None) != self.is_refund:
            raise ValueError(
                "refund_of is non-None iff is_refund; got "
                f"is_refund={self.is_refund!r} refund_of={self.refund_of!r}"
            )
        if (self.bin_hash is not None) != (self.instrument in CARD_INSTRUMENTS):
            raise ValueError(
                "bin_hash is non-None iff the instrument carries an issuer BIN; got "
                f"instrument={self.instrument!r} bin_hash={self.bin_hash!r}"
            )


TRANSACTION_SCHEMA: dict[str, pl.DataType] = {
    "event_id": pl.String(),
    "merchant_id": pl.String(),
    "payer_id": pl.String(),
    "event_time": TIMESTAMP,
    "event_date": pl.Date(),
    "amount_inr": pl.Float64(),
    "instrument": pl.String(),
    "is_cnp": pl.Boolean(),
    "is_international": pl.Boolean(),
    "bin_hash": pl.String(),
    "device_hash": pl.String(),
    "ip_hash": pl.String(),
    "status": pl.String(),
    "decline_code": pl.String(),
    "mcc": pl.String(),
    "is_refund": pl.Boolean(),
    "refund_of": pl.String(),
    "schema_version": pl.Int32(),
}


# ─────────────────────────────────────────────────────────────────────────────
# §2  MerchantProfile — onboarding-time facts
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MerchantProfile:
    """Known at ``onboarded_at``, constant thereafter. Readable by the feature layer.

    ``declared_monthly_gmv`` is the denominator of ``v_declared_ratio`` — the one signal
    that exists only *after* onboarding, because it compares what the merchant said it
    would do against what it then did. Bumblebee cannot see it. That gap is the project.
    """

    merchant_id: str
    onboarded_at: datetime
    mcc: str
    mcc_group: str
    declared_monthly_gmv: float
    kyc_tier: int
    vintage_months: int
    city_tier: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.onboarded_at.tzinfo is None:
            raise ValueError(f"onboarded_at must be tz-aware UTC; got {self.onboarded_at!r}")
        if self.declared_monthly_gmv <= 0:
            raise ValueError(
                "declared_monthly_gmv is a denominator and must be > 0; got "
                f"{self.declared_monthly_gmv!r} for {self.merchant_id!r}"
            )
        if not 1 <= self.kyc_tier <= 3:
            raise ValueError(f"kyc_tier is ordinal 1-3; got {self.kyc_tier!r}")
        if not 1 <= self.city_tier <= 3:
            raise ValueError(f"city_tier is ordinal 1-3; got {self.city_tier!r}")


PROFILE_SCHEMA: dict[str, pl.DataType] = {
    "merchant_id": pl.String(),
    "onboarded_at": TIMESTAMP,
    "mcc": pl.String(),
    "mcc_group": pl.String(),
    "declared_monthly_gmv": pl.Float64(),
    "kyc_tier": pl.Int32(),
    "vintage_months": pl.Int32(),
    "city_tier": pl.Int32(),
    "schema_version": pl.Int32(),
}


# ─────────────────────────────────────────────────────────────────────────────
# §3  Payout — settlement events
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Payout:
    payout_id: str
    merchant_id: str
    requested_at: datetime
    settled_at: datetime | None
    amount_inr: float
    balance_before_inr: float
    is_accelerated: bool
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.requested_at.tzinfo is None:
            raise ValueError(f"requested_at must be tz-aware UTC; got {self.requested_at!r}")
        if self.settled_at is not None and self.settled_at < self.requested_at:
            raise ValueError(
                f"settled_at {self.settled_at!r} precedes requested_at "
                f"{self.requested_at!r} for payout {self.payout_id!r}"
            )
        if self.balance_before_inr <= 0:
            raise ValueError(
                "balance_before_inr is the denominator of s_balance_drawdown and must be "
                f"> 0; got {self.balance_before_inr!r} for payout {self.payout_id!r}"
            )


PAYOUT_SCHEMA: dict[str, pl.DataType] = {
    "payout_id": pl.String(),
    "merchant_id": pl.String(),
    "requested_at": TIMESTAMP,
    "settled_at": TIMESTAMP,
    "amount_inr": pl.Float64(),
    "balance_before_inr": pl.Float64(),
    "is_accelerated": pl.Boolean(),
    "schema_version": pl.Int32(),
}


# ─────────────────────────────────────────────────────────────────────────────
# §4  Label — the delayed supervision signal
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Label:
    """Readable by eval and by training, never by the feature layer.

    ``label_available_at`` is the field that makes v2 honest. v1 trained against labels
    the instant the fraud occurred, which measured a system that cannot exist. Every
    training query must filter ``label_available_at <= as_of``, and must do it through
    ``eval.splits.available_labels(as_of)`` rather than inline — one implementation, one
    place to get it wrong.
    """

    merchant_id: str
    label: int | None
    label_event_at: datetime | None
    label_available_at: datetime | None
    label_source: LabelSource
    is_censored: bool
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.label not in (0, 1, None):
            raise ValueError(f"label is 1 fraud / 0 good / None censored; got {self.label!r}")
        # The third leg of the hard invariant — `>= drift_onset_at` — lives in GroundTruth
        # and is property-tested across the join in tests/unit/test_labels.py, because
        # drift_onset_at is quarantined and must not be reachable from this dataclass.
        if self.label_available_at is not None and self.label_event_at is not None:
            if self.label_available_at <= self.label_event_at:
                raise ValueError(
                    "label_available_at must be strictly after label_event_at "
                    f"({self.label_available_at!r} <= {self.label_event_at!r}) for "
                    f"{self.merchant_id!r}: a label is never usable the instant it occurs"
                )
        if self.is_censored and self.label is not None:
            raise ValueError(
                "a censored merchant has no resolved label; got "
                f"label={self.label!r} for {self.merchant_id!r}"
            )


LABEL_SCHEMA: dict[str, pl.DataType] = {
    "merchant_id": pl.String(),
    "label": pl.Int8(),
    "label_event_at": TIMESTAMP,
    "label_available_at": TIMESTAMP,
    "label_source": pl.String(),
    "is_censored": pl.Boolean(),
    "schema_version": pl.Int32(),
}


# ─────────────────────────────────────────────────────────────────────────────
# §5  GroundTruth — quarantined
# ─────────────────────────────────────────────────────────────────────────────

#: Prime Directive 3. Any of these names appearing anywhere under
#: ``src/rakshak/features/`` or ``src/rakshak/models/`` fails CI
#: (``tests/gates/test_no_ground_truth_import.py``). They are importable only by
#: ``src/rakshak/eval/`` and ``src/rakshak/generator/``.
RADIOACTIVE_FIELDS: frozenset[str] = frozenset(
    {
        "GroundTruth",
        "drift_onset_at",
        "is_unreported",
        "persona_id",
        "risk_typology_id",
        "true_loss_amount_inr",
    }
)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Generator truth: ablation slicing, per-typology recall, time-to-detection, and the
    oracle knapsack. Nothing else, ever.

    If you find yourself wanting one of these fields inside a feature or a model, that is
    the signal that the feature is cheating — not that the quarantine is inconvenient.
    """

    merchant_id: str
    persona_id: PersonaId
    risk_typology_id: TypologyId | None
    drift_onset_at: datetime | None
    true_loss_amount_inr: float
    is_unreported: bool
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (self.risk_typology_id is None) != (self.drift_onset_at is None):
            raise ValueError(
                "a merchant has a typology iff it has a drift onset; got "
                f"typology={self.risk_typology_id!r} onset={self.drift_onset_at!r} for "
                f"{self.merchant_id!r}"
            )
        if self.risk_typology_id is None and self.true_loss_amount_inr != 0.0:
            raise ValueError(
                "a merchant with no typology causes no loss; got "
                f"true_loss_amount_inr={self.true_loss_amount_inr!r} for {self.merchant_id!r}"
            )
        if self.risk_typology_id is not None and self.true_loss_amount_inr <= 0:
            raise ValueError(
                "every fraud merchant must carry a positive true_loss_amount_inr — it is "
                f"the weight in the oracle knapsack; got {self.true_loss_amount_inr!r} "
                f"for {self.merchant_id!r}"
            )


GROUND_TRUTH_SCHEMA: dict[str, pl.DataType] = {
    "merchant_id": pl.String(),
    "persona_id": pl.String(),
    "risk_typology_id": pl.String(),
    "drift_onset_at": TIMESTAMP,
    "true_loss_amount_inr": pl.Float64(),
    "is_unreported": pl.Boolean(),
    "schema_version": pl.Int32(),
}


# ─────────────────────────────────────────────────────────────────────────────
# §9  FeatureVector — feature layer → model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Column order is part of the contract.

    A model trained on one column order and scored on another fails *silently*, which is
    the worst failure mode available to this repo. ``values`` is ordered by
    ``registry.ORDER`` and by nothing else.
    """

    merchant_id: str
    as_of: date
    values: np.ndarray
    feature_schema_version: int
    computed_by: Literal["online", "offline"]
    stage_reached: int

    def __post_init__(self) -> None:
        if self.values.dtype != np.float64:
            raise ValueError(f"FeatureVector.values must be float64; got {self.values.dtype}")
        if self.stage_reached not in (0, 1, 2):
            raise ValueError(f"stage_reached is 0/1/2 in the cascade; got {self.stage_reached!r}")


# ─────────────────────────────────────────────────────────────────────────────
# §10  Decision — model → eval
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Decision:
    merchant_id: str
    as_of: date
    score: float
    action: Action
    reason_codes: list[str]
    model_version: str
    eval_run_id: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score is a calibrated probability in [0,1]; got {self.score!r}")
        # FR-014: every non-PASS decision is merchant-readable, or it is not shippable.
        want = 0 if self.action is Action.PASS else 3
        if len(self.reason_codes) != want:
            raise ValueError(
                f"action={self.action!r} requires exactly {want} reason codes; got "
                f"{len(self.reason_codes)} for {self.merchant_id!r} on {self.as_of!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# §11  EvalResult — harness output
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EvalResult:
    """One row per (rung, split, prevalence, cost_scenario).

    The provenance fields at the bottom are load-bearing, not bookkeeping. A results table
    you cannot trace back to an exact commit, lock, and open count is a results table
    nobody should believe — including you, in three days, when a number moves.
    """

    rung: int
    split: Split
    prevalence: float
    pr_auc: float
    roc_auc: float
    ece: float
    savings: float
    savings_floor_random: float
    savings_floor_all_pass: float
    savings_floor_all_hold: float
    savings_floor_volume_rank: float
    precision_at_k: float
    recall_at_k: float
    alerts_per_day: float
    ttd_median_days: float
    detection_rate_d7: float
    detection_rate_d14: float
    detection_rate_d30: float
    gap_to_oracle: float
    alert_jaccard_wow: float
    recall_by_typology: dict[TypologyId, float]
    p99_latency_ms: float
    state_bytes_p99: float
    model_size_mb: float
    eval_lock_sha: str
    open_count: int
    git_sha: str
    cost_scenario: str = "base"
    floor_fail: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # FR-021: prevalence is always present. v1's headline was computed at 20%
        # prevalence against a real rate near 1.5%, and reported without saying so.
        if not 0.0 < self.prevalence < 1.0:
            raise ValueError(
                "prevalence must be a reported rate in (0,1) — it is never optional "
                f"(FR-021); got {self.prevalence!r}"
            )

    @property
    def beats_all_floors(self) -> bool:
        """A rung that loses to any floor is FLOOR-FAIL regardless of its PR-AUC.

        Savings below the ``all_pass`` floor means the system costs more than doing
        nothing at all, which no amount of ranking quality redeems.
        """
        return self.savings > max(
            self.savings_floor_random,
            self.savings_floor_all_pass,
            self.savings_floor_all_hold,
            self.savings_floor_volume_rank,
        )
