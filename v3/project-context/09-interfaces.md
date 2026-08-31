<!-- HEAD
FILE:     09-interfaces.md
PHASE:    2 — SPECIFY
UPDATED:  2026-08-31
STATUS:   frozen at end of Block 1 — changes after that require a DESCEND
SUMMARY:  The data contracts between generator, feature layer, models, and eval harness.
          This file is what makes parallel Claude Code sessions possible: two tickets can
          proceed independently only if the boundary between them is frozen here. All of
          it lands in src/rakshak/schemas.py as the single source of truth.
OPEN:     none — freeze this before Block 2 starts.
-->

# 09 — Interfaces

Everything here becomes a dataclass or Enum in `src/rakshak/schemas.py`. No module
exchanges an ad-hoc dict with another module. If you need a new field, add it here
first, then to `schemas.py`, then use it.

**Conventions, applied everywhere without exception**
- Timestamps: UTC, tz-aware, nanosecond precision. No naive datetimes cross a boundary.
- Money: `float64`, whole rupees. Not Decimal, not paise-integers.
- IDs: `str`. Hashes: 16-hex-char `str`.
- Nullable is explicit: `X | None`, never a sentinel like `-1` or `""`.
- Every persisted table carries a `schema_version: int`.

---

## 1. `Transaction` — the event stream

The only table the feature layer may read. Written to
`data/v2/transactions.parquet`, partitioned by `event_date`.

| Field | Type | Notes |
|---|---|---|
| `event_id` | str | unique |
| `merchant_id` | str | |
| `payer_id` | str | pseudonymous; stable across merchants (needed for F4) |
| `event_time` | datetime64[ns, UTC] | **monotonic within merchant** — property-tested |
| `event_date` | date | partition key, derived |
| `amount_inr` | float64 | > 0 always, including refunds (sign carried by `is_refund`) |
| `instrument` | `Instrument` enum | see §6 |
| `is_cnp` | bool | card-not-present |
| `is_international` | bool | |
| `bin_hash` | str \| None | issuer BIN, hashed; None for non-card |
| `device_hash` | str \| None | |
| `ip_hash` | str \| None | |
| `status` | `TxnStatus` enum | `CAPTURED` / `FAILED` / `PENDING` |
| `decline_code` | str \| None | non-None iff `status == FAILED` |
| `mcc` | str | merchant category code, denormalised for convenience |
| `is_refund` | bool | |
| `refund_of` | str \| None | `event_id` of the original; non-None iff `is_refund` |
| `schema_version` | int | |

**Not present, and must never be added:** anything about chargebacks, disputes,
personas, typologies, or drift onset. Chargebacks are the label. (FR-006)

---

## 2. `MerchantProfile` — onboarding-time facts

Known at `onboarded_at`, constant thereafter. Readable by the feature layer.

| Field | Type | Notes |
|---|---|---|
| `merchant_id` | str | |
| `onboarded_at` | datetime64[ns, UTC] | |
| `mcc` | str | |
| `mcc_group` | str | coarse grouping, used for cohorts |
| `declared_monthly_gmv` | float64 | **denominator of `v_declared_ratio`** — the signal that only exists post-onboarding |
| `kyc_tier` | int | 1–3, ordinal |
| `vintage_months` | int | business age at onboarding |
| `city_tier` | int | 1–3, ordinal |
| `schema_version` | int | |

---

## 3. `Payout` — settlement events

| Field | Type | Notes |
|---|---|---|
| `payout_id` | str | |
| `merchant_id` | str | |
| `requested_at` | datetime64[ns, UTC] | |
| `settled_at` | datetime64[ns, UTC] \| None | None while pending |
| `amount_inr` | float64 | |
| `balance_before_inr` | float64 | `s_balance_drawdown` = amount / balance_before |
| `is_accelerated` | bool | merchant requested faster-than-cycle settlement |

---

## 4. `Label` — the delayed supervision signal

`data/v2/labels.parquet`. **Readable by eval and by training, never by the feature
layer.**

| Field | Type | Notes |
|---|---|---|
| `merchant_id` | str | |
| `label` | int8 \| None | 1 fraud, 0 good, None if censored |
| `label_event_at` | datetime64[ns, UTC] \| None | when the dispute occurred |
| `label_available_at` | datetime64[ns, UTC] \| None | **when the system may use it** |
| `label_source` | `LabelSource` enum | `CHARGEBACK` / `MANUAL_REVIEW` / `NONE` |
| `is_censored` | bool | window extends past simulation end |

**Hard invariant, property-tested:** `label_available_at > label_event_at >= drift_onset_at`.

Any training query must filter `label_available_at <= as_of`. Use
`eval.splits.available_labels(as_of)` rather than writing the filter inline — one
implementation, one place to get it wrong.

---

## 5. `GroundTruth` — quarantined

`data/v2/ground_truth.parquet`. **Importable only by `src/rakshak/eval/` and
`src/rakshak/generator/`.** An AST scan fails CI if `features/` or `models/` touches it.

| Field | Type | Used for |
|---|---|---|
| `merchant_id` | str | |
| `persona_id` | `PersonaId` enum | ablation slicing only |
| `risk_typology_id` | `TypologyId` enum \| None | **per-typology recall reporting** |
| `drift_onset_at` | datetime64[ns, UTC] \| None | **time-to-detection** |
| `true_loss_amount_inr` | float64 | **oracle knapsack, savings metric** |
| `is_unreported` | bool | fraud that never produced a label |

---

## 6. Enums

```python
class Instrument(StrEnum):
    CARD_CREDIT = "card_credit"; CARD_DEBIT = "card_debit"; UPI = "upi"
    NETBANKING = "netbanking";   WALLET = "wallet";        EMI = "emi"
    INTL_CARD = "intl_card"

class TxnStatus(StrEnum):
    CAPTURED = "captured"; FAILED = "failed"; PENDING = "pending"

class Action(StrEnum):
    PASS = "pass"; REVIEW = "review"; HOLD = "hold"

class LabelSource(StrEnum):
    CHARGEBACK = "chargeback"; MANUAL_REVIEW = "manual_review"; NONE = "none"

class PersonaId(StrEnum):   L1 = "L1"; ... ; L8 = "L8"
class TypologyId(StrEnum):  R1 = "R1"; ... ; R9 = "R9"
class ConfounderId(StrEnum): P1 = "P1"; ... ; P6 = "P6"
class Tier(IntEnum):        T1 = 1; T2 = 2; T3 = 3
```

---

## 7. `FeatureSpec` — the dual-runner contract

The single most important interface in the repo. One definition, two runners, parity
asserted to 1e-9. (FR-010, NFR-08)

```python
@dataclass(frozen=True)
class FeatureSpec(Protocol):
    name: str                    # matches an ID in 07-feature-register.md
    tier: Tier
    family: str                  # "F1".."F9"
    state_bytes: int             # declared budget; summed and checked against NFR-04
    human_template: str          # e.g. "GMV is {z:.1f}σ above this merchant's norm"
    has_cohort_residual: bool

    def init_state(self) -> FeatureState: ...

    def update(self, state: FeatureState, event: Transaction) -> None:
        """O(1). Mutates state. Called once per event, in time order."""

    def value(self, state: FeatureState, as_of: datetime) -> float:
        """Read the current value. Must not mutate state."""

    def batch(self, frame: pl.LazyFrame, as_of: datetime) -> pl.Series:
        """Offline equivalent over the full history. Same answer as value()."""
```

**The parity test is not optional and it is not a formality.** It replays the event
stream through `update()` epoch by epoch and compares against `batch()` at every epoch
for every merchant. If a feature disagrees, the feature is wrong — and it is almost
always the batch version, because that is the one that quietly sees the future.

---

## 8. `MerchantState` — the online state object

```python
@dataclass
class MerchantState:
    merchant_id: str
    profile: MerchantProfile
    baseline: BaselineStats        # Welford accumulators, frozen after warmup_days
    feature_states: dict[str, FeatureState]
    last_event_time: datetime | None
    schema_version: int
```

Serialized size ≤ **4 KB** (NFR-04), asserted in `tests/perf/test_state_size.py`. The
`state_bytes` declarations across the registry are summed at import time and the module
refuses to load if the total exceeds budget — fail at startup, not at scale.

Baseline is computed over `warmup_days` (default 30) after onboarding, then **frozen**.
A rolling baseline would let a slow-ramp adversary walk the baseline along with it,
which is exactly how R2 defeats naive drift detection. Freezing the baseline is a
deliberate anti-R2 design choice and should be named as such in the writeup.

---

## 9. `FeatureVector` — feature layer → model

| Field | Type |
|---|---|
| `merchant_id` | str |
| `as_of` | date |
| `values` | `np.ndarray[float64]`, order fixed by `registry.ORDER` |
| `feature_schema_version` | int |
| `computed_by` | `"online"` \| `"offline"` |
| `stage_reached` | int (0/1/2) |

`registry.ORDER` is a module-level tuple. Column order is part of the contract — a model
trained on one order and scored on another fails silently, which is the worst kind.

---

## 10. `Decision` — model → eval

| Field | Type | Notes |
|---|---|---|
| `merchant_id` | str | |
| `as_of` | date | |
| `score` | float64 | calibrated probability in [0,1] |
| `action` | `Action` | |
| `reason_codes` | `list[str]`, len == 3 iff action != PASS | from `pred_contrib` |
| `model_version` | str | git SHA + rung number |
| `eval_run_id` | str | ties back to EVAL-LOCK |

---

## 11. `EvalResult` — harness output

One row per (rung, split, prevalence, cost_scenario). Written to
`docs/results_v2.parquet` and rendered by `make report`.

| Field | Type |
|---|---|
| `rung` | int |
| `split` | `"train"` \| `"val"` \| `"test"` |
| `prevalence` | float64 — **always present** (FR-021) |
| `pr_auc`, `roc_auc`, `ece` | float64 |
| `savings`, `savings_floor_random`, `savings_floor_all_pass`, `savings_floor_all_hold`, `savings_floor_volume_rank` | float64 |
| `precision_at_k`, `recall_at_k`, `alerts_per_day` | float64 |
| `ttd_median_days`, `detection_rate_d7`, `detection_rate_d14`, `detection_rate_d30` | float64 |
| `gap_to_oracle` | float64 |
| `alert_jaccard_wow` | float64 |
| `recall_by_typology` | struct<R1..R9: float64> |
| `p99_latency_ms`, `state_bytes_p99`, `model_size_mb` | float64 |
| `eval_lock_sha`, `open_count`, `git_sha` | str/int |

Provenance columns are load-bearing. A results table you cannot trace back to an exact
commit, lock, and open count is a results table nobody should believe.

---

## 12. Freeze rule

This file freezes at the end of Block 1. After that, a change here is a `DESCEND` — it
means work in flight is being built against a moving contract, and the correct response
is to stop, change the interface deliberately, and restart the affected tickets. Do not
add a field "quickly" mid-sprint.
