<!-- HEAD
FILE:     06-requirements-v2.md
PHASE:    2 — SPECIFY
UPDATED:  2026-08-31
STATUS:   active
SUMMARY:  Numbered, testable functional and non-functional requirements for Rakshak v2.
          Every NFR carries a number and a unit. Includes the floor/oracle/frozen-eval
          declaration that must be written before any model is specified, the rung
          adoption gates, and the non-goals list.
OPEN:     NFR-03 and NFR-10 depend on the confirmed merchant population (charter §10.3).
-->

# 06 — Requirements v2

Priority: **P0** must ship · **P1** should ship · **P2** cut first.

---

## A. Floor, oracle, frozen eval — declared first

Written before any method is specified, deliberately.

**Floors (Rung 0).** Reported alongside every headline number, always, no exceptions.
- `all_pass` — never alert. Establishes the cost of doing nothing.
- `all_hold` — alert on everything. Establishes the cost of paranoia.
- `random_at_K` — alert on K merchants uniformly at random per day.
- `volume_rank` — alert on the K highest-GMV merchants. The dumbest non-random heuristic.

The v1 finding that `random` won on savings at 20% prevalence was a real methodology
result, and the remedy is to make floors mandatory output rather than to hide the
metric. If `random` beats a rung at 1.47% prevalence, that rung is worthless and the
harness will say so on the same line.

**Incumbent (Rung 2).** LightGBM on windowed merchant aggregates. **This is the bar.**

**Oracle (ceiling).** Perfect-foresight knapsack: given true labels and true loss
amounts, choose the K merchants per day that maximise prevented loss. Results are
reported as **gap-to-oracle**, not as unanchored absolutes.

**Frozen eval.** `EVAL-LOCK.json` contains: SHA256 of the scenario config, SHA256 of
`src/rakshak/eval/`, the seed list, the split boundaries, the metric list, the declared
adoption margins from charter §2, and `open_count`. Written in Block 5. `make eval`
verifies the hashes match and refuses to run against the test split unless
`RAKSHAK_UNLOCK=1`; when it does run, it increments and commits `open_count`.

---

## B. Functional requirements

### B1 — Generator (`src/rakshak/generator/`)

```
FR-001 | The generator shall emit a transaction stream for N merchants over D days
        from a single integer seed and a YAML scenario config.
  Priority:    P0
  Rationale:   every downstream comparison is only as trustworthy as this
  Acceptance:  GIVEN seed=42 and configs/scenario_v2.yaml WHEN `make gen` runs twice
               THEN the SHA256 of the output parquet is identical both times
  Verified by: tests/unit/test_determinism.py

FR-002 | The generator shall assign each merchant exactly one legitimate persona from
        L1–L8, and optionally one risk typology from R1–R9.
  Priority:    P0
  Rationale:   a fraud merchant is a legitimate merchant that turned; the persona is
               the pre-drift behaviour and must exist independently of the typology
  Acceptance:  GIVEN any generated population THEN every merchant has a persona_id,
               and merchants with a risk_typology_id also have a drift_onset_at that
               falls strictly after their onboarded_at
  Verified by: tests/unit/test_generator_invariants.py

FR-003 | The generator shall produce transaction inter-arrival times from an
        overdispersed process whose daily-count Fano factor is configurable.
  Priority:    P0
  Rationale:   measured Fano 12.25 vs Poisson 1.0; misspecified emissions were a
               named cause of the v1 failure
  Acceptance:  GIVEN target_fano=12.25 THEN the realised Fano factor over the
               population is 12.25 ± 1.0
  Verified by: tests/gates/test_g1_marginal_parity.py

FR-004 | The generator shall emit platform-wide confounder events P1–P6 that shift the
        features of ALL merchants without any fraud occurring.
  Priority:    P0
  Rationale:   distinguishing adversarial drift from natural platform drift is the
               open research problem; it cannot be tested without generating it
  Acceptance:  GIVEN prevalence=0.0 and confounders enabled THEN the ground-truth
               fraud count is 0 and at least 3 distinct platform events are present
  Verified by: tests/gates/test_g5_confounder_null.py

FR-005 | The generator shall emit labels with a delay drawn from U(45,120) days, a
        configurable false-negative rate for unreported fraud, and a censoring flag
        for merchants whose label window extends past the simulation end.
  Priority:    P0
  Acceptance:  GIVEN any label row THEN label_available_at > label_event_at, and rows
               beyond the horizon carry is_censored=true and label=NULL
  Verified by: tests/unit/test_labels.py

FR-006 | The generator shall write ground-truth fields (persona_id, risk_typology_id,
        drift_onset_at, true_loss_amount) to a physically separate `ground_truth`
        table, never into the transaction stream.
  Priority:    P0
  Rationale:   leakage prevention by construction beats leakage prevention by care
  Acceptance:  GIVEN the transaction parquet THEN none of the forbidden column names
               are present
  Verified by: tests/gates/test_no_ground_truth_import.py
```

### B2 — Features (`src/rakshak/features/`)

```
FR-010 | Every feature shall implement both a batch runner and an online update, from
        a single FeatureSpec definition.
  Priority:    P0
  Acceptance:  GIVEN any registered feature and any generated stream THEN
               max|batch_value - online_value| <= 1e-9 at every epoch
  Verified by: tests/parity/test_feature_parity.py  (NFR-08)

FR-011 | The feature layer shall express every drift feature relative to the merchant's
        own post-onboarding baseline, not as an absolute level.
  Priority:    P0
  Rationale:   merchant heterogeneity; a ₹2L ticket is normal for jewellery
  Acceptance:  GIVEN two merchants with identical drift shape but 100x different scale
               THEN their drift feature values agree within 5%
  Verified by: tests/unit/test_scale_invariance.py

FR-012 | The feature layer shall emit, for each Tier-1 drift feature, a cohort-residual
        variant: the merchant's z-score minus the leave-one-out cohort median z-score
        at the same epoch.
  Priority:    P0  (this is the sprint's hypothesis — see charter K-1)
  Acceptance:  GIVEN a population under confounder P1 with prevalence=0 THEN the mean
               absolute cohort residual is < 0.25 while the mean absolute raw z-score
               is > 1.0
  Verified by: tests/gates/test_g5_confounder_null.py

FR-013 | Cohort assignment shall be (mcc_group, gmv_decile, vintage_bucket), backing
        off to mcc_group and then global when membership < 30, with empirical-Bayes
        shrinkage of the merchant's own baseline toward the cohort prior.
  Priority:    P0
  Acceptance:  GIVEN a merchant with < 14 days of history THEN its baseline is
               shrunk toward the cohort prior with weight n/(n+k), k from config
  Verified by: tests/unit/test_cohort.py

FR-014 | Feature computation shall run as a three-stage cascade: Stage 0 (T1 only, all
        merchants), Stage 1 (T1+T2+cohort, top 10% by Stage-0 score), Stage 2
        (reason codes, non-PASS decisions only).
  Priority:    P1
  Rationale:   this is how NFR-03 is met; without it the daily sweep does not fit
  Acceptance:  GIVEN a 10k-merchant day THEN Stage 1 runs on ≤1,000 merchants and the
               end-to-end sweep meets NFR-03
  Verified by: tests/perf/test_sweep_budget.py
```

### B3 — Eval harness (`src/rakshak/eval/`)

```
FR-020 | Splits shall be simultaneously temporal, merchant-group disjoint, and
        label-availability aware: training at decision time t may use only labels with
        label_available_at <= t.
  Priority:    P0
  Rationale:   the 45–120 day chargeback window is the defining domain constraint;
               ignoring it is the most common silent cheat in this literature
  Acceptance:  GIVEN any training fold THEN no merchant_id appears in two folds AND no
               label with label_available_at > fold_end is present
  Verified by: tests/unit/test_splits.py

FR-021 | Every reported metric shall be accompanied by all four Rung-0 floors and by
        the declared evaluation prevalence, in the same output table.
  Priority:    P0
  Rationale:   AP-06 — savings-style metrics degenerate at inflated prevalence
  Acceptance:  GIVEN any results table THEN it contains a `prevalence` column and four
               floor rows
  Verified by: tests/unit/test_report.py

FR-022 | The harness shall compute time-to-detection as (first epoch with action in
        {REVIEW, HOLD}) − drift_onset_at, reporting median TTD over uncensored
        positives and detection rate at days 7, 14, 30.
  Priority:    P0
  Rationale:   PR-AUC hides latency; operational value is days of loss prevented
  Verified by: tests/unit/test_metrics.py

FR-023 | The harness shall compute precision@K and alerts-per-day under the configured
        analyst capacity K, and gap-to-oracle against the perfect-foresight knapsack.
  Priority:    P0
  Verified by: tests/unit/test_capacity.py, tests/unit/test_oracle.py

FR-024 | The harness shall compute week-over-week alert-set Jaccard stability.
  Priority:    P1
  Rationale:   an alert set that churns every week is unusable by an ops team
               regardless of its PR-AUC
  Verified by: tests/unit/test_metrics.py

FR-025 | `make eval` shall verify EVAL-LOCK hashes before running and shall refuse the
        test split unless RAKSHAK_UNLOCK=1, incrementing open_count when it does.
  Priority:    P0
  Verified by: tests/unit/test_lock.py

FR-026 | The harness shall provide a BAF adapter that maps BAF columns onto the
        internal schema so any rung can be scored on real data.
  Priority:    P1
  Rationale:   the only defence against synthetic circularity
  Verified by: tests/gates/test_g2_baseline_transfer.py
```

### B4 — Models and decision layer

```
FR-030 | Rungs 0–2 shall be implemented: floors, static rule engine, LightGBM on
        windowed aggregates.
  Priority:    P0

FR-031 | Rung 3 shall be LightGBM on the Rung-2 feature set plus the cohort-residual
        features, with no other change, so the delta is attributable.
  Priority:    P0
  Rationale:   a clean single-variable experiment; anything else confounds the test
               of charter K-1

FR-032 | Rung 4 shall place the instance-dependent cost inside the training objective
        rather than applying Bayes-minimum-risk after the fact.
  Priority:    P2  (first cut)

FR-033 | Every non-PASS decision shall carry three reason codes derived from LightGBM
        pred_contrib, mapped to human-readable strings via the feature registry.
  Priority:    P0
  Rationale:   replaces the Viterbi path as the audit trail, at near-zero cost
  Acceptance:  GIVEN any HOLD decision THEN exactly 3 non-empty reason strings are
               attached, each naming a feature and a direction
  Verified by: tests/unit/test_reason_codes.py

FR-034 | The action selector shall choose PASS/REVIEW/HOLD to minimise expected cost
        subject to at most K REVIEW+HOLD actions per day.
  Priority:    P0
  Verified by: tests/unit/test_capacity.py
```

---

## C. Non-functional requirements

Every one carries a number and a unit. An NFR without a number is a wish.

| ID | Category | Requirement | Verified by |
|---|---|---|---|
| **NFR-01** | Latency | Stage-0 screen: p99 ≤ **0.5 ms** per merchant-epoch, 1 core | `tests/perf/test_stage0_latency.py` |
| **NFR-02** | Latency | Stage-1 full scoring: p99 ≤ **10 ms** per merchant-epoch, 1 core | `tests/perf/test_stage1_latency.py` |
| **NFR-03** | Throughput | Full daily sweep of **10,000 merchants ≤ 30 s** on 4 cores (scales to ≤ 5 min at 100k) | `tests/perf/test_sweep_budget.py` |
| **NFR-04** | Memory | `MerchantState` ≤ **4 KB** serialized per merchant | `tests/perf/test_state_size.py` |
| **NFR-05** | Artifact | Trained model file ≤ **20 MB** | `tests/perf/test_model_size.py` |
| **NFR-06** | Training | Any rung trains in ≤ **20 min** on 4 cores, no GPU | `tests/perf/test_train_budget.py` |
| **NFR-07** | Determinism | Same seed ⇒ **byte-identical** output SHA256, across two clean runs | `tests/unit/test_determinism.py` |
| **NFR-08** | Correctness | Online vs offline feature agreement: max abs diff ≤ **1e-9** | `tests/parity/test_feature_parity.py` |
| **NFR-09** | Stability | Week-over-week alert-set Jaccard ≥ **0.60** on non-drifting merchants | `tests/unit/test_metrics.py` |
| **NFR-10** | Generator speed | 10,000 merchants × 180 days ≤ **3 min** on 4 cores | `tests/perf/test_gen_budget.py` |
| **NFR-11** | Coverage | ≥ **80%** line coverage on `generator/`, `features/`, `eval/` | CI |
| **NFR-12** | Reproducibility | `make all` passes from a **clean `git clone`** on a fresh env | CI job `clean-clone` |
| **NFR-13** | Typing | `mypy --strict src/` returns **0 errors** | CI |
| **NFR-14** | Portability | Runs on Linux and macOS, Python 3.11, **CPU only**, no network at runtime | CI matrix |

NFR-12 gets special attention: the v1 build's `make eval` did not reproduce on a clean
checkout, and that was assessed as the highest-risk demo-day failure mode. It is now a
blocking CI job rather than a hope.

---

## D. Rung adoption gates

A rung enters the scoring path only if **all four** hold. Declared here, before results.

1. Beats the rung below by ≥10% relative PR-AUC **or** ≥3 days median TTD at equal
   alerts/day.
2. Beats all four Rung-0 floors on **every** reported metric.
3. Meets NFR-01 through NFR-06.
4. Gate G5 (confounder null) stays green — the rung must not alert on platform drift.

A rung that fails any gate is written into `LIMITATIONS.md` with its number and dropped.
It is not tuned until it passes; that is what the validation split is for, and once the
test split has been opened, tuning is over.

---

## E. Non-goals

Explicitly out of scope for v2. Listing them is what prevents scope creep at hour 30.

- **No UI, dashboard, or web service.** CLI and parquet only.
- **No online/incremental model learning.** Features are online; the model is batch
  retrained. Delayed labels make anything else unsound and unverifiable in 48 hours.
- **No GNN, no transformer, no RNN, no neural TPP.** Deferred with reasons in the
  lit survey; the ADR rejecting GNNs stands on synthetic-graph circularity.
- ~~No MIL layer (Rung 5) and no HSMM explanation layer (Rung 7) in this sprint~~ —
  **reversed 2026-08-31 (GitHub #51)**. Rungs 5–7 were built and scored in cycle 4; see
  `LIMITATIONS.md` §9.9. Rung 8 (neural conditional intensity) is still a non-goal for
  this sprint — `status: planned` in `configs/rung_roster.yaml`.
- **No real-time streaming infrastructure** (Kafka, Flink). The online feature runner
  proves streaming-computability; deploying it is not this sprint's job.
- **No multi-currency, no cross-border settlement modelling.** INR only.
- **No attempt to beat BAF's published leaderboard.** BAF is used as an anchor for
  transfer and prevalence, not as a competition.
