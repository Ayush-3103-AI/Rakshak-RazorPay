<!-- HEAD
FILE:     06-requirements.md
PHASE:    2 — SPECIFY
UPDATED:  2026-08-28
STATUS:   gated
SUMMARY:  21 FRs and 9 NFRs, each with GIVEN/WHEN/THEN acceptance. Headline gate: NFR-001,
          savings score >=20% relative over the static rule engine at equal review budget on a
          temporally-and-group-split held-out set. Floor = rule engine; ceiling = perfect-
          foresight knapsack oracle; frozen eval defined in section 4 BEFORE any model exists.
          FR-005 mandates reporting degraded recall on the slow-ramp typology. Non-goals listed
          at the end. Every requirement traces to a clause in the Track 02 published bar.
OPEN:     Cost matrix values in 07-math.md §5 are provisional pending FR-020 sensitivity analysis.
-->

# 06 — Requirements

Every requirement traces to a clause in the Track 02 bar: *"a working detector, verifier or auto-responder for one class of loss, with measured precision and recall on a held-out test set... honest metrics including false-positive cost... strictly defense-only."*

---

## 1. Functional requirements

### Data generation

```
FR-001 | Generate synthetic merchant transaction streams with configurable merchant count and horizon
  Priority:    MUST
  Rationale:   No public merchant-sequence data exists — ADR-0007
  Acceptance:  GIVEN a seed and a config WHEN generate.py runs THEN it emits a parquet of
               transactions with merchant_id, timestamp, amount, payer_id, method, mcc,
               is_refund, is_chargeback — and a separate ground-truth table of per-merchant
               state paths with transition timestamps
  Verified by: schema test + row-count assertion + determinism test (same seed → identical hash)
```

```
FR-002 | Merchant population spans realistic heterogeneity
  Priority:    MUST
  Rationale:   Within-merchant normalisation (P-02) is only meaningful if merchants differ
  Acceptance:  GIVEN generated data WHEN summarised THEN AOV spans at least two orders of
               magnitude, monthly volume spans at least two orders of magnitude, and at least
               6 distinct MCC segments are present
  Verified by: test asserting the spread
```

```
FR-003 | Ground truth records the transition TIME, not only the label
  Priority:    MUST
  Rationale:   Enables detection-lag measurement (FR-017), which is more informative than
               precision alone and maps directly to money lost
  Acceptance:  GIVEN a merchant that transitions THEN ground truth contains the exact index
               and timestamp at which the latent state changed
  Verified by: test on a known-transition fixture
```

```
FR-004 | Inject exactly four fraud typologies
  Priority:    MUST
  Rationale:   D-14. Four gives credible per-class metrics; five spreads thin
  Acceptance:  GIVEN the generator THEN it supports and labels:
                 (a) BUST_OUT           — legitimate history, then hard volume ramp, then vanish
                 (b) LAUNDERING_ENDPOINT — normal tickets, abnormal payer graph: many payers,
                                           no repeats, no organic returns
                 (c) CATEGORY_DRIFT     — silently shifts ticket-size and timing profile to a
                                           different business category
                 (d) REFUND_COLLUSION   — merchant and a small payer set cooperate to extract
                                           value via refunds/chargebacks
  Verified by: per-typology fixture test; each is separably detectable by at least one emission
```

```
FR-005 | Provide a fifth ADVERSARIAL typology: slow-ramp evader
  Priority:    MUST
  Rationale:   D-15. A measured failure is the strongest honesty signal available
  Acceptance:  GIVEN the evaluation THEN recall on SLOW_RAMP is reported in its own row of the
               results table, and the README states plainly that performance degrades on it
  Verified by: presence of the row in results/typology_breakdown.md
  NOTE:        DO NOT TUNE THIS AWAY. It is included in order to be reported as a weakness.
```

```
FR-006 | The generator is documented as an evaluation artifact, not a fraud toolkit
  Priority:    MUST
  Rationale:   Track rule — "strictly defense-only: anything offense-capable is disqualified"
  Acceptance:  GIVEN src/rakshak/generator/ THEN its module docstring and the README state that
               it exists to test detection and produces no output usable against a live system
  Verified by: inspection; documented in README §Scope and Safety
```

### Feature layer

```
FR-007 | Emissions are standardised WITHIN merchant, not across the population
  Priority:    MUST
  Rationale:   P-02 — the single most important modelling decision; it is what prevents the
               2008-era cardholder-HMM false-positive failure
  Acceptance:  GIVEN a merchant's window features WHEN standardised THEN the location and scale
               parameters are estimated from that merchant's own burn-in window only, with
               shrinkage to segment for short histories
  Verified by: unit test — two merchants with identical relative behaviour but 100× different
               AOV produce near-identical standardised emissions
```

```
FR-008 | Emission vector includes graph-derived scalar features
  Priority:    MUST
  Rationale:   ADR-0002 — approximates the GNN signal on CPU
  Acceptance:  GIVEN a window THEN the emission vector contains at minimum: payer-set entropy,
               repeat-payer ratio, payer-set Jaccard similarity vs. previous window, and
               Herfindahl concentration on payer volume
  Verified by: schema test; ablation in FR-018
```

```
FR-009 | Emission vector includes behavioural and financial features
  Priority:    MUST
  Acceptance:  GIVEN a window THEN it contains: log ticket-size mean and variance, transaction
               velocity, refund ratio, chargeback ratio, chargeback lag, hour-of-day entropy,
               payment-method mix entropy, new-payer ratio
  Verified by: schema test
```

```
FR-010 | Vulcan-proxy risk score is consumable as an emission
  Priority:    SHOULD
  Rationale:   D-03 — the architecture must show Rakshak consuming Razorpay's existing
               transaction-level score rather than competing with it
  Acceptance:  GIVEN the config WHEN a per-transaction risk score column is present THEN its
               window-aggregated mean and 95th percentile enter the emission vector; WHEN absent
               THEN the pipeline runs without it and logs the omission
  Verified by: test with the column present and absent
```

```
FR-011 | Merchants are assigned to segments for pooling
  Priority:    MUST
  Rationale:   ADR-0006 — shrinkage requires a defensible segment definition
  Acceptance:  GIVEN a merchant THEN it maps to exactly one segment defined as MCC × AOV-band,
               and every segment in the training set contains at least 20 merchants
  Verified by: test asserting minimum segment population
```

### Model layer

```
FR-012 | Hand-written HMM with forward, backward, Viterbi, and Baum-Welch
  Priority:    MUST
  Rationale:   ADR-0001
  Acceptance:  GIVEN a sequence THEN forward returns log-likelihood, Viterbi returns the MAP
               state path, and Baum-Welch monotonically increases log-likelihood across
               iterations; all computation in log space
  Verified by: (a) log-likelihood monotonicity test; (b) comparison against a brute-force
               enumeration of all state paths on a tiny 3-state, 5-observation case
```

```
FR-013 | HMM recovers injected states on data with known ground-truth state paths
  Priority:    MUST — THIS IS THE PROJECT-KILLER (A-002)
  Acceptance:  GIVEN generator data with known state paths WHEN Baum-Welch is fit and Viterbi
               decoded THEN adjusted Rand index between the recovered and true state sequences
               exceeds 0.5 after optimal label permutation
  Verified by: tests/test_hmm_recovery.py
  NOTE:        If this fails by Sat 29 Aug EOD, trigger kill criterion K1 and DESCEND.
```

```
FR-014 | Every flagged merchant receives a human-readable reason derived from the Viterbi path
  Priority:    MUST
  Rationale:   P-03 — this is the differentiator no other submission will have, and it addresses
               the merchant-experience wound Razorpay documented themselves
  Acceptance:  GIVEN a merchant flagged at time t THEN the system emits a string naming (a) the
               state transitioned into, (b) the date of transition, (c) the top 3 emissions by
               contribution to the transition, in merchant-facing language
  Verified by: golden-file test on fixed fixtures
  Example:     "On 14 May this account moved into a pattern we flag as rapid-volume-escalation.
                Transaction volume rose 6× above your own 90-day norm, average ticket size fell
                62%, and 94% of payers were first-time. Documents that would resolve this: ..."
```

```
FR-015 | Three-action policy: pass / review / hold
  Priority:    MUST
  Rationale:   ADR-0005
  Acceptance:  GIVEN a belief state and a merchant's cost parameters THEN the policy returns
               exactly one of {PASS, REVIEW, HOLD} and the expected cost of each option
  Verified by: unit test over the decision boundary
```

```
FR-016 | Decisions minimise example-dependent expected cost (Bayes Minimum Risk)
  Priority:    MUST
  Rationale:   Elkan (2001); Bahnsen (2015). ADR-0006
  Acceptance:  GIVEN a cost matrix and a calibrated posterior THEN the chosen action is the
               argmin of expected cost, and the savings score is computed per Bahnsen et al. (2016)
  Verified by: unit test against hand-computed values on a 2-merchant fixture
```

```
FR-017 | Global review-capacity constraint is enforced
  Priority:    MUST
  Rationale:   ADR-0005
  Acceptance:  GIVEN a budget of K analyst-hours per period THEN the total review time implied
               by the policy's REVIEW actions does not exceed K, and the binding constraint is
               reported
  Verified by: test asserting the constraint holds across the frontier
```

### Evaluation layer

```
FR-018 | Ablation table isolating every component's contribution
  Priority:    MUST
  Rationale:   AP-04 — a component whose removal changes no number is decoration
  Acceptance:  GIVEN make eval THEN results/ablations.md contains rows for: HMM on/off,
               graph features on/off, within-merchant standardisation on/off, shrinkage on/off,
               NSGA vs. grid search — each with the headline metric and its delta
  Verified by: presence and completeness of the table
```

```
FR-019 | Every headline number reported in two vocabularies
  Priority:    MUST
  Rationale:   P-09, 02-stakeholders.md
  Acceptance:  GIVEN a result THEN it appears both as an ML metric (PR-AUC, precision@K) and as
               an operational quantity (₹ saved, analyst-hours consumed, merchants held per 1000)
  Verified by: inspection of results/summary.md
```

```
FR-020 | Sensitivity analysis over the cost matrix
  Priority:    SHOULD
  Rationale:   The cost values are assumptions and the panel will say so first if we do not
  Acceptance:  GIVEN the false-positive cost varied over ±50% THEN the resulting change in
               optimal thresholds and in savings is reported as a table or figure
  Verified by: results/sensitivity.md
```

```
FR-021 | Decision layer additionally validated on BAF
  Priority:    SHOULD
  Rationale:   ADR-0007 — provenance credibility
  Acceptance:  GIVEN the BAF Base variant THEN the cost/threshold layer is run against it and
               savings are reported on its native temporal split
  Verified by: results/baf_validation.md
```

---

## 2. Non-functional requirements

Every NFR carries a number and a unit. "Robust" is an adjective waiting to be argued about at acceptance.

```
NFR-001 | HEADLINE GATE — savings improvement over the floor
  Metric:      Bahnsen savings score, Rakshak vs. static rule engine, at equal review budget
  Threshold:   ≥ 20% relative improvement
  Conditions:  Held-out test window (months 8–9), merchants unseen in training, seed 42
  Verified by: results/summary.md, regenerable by `make eval`
```

```
NFR-002 | Leakage — zero merchant overlap between splits
  Threshold:   0 merchant IDs appear in more than one split
  Verified by: tests/test_splits.py — fails the build if violated
```

```
NFR-003 | Reproducibility — identical results across runs
  Threshold:   Two runs at the same seed produce byte-identical results/*.md
  Verified by: tests/test_determinism.py
```

```
NFR-004 | Runtime — full evaluation completes quickly enough to iterate
  Threshold:   `make eval` < 15 minutes wall-clock on 8-core CPU, 16 GB RAM
  Verified by: timed CI run
```

```
NFR-005 | Compute — no GPU anywhere
  Threshold:   0 CUDA/MPS dependencies; runs on a machine with no accelerator
  Verified by: dependency audit in CI
```

```
NFR-006 | Calibration — posterior probabilities are meaningful
  Threshold:   Brier score better than an uncalibrated baseline; reliability diagram produced
  Verified by: results/calibration.md + figure
```

```
NFR-007 | Cold start — new merchants receive a usable threshold
  Threshold:   A merchant with ≥ 20 transactions receives a segment-shrunk threshold; below 20,
               the segment default applies and is logged as such
  Verified by: tests/test_shrinkage.py
```

```
NFR-008 | Licence hygiene
  Threshold:   100% of dependencies MIT / BSD / Apache-2.0
  Verified by: pip-licenses check in CI
```

```
NFR-009 | Setup — a stranger can reproduce everything
  Threshold:   `git clone && make setup && make eval` succeeds from a clean environment in
               ≤ 3 commands and ≤ 20 minutes including data generation
  Verified by: clean-container test before submission
```

---

## 3. Floor, ceiling, frozen eval

**Written before any model exists. Freezing the eval after seeing results is the most common form of self-deception in technical work.**

### Floor — the dumbest thing that works
A **static rule engine**: flag if 7-day transaction velocity exceeds 3× the merchant's trailing 90-day mean, OR refund ratio exceeds 15%, OR chargeback ratio exceeds 1%. Fixed global thresholds, no learning.
*If the sophisticated approach cannot beat this, that is a bug or a finding — not a result to hide.*

### Additional baselines
| Baseline | Isolates |
|---|---|
| Random selection at equal budget | The absolute floor |
| LightGBM on windowed aggregates, no HMM | Whether latent-state modelling earns its place (A-005) |
| BOCPD + identical cost layer | HMM vs. changepoint detection |
| Uncoupled grid search over thresholds | Whether NSGA-II earns its place (A-006) |

### Ceiling — perfect-foresight oracle
A **knapsack allocation** of the K available review-hours, computed with full hindsight knowledge of which merchants actually transitioned and when. Exactly computable on synthetic data.
**All results reported as gap-to-oracle, not as unanchored absolutes.**

### Frozen eval — version-controlled now
| Element | Value |
|---|---|
| **Split** | Temporal: train months 1–6, validate month 7, test months 8–9. AND merchant-group: no merchant ID crosses splits. Both enforced in `eval/splits.py`. |
| **Test set touched** | Exactly once, at the end. All hyperparameters and thresholds chosen on the validation window. |
| **Primary metric** | Bahnsen savings score at equal review budget |
| **Secondary** | PR-AUC, precision@K (K = review budget), Brier score, median detection lag in days, gap-to-oracle |
| **Prohibited as headline** | ROC-AUC (flatters imbalance), raw accuracy |
| **Per-typology reporting** | Mandatory, including the slow-ramp failure row |
| **Verdict rendered by** | `make eval` writing `results/summary.md`. The README includes that file; it does not restate it. |
| **Seed** | 42, set in `src/rakshak/config.py` |

---

## 4. Non-goals

Reopening any of these is a DESCEND, not a quick addition.

- Transaction-level fraud scoring — Vulcan owns it
- Onboarding-gate merchant review — Bumblebee owns it
- RTO / COD return fraud — Thirdwatch owns it; most crowded idea in the pool
- KYC deepfake or synthetic-document detection — vision models, GPU
- Chargeback evidence generation — no ground truth for "would this have won"
- Any GPU-dependent method
- Production-grade UI
- Real Razorpay API integration
- Multi-analyst routing (which analyst gets which case) — one analyst pool assumed
- Online / streaming deployment — batch evaluation only
