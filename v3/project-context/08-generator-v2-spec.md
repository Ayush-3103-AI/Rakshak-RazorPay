<!-- HEAD
FILE:     08-generator-v2-spec.md
PHASE:    2 — SPECIFY
UPDATED:  2026-08-31
STATUS:   active
SUMMARY:  Specification for the Rakshak v2 hybrid data generator: 8 legitimate personas,
          9 risk typologies, 6 platform-level confounders, an overdispersed marked point
          process for arrivals, delayed/censored label emission, and five GREEN/RED
          parity gates that decide whether the generator is trustworthy enough to draw
          conclusions from. Implements FR-001..FR-006.
OPEN:     effect sizes in 07-feature-register.md are priors to be calibrated by G1.
-->

# 08 — Generator v2 Specification

## Why the generator comes first

Until the generator is right, every model comparison is measuring the generator. v1
evaluated at 20% prevalence with Poisson-ish arrivals, and both choices independently
distorted the results — the inflated prevalence produced the `random`-wins-on-savings
artefact (AP-06), and the under-dispersed arrivals misspecified the sequence model's
emissions. Fixing these is not preparation for the real work. It *is* the work.

## Design stance

Three properties that the v1 generator lacked, in order of importance:

1. **A confounder layer.** Platform-wide events that move everyone's features with zero
   fraud. Without these you cannot test the one thing that makes this problem hard.
2. **Hard negatives that are hard on purpose.** L3 (high-growth genuine) must be
   genuinely difficult to separate from R1 (bust-out). L8 (travel/OTA) must break naive
   refund features. A generator whose negatives are easy produces a model that fails on
   contact with reality.
3. **Ground truth that supports the metrics we actually care about.** `drift_onset_at`
   for time-to-detection, `true_loss_amount` for the oracle, `is_censored` for the
   delayed-label protocol.

---

## 1. Arrival process

Transactions per merchant are a **marked point process**. Two layers:

**Layer 1 — daily intensity.** λ(m, d) = base_λ(m) × persona_shape(d) × typology_mult(m, d)
× confounder_mult(d) × dow_factor(d).

**Layer 2 — overdispersed counts.** Daily count `n ~ NegBinomial(mean=λ, dispersion=r)`
where `r` is solved so that the realised Fano factor hits `target_fano` (default 12.25,
matching the v1 measurement). Fano for NB is `1 + λ/r`, so `r = λ / (target_fano − 1)`.

**Layer 3 — within-day times.** Given `n`, draw arrival times from the persona's
hour-of-day distribution (a 24-bin categorical), then jitter uniformly within the hour.
For typologies with bursty signatures (R3 card testing), overlay a **Hawkes-style
self-excitation**: each event raises the intensity for the next 10 minutes by
`excitation`, decaying exponentially. This is what makes `f_retry_burst_rate` and
`h_interarrival_cv` detectable — without it those features are noise.

**Marks** (per transaction): amount, instrument, payer, device, IP, status, decline
code, refund linkage. Each mark is drawn from a persona-conditioned distribution that
the typology can shift over time. See `configs/scenario_v2.yaml` for the full parameter
surface.

**Amounts** are lognormal per persona: `amount ~ LogNormal(μ_m, σ_m)`, with μ shifted by
typology and by `t_ticket_shift`. Lognormal is right because payment amounts are
multiplicative and heavy-tailed; Gaussian amounts would make `t_wasserstein_7d` trivial.

---

## 2. Legitimate personas (L1–L8)

Every merchant gets exactly one. Fraud merchants are legitimate merchants that turned —
they carry a persona for their pre-onset behaviour and a typology after `drift_onset_at`.

| ID | Persona | Signature | Population share | Why it's here |
|---|---|---|---|---|
| **L1** | Steady small retail (kirana) | low variance count, tight ticket distribution, daily rhythm | 30% | the bulk; the easy negative |
| **L2** | Seasonal D2C | large festival/sale spikes, otherwise flat | 15% | **hard negative for `v_gmv_z`** — spikes without fraud |
| **L3** | High-growth genuine | sustained upward ramp in count and GMV over 90+ days | 8% | **the hardest negative.** Looks exactly like R1 on volume features. Separable only by convexity (`v_gmv_accel`), payer diversity, and payout behaviour. |
| **L4** | Lumpy B2B | few txns, very high ticket, irregular | 10% | breaks count-based features; high `t_p95_median_ratio` legitimately |
| **L5** | Subscription/recurring | near-deterministic arrival times, low `h_interarrival_cv` | 12% | **hard negative for `h_interarrival_cv`** — scripted-looking but legitimate |
| **L6** | Marketplace sub-merchant | many payers, high new-payer ratio, aggregator patterns | 12% | **hard negative for `g_new_payer_ratio`** and adjacent to R4 laundering |
| **L7** | Dormant-then-revived | 30–60 day gap, then genuine resumption | 5% | **hard negative for `v_dormant_burst`** |
| **L8** | Travel/OTA | high refund rate, long refund latency, high ticket variance | 8% | **hard negative for the whole F6 family** |

The hard-negative annotations are the point. Each names a feature family that would
otherwise produce false positives, and each must be present in sufficient number for the
false-positive cost to be real. If L3 is 0.5% of the population, `v_gmv_accel` never
gets tested.

---

## 3. Risk typologies (R1–R9)

Each carries a `drift_onset_at`, a `difficulty` tier, and a `true_loss_amount`.

| ID | Typology | Behaviour after onset | Dominant features | Difficulty |
|---|---|---|---|---|
| **R1** | Classic bust-out | 14–21 day convex GMV ramp, payout urgency spikes, then vanish | F1, F8 | easy |
| **R2** | **Slow-ramp bust-out** | 60–90 day ramp, each week's change < 1σ of own baseline | F1 (weakly), F8 | **hard — v1 failed here** |
| **R3** | Card-testing host | micro-amount bursts, high auth-fail, high BIN diversity, Hawkes self-excitation | F5, F2, F3, F7 | easy |
| **R4** | Transaction laundering | MCC-inconsistent basket, ticket distribution shift, international share up, payer base changes | F2, F3, F9 | medium |
| **R5** | Collusive refund fraud | capture then rapid refund to a small repeat payer set | F6, F4 | medium |
| **R6** | Merchant account takeover | **abrupt** change on a long clean history; instrument mix and hour-of-day flip overnight | F3, F7 | easy-medium |
| **R7** | Mule-ring participant | shares payers/devices with other ring members; individually unremarkable | F4 (cross-merchant) | **hard — needs T3 features** |
| **R8** | Promo/coupon abuse | concentrated payers, refund-value ratio elevated, low ticket | F6, F4 | medium |
| **R9** | Prohibited-category drift | gradual basket shift away from declared MCC | F2, F9 | hard |

**R2 is the one that matters.** v1's slow-ramp adversarial typology was the documented
failure. It is a known-hard open problem, not a build defect — the research agenda for
this domain names distinguishing gradual adversarial drift from sudden natural drift as
an open need. v2 keeps R2 in the population and reports its per-typology recall
separately, so the failure (if it repeats) is visible and attributed rather than
averaged away.

**Per-typology recall is a required output.** A single aggregate recall lets R1's easy
wins hide R2's and R7's failures. `make report` breaks recall out by typology.

**Prevalence.** Merchant-level positive rate defaults to **1.47%** (BAF-native),
configurable. Typology mix within positives is configurable; default weights the easy
typologies down (R1 20%, R2 20%, R3 15%, R4 10%, R5 10%, R6 8%, R7 7%, R8 5%, R9 5%) so
the population is not dominated by the easiest case.

---

## 4. Platform confounders (P1–P6)

**This is the v2 contribution.** These events shift features platform-wide with zero
fraud, and gate G5 tests whether the detector can tell the difference.

| ID | Event | Feature effect | Shape | Default schedule |
|---|---|---|---|---|
| **P1** | Festival / sale spike | `v_gmv_z`, `v_txn_count_z` up 2–4σ for all merchants | 5-day bump | 2 occurrences |
| **P2** | Gateway outage | `f_auth_fail_rate_z` up 4σ, count down | 6-hour spike | 3 occurrences |
| **P3** | Fee/pricing change | `i_mix_jsd` — instrument mix shifts toward cheaper rail | step change, permanent | 1 occurrence, day 60 |
| **P4** | New payment method launch | `i_mix_jsd` — new instrument gains share over 30 days | S-curve | 1 occurrence, day 90 |
| **P5** | Tokenisation/regulatory mandate | `i_bin_hhi`, `i_cnp_share` shift; brief failure elevation | step + 10-day transient | 1 occurrence, day 120 |
| **P6** | Macro seasonality | slow sinusoidal modulation of all volume | continuous | always on |

Confounders apply a **multiplicative modifier to the intensity and mark distributions of
every merchant**, including fraud merchants. They must be implemented as a separate
layer in `confounders.py` that the persona and typology layers do not know about — that
separation is what makes the `prevalence=0, confounders=on` null test meaningful.

Cohort heterogeneity matters: P1 (festival) should hit L2 (seasonal D2C) harder than L4
(B2B). Confounder effects are therefore `base_effect × persona_sensitivity[persona_id]`.
Without this, the cohort residual has an unfairly easy job and G5 passes for the wrong
reason.

---

## 5. Settlement and payout layer

Often skipped; do not skip it. The F8 family and `s_payout_freq_z` are among the
strongest bust-out signals and they are untestable without it.

Model: captured funds accrue to a merchant balance with a T+2 default settlement cycle.
Merchants request payouts at a persona-specific frequency. R1 and R2 raise
`payout_urgency` after onset — more frequent requests, larger drawdown fraction,
occasionally requesting accelerated settlement. Emit a `payout` event table joined on
`merchant_id`.

---

## 6. Label emission

```python
label_event_at   = drift_onset_at + Exponential(mean=fraud_to_dispute_days)
label_available_at = label_event_at + Uniform(45, 120) days
```

Four label states, all required:

| State | Condition | Row |
|---|---|---|
| Observed positive | fraud, disputed, within horizon | `label=1`, `is_censored=false` |
| **Unreported positive** | fraud, but `unreported_rate` (default 15%) fires | `label=0`, `is_censored=false`, `true_typology` set in ground_truth |
| Censored | `label_available_at > simulation_end` | `label=NULL`, `is_censored=true` |
| **False positive label** | legitimate merchant, `spurious_chargeback_rate` (default 0.3%) | `label=1` on a good merchant |

Unreported positives and spurious labels are what make this a realistic weak-supervision
problem rather than a clean classification task. A model that assumes labels are correct
will overfit to label noise, and the harness should be able to show that.

---

## 7. Parity gates — GREEN/RED

All five must be GREEN before any model trains. `make gates` runs them.

| Gate | Test | GREEN condition | If RED |
|---|---|---|---|
| **G1** | Marginal parity vs BAF | For each shared feature analogue, two-sample KS statistic ≤ 0.15 against the BAF marginal; realised Fano = target ± 1.0 | Recalibrate persona parameters. One attempt, then charter K-3. |
| **G2** | Baseline transfer | LightGBM trained on generator → scored on BAF achieves PR-AUC ≥ 0.5 × its in-domain PR-AUC, and vice versa | The generator is fiction. Charter K-3 fires. |
| **G3** | Determinism | Two clean runs at the same seed produce identical output SHA256 | Hunt the unseeded RNG. Blocking; nothing proceeds. |
| **G4** | No leakage | AST scan finds no forbidden symbol in `features/` or `models/`; point-in-time recomputation at time t matches the stored feature vector exactly | Fix immediately. Any leakage invalidates every number. |
| **G5** | **Confounder null** | With `prevalence=0` and confounders on, the trained detector's alert rate stays ≤ the nominal FPR + 2 percentage points across all six confounder windows | The system cannot distinguish platform drift from fraud. Charter K-4 fires — report as the central negative finding. |

**G5 is the gate worth building the demo around.** It is a direct, visual test of the
one property that distinguishes a merchant sentinel from a transaction scorer, and it
produces a plot — alert rate over time with confounder windows shaded, raw-feature model
spiking and cohort-residual model flat — that a Head of Risk Ops understands in three
seconds. If only one figure survives into the video, make it this one.

---

## 8. Config surface

`configs/scenario_v2.yaml` — everything below is a named, commented parameter. No magic
numbers in `src/`.

```yaml
seed: 42
population:
  n_merchants: 10000
  n_days: 180
  prevalence: 0.0147          # BAF-native. AP-06: NEVER evaluate at 0.20.
arrivals:
  target_fano: 12.25          # measured in v1; G1 calibrates against this
  hawkes_excitation: 0.35     # only active for bursty typologies
personas:                     # shares must sum to 1.0
  L1: 0.30
  L2: 0.15
  L3: 0.08                    # hard negative — do not shrink below 0.05
  L4: 0.10
  L5: 0.12
  L6: 0.12
  L7: 0.05
  L8: 0.08
typology_mix: {R1: 0.20, R2: 0.20, R3: 0.15, R4: 0.10, R5: 0.10,
               R6: 0.08, R7: 0.07, R8: 0.05, R9: 0.05}
confounders:
  enabled: true
  P1_festival: {count: 2, magnitude_sigma: 3.0, duration_days: 5}
  P2_outage:   {count: 3, magnitude_sigma: 4.0, duration_hours: 6}
  P3_fee_change:   {day: 60}
  P4_new_method:   {day: 90, ramp_days: 30}
  P5_regulatory:   {day: 120, transient_days: 10}
  P6_macro:        {amplitude: 0.15, period_days: 90}
labels:
  dispute_delay_days: [45, 120]
  fraud_to_dispute_mean_days: 21
  unreported_rate: 0.15
  spurious_chargeback_rate: 0.003
settlement:
  cycle_days: 2
capacity:
  analyst_reviews_per_day: 50   # K. Load-bearing — see charter §10.4
costs:
  fraud_loss_multiplier: 1.0    # of true_loss_amount
  false_hold_cost_inr: 8000     # merchant churn + support
  review_cost_inr: 250          # analyst time
```

The cost block deserves a note. v1 measured cost asymmetry at 47.5 / 13.1 / 61,368
against a literature band of 400–600 — a three-orders-of-magnitude spread that says the
asymmetry cannot be assumed, only measured per deployment. Treat these three numbers as
**swept parameters**, not constants: `make report` should show how the rung ranking
changes across an asymmetry sweep. A ranking that is stable across the sweep is a much
stronger claim than one tuned to a single guessed ratio.
