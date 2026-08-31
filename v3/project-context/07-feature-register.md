<!-- HEAD
FILE:     07-feature-register.md
PHASE:    2 — SPECIFY
UPDATED:  2026-08-31
STATUS:   active
SUMMARY:  The catalogue of drift features. Each entry names the fraud mechanism it
          detects, the persona/typology it fires on, the generator parameter that
          controls it, its tier (T1 cheap/always, T2 conditional, T3 deferred), its
          online state cost, and its leakage risk. This file IS the contract between
          08-generator-v2-spec.md and src/rakshak/features/registry.py — a feature
          here that the generator does not produce is untestable, and vice versa.
OPEN:     T3 sketch features are P2 and may be cut entirely.
-->

# 07 — Feature Register

## On "weightage"

Do not hand-assign importance weights. Weights are learned by the model; hand-weighting
before training is how you smuggle your hypothesis into the result and then discover it
in the ablation. What this register carries instead is a **tier**, which is a decision
about *compute*, not about *importance*:

- **T1** — computed for every merchant, every epoch, in Stage 0. ~14 features. Must be
  O(1) update with < 200 bytes of state each.
- **T2** — computed only for merchants that clear the Stage-0 screen (top 10%). May
  hold a small histogram or ring buffer. Up to ~1.5 KB of state each.
- **T3** — deferred / P2. Sketch-based graph features. Cut first.

The **prior effect size** column is different: that is an input to the *generator*, not
to the model. It says how strongly the generator should make each typology move each
feature. It is the thing that makes the synthetic problem neither trivial nor
impossible, and it is a config value that gets calibrated, not a belief.

## Two rules that gate entry into this register

1. **Self-referential, not absolute.** Every drift feature is expressed as a z-score
   against the merchant's *own* post-onboarding baseline (with empirical-Bayes shrinkage
   toward the cohort prior during cold start). Absolute levels are merchant-specific and
   will not generalise. (FR-011)
2. **Online-computable or out.** If it cannot be maintained incrementally from a bounded
   `MerchantState`, it does not go in. This kills several attractive features and that
   is the point — the data arrives as a stream. (FR-010, Prime Directive 4)

---

## The cohort-residual layer — read this before the table

Every T1 feature below gets a companion:

```
r_f(m, t) = z_f(m, t) − median_{m' ∈ cohort(m), m' ≠ m} z_f(m', t)
```

This is the sprint's hypothesis and the thing charter K-1 kills if it fails.

**Why it matters.** The named open problem in this domain is separating the fraudster's
gradual adversarial drift from the platform's sudden natural drift. When a festival
spikes GMV platform-wide, every merchant's `z_gmv` rises together and every residual
stays near zero. When one merchant ramps alone, its residual explodes. The confounder
layer P1–P6 in the generator exists to test exactly this, and gate G5 is its verdict.

**Cost.** Cohort statistics are computed once per epoch in a single pass over all
merchants — O(N log N) for the medians once per day, then O(1) per merchant. This is
cheap because the epoch is daily. Leave-one-out is computed by the standard
median-excluding-self trick on the sorted cohort array, not by recomputing N times.

**Cohort definition.** `(mcc_group, gmv_decile, vintage_bucket)`, backing off to
`mcc_group` then global at membership < 30. (FR-013)

---

## F1 — Volume and velocity drift

| ID | Feature | Mechanism detected | Fires on | Generator knob | Prior effect | Tier | State |
|---|---|---|---|---|---|---|---|
| `v_txn_count_z` | daily txn count, z vs own baseline | bust-out ramp | R1, R2, R3 | `ramp_rate` | large | T1 | 24 B (Welford) |
| `v_gmv_z` | daily GMV, z vs own baseline | bust-out ramp | R1, R2 | `ramp_rate` | large | T1 | 24 B |
| `v_gmv_accel` | 2nd difference of 7d GMV | *accelerating* ramp vs organic growth | R1 vs L3 | `ramp_convexity` | medium | T1 | 32 B |
| `v_declared_ratio` | trailing-30d GMV ÷ onboarding-declared monthly GMV | **promise vs reality gap** | R1, R2, R4 | `declared_gmv_error` | large | T1 | 16 B |
| `v_fano_trailing` | Fano factor of daily counts, 28d | burstiness change | R3, R5 | `fano_shift` | medium | T1 | 40 B |
| `v_dormant_burst` | days-since-last-txn × today's count z | dormancy then spike | R1, R6 | `dormancy_days` | medium | T1 | 16 B |

`v_declared_ratio` deserves a note: it is only available **because** the merchant was
onboarded. A transaction scorer never sees it. It is the clearest example of a signal
that exists exclusively in the post-onboarding surveillance position, and it should be
named as such in the writeup.

`v_gmv_accel` is the discriminator against persona L3 (high-growth genuine), which is
the hardest negative in the population. Organic growth is roughly linear-to-log;
bust-out is convex.

---

## F2 — Ticket-size distribution drift

| ID | Feature | Mechanism | Fires on | Generator knob | Prior effect | Tier | State |
|---|---|---|---|---|---|---|---|
| `t_wasserstein_7d` | 1-Wasserstein between trailing-7d ticket distribution and baseline | any distributional shift | R1, R2, R4 | `ticket_shift` | large | T2 | 1.0 KB (32-bin log histogram) |
| `t_p95_median_ratio` | p95 ÷ median ticket | tail growth | R1, R4 | `ticket_tail` | medium | T1 | 64 B (P² estimators) |
| `t_round_amount_share` | share of txns at round values (100/500/1000/5000) | card testing, laundering | R3, R4 | `round_amount_rate` | medium | T1 | 16 B |
| `t_micro_share` | share of txns ≤ ₹10 | card-testing probes | R3 | `probe_rate` | large | T1 | 16 B |
| `t_new_max_event` | 1 if today's max > 3× historical max | single large exfil | R1, R5 | `exfil_multiple` | medium | T1 | 8 B |

Use a **fixed 32-bin log-spaced histogram** for `t_wasserstein_7d`, not a stored sample.
The histogram is O(1) to update, bounded in memory, and Wasserstein on binned CDFs is a
single vectorised pass. A reservoir sample would blow NFR-04.

---

## F3 — Payment-instrument mix drift

| ID | Feature | Mechanism | Fires on | Generator knob | Prior effect | Tier | State |
|---|---|---|---|---|---|---|---|
| `i_mix_jsd` | Jensen-Shannon divergence of 7d instrument mix vs baseline mix | channel shift | R3, R4, R6 | `mix_shift` | large | T2 | 96 B (7-way counters) |
| `i_intl_share` | share of international cards | cross-border exfil, laundering | R4, R7 | `intl_rate` | medium | T1 | 16 B |
| `i_cnp_share` | card-not-present share | testing, ATO | R3, R6 | `cnp_shift` | medium | T1 | 16 B |
| `i_bin_hhi` | Herfindahl over issuer BINs, 7d | concentrated stolen-card source | R3, R7 | `bin_concentration` | medium | T2 | 512 B (top-k counter) |

`i_mix_jsd` is the feature most at risk of firing on confounder P3 (fee change) and P4
(new payment method launch). That is intentional — it is the best test case for whether
the cohort residual works, because the raw feature *should* fire platform-wide and the
residual *should not*.

---

## F4 — Counterparty / payer-graph drift

Graph signal, delivered as scalars. The v1 ablation found these load-bearing for
sequence models (−0.1006 savings on removal) and near-irrelevant to LightGBM (+0.0047) —
a genuine finding worth re-testing under the corrected generator.

| ID | Feature | Mechanism | Fires on | Generator knob | Prior effect | Tier | State |
|---|---|---|---|---|---|---|---|
| `g_unique_payers_z` | distinct payers/day, z | volume authenticity | R1, R5 | `payer_pool_size` | medium | T1 | 24 B + HLL |
| `g_new_payer_ratio` | share of payers never seen before | churned/synthetic payer base | R1, R3 | `new_payer_rate` | large | T1 | 256 B (HLL, 12-bit) |
| `g_payer_hhi` | Herfindahl over payer IDs, 28d | collusive self-churn | R5, R8 | `collusion_concentration` | large | T2 | 512 B (top-k) |
| `g_shared_payer_count` | payers this merchant shares with ≥3 other flagged merchants | mule ring | R7 | `ring_size` | large | **T3** | sketch, cross-merchant |
| `g_device_reuse_rate` | distinct payers per distinct device hash | one actor, many "payers" | R3, R5, R7 | `device_reuse` | large | T2 | 256 B (HLL pair) |

`g_shared_payer_count` is the only feature requiring cross-merchant state at update
time. It is T3 and P2 for exactly that reason. If it is cut, say so in LIMITATIONS —
it is the feature a real deployment would most want.

Use **HyperLogLog** for cardinality (`g_unique_payers_z`, `g_new_payer_ratio`) — exact
distinct-counting over a 28-day window blows NFR-04. Accept ~2% relative error; it is
far below the effect sizes being detected. Record the choice in the LOGBOOK.

---

## F5 — Failure and retry signature

| ID | Feature | Mechanism | Fires on | Generator knob | Prior effect | Tier | State |
|---|---|---|---|---|---|---|---|
| `f_auth_fail_rate_z` | auth failure rate, z vs baseline | card testing | R3 | `fail_rate_shift` | large | T1 | 24 B |
| `f_retry_burst_rate` | rate of (same payer, ≥3 attempts, ≤10 min) | enumeration | R3 | `retry_burst` | large | T1 | 64 B (short ring) |
| `f_decline_entropy` | Shannon entropy of decline codes, 7d | broad card-source testing | R3, R7 | `decline_spread` | medium | T2 | 192 B |

`f_auth_fail_rate_z` is the feature that confounder **P2 (gateway outage)** will slam
platform-wide. It is the cleanest single demonstration of the cohort-residual idea:
during P2, every merchant's `f_auth_fail_rate_z` spikes and every residual stays flat.
Build the G5 test around this feature first.

---

## F6 — Refund and dispute precursors

| ID | Feature | Mechanism | Fires on | Generator knob | Prior effect | Tier | State |
|---|---|---|---|---|---|---|---|
| `d_refund_rate_z` | refund count ÷ txn count, z | refund fraud, buyer collusion | R5, R8 | `refund_shift` | large | T1 | 24 B |
| `d_refund_latency_med` | median hours from capture to refund | instant-refund laundering | R5 | `refund_latency` | medium | T2 | 64 B (P²) |
| `d_refund_amount_ratio` | refunded ₹ ÷ captured ₹, 7d | value siphon | R5, R8 | `refund_value_ratio` | large | T1 | 24 B |

**Leakage warning.** Do not build features from chargebacks or disputes. A chargeback is
the label, delayed. Any chargeback-derived feature at decision time t is either
unavailable (correct) or leakage (fatal). Refunds are merchant-initiated and observable
immediately — those are fine. This distinction must be enforced in the schema: refunds
live in the transaction stream, chargebacks live in the label table.

Persona **L8 (travel/OTA)** exists specifically as the hard negative for this family —
legitimately high refund rate and high refund latency.

---

## F7 — Temporal pattern drift

| ID | Feature | Mechanism | Fires on | Generator knob | Prior effect | Tier | State |
|---|---|---|---|---|---|---|---|
| `h_hourly_jsd` | JS divergence of hour-of-day histogram vs baseline | operator change, automation | R3, R6, R7 | `hour_shift` | medium | T2 | 192 B (24 counters) |
| `h_interarrival_cv` | coefficient of variation of inter-arrival times, 7d | scripted vs human traffic | R3 | `arrival_regularity` | large | T1 | 32 B |
| `h_weekend_share_z` | weekend share of GMV, z | business-pattern change | R4, R6 | `weekend_shift` | small | T1 | 24 B |

`h_interarrival_cv` is the one feature that directly exploits the continuous-time
structure the v1 fixed-window design threw away. It is cheap and it should be in T1.

---

## F8 — Settlement and payout behaviour

| ID | Feature | Mechanism | Fires on | Generator knob | Prior effect | Tier | State |
|---|---|---|---|---|---|---|---|
| `s_payout_freq_z` | payout requests/week, z | cash-out urgency before bust-out | R1, R2 | `payout_urgency` | large | T1 | 24 B |
| `s_balance_drawdown` | share of available balance withdrawn per payout | exfil behaviour | R1, R5 | `drawdown_rate` | medium | T1 | 16 B |

Payout urgency is one of the strongest real-world bust-out tells and it costs almost
nothing. If the generator does not model payouts, this family is untestable — make sure
`08-generator-v2-spec.md` §Settlement is implemented, not skipped.

---

## F9 — Static profile and mismatch

Not drift features; they modulate the others and enter the model directly.

| ID | Feature | Note | Tier |
|---|---|---|---|
| `p_mcc_group` | categorical, one-hot | T1 |
| `p_days_since_onboarding` | risk is non-monotonic in this — bust-out has a characteristic latency; let the tree find it | T1 |
| `p_kyc_tier` | ordinal | T1 |
| `p_vintage_months` | business age at onboarding | T1 |
| `p_declared_monthly_gmv` | the denominator of `v_declared_ratio` | T1 |
| `p_city_tier` | ordinal | T1 |

**Forbidden as features, permanently:** `persona_id`, `risk_typology_id`,
`drift_onset_at`, `true_loss_amount`, anything from the `ground_truth` table, and any
chargeback-derived quantity. Enforced by AST scan in
`tests/gates/test_no_ground_truth_import.py`.

---

## Registry summary and budget check

| Tier | Count (base) | + cohort residual | State/merchant | Stage |
|---|---|---|---|---|
| T1 | 23 | +14 (residuals on the drift subset) | ~0.9 KB | 0 — all merchants |
| T2 | 9 | +4 | ~2.9 KB | 1 — top 10% |
| T3 | 1 | — | cross-merchant sketch | deferred, P2 |

T1 + T2 state ≈ 3.8 KB, inside the 4 KB NFR-04 budget with ~5% headroom. If it goes
over, cut `i_bin_hhi` (512 B) first — it is the most redundant with `g_new_payer_ratio`.

**Ablation plan** (Rung 3 vs Rung 2 is the headline, but run these too):
1. Full T1 vs T1-minus-cohort-residuals — the K-1 test.
2. Leave-one-family-out across F1–F8 — which family is load-bearing.
3. T1-only vs T1+T2 — does the expensive tier earn its compute.
4. Re-run v1's graph-feature ablation under the corrected generator — does the
   +0.0047 / −0.1006 asymmetry survive? That is a publishable answer either way.
