<!-- HEAD
FILE:     docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md
PHASE:    pre-registration (cycle 4)
UPDATED:  2026-09-01
STATUS:   sealed — committed before EVAL-LOCK-CYCLE4.json, before the cycle-4 dataset
          exists, and before a line of any rung named below is written
SUMMARY:  Fixes cycle 4 in advance: the regeneration parameters, the seeds, the one new
          rung by name, its numeric adoption gate, the pre-registered decision-policy
          A/B, and the numeric rule that conditionally opens the test split.
OPEN:     Nothing is left open by design. Two items are declared CUTTABLE with a stated
          cut order, so a schedule overrun removes work in a sequence fixed in advance
          rather than in whatever sequence flatters the result.
-->

# Pre-registration — Cycle 4

**Written and committed BEFORE `EVAL-LOCK-CYCLE4.json` is sealed, BEFORE the cycle-4
dataset is generated, and BEFORE any line of `rung8_realised_exposure.py` or
`rung9_rank_cusum.py` exists.** Its whole value is that it predates all of them. Check its
position in the git history: if it does not sit before the commit that adds either module,
it is worthless and should be read as such.

**Date:** 2026-09-01 · **Author:** lead · **Supersedes:** nothing. Adds a fourth cycle lock.
**Spec:** `project-context/12-spec-cycle4.md` · **Surveys:** `project-context/13a`, `13b`, `13c`
**Cycle-3 ladder:** tagged `cycle3-ladder-immutable` before any of this began.

---

## 1. Why this is happening

Cycle 3 recorded two failures. Investigation has established that **the first was not a
failure and the second was not the failure it was reported as.** Both findings predate this
document, are recorded in `LIMITATIONS.md` §8.7a and §8.3a, and depend on nothing below.

### 1.1 Time-to-detection was never measurable

`detection_rate_d7`, `detection_rate_d14` and `detection_rate_d30` read `0.000` for every
rung **and every floor** on the cycle-3 validation split. Drift onsets were confined by
config to days 30–240; the validation window opens on day 240.

Measured on the committed cycle-3 ground truth — 294 fraud merchants, onset min 30, median
108.5, max 217:

| metric | needs onset ≥ | qualifying | achievable by anything |
|---|---|---|---|
| `detection_rate_d7` | 233 | **0 of 294** | 0.000 |
| `detection_rate_d14` | 226 | **0 of 294** | 0.000 |
| `detection_rate_d30` | 210 | 4 of 294 | ~0.014 before the fold and censoring cuts |

An oracle alerting on every merchant on day 240 scores 0.000 on d7 and d14. Seven policies
of seven scored identically, which is what a metric looks like when it measures the calendar.

### 1.2 The savings floor-fail is an exposure-estimator defect, not a ranking failure

At the same K = 15 on validation, seed 42:

| policy | PR-AUC | ECE | precision@K | recall@K | savings |
|---|---|---|---|---|---|
| `volume_rank` | 0.2169 | 0.4866 | 0.5714 | 0.1951 | **0.6017** |
| Rung 3 | 0.8578 | 0.0077 | **0.8688** | **0.3150** | 0.4354 |

Rung 3 finds **more** fraud merchants at **higher** precision on the **same** alert budget,
is 63× better calibrated, and saves 27% less money. **No ranking-quality hypothesis can
produce that**: a ranking story must predict lower precision or lower recall, and neither
holds. The cost-sensitive literature's headline prescription — rank by expected value rather
than by probability — is *already implemented*: `eval/capacity.py::select_actions` ranks on
`0.8 · p · exposure_inr − 250`.

The difference is the exposure variable, and only that:

- Rungs are handed `exposure_inr = p_declared_monthly_gmv` (`cli.py:688`) — the figure the
  merchant **declared at onboarding**, which the generator corrupts on purpose
  (`declaration_error_sigma: 0.55`).
- `volume_rank` ranks on **observed captured GMV**, an event-stream quantity.
- `true_loss_amount_inr` = `loss_fraction × post-onset realised captured GMV`.

Measured against realised loss over the 294 fraud merchants:

| exposure estimator | Spearman ρ vs realised loss | share of total loss in top K=15 |
|---|---|---|
| `declared_monthly_gmv` — every rung | **+0.533** | **20.51%** |
| observed pre-window GMV — `volume_rank` | **+0.929** | **37.83%** |
| perfect foresight | 1.000 | 46.18% |

`volume_rank` is not a dumb floor that happens to win. It is an exposure estimator at
ρ = 0.93 against the quantity the savings metric integrates, beating rungs whose excellent
`p` is multiplied by an estimator at ρ = 0.53.

### 1.3 Disclosure of what informed the choices below

**Both findings above were derived from cycle-3 data, which this cycle replaces.** Choosing
a cycle-4 hypothesis using cycle-3 evidence is legitimate and is what a previous cycle is
for. What would not be legitimate is choosing after seeing cycle-4 results, and that is what
this document exists to prevent. Stated explicitly so no reader has to infer it:

- §1.1 was found by checking the spec's arithmetic against `data/v2/ground_truth.parquet`.
- §1.2 was found by reading `eval/capacity.py` to answer a question survey 13b raised —
  whether the decision layer ranks by probability or by expected value — and then measuring
  the single remaining hypothesis. Survey 13c reached the same mechanism independently by
  tracing `cli.py` and `engine.py`, and additionally identified the REVIEW-only comparison
  asymmetry recorded in §4.3.
- **No cycle-4 number of any kind exists at the time of writing.** The dataset has not been
  generated. Every gate below is therefore declared against results that cannot yet exist.

---

## 2. The regeneration — parameters, already committed

Committed at `52e3ae5` in `configs/scenario_v2.yaml`, before this document and before the
lock. Restated here so the pre-registration is self-contained.

| parameter | cycle 3 | cycle 4 | why |
|---|---|---|---|
| `n_merchants` | 20,000 | **40,000** | evaluation denominator, without touching prevalence |
| `n_days` | 365 | 365 (unchanged) | lengthening spreads a fixed onset count thinner |
| `prevalence` | 0.0147 | **0.0147, held** | BAF-native; raising it reopens AP-06 |
| `onset_window_max_day` | 240 | **364** | in-window onsets; this is the cycle |
| onset placement | uniform | **uniform** | stratifying into eval windows would be enrichment |
| per-typology onset bounds | — | affine `[30,240] → [30,364]` | preserves each typology's relative position and spread |
| `label_resolution_horizon_day` | (n/a) | **500** | censoring stops being an artefact of where simulation halts |
| split boundaries | 0–239 / 240–299 / 300–364 | **unchanged** | |
| seeds | 42, 43, 44, 45, 46 | **unchanged** | |
| capacity K | 100 (derived) | **200 (derived)** | the 50-per-10,000 rate is unchanged; K follows by rule |

**Expected yield, Monte Carlo over the real label pipeline, five seeds, run before the lock**
(`assign_typologies` + `emit_labels`, no transactions generated):

| quantity | mean | range over seeds |
|---|---|---|
| fraud merchants | 588 | 588 |
| trainable positives at the day-239 boundary | 233.6 | 222–247 |
| in-window resolved onsets, **validation** fold | **10.4** | **7–14** |
| in-window resolved onsets, **test** fold | **6.4** | **3–11** |
| labelled positives in test fold | 159.2 | 140–178 |

### 2.1 A correction to the spec's own yield estimate, declared before the run

`12-spec-cycle4.md` predicted "14 in-window onsets in the validation fold, 13 in the test
fold." The validation figure is right. **The test figure is not: the measured mean is 9.0
in-window onsets, 6.4 of them resolving, against a predicted 13.** Verified analytically as
well as by Monte Carlo, so it is not seed noise:

The affine rescale preserves each typology's *relative* position, so the top of the horizon
is reachable by only a few typologies. `P(onset ∈ test window)` by typology: R1 0.000,
R2 **0.000**, R3 0.108, R4 0.000, R5 0.030, R6 0.268, R7 0.100, R8 0.104, R9 **0.000** —
population weighted, 0.0528, giving 31 of 588 platform-wide and 7.8 in the 25% test fold.

**Two consequences are declared now rather than discovered later:**

1. **R2 and R9 — 25% of the fraud mix — have zero probability of onsetting in either
   evaluation window.** R2 is the slow-ramp bust-out, *the typology v1 failed on and the one
   kept in the population specifically so the failure stays visible*. Per-typology TTD is
   therefore structurally uncomputable for R2 and R9 in cycle 4, and any per-typology
   latency table must show them as **structurally absent, not as zero**.
2. **The test split's latency denominator is ~6 merchants**, carrying a standard error near
   ±20 pp. This is a reason the test-split gate in §5 is strict, and a reason a test-split
   TTD number will be reported with its denominator attached or not at all.

**This is not patched around.** Widening the per-typology windows to reach the horizon would
make R2 and R9 easier relative to their peers, which the spec explicitly rejected. The spec's
decision stands and its cost is recorded here instead.

---

## 3. The ordering claim, and how to check it

The enforced `eval_module_sha256` is expected to be **byte-identical** between cycle 3 and
cycle 4. Verified at the time of writing:

| hash | cycle 3 | cycle 4 | |
|---|---|---|---|
| `eval_module_sha256` | `c009e38d…` | `c009e38d…` | **UNCHANGED** |
| `generator_module_sha256` | `816782d0…` | `e750edbf…` | changed, as it must |
| `scenario_config_sha256` | `d64a5098…` | `683c435c…` | changed, as it must |

`src/rakshak/eval/` is not edited in this cycle. **If that hash ever differs, the claim
"we only moved the data" is broken and this document should be read as void.** No rung below
may require an eval-package edit; a candidate that needs one is the wrong candidate.

---

## 4. What is adopted, and against what numbers

### 4.1 The one new scoring rung — `rung9_rank_cusum`

**Named now, before its code exists.** Survey 13a's first-place recommendation.

Per-merchant Page/CUSUM on the within-day cross-sectional rank of the incumbent
cohort-residual score, with the existing capacity layer performing top-K selection:

```
u_t   = within-day, within-cohort rank of the incumbent Rung-3 score, in (0,1)
C_t   = min(20, max(0, C_{t-1} + Φ⁻¹(u_t) − 0.25))          # 8 bytes of state
score = σ( a·logit(incumbent) + b·C_t + c )                  # 3 params, fit on train only
```

Rationale for weighting it above the alternatives: its objective **is** detection delay
rather than classification accuracy with latency inherited as a side effect, and the rank
transform is invariant to any monotone platform-wide shock — a strictly stronger form of what
the cohort residual does additively. It consumes no labels beyond the incumbent's, needs no
autograd, adds no dependency, enters through the existing rung interface, and reaches the
capacity layer through the existing decision-policy seam.

**Adoption gate, declared now.** Rung 9 is adopted only if **both** hold on validation,
across the five locked seeds:

- **Primary (latency):** a paired McNemar test on `detection_rate_d30` discordant pairs
  against the best incumbent rung, significant at p < 0.05. Unpaired proportions are
  hopeless at n ≈ 10; the pairing is what buys any power at all.
- **Mechanism (well-powered):** median week-over-week `alert_jaccard_wow` in **[0.30, 0.85]**.
  Below 0.30 the detector is churning; at 1.000 it is `volume_rank` wearing a hat.
- **Do-no-harm:** `savings` not worse than the best incumbent by more than 0.02 absolute, and
  p99 scoring latency ≤ 10 ms per merchant.

**INSEPARABLE is pre-declared as a likely and acceptable outcome.** With ~10 evaluable
merchants a ±13 pp standard error means two rungs differing by under ~25 pp cannot be told
apart. If the McNemar test does not reach p < 0.05, **Rung 9 is reported as not adopted and
is not tuned to rescue it.** It stays in the tree as a negative result.

### 4.2 The decision-policy A/B — `realised_exposure(capacity_topk)`

Survey 13c's first-place recommendation. **This is deliberately NOT registered as a
competing rung, and the distinction is load-bearing.** §1.2 establishes that handing the
decision layer `p_declared_monthly_gmv` is a defect in harness wiring, not a modelling
choice. Correcting a defect silently, mid-comparison, would confound the geometry result
with a wiring result and make the cycle-3 → cycle-4 story unreadable.

So it is registered as a **controlled A/B, run over the entire ladder — every floor and
every rung — on identical scores**:

- **Arm A, `capacity_topk`** — `exposure_inr = p_declared_monthly_gmv`. Exactly cycle 3's
  wiring. This is the arm the cycle-3 → cycle-4 comparison is made on.
- **Arm B, `realised_exposure(capacity_topk)`** — a `DecisionPolicy` wrapper that
  `dataclasses.replace`s `DecisionRequest.exposure_inr` with trailing-30d realised captured
  GMV and forwards to the unchanged `CapacityTopK`. The exposure vector is
  `v_declared_ratio × p_declared_monthly_gmv`, which is trailing-30d captured GMV
  identically — **both are already registered, point-in-time, leakage-gated features**
  (`features/tier1.py::DeclaredRatio`, verified: `trailing-30d GMV ÷ declared_monthly_gmv`).
  No new feature, no new dependency, no training, no labels, no locked file touched.

**What is declared in advance about arm B:**

- **The prediction:** arm B raises savings for every scoring rung, and raises it more for
  rungs with better `p`, because the two terms of the expected-value product are
  complementary. **If arm B does not raise savings, §1.2's mechanism is wrong** and that is
  reported as a falsification of it rather than explained away.
- **The gate for calling the floor-fail closed:** the best rung under arm B must reach
  **savings ≥ 0.7017** — the cycle-3 `volume_rank` floor of 0.6017 plus 0.10 absolute —
  holding at ≥ 4 of 5 cost-asymmetry sweep ratios and ≥ 4 of 5 seeds. The margin is set
  above the noise a heavy-tailed loss distribution produces at this sample size, not at the
  bare floor.
- **Anti-degeneracy, and this is the important one.** Arm B must not win by turning every
  rung into `volume_rank`. Both must hold or the win is reported as degenerate:
  `alert_jaccard_wow < 0.95` and `alerts_per_day ≥ 0.9 · K`.
- **The floors are rescored under arm B too**, so `volume_rank` is beaten at the budget and
  under the wiring the rungs actually face.

### 4.3 A comparison asymmetry, disclosed now, deliberately not fixed

Survey 13c found that ranking floors are scored REVIEW-only (₹250 per error via
`savings_of_ranking(..., action=REVIEW)`) while rungs are scored on their own chosen actions,
which may HOLD at ₹8,250 per error — a 33× asymmetry. `savings_of_ranking`'s docstring claims
the comparison differs only in the score vector; that is true floor-vs-floor and **false
floor-vs-rung**. Rung 6 confirms it from the other side: it raised precision@K to 0.917 and
halved savings by rewriting HOLD → REVIEW.

**This is disclosed rather than fixed, and the reason is that fixing it requires editing the
locked eval package**, which §3 forbids and which is a deliverable of this cycle. The cycle-4
report will state, wherever a floor is compared to a rung on savings, that the floor is
priced REVIEW-only. A future cycle should re-freeze a harness in which floors and rungs are
priced identically; this cycle names the defect and leaves the hash alone.

### 4.4 Cut order, fixed in advance

If the schedule does not hold, work is removed in **this** order and no other:

1. **Rung 9 (`rank_cusum`)** — cut first. The cycle's results do not depend on it.
2. **The test-split opening** — cut second; the validation result stands on its own.
3. Everything above §4.2 is **not cuttable**. The regeneration, the full-ladder rescore and
   the arm A/B are the cycle.

---

## 5. The test split — the numeric rule, written before validation is read

The test split has been opened **zero** times. It opens **once, and only if all four of the
following hold on validation**, evaluated after the full ladder is rescored:

1. The best rung under arm B reaches **savings ≥ 0.7017** at ≥ 4 of 5 sweep ratios and
   ≥ 4 of 5 seeds (§4.2's gate).
2. That rung passes both anti-degeneracy conditions in §4.2.
3. `detection_rate_d30` is **non-zero for at least one policy** on validation — i.e. the
   geometry fix demonstrably worked, rather than being asserted from the config.
4. The lock verifies and `eval_module_sha256` still equals `c009e38d…`.

**If any of the four fails, the test split stays shut and the cycle reports validation
numbers only.** A held-out number is not worth spending the one-way door on a rung that did
not earn it.

**The conditionality is itself a finding and must be disclosed.** Wherever a test-split
number appears in the report, it carries the statement that its *existence* was contingent on
the rule above — a reader must be able to tell that the number was not guaranteed to be
produced. If the split stays shut, the report says so and says which condition failed.

`cli.py` must call **both** `require_unlocked_or_refuse(split)` and `verify_lock()` before
any scoring path, per the standing carry-forward. The primitive refuses on anything but the
literal string `"1"`.

---

## 6. Seeds

**42, 43, 44, 45, 46** — unchanged from `EVAL-LOCK.json`, fixed before any cycle-4 model
trains. Every seed in the list appears in every reported table. A rung's headline number is
the mean over all five, never a selected seed.

Cycle 3's ladder was scored single-seed (`n_seeds: 1, seeds: [42]`). With ~41 fraud merchants
in the validation fold and a heavy-tailed loss distribution, the effective sample for savings
is the count of *large-loss* frauds, plausibly 5–10. **Every four-decimal number in the
cycle-3 table, including the 0.6017 floor this cycle is measured against, is weaker than it
looks**, and the cycle-4 ladder is scored on all five seeds for exactly that reason.

---

## 7. What would falsify each claim

Stated so that each is a claim rather than a hope.

| claim | falsified by |
|---|---|
| The geometry fix makes TTD measurable | `detection_rate_d30` still 0.000 for every policy on cycle-4 validation |
| The harness did not change | `eval_module_sha256` ≠ `c009e38d…` |
| The exposure estimator explains the floor-fail | arm B does not raise savings for the scoring rungs |
| The two failures were one failure | arm A's floor-fail closes on its own, with no exposure correction |
| Rung 9 improves latency | McNemar on d30 discordant pairs does not reach p < 0.05 |

**The fourth row is the one to watch.** If in-window onsets alone close the floor-fail under
arm A, then `volume_rank` was winning because the window was stationary, and the exposure
finding — while still true as arithmetic — was not what was deciding the comparison. That is
a hypothesis this cycle can support but not establish, and the report must write it as such.

---

## 8. Clerical corrections after sealing

Recorded rather than silently applied. **Neither changes a method, a gate, a threshold or a
number**; both are naming. Anything that did change one of those would be a new
pre-registration, not a note.

**2026-09-02 — the exposure wrapper is not "Rung 8".** §4.2 above and the paragraph at the
head of this document refer to `rung8_realised_exposure.py`. That filename was wrong on two
counts. `configs/rung_roster.yaml` already assigns Rung 8 to `tpp_hawkes_nb` (T-0125, #59),
so the number was taken; and §4.2 itself registers this as a decision-policy A/B run over
the whole ladder **and explicitly not as a competing rung**, so it should never have carried
a rung number. The module is `src/rakshak/models/decision_realised_exposure.py` and the
policy names itself `realised_exposure(capacity_topk)`, which is what appears beside every
number it produces. Rung 9 is unaffected — 9 was and is free.
