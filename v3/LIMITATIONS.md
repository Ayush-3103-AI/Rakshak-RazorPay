# LIMITATIONS — Rakshak v2

> Honest failures, with numbers. Prime Directive 6: *a rung that loses is a finding, not an
> embarrassment.* This file accumulates through the sprint and is finalised at T-152. Every
> failed rung, every fired kill criterion, and every cut feature is named here with its
> number and its reason.
>
> Nothing in this file is here because it was too awkward to fix elsewhere. Each entry names
> what was cut, why, and what it would take to un-cut it.

**Status:** accumulating · last updated 2026-08-31, after Lane D (T-140…T-143). §8 holds the rung results.

---

## 1. Features specified in the register but not built

Eight of the register's rows are not in `registry.ORDER`. The register is the contract
between the generator spec and the feature layer, so a row that is not built is a gap in
that contract and is named here rather than quietly dropped.

| Feature | Family | Why it was cut | What would un-cut it |
|---|---|---|---|
| `s_payout_freq_z` | F8 | **Structural.** `FeatureSpec.update(state, event)` takes a `Transaction`, and `assert_parity(spec, txns, profiles)` has no payout channel. There is no path for a `Payout` to reach a feature at all. | `update(state, event: Transaction \| Payout)` and a `payouts=()` argument on the parity harness — an amendment to the frozen `09-interfaces.md` §7. Declined 2026-08-31 to keep the Block-1 freeze intact. |
| `s_balance_drawdown` | F8 | Same. | Same. |
| `d_refund_latency_med` | F6 | **Rule 2 — online-computable or out.** The median latency between a capture and its refund requires the capture timestamp of an arbitrary past event, and the set of still-open captures is unbounded. It cannot be maintained in bounded state. | A `latency` (or original-capture-time) field carried on the refund event itself, in `schemas.py`. This is a permanent cut under the current schema, not a deferral. |
| `i_bin_hhi` | F3 | State budget. | The shared daily ring below. |
| `g_payer_hhi` | F4 | State budget. | The shared daily ring below. |
| `g_device_reuse_rate` | F4 | State budget. | The shared daily ring below. |
| `g_unique_payers_z` | F4 (T1) | **Ticket gap, not a technical cut.** T-120's scope names F1/F2/F5/F6/F7/F8/F9; T-122's names F2/F3/F4 *T2 rows only*. These two T1 rows fall between the two tickets and are in no ticket's scope. | A ticket. Flagged so the omission is not mistaken for a judgement about the feature. |
| `g_new_payer_ratio` | F4 (T1) | Same. | Same. |

**The shared daily ring.** Six T1 features keep private ring buffers over substantially the
same per-merchant daily series. One shared ring on `MerchantState` would return roughly 1 KB
and make the three budget-cut features affordable. It needs a slot in `state.py`, which
`09-interfaces.md` §8 freezes. Declined 2026-08-31 for the same reason as F8: reopening the
Block-1 contract mid-sprint costs more than the three features are worth, and the freeze is
itself a claim the project makes.

---

## 2. Failed and unmet acceptance clauses

### T-120 — the T1 state bar is not met
The ticket requires *total declared `state_bytes` for T1 < 1024*. **Measured: 1720 B across
24 T1 features.** The per-feature byte figures in `07-feature-register.md` are not physically
achievable as written — a trailing-30-day sum cannot be maintained in 16 B. The register's
budget column is an estimate that was never costed against an implementation.

Total declared state across all 28 features is **3968 B against NFR-04's 4096 B**, so the
*declared* budget is met; it is the T1-only sub-budget that is not. **The measured budget is
not met either — see the NFR-04 entry below**, which supersedes the reading that the binding
budget was satisfied. The declarations are 2.4× under what the same features actually
serialize to.

### NFR-04 — the state budget is exceeded by 2.35×, and it is not a serialization problem
*Resolved at T-150. The earlier entry here diagnosed this as pickle framing and carried it
forward; the diagnosis was right about the cause and wrong about the size of it.*

| | bytes | |
|---|---|---|
| pickled, fully warmed | **13,145** | not the 7,091 B recorded at T-120 — that state was measured before its trailing windows had filled |
| packed (`features/state.py`) | **9,634** | 27% smaller; a real serialization, `unpack(pack(x)) == x` asserted |
| zero-framing float64 floor | **9,734** | every scalar at its natural width, no tags, no lengths, no class names |
| NFR-04 budget | **4,096** | |
| declared across 28 features | 3,968 | the declarations were never costed against an implementation |

T-150 built the packed representation the previous entry asked for: one tagged
little-endian buffer, float lists written as contiguous arrays, and count arrays narrowed
losslessly to uint32. It removed the framing. **The budget is still exceeded by 2.35×, and
the packed form is already smaller than the zero-framing floor of the same values** —
narrowing histogram counts buys more than the tag bytes cost. There is no serialization
left to do: 9.6 KB is the information content of what the 28 registered features hold, at
the float64 precision NFR-08's 1e-9 parity requires.

So this is a feature problem after all. The levers, in order of size:

- **The two T2 histogram features are 4.1 KB of the 9.6 KB.** `t_wasserstein_7d` (32 bins)
  and `h_hourly_jsd` (24 bins) each hold a frozen baseline plus a week of completed daily
  histograms. A shared daily histogram across the four T2 features, or a shorter T2 window,
  is the single biggest change available and it is a change to `tier2.py`.
- The five windowed T1 features hold 20–30 completed days of `(date, num, den, aux)`.
  Packing those column-wise instead of per-tuple is worth roughly 1.5 KB more and still does
  not reach 4,096 B, which is why it was not built.
- Cutting features. NFR-04 exists to force exactly that trade, and this is the trade.

**The budget was not raised.** `tests/perf/test_state_size.py` keeps the 4,096 B assertion
live as `xfail(strict=True)` — the same treatment T-121's unreachable clause got — so that
if anyone ever makes it pass, the strict xfail fails and the finding has to be rewritten.
Beside it, a live fence at 10,240 B fails CI if the state grows further, because a real
budget sitting in xfail can no longer catch a regression.

One consequence worth naming: the packed codec is pure Python and roughly 6× slower than
pickle in both directions (pack 1.50 ms, unpack 0.85 ms, against 0.23 / 0.14). At 10,000
merchants that is the largest single line in the NFR-03 sweep — and NFR-03 still passes with
a 1.3× margin, so it was left alone rather than optimised.

### T-121 — the original clause was unreachable, and is preserved as a strict xfail
The ticket asked for `mean |residual| < 0.25` under confounder P2 at prevalence=0. Measured
against real generator output at both 3,000 and 10,000 merchants: mean |raw z| = 3.301, mean
|residual| = 3.117.

The null day is what settles it. **On a day with no confounder at all, mean |residual| was
already 0.558.** A z-score has unit scale by definition and E|N(0,1)| = 0.798, so subtracting
a cohort median cannot bring the mean absolute value below roughly 0.6 unless every merchant
within a cohort agrees to within 0.3σ, which no per-merchant z-score does. **A perfect
residual layer fails this clause and so does a useless one.**

The clause was amended on the board (dated, with the arithmetic) to a three-part mechanism
check. The original assertion is kept verbatim in `tests/unit/test_cohort.py` as
`xfail(strict=True)` rather than deleted. It was not weakened to go green.

**This is not a K-1 falsification.** K-1's verdict is rendered at T-142 on the Rung 3 vs
Rung 2 validation delta. Reporting a falsification caused by a mis-specified metric would be
a false claim about the hypothesis.

---

## 3. What the cohort-residual layer does and does not do

Measured, prevalence=0, confounder P2 (gateway outage) active, feature
`f_auth_fail_rate_z`, at 3,000 and 10,000 merchants:

| | mean \|z\| | mean \|residual\| | median z | median residual | alert rate at \|·\|>3 |
|---|---|---|---|---|---|
| **P2 day** | 3.301 | 3.117 | +1.790 | +0.005 | 0.390 → 0.231 |
| **null day** | 0.691 | 0.558 | −0.378 | −0.000 | 0.024 → 0.028 |

**It removes the common mode exactly.** Median z of +1.790 becomes a median residual of
+0.005. That is the mechanism working precisely as designed.

**It cuts confounder-driven alerts materially but not to zero** — 39.0% to 23.1% of the
population at a |·| > 3 threshold. A 41% relative reduction, and still a G5 catastrophe on
this feature alone.

**It cannot remove P2's per-merchant heterogeneity.** The generator scales each confounder
per merchant by `sqrt(target_fano / λ)` times persona sensitivity, so a six-hour outage hits
a five-transaction-a-day merchant far harder than a five-hundred-a-day one. A median removes
only what is shared, and most of P2's effect on this feature is not shared. Cohorting on
`gmv_decile` is the lever that would make more of it shared; at 3,000 merchants the backoff
chain rarely reaches the full key, and at 10,000 the improvement was marginal (mean
|residual| 3.135 vs 3.117).

**This is the honest framing for the writeup:** the residual layer is a partial defence
against platform-wide confounders, not a solution to them. Claiming otherwise would not
survive the first question from a risk operations panel.

---

## 4. Accepted deviations from spec

### `EVAL-LOCK.json` enforces one hash of three (eval-harness-spec §6)
§6 requires all three recorded hashes to hard-fail on mismatch. The harness was frozen while
the generator was still in flight — which `STATE.md` calls *preferable*, since a harness
frozen before the generator is harder to accuse of hindsight. Enforcing the generator and
config hashes would have hard-failed on the generator's next commit, someone would have added
`--force`, and within a day it would have been reflex. **A lock that is routinely overridden
is worse than no lock, because it still looks like evidence.**

`eval_module_sha256` is enforced — that is the actual claim. The other two are recorded as
freeze-time provenance and reported as drift, never silently. `verify_lock(strict=True)`
promotes all three once the generator freezes. The reasoning is written into the lock file's
own `enforcement_note`, readable by a reviewer who opens nothing else.

### `RungOutput`, `Truth` and `CostParams` live in `metrics.py`, not `schemas.py`
Against the "all types in `schemas.py`" convention. `schemas.py` froze at the end of Block 1
and a DESCEND to add three intra-package types would have cost more than it bought. Nothing
outside `rakshak.eval` constructs them; if that stops being true, they move.

### `07-feature-register.md`'s own counts are internally inconsistent
The document states T1 = 23 and T2 = 9. Its own tables list 28 and 8. The "+14 residuals"
figure is likewise stale — 21 features now carry `has_cohort_residual`. The register was not
corrected during the sprint because it is an input document and editing it mid-flight would
have invalidated the diff against the implementation that is its purpose.

---

## 5. The external anchor: what it actually supports, and what it cannot

**Updated 2026-08-31 by T-116b.** BAF now runs. Reproduce with:

```bash
RAKSHAK_BAF_PATH=../data/external/baf.zip make gates
```

The dataset is **not vendored and must not be** — CC BY-NC-SA 4.0, non-commercial and
share-alike. It is git-ignored, it is not in this tree, and `make gates` was re-verified to
pass on a machine without it: with `RAKSHAK_BAF_PATH` unset, and again with it pointed at a
path that does not exist, G1b/G1c/G1d/G2 all record `SKIP` with their reason and the suite
exits 0.

### 5.1 What BAF is, stated before any number is quoted

BAF is **bank account-opening applications**. One row per application: no amount, no
timestamp, no payer, no merchant, no sequences. `data/external/baf.manifest.json` says so in
the project's own words, and v1's ADR-0007 concluded it informs **none** of the generator's
marginals. Rakshak is post-onboarding merchant behaviour, thousands of rows per merchant.

**BAF is also itself synthetic** — Jesus et al. fit a CTGAN to a real anonymised application
dataset under differential privacy. What it carries is a *real label distribution with real
temporal drift*, which is genuine and valuable; it is not raw observation, and the
fingerprints are visible in the data (`velocity_6h` takes negative values; the `velocity_*`
columns are non-integer despite being described as counts).

So `08-generator-v2-spec.md` §7's premise for G1 and G2 — "each shared feature analogue" —
assumes a shared feature space that **largely does not exist**. That is the finding this
section exists to record. It was not worked around.

### 5.2 The gate that could not fail

**G1b as it stood before T-116b was vacuous.** It rank-normalised both marginals before the
KS test. Mapping a sample onto its own ranks makes its empirical CDF uniform *by
construction*, so the KS between two rank-normalised samples is ~0 regardless of what they
are. Measured at a million draws each:

> `KS(rank(Normal(0,1)), rank(Exponential(1))) = 0.0` — exactly zero.

G1b would have printed GREEN against any dataset, including one of pure noise. **Had the
anchor been vendored without this being caught, the project's single external claim would
have been a number that could not move.** It is replaced by a robust standardisation
(centre on the median, scale by the IQR): units are removed — they genuinely do not
correspond — while skew and tail, the properties G1 is meant to be asking about, survive.

### 5.3 G1b — RED. Two analogues of three, and four columns that are not analogues at all

Gate population, 1,200 merchants x 180 days, seed 20260831, against all 1,000,000 BAF rows:

| analogue | Rakshak side | BAF column | KS | ceiling | verdict |
|---|---|---|---|---|---|
| `count_28d` | txns per merchant per 28d | `zip_count_4w` | **0.0922** | 0.15 | GREEN |
| `count_56d` | txns per merchant per 56d | `bank_branch_count_8w` | **0.2679** | 0.15 | **RED** |
| `cross_border` | `is_international` | `foreign_request` | **0.0056** | 0.15 | GREEN |

**G1b is RED on its worst analogue and that is the verdict recorded.** The pairing rule was
fixed before the numbers were seen — *count of events at one entity over one matched
window* — and `count_56d` was kept precisely because dropping the analogue that disagrees
would leave the gate resting on the one window that happened to agree.

The RED is not a transform artefact: on a log scale it is 0.1531, still above the ceiling.
But it is weaker evidence against the generator than it looks, and the reason is legible in
the quantiles. BAF's applications-per-bank-branch runs q10/q50/q90 = **0 / 9 / 750**; our
transactions-per-merchant runs **102 / 406 / 790**. Bank branches are vastly more
heterogeneous in size than a merchant population is — an 83x median-to-p90 spread against
our 1.9x. `count_28d`'s entity is a zip code, whose size distribution is far closer to a
merchant population's, and it passes comfortably. **The honest reading is that `count_56d`
measures the difference between a branch population and a merchant population at least as
much as it measures the generator, and this file does not claim to separate the two.**
Charter K-3's trigger is G2, not G1b; G1b RED calls for one recalibration attempt, and no
recalibration of persona parameters would change a fact about bank branches.

**Four columns the spec's "shared feature analogue" implies and the data refuses.** They are
printed by the gate with their reasons rather than dropped, because a quietly shortened list
is how a weak anchor comes to look strong:

- `proposed_credit_limit` — 12 distinct values on a bank's offer grid (190, 200, 210, ... 2100).
  A limit the bank offered, not money that moved. Our `amount_inr` is continuous over four
  orders of magnitude. One side is a menu.
- `device_distinct_emails_8w` — four distinct values, 96.8% of a million rows exactly `1`,
  and `-1` is a missing sentinel, not a count. Fano **0.032**: underdispersed almost to a
  constant. A KS against it measures a censoring rule at Feedzai.
- `velocity_4w` / `velocity_24h` / `velocity_6h` — **the pairing that looked most obviously
  right, and is wrong.** Not counts: non-integer, `velocity_6h` goes negative, skew -0.06,
  coefficient of variation 0.19. A merchant's daily transaction count is heavy-tailed and
  overdispersed (skew ~16). Same word, different family.
- `session_length_in_minutes` — Rakshak has no session. There is nothing to pair it with.

### 5.4 G1c — the one external result that matters, and it is GREEN

v2's central correction over v1 is that **real counts are overdispersed** where v1 assumed
Poisson. Every other gate in the suite is internal: G1a checks the generator against its own
target, G3 determinism, G4 the quarantine, G5 the generator against a detector built on the
same generator. G1c is the only place a number arrives from outside, from a dataset that
informed none of the generator's parameters.

| BAF column | window | mean | variance | **Fano** | Poisson null |
|---|---|---|---|---|---|
| `zip_count_4w` | 4 weeks | 1,572.69 | 1,010,777 | **642.71** | 1.0 |
| `bank_branch_count_8w` | 8 weeks | 184.36 | 211,255 | **1,145.87** | 1.0 |
| `date_of_birth_distinct_emails_4w` | 4 weeks | 9.50 | 25.34 | **2.67** | 1.0 |
| `device_distinct_emails_8w` | 8 weeks | 1.02 | 0.03 | **0.032** | reported, not gated |

**The Poisson assumption is rejected on every non-degenerate count column in real-derived
fraud data, by two to three orders of magnitude.** That is genuine external support for the
premise the whole of v2 rests on, and it is the strongest external statement this project
can make.

Three caveats, none of which are optional:

1. **`device_distinct_emails_8w` is underdispersed at 0.032 and is excluded** by a rule
   declared in `baf_adapter.py` before it is applied (fewer than 10 distinct values means no
   dispersion left to measure). Its number is printed by the gate anyway. It is excluded
   because it is degenerate, not because it is inconvenient — but a reader is entitled to
   notice that the excluded column is the one that disagrees.
2. **The magnitudes do not calibrate 12.25.** Fano is neither scale-free nor window-free.
   BAF's counts pool across entities and BAF has **no entity id**, so a *per-entity* Fano —
   which is what the generator's 12.25 target actually is — cannot be computed on BAF at
   all. The comparable statistic is the pooled one: the generator's pooled 28-day count Fano
   is **439.8**, inside BAF's [2.7, 1,146]. That is a same-order-of-magnitude agreement, not
   a calibration, and it must not be reported as one.
3. **This supports the direction of v2's correction, not its size.** The claim BAF licenses
   is "real counts are far from Poisson". It does not license "and the right number is
   12.25".

### 5.5 G1d — prevalence and drift, anchored

The only two quantities BAF supplies that need no analogue argument at all.

- **Prevalence.** BAF **1.103%** against the shipped scenario's **1.47%** — a factor of
  1.33. v1's 20% is **18.1x BAF's rate** and is rejected. The gate's clause requires both
  halves, because "within 2x of BAF" alone would pass for any small number.
- **Drift.** BAF's monthly fraud rate runs **0.87% to 1.47% across eight months, a 1.69x
  swing.** Real drift in a real label distribution. Reported, **not gated**: eight months of
  account-opening fraud is not 180 days of merchant behaviour, and asserting the two agree
  would be inventing a correspondence again.

### 5.6 G2 — NOT WELL-POSED. Not green, not red

**G2 is not run as a pass/fail gate, and this is a deliberate refusal, not a skip for
convenience.** The spec asks for a LightGBM trained on the generator, scored on BAF, at
>= 0.5x its in-domain PR-AUC. After honest pruning the shared subspace is three columns:
`zip_count_4w`, `bank_branch_count_8w`, `foreign_request`. The gate measures the following
every run and refuses on the result:

| quantity | value |
|---|---|
| BAF month-7 prevalence — the random-scorer floor | **0.0147** |
| BAF in-domain PR-AUC over the 3 shared columns | **0.0197** (lift **1.34x** over random) |
| the spec's bar: 0.5 x in-domain | **0.0099** |
| BAF in-domain PR-AUC over 15 numeric BAF columns, for scale | 0.0918 |
| measured transfer, generator to BAF | 0.0169 (ratio **0.857**, would print GREEN) |
| **a seeded uniform-random scorer** | 0.0153 (ratio **0.776**, **would also print GREEN**) |
| reverse, BAF to generator | 0.0827 vs generator in-domain 0.0960 |
| resolved positives on the generator side | **15**, across 1,195 merchants |

**The spec's bar (0.0099) sits below the random floor (0.0147). A coin flip passes G2.**
A gate a random number generator clears is not evidence about the generator, and recording
it GREEN would have been the most misleading number in this repository. Two further reasons
the test cannot measure what it was written to measure: the rank alignment used to make a
tree's thresholds transfer forces the two marginals to coincide by construction, so marginal
mismatch — the thing G2 is supposed to detect — cannot cause a failure; and 15
merchant-level positives cannot support a PR-AUC either way.

**What a genuine external anchor would require**, stated so this is a specification and not
an excuse: a labelled public dataset of **entities observed over time** — per-entity event
counts, an amount, and a timestamp, with a label attached to the entity rather than the
event. BAF has none of the four. IEEE-CIS and the Sparkov/PaySim family are
transaction-level with no durable entity to watch drift on; the ULB credit-card set is
PCA-anonymised and 48 hours long. **This project could not find one, and says so rather than
approximating one.**

### 5.7 So how much external evidence does this project actually have?

Stated plainly, because this remains the weakest point in the argument:

**It has three external facts, and none of them is a transfer test.**

1. Real count data rejects Poisson decisively (G1c) — supporting the *direction* of v2's
   central correction, from data that informed none of its parameters.
2. One of three marginal-shape analogues agrees within the spec's tolerance and one does
   not (G1b), on a comparison whose disagreeing half is confounded by entity-population
   differences this file cannot separate out.
3. A real fraud base rate near 1.1% and real drift of 1.69x (G1d), which anchors the
   prevalence regime and refutes v1's 20%.

**It has no evidence at all that a model trained on this generator would work on real
payments data**, because no public dataset exists that could supply it. The generator's
*joint* structure — the thing that decides whether a model transfers — is anchored by
nothing. Every rung comparison in this project is a comparison on the generator, and the
correct summary sentence is:

> *Four of the five gates return a verdict; three are green, G1b is RED on one of three
> analogues, and G2 is recorded NOT WELL-POSED because the shared feature space needed to
> pose it does not exist. Describing this suite as "five gates, all passing" would be false.*

---

## 6. G5 is green for the RAW detector too — the demo premise does not hold

The narrative the project was built around is that a platform-wide confounder makes raw
z-scores spike while cohort residuals stay flat. **Measured on the real generator at
prevalence = 0, that is not what happens.**

| Detector | Worst window excess over nominal | Verdict |
|---|---|---|
| raw z-score | **+1.27pp** | GREEN (allowance +2pp) |
| cohort residual | **+0.72pp** | GREEN |

Both pass. The residual is better — 0.72 against 1.27 — but the raw detector did not fail,
so the confounder does not demonstrate the failure the residual is supposed to repair.

Three explanations, and this lane could not distinguish them:
1. The **+2pp allowance is generous** against a 0.5% nominal alert rate — it permits a 5x
   inflation before firing.
2. A **single-feature z-threshold is too weak a stand-in for Rung 2.** The real question is
   whether a trained model alerts on confounders, and a one-feature rule is not that model.
3. The **trailing 28-day baseline already absorbs a 5-day festival**, so by the time the
   confounder window opens the baseline has partly moved with it.

**T-151 re-runs G5 against Rungs 2 and 3, and that is the number that counts.** If the raw
detector still does not fail there, then charter K-1 has fired on its own terms and **the
figure to publish is the falsification**: the cohort-residual layer would be a solution to a
problem this generator does not create. That would be a real finding and it must not be
softened into "the residual was slightly better".

---

## 7. Findings about the tests themselves

**Parity is necessary and nowhere near sufficient.** T-120 found a real bug that the parity
suite was structurally incapable of catching: the warmup window was anchored on
`onboarded_at`, but the event stream starts at day 0 while merchants onboarded up to 120 days
earlier — so for most of the population the warmup window contained no events at all, and
every baseline was empty. Both runners agreed perfectly on the wrong answer, and parity
stayed green throughout. It was found only when real generator output arrived.

**Parity says two runners agree. It never says they agree about something meaningful.** The
fix re-anchors warmup to the first *observed* event date. The lesson belongs in the writeup:
the dual-runner design protects against one specific failure — a feature that cannot be
served — and buys nothing at all against a feature that is simply wrong.

---

## 8. Lane D — what the rungs actually measured (T-140…T-143)

Every number below is the **validation** split, days 120–149, 1,723 merchants, 51,690
merchant-days of which 51,180 are scored and 510 dropped as label-censored. Capacity
K = **9** (the lock's 50-per-10,000 rate applied to the merchants actually scored).
Merchant-level prevalence among uncensored merchants **0.996%**. The test split was never
read: the panel was materialised with `last_day = 149` and no event after it was ever
opened. `verify_lock` still passes on `eval_module_sha256` and `open_count` is still 0.

### 8.1 Eight labelled positive merchants. That is the binding constraint, not the model.

`available_labels(day 119)` — the training decision time — returns **12 positive merchants
platform-wide, of which 8 fall in the train fold**. Those 8 become 960 positive rows out of
801,240. Every other merchant is a negative by assumption, including the ones whose
disputes have not landed yet.

| as_of | resolved positives available | censored | pending |
|---|---|---|---|
| day 119 (train boundary) | **12** | 81 | 9,907 |
| day 149 (val boundary) | 39 | 81 | 9,880 |
| day 179 (end of horizon) | 77 | 81 | 9,842 |

The generator emits **147** fraud merchants. 81 merchants are censored and 60 of those are
fraud, so 41% of the positive class never resolves inside the horizon at all.

The arithmetic is the spec's, not an accident: onset is uniform on days 20–165, the
fraud→dispute gap is Exponential(21 days), and the dispute→availability gap is
Uniform(45, 120). A merchant is trainable at day 119 only if onset + 66…141 ≤ 119.
**This is FR-020 and 10-eval-harness-spec §1 working exactly as written**, and it is the
single largest fact about every Lane D result. A 180-day horizon and a real chargeback delay
leave a supervised model almost nothing to learn from. v1 avoided this by handing the model
labels the instant the fraud occurred, which is part of what made v1's numbers meaningless.

**It is not fixable inside this sprint and it should not be patched around.** The honest
mitigations are a longer horizon, or a positive-unlabelled formulation that models the
censoring instead of assuming it away. Neither is in scope, and inventing a lead-time
constant to relabel the positive window would be exactly the tuning this project exists to
avoid.

### 8.2 Charter K-1 — the verdict, and why it is not a clean one

Rung 3 is Rung 2 plus **exactly 21 cohort-residual columns** (28 → 49; nothing removed,
`assert_single_variable` refuses to train otherwise), with the same hyperparameters, the
same seed, the same rows, the same labels and the same training as_of. Relative deltas on
validation, across the **five seeds EVAL-LOCK.json declared before any model existed**
(42, 43, 44, 45, 46) — all five reported, including the one that flips sign:

| metric | 42 | 43 | 44 | 45 | 46 | mean | median | positive |
|---|---|---|---|---|---|---|---|---|
| **PR-AUC** | +21.20% | +7.74% | +9.25% | +12.69% | **−16.43%** | **+6.89%** | +9.25% | 4/5 |
| **savings** | +8.07% | +5.78% | −10.68% | −11.56% | −29.92% | **−7.67%** | −10.68% | 2/5 |
| precision@K | +11.32% | +5.62% | −11.32% | −10.78% | −38.39% | −8.71% | −10.78% | 2/5 |
| recall@K | +10.87% | +8.82% | −11.32% | −13.79% | −30.19% | −7.12% | −11.32% | 2/5 |
| ROC-AUC | +2.33% | −1.34% | +1.80% | +1.41% | −0.43% | +0.75% | +1.41% | 3/5 |

Read literally against K-1's stated threshold — *under 5% relative and K-1 has fired* —
**K-1 did not fire on PR-AUC**: +6.89% mean, +9.25% median, +21.20% at the lock's first
seed. It did fire on savings, at −7.67%.

Three things have to be said in the same breath and none of them can be dropped:

1. **The seed-to-seed spread is larger than the effect.** PR-AUC ranges −16.43% to +21.20%
   around a +6.89% mean. With 8 labelled positive merchants training the model and roughly
   17 scoreable fraud merchants in the validation fold, that is what should be expected.
   **This measurement does not have the resolution to render K-1's verdict either way**, and
   reporting +21.20% from seed 42 alone would have been a result manufactured by a seed.
2. **The charter's own adoption margin is not met.** Gate D.1 requires ≥10% relative PR-AUC
   or ≥3 days median TTD. The mean is 6.89% and median TTD is infinite for every rung, so
   **Rung 3 is not adopted**. Gate D.2 also fails — see §8.3.
3. **On the metric with rupees attached, the residual layer is worse on average.** Savings,
   precision@K and recall@K are all negative in the mean and positive in only 2 of 5 seeds.
   A layer that improves the ranking slightly and the decisions not at all is not a layer a
   risk team would ship.

**This is consistent with what §6 already said**, and it should be read that way rather than
as a surprise: G5 is green for the raw detector too (+1.27pp against a +2pp allowance), so on
this generator the platform confounders do not produce the failure the residual layer exists
to repair. §3 measured the same ceiling from the other side — the layer removes the common
mode exactly (median z +1.790 → +0.005) but cuts confounder alerts only 39.0% → 23.1%,
because it cannot remove per-merchant heterogeneity.

**No features were added, no hyperparameter was tuned, and no seed was chosen after the
fact.** The seed list is EVAL-LOCK.json's, frozen before any model existed, and every seed in
it is in the table.

### 8.3 `volume_rank` is not a dumb floor on this generator, and every rung fails against it

| policy | savings |
|---|---|
| `all_pass` | 0.00000 |
| `random_at_k` | −0.00209 |
| `all_hold` | **−28.128** |
| **`volume_rank`** | **+0.16999** |
| Rung 1 (rules) | 0.07457 |
| Rung 2 (LightGBM) | 0.09358 |
| Rung 3 (+ residuals) | 0.10112 |
| Rung 4 (cost-in-loss) | 0.08106 |
| perfect-foresight oracle | 0.91548 |

**Every rung is FLOOR-FAIL on savings**, and the floor that beats them is `volume_rank` —
alert on the K largest merchants by pre-window GMV, the same nine merchants every day (its
week-over-week alert Jaccard is exactly 1.000). It is not a dumb heuristic here: every
typology raises transaction intensity (`intensity_multiple` runs 1.5× to 22×) and
`true_loss_amount_inr` scales with volume, so by the time the validation window opens the
fraud merchants *are* among the largest merchants. Ranking by size is a partially-informed
detector on this generator, and none of the five rungs beat it.

Two readings, and this lane cannot settle between them: either the generator makes fraud too
legible in volume alone, or a supervised model with 8 positives cannot recover what a size
ranking gets for free. Both are worth stating; the first is a criticism of the generator and
belongs next to §5's.

### 8.4 The static rule engine out-ranks LightGBM by 5×, and loses on everything else

| rung | PR-AUC | ROC-AUC | ECE | savings | P@K | R@K | d30 | median TTD | Jaccard |
|---|---|---|---|---|---|---|---|---|---|
| `all_pass` | 0.00996 | 0.5000 | 0.0100 | 0.00000 | — | 0.000 | 0.000 | ∞ | — |
| `random_at_k` | 0.01048 | 0.4976 | 0.4908 | −0.00209 | 0.011 | 0.006 | 0.000 | ∞ | 0.011 |
| `volume_rank` | 0.03143 | 0.6858 | 0.4890 | 0.16999 | 0.111 | 0.059 | 0.000 | ∞ | 1.000 |
| **Rung 1** rules | **0.33644** | 0.8317 | 0.0163 | 0.07457 | 0.064 | 0.029 | 0.000 | ∞ | 0.203 |
| **Rung 2** LGBM | 0.06570 | 0.8019 | 0.0099 | 0.09358 | 0.186 | 0.090 | 0.118 | ∞ | 0.643 |
| **Rung 3** +resid | 0.07963 | 0.8206 | 0.0099 | 0.10112 | 0.207 | 0.100 | 0.118 | ∞ | 0.624 |
| **Rung 4** cost | 0.07681 | 0.8066 | 0.0097 | 0.08106 | 0.231 | 0.120 | 0.059 | ∞ | 0.768 |

(seed 42; `all_hold` cannot appear as a row — see §8.6.)

The rule engine's PR-AUC is **5.1× Rung 2's**, on a task where the charter said the bar is
LightGBM and not the rule engine. That claim was made against v1's numbers and v1's label
handling. Under the delayed-label regime the rule engine needs no labels at all and the model
has eight merchants, and the ranking metric says so.

It is not a reversal, because Rung 1 loses on every operational metric that follows the
ranking: precision@K 0.064 against 0.207, recall@K 0.029 against 0.100, day-30 detection
0.000 against 0.118, week-over-week alert stability 0.203 against 0.624, and an ECE 1.6×
worse. **A rule engine that ranks well and cannot fill an analyst queue is not the better
system**, and the divergence is itself the finding: PR-AUC and performance-under-capacity-K
are measuring different things here, which is the whole reason FR-021 puts them on one row.

### 8.5 Rung 4 (FR-032) loses on every seed. Cut.

Cost-in-the-objective, against Rung 3, same five declared seeds:

| metric | 42 | 43 | 44 | 45 | 46 | mean | positive |
|---|---|---|---|---|---|---|---|
| PR-AUC | −3.55% | −4.52% | −33.20% | −20.27% | −8.10% | **−13.93%** | **0/5** |
| savings | −19.84% | −12.43% | −43.33% | −26.26% | +22.71% | **−15.83%** | 1/5 |

Zero of five seeds improve PR-AUC and one of five improves savings. Weighting the 960
positive rows by an exposure ratio concentrates an already tiny positive class onto its
largest few merchants, which is the opposite of what eight positives need. **Rung 4 is
dropped from the scoring path.** It is implemented and tested and it stays in the tree as a
negative result rather than being deleted.

### 8.6 `all_hold` cannot produce an `EvalResult` row, so T-140's clause is met three ways of four

T-140 asks that "all four floors and the rule engine score on the validation split and
produce complete `EvalResult` rows". `all_pass`, `random_at_k` and `volume_rank` do.
`all_hold` alerts on every merchant every day, so `alerts_per_day` is 1,723 against K = 9 and
`build_eval_result` refuses — correctly — to compute metrics above capacity. Its savings is
on **every** row as `savings_floor_all_hold` (−28.128), which is where FR-021 actually puts
it. The refusal is asserted in `tests/unit/test_rungs_0_1.py` rather than worked around.

### 8.7 Nothing detects anything quickly. Median TTD is infinite for every rung.

Detection rate at day 7 and at day 14 is **0.000 for every rung and every floor**. The best
day-30 rate is 0.118 (Rungs 2 and 3). Median TTD is `inf` across the board, which means more
than half of the scoreable positives are never alerted on inside a 30-day window at K = 9.
Per-typology recall at day 30 is 0.0 for R1, R2, R3 and R5 and 0.5 for R4; R6–R9 have no
day-30-eligible members in the validation fold and are `nan`, not 0.

Charter §2 makes TTD an equal-standing win condition. On this measurement it does not
discriminate between rungs at all, because no rung achieves a finite median.

### 8.8 Three registered features are effectively constant on real generator output

Measured over all 852,930 materialised merchant-days:

| feature | non-zero rows | share |
|---|---|---|
| `f_retry_burst_rate` | 584 | 0.07% |
| `t_micro_share` | 1,030 | 0.12% |
| `t_new_max_event` | 1,651 | 0.19% |

All three sit at 0.000 through the 99th percentile. They are not broken — the parity suite
and G4b agree online and offline on them — they simply almost never fire on this generator,
so the tree ensemble can gain nothing from them and Rung 1's micro-ticket and retry-burst
rules are close to dead weight. `f_retry_burst_rate` is the one that should be looked at: R3
is 15% of the typology mix and Hawkes self-excitation is switched on for it precisely so that
this feature has something to see.

### 8.9 The loss is amortised per day, and that is a decision made in `cli.py`

The harness charges `row_cost` per merchant-day and `Truth.loss_inr` is a merchant-level
total. Handed in unamortised, a merchant that turns on day 40 of 180 is billed 140 times its
own loss, which inflates the all-PASS denominator until `all_hold` shows **+0.685 savings** —
paranoia looking profitable. `cli._build_truth` divides `true_loss_amount_inr` by the days
from onset to the end of the horizon, so the loss summed over any window is the loss accrued
in that window. Made in `cli.py`, not in the frozen `eval/`, and named here because it moves
every savings number in §8.3 and a reader is entitled to know it was a choice rather than the
only reading.

### 8.10 What the rungs were trained on, precisely

Trained on the **train** split only — train-fold merchants, days 0–119 — with labels
available at day 119, and measured on the **val** split, which is disjoint in both merchant
and time. The board's "trains on train+val ONLY" is the constraint that no test data is used;
training on val and reporting val would be self-evaluation. The T-151 refit on train+val at an
as_of of day 149 would see 39 available positives rather than 12, so the final test-split
numbers should be expected to differ from these by more than seed noise.

Rung 2 trains in **7.6 s** on 4 cores (NFR-06 budget 20 min) and the artifact is **0.489 MB**
(NFR-05 budget 20 MB). Compute was never the constraint. Labels were.
