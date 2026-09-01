<!-- HEAD
FILE:     project-context/13b-survey-delayed-feedback-pu.md
PHASE:    survey (cycle 4)
UPDATED:  2026-09-01
STATUS:   recommendation-ready — one method named, one numeric gate pre-registered
SUMMARY:  The delayed-feedback literature is almost entirely inapplicable here: every method
          in it needs an observable event-time origin, and ours (drift_onset_at) is
          radioactive. The PU literature is applicable but corrects the direction that
          barely matters — the unlabelled set is 98.9% clean. The direction that does
          matter is the one both literatures cover thinly: 35-47% of the labelled
          POSITIVES at the day-239 boundary are merchants that did nothing wrong.
          First place is a two-sided, known-rate affine posterior correction
          (Elkan-Noto rescaling generalised by the Scott/Natarajan two-sided noise
          identity), ~15 lines, no new dependency, gated on ECE not on TTD.
OPEN:     (1) Whether the cycle-3 decision layer assigns PASS/REVIEW/HOLD by a
          cost-derived probability threshold or purely by top-K rank. If purely by rank,
          the recommended rung is a no-op on savings and should not be adopted. This must
          be read out of the existing decision-layer code BEFORE pre-registration.
          (2) Whether spurious chargebacks pass through the same Uniform(45,120)
          availability delay as genuine ones, or are emitted immediately. This changes
          the count of false positive labels available at the training boundary from ~70
          to ~118 and therefore the positive-class precision from 0.66 to 0.53. The
          generator spec §6 does not say. Settle it by counting, not by reading.
-->

# Survey — delayed feedback and positive-unlabelled learning, for the LABEL constraint

---

## 1. What this literature is for

### 1.1 The label mechanism, restated exactly

From `08-generator-v2-spec.md` §6 and its `labels:` config block:

```
label_event_at     = drift_onset_at + Exponential(mean = 21 days)
label_available_at = label_event_at + Uniform(45, 120) days
unreported_rate           = 0.15    # fraud merchant, never disputed → label = 0
spurious_chargeback_rate  = 0.003   # legitimate merchant → label = 1
prevalence                = 0.0147
```

Two facts about these parameters that are load-bearing and easy to miss:

- `fraud_to_dispute_mean_days: 21` is a **single global scalar**. It is not per-typology,
  not per-persona, not per-tier. The delay draw is i.i.d. and independent of every covariate.
- `unreported_rate: 0.15` is likewise a **single global scalar**. The reporting decision does
  not depend on merchant size, typology, or feature values.

Both of those are why §2 comes out the way it does.

### 1.2 The delay distribution is not exponential and this matters

`D = Exp(21) + U(45, 120)`:

| quantity | value |
|---|---|
| support | **[45, ∞)** — a hard floor, zero density below 45 days |
| mean | 21 + 82.5 = **103.5 days** |
| variance | 21² + 75²/12 = 441 + 468.75 = **909.75**, sd = **30.16** |
| coefficient of variation | 30.16 / 103.5 = **0.29** |
| skewness | 2·21³ / 909.75^1.5 = **0.68** |

An exponential has CV = 1 and skewness = 2. The convolution has CV = 0.29 and skewness 0.68.
It is far closer to a Gamma with large shape — near-Normal with a modest right tail — than to
an exponential. Concretely, **an exponential fitted to the correct mean of 103.5 days places
35.3% of its probability mass below day 45, where the truth is exactly 0%**
(`1 − e^{−45/103.5} = 0.353`), and 18.4% below day 21. Every delayed-feedback method that
assumes an exponential delay with a fitted rate — which is most of the family descending from
Chapelle — would, on this data, declare roughly a third of not-yet-disputed drifted merchants
"overdue, therefore probably negative" during a window in which by construction *no* dispute
can have surfaced.

### 1.3 The arithmetic at the day-239 training boundary

40,000 merchants, π = 1.47%, onsets uniform on days 30–364, spec Monte Carlo yield ≈ **134
trainable positives**.

| quantity | count | derivation |
|---|---|---|
| genuinely fraudulent merchants | **588** | 0.0147 × 40,000 |
| ever reported (85%) | 500 | 0.85 × 588 |
| already drifted by day 239 | ≈ 368 | 588 × (239−30)/(364−30) = 588 × 0.626 |
| **labelled positive at day 239** | **≈ 134** | spec Monte Carlo (independent estimate: ≈ 155 pre-fold, consistent) |
| **spurious positives, whole run** | **≈ 118** | 0.003 × (40,000 − 588) |
| spurious positives available by day 239 | **~70 to 118** | depends on OPEN item (2) |
| hidden positives inside the label=0 set | ≈ 454 | 588 − 134; of these 88 permanently unreported, 366 not-yet-available |
| unlabelled ("negative") set size | ≈ 39,750 | |

Two ratios fall out, and they are the entire finding of this survey:

> **Contamination of the negative set: 454 / 39,750 = 1.14%.**
> **Corruption of the positive set: 70–118 false out of 204–252 labelled = 35% to 47%.**

The PU literature exists to correct the first number. It is 1.14 percentage points. A
gradient-boosted tree splitting on 39,750 rows does not notice 1.14% of them carrying the
wrong sign. The second number — that between a third and a half of the 200-odd rows carrying
`label = 1` belong to merchants that did nothing — is 30× larger and is the direction both
literatures cover thinly.

Effective noise rates, in Natarajan's notation:

```
ρ₊ = P(ỹ=0 | y=1) ≈ 1 − 134/588 = 0.772     (dominated by DELAY, not by unreported_rate)
ρ₋ = P(ỹ=1 | y=0) = 0.003
```

Note that `unreported_rate = 0.15` contributes only 0.15 of that 0.772. **Five sixths of the
false-negative rate is the delay, not the non-reporting.** Anyone who reads "15% unreported"
and reaches for PU learning has mis-attributed the problem by a factor of five.

The label frequency (Elkan–Noto's `c`) is therefore:

```
c = P(ỹ=1 | y=1) ≈ 134 / 588 = 0.228
```

and the posterior probability that a labelled positive is genuine is

```
π_pos = cπ / (cπ + ρ₋(1−π)) = 0.003352 / (0.003352 + 0.002956) = 0.531
```

Roughly a coin flip.

### 1.4 Which cycle-3 failure this literature can address

**Failure 1 — TTD was arithmetically unmeasurable.** This literature addresses it **not at
all**. It was a geometry defect in the onset window and the cycle-4 spec already fixes it by
moving the data. No delayed-feedback or PU method makes an unmeasurable quantity measurable.
Anything in this survey that claims to improve TTD is claiming to improve a number carrying a
±13 pp standard error over 14 merchants; see §6.

**Failure 2 — every learned model loses on savings to `volume_rank` (0.6016 vs 0.4348)
despite ranking far better (PR-AUC 0.836 vs 0.217).** This literature addresses it
**partially, and only through one specific channel**. A ranking win coexisting with a cost
loss is the signature of a *decision-threshold* problem, not a *ranking* problem — and under
the label mechanism above the model's scores are deflated by a factor of ≈ 4.4 (it was
trained to predict "a dispute is available by day 239", which happens for only 22.8% of
fraudulent merchants). A probability that is systematically 4.4× too small, fed into a
cost-aware decision layer that compares it against a cost-derived threshold, will
systematically under-act. That is a real, mechanism-derived, correctable defect.

It is also, honestly, most of what this literature can do here. See §9.

---

## 2. Does SCAR hold here?

**Answer: SCAR holds for the non-reporting mechanism exactly, and fails for the delay
mechanism — but it fails in a way that is neutralised by the point-in-time filter, and the
residual failure is second-order.**

SCAR (Elkan & Noto 2008; formalised as an assumption class by Bekker & Davis 2020) says the
propensity `e(x) = P(labelled | y=1, x)` is a constant `c`, independent of `x`. SAR relaxes
this to `e(x)` varying with the covariates.

**The reporting mechanism is SCAR, exactly.** `unreported_rate = 0.15` is a flat scalar in the
config, drawn independently per fraudulent merchant with no dependence on typology, persona,
tier, GMV, or any feature. There is no more perfectly SCAR mechanism than one implemented as
a single Bernoulli constant. If `unreported_rate` were the whole story, the SCAR half of the
PU literature would apply without qualification.

**The delay mechanism is not SCAR, and the reason is subtle.** The delay draw itself is
covariate-independent (§1.1). But whether a positive is *labelled at boundary T* depends on
`T − drift_onset_at`, and the merchant's feature vector at time T also depends on
`T − drift_onset_at`, because the fraud typologies ramp their behaviour after onset. A
merchant that drifted long ago is *both* more likely to have a matured dispute *and* more
likely to look fraudulent in features. The propensity is therefore correlated with `x`, which
is the definition of SAR:

```
e(x, T) = 0.85 · P(D ≤ T − onset(x))     and onset is inferable from x
```

**Three things stop this from being fatal:**

1. **The 45-day floor makes the dependence a step, not a gradient, over most of the range.**
   No merchant that drifted within 45 days of the boundary can be labelled, whatever its
   features. For those merchants `e(x) = 0` uniformly — which is not SAR-with-varying-
   propensity so much as deterministic administrative censoring, and administrative censoring
   is what point-in-time filtering already handles correctly.
2. **The covariate dependence is monotone and rank-preserving.** Under SAR the danger is that
   the labelling mechanism *reorders* the score. Here the propensity increases with
   time-since-onset and so does the true posterior; the bias inflates confidence in
   long-drifted merchants and deflates it for recent ones, but it does not invert any pair.
   The practical consequence is a TTD bias — the model is systematically better at detecting
   merchants that drifted long ago, which is exactly the wrong direction for a latency
   metric, and should be *reported* rather than corrected.
3. **The SAR-correct half of the literature is unaffordable.** Bekker et al. (2019) and
   Gerych et al. estimate a propensity function `e(x)` from PU data; the identification
   requires either a propensity-attribute assumption or a labelled-positive sample large
   enough to fit a second model. With 134 positives, fitting a second model to estimate a
   propensity surface is not a defensible use of the sample. Teisseyre et al. (2024) give a
   *test* for SCAR that could be run cheaply, but a test whose rejection leads to a method we
   cannot afford is a test that changes nothing.

**Ruling for this cycle: treat the labelling as SCAR with a single scalar `c` estimated at the
training boundary, declare the SAR residual as a known, signed, TTD-directional bias in
`LIMITATIONS.md`, and do not attempt to correct it.** The SCAR half of the PU literature is
admissible. The SAR half is admissible in principle and unaffordable in practice.

---

## 3. The map

### 3.1 Delayed-feedback families

| family | representative work | what it assumes | does the assumption survive here? |
|---|---|---|---|
| **Wait for maturity** (the baseline everyone reports against) | Chapelle 2014 §2 | you can afford to discard the most recent window of data | Survives. Costs 103 days of feature freshness and introduces train/serve skew. This is the only member of the family that needs no event-time origin. |
| **Parametric delay model (DFM)** | Chapelle 2014 (KDD, DOI 10.1145/2623330.2623634) | delay ~ Exponential with rate `exp(w·x)`; **click time observed** | **Fails twice.** The delay is not exponential (§1.2 — an exponential fit misplaces 35.3% of mass below the hard floor), and the event-time origin is `drift_onset_at`, which is radioactive. |
| **Non-parametric delay model** | Yoshikawa & Imai, AISTATS 2018 | delay distribution unknown but **elapsed time observed** | Fixes the distributional half, not the origin half. Still needs an observable clock start. **Inapplicable.** |
| **Fake-negative weighting / calibration (FNW, FNC)** | Ktena et al., RecSys 2019, arXiv:1907.06558 | a *streaming* ingest where every arrival is labelled negative and a duplicate positive is inserted on conversion | Our training set is a single point-in-time snapshot, not a stream, and the duplicate-insertion mechanic has no merchant-level analogue. **Inapplicable as published**; its importance-weight identity `p/(1+p)` is a special case of the affine correction in §5. |
| **Feedback-shift importance weighting (FSIW)** | Yasui et al., WWW 2020, DOI 10.1145/3366423.3380032, arXiv:2002.02068 | delayed-observed vs true conditional distributions differ by a weight estimable from a long-elapsed subsample | Needs elapsed time to define "long-elapsed". **Inapplicable.** |
| **Elapsed-time sampling (ES-DFM)** | Yang et al., AAAI 2021 35(5):4582–4589, arXiv:2012.03245 | an explicit elapsed-time window per training example | Same. **Inapplicable.** |
| **Label correction / influence functions** | Wang et al., WWW 2022 (DOI 10.1145/3485447.3511965); Chen et al., arXiv:2502.01669 | neural, differentiable, streaming | Autograd. **GATED** (see §4). |
| **Applied payments/credit variants** | "Mind the Gap", KDD 2025 (DOI 10.1145/3711896.3737247); arXiv:2409.10111 | delayed labels in nonpayment / tabular fraud streams | Useful as evidence that the problem is real in payments. No transferable method that avoids the origin problem. |

> **The single most important sentence in this half of the survey:** every method in the
> delayed-feedback canon except "wait for maturity" requires an **observable event-time
> origin** — in CVR prediction, the click timestamp. Ours is `drift_onset_at`, which the
> constraint list marks radioactive and which a model may never read. There is no legal
> surrogate: all merchants exist from day 0, so time-since-onboarding is constant across the
> population and carries no maturity signal. **This family does not transfer.**

### 3.2 Survival / time-to-event framing

Two distinct framings, with different legality:

**(a) Model the delay as a duration.** Origin = `drift_onset_at`, event = `label_event_at`,
duration = the delay. This is the natural framing and it is **disqualified**: it needs the
true onset at training time.

**(b) Model time-to-label-availability from a legal origin.** Origin = day 0, event =
`label_available_at`, administratively right-censored at the training boundary. This is
**legal** — `label_available_at` is a label field, not a ground-truth field — and it has one
genuine attraction: a discrete-time hazard model is fitted as *one binary classification per
merchant-day on the risk set* (Suresh et al., BMC Med Res Methodol 2022,
DOI 10.1186/s12874-022-01679-6), so it needs no survival library at all, runs in LightGBM
with a time index as a feature, and emits exactly the per-merchant-per-day score vector the
rung interface wants.

Be honest about what it buys:

- **Cox proportional hazards** on 134 events with 28 features gives events-per-variable ≈ 4.8,
  under half the conventional EPV ≥ 10 floor. The proportional-hazards assumption is also
  violated by construction: the hazard of label availability is exactly zero for 45 days after
  onset and then rises, which is not a proportional shift of any common baseline.
- **Discrete-time hazard as sequential binary classification** avoids the PH assumption and is
  implementable with zero new dependencies. But over the fixed 60-day validation window, the
  cumulative-incidence ranking it produces is very nearly a monotone transform of the
  classifier's posterior — you pay a person-period row expansion (40,000 × 240 ≈ 9.6M rows for
  training) for a re-parameterisation, not for new information.
- **Competing risks (Fine–Gray)** requires a competing event. Ours is administrative censoring
  at the horizon, which is not a competing risk. Merchant churn would be, and the generator
  does not model it. Nothing to compete with; the framing is empty here.

**Ruling: the survival framing is legal, implementable without a library, and does not earn
its row expansion at 134 events.** Listed in the candidates table, not recommended.

Library note, since the brief flagged it: **`scikit-survival` is GPL-3.0-or-later** (verified
at `sebp/scikit-survival/COPYING`, GitHub API `spdx_id` also reports GPL-3.0-or-later) and is
therefore **DISQUALIFIED outright**, not gated. **`lifelines` is MIT** (verified at
`CamDavidsonPilon/lifelines/LICENSE`, "Copyright (c) 2017 Cameron Davidson-Pilon") and would
be admissible — but under the ruling above there is nothing to use it for.

### 3.3 PU families

| family | representative work | what it assumes | violated here? |
|---|---|---|---|
| **Two-step / reliable negatives** | Liu et al., ICDM 2003 (S-EM, spy technique); Yu et al. (PEBL) | a reliably identifiable negative core exists in U | Our U is already 98.9% negative. Identifying "reliable negatives" in a set that is 98.9% negative is a solved problem you do not have. **Redundant.** |
| **PU bagging** | Mordelet & Vert, Pattern Recognit. Lett. 37:201–209, 2014, DOI 10.1016/j.patrec.2013.06.010 | few positives, low contamination in U | Assumption *holds* — this is the regime the paper was written for. But at 1.14% contamination its PU-specific benefit is ~nil and the residual benefit is ordinary bagging variance reduction, which `bagging_fraction` already gives. **Applicable but not PU-motivated.** |
| **Cost-sensitive / class-prior-corrected posterior rescaling** | Elkan & Noto, KDD 2008, DOI 10.1145/1401890.1401920 | SCAR, and `c` known or estimable | **Holds** (§2). Model-agnostic, weight-only, no autograd. |
| **Unbiased PU (uPU)** | du Plessis, Niu & Sugiyama, NeurIPS 2014; convex form ICML 2015, PMLR 37:1386–1394 | SCAR, π known, and a **negative weight** on the P-as-negative term | Assumption holds; the *implementation* does not — see §3.4. |
| **Non-negative PU (nnPU)** | Kiryo et al., NeurIPS 2017, arXiv:1703.00593 | as uPU, plus a flexible model that would otherwise overfit the negative-risk region | Assumption holds; published implementation is a neural one. **GATED**, see §4. |
| **PU + biased negatives (PUbN)** | Hsieh, Niu & Sugiyama, ICML 2019, PMLR 97, arXiv:1810.00846 | a biased negative set is available alongside P and U | We do not have a separately-collected biased-negative set. **Inapplicable.** |
| **Class-prior / mixture-proportion estimation** | Elkan–Noto e1/e2/e3; Ramaswamy, Scott & Tewari (KM1/KM2), ICML 2016, arXiv:1603.02501; Bekker & Davis (TIcE), AAAI 2018, DOI 10.1609/aaai.v32i1.11715; Ivanov (DEDPUL), ICMLA 2020, arXiv:1902.06965; Garg et al. (BBE/CVIR/TED^n), NeurIPS 2021, arXiv:2111.00980 | various | **Not needed.** π = 0.0147 by construction. See §3.5. |
| **PU with noisy positives** | Jain, White & Radivojac, NeurIPS 2016, arXiv:1606.08561 | positives are contaminated *and* U is a mixture | **This is our exact situation.** It is also the thinnest corner of the literature; see §8. |
| **Class-conditional label noise, both directions** | Natarajan et al., NIPS 2013, pp. 1196–1204; journal version Natarajan et al., JMLR 18(155), 2017 | ρ₊, ρ₋ known or estimable, ρ₊ + ρ₋ < 1 | **Holds.** ρ₊ ≈ 0.772, ρ₋ = 0.003, sum = 0.775 < 1. The identity behind §5's recommendation. |

### 3.4 Can nnPU be carried to LightGBM without autograd? — an explicit answer

The brief asks for this explicitly, so here it is, with the mechanics.

The uPU risk estimator is

```
R̂_uPU(f) = π·Ê_P[ℓ(f(x), +1)]  +  ( Ê_U[ℓ(f(x), −1)] − π·Ê_P[ℓ(f(x), −1)] )
```

The third term carries a **negative coefficient on positive examples evaluated as negatives**.
As a sample-weighting scheme that is a weight of `−π` on a duplicated row.

- **Via `sample_weight`: NO.** LightGBM's documentation states, for `Dataset.weight` and for
  `set_weight`, verbatim: *"Weight for each instance. Weights should be non-negative."*
  (verified at `lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Dataset.html`). uPU is
  therefore **not** expressible through the sample-weight path.
- **Via a custom objective: YES, mechanically — and it is a bad idea.** LightGBM's `fobj` is a
  plain Python callable returning `(grad, hess)` numpy arrays. No autograd is involved; the
  gradient and Hessian of the logistic loss are closed-form, so the signed uPU combination can
  be assembled by hand in ~20 lines of numpy, and nnPU's `max(0, ·)` correction is
  implementable exactly as Kiryo publishes it — compute the bracketed U-term risk over the
  current batch and, when it goes negative, flip the gradient of the correction term. This
  does not require torch.
- **Why it is still a bad idea here.** Negative per-sample Hessian contributions make
  `sum_hessian` in a leaf negative, and LightGBM's leaf value is `−sum_grad / (sum_hess + λ)`;
  a near-zero or negative denominator produces unbounded leaf outputs. `min_child_weight`
  (the min sum of Hessian in a leaf) becomes meaningless. This is precisely the difficulty
  Zhao et al. document when building a boosting algorithm for PU data — AdaPU
  (arXiv:2205.09485; Statistics and Computing, DOI 10.1007/s11222-024-10529-y) exists
  *because* the negative-weight problem is not trivially delegated to an off-the-shelf
  booster, and their fix is a bespoke weak-learner fitting procedure, not a `fobj`.
- **And the payoff is nil.** The entire uPU/nnPU correction operates on the U-term, i.e. on
  the 1.14% of the unlabelled set that is secretly positive. It buys you a corrected risk over
  454 hidden merchants out of 39,750, using a correction whose variance is set by 134 positive
  rows. §1.3 says this is the direction that does not matter.

**Ruling: nnPU's risk estimator is portable to LightGBM without autograd, via a hand-derived
custom objective; the off-the-shelf implementation (`kiryor/nnPUlearning`) is not usable and
is GATED regardless. Do not adopt it. Not because it is illegal — because it corrects the
wrong 1.14%.**

### 3.5 Knowing the prior — what it actually buys

The project knows π = 0.0147 by construction. In the PU literature, mixture-proportion
estimation is the single largest error source and the subject of an entire sub-field
(Elkan–Noto's e1/e2/e3 estimators, which du Plessis & Sugiyama showed systematically
*over*-estimate; KM1/KM2's kernel embeddings; TIcE's decision-tree induction; DEDPUL's
density-difference postprocessing; BBE's best-bin estimation). Bekker & Davis's survey reports
KM2 and TIcE as the most accurate on SCAR PU data — and all of them degrade sharply at small π
and small n, which is where we are.

So knowing π **removes the largest error term in a PU pipeline for free**, and that is worth
saying out loud in the write-up.

But be honest about the size of the prize: knowing π lets you compute a correction whose
magnitude is proportional to π. At π = 0.0147, the correction to the negative class is 1.14
percentage points. **Knowing the prior perfectly is worth a great deal relative to estimating
it badly, and very little relative to not correcting at all.** What knowing π *does* buy here
is the ability to identify `c` in closed form from an observable count (§5, step 1), which is
the one place the known prior is genuinely load-bearing.

---

## 4. Candidates table

Ranked. "Positives required" is a judgement about the sample size at which the method's
estimated quantities stop being noise-dominated, against the 134 available.

| # | method | assumption it needs | holds here? | LightGBM, no autograd? | positives needed vs 134 | licence | expected effect given the power ceiling |
|---|---|---|---|---|---|---|---|
| **1** | **Two-sided affine posterior correction** (Elkan–Noto SCAR rescaling generalised by the Scott/Natarajan two-sided noise identity), `p = (s − ρ₋)/(c − ρ₋)` | SCAR labelling; ρ₋ known; ρ₋ < c | **Yes** (§2); both rates known by construction | **Yes** — post-hoc scalar map on the score column, zero training change | ~100 for `c` to ±11% relative; **134 suffices** | none — numpy only (BSD-3, pinned) | **ECE: large and near-certain** (score scale off by 4.4×). **PR-AUC / precision@K / recall@K / TTD: exactly zero by construction** (affine monotone). **Savings: small, uncertain, conditional on OPEN item (1).** |
| 2 | Iterative positive-set cleaning (small-loss / self-cleaning against `spurious_chargeback_rate`) | corrupted positives are separable from genuine ones by model score | Partly — but early-stage genuine fraud also scores low | Yes — refit loop, no new dep | wants ≳ 1,000; **134 is far too few** | none | Could move PR-AUC either way. **Rejected**: it preferentially deletes weak-signal early-stage positives, i.e. exactly the merchants TTD exists to measure. Actively dangerous for the cycle's headline metric. |
| 3 | PU bagging (Mordelet & Vert) with LightGBM base learners | few positives, low U contamination | Assumption holds; PU *motivation* does not (1.14%) | Yes — sklearn + lightgbm | works at 134 | lightgbm MIT (verified `microsoft/LightGBM/LICENSE`); scikit-learn BSD-3 | Variance reduction indistinguishable from `bagging_fraction`. ~100 model fits, latency risk against the 10 ms p99. Not separable at ±13 pp. |
| 4 | Wait-for-maturity (train on features as of day 136, labels as of day 239) | you can afford 103 days of staleness | Holds | Yes — a date change | works at 134 | none | Introduces train/serve skew on 28 features. Plausible but unmeasurable at this power; would need a ranking metric to move by ≥ 25 pp. |
| 5 | Discrete-time hazard as sequential binary classification (person-period expansion, day index as feature) | legal origin + administrative censoring | Holds if origin = day 0, event = `label_available_at` | **Yes** — no survival library needed | 134 events / 28 features → EPV 4.8, **below the EPV ≥ 10 floor** | none (would not need `lifelines`) | ≈ monotone re-parameterisation of the classifier over a fixed 60-day window. Pays a 9.6M-row expansion for no new information. |
| 6 | uPU / nnPU risk estimator as a hand-derived LightGBM `fobj` | SCAR, π known | Holds | **Mechanically yes**, but negative Hessians destabilise leaf values (§3.4) | nnPU's whole point is small-P robustness, so 134 is arguably in range | risk estimator is a formula, not a library; `kiryor/nnPUlearning` is **NOASSERTION** (verified via GitHub API) → **unusable** | Corrects the 1.14% direction. Immeasurable. |
| 7 | Cox PH on time-to-label-availability | proportional hazards | **No** — hazard is identically zero for 45 days post-onset then rises | via `lifelines` only | EPV 4.8 | `lifelines` **MIT** (verified `LICENSE`); **`scikit-survival` GPL-3.0-or-later — DISQUALIFYING** (verified `COPYING`) | None. PH assumption violated by construction. |
| 8 | Off-the-shelf PU classifiers (`pulearn`: `ElkanotoPuClassifier`, `BaggingPuClassifier`) | SCAR | Yes | Yes, sklearn-compatible | fine | **BSD-3-Clause** (verified GitHub API `pulearn/pulearn` and PyPI; latest 0.2.0, 2026-03-14) | Would work, but wraps ~15 lines. **Adding a dependency to avoid 15 lines fails the standing rule.** Named here so the licence check is on record. |
| 9 | Chapelle DFM / ES-DFM / FSIW / FNW-FNC | observable elapsed time from an event-time origin | **No — origin is `drift_onset_at`, radioactive** | n/a | n/a | `ThyrixYang/es_dfm` **NOASSERTION** (verified via GitHub API) → unusable anyway | **Disqualified.** No legal surrogate origin exists (§3.1). |
| 10 | Neural delayed-feedback (DGNN, influence-function DFM, Dist-PU, Self-PU) | differentiable model | — | **No — requires torch** | — | — | **GATED.** Visible and revisitable; blocked by the standing no-autograd ADR, not by merit. Revisit only if that ADR is reopened. |
| 11 | Jain–White–Radivojac noisy-positive PU (arXiv:1606.08561) | positives contaminated, U a mixture, univariate transform preserves the prior | **This is our exact setting** | Yes in principle (classifier-agnostic, uses univariate transforms + KDE) | needs enough positives for a stable density estimate; **134 is marginal at best** | `DEDPUL` (related, MIT, verified) — the 2016 method itself has no maintained reference package I could verify → **UNVERIFIED** | The right paper for our situation. Not enough data to run it. Cite it as the gap (§8). |

---

## 5. First place, unambiguous

### The method

**A two-sided, known-rate affine posterior correction: Elkan–Noto SCAR rescaling generalised
by the Scott/Natarajan class-conditional-noise identity, applied as a post-hoc scalar map on
the existing LightGBM rung's score.**

### Why it wins

It wins because it is the only candidate whose correction is aimed at a defect this data
actually has, whose two required constants are both known by construction rather than
estimated from 134 rows, and whose variance contribution is therefore essentially zero — and
because it is the only candidate that speaks directly to cycle-3 failure 2. Every alternative
either targets the 1.14% contamination of the negative set (uPU, nnPU, PU bagging, two-step
methods — a direction that is arithmetically negligible here), or requires an event-time
origin the constraint list forbids (the entire Chapelle lineage), or requires more positives
than exist (positive-set cleaning, Jain et al. 2016, any survival regression at EPV 4.8), or
adds a dependency to avoid fifteen lines. This method instead corrects the one quantity that
is provably and largely wrong: the *scale* of the predicted probability, which is off by a
factor of ≈ 4.4 because the model was trained to predict "a dispute is available by day 239",
an event that occurs for only 22.8% of fraudulent merchants. Because the map is affine and
monotone, it cannot damage any ranking metric — which makes its adoption gate an
implementation check as much as a scientific one — and because it is a scalar map on an
existing column, it costs two floating-point operations per scored row and cannot threaten
the 10 ms p99 budget. It is the laziest correct thing, and at this sample size laziness and
correctness point the same way.

### The identity

Let `s(m,t) = P̂(ỹ = 1 | x(m,t))` be the existing rung's output — the probability that a
dispute for merchant `m` is *available and positive* at the training boundary. Let
`p(m,t) = P(y = 1 | x(m,t))` be what the decision layer actually needs. Then

```
P(ỹ=1 | x) = P(ỹ=1 | y=1)·p  +  P(ỹ=1 | y=0)·(1−p)
           = c·p + ρ₋·(1−p)
           = ρ₋ + (c − ρ₋)·p
```

Invert:

> ```
> p̂(m,t) = clip( ( s(m,t) − ρ₋ ) / ( c − ρ₋ ),  0,  1 )
> ```

That is the whole method. With `ρ₋ = 0` it collapses to Elkan–Noto's `p = s/c`. With
`ρ₋ > 0` it is the two-sided correction, and it is exactly the inverse of Natarajan et al.'s
class-conditional noise transition applied at the posterior rather than at the loss.

### Estimating `c` from observables only — no ground-truth field is read

```
c = ( N₁(T)/M  −  ρ₋·(1 − π) ) / π
```

where

- `N₁(T)` = count of distinct merchants carrying an available `label = 1` at the training
  boundary `T = 239`. This reads the **label** table, not `ground_truth`.
- `M` = merchant count (40,000), `π` = 0.0147 (the declared prevalence, already reported by
  the eval harness on every row), `ρ₋` = 0.003 (the declared spurious rate).

Nothing here touches `drift_onset_at`, `risk_typology_id`, or `persona_id`. The method needs
the **label** timestamp only, and in fact needs only the label *count*.

Expected value at `T = 239`: with `N₁ ≈ 234` (134 genuine + ~100 spurious matured),
`c = (234/40000 − 0.003·0.9853)/0.0147 = (0.005850 − 0.002956)/0.0147 = 0.197`. If spurious
labels do not mature (OPEN item 2 resolves the other way), `N₁ ≈ 204` and `c = 0.146`. Against
the ground-truth-derived value 134/588 = 0.228, the estimator recovers the right order and
sits within its own sampling error — see §9 for why that residual matters less than it looks.

**Sampling error on `c`:** `Var(N₁) ≈ N₁` (Poisson) → `sd(N₁) ≈ 15.3` →
`sd(c) = 15.3/(M·π) = 15.3/588 = 0.026`, i.e. **c is known to about ±11% relative**. That
propagates to `p̂` as a ±11% multiplicative uncertainty, against a correction of ×4.4 to ×6.8.
The correction dominates its own error by a factor of ~40.

### Producing the per-merchant-per-day score vector

The rung is a thin wrapper, not a new model:

1. **Train** exactly as the existing LightGBM rung does — same features, same
   point-in-time-filtered binary target `ỹ`, same hyperparameters, same 5 seeds. No change.
   The correction deliberately does not touch training, so the two rungs are comparable
   row-for-row.
2. **Compute `c` once**, at fit time, from the formula above, using the training-boundary
   label count. Store it in the model artefact alongside `model_version`.
   **Use `c(T)`, the training-boundary value — not a scoring-day value.** The model was fitted
   to "available by `T`"; that is the event whose propensity must be inverted. Recomputing `c`
   per scoring day is the obvious trap and is wrong.
3. **Score** for each `(merchant_id, as_of)` in the validation window exactly as today, giving
   `s(m,t)`.
4. **Map**: `score = clip((s − 0.003) / (c − 0.003), 0.0, 1.0)`. Emit this as the `score`
   field of the `Decision` row. Two float ops per row.
5. **Actions** are derived by the *unchanged* decision layer from the corrected `score`.
   The rung must not touch the decision layer, the cost model, or the eval package.
6. **Diagnostics to emit** (cheap, and they are what makes the result readable):
   the fitted `c`, its Poisson standard error, and the **clip rate** — the fraction of scored
   rows saturating at 1.0. A clip rate above a few percent means the SCAR approximation is
   straining and should be reported.

No new dependency. No autograd. Roughly 15 lines plus the diagnostic emission. Latency delta:
unmeasurable.

---

## 6. The pre-registered numeric adoption gate

Decided in advance, on a metric the sample size can actually resolve.

**Primary gate — calibration.**

> Adopt the rung if, on the **validation** split, averaged over the five declared seeds, the
> corrected rung's **ECE (10 equal-mass bins)** is at most **0.60 ×** the uncorrected
> LightGBM rung's ECE on the same split and seeds. That is, **a ≥ 40% relative reduction in
> ECE**.

Rationale for the threshold: the correction rescales the score by 1/(c − ρ₋) ≈ 4.4–6.8. If the
uncorrected rung's calibration error is dominated by that scale factor — which the mechanism
predicts — the reduction should be far larger than 40%. Setting the bar at 40% leaves room for
the SAR residual (§2), the ±11% error on `c`, and clipping, while still being a bar that a
no-op cannot clear. ECE is computed over ~40,000 merchants × 60 days ≈ 2.4M decisions, so it
is the only headline metric in the suite that is not power-starved.

**Guard 1 — implementation check (must pass).**

> `|ΔPR-AUC| ≤ 0.005` between corrected and uncorrected rung on the same split and seeds.

The map is affine and monotone, so PR-AUC is *mathematically identical*. Any deviation beyond
floating-point noise means the implementation is wrong — a clip applied before the affine map,
`c` recomputed per scoring day, or the correction leaking into training. This guard costs
nothing and catches the three most likely bugs.

**Guard 2 — no operational regression (must pass).**

> `precision@K`, `recall@K` and `alert_jaccard_wow` unchanged to within floating-point noise,
> for the same reason.

**Secondary, reported but NOT gated — the savings question.**

> Report whether validation `savings` clears the `volume_rank` floor (cycle-3 value 0.6016) at
> the declared cost point, and across the required `false_hold_cost / fraud_loss` sweep
> {0.01, 0.1, 1, 10, 100}. Pre-register the reading, not a pass/fail: **a win on ≥ 3 of the 5
> sweep points on ≥ 3 of 5 seeds is reported as evidence the savings loss was a calibration
> defect; anything less is reported as evidence it was not.**

This is deliberately not a gate, because it is conditional on OPEN item (1) — if the decision
layer is purely rank-based, savings cannot move and a gate on it would fail the rung for a
reason unrelated to its merit.

**Explicitly NOT gated on, and why.**

> `detection_rate_d7 / d14 / d30` and `ttd_median_days` are **excluded from the gate**. With
> ~14 in-window onsets in the validation fold, a detection rate carries a standard error of
> `sqrt(0.25/14) = 13.4 pp` at p = 0.5, and a *difference* between two rungs carries
> `sqrt(2) × 13.4 = 19 pp`, giving a 95% interval half-width of ~37 pp (~30 pp at p = 0.2).
> The cycle-4 spec's "under ~25 pp is not separable" is the right number and it is fatal to any
> TTD-based gate. **Gating adoption on a quantity we cannot measure would be exactly the
> cycle-3 mistake in a new costume.** TTD is reported with its interval and is not a gate.

---

## 7. ADR stub

**ADR-C4-LABEL — Correct the label-observation bias with a two-sided affine posterior map, not
with a PU risk estimator.**

**Context.** Merchant labels arrive after `Exp(21) + U(45,120)` days, 15% never arrive, and
0.3% of clean merchants are labelled fraudulent. At the day-239 training boundary this yields
≈134 genuine labelled positives against ≈588 truly fraudulent merchants (`c ≈ 0.228`,
`ρ₊ ≈ 0.772`, of which only 0.15 is non-reporting and the rest is delay) and 70–118 spurious
positives, so 35–47% of the positive class is wrong while only 1.14% of the negative class is.
Cycle 3 recorded a model that ranks far better than the `volume_rank` floor (PR-AUC 0.836 vs
0.217) yet loses to it on savings (0.4348 vs 0.6016) — the signature of a decision-threshold
defect, and the model's scores are deflated ×4.4 by construction. The cycle-4 spec permits one
new rung and forbids autograd, GPL dependencies, and any read of `drift_onset_at`.

**Decision.** Adopt one rung that leaves training untouched and applies
`p̂ = clip((s − ρ₋)/(c − ρ₋), 0, 1)` to the existing LightGBM rung's score, with `ρ₋ = 0.003`
declared and `c` identified from the observable labelled-positive count and the declared
prevalence: `c = (N₁(T)/M − ρ₋(1−π))/π`. Emit `c`, its standard error, and the clip rate as
diagnostics. Gate adoption on a ≥40% relative ECE reduction on validation with `|ΔPR-AUC| ≤
0.005`; do not gate on TTD.

**Consequences.**
- *Positive:* the only two constants the method needs are known by construction, so the
  correction adds essentially no variance to a 134-positive problem. No new dependency, no
  autograd, ~15 lines, negligible latency. PR-AUC-invariance turns the primary guard into a
  free implementation test.
- *Negative:* the method **cannot** improve any ranking metric — PR-AUC, precision@K,
  recall@K, per-typology recall, `alert_jaccard_wow`, and TTD are unchanged by construction.
  It moves ECE, and it moves savings only if the decision layer contains a cost-derived
  probability threshold (OPEN item 1). If it does not, the rung is a no-op on everything but
  ECE and should be reported as such rather than dressed up.
- *Accepted risk:* the SCAR approximation is imperfect (the delay makes the propensity weakly
  covariate-dependent through latent onset), leaving a known, signed bias that favours
  long-drifted merchants — the wrong direction for TTD. Recorded in `LIMITATIONS.md`, not
  corrected.

**Alternatives rejected.**
- *uPU / nnPU risk estimators* — the risk estimator is portable to LightGBM via a hand-derived
  `fobj` with no autograd, but LightGBM forbids negative `sample_weight` (documented), the
  custom-objective route produces negative leaf Hessians, and the entire correction targets the
  1.14% of the unlabelled set that is secretly positive. Rejected on payoff, not legality. The
  reference implementation `kiryor/nnPUlearning` carries no licence (NOASSERTION) and is
  unusable independently.
- *Chapelle's DFM and its whole lineage (NoDeF, FNW/FNC, FSIW, ES-DFM)* — all require an
  observable event-time origin. Ours is `drift_onset_at`, which is radioactive, and no legal
  surrogate exists because all merchants are present from day 0. Additionally, the delay is
  `Exp(21) + U(45,120)`, CV = 0.29, with a hard 45-day floor; an exponential DFM would place
  35.3% of its mass in a region of zero true probability.
- *Survival regression on time-to-label-availability* — legal with origin = day 0 and event =
  `label_available_at`, and implementable as discrete-time sequential binary classification
  without any library, but 134 events against 28 features gives EPV 4.8, the PH assumption is
  violated by the 45-day floor, and over a fixed 60-day window the result is a monotone
  re-parameterisation. `scikit-survival` is GPL-3.0-or-later and disqualified regardless;
  `lifelines` is MIT and admissible but has nothing to do here.
- *Iterative positive-set cleaning* — the only candidate that could reduce the 35–47% positive
  corruption, but at 252 positive rows it is confirmation-bias-dominated and it preferentially
  deletes weak-signal early-stage fraud, i.e. the merchants TTD exists to measure.
- *PU bagging (Mordelet & Vert)* and *two-step reliable-negative methods* — sound, cheap, and
  aimed at a contaminated negative set that is 98.9% clean.
- *`pulearn`* (BSD-3-Clause, verified) — would work; wraps fifteen lines; adding a dependency
  to avoid fifteen lines fails the standing rule.
- *Neural delayed-feedback and neural PU (Dist-PU, Self-PU, influence-function DFM, DGNN)* —
  **GATED, not dropped.** Blocked by the standing no-autograd ADR. Revisit only if that ADR is
  reopened; none of them is promising at 134 positives regardless.

---

## 8. Where the literature is thin

1. **Simultaneous two-sided noise with a tiny prior is nearly unstudied.** Natarajan et al.
   (2013/2017) give the general class-conditional-noise machinery for `(ρ₊, ρ₋)` and it is
   correct, but essentially every empirical study of it uses balanced or near-balanced classes
   with symmetric or mildly asymmetric rates. Our regime — `π = 0.0147`, `ρ₊ = 0.77`,
   `ρ₋ = 0.003` — is extreme on all three axes at once. The unbiased-estimator variant divides
   by `1 − ρ₊ − ρ₋ = 0.225`, inflating variance ~4.4×, and I found no finite-sample study of
   that estimator at n_pos ≈ 10². The affine posterior correction recommended here sidesteps
   the variance blow-up by acting on the posterior rather than the loss, but I could not find a
   paper that states it in exactly that form for the two-sided case — it is a two-line
   consequence of the transition identity, so it is more likely folklore than novel, but I am
   marking it **not attributable to a single citation**.
2. **PU with contaminated positives is one paper deep.** Jain, White & Radivojac
   (NeurIPS 2016, arXiv:1606.08561) is the closest match to our situation and, as far as I can
   find, the canonical one. Everything downstream of it (Robust PU / noise-negative
   self-correction, KDD 2023, arXiv:2308.00279; PU-via-noisy-labels, arXiv:2103.04685) is
   neural and assumes thousands of positives. There is no tree-based, small-sample treatment.
3. **PU at ~100 positives is outside the empirical envelope of every method in §3.3.** nnPU is
   advertised as robust with "limited P data" and its experiments use MNIST/CIFAR scale —
   thousands of positives. TIcE, KM2, DEDPUL and BBE all report degradation as the labelled
   set shrinks and none reports at n_pos ≈ 10² with π ≈ 10⁻². Anyone citing "nnPU works with
   few positives" at our scale is extrapolating three orders of magnitude.
4. **Delayed feedback without an observable origin has no literature at all.** The entire
   canon is written for ad conversion, where the click timestamp is free. I could find no
   treatment of the case where the clock start is latent and legally unreadable. The nearest
   things are the payments-specific papers ("Mind the Gap", KDD 2025; arXiv:2409.10111) and
   they describe the problem without solving this version of it. Several very recent
   arXiv preprints surfaced in search that appear to address exactly this (censored/corrupted
   feedback in card networks, causal label recovery in payment networks) — I did not verify
   their contents and am **not** citing them; see §10.
5. **Nobody reports what a PU correction does to a *cost* metric.** The literature evaluates on
   accuracy, F1, AUC, and occasionally calibration. The one place a monotone posterior
   correction can possibly matter — a downstream decision rule with an asymmetric cost model
   and a hard capacity budget — is exactly where the literature stops. That gap is why §6's
   savings reading is pre-registered as a *reading* and not as a gate.

---

## 9. Contrarian view — argue against first place

**The strongest case against the recommendation is that it is arguably a bug fix wearing a
rung's clothing, and that at 134 positives no label-side correction is separable from noise.**

*Argument A — it is provably a no-op on every ranking metric.* The map is affine and monotone.
PR-AUC, ROC-AUC, precision@K, recall@K, per-typology recall, `alert_jaccard_wow`, `TTD` and
every detection rate are *invariant* under it. If the decision layer assigns PASS/REVIEW/HOLD
by taking the top-K scores each day — which the capacity constraint `alerts_per_day ≤ K`
strongly suggests — then `savings` is invariant too, and the rung moves exactly one number in
the entire metric suite: ECE. Adopting a "new rung" that changes one metric by construction and
cannot change any other is a defensible bookkeeping decision and an indefensible research
claim. **This is OPEN item (1) and it must be settled by reading the decision layer before
pre-registration, not after.** If the decision layer is rank-only, the correct move is to
withdraw this recommendation and adopt nothing.

*Argument B — the savings loss probably is not a calibration problem.* `volume_rank` alerts on
the K largest merchants by GMV and wins on a loss-weighted metric. It wins because the cost
function is `true_loss_amount_inr`-weighted and loss scales with GMV, so the floor is ranking
by *expected loss* while every model ranks by *probability*. The fix for that is to rank by
`p̂ × E[loss | fraud]`, not to rescale `p̂`. And a uniform rescale of `p̂` does not change
`p̂ × loss` rankings either. So the honest statement is: **this rung supplies a calibrated
probability, which is a necessary input to an expected-loss decision rule, and is not itself
that rule.** If the expected-loss rule is what actually closes the gap, the rung is
infrastructure and the credit belongs elsewhere.

*Argument C — `c` is estimated from the same 134 positives whose scarcity is the problem.*
`sd(c) = 0.026` on `c ≈ 0.20`, i.e. ±11%. That is small relative to the ×4.4 correction, which
is why I ranked it first — but it means the corrected posterior carries an ±11% multiplicative
band that no downstream number acknowledges. And the estimator disagrees with the
ground-truth-derived `c` (0.197 vs 0.228) by more than one standard error, which is either the
spurious-maturation uncertainty (OPEN item 2) or a sign the SCAR approximation is worse than
§2 concludes.

*Argument D — the SAR residual points the wrong way for the cycle's headline metric.* §2
concedes that the labelling propensity rises with time-since-onset, and so does the true
posterior. A model trained on this data is systematically better at merchants that drifted long
ago. Cycle 4 exists to make TTD measurable. Correcting the *scale* of the posterior while
leaving that bias in place produces a well-calibrated model that is still structurally late,
and a well-calibrated late model may read as a *worse* result than an uncalibrated one, because
now the lateness is visible and unattributable to miscalibration.

*Argument E — the honest move may be to report the impossibility.* With ~14 evaluable merchants
per fold, a 30 pp difference in detection rate is at the edge of resolution and a 10 pp
difference is invisible. Every method in §4 differs from every other by less than that on the
metrics anyone cares about. **A defensible cycle-4 outcome is: "we surveyed the label-constraint
literature, established that the delayed-feedback canon is inapplicable because our event-time
origin is unobservable, established that PU learning corrects a 1.14% contamination while the
real corruption is a 35–47% error rate in the positive class, established that the only
admissible correction is monotone and therefore cannot move a ranking metric, and adopted no
new rung."** That outcome costs one cycle and buys a genuinely publishable negative finding.
It is the second-best answer here and it is close.

**Why I still rank the recommendation first:** because it is the only candidate whose cost is
near zero. Fifteen lines, no dependency, no training change, no latency, and an
adoption gate that doubles as an implementation test. If Argument A resolves against it, the
rung is withdrawn at a cost of an afternoon. Every other candidate in §4 costs days and is
equally unseparable at the power ceiling. When nothing is measurable, prefer the cheap correct
thing over the expensive plausible thing.

---

## 10. References

Verified at source unless marked. arXiv identifiers and DOIs were resolved during this survey;
licences were read from the named `LICENSE`/`COPYING` file, the GitHub API `license.spdx_id`
field, or the PyPI project page.

**Delayed feedback**

1. Chapelle, O. (2014). *Modeling delayed feedback in display advertising.* KDD '14.
   DOI [10.1145/2623330.2623634](https://dl.acm.org/doi/10.1145/2623330.2623634). — The
   exponential-delay DFM; the origin of the "too early to tell" discard rule.
2. Yoshikawa, Y. & Imai, Y. (2018). *A nonparametric delayed feedback model for conversion
   rate prediction.* AISTATS 2018. — **UNVERIFIED** venue/pages; the work is real and is cited
   as NoDeF throughout the ES-DFM line, but I did not open a canonical record.
3. Ktena, S. I., Tejani, A., Theis, L., Myana, P. K., Dilipkumar, D., Huszár, F., Yoo, S. &
   Shi, W. (2019). *Addressing delayed feedback for continuous training with neural networks in
   CTR prediction.* RecSys '19. [arXiv:1907.06558](https://arxiv.org/abs/1907.06558);
   DOI [10.1145/3298689.3347002](https://dl.acm.org/doi/10.1145/3298689.3347002). — FNW / FNC.
4. Yasui, S., Morishita, G., Fujita, K. & Shibata, M. (2020). *A feedback shift correction in
   predicting conversion rates under delayed feedback.* WWW '20, pp. 2740–2746.
   [arXiv:2002.02068](https://arxiv.org/abs/2002.02068);
   DOI [10.1145/3366423.3380032](https://doi.org/10.1145/3366423.3380032). — FSIW.
5. Yang, J.-Q., Li, X., Han, S., Zhuang, T., Zhan, D.-C., Zeng, X. & Tong, B. (2021).
   *Capturing delayed feedback in conversion rate prediction via elapsed-time sampling.*
   AAAI 2021, 35(5):4582–4589. [arXiv:2012.03245](https://arxiv.org/abs/2012.03245). — ES-DFM.
   Reference code `ThyrixYang/es_dfm` carries **no licence (NOASSERTION)** — unusable.
6. Wang, Y. et al. (2022). *Asymptotically unbiased estimation for delayed feedback modeling
   via label correction.* WWW '22.
   DOI [10.1145/3485447.3511965](https://dl.acm.org/doi/10.1145/3485447.3511965);
   [arXiv:2202.06472](https://arxiv.org/abs/2202.06472).
7. *Mind the gap: delayed label bias-variance tradeoffs in predicting likelihood of
   nonpayment.* KDD '25 v.2.
   DOI [10.1145/3711896.3737247](https://dl.acm.org/doi/10.1145/3711896.3737247). — Applied
   delayed-label evidence in a payments-adjacent setting. Author list **UNVERIFIED**.
8. *Evaluating the efficacy of instance incremental vs. batch learning in delayed label
   environments: an empirical study on tabular data streaming for fraud detection.*
   [arXiv:2409.10111](https://arxiv.org/abs/2409.10111). Author list **UNVERIFIED**.
9. Search also surfaced several 2026 arXiv preprints on censored/corrupted feedback in card
   payment networks (e.g. 2605.27557, 2605.29272). **I did not open or verify these and am not
   relying on them.** Flagged so a reviewer knows they exist and were deliberately excluded.

**Positive-unlabelled learning**

10. Liu, B., Dai, Y., Li, X., Lee, W. S. & Yu, P. S. (2003). *Building text classifiers using
    positive and unlabeled examples.* ICDM '03. — The spy technique / S-EM; the two-step family.
11. Elkan, C. & Noto, K. (2008). *Learning classifiers from only positive and unlabeled data.*
    KDD '08, pp. 213–220.
    DOI [10.1145/1401890.1401920](https://dl.acm.org/doi/10.1145/1401890.1401920). — SCAR, the
    `P(ỹ=1|x) = c·P(y=1|x)` identity, and the case-weighting scheme. **The basis of §5.**
12. Mordelet, F. & Vert, J.-P. (2014). *A bagging SVM to learn from positive and unlabeled
    examples.* Pattern Recognition Letters 37:201–209.
    DOI [10.1016/j.patrec.2013.06.010](https://dl.acm.org/doi/10.1016/j.patrec.2013.06.010).
13. du Plessis, M. C., Niu, G. & Sugiyama, M. (2014). *Analysis of learning from positive and
    unlabeled data.* NeurIPS 2014. — uPU; requires a non-convex loss to avoid a superfluous
    penalty.
14. du Plessis, M. C., Niu, G. & Sugiyama, M. (2015). *Convex formulation for learning from
    positive and unlabeled data.* ICML 2015, PMLR 37:1386–1394.
    [proceedings.mlr.press/v37/plessis15.html](https://proceedings.mlr.press/v37/plessis15.html).
15. Kiryo, R., Niu, G., du Plessis, M. C. & Sugiyama, M. (2017). *Positive-unlabeled learning
    with non-negative risk estimator.* NeurIPS 2017.
    [arXiv:1703.00593](https://arxiv.org/abs/1703.00593). — nnPU. Reference implementation
    `kiryor/nnPUlearning` carries **no licence (NOASSERTION)**, verified via the GitHub API.
16. Bekker, J. & Davis, J. (2020). *Learning from positive and unlabeled data: a survey.*
    Machine Learning 109:719–760.
    DOI [10.1007/s10994-020-05877-5](https://link.springer.com/article/10.1007/s10994-020-05877-5);
    [arXiv:1811.04820](https://arxiv.org/abs/1811.04820). — The SCAR/SAR taxonomy used in §2.
17. Bekker, J. & Davis, J. (2018). *Estimating the class prior in positive and unlabeled data
    through decision tree induction.* AAAI 2018, 32(1).
    DOI [10.1609/aaai.v32i1.11715](https://doi.org/10.1609/aaai.v32i1.11715). — TIcE.
18. Ramaswamy, H., Scott, C. & Tewari, A. (2016). *Mixture proportion estimation via kernel
    embeddings of distributions.* ICML 2016, PMLR 48.
    [arXiv:1603.02501](https://arxiv.org/abs/1603.02501). — KM1 / KM2.
19. Ivanov, D. (2020). *DEDPUL: difference-of-estimated-densities-based positive-unlabeled
    learning.* ICMLA 2020. [arXiv:1902.06965](https://arxiv.org/abs/1902.06965). Repository
    `dimonenka/DEDPUL` is **MIT** (verified via GitHub API).
20. Garg, S., Wu, Y., Smola, A., Balakrishnan, S. & Lipton, Z. C. (2021). *Mixture proportion
    estimation and PU learning: a modern approach.* NeurIPS 2021 (spotlight).
    [arXiv:2111.00980](https://arxiv.org/abs/2111.00980). — BBE, CVIR, TED^n.
21. Hsieh, Y.-G., Niu, G. & Sugiyama, M. (2019). *Classification from positive, unlabeled and
    biased negative data.* ICML 2019, PMLR 97.
    [arXiv:1810.00846](https://arxiv.org/abs/1810.00846). — PUbN.
22. Jain, S., White, M. & Radivojac, P. (2016). *Estimating the class prior and posterior from
    noisy positives and unlabeled data.* NeurIPS 2016.
    [arXiv:1606.08561](https://arxiv.org/abs/1606.08561). — **The closest published match to
    this project's two-sided situation.**
23. Jain, S., White, M. & Radivojac, P. (2017). *Recovering true classifier performance in
    positive-unlabeled learning.* AAAI 2017.
    [arXiv:1702.00518](https://arxiv.org/abs/1702.00518). — Relevant to reporting metrics under
    PU contamination, which the eval harness currently does not adjust for.
24. Zhao, Y., Zhang, M., Zhang, C., Chen, W., Ye, N. & Xu, M. (2022/2024). *A boosting algorithm
    for positive-unlabeled learning* / *A boosting framework for positive-unlabeled learning.*
    [arXiv:2205.09485](https://arxiv.org/abs/2205.09485); Statistics and Computing,
    DOI [10.1007/s11222-024-10529-y](https://link.springer.com/article/10.1007/s11222-024-10529-y).
    — AdaPU. Cited in §3.4 as evidence that PU risk estimators do **not** drop cleanly into an
    off-the-shelf booster.
25. Teisseyre, P., Furmańczyk, K. & Mielniczuk, J. (2024). *Verifying the selected completely at
    random assumption in positive-unlabeled learning.*
    [arXiv:2404.00145](https://arxiv.org/abs/2404.00145). — A cheap SCAR test; noted in §2 and
    not adopted.

**Label noise, both directions**

26. Natarajan, N., Dhillon, I. S., Ravikumar, P. & Tewari, A. (2013). *Learning with noisy
    labels.* NIPS 2013, pp. 1196–1204.
    [dblp:conf/nips/NatarajanDRT13](https://dblp.org/rec/conf/nips/NatarajanDRT13.html). —
    Class-conditional `(ρ₊, ρ₋)`, the unbiased-estimator and α-weighted surrogate methods. **The
    transition identity inverted in §5.**
27. Natarajan, N., Dhillon, I. S., Ravikumar, P. & Tewari, A. (2017). *Cost-sensitive learning
    with noisy labels.* JMLR 18(155):1–33.
    [jmlr.org/papers/v18/15-226.html](https://jmlr.org/papers/v18/15-226.html). — Journal
    version, with the cost-sensitive treatment used in §4 row 1's reasoning.
28. *Robust positive-unlabeled learning via noise negative sample self-correction.* KDD 2023.
    DOI [10.1145/3580305.3599491](https://dl.acm.org/doi/10.1145/3580305.3599491);
    [arXiv:2308.00279](https://arxiv.org/abs/2308.00279). Author list **UNVERIFIED**.
29. *A novel perspective for positive-unlabeled learning via noisy labels.*
    [arXiv:2103.04685](https://arxiv.org/abs/2103.04685). Author list **UNVERIFIED**.

**Survival / time-to-event**

30. Suresh, K., Severn, C. & Ghosh, D. (2022). *Survival prediction models: an introduction to
    discrete-time modeling.* BMC Medical Research Methodology 22:207.
    DOI [10.1186/s12874-022-01679-6](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-022-01679-6).
    — Discrete-time survival as per-period binary classification, which is why §3.2(b) needs no
    survival library.

**Libraries and licences — all checked at source**

| library | version checked | licence | source of check | verdict |
|---|---|---|---|---|
| `lightgbm` | ≥ 4.3 (pinned) | **MIT** | `microsoft/LightGBM/LICENSE` | admissible (already pinned) |
| `numpy`, `scipy`, `scikit-learn` | pinned | **BSD-3-Clause** | already in the pinned set | admissible |
| `lifelines` | 0.30.3 | **MIT**, © 2017 Cameron Davidson-Pilon | `CamDavidsonPilon/lifelines/LICENSE` | admissible — **but not needed** |
| `scikit-survival` | 0.24.x | **GPL-3.0-or-later** | `sebp/scikit-survival/COPYING` + GitHub API | **DISQUALIFIED** |
| `pulearn` | 0.2.0 (2026-03-14) | **BSD-3-Clause** | PyPI project page + GitHub API | admissible — **rejected on the no-new-dependency rule** |
| `dimonenka/DEDPUL` | repo | **MIT** | GitHub API `license.spdx_id` | admissible — not needed |
| `kiryor/nnPUlearning` | repo | **NOASSERTION (no licence)** | GitHub API | **unusable** |
| `ThyrixYang/es_dfm` | repo | **NOASSERTION (no licence)** | GitHub API | **unusable** |
| `torch`, `transformers` | — | — | — | **GATED** by the standing no-autograd ADR |

**Documented behaviour relied on**

31. LightGBM Python API, `lightgbm.Dataset` / `set_weight`: *"Weight for each instance. Weights
    should be non-negative."*
    [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Dataset.html)
    — the fact that rules out implementing uPU through `sample_weight` (§3.4).
