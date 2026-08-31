# Rakshak — post-onboarding merchant risk sentinel

Submission for the **Razorpay AI Buildathon 2026, Track 02 (AI Risk Manager)**. Solo build,
4-day window. This document is written for one reader: a Razorpay Head of Risk Operations on the
panel. It states what was measured, how, and where the pre-registered claim failed.

**Architecture and design rationale live in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — the
four-layer shape, the ADRs, and what was cut versus rejected. This file does not repeat that
diagram; it reports results and their provenance.

---

## The one-line verdict

> **K2 FAILED.** The pre-registered claim — `hmm` beats the static rule engine `rules` by ≥20%
> relative on the Bahnsen savings score — does not hold on the held-out test window. The measured
> margin is **+5.9% relative**, and it holds at *no* point across the full swept cost-asymmetry
> range. Per `00-charter.md` §3's kill criterion, this result is reported, not tuned away, and the
> pitch is built on explainability and the cost frontier instead of a savings win.

A second, larger finding sits underneath it: on this same test window, a **uniform random score
beats every fitted model** on the savings metric. Every savings number in this document is
therefore printed net of that floor, with PR-AUC beside it — an unadorned savings figure is not a
claim about detection on this project's numbers.

---

## What this is

Razorpay's **Vulcan** scores every *transaction* in milliseconds. Razorpay's **Bumblebee** reviews
every *merchant* once at onboarding. Nothing watches a merchant who was already cleared drift from
good to bad over the following weeks — so bust-outs, laundering endpoints, category drift and
refund collusion surface only when chargebacks land 45–120 days later. Rakshak runs a per-merchant
Hidden Markov Model over the transaction stream, updates a belief over four latent risk states with
every new window of transactions, and converts that belief into pass / review / hold under
per-merchant cost economics and a fixed analyst-hour budget.

> **Sequence-layer metrics are measured on synthetic merchant streams with injected typologies;
> the generator is in this repo.** The decision layer is additionally validated on BAF (Feedzai,
> NeurIPS 2022), a public benchmark derived from real bank data.

That sentence is required verbatim by this project's non-negotiables (`CLAUDE.md`) and is repeated
at the top of every file under `results/`. Read it before any number below.

---

## T-0011 — the verdict, on the held-out test window

**Produced by** `python -m rakshak.eval.verdict --seed 42` · **seed 42** · **split** `test`
(days 210–269, unlocked once with ticket T-0011) · **100 merchants, 20 truly bad (20.0%
prevalence)** · **review budget K = 5** (0.40 analyst-hours) · **cited central cost asymmetry
13.1** INR FP-cost per INR 100 of fraud loss, swept 0.7–146.9. Full table, sweep and ceilings:
[`results/verdict.md`](results/verdict.md).

| model | savings | savings − `random` | PR-AUC | precision@5 | Brier | median lag (days) | flagged frac |
|---|---|---|---|---|---|---|---|
| random | 0.5365 | 0.0000 | 0.2449 | 0.2000 | 0.3069 | n/a | 0.00 |
| rules | 0.4889 | -0.0475 | 0.5547 | 1.0000 | 0.1358 | 5.0 | 0.65 |
| gbdt | 0.5069 | -0.0296 | 0.6523 | 1.0000 | 0.1453 | 10.0 | 0.65 |
| hmm | 0.5176 | -0.0188 | 0.3347 | 0.4000 | 0.4321 | 11.0 | 0.75 |

### K2 fired: FAIL

> `00-charter.md` §2 (amended 2026-08-28, before any swept number existed): *"Rakshak beats a
> static velocity/refund-ratio rule engine by ≥20% relative on the Bahnsen savings score at the
> cited central cost asymmetry ... with the relative improvement reported across the full
> plausible asymmetry range and the boundary at which the claim fails stated explicitly."*

At the central asymmetry (13.1), `hmm` savings (0.5176) beats `rules` savings (0.4889) by an
absolute 0.0287 — **+5.9% relative**, against the pre-registered **≥20%** bar. **There is no
boundary asymmetry to report because the claim holds at no swept point** across the full derived
range 0.7–146.9; the best point in the whole sweep is +14.3% relative, at 146.9. The weaker
question — does `hmm` beat `rules` at all — crosses positive between asymmetry 2.6 and 5.1 on test
(between 18.5 and 36.2 on validate), and the margin is **non-monotone** across the sweep (+5.9% at
13.1, dipping to +3.7% at 19.6, then rising). None of this was tuned after the fact: `00-charter.md`
§2's bar was pre-registered a day before T-0007b's sweep first ran.

Per `CLAUDE.md`'s first non-negotiable — *"if a baseline beats the HMM, report that the baseline
beat the HMM"* — the ranking and calibration story is also unambiguous. On PR-AUC, `hmm` (0.3347)
trails both `rules` (0.5547) and `gbdt` (0.6523); on Brier, `hmm` (0.4321) is the worst of the four
rows. Its one measured advantage is coverage: it flags 0.75 of truly-bad merchants against
`gbdt`'s and `rules`'s 0.65.

### The finding that outranks the verdict: `random` wins the primary metric

**A uniform random score posts 0.5365 savings on this window — the highest of any row — while
ranking at PR-AUC 0.2449, i.e. at the prevalence, with no discriminating power whatsoever.**
Savings net of that floor is negative for every fitted model: `rules` -0.0475, `gbdt` -0.0296,
`hmm` -0.0188. Nothing about any model produced the random row's score; the cost matrix did. When
the per-merchant false-positive cost is small relative to the realised-loss stock, a merchant lands
on the correct side of its own threshold most of the time whatever the score is. This is
`07-math.md` §6's AP-06 guard arriving as a measurement, first seen on `validate` at T-0007b
(`random` 0.0051 *behind* `rules` there — on `test` it is *ahead of everything*).

**Consequence stated once and applied everywhere in this document: no savings number is quoted
without PR-AUC beside it, and no headline of the form "Rakshak saves X%" is made without
subtracting the random floor.**

Read that alongside BAF, below: at BAF's realistic 1.47% prevalence, `random` scores **-28.2169** —
catastrophically negative, not a near-miss. That points at this generator's `FRAUD_MERCHANT_RATE =
0.20` (chosen for per-typology sample size, not realism), not at the savings metric itself — the
metric's sensitivity to a uniform baseline is a property of prevalence, and this project's synthetic
prevalence is far above a real merchant book's.

**In INR** (`results/verdict.md` §FR-019, Cost_l = INR 555,961 on this split): every model saves
money against holding/passing everyone by default, but **relative to `random`, every model on this
window is negative** — `rules` −26,412, `gbdt` −16,443, `hmm` −10,475. No model here saves money
relative to scoring merchants at random.

---

## Ablations (FR-018) — one component is close to decoration, and it is not the one expected

**Produced by** `python -m rakshak.eval.ablations --seed 42` · same split, seed, population as
above · 6 fits. Full table: [`results/ablations.md`](results/ablations.md).

| ablation | Δ savings | Δ PR-AUC | Δ precision@5 | Δ Brier |
|---|---|---|---|---|
| HMM off (= `gbdt` path, same features) | -0.0107 | **+0.3176** | **+0.6000** | **-0.2868** (better) |
| graph features (FR-008) off — HMM | -0.1006 | -0.0390 | 0 | -0.0144 |
| graph features (FR-008) off — `gbdt` | +0.0047 | -0.1217 | -0.2000 | -0.0038 |
| within-merchant standardisation (FR-007) off — HMM | -0.0318 | -0.0475 | 0 | **+0.1729** (worse) |
| within-merchant standardisation (FR-007) off — `gbdt` | **-0.0001** | **+0.0032** | **0** | -0.0081 |
| empirical-Bayes shrinkage (ADR-0006) | **not measured** — T-0008 cut | | | |
| NSGA-II vs. grid search (ADR-0004) | **not measured** — T-0009 cut | | | |

**The incumbent out-ranks the proposal on every ML metric, and the row that is closest to
decoration is this project's own headline modelling decision.** Swapping the HMM for LightGBM on
the identical feature pipeline degrades savings by only 0.0107 while improving PR-AUC by 0.3176,
precision@5 by 0.60 and Brier by 0.2868 — the incumbent wins the ranking argument outright.
**FR-007 within-merchant standardisation — described in `docs/ARCHITECTURE.md` as "the single most
important modelling decision in the project" — moves LightGBM's savings by -0.0001 and its PR-AUC
by +0.0032, i.e. it is nearly free for the incumbent** (it is load-bearing for the HMM: turning it
off costs the HMM 0.0475 PR-AUC and 0.1729 Brier). The graph-derived scalars (ADR-0002's CPU
stand-in for a GNN) are not decoration for either model.

**No sequence-aware baseline other than the HMM was ever measured** — the BOCPD changepoint
baseline (T-0010) was cut in the 2026-08-28 re-plan — so whether any margin here comes from
*sequence modelling* or from *this particular HMM* is left open and stated as open, not answered
by implication.

---

## The K1 story — recovering latent risk states

**Pinned by** `python -m pytest tests/test_hmm_recovery_fullscale.py --seed 42` against the
full-scale generator population (`docs/adr/ADR-0009-k1-label-informed-hmm.md` records the same
numbers as the decision's Context). These are regression-locked assertions, not a `results/` table
`make eval` writes — the K1 gate was decided once, on `train`/`validate`, before T-0006 built a
scoring path, and is re-verified on every test run rather than re-measured.

FR-013 required four-way latent-state ARI > 0.5 against ground truth. The unsupervised HMM (T-0004)
measured **0.091**. The load-bearing number next to it is the **oracle-parameterised ceiling of
0.404** (0.381 on the validate merchant group) — with HMM parameters read directly off ground
truth, the gate is unreachable by any correctly-implemented HMM on this generator. Root cause:
per-state overlap. The early-warning `RAMP` state sits **1.19σ** from `HEALTHY`, which holds ~90%
of windows — the gate failed precisely on the state the product exists to catch.

A literature survey (`project-context/12-lit-survey-k1.md`) established that ARI is the wrong index
for a 90/6.4/3.4/2.2 reference distribution (Romano et al., JMLR 17, 2016). FR-013 was amended —
dated, cited, after the gate failed, not before — to AMI + per-state recall + binary PR-AUC +
detection lag, **with ARI and the oracle ceiling retained and reported permanently** rather than
replaced.

T-0004b's partially-supervised, label-informed fit (ADR-0009) then measured: ARI 0.134 → 0.319, AMI
0.102 → 0.218, binary PR-AUC 0.109 → 0.327 — roughly 85% of the way to the ceiling and never above
it.

**The unflattering finding that must reach this document: supervision made `RAMP` recall *worse* —
0.328 → 0.234 — while doubling every headline metric.** Labels help rare *separable* states, not
rare *overlapping* ones. The configuration that wins on aggregate metrics is the one that goes
blind on the state this project exists to catch, and it ships anyway, with both configurations
reported side by side. A pre-registered `RAMP`-recall ≥0.35 bar was recorded before this
measurement and failed at 0.234; it is committed in the test suite as a strict `xfail` and left
there rather than removed.

---

## Explainability — the Viterbi path, FR-014

Every flagged merchant gets a machine-generated reason string naming the state transitioned into,
the date, and the top-3 emission features by contribution, in merchant-facing language — derived
directly from the Viterbi decode, not from a post-hoc surrogate.

**Produced by** `python -m rakshak.explain.reasons --seed 42` · split `test` · model `hmm` ·
committed at [`results/reasons.json`](results/reasons.json) (golden-file tested: same seed → the
same file, byte for byte, `tests/test_reasons.py`). Of 100 merchants, **56 were flagged, 15 of
those truly bad, 41 healthy; the Viterbi decode disagreed with the raw flag on 3 of them** and that
disagreement is recorded in the file rather than reconciled away.

One reason string, taken verbatim from `results/reasons.json` (merchant M00131, flag day 237 / 26
August 2026, no early-warning claim implied — flag day and decode day coincide here):

> *"On 26 August 2026 this account moved into a pattern we flag as sustained-abnormal-activity. The
> measurements that moved it there, each compared against this account's own trading history rather
> than against other merchants: share of repeat payers ran far above this account's own baseline
> (+8.18 SD); overlap between this week's payers and last week's ran far above this account's own
> baseline (+5.41 SD); share of first-time payers ran far below this account's own baseline (-6.84
> SD). What would resolve this: settlement-level invoices for the flagged period, and contact
> details for the payers behind the largest transactions."*

Every deviation is stated **against the merchant's own baseline**, in standard deviations — never
against the population — for the same reason FR-007 standardises within merchant: a jeweller and a
grocer live three orders of magnitude apart, and comparing either to the other is the 2008-era
cardholder-HMM failure mode this project exists to avoid.

**What this claims, and what it does not.** The closest deployed system in the published record
(Zhang et al., arXiv:2607.17586, a production money-mule detector — LightGBM over 280 features,
TreeSHAP attribution, LLM narration, reported analyst yield 61% → 89%, no sequence model) **is an
existing, deployed answer to "can I explain this decision to the merchant"**. This document does
not claim otherwise. What it claims is narrower and, on inspection of that paper, still true: the
Viterbi path is a **different kind** of explanation — intrinsic and temporal (the explanation *is*
the state the model already believed the merchant was in, decoded jointly with a specific date of
transition) rather than post-hoc and static (a feature-attribution snapshot of one classifier
output, narrated afterward by a separate model). It does not claim to be a *better* explanation;
that comparison is unmeasured and unmeasurable here — this project has no analysts and no user
study, and arXiv:2607.17586's own evidence for reduced cognitive load is theirs, not this repo's.

---

## Recall by typology (FR-005) — where the model is meant to fail

**Produced by** `python -m rakshak.eval.typology --seed 42` · same test-window population as the
verdict · **4 truly-bad merchants per typology** — every cell below is a proportion over about
four merchants, reported with Wilson 95% intervals because the sample is this small, not despite
it. Full table: [`results/typology_recall.md`](results/typology_recall.md).

`SLOW_RAMP` is FR-005's deliberately adversarial typology: a monotone, changepoint-free drift built
to defeat exactly the state-transition logic this project is made of. `CLAUDE.md` forbids tuning it
away, and its recall is reported degraded here rather than curated out:

| metric | `hmm` on `SLOW_RAMP` | `hmm` on the other four typologies |
|---|---|---|
| acted on (policy chose REVIEW/HOLD) | 0.75 (3/4) | 1.00 |
| **own-score flagged, unconstrained by capacity** | **0.50 (2/4)** | **1.00** |

The sharper degradation is in the model's own score, unconstrained by the review-capacity policy:
`hmm` flags only **2 of 4** `SLOW_RAMP` merchants on its own threshold, against 1.00 (16/16) on the
other four typologies combined. `gbdt` degrades further on this measure — **0 of 4 flagged** on
`SLOW_RAMP`. **A delta at or above zero here is not evidence the adversarial typology was solved**:
with four merchants per cell, none of these deltas are separable from sampling noise at any
conventional significance level, and the table is published with that stated rather than narrowed
to look better.

---

## Detection lag — corrected, and a banned claim

**Produced by** `python -m rakshak.eval.lag_probe --seed 42` · splits `validate` and `test`. Full
probe, including the leakage clearance: [`results/lag_probe.md`](results/lag_probe.md).

T-0011 found and fixed a reporting defect: `gbdt` and `hmm` were attributing a flag to the *start*
day of the 7-day window whose evidence raised it, crediting them with up to 6 days of earliness
they never had, while `rules` had always reported the window's *end* day. The two conventions were
printed in the same column with no note. **Corrected at source** (both scorers now attribute to the
window's last day, the first day on which the flag could actually have fired): on `validate`, the
median lag for both `gbdt` and `hmm` moves from -1.0 d (superseded — read at the time as detection
*before* onset) to **5.0 d**. On `test`, it moves from 4.0 d (superseded) to **10.0 d** for
`gbdt`, and from 5.0 d (superseded) to **11.0 d** for `hmm`. `rules` is unaffected (already
end-attributed).

A merchant-clustered permutation test over pre-onset windows (n = 499 relabelling draws) confirms
this was aliasing, not leakage: largest per-feature |AUC − 0.5| is 0.173 (validate) / 0.159 (test)
against null 95th percentiles of 0.222 / 0.215 — **p = 0.164 / 0.310**, underpowered but not
significant, and not a reason to trust the result more than the confidence interval allows.

> **The phrase "Rakshak detects N days before the fraud starts" is not a claim this repo can make,
> and it does not appear anywhere in this document, the architecture doc, or any file under
> `results/`.** Under the corrected attribution, no model here detects before onset; the honest
> claim is about how many days *after* onset a merchant is flagged, and how many bad merchants are
> flagged at all.

---

## BAF validation of the decision layer (FR-021)

**Produced by** `python -m rakshak.eval.baf --seed 42` · BAF Base, 1,000,000 rows, BAF's own
native temporal split, **month 7 reported** (96,843 applications, **1.47% prevalence**), CC
BY-NC-SA 4.0, git-ignored and not vendored. Full table and both assumptions stated:
[`results/baf_validation.md`](results/baf_validation.md).

| model | savings | PR-AUC | precision@K | Brier | held |
|---|---|---|---|---|---|
| random | -28.2169 | 0.0143 | 0.0137 | 0.3340 | 4033 |
| credit_risk_score | -5.2810 | 0.0403 | 0.0560 | 0.3200 | 469 |
| gbdt | **+0.0294** | **0.2179** | **0.1436** | **0.0129** | 1 |

`gbdt` is the only positive row at **every one of the ten swept cost asymmetries** (5,497 to
519,634), so the ordering is not an artefact of one cost matrix. **BAF is bank account-opening
applications** — no amount, no timestamp, no payer, no merchant, no sequences — so the HMM does not
and cannot run here. This validates the decision layer (Bayes Minimum Risk, the capacity
constraint, the savings score) against real bank data and real temporal drift; it says nothing
about the sequence layer, and nothing here should be read as grounding the generator.

The native asymmetry on BAF (61,368) never falls inside the range this project sweeps on the
synthetic split (0.7–146.9) — BAF's own credit-limit units against this project's absolute INR
support and review costs put false positives in an extreme corner where the correct policy is to
hold almost nobody, and BMR does exactly that. **What BAF validates is that the decision layer does
the economically correct thing when false positives are overwhelmingly expensive — not the
review-versus-hold trade-off at this project's own operating point**, which no public dataset
available to this project puts real money on both sides of.

---

## Scope and Safety

**Defense only.** Rakshak is a detector. Every component in this repository exists to identify risk
on a merchant portfolio a payment aggregator already owns, and to price the cost of acting on it.
There is no component that probes, enumerates, or exploits any payment system. Any Razorpay API
usage, if it appears at all, is test-mode only.

**The synthetic generator is an evaluation artifact, not a fraud toolkit (FR-006).**
`src/rakshak/generator/` produces labelled synthetic merchant transaction streams so detection can
be *measured* against ground truth this project controls. It writes rows to a local parquet file.
It has no payment-system client, no credential handling, and makes no network calls.

Its five "typologies" are coarse statistical caricatures pitched at the level a risk analyst would
sketch on a whiteboard — a volume ramp, a flat payer graph, a ticket-size shift, a refund rate. They
encode no operational tradecraft and produce nothing usable against a live system. Read as an
attack recipe, the module says only that fraud changes a merchant's statistics, which is the
premise of every fraud-detection paper ever published. The four realistic typologies plus the
deliberately-hard fifth (`SLOW_RAMP`) exist to be reported against, including — see above — where
Rakshak fails to catch them.

**Explicitly out of scope:** deepfake or KYC document analysis, RTO/COD return fraud (Razorpay's
Thirdwatch has owned this since 2019), and any capability that could be repurposed offensively. See
`03-landscape.md`.

### Regenerating the synthetic dataset

```
python -m rakshak.generator --seed 42
```

Writes `data/synthetic/transactions.parquet` and `data/synthetic/state_paths.parquet`
(git-ignored). At the defaults — 500 merchants, 270 days, seed 42 — this is ~747,000 transactions
in a few seconds on a laptop CPU.

### How synthetic is the synthetic data? — the measured answer

`results/calibration_gap.md` (`python -m rakshak.data.profile --seed 42`, T-0015) compares eleven
marginals of the generator against **Online Retail II** (UCI 502, CC BY 4.0) — a real transaction
stream, hashed and manifested at `data/external/online_retail_ii.manifest.json`.

**Of the eight ratio-scale marginals, five diverge by 1.9x or more and four by 5x or more; the
widest is 32.5x.** One divergence is structural, not parametric: real daily transaction counts are
over-dispersed (Fano factor 12.3) while the generator's Poisson process has a Fano factor of 1.0 by
construction — no choice of rate closes that gap.

**What the profile cannot inform, and what therefore remains entirely synthetic and uncalibrated:**
latent risk states and their transitions, merchant-level fraud labels and prevalence, the
chargeback rate, the cross-merchant payer graph, MCC/category structure and absolute ticket size,
and every typology dynamic. No public merchant-sequence dataset with merchant-level risk labels
exists (`06-requirements.md`, ADR-0007) — a literature survey (T-0023,
`project-context/15-lit-survey-drift-detection.md`) went looking specifically for one and found
that the closest prior art on post-onboarding merchant drift sits almost entirely in card-network
and acquirer patent filings rather than published papers, which is itself the honest state of this
problem's literature rather than a gap in this project's search.

---

## Reproducing every number in this document

```
make setup
make eval      # harness -> verdict -> ablations -> lag_probe -> typology -> reasons -> baf
make test
```

`make eval` runs, in order: `rakshak.eval.harness` (validate-window baselines, `results/summary.md`
— superseded for headline purposes by `verdict.md` but the split-population and cross-check
tables live there), `rakshak.eval.verdict` (the K2 table above, `results/verdict.md`),
`rakshak.eval.ablations` (`results/ablations.md`), `rakshak.eval.lag_probe`
(`results/lag_probe.md`), `rakshak.eval.typology` (`results/typology_recall.md`),
`rakshak.explain.reasons` (`results/reasons.json`), and `rakshak.eval.baf`
(`results/baf_validation.md`, tolerant of a missing git-ignored download on a clean checkout). Every
run in this document used **seed 42**. Full `make eval` runtime is under the 15-minute NFR budget
on a laptop CPU; `make` is not installed on the machine this was built on, so `./make.ps1 eval` is
the exercised path and `Makefile` itself has not been run on a Linux checkout.

---

## What this project does not establish

- **Whether any margin comes from sequence modelling or from this particular HMM is open.** The
  BOCPD changepoint baseline (T-0010) was cut, so the HMM is the only sequence-aware model measured
  anywhere in this repo.
- **No calibration happens anywhere in this repo.** Empirical-Bayes shrinkage (T-0008, ADR-0006)
  was cut. Bayes Minimum Risk consumes every model's raw score, clipped to [0, 1], as if it were a
  calibrated posterior — under a rank-only policy that would only cost a calibration metric; under
  BMR it moves the decision itself.
- **The perfect-hindsight oracle dominates by construction and proves nothing**; it is printed as a
  scale, not an achievement. The review-knapsack ceiling clears hold-everything on this split only
  because realised loss happens to be concentrated in a few merchants — not a general property.
- **Every sequence-layer number is measured on a generator this repo wrote**, at a 20% merchant
  fraud rate chosen for per-typology sample size, not realism.
- **No Pareto frontier exists.** `optimize/nsga.py`'s NSGA-II threshold search (ADR-0004) was cut
  before implementation; its mandatory grid-search comparison is undischarged.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5 for the full list of what was rejected on the
merits versus decided-in-then-cut-on-schedule, and `STATE.md` for the complete build history.

---

## Future work

From the T-0023 literature survey (`project-context/15-lit-survey-drift-detection.md`), which
audited the locked stack against current published work and reversed two ADR rationales without
moving a shipped number:

- **A calibration layer is the highest-leverage next piece of work, and it is smaller than it
  looks.** ADR-0006's empirical-Bayes shrinkage was cut believing it would calibrate model scores
  into posteriors for the Bayes-Minimum-Risk policy; the survey found that ADR-0006 actually
  shrinks per-merchant *cost* parameters, not scores, so reinstating it as originally specified
  would not have closed the gap. What would is a separate, smaller ticket — isotonic or Platt
  calibration of each model's score on `validate`, using `scikit-learn`, already a dependency.
- **A population-level drift detector (ADWIN/DDM/EDDM, via `river` 0.26.1, BSD-3-Clause) was never
  considered.** Every decision in this repo treats drift as per-merchant; the streaming-fraud
  literature's default primitive is a detector over the aggregate score or error stream, and one
  2025 benchmark (ROSFD, arXiv:2504.10229) reports ADWIN as its strongest drift method. This is the
  survey's single most actionable unexplored finding.
- **A second sequence-aware baseline (BOCPD, cut at T-0010) remains the cleanest way to answer
  whether any margin here is from sequence modelling or from this particular HMM** — a question this
  repo currently cannot answer at all.
- **Self-supervised event-sequence representations (CoLES-style contrastive pre-training, and its
  2026 hybrid with selective state-space models, arXiv:2607.20228) would remove the sequence
  layer's dependency on this project's own generator labels** for the representation, though not
  for the evaluation. GPU-oriented and out of scope for a CPU-only 4-day build; the honest answer to
  "what would you build with three months instead of four days".
