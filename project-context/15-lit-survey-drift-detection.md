# 🔬 Literature Survey: Post-Onboarding Merchant Risk Drift Detection (stack re-check against ADR-0001 … ADR-0009)

**Date:** 2026-08-30 · **Ticket:** T-0023 · **Trigger:** the locked stack has gone unchallenged
since the ADRs were written on 2026-08-29, and K2 fired **FAIL** at T-0011 the day before
(`STATE.md`, "T-0011 — the verdict"). A stack that has just failed its own pre-registered bar is
the one that most needs re-checking against prior art — and the one where a flattering survey
would do the most damage.
**Sources consulted:** 24 web searches · 10 primary sources opened and read (1 refused, HTTP 403) ·
6 US patents (read through the USPTO full-text index) · 3 package indices (PyPI JSON API)
**Claims verified at primary source:** 9 (marked `[VERIFIED]`). Everything else is
`[snippet-inferred]` and labelled as such.

> **Honest-measurement note, applied to this survey as it is to the project.** Two of the nine
> ADRs come out **reconsidered**, and one of those two — ADR-0006 — is reconsidered because the
> repo has been describing a decision as doing something it could not have done. The single most
> relevant paper found (a production money-mule detector, arXiv:2607.17586) uses **no sequence
> model at all**: LightGBM plus TreeSHAP. That is uncomfortable and it is consistent with K2's
> verdict, so it is the first thing in the Landscape section rather than the last. Where the
> academic literature is empty — and on post-onboarding *merchant* drift it very nearly is — that
> is said outright, and the patent literature that fills the gap is cited instead.

**This ticket produced no code and no new model.** Per `11-tickets/T-0023.md`, "What this ticket
must NOT do": nothing under `src/`, `MODEL_REGISTRY` or `results/` was touched, and no candidate
surfaced here was implemented or benchmarked. Every compelling finding below is future work.

---

## 📌 Problem Framing

Rakshak's problem, stated so that prior art can be matched against it: a merchant already cleared
at onboarding emits a transaction stream; something in that stream changes; the change must be
detected **at the merchant level**, **before chargebacks land 45–120 days later**, **on CPU**
(`CLAUDE.md:50`), and the decision must be **explainable to the merchant who calls to shout**
(`CLAUDE.md:152`).

That decomposes into four questions the literature can be asked:

| Question | Rakshak's answer today | Where it is locked |
|---|---|---|
| What carries the temporal signal? | Per-merchant pooled Gaussian HMM, hand-written | `CLAUDE.md:64`, ADR-0001 |
| What is the incumbent it must beat? | LightGBM over windowed features | `CLAUDE.md:66` |
| How does a belief become an action under scarce analyst hours? | Bayes Minimum Risk, 3 actions, hard capacity | `CLAUDE.md:67`, ADR-0005, ADR-0008 |
| Where does the data come from? | Own generator (sequences) + BAF (decision layer) | `CLAUDE.md:70`, ADR-0007 |

Three measured facts constrain every recommendation below, and none of them may be argued away:

| Fact | Value | Source |
|---|---|---|
| HMM vs `rules` at the central cost asymmetry | **+5.9% relative**, against a pre-registered ≥20% bar | `results/verdict.md`; `STATE.md` "K2 FIRED" |
| `random` savings on the test split | **0.5365 — beats every fitted model** | `STATE.md`, T-0011 table |
| `random` savings on BAF at 1.47% prevalence | **−28.2169** | `results/baf_validation.md` |
| Oracle-parameterised four-way ARI ceiling | 0.378 (0.404 after T-0003b) | ADR-0009, "Context" |

**The survey's job is therefore not "how do we win".** It is: given a sequence layer that did not
clear its bar, does the *published* record say the architecture was wrong, the problem is hard, or
both — and which locked decisions should change on the evidence.

---

## 🗺️ Landscape Overview

### The finding that goes first, because it is the unflattering one

**The most directly comparable production system in the current literature does not use a sequence
model.** Zhang et al., *Detection, Attribution, Narration: An End-to-End Pipeline for Explainable
Money Mule Identification* (arXiv:2607.17586, submitted 2026-07-20) describes a deployed
customer-level mule detector built from three stages: **a LightGBM classifier over 280 engineered
features** spanning transaction patterns, account demographics, network topology and temporal
behaviour; **a TreeSHAP attribution layer**; and an LLM that turns SHAP attributions into
analyst-facing narratives. In live AML production it reports a **yield rate of 89%, up from 61%
under the incumbent rule-based system**, with monthly alert volume rising 211 → 302, described as
broader true-positive coverage rather than added noise, and **60% incremental adverse detection
beyond existing review workflows** `[VERIFIED at arXiv abstract]`.

Read against this repo, that paper says three things:

1. **The entity-level drift problem is being solved in production with a GBDT plus engineered
   temporal features** — i.e. with the thing `CLAUDE.md:66` calls "the incumbent that must be
   beaten". T-0011 measured `gbdt` at PR-AUC 0.6523 against `hmm`'s 0.3347. The literature does
   not contradict that ordering; it is the ordering the deployed system also chose.
2. **Explainability is not the HMM's private property.** `CLAUDE.md:154` states the Viterbi path
   is the centrepiece because "Question 3 is the one nobody else in the submission pool will
   answer". TreeSHAP-over-LightGBM plus narration is a *different, deployed* answer to the same
   question, and it reports analyst-measured cognitive-load benefit. This does not invalidate the
   Viterbi-path pivot — a state path is a genuinely different object from a feature attribution,
   and it is causal-in-the-model rather than post-hoc — but the claim "nobody else answers
   Question 3" is no longer true of the field, and the README should not make it.
3. **Yield rate and alert volume are the operational metrics.** That is precision@K under a
   capacity constraint. ADR-0008's decision to express capacity per 1000 merchants and report the
   binding constraint is the same instinct, arrived at independently.

### The domain literature is thin in a specific, citable way

The K1 survey already recorded that there is "essentially no published work on merchant-level
*post-onboarding* latent-risk-state HMMs" (`12-lit-survey-k1.md`, honest-measurement note). This
survey extends that finding and narrows it: **the prior art for post-onboarding merchant drift is
predominantly patent literature held by the card networks and acquirers, not papers.** Searching
for MCC drift, transaction laundering and bust-out returns:

- **US 11334895** — *Methods, systems, and apparatuses for detecting merchant category code shift
  behavior*: a merchant trading across two MCCs, with "a sudden decline in one and a significant
  increase in another with no significant change in overall transaction volume", is flagged as
  potentially miscoded or laundering `[snippet-inferred from the USPTO full-text index]`. That is
  *exactly* Rakshak's category-drift typology, patented.
- **US 11651462** — *Methods and systems for detecting transaction laundering*: a DNN over ~98
  merchant-descriptive variables including MCC, account age, cross-border percentage, number of
  identical MIDs, and outlier transaction volatility `[snippet-inferred]`.
- **US 11514533 / US 12094011** — *Systems and methods for identifying a MCC-misclassified
  merchant* `[snippet-inferred]`.
- **US 7428509 / US 8001042** — bust-out detection from portfolio and credit-bureau signals; the
  1980s-vintage framing that bust-out is a *behavioural trajectory* (build limit, then draw it
  down fast) rather than a point anomaly `[snippet-inferred]`.

**Consequence for the README, and it is a positive one:** Rakshak has no academic benchmark to
lose to, and the commercial prior art it does have is closed. `ADR-0007`'s statement that "no
public merchant-sequence dataset with merchant-level risk labels exists" (ADR-0007, "Context") is
**reaffirmed by this search** — and the reason is now sharper than "nobody published it": the
people who have the data patented the method instead.

### Four families, and which of Rakshak's problems each attacks

| Family | Attacks | Representative work | Maturity | Rakshak's position |
|---|---|---|---|---|
| **Entity-level behavioural profiling** (peer-group analysis, break-point analysis) | Per-merchant drift against a peer baseline | Bolton & Hand, *Peer Group Analysis* / *Unsupervised Profiling Methods for Fraud Detection* (2001) | Consensus, 25 years old | **Not cited anywhere in the repo.** Closest classical prior art to the product thesis |
| **Population concept-drift detection** (ADWIN, DDM, EDDM) | A shock or regime change hitting *many* entities at once | ROSFD, arXiv:2504.10229 (2025); `river` | Consensus | Not considered by any ADR. Directly relevant to T-0022 |
| **Per-entity changepoint detection** (BOCPD, CUSUM) | Abrupt parameter shifts in one stream | Adams & MacKay 2007; Altamirano et al., ICML 2023 | Consensus | `CLAUDE.md:65`; T-0010 **cut**, so never measured |
| **Learned sequence representations** (CoLES, SSM/Mamba, contrastive transformers) | Representation quality without hand-designed states | arXiv:2502.04899 (survey); arXiv:2607.20228; arXiv:2605.21490 | Emerging | Rejected at ADR-0002 on compute; the rejection still holds |

The row that matters most and is missing from every ADR is the **second** one. Rakshak's entire
design treats drift as a *per-merchant* phenomenon — the generator "models every merchant as a
fully independent stochastic process" (`14-spec-blackswan-and-drift-survey.md`, Problem Statement
¶1). The streaming-fraud literature's default drift primitive is the opposite: a
**population-level** detector over the score or error stream. ROSFD (arXiv:2504.10229, 2025)
compares DDM, EDDM and ADWIN inside a streaming fraud pipeline and reports that "**ROSFD utilizing
ADWIN as the drift detection method demonstrated the best performance among the employed
methods**", with Adaptive Random Forest the strongest base learner (highest AUC on four of five
datasets) `[VERIFIED at arXiv abstract]`. `river` ships ADWIN, DDM and EDDM; **0.26.1, BSD-3-Clause,
requires-python ≥3.11** `[VERIFIED at PyPI JSON API]` — licence-clean under `CLAUDE.md:56` and
version-compatible with `CLAUDE.md:71`.

This is the survey's single most actionable finding and it is **future work, not code** (T-0023
"must NOT" clause 2). See §"What this hands to T-0022c and T-0013".

---

## Q1 — Verdicts on the locked stack (`CLAUDE.md:64–71`)

Every row of the stack table gets a verdict. Rows with an ADR are cross-referenced; rows without
one are judged against the CLAUDE.md line directly.

### Hand-written HMM (`CLAUDE.md:64`) — see **ADR-0001**, below. **Reaffirmed with a caveat.**

### BOCPD as the changepoint baseline (`CLAUDE.md:65`) — **reaffirmed with a caveat. No ADR covers this row.**

The choice of Adams & MacKay's BOCPD as *a baseline* is unchallenged: BOCPD remains the reference
online Bayesian changepoint method and 2025 work continues to benchmark against it, reporting that
"the BOCPD algorithm outperforms classical methods across most scenarios" on financial series
while CUSUM/GLR remain strong classical benchmarks `[snippet-inferred, ACM
10.1145/3795154.3795291]`. The K1 survey's negative reading — BOCPD assumes **abrupt** parameter
change and RAMP is gradual by construction — is unchanged and remains the correct prediction
(`12-lit-survey-k1.md` §2.4, verified against the Adams & MacKay abstract).

Two caveats, both naming methods that were not feasible here:

- **Robust BOCPD.** Altamirano, Briol & Knoblauch, *Robust and Scalable Bayesian Online Changepoint
  Detection* (PMLR 202, ICML 2023) addresses BOCPD's known fragility to outliers — relevant
  because a black-swan shock is precisely an outlier that a naive BOCPD would read as a changepoint
  in every merchant simultaneously `[snippet-inferred]`.
- **CUSUM with tuned hyperparameters** is repeatedly reported as a hard-to-beat cheap benchmark,
  with 2025 work on meta-recommending its hyperparameters online because "a single setup may not
  be universally suitable across the entire time series" `[snippet-inferred, PMC12074366]`.

**The caveat that actually costs the repo something is not about BOCPD's design — it is that
T-0010 was cut, so no sequence-aware baseline other than the HMM was ever measured**
(`STATE.md`, ablations section; ADR-0009 decision item 5). The literature makes the untested
prediction sharper, not weaker: BOCPD should win on BUST_OUT and REFUND_COLLUSION and lose on
SLOW_RAMP. That prediction is still unfalsified and the README must say so.

### LightGBM as the discriminative baseline (`CLAUDE.md:66`) — **reaffirmed, strongly. No ADR covers this row.**

Grinsztajn, Oyallon & Varoquaux, *Why do tree-based models still outperform deep learning on
tabular data?* (arXiv:2207.08815, NeurIPS 2022): across "a standard set of 45 datasets from varied
domains with clear characteristics of tabular data", "**tree-based models remain state-of-the-art
on medium-sized data (~10K samples) even without accounting for their superior speed**"
`[VERIFIED at arXiv abstract]`. Rakshak's window matrix is squarely in that regime. The 2025
follow-up literature keeps the finding while adding nuance about size- and dimension-dependence
`[snippet-inferred, ACM Computing Surveys 10.1145/3807777]`, and arXiv:2607.17586's production
system is a LightGBM `[VERIFIED]`.

**This is the one row where the literature agrees with the repo's measured result rather than
merely permitting it.** `gbdt` leading `hmm` on PR-AUC (0.6523 vs 0.3347), precision@5 and Brier
is the expected outcome, not an anomaly to be explained away.

### Bayes Minimum Risk + savings score (`CLAUDE.md:67`) — see **ADR-0005**. **Reaffirmed with a caveat.**

### Empirical-Bayes shrinkage (`CLAUDE.md:68`) — see **ADR-0006**. **RECONSIDERED.**

### pymoo NSGA-II ≥0.6.2 (`CLAUDE.md:69`) — see **ADR-0004**. **RECONSIDERED.**

`pymoo` **0.6.2, Apache-2.0** `[VERIFIED at PyPI JSON API]` — licence still clean under
`CLAUDE.md:56`. The problem is not the package.

### Data: own generator + BAF (`CLAUDE.md:70`) — see **ADR-0007**. **Reaffirmed with a caveat.**

### Python ≥3.11 (`CLAUDE.md:71`) — **reaffirmed.** Nothing found requires otherwise; the one new
library named anywhere in this survey (`river` 0.26.1) requires ≥3.11 `[VERIFIED]`.

---

## Q2 — Verdicts on the rejection table (`CLAUDE.md:77–83`)

### Graph neural networks (`CLAUDE.md:77`, ADR-0002) — **reaffirmed with a caveat.**

The 2025 reviews continue to report GNNs outperforming tabular baselines on relational fraud
`[snippet-inferred: arXiv:2411.05815; a Medium benchmark quotes RF 0.78 / XGBoost 0.81 / GNN 0.89
recall — not verified at a primary source and not relied on here]`. **ADR-0002's rejection does
not rest on that comparison and is untouched by it.** Its load-bearing objection is
evaluation-validity, stated in its own Consequences: "If a GPU appeared tomorrow the circularity
objection to (a) would still stand — it is an evaluation-validity problem, not a compute problem."
Nothing in the 2025–2026 literature dissolves that. Scoring a GNN on the only merchant×payer graph
available — the one this repo's generator wrote — measures the generator.

**The caveat names a method ADR-0002 did not consider and that is neither GPU-bound nor circular:**
IBM's *Graph Feature Preprocessor* (ACM 10.1145/3677052.3698674) — real-time subgraph-based feature
extraction feeding a **GBDT**, i.e. the CPU-feasible middle ground between ADR-0002's option (a)
and option (c) `[snippet-inferred]`. It is blocked here by **data, not compute**: ADR-0002's own
Consequences record that "the generator scopes payers to a single merchant by design, so no
cross-merchant collusion structure exists to detect". A subgraph feature extractor over a graph
with no cross-merchant edges extracts nothing. That is the honest reason it is not in the repo,
and it is a better reason than the one ADR-0002 gives.

### Sequence transformers (`CLAUDE.md:78`, ADR-0002) — **reaffirmed. The citation is stronger than the repo's paraphrase.**

`CLAUDE.md:78` and ADR-0002 cite NICE Actimize, arXiv:2605.21490, for "parity with feature
engineering". Read at source: Butvinik, Marcus, Tal & Azoulay (NICE Actimize), *Temporal
Contrastive Transformer for Financial Crime Detection*. Embeddings alone reach **AUC 0.8644**;
combined with domain-engineered features, "**no measurable improvement is observed**" against the
baseline — **0.9245 vs 0.9205** — and the work is explicitly "**not yet production-ready**"
`[VERIFIED at arXiv abstract]`. The paper frames parity itself as the meaningful outcome. **The
repo's paraphrase is accurate and the underlying numbers are more favourable to the rejection than
the paraphrase suggested.** Quote the 0.9245/0.9205 pair in the video; it is a vendor's own
research team reporting a 0.4-point delta.

The one thing to name as newer: **CoLES-style self-supervised event-sequence representations**
(survey: arXiv:2502.04899), and the CoLES + selective-SSM (Mamba) hybrid of arXiv:2607.20228,
which reports convergence "2–3× faster than the plain SSM baseline" and uses discretisation-step
maps and Integrated Gradients for explanation `[VERIFIED at arXiv abstract]`. This is the family
that would learn merchant behaviour representations **without generator labels** — the exact
limitation ADR-0009 books as its worst consequence. It is GPU-oriented and its published
evaluations are on age-group / product-acquisition / purchase-prediction benchmarks, not fraud.
**Headline future-work item.**

### Reinforcement learning (`CLAUDE.md:79`, ADR-0003) — **reaffirmed.**

See ADR-0003 below. The only change is that ADR-0003's central empirical premise now has a
citation it lacked.

### `hmmlearn` (`CLAUDE.md:80`, ADR-0001) — **reaffirmed.**

PyPI, checked today: **hmmlearn 0.3.3, BSD licence**, and the project description still carries
"**Note: This package is under limited-maintenance mode**" `[VERIFIED at PyPI JSON API]`. The
factual claim at `CLAUDE.md:64` and in ADR-0001's Context ¶1 is current, not stale. Licence was
never the objection and still isn't.

### NSGA-III (`CLAUDE.md:81`, ADR-0004) — **RECONSIDERED.** See ADR-0004 below.

### Deepfake / KYC document analysis (`03-landscape.md`) — **reaffirmed.**

The threat is real and growing — 2026 industry reporting describes real-time face-swap, camera
injection and voice cloning sold as fraud-as-a-service, and an RBI annual report figure of a 46.4%
year-on-year rise in banking fraud cases `[snippet-inferred, vendor and press sources; the RBI
figure was not verified at rbi.org.in and must not be quoted in the README]`. It changes nothing
here: it needs vision models and GPU (`CLAUDE.md:50`), and — the more important reason — **it is
an onboarding-time control, and Rakshak's entire thesis is that the gap is *after* onboarding**
(`CLAUDE.md:13`). A stronger deepfake gate makes Rakshak's problem *more* acute, not less.

### RTO / COD return fraud (`03-landscape.md`) — **reaffirmed.**

Nothing found suggests the space has opened up. Razorpay's Thirdwatch has owned it since 2019 and
the crowding argument is unchanged. Refund *collusion* — merchant-side, the typology this repo
actually models — is a different problem and the published work on it is thin: the searchable
material is vendor guidance plus one 2026 qualitative study of GenAI-enabled refund fraud in
Chinese e-commerce (arXiv:2606.03215) `[snippet-inferred]`, which studies the human ecosystem, not
detection. **No detection benchmark for merchant-side refund collusion was found.** That belongs
in the README's limitations, beside the merchant-sequence dataset gap.

---

## Q3 — The nine ADR verdicts

Every ADR from 0001 to 0009 gets exactly one verdict. Two are **reconsidered**; the addendum text
for both is in the final section.

| ADR | Subject | Verdict |
|---|---|---|
| **ADR-0001** | Hand-written HMM; `hmmlearn` rejected | **Reaffirmed with a caveat** |
| **ADR-0002** | No GNN, no sequence transformer; graph scalars instead | **Reaffirmed with a caveat** |
| **ADR-0003** | RL rejected for the build, retained as a pitch slide | **Reaffirmed** |
| **ADR-0004** | NSGA-II not NSGA-III, plus the grid-search obligation | **RECONSIDERED** |
| **ADR-0005** | Three actions under a hard review-capacity constraint | **Reaffirmed with a caveat** |
| **ADR-0006** | Per-merchant cost parameters shrunk to segment, closed form | **RECONSIDERED** |
| **ADR-0007** | Hybrid data: own generator + public benchmark | **Reaffirmed with a caveat** |
| **ADR-0008** | Review capacity expressed per 1000 merchants | **Reaffirmed** |
| **ADR-0009** | Label-informed HMM estimation + re-specified FR-013 | **Reaffirmed with a caveat** |

### ADR-0001 — hand-written HMM, `hmmlearn` rejected → **reaffirmed with a caveat**

**Reaffirmed on its own terms.** All three grounds in ADR-0001's Context hold today. Maintenance:
`hmmlearn` 0.3.3 still declares limited-maintenance mode `[VERIFIED at PyPI]`. Expressiveness: the
K1 response required a *weighted* likelihood (ADR-0009 decision item 2) that no library exposes,
which ADR-0001's Consequences already record as an unforeseen retrospective benefit — that record
is accurate. Judgement signal: unchanged, and it is a claim about a panel, not about literature.

**The caveat is not about the library choice; it is about what the choice bought.** ADR-0001
decides *how* to build an HMM, not *whether* the sequence layer should be primary. Two findings
bear on the latter and belong beside this ADR rather than inside it:

1. The nearest deployed system to Rakshak's problem uses a GBDT and post-hoc attribution, not a
   latent-state model (arXiv:2607.17586) `[VERIFIED]`.
2. The published HMM fraud literature remains overwhelmingly *card-level* transaction-sequence
   work — Srivastava et al. (IEEE TDSC 2008) and its long tail — and secondary sources
   consistently report high false-positive rates and difficulty choosing an adequate emission
   family for multivariate observations as its standing limitations `[snippet-inferred]`. Rakshak
   measured both independently: `hmm` flags 0.75 of the population against `gbdt`'s 0.65 on test,
   and the pooled-Gaussian emission is the documented weak point (`STATE.md`, ablations —
   standardisation off moves HMM Brier +0.1729 and `gbdt`'s by −0.0081).

Neither reopens ADR-0001. Both mean the video's framing must be "we hand-wrote the estimator, and
here is the honest measurement of what the estimator bought" — which is already the pivot
`00-charter.md` §3 mandates.

### ADR-0002 — no GNN, no transformer → **reaffirmed with a caveat**

Covered in Q2 above. Summary: the transformer half is reaffirmed with a **stronger** primary
citation than the repo currently quotes (0.9245 vs 0.9205, "not yet production-ready"
`[VERIFIED]`). The GNN half is reaffirmed on ADR-0002's own evaluation-validity ground, which no
2026 result touches. The caveat names the *Graph Feature Preprocessor* → GBDT path as the
CPU-feasible middle ground ADR-0002 did not consider, and records that it is blocked here by the
generator's single-merchant payer scoping — a data limitation ADR-0002 already documents — rather
than by compute.

### ADR-0003 — RL rejected → **reaffirmed**

Both of ADR-0003's disqualifying facts survive contact with the literature, and the first one now
has a citation it did not have. Dal Pozzolo, Boracchi, Caelen, Alippi & Bontempi, *Credit Card
Fraud Detection and Concept-Drift Adaptation with Delayed Supervised Information* (IJCNN 2015), and
the follow-on *Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy*
(IEEE TNNLS 2018), formalise **verification latency** as a first-class property of the fraud
problem and find that "learning from feedbacks is a different problem than learning from delayed
samples", with the winning strategy being two separate classifiers aggregated
`[snippet-inferred — abstracts read via search index; the Politecnico PDF was not opened]`. That
is the published version of ADR-0003's claim that ground truth arriving 45–120 days late leaves an
agent nothing to learn from inside a 4-day build.

ADR-0003's second fact — training inside our own simulator learns the simulator — is
methodological and needs no external support; it is the same objection ADR-0002 raises against
GNNs, and this survey found nothing that weakens it.

**No caveat is warranted.** The one nearby method worth naming is not RL: it is Dal Pozzolo's
*delayed-label handling*, which is a supervised-learning design, and it would matter to a real
deployment rather than to this build.

### ADR-0004 — NSGA-II, not NSGA-III → **RECONSIDERED**

**This is the clearest reversal in the survey, it rests on a verified theorem, and it contradicts
the ADR's stated rationale — not its outcome.**

ADR-0004's Context ¶2 argues: *"Three objectives is not many-objective. NSGA-III's
reference-direction machinery exists to maintain diversity when the Pareto front is
high-dimensional, which begins to matter at four or more objectives."* `CLAUDE.md:69` and
`CLAUDE.md:81` repeat it as a locked constraint.

Zheng & Doerr, *Runtime Analysis for the NSGA-II: Proving, Quantifying, and Explaining the
Inefficiency For Many Objectives* (arXiv:2211.13084; IEEE TEVC; GECCO 2024) prove the opposite
boundary. Verified at source: on the m-objective generalisation of OneMinMax — a benchmark where
**every solution is Pareto optimal** — NSGA-II "cannot compute the full Pareto front (objective
vectors of all Pareto optima) in sub-exponential time **when the number of objectives is at least
three**", even with large population sizes. The mechanism is named exactly: the inefficiency
"lies in the fact that in the computation of the crowding distance, the different objectives are
regarded independently", a property that is harmless at two objectives — where sorting by one
objective is the inverse sorting by the other — and breaks beyond two `[VERIFIED at arXiv
abstract]`.

**Three is not the safe side of the line. Three is the first failing case.** The
degradation-at-three reading is corroborated in the applied literature ("NSGA-II optimizes
bi-objective problems efficiently while it performs less when dealing with three or more
objectives") `[snippet-inferred]`.

**What this does and does not change:**

- It **does** overturn the reason recorded in ADR-0004 and repeated at `CLAUDE.md:69` and
  `CLAUDE.md:81`. That reason is wrong at the exact objective count this project has.
- It does **not** promote NSGA-III. NSGA-III is a many-objective algorithm whose own runtime
  analyses target ≥4 objectives; nothing found recommends it at 3 for this problem shape, and
  reversing to it would be trading one under-evidenced choice for another.
- It **strengthens** the half of ADR-0004 that matters most — option (c), the uncoupled
  per-segment grid search, and the obligation that NSGA-II must dominate it in hypervolume or be
  deleted. If NSGA-II's diversity mechanism is provably degraded at 3 objectives, the grid search
  is no longer merely a mandatory baseline; it is the **better default** for a 3-objective,
  low-dimensional, cheap-to-evaluate threshold search, and a dominance result would have been the
  surprising outcome.
- It costs the project nothing today. **T-0009 was cut** and neither algorithm was built
  (ADR-0004, Consequences ¶1; `STATE.md`). The repo makes no frontier claim.
- It makes T-0020's `pymoo` decision easy. ADR-0004's Consequences already flag "`pymoo` remains
  in `pyproject.toml` as a declared dependency for work that did not happen — that is a defect to
  resolve before freeze". **Drop it**, and record the reason as *the rationale did not survive
  review*, not merely *the work was cut*.

Dated addendum text is in the final section.

### ADR-0005 — three actions under a hard capacity constraint → **reaffirmed with a caveat**

**Reaffirmed.** Bayes Minimum Risk over example-dependent costs remains the standard formulation
(Elkan 2001; Bahnsen, Stojanovic, Aouada & Ottersten, *Cost Sensitive Credit Card Fraud Detection
using Bayes Minimum Risk*, ICMLA 2013; Bahnsen et al. 2016's savings score) `[snippet-inferred —
the Bahnsen ICMLA PDF was located at albahnsen.github.io but read via search index]`, and the
industry literature on AML/fraud triage describes the operating problem in ADR-0005's own terms:
alert volume against limited investigator capacity, with risk-based prioritisation as the lever
(arXiv:2112.07508; arXiv:2604.19755) `[snippet-inferred]`. arXiv:2607.17586's headline production
metric — **yield rate**, 61% → 89% `[VERIFIED]` — is precision among worked alerts under a fixed
review budget. That is `precision@K` with K set by capacity, which is what ADR-0008 makes the
harness compute.

**The caveat is the one the repo already found by measurement, and the literature confirms the
danger rather than the remedy.** The cost-metric literature warns explicitly that cost-matrix
evaluation is sensitive to the choice of baseline — "there is risk of applying different baselines
when using a cost-matrix to measure overall cost" — and that cost-based scores should be
supplemented, never used alone `[snippet-inferred]`. Rakshak's `random` row is the sharpest
instance of that warning found anywhere: **0.5365 on test, beating every fitted model, at PR-AUC
0.2449** (`STATE.md`). No paper found predicts that outcome, but the discipline the repo adopted
in response — always print savings net of the `random` floor with PR-AUC beside it — is exactly
what this literature prescribes. **Keep it, and cite it.**

Second caveat, unchanged from ADR-0005's own Consequences: BMR is myopic and consumes an
uncalibrated score as a posterior. See ADR-0006.

### ADR-0006 — empirical-Bayes shrinkage of per-merchant cost parameters → **RECONSIDERED**

**The decision described in ADR-0006 is sound. What is wrong is what the repo believes it would
have delivered, and that belief is load-bearing in two other ADRs.**

ADR-0006's Related line reads: *"ADR-0005 (the policy that consumes the calibrated posterior this
ADR was to produce)."* ADR-0005's Consequences read: *"ADR-0006's shrinkage was cut, so no
recalibration happens anywhere in this repo."* `STATE.md` repeats it as "the strongest argument for
un-cutting T-0008".

**Those two things are not the same object.** ADR-0006's Context is explicit that the shrinkage is
over *per-merchant cost parameters* — `theta_hat_m = w_m·theta_MLE_m + (1−w_m)·theta_segment`,
partial pooling of `L_m` and `c_fp(m)` toward the MCC × AOV-band segment (FR-011). That shrinks the
**cost side** of the BMR argmin. The thing ADR-0005 is missing is calibration of the **score
side** — `P(bad | merchant)`, which the harness currently takes raw from each model. Shrinking
`L_m` toward a segment mean does not make `hmm`'s 0.3347-PR-AUC score with Brier 0.4321 into a
posterior. Reinstating T-0008 exactly as written would have left BMR consuming the same
uncalibrated scores.

The literature is unambiguous about which of the two BMR actually requires: for cost-sensitive
thresholding to work, the model must produce **well-calibrated probability estimates**, and
calibrated scores are "the prerequisite of some cost-sensitive learning techniques"; the standard
remedies are Platt scaling, isotonic regression and beta calibration `[snippet-inferred — multiple
concurring secondary sources; no single primary source was opened]`. Bahnsen's own BMR line of
work is built on calibrated posteriors, not on shrunken cost parameters `[snippet-inferred]`.

**What should change:**

- ADR-0006 keeps its decision (closed-form EB shrinkage, no MCMC — `CLAUDE.md:68` stands) but must
  stop claiming to produce the calibrated posterior ADR-0005 needs. That is a **different, cheaper,
  and more urgent** piece of work: an isotonic or Platt calibration of model scores fitted on
  `validate`, which is a handful of lines against `sklearn` (BSD-3, already a dependency) and
  strictly smaller than T-0008 ever was.
- **This is not authorised as build work by this ticket and must not be built now** (T-0023 "must
  NOT" clause 2; freeze Tue 1 Sep). It is a correction to the record and a future-work item.
- The correction cuts **against** the project's interest, which is why it is in this survey:
  it removes the tidy story that one cut ticket explains the calibration gap. The gap is real,
  it is worse than described, and T-0008 would not have closed it.

Dated addendum text is in the final section.

### ADR-0007 — hybrid data strategy → **reaffirmed with a caveat**

**The load-bearing claim survives an active search.** ADR-0007's Context states, and
`06-requirements.md:28` records: *"No public merchant-sequence dataset with merchant-level risk
labels exists."* Searching the 2023–2026 record for anything that would falsify it turns up the
closest candidates and none of them qualifies:

- **AMLworld / *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models***
  (IBM Research + ETH Zurich, NeurIPS 2023 Datasets & Benchmarks; six datasets on Kaggle, up to
  175–180M transactions with complete laundering ground truth) — **account-level, and synthetic
  from an agent-based generator** `[snippet-inferred]`.
- **CFDB, *A Customer-Level Fraud Detection Benchmark*** (arXiv:2404.14746) — built on AMLworld's
  HI-Small / LI-Small subsets; **customer**-level, with "customer-centric features". Its abstract
  does not state the licence or access terms `[VERIFIED that the abstract does not state them]`.
- **TransXion** (arXiv:2604.17420), **BlazingAML** (arXiv:2604.12241) — AML graph benchmarks,
  entity/account-level `[snippet-inferred]`.
- A labeled synthetic mobile-money transaction dataset (2025, PMC12036017) — synthetic, and
  customer- rather than merchant-centric `[snippet-inferred]`.

**Nothing merchant-level with merchant risk labels. ADR-0007's boundary statement is reaffirmed,
and it is now reaffirmed against a named search rather than asserted.**

**The caveat is that ADR-0007's option set was incomplete.** Its options were (a) public only,
(b) own generator only, (c) hybrid. There is a fourth that was never considered: **use somebody
else's generator's output as an external synthetic benchmark** — AMLworld/CFDB. ADR-0007's own
circularity objection to option (b) applies only partly: an external generator still encodes
*someone's* assumptions, but not *ours*, so it would test whether the sequence layer generalises
past the assumptions this repo wrote. That is a real check the repo does not have, and T-0015
rejected the nearest instances (PaySim, Sparkov) on exactly the circularity ground without
distinguishing *our* assumptions from *another team's* (`STATE.md`, T-0015). The distinction is
worth making in the README's limitations; the work is not affordable before freeze.

Second, smaller caveat: `12-lit-survey-k1.md` recorded that "there is no published benchmark for
post-onboarding merchant latent-risk-state recovery". **Still true.** The reason found by this
survey — the method is in the patent literature (§Landscape) — should be added, because
"no benchmark exists" reads as a weak search until it is paired with where the work actually went.

### ADR-0008 — capacity per 1000 merchants → **reaffirmed**

Nothing in the literature contradicts it, and the operational framing it produces is the framing
production systems report. arXiv:2607.17586's production numbers are **monthly alert volume**
(211 → 302) and **yield rate** (61% → 89%) `[VERIFIED]` — a volume budget and a precision under
that budget, which is precisely `K` and `precision@K`. ADR-0008's own choice of ~6% of the book per
decision period as "a plausible load for a risk-ops desk" is an assumption, and the AML triage
literature's recurring complaint of 95–98% false-positive rates in rule-based alerting
`[snippet-inferred, arXiv:2112.07508]` suggests real desks run at far worse precision than
Rakshak's swept range — i.e. the assumption is, if anything, generous to the incumbent baseline.

**No caveat.** ADR-0008 is the ADR that has aged best, and it is the least glamorous one.

### ADR-0009 — label-informed HMM estimation + re-specified FR-013 → **reaffirmed with a caveat**

**Reaffirmed.** Nothing found in this survey disturbs the K1 chain of reasoning: Romano et al.
(JMLR 17, 2016) on AMI for unbalanced references with small clusters, Elworthy (1994) and Merialdo
(1994) on unsupervised Baum-Welch degrading a model when labels exist, Li et al. (2024) on EM's
contraction under rare-event mixtures, Sidrow et al. (2025) on weighted-likelihood partial labels,
Ruiz-Suarez et al. (2021) on overlap rather than misspecification being the binding constraint —
all of these were verified at source in `12-lit-survey-k1.md` on 2026-08-28 and none has been
superseded in the two days since. The discipline of retaining the 0.091 ARI and the 0.378 oracle
ceiling permanently is what the metric-choice literature demands of anyone who changes a metric
after seeing it fail.

**The caveat is ADR-0009's own worst consequence, and the literature now names the method that
addresses it.** ADR-0009 records: *"The sequence layer is now label-informed on SYNTHETIC generator
labels. This is a stronger limitation than the plain synthetic-data caveat."* The current answer
to "learn behaviour representations from transaction streams without labels" is **self-supervised
event-sequence modelling** — CoLES-style contrastive learning over account/entity sequences,
surveyed in arXiv:2502.04899, and its 2026 hybrid with selective state-space models
(arXiv:2607.20228, CoLES + Mamba, with Integrated Gradients and discretisation-step maps for
explanation) `[VERIFIED at abstract]`. Pre-train unsupervised on unlabelled merchant streams, then
attach a small supervised head. It removes the dependency on generator labels for the
*representation*, though not for the *evaluation*.

Out of scope here for the reasons ADR-0002 already gives — GPU-oriented, and its published
evaluations are on age-group, product-acquisition and purchase-prediction benchmarks rather than
fraud. **This is the survey's headline future-work item**, and it is the honest answer to a panel
member who asks "what would you build with three months instead of four days?"

---

## ⚡ Must-Read (in priority order, ~90 minutes total)

1. **Zhang et al. (2026), arXiv:2607.17586** — https://arxiv.org/abs/2607.17586 — the deployed
   system closest to this problem. LightGBM + TreeSHAP + narration, 61% → 89% yield. Read the
   abstract before writing the README's explainability claim.
2. **Zheng & Doerr, arXiv:2211.13084** — https://arxiv.org/abs/2211.13084 — the abstract alone
   overturns ADR-0004's stated rationale. Ten minutes.
3. **Butvinik et al. (NICE Actimize), arXiv:2605.21490** — https://arxiv.org/abs/2605.21490 —
   already cited by ADR-0002; read it for the 0.9245-vs-0.9205 pair, which is a better line for
   the video than the current paraphrase.
4. **Grinsztajn, Oyallon & Varoquaux (2022), arXiv:2207.08815** — https://arxiv.org/abs/2207.08815
   — why `gbdt` beating `hmm` on PR-AUC is the expected result.
5. **ROSFD, arXiv:2504.10229** — https://arxiv.org/abs/2504.10229 — ADWIN as the streaming-drift
   default; the family Rakshak never considered.
6. **Bolton & Hand (2001), *Peer Group Analysis* / *Unsupervised Profiling Methods for Fraud
   Detection*** — the 25-year-old classical framing of exactly Rakshak's product thesis. Not
   cited anywhere in this repo and it should be.
7. **Dal Pozzolo et al. (2015/2018)** — verification latency as a first-class property; the
   citation ADR-0003's central claim was missing.

---

## 🔧 Libraries touched by this survey (all licence-checked today)

| Package | Version | Licence | Python | Status | Relevance |
|---|---|---|---|---|---|
| `pymoo` | **0.6.2** | **Apache-2.0** | — | healthy | `CLAUDE.md:69`. Licence fine; **the rationale is not** — see ADR-0004. Drop it at T-0020 |
| `hmmlearn` | **0.3.3** | BSD | — | *"under limited-maintenance mode"*, per its own PyPI description | `CLAUDE.md:80` / ADR-0001 — **stays rejected**, on current evidence |
| `river` | **0.26.1** | **BSD-3-Clause** | **≥3.11** | healthy | ADWIN / DDM / EDDM. Licence- and version-clean. **Future work only — not added** |
| `scikit-learn` | current | BSD-3 | — | existing dep | Isotonic / Platt calibration — the thing ADR-0006 was believed to provide and does not |

All four rows `[VERIFIED at the PyPI JSON API on 2026-08-30]`, except `scikit-learn`, which is an
existing declared dependency.

**No dependency was added by this ticket.** `CLAUDE.md:103` requires licence-checking and a
`LOGBOOK.md` entry for any addition; there is nothing to record because nothing was added.

---

## ⚠️ Where the literature is thin, and what this survey could not settle

- **Post-onboarding merchant drift has no public academic benchmark and no published detection
  baseline.** The method exists in patents (§Landscape). Rakshak has nothing to lose to and no
  independent validation of its problem framing. This is the second survey in this repo to reach
  that conclusion and it should now be stated in the README as a finding, not an apology.
- **Merchant-side refund collusion has no detection benchmark at all** that this search could
  find. The one recent academic paper on refund fraud (arXiv:2606.03215) studies the human
  ecosystem `[snippet-inferred]`.
- **No paper found predicts or explains the `random`-beats-everything savings result.** The
  cost-metric literature warns about baseline sensitivity in general terms `[snippet-inferred]`;
  the specific mechanism — a uniform score landing most merchants on the correct side of a
  merchant-specific threshold when `c_fp` is small relative to `L_m` — is documented only in this
  repo (`07-math.md` §6 AP-06; `STATE.md`, T-0007b). If that is right, it is a small original
  contribution and should be described as one, cautiously.
- **Whether the HMM's Viterbi path is a *better* explanation than TreeSHAP + narration is
  unmeasured and unmeasurable here.** arXiv:2607.17586 reports qualitative analyst feedback on
  reduced cognitive load `[VERIFIED]`; Rakshak has no analysts and no user study. The README may
  claim the Viterbi path is a *different kind* of explanation — intrinsic and temporal rather than
  post-hoc and static — and must not claim it is better.
- **Several conclusions here rest on secondary sources.** Bolton & Hand, the Bahnsen ICMLA paper,
  Dal Pozzolo, the calibration literature and the MCC-shift patents were all read through search
  indices and full-text abstracts, not opened as primary PDFs. They are marked
  `[snippet-inferred]` throughout and none of them carries a *reconsidered* verdict on its own.
  **Both reconsiderations rest on `[VERIFIED]` primary sources** (arXiv:2211.13084 for ADR-0004;
  the internal contradiction between ADR-0006's Context and ADR-0005's Consequences for ADR-0006,
  which is checkable inside this repo without any external source at all).

---

## 💡 Contrarian view

**The survey's most useful output is not a method. It is that two of the nine ADRs contain a wrong
reason for a defensible decision — and that both were found by reading the ADRs against the
literature rather than by reading the code.**

ADR-0004 rejects NSGA-III with an argument that the runtime-analysis literature reverses.
ADR-0006 claims to supply a calibrated posterior that the technique it specifies cannot supply.
Neither error changed a shipped number, because **both decisions were cut before implementation**.
That is luck, not process — and it is worth saying on camera, because a panel that has seen a
thousand fraud demos has seen exactly zero submissions that audited their own architecture
decisions against prior art after the results came in, found two broken rationales, and printed
them.

`CLAUDE.md:28`: *"If a baseline beats the HMM, report that the baseline beat the HMM."* The same
rule applied to decision records rather than to models is what this document is.

---

## 🔍 What would change our mind

Stated in advance, as falsifiable conditions:

1. **A public merchant-sequence dataset with merchant-level risk labels appears.** ADR-0007's
   entire hybrid strategy, and the synthetic-data caveat that travels with every number in the
   repo, become renegotiable in one step. Cheapest check: re-run the search that produced §ADR-0007
   above, ~30 minutes.
2. **A GPU becomes available AND a cross-merchant graph exists.** Both conditions, not either.
   ADR-0002's compute objection falls to the first; its circularity objection falls only to a
   graph this repo did not generate. One without the other changes nothing.
3. **Someone measures a CoLES-style self-supervised encoder on merchant streams and reports a gain
   over engineered features.** arXiv:2605.21490 currently reports **parity** on a vendor's real
   data `[VERIFIED]`. A published win would move the sequence-representation question from
   "future work" to "reopen ADR-0002".
4. **T-0010 is un-cut and BOCPD beats the HMM on SLOW_RAMP.** That falsifies the Adams & MacKay
   reading carried since K1 and would justify promoting changepoint detection. The prediction has
   been on record since 2026-08-28 and is still untested.
5. **A calibration layer is added and the `random` floor stops winning on savings.** That would
   localise the AP-06 finding to miscalibration rather than to the cost matrix, and would
   substantially change what the video says about the savings metric. It would also confirm the
   ADR-0006 reconsideration below.
6. **The panel's criteria turn out to reward architectural novelty over measurement discipline.**
   Then this survey's contrarian view is a liability rather than an asset, and everything in it
   should be compressed to one slide.

---

### Provenance

Every claim marked `[VERIFIED]` was read at its primary source — arXiv abstract page or the PyPI
JSON API — during this survey on **2026-08-30**. Claims marked `[snippet-inferred]` come from
search-result summaries or secondary pages only and were not confirmed at a primary source; none
of them carries a *reconsidered* verdict. Two sources could not be opened: the MDPI *Journal of
Risk and Financial Management* article on economic performance of fraud models returned HTTP 403,
and the USPTO full-text pages were read through the search index rather than fetched. Both are
labelled accordingly and neither is load-bearing.

Repo-internal claims cite the file and line or the ADR section they rest on
(`CLAUDE.md:NN`, `ADR-000X`, `STATE.md` section headings, `results/*.md`) so that a reader can
check each one without trusting this document.

---

## ADR addenda — pending application by the lead

**Deviation from `11-tickets/T-0023.md`, recorded here rather than acted on.** The ticket's Build
step 4 says: *"For any reconsidered finding, append a dated addendum block to the relevant
`docs/adr/ADR-000X-*.md` file, in that file's own house format — never a silent rewrite."* The
lead imposed a file-ownership restriction for this session: `docs/adr/**` is owned by another
agent running in parallel, and a concurrent write to `ADR-0004` would race. **No file under
`docs/adr/` was read-modified, created or deleted by this ticket.** The two addenda are reproduced
below verbatim, in the target files' house format, for the lead to append. Nothing in either
addendum rewrites or deletes existing ADR text; both are append-only blocks.

---

### → append to `docs/adr/ADR-0004-nsga-ii-not-nsga-iii.md`

```markdown
## Addendum — 2026-08-30 (T-0023, drift-detection literature survey)

**The rationale in "Context" above is wrong at exactly three objectives. The decision it
supports is unchanged; the reason given for it is not.**

Zheng & Doerr, *Runtime Analysis for the NSGA-II: Proving, Quantifying, and Explaining the
Inefficiency For Many Objectives* (arXiv:2211.13084; IEEE TEVC; GECCO 2024) prove that on the
m-objective generalisation of the OneMinMax benchmark — where every solution is Pareto optimal —
NSGA-II "cannot compute the full Pareto front ... in sub-exponential time when the number of
objectives is at least three", even with large population sizes. The stated mechanism is the
crowding distance itself: "in the computation of the crowding distance, the different objectives
are regarded independently", which is harmless at two objectives and breaks beyond two. Verified
at the arXiv abstract on 2026-08-30.

"Context" ¶2 asserts that reference-direction machinery "begins to matter at four or more
objectives". **Three is the first failing case, not the last safe one.** The same claim is
repeated at `CLAUDE.md:69` and `CLAUDE.md:81` and is wrong in both places.

**What changes:**

* The rejection of NSGA-III is **not** reversed. NSGA-III is a many-objective algorithm whose own
  runtime analyses target four or more objectives; nothing in the surveyed literature recommends
  it at three for a low-dimensional threshold search. Reversing to it would trade one
  under-evidenced choice for another.
* **Option (c), the uncoupled per-segment grid search, is promoted from mandatory baseline to
  preferred default.** If NSGA-II's diversity mechanism is provably degraded at three objectives,
  a dominance result for the coupled solution would have been the surprising outcome, not the
  expected one.
* **`pymoo` should be dropped from `pyproject.toml` at T-0020**, and the reason recorded as *the
  rationale did not survive review*, not merely *the work was cut*. The Consequences section above
  already flags the dependency as a defect to resolve before freeze; this addendum removes the
  argument for keeping it.

**What does not change:** no shipped number. T-0009 was cut on 2026-08-28 and neither algorithm
was ever built, so the repo has never made a Pareto-frontier claim. The obligation recorded in
"Decision" — that a GA must dominate the grid search in hypervolume or be deleted — remains
undischarged and is now, if anything, easier to discharge in the grid search's favour.

Source: `project-context/15-lit-survey-drift-detection.md` §Q3, ADR-0004.
```

---

### → append to `docs/adr/ADR-0006-empirical-bayes-shrinkage.md`

```markdown
## Addendum — 2026-08-30 (T-0023, drift-detection literature survey)

**The decision above is sound. The claim attached to it — that it would have supplied the
calibrated posterior ADR-0005's BMR policy consumes — is not, and that claim has propagated.**

The header of this ADR relates it to "ADR-0005 (the policy that consumes the calibrated posterior
this ADR was to produce)". ADR-0005's Consequences read "ADR-0006's shrinkage was cut, so no
recalibration happens anywhere in this repo", and `STATE.md` calls reinstating T-0008 "the
strongest argument" for closing the calibration gap.

**Two different objects have been conflated.** The "Context" section above is explicit that the
shrinkage estimator applies to *per-merchant cost parameters* — the partial pooling of `theta_m`
toward its MCC × AOV-band segment mean under FR-011, i.e. `L_m` and `c_fp(m)`. That shrinks the
**cost side** of the BMR argmin. What ADR-0005 lacks is calibration of the **score side**:
`P(bad | merchant)`, which `eval/harness.py` currently passes to the policy raw. Shrinking `L_m`
toward a segment mean does not turn `hmm`'s raw score — PR-AUC 0.3347, Brier 0.4321 on test — into
a posterior. **Reinstating T-0008 exactly as specified would have left BMR consuming the same
uncalibrated scores it consumes today.**

The cost-sensitive-learning literature is consistent that calibrated probability estimates are the
prerequisite for cost-sensitive thresholding, with Platt scaling, isotonic regression and beta
calibration as the standard remedies (secondary sources; read via search index on 2026-08-30 and
marked `[snippet-inferred]` in the survey).

**What changes:**

* This ADR retains its decision — closed-form empirical-Bayes shrinkage, no MCMC, per
  `CLAUDE.md:68` — and **withdraws the claim that it produces the posterior ADR-0005 needs.**
* The work that would close ADR-0005's calibration gap is **a different, smaller ticket**: an
  isotonic or Platt calibration of model scores fitted on `validate` and applied before the BMR
  argmin, using `scikit-learn` (BSD-3, already a declared dependency). It is strictly cheaper than
  T-0008 was.
* **It was not built and must not be built before the Tue 1 Sep freeze.** T-0023 is a
  documentation ticket and forbids implementing any candidate it surfaces. This addendum records
  the correction; it does not authorise the work.

**Why this is recorded even though it is unflattering:** it removes a tidy explanation the repo
had been relying on. The calibration gap is not the by-product of one cut ticket that could be
un-cut; it is a gap nothing on the board ever addressed. `CLAUDE.md`'s first non-negotiable
applies to decision records as well as to metrics.

Source: `project-context/15-lit-survey-drift-detection.md` §Q3, ADR-0006.
```
