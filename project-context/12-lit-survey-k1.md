# 🔬 Literature Survey: Recovering a Rare, Low-Separation Transitional Latent State (K1 kill-criterion response)

**Date:** 2026-08-28 · **Trigger:** kill criterion K1 fired in T-0004 (four-way state-recovery
ARI 0.091 vs required 0.5; oracle-parameterised ceiling 0.378)
**Sources consulted:** 14 papers · 6 repos/package indices · 12 web searches
**Claims verified at primary source:** 9 (marked `[VERIFIED]`). Everything else is
`[snippet-inferred]` and labelled as such.

> Honest-measurement note, applied to this survey as well as to the project: two of the four
> hypothesised mechanisms are **not** well supported by the literature, and one is **refuted by
> Rakshak's own numbers**. Those are stated plainly below rather than hedged. Where the
> literature is thin (there is essentially no published work on merchant-level *post-onboarding*
> latent-risk-state HMMs, as opposed to card-level transaction HMMs), that is said outright.

---

## 📌 Problem Framing

Rakshak fits a per-merchant-pooled, K=4, Gaussian diagonal-covariance HMM by fully unsupervised
Baum-Welch over 500 merchants × 39 weekly windows × 14 within-merchant-standardised features.
Ground-truth latent occupancy is HEALTHY 90.0% / FRAUD 6.4% / RAMP 3.4% / DORMANT 2.2%.

Measured facts that constrain every recommendation below:

| Fact | Value | What it rules out |
|---|---|---|
| Fitted four-way ARI | 0.091 | — |
| **Oracle-parameterised ARI** (params read from ground truth, no fitting) | **0.378** | Any claim that this is an initialisation / seed / restart problem |
| KMeans(4) on same emissions | 0.107 | Any claim that Markov structure is currently contributing |
| Per-segment fit | 0.021 | Hierarchical/per-segment pooling as a fix |
| RAMP separation from HEALTHY | **1.07 σ** | Any estimator-side fix reaching high four-way recovery |
| Oracle per-state recall, RAMP | 0.373 | Same |
| Binary "non-healthy window" from the *same* fitted model | PR-AUC 0.369 @ 10% base rate, ROC-AUC 0.729 | Any claim that the emissions carry no signal |

**The decisive structural fact.** *The oracle-parameterised HMM **is** the fully-supervised
maximum-likelihood HMM.* Reading emission means/covariances and the transition matrix off the
ground-truth state path is exactly the supervised MLE estimator (Rabiner 1989, §III.C — counting
estimator when state paths are observed). So Rakshak has already, without meaning to, measured
the supervised upper bound of its current model class: **ARI 0.378, RAMP recall 0.373.**

That splits the failure cleanly into two independent gaps, and the whole survey turns on the split:

- **Estimation gap: 0.091 → 0.378 (4.2×).** Caused by the learning objective and the imbalance.
  This gap is *closable*, cheaply, and the literature is unambiguous about how.
- **Representation gap: 0.378 → 1.0.** Caused by RAMP overlapping HEALTHY at ~1 σ in a space
  where HEALTHY holds 90% of the mass. This gap is **not closable** by any estimator, any
  emission family, or any sequence model, on these emissions. The literature is equally
  unambiguous about that.
- **Metric gap.** ARI is, per its own authors' successor paper, the wrong chance-corrected index
  for a reference partition of this shape. Some of the "failure" is measurement artefact.

---

## 🗺️ Landscape Overview

Four paradigm families are relevant, and they attack different gaps. Confusing which gap a method
attacks is the main way to waste the remaining two days.

| Family | Attacks | Representative work | Maturity |
|---|---|---|---|
| **Label-informed generative estimation** (supervised MLE, partially-supervised / weighted-likelihood EM) | Estimation gap | Elworthy 1994; Merialdo 1994; Sidrow et al. 2025; Li et al. 2024 | Consensus |
| **Discriminative sequence labelling** (linear-chain CRF, MMI/MPE-trained HMM) | Estimation gap + conditional-independence violation | Lafferty et al. 2001; Povey & Woodland | Consensus |
| **Structural enrichment** (HSMM durations, sticky priors, IOHMM, richer emission families) | Representation gap — *partially* | Yu 2010; Fox et al. 2011; Bengio & Frasconi 1995; Ruiz-Suarez et al. 2021 | Consensus (methods), Emerging (payoff here) |
| **Changepoint detection** (BOCPD) | A *different problem*: abrupt parameter shifts | Adams & MacKay 2007 | Consensus |

Comparison against Rakshak's constraints:

| Approach | Fixes | Dev-hours | CPU | Licence of any new dep | Explainable reason string? | Maturity | Fit /10 |
|---|---|---|---|---|---|---|---|
| Metric re-specification (AMI + per-state recall + binary PR-AUC + detection lag) | Metric gap | 3 | free | none (sklearn, BSD-3) | n/a | Consensus | **10** |
| **Label-weighted partially-supervised HMM (PHMM)** | Estimation gap | 5 | seconds | **none — edit existing `hmm.py`** | Yes — same Viterbi path, better decoded | Consensus | **9** |
| Deterministic DORMANT + K=3 on non-sparse windows | Frees a wasted state | 2 | seconds | none | Yes | folk-standard | 8 |
| Dirichlet / sticky transition prior + variance floors | EM degeneracy | 2 | free | none | Yes | Consensus | 7 |
| Linear-chain CRF | Estimation gap + feature correlation | 8 | seconds | sklearn-crfsuite 0.5.0, **MIT** | Yes — feature weights, arguably richer | Consensus | 6 |
| BOCPD (already T-0010) | Abrupt typologies only | 6 (already scoped) | seconds | hand-written per ADR | Yes — run-length | Consensus | 5 as baseline, **2 as primary** |
| Emission-family correction (zero-inflated beta / Bernoulli) | Representation gap, second-order | 6 | seconds | none | Yes | Consensus (method) | 4 |
| Hidden semi-Markov model | Duration realism only | 14+ | O(TD²K²) | pyhsmm MIT but **dead**; hsmmlearn **GPLv3 ✗** | Yes | Consensus | **2** |
| IOHMM | Nothing available here | 10 | seconds | none | Yes | Consensus | 2 |

---

## Q1 — Diagnosis

### Q1a. Is ARI the right metric here? **No — and this is settled, published, and citable.**

Romano, Vinh, Bailey & Verspoor, *Adjusting for Chance Clustering Comparison Measures*, JMLR 17
(2016) — the direct successor to Vinh, Epps & Bailey (JMLR 11, 2010) — closes with an explicit
usage guideline. Verbatim from the abstract `[VERIFIED at source]`:

> "ARI should be used when the reference clustering has large equal sized clusters; AMI should be
> used when the reference clustering is unbalanced and there exist small clusters."
> — https://arxiv.org/abs/1512.01286

Rakshak's reference partition is 90.0 / 6.4 / 3.4 / 2.2. That is the textbook definition of
"unbalanced with small clusters". FR-013 specifies the index the literature says not to use for
this shape. ARI is pair-counting based (Hubert & Arabie 1985): the pair count is dominated by the
~81% of pairs that are HEALTHY–HEALTHY, so a partition can be excellent on the three minority
states and still score near zero.

There is a further, independent criticism of ARI's null model: its chance adjustment assumes a
hypergeometric (fixed marginals) null, which *"is not appropriate when the two clusterings are
dependent, it forces the size of the clusters, and it ignores the randomness of the sampling"*
(Adjusting the adjusted Rand Index: A multinomial story, *Computational Statistics* 38(1), 2023)
`[snippet-inferred]`.

**Defensible replacements, in order:**

1. **AMI** (`sklearn.metrics.adjusted_mutual_info_score`, BSD-3, present in current scikit-learn
   `[VERIFIED at docs]`) — the literature's own named substitute for exactly this situation.
   Cost: one line. Report ARI beside it, never instead of it.
2. **Per-state recall reported separately** (already computed in T-0004: 0.868 / 0.373 / 0.550 /
   0.984) and its macro-average, i.e. balanced accuracy — the standard remedy for a metric being
   swamped by a dominant class.
3. **Binary PR-AUC** on "non-healthy window" — Saito & Rehmsmeier, PLOS ONE 10(3):e0118432 (2015),
   the canonical citation that PR curves are more informative than ROC under strong imbalance
   `[VERIFIED at journal page]`.

**Honest caveat, and it matters for the video:** AMI is not a free pass. AMI will read higher than
ARI here largely *because* it weights the small clusters more, which is the point — but swapping
in the metric that flatters you, without also publishing the ARI and the oracle ceiling, is the
goalpost move the panel will smell. See Q3.

### Q1b. Ranking the four hypothesised mechanisms

**#1 — Mechanism 4 (objective mismatch): STRONGLY SUPPORTED. Explains the entire 4.2× estimation gap.**

This is the oldest and best-replicated result in the applied-HMM literature. Elworthy,
*Does Baum-Welch Re-estimation Help Taggers?*, ANLP 1994 (255 citations) finds three distinct
re-estimation regimes, and **in two of them Baum-Welch re-estimation reduces accuracy rather than
improving it**; the paper's headline conclusion is that *"initial biasing of either lexical or
transition probabilities is essential to achieve a good accuracy"*, and which regime you land in
is predictable from the quality of the initial model `[VERIFIED via Semantic Scholar record]` —
https://aclanthology.org/A94-1009/. Merialdo, *Tagging English Text with a Probabilistic Model*,
Computational Linguistics 20(2):155–171 (1994) reaches the same conclusion independently
`[snippet-inferred]`: unsupervised EM degrades a model whenever labels exist and are used.

The mechanism is precisely Rakshak's: unsupervised Baum-Welch maximises `P(observations)`. Under a
90/6.4/3.4/2.2 occupancy, likelihood is overwhelmingly determined by fitting HEALTHY well.
Allocating a state to a 3.4%-mass class sitting 1 σ away *lowers* the likelihood relative to using
that state to model a mode inside HEALTHY. **EM is behaving correctly; it is optimising the wrong
thing.** The 0.091-vs-0.378 gap is the numerical size of "wrong thing".

**#2 — Mechanism 1 (severe latent-class imbalance): STRONGLY SUPPORTED, and it is *why* #4 bites.**

Li, Zhou & Wang, *Gaussian Mixture Model with Rare Events* (arXiv:2405.16859, 2024) analyse EM
under rare-event mixtures as a contraction operator and show *"the spectral radius of the
contraction operator in this case could be arbitrarily close to 1 asymptotically"* — i.e. EM
convergence degenerates as the minority proportion shrinks. **Their proposed remedy is a Mixed-EM
algorithm that leverages partially labelled data** `[VERIFIED at arXiv abstract]`.

Ou, Sen, Young & Dunson, *Targeted stochastic gradient MCMC for HMMs with rare latent states*
(arXiv:1810.13431) frame the same problem for HMMs specifically: with imbalanced data, standard
inference *"often exclude[s] rare latent state data, leading to inaccurate inference"*, and their
fix is deliberate over-sampling / re-weighting of rare-state observations `[VERIFIED at arXiv abstract]`.

Both point the same direction: **re-weight the rare class, or use labels.** Note this also explains
the per-segment collapse (0.021): ~24 merchants per segment with ~2 ever non-healthy leaves EM
almost no minority mass at all — an even more extreme instance of the same mechanism.

Additionally, imbalance is what makes ARI the wrong index (Q1a). Mechanism 1 therefore contributes
to *both* the estimation gap and the metric gap.

**#3 — Mechanism 3 (emission-family mismatch): WEAKLY SUPPORTED. Second-order here, with one
specific exception.**

The most directly relevant paper is Ruiz-Suarez, Leos-Barajas & Morales, *Hidden Markov and
semi-Markov models: When and why are these models useful for classifying states in time series
data?* (arXiv:2105.11490, 2021). Its central finding, verified at source: **model misspecification
does not substantially harm classification performance *unless* there is high overlap between the
state-dependent distributions** `[VERIFIED]`.

That is close to a direct verdict on Rakshak. RAMP-vs-HEALTHY at 1.07 σ *is* the high-overlap
regime. In that regime the paper says the difficulty is intrinsic, not a family-choice artefact —
so switching to zero-inflated beta emissions or a full covariance buys little on the state that
matters. Correcting the family is honest engineering; it is not the fix for K1.

**The one exception worth two hours.** Putting the binary `sparse` indicator through a Gaussian is
a genuine defect, and it is plausibly *costing a state*. DORMANT sits 14.61 σ from HEALTHY driven
almost entirely by `sparse` (11.70). A K=4 model that spends one state on a deterministically
identifiable class has three states left for a 90%-mass HEALTHY plus FRAUD plus RAMP. Removing
DORMANT by rule and fitting K=3 on the remaining windows is a two-hour change with a real chance of
moving the number.

**#4 — Mechanism 2 (7-day window aggregation): REFUTED BY RAKSHAK'S OWN DATA. Do not spend time here.**

The window-size literature is real — a larger window "may smooth over important short-term
anomalies" while a smaller one gives unstable statistics `[snippet-inferred, e.g. arXiv:2504.15375]`
— but it does not apply at Rakshak's numbers. 662 RAMP windows spread over ~100 typology-affected
merchants is **≈6.6 consecutive RAMP windows per affected merchant**, i.e. a ~6-week ramp resolved
into ~7 samples. RAMP is not being averaged into a single window; it is well-resolved and *still*
only 1.07 σ from HEALTHY. Shrinking the window to 3 days would roughly double the sample count and
roughly halve the per-window SNR — the σ figure would likely get *worse*, not better.

**#5 — the mechanism that was not on the list, and is the largest single one.**

**RAMP is 1.07 σ from HEALTHY in the generator itself.** This is a property of the data design, not
of the model. Oracle RAMP recall of 0.373 is the honest ceiling of *any* method operating on these
14 features at this window size. No estimator, no emission family, no duration model, and no
sequence architecture recovers a class that overlaps the 90%-mass class at one standard deviation.
This is not a defect to be fixed in two days; it is a finding to be reported, and it is precisely
the kind of finding CLAUDE.md's non-negotiable #1 exists to protect.

---

## Q2 — Method candidates

### 2.1 Supervised / semi-supervised HMM estimation — **the recommendation**

**What it is.** Two variants, both implementable inside the existing hand-written `hmm.py`:

- *Supervised MLE*: with observed state paths, transition and emission parameters are closed-form
  counts and per-state moment estimates (Rabiner 1989). Rakshak has already run this — it is the
  oracle at ARI 0.378.
- *Partially-supervised / weighted-likelihood EM (PHMM)*: run Baum-Welch, but on labelled windows
  clamp the posterior `γ` to the known state, and **up-weight** labelled observations in the M-step
  so a rare labelled class is not drowned by unlabelled majority mass.

**Primary evidence.** Sidrow, Heckman, McRae, Volpov, Trites, Fortune & Auger-Méthé,
*Incorporating sparse labels into hidden Markov models using weighted likelihoods improves accuracy
and interpretability in biologging studies*, PLOS ONE 20(6):e0325321 (2025) / arXiv:2409.18091.
The method *"increases the relative influence of labelled observations when they are infrequent"*
and reports *"more accurate and understandable decoded latent processes compared to existing
methods"* `[VERIFIED at arXiv abstract]`. Reference implementation (R) at
https://github.com/evsi8432/PHMM — repository licence not verified; **do not vendor it**, the
method is ~30 lines in numpy and Rakshak's ADR-0001 already commits to hand-written HMM code.

Supporting: Elworthy 1994 and Merialdo 1994 (above); Li et al. 2024's Mixed-EM (labels fix the EM
contraction problem); constrained Baum-Welch for partial labels in BMC Bioinformatics 22 (2021),
https://doi.org/10.1186/s12859-021-04080-0 `[snippet-inferred — Springer page behind an auth
redirect, abstract read via search index only]`. Also relevant: semi-supervised HMM work in
*Bioinformatics* 35(13):2208 (2019) reporting up to ~10% accuracy gains over supervised training
when labelled sequences are scarce `[snippet-inferred]`.

**What it buys against THIS failure mode.** It closes the estimation gap and nothing else: expect
to land at or near the 0.378 oracle, i.e. a ~4× improvement on the fitted number, with RAMP recall
around 0.37 rather than near-chance. It will **not** clear a 0.5 four-way ARI gate, and any plan
that assumes it will is fooling itself — the oracle already told us so.

**Cost.** ~5 developer-hours. No new dependency. CPU cost unchanged (15–50 s fit, well inside the
15-minute `make eval` budget). Viterbi path and therefore `explain/reasons.py` are untouched —
in fact decoded paths become *more* interpretable, which is the Sidrow paper's own second claim.

**The honesty cost, which is real and must be paid in the README and the video.** Training the
sequence layer on the generator's own state labels means the model learns Rakshak's assumptions
about what a ramp looks like. That is a stronger version of the synthetic-data limitation already
disclosed. Mitigations, all mandatory: fit labels only on the training merchants under the existing
merchant-group + temporal split; report the unsupervised number beside the label-informed one;
state in the results table that the sequence layer is label-informed on synthetic labels.

### 2.2 Hidden Semi-Markov Models — **reject for this build window**

Explicit duration modelling is genuinely the right structural criticism: an HMM's implicit
geometric dwell time is a poor model for a ramp with a characteristic ~6-week length, and the
prognostics literature does report gains (e.g. duration-dependent HSMMs for remaining-useful-life
estimation, *Mathematical Problems in Engineering*, 2014/2015) `[snippet-inferred]`.

It still loses on every Rakshak constraint:

- **Cost.** Naive HSMM inference is O(TD²K²) versus O(TK²) (Murphy's HSMM notes; Yu, *Hidden
  semi-Markov models*, Artificial Intelligence 174(2), 2010) `[VERIFIED via search of both
  sources]`. With D≈39 that is a ~1500× constant on the message passing. Still CPU-feasible at
  this data size, but no longer a 30-second fit.
- **Libraries are unusable.** `hsmmlearn` is **GPL v3** because it ports R's `hsmm` — a hard
  violation of Rakshak's permissive-licence constraint `[verified at the project's own README]`.
  `pyhsmm` is MIT but carries a *"this package is not maintained anymore"* banner, was last tested
  on **Python 3.7**, and requires a C++11 compiler `[VERIFIED at repo]` — dead on a Python ≥3.11
  target.
- **Hand-writing it** is ~14 hours minimum (duration-augmented forward/backward, right-censoring at
  sequence ends, a duration M-step), which is the entire remaining build window.
- **And it targets the wrong gap.** Ruiz-Suarez et al. (2021) — a paper specifically about HMM *vs*
  HSMM for state classification — says the binding constraint is state-distribution overlap, not
  duration realism `[VERIFIED]`. HSMM sharpens the boundaries of states you can already tell apart;
  RAMP is not one of those.

### 2.3 Linear-chain CRF — **strong second-place candidate**

Directly optimises `P(states | observations)`, so it inherits the same objective-alignment benefit
as supervised HMM training *and* drops the conditional-independence assumption that the correlated
payer-graph features violate (Lafferty, McCallum & Pereira, ICML 2001;
https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf). It attacks mechanisms 3 and 4 at once,
and it is the one candidate that could beat the 0.378 oracle, because the oracle is the ceiling of
the *generative* model class, not of all models.

- **Library:** `sklearn-crfsuite` **0.5.0**, released **2024-06-18**, **MIT** `[VERIFIED at PyPI
  JSON API]`. Flag: upstream `CHANGES.rst` stops at 0.3.6 (2017) `[VERIFIED]` and the repo reads as
  minimally maintained; a community fork exists. Wraps the C++ CRFsuite binary — an extra build
  surface on a laptop, and a licence to re-check for the wheel.
- **Explainability:** not a penalty. Viterbi decoding still yields a per-window state path, and
  CRF feature weights give a *directly attributable* "this merchant was flagged because refund
  ratio and new-payer ratio moved together" string — arguably a better answer to the panel's
  Question 3 than an HMM emission mean.
- **Why second, not first:** ~8 hours plus a new dependency plus a new build surface, against ~5
  hours of pure numpy inside a module that already exists and is already tested. It also abandons
  the "belief over latent risk states, updated per transaction" framing that the pitch is built on,
  and CLAUDE.md locks the hand-written HMM as *"the clearest proof of mathematical depth in the
  repo"*. Adopting a CRF as the primary sequence model is a bigger narrative change than the
  remaining time supports.

### 2.4 BOCPD — **keep as the T-0010 baseline. Do NOT promote to primary.**

This is the survey's clearest negative finding, and it rests on a verbatim quote from the primary
source. Adams & MacKay (arXiv:0710.3742, 2007) open with `[VERIFIED at arXiv abstract]`:

> "Changepoints are **abrupt** variations in the generative parameters of a data sequence… we
> examine the case where the model parameters before and after the changepoint are **independent**."

RAMP is, by construction and by name, a **gradual** transition — and SLOW_RAMP is adversarially
gradual on purpose. BOCPD's generative assumption is violated by exactly the state the product
exists to catch. The changepoint literature acknowledges the limitation explicitly: *"changes in
the behavior of some processes may occur gradually, taking time to reach their full effect"*, which
is the stated motivation for gradual-change extensions (arXiv:2205.01054) `[snippet-inferred]`.

Predicted behaviour, and this is a falsifiable prediction worth writing into T-0010's expectations:
**BOCPD should do well on BUST_OUT and REFUND_COLLUSION (abrupt, and already the two typologies
with the best per-typology ARI: 0.542 and 0.608) and poorly on SLOW_RAMP and LAUNDERING_ENDPOINT.**
If it does, that is a genuinely good result to report — a baseline that is strong exactly where the
HMM is strong and weak exactly where it is weak tells the panel something real about the problem.

If Rakshak wants a changepoint primary anyway, the off-the-shelf option is `ruptures` **1.1.10**,
released **2025-09-10**, **BSD-2-Clause**, Python 3.9–3.13 `[VERIFIED at PyPI]` — but it is
*offline* segmentation, which contradicts the streaming-sentinel product story, and ADR already
commits to a hand-written BOCPD.

### 2.5 Cost-sensitive / class-weighted EM, minority priors, sticky priors — **cheap adjunct, take the 2 hours**

- **Dirichlet prior on the transition rows + a variance floor on emissions.** Standard EM
  regularisation: add prior pseudo-counts to expected counts in the M-step. Directly prevents the
  rare-state degeneracies observed (T-0004 already hit a zero-variance scale cascade sending
  z-scores to 1e8). ~2 hours, no dependency.
- **Sticky self-transition bias**, from Fox, Sudderth, Jordan & Willsky, *An HDP-HMM for systems
  with state persistence* (ICML 2008) and *A sticky HDP-HMM with application to speaker
  diarization*, Annals of Applied Statistics 5(2A):1020–1056 (2011). The finding that matters:
  *"without an extra self-transition bias, the HDP-HMM rapidly transitions among redundant states"*
  `[snippet-verified at https://ics.uci.edu/~sudderth/papers/icml08.pdf]`. A κ term on the diagonal
  of the transition prior is ~5 lines and is the single cheapest way to make the Markov structure
  earn its keep — remember KMeans currently ties the HMM at 0.107, which is evidence the temporal
  structure is contributing nothing.
- **Full sticky HDP-HMM is out of scope** — it needs MCMC, which CLAUDE.md rules out ("no MCMC, no
  PyMC"). Take the κ prior, leave the nonparametrics.
- **Class-weighted EM** is the same idea as the PHMM weighting in §2.1 and should be implemented
  once, there.

### 2.6 Emission-family correction — **partial, cheap slice only**

Full mixed-type emissions (Bernoulli for `sparse`, zero-inflated beta for bounded ratios, Gaussian
for log-features) is well-established — e.g. HMMs with state-dependent zero-inflated beta
distributions for bounded, inflated series (*Wind Energy*, 2026) `[snippet-inferred]` — and would
cost ~6 hours. Ruiz-Suarez et al. say the payoff under high overlap is small `[VERIFIED]`. Take
only the two-hour slice: handle `sparse` outside the Gaussian block (see mechanism #3 above).

### 2.7 IOHMM — **reject**

Input-Output HMMs (Bengio & Frasconi 1995) let transitions and emissions depend on an exogenous
input stream. Rakshak has no such stream: MCC and AOV band are static segment labels already used
for segmentation, and everything else time-varying is already an emission. IOHMM would be
re-labelling existing features as inputs for no informational gain. ~10 hours for nothing.

### 2.8 Anything better under the constraints?

Nothing found. The searched-and-rejected list, for the record: sequence transformers and GNNs
(ADR-0002, and both need GPU); neural-flow emission HMMs (GenHMM) — GPU; deep switching state-space
models (DS³M, arXiv:2106.02329) — GPU and no reason string; sticky HDP-HSMM (Johnson & Willsky,
JMLR 14, 2013) — MCMC, ruled out. **No reopen proposal is warranted. Every ADR-rejected item stayed
rejected on the evidence found; none of it rose anywhere near "extraordinary".**

---

## Q3 — Metric and framing: is re-scoping defensible, or is it moving the goalposts?

**It is defensible, under three conditions, and indefensible without them.**

The re-scoping itself has clean literature support. Evaluating latent-state models on the
downstream decision rather than on state recovery is standard practice in regime-switching work
(state-recovery indices and downstream task performance are typically reported *together*, not
substituted) `[snippet-inferred]`. And Saito & Rehmsmeier (2015) is the canonical justification for
scoring a rare-positive detection task with PR-AUC `[VERIFIED]`. A binary "is this merchant worth an
analyst-hour this week" target is also the objective the cost layer and FR-014 actually consume —
the four-way partition was always an instrument, never the product.

What makes it a goalpost move is *sequencing and omission*, not the re-scope itself. Three
conditions:

1. **Publish the old number.** The 0.091 four-way ARI stays in the results table, permanently.
2. **Publish the ceiling next to it.** The 0.378 oracle is what makes the re-scope credible rather
   than convenient — it demonstrates the ceiling was measured *before* the goalposts moved, not
   after. T-0004 already makes this argument; the survey endorses it.
3. **Publish per-state recall separately**, RAMP's 0.373 included, and say in plain words that
   Rakshak's early-warning state is its weakest state. This is the same discipline that
   CLAUDE.md already mandates for the SLOW_RAMP typology: *"Do not tune it away. Do not hide it."*

**Recommended metric suite for the amended FR-013:**

| Metric | Why | Provenance |
|---|---|---|
| **AMI** (four-way) | The literature's named index for unbalanced references with small clusters | Romano et al., JMLR 17 (2016) `[VERIFIED]` |
| ARI (four-way) | Retained for continuity and honesty; reported as known-pessimistic under this skew | Hubert & Arabie 1985 |
| Per-state recall + macro-average | Prevents the 90% class from setting the headline | balanced-accuracy convention |
| Binary PR-AUC @ base rate | Matches the shipped decision and the cost layer | Saito & Rehmsmeier 2015 `[VERIFIED]` |
| **Detection lag** (windows from true RAMP onset to first alert) | The product claim is *earliness*; a metric that ignores time cannot measure it | early-detection metric literature, e.g. PATE (arXiv:2405.12096) and the TSAD metric taxonomy (arXiv:2511.18739) `[snippet-inferred]` |
| Oracle ceiling for each of the above | Non-negotiable #1 | T-0004 precedent |

Detection lag is the one addition that is *more* demanding than the current gate, not less — which
is exactly what a re-scope needs in order to read as a sharpening rather than a retreat.

---

## ⚡ Must-Read (in priority order, ~3 hours total)

1. **Romano, Vinh, Bailey & Verspoor (2016), JMLR 17** — https://arxiv.org/abs/1512.01286 —
   abstract alone settles the metric question. *Read this first; it is 10 minutes and it changes
   what FR-013 should say.*
2. **Ruiz-Suarez, Leos-Barajas & Morales (2021)** — https://arxiv.org/abs/2105.11490 — the paper
   that tells you HSMM will not save you and that overlap is the binding constraint.
3. **Sidrow et al. (2025), PLOS ONE 20(6):e0325321** — https://arxiv.org/abs/2409.18091 — the
   method to implement, plus the interpretability argument you can quote in the video.
4. **Elworthy (1994), ANLP** — https://aclanthology.org/A94-1009/ — 30-year-old proof that
   unsupervised Baum-Welch can be worse than not running it.
5. **Li, Zhou & Wang (2024)** — https://arxiv.org/abs/2405.16859 — the theory of *why* EM
   degenerates under rare events, and why labels fix it.
6. **Adams & MacKay (2007)** — https://arxiv.org/abs/0710.3742 — read the first sentence and write
   it into T-0010's expected-results section.
7. **Fox et al. (2008), ICML** — https://ics.uci.edu/~sudderth/papers/icml08.pdf — the κ
   self-transition bias, 5 lines of code.
8. **Saito & Rehmsmeier (2015), PLOS ONE** —
   https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432 — the PR-AUC citation.
9. **Ou, Sen, Young & Dunson** — https://arxiv.org/abs/1810.13431 — rare latent states in HMMs
   specifically; skim the abstract only.

---

## 🔧 Open-Source Starting Points (all licence-checked)

| Repo / package | Version | Licence | Status | Use |
|---|---|---|---|---|
| `scikit-learn` (`adjusted_mutual_info_score`, `precision_recall_curve`) | current | BSD-3 | ✅ already a dep | The whole metric fix |
| `sklearn-crfsuite` | 0.5.0 (2024-06-18) `[VERIFIED PyPI]` | **MIT** | ⚠️ upstream changelog stale since 2017; wraps C++ CRFsuite | Only if the CRF fallback is taken |
| `ruptures` | 1.1.10 (2025-09-10) `[VERIFIED PyPI]` | BSD-2 | ✅ healthy | Cross-check only; offline, not the streaming story |
| `evsi8432/PHMM` | — | not verified | R reference code | **Read, do not vendor** |
| `pyhsmm` | — | MIT | ❌ *"not maintained anymore"*, Python 3.7, C++ build `[VERIFIED repo]` | Do not use |
| `hsmmlearn` | — | **GPL-3.0** `[per its own README]` | ❌ **licence violation** | Do not use |
| `pomegranate` | 1.1.2 | MIT | ⚠️ torch backend, low recent activity `[snippet-inferred]` | Not needed |
| `hmmlearn` | 0.3.3 | BSD-3 | ❌ ADR-0001 | Stays rejected |

---

## ⚠️ Open Problems, Risks & Where the Literature Is Thin

- **The domain literature is genuinely absent.** Published HMM fraud work is almost entirely
  *card-level* transaction-sequence modelling (Srivastava et al. 2008 and its long tail of
  re-implementations). The nearest merchant-centric work is prepaid-card *store-level* HMM
  divergence monitoring (*Expert Systems with Applications*, 2017). There is **no published
  benchmark for post-onboarding merchant latent-risk-state recovery**. Rakshak has no external
  yardstick, which cuts both ways: nothing to lose to, and no independent validation of the
  problem framing. Say this in the README.
- **Everything above is measured on synthetic data with generator-defined labels.** Adopting a
  label-informed estimator deepens that dependency. It does not invalidate the approach — the
  entire sequence layer already carries this caveat — but it moves the caveat from "we trained
  unsupervised on synthetic streams" to "we trained on synthetic labels", which is a materially
  stronger claim for a panel to discount.
- **AMI has its own known biases** under differing cluster-count assumptions; Gates & Ahn's
  element-centric similarity (arXiv:1706.06136) argues both pair-counting and IT measures have
  blind spots `[snippet-inferred]`. AMI is defensible, not perfect. Report both.
- **RAMP at 1.07 σ may be a generator artefact rather than a fact about fraud.** If real ramps
  separate more cleanly than Rakshak's generator produces, the pessimism here is over-stated; if
  they separate less, it is under-stated. There is no way to know without real data, and that
  uncertainty belongs in the video.
- **The κ / Dirichlet priors are effectively free but effectively untested here.** No claim is
  made about their magnitude of effect; measure, then report.

---

## 💡 Contrarian View

**The most valuable thing in this failure may be the failure itself.**

Rakshak has a measured oracle ceiling (0.378), a measured unsupervised floor (0.091), a measured
non-sequential control (KMeans 0.107), and a measured per-state overlap table. Almost no buildathon
submission will have any of those, and none will have all four. A results section that says *"our
early-warning state is 1.07 σ from healthy, here is the perfect-information ceiling, here is what
we actually achieved, and here is the gap we cannot close"* is a stronger artefact for a Head of
Risk Operations than a submission claiming 0.94 on a metric nobody interrogated.

The contrarian recommendation, then: spend the ~8 hours on the estimation gap and the metric suite,
and spend the *rest* of the remaining time on the failure-analysis section and the cost layer —
not on chasing the four-way ARI. FR-013 was an instrument that has now done its job. It told you
the emissions cannot support four-way recovery. That is information, and it was expensive to buy;
spending two more days trying to unbuy it is the actual risk.

---

## 🏆 RANKED RECOMMENDATION TABLE

| # | Approach | What it fixes | Dev-hours | CPU cost | Licence | Confidence |
|---|---|---|---|---|---|---|
| **1** | **Metric re-specification: AMI + per-state recall + macro-recall + binary PR-AUC + detection lag, ARI and oracle retained** | The **metric gap**. FR-013 currently specifies an index its own authors' successor paper says not to use for a 90/6.4/3.4/2.2 reference | **3** | negligible | BSD-3 (sklearn, existing dep) | **High** — verbatim guideline verified at source |
| **2** | **Label-weighted partially-supervised HMM (PHMM) in existing `hmm.py`: clamp γ on labelled training windows, up-weight rare labelled states in the M-step** | The **estimation gap**, 0.091 → ~0.378 (~4×) | **5** | unchanged (~50 s fit) | none — no new dependency | **High** — Sidrow 2025 + Elworthy 1994 + Li 2024, all verified |
| **3** | Deterministic DORMANT rule on `sparse`, then fit K=3 on the remainder | Stops a trivially separable state consuming 25% of model capacity; also fixes the worst emission-family violation | **2** | unchanged | none | Medium — mechanism is sound, magnitude unmeasured |
| **4** | Dirichlet transition prior + κ self-transition bias + emission variance floors | EM degeneracy; makes the Markov structure earn its keep (KMeans currently ties the HMM) | **2** | negligible | none | Medium — Fox et al. 2008 |
| **5** | Linear-chain CRF as a *reported alternative model*, not a replacement | Estimation gap + conditional-independence violation together; only candidate that can exceed the generative oracle | **8** | seconds | MIT (`sklearn-crfsuite` 0.5.0) | Medium — strong method, weak library maintenance, narrative cost |
| **6** | BOCPD **as the T-0010 baseline, with a written prediction that it wins on abrupt typologies and loses on RAMP** | Nothing about RAMP. Provides a diagnostic contrast that is itself a result | 6 (already scoped) | seconds | hand-written | **High** — Adams & MacKay's own abstract |
| **7** | Full mixed-type emissions (Bernoulli / zero-inflated beta) | Representation gap, second-order | 6 | seconds | none | Low payoff — Ruiz-Suarez 2021 verified |
| **8** | Hidden Semi-Markov Model | Duration realism only | **14+** | O(TD²K²), ~1500× constant | pyhsmm dead / **hsmmlearn GPLv3 ✗** | **Reject** |
| **9** | IOHMM | Nothing — no exogenous inputs exist | 10 | seconds | none | **Reject** |
| **10** | Promoting BOCPD to primary | Would make the failure worse on the state that matters | — | — | — | **Reject** |

---

## ✅ PRIMARY RECOMMENDATION

**Do #1 + #2 + #3 + #4 — a single ~12-hour work package — and amend FR-013 rather than abandon it.
Keep the hand-written HMM. Do not build an HSMM. Do not promote BOCPD.**

**Reasoning.** The T-0004 measurements have already partitioned this failure for us, and the
partition determines the answer. The oracle-parameterised run *is* the supervised-MLE HMM, so
Rakshak has measured both an estimation gap (0.091 → 0.378, closable) and a representation gap
(0.378 → 1.0, not closable on these emissions). The literature is decisive on both halves and it
points in opposite directions: on estimation, Elworthy (1994) and Merialdo (1994) established
thirty years ago that unsupervised Baum-Welch degrades a model whenever labels exist, Li et al.
(2024) explain analytically why EM degenerates as a class becomes rare, and Sidrow et al. (2025)
give a weighted-likelihood recipe that is thirty lines of numpy inside a module Rakshak already
owns and already tests — so half a day recovers a 4× improvement with no new dependency, no CPU
cost, and no damage to the Viterbi-path explainability that is the project's differentiator. On
representation, Ruiz-Suarez et al. (2021) state directly that misspecification is largely benign
*except* under high overlap between state-dependent distributions, which is precisely the
1.07 σ RAMP-vs-HEALTHY regime — so every expensive structural remedy (HSMM durations, mixed-type
emissions, IOHMM) is aimed at a gap it cannot close, at a cost of one to two days each, and two of
the three are additionally blocked by a GPL licence or a dead Python-3.7 repository. Meanwhile
Romano et al. (JMLR 2016) supply a verbatim, citable rule that ARI is the wrong chance-corrected
index for an unbalanced reference with small clusters, which means part of the measured "failure"
is a measurement artefact that costs three hours and one scikit-learn call to correct. The package
therefore buys the entire closable gap plus a correctly-specified gate for roughly one and a half
of the two remaining days, and leaves the rest for the cost layer and the failure-analysis write-up
— which, per the contrarian section, is where the marginal hour is actually worth most.

**What success looks like (state these as the amended gate before running, not after):**
four-way AMI reported with ARI beside it; label-informed four-way ARI in the 0.30–0.40 band
(matching the oracle, *not* exceeding it — exceeding it means a leak, check `eval/splits.py`);
RAMP recall ≥ 0.35; binary PR-AUC > 0.40 at a 10% base rate; detection lag reported per typology;
oracle ceiling reported for every one of these.

**What it explicitly does NOT promise:** a four-way ARI of 0.5. That number is not reachable on
these emissions and the oracle proved it before this survey started.

## 🛟 FALLBACK (if #2 has not landed by end of day 1)

**Ship the binary decision framing on the already-fitted unsupervised model, unchanged.**

T-0004 already measured it: PR-AUC 0.369 at a 10% base rate, ROC-AUC 0.729, a 3.7× lift, from a
model in hand. Zero new modelling, zero new risk. Combine with #1 (metrics, 3 h) and #6 (BOCPD as
the scoped T-0010 baseline) and Rakshak still has a defensible, honestly-measured, fully
explainable submission: a per-merchant belief over latent states, a Viterbi reason string, a cost
layer under a capacity constraint, a changepoint baseline, and a published account of exactly where
and why the four-way recovery failed. The fallback is not a degraded product — it is the same
product with a smaller claim.

**Explicit non-fallback:** do not, under time pressure, collapse RAMP into HEALTHY and report a
3-state recovery. It would raise the number and delete the early-warning claim that the entire
project premise rests on. T-0004's recommendation against option (2) stands and this survey
seconds it.

---

## 📄 ADR STUB

```
# ADR-0009 — Response to K1: label-informed HMM estimation and re-specified recovery metrics
# (drafted here as ADR-0005; renumbered 2026-08-29 because 0005 was already the
#  three-action policy. PROMOTED to docs/adr/ADR-0009-k1-label-informed-hmm.md,
#  which is now authoritative and records what T-0004b actually measured.)

Status: PROPOSED
Date: 2026-08-28
Supersedes: none. Amends: FR-013 (06-requirements.md). Related: ADR-0001 (hand-written HMM).

## Context

FR-013's four-way state-recovery gate (ARI > 0.5) failed at 0.091 (T-0004). The
oracle-parameterised HMM — which is the fully-supervised MLE estimator — reaches only 0.378,
so the ceiling sits below the gate. RAMP, the commercially load-bearing early-warning state,
is 1.07 sigma from HEALTHY, which holds 90% of window mass; oracle RAMP recall is 0.373.

## Options considered

A. Keep FR-013 as written; report the failed gate. (Honest, hands a competitor the headline.)
B. Collapse RAMP into HEALTHY, score 3-state recovery. (Deletes the early-warning claim.)
C. Build an HSMM with explicit RAMP duration. (14+ h; hsmmlearn is GPLv3, pyhsmm is
   unmaintained/Py3.7; Ruiz-Suarez 2021 indicates overlap, not duration, is the binding
   constraint.)
D. Promote BOCPD from baseline to primary. (Adams & MacKay define changepoints as ABRUPT
   parameter variations; RAMP is gradual by construction. Wrong tool for the target state.)
E. Replace the HMM with a linear-chain CRF. (8 h + new dependency + abandons the locked
   generative framing; retained as a reported alternative model, not as the primary.)
F. Label-informed (partially-supervised, weighted-likelihood) estimation inside the existing
   hand-written HMM, PLUS re-specified recovery metrics.

## Decision

Adopt F, with A retained inside it: the 0.091 four-way ARI and the 0.378 oracle ceiling are
reported permanently alongside every new number.

1. Amend FR-013's metric suite to: AMI (primary four-way index), ARI (retained), per-state
   recall and its macro-average, binary non-healthy PR-AUC at base rate, and per-typology
   detection lag — each reported with its oracle ceiling.
2. Replace fully-unsupervised Baum-Welch with weighted-likelihood partially-supervised EM:
   clamp gamma to known states on labelled TRAINING-split windows only, up-weight rare
   labelled states in the M-step.
3. Handle DORMANT by a deterministic rule on `sparse` and fit K=3 on the remainder.
4. Add a Dirichlet transition prior with a sticky self-transition term and emission variance
   floors.
5. Keep BOCPD as the T-0010 baseline with a pre-registered prediction: strong on BUST_OUT and
   REFUND_COLLUSION, weak on SLOW_RAMP.

## Rationale

Romano/Vinh/Bailey/Verspoor (JMLR 17, 2016) state that ARI is for balanced references and AMI
for unbalanced references with small clusters; Rakshak's reference is 90/6.4/3.4/2.2.
Elworthy (ANLP 1994) and Merialdo (CL 1994) establish that unsupervised Baum-Welch degrades
accuracy when labels exist; Li/Zhou/Wang (2024) show EM's contraction radius approaches 1 under
rare-event mixtures and that partial labels fix it; Sidrow et al. (PLOS ONE 2025) give the
weighted-likelihood recipe and report improved accuracy AND interpretability. Ruiz-Suarez et al.
(2021) find misspecification is largely benign except under high state-distribution overlap,
which demotes emission-family and duration remedies here. Total cost ~12 developer-hours, no new
dependency, CPU-only, no change to Viterbi-path explainability.

## Consequences

+ Estimation gap closes (~4x); metric matches the reference partition's shape.
+ No new dependency, no licence exposure, Viterbi reason strings unaffected.
- The sequence layer becomes label-informed on SYNTHETIC generator labels. This is a stronger
  limitation than the existing synthetic-data caveat and MUST be stated verbatim in the README,
  the video, and every results table.
- Four-way ARI will still not reach 0.5. FR-013's original threshold is retired as unreachable
  on these emissions, with the oracle as evidence.

## Revisit trigger

- Label-informed four-way ARI materially EXCEEDS 0.378 -> suspect leakage; audit eval/splits.py
  before believing it.
- Any real (non-synthetic) merchant stream becomes available -> re-measure RAMP separation; the
  1.07 sigma figure is a property of the generator, not of fraud.
- A future build window >= 3 days AND a permissively-licensed maintained HSMM library -> revisit
  option C.
```

---

## 🔍 WHAT WOULD CHANGE OUR MIND

Stated in advance, as falsifiable conditions:

1. **RAMP separation turns out to be a generator artefact.** If re-parameterising the generator
   (or any real data) puts RAMP at ≥ 2 σ from HEALTHY, the representation gap largely evaporates,
   the four-way gate becomes reachable, and structural remedies (HSMM, mixed-type emissions)
   become worth their cost. **Cheapest test: 30 minutes — re-run the σ table with the generator's
   ramp amplitude scaled 2×.** If ARI jumps, the problem was the data design all along and this
   survey's ranking inverts. *This is the single highest-value half-hour available.*
2. **Label-informed fitting overshoots the oracle.** If the PHMM lands materially above 0.378,
   something is wrong — most likely label leakage across the merchant-group or temporal split. Do
   not celebrate; audit `eval/splits.py`.
3. **Label-informed fitting lands near 0.091.** If clamping γ on labelled training windows barely
   moves the number, mechanism 4 is wrong and the estimation gap is not where we think it is. Next
   suspect: the pooled-vs-per-merchant parameterisation, since the oracle reads per-*state*
   parameters that a pooled fit may be unable to represent at all. Escalate to the CRF (#5), which
   sidesteps the generative parameterisation entirely.
4. **BOCPD beats the HMM on SLOW_RAMP.** That would falsify the Adams & MacKay reading applied
   here and would justify promoting changepoint detection. Prediction stated in advance so it can
   be checked rather than rationalised.
5. **The κ sticky prior alone closes most of the gap.** Would mean the failure was EM degeneracy
   rather than objective mismatch, promoting mechanism 1 over 4 and making the whole label-informed
   package unnecessary. Two hours to find out; run it before the PHMM if time is tight.
6. **A maintained, permissively-licensed HSMM library appears** (or the build window extends past
   Monday). Reopens option C — but only in combination with #1, since duration modelling cannot
   fix overlap.
7. **The panel signal changes.** If Razorpay's evaluation criteria were to emphasise state-recovery
   fidelity over decision quality and honest measurement, the re-scope in Q3 becomes much harder to
   defend and option A (report the failed gate as-is, change nothing) becomes the better play.

---

### Provenance

Every claim marked `[VERIFIED]` was read at its primary source (arXiv abstract page, PyPI JSON API,
GitHub repository, or scikit-learn documentation) during this survey on 2026-08-28. Claims marked
`[snippet-inferred]` come from search-result summaries only and were not confirmed at the source —
none of them carries a top-three recommendation. Two sources could not be opened directly and are
labelled accordingly: the Springer page for BMC Bioinformatics 22 (2021) redirects to
authentication, and the Elworthy (1994) PDF would not parse, so its content was verified via the
Semantic Scholar record for DOI 10.3115/974358.974371 rather than the PDF itself.
