<!-- HEAD
FILE:     project-context/13a-survey-quickest-change-detection.md
PHASE:    survey (cycle 4)
UPDATED:  2026-09-01
STATUS:   ready — one method named, one numeric gate pre-registered, no new dependency
SUMMARY:  Quickest change detection (QCD) surveyed against the two cycle-3 failures. QCD
          cannot fix the unmeasurable-TTD failure (that was split geometry, fixed by the
          onset-window widening); it is the right literature for the volume_rank failure,
          because a static ranking is by construction a zero-delay-information detector.
          First place: a per-merchant Page/CUSUM recursion run on the WITHIN-DAY CROSS-
          SECTIONAL RANK of the incumbent cohort-residual score, ranked top-K. 8 bytes of
          state per merchant, ~25 lines of numpy, 3 fitted parameters, no new library.
OPEN:     (a) No literature found on QCD under a HARD per-period review capacity K rather
          than an ARL or FDR constraint — the operating constraint here has no theory.
          (b) No literature found on delay-optimality when the loss rate is proportional to
          a stream-specific exposure (merchant GMV). That gap is exactly why volume_rank
          wins on money and QCD may not take it back.
          (c) At ~14 evaluable onsets the d7/d14 arms are almost certainly unpowered; the
          gate below is written as a PAIRED test for that reason and may still return
          INSEPARABLE.
-->

# 13a — Survey: quickest change detection for a detection-latency objective

## 1. What this literature is for

Two things failed in cycle 3. They are not the same kind of thing and this literature
speaks to only one of them.

**Failure 1 — time-to-detection was never measurable.** `detection_rate_d7`, `d14` and
`d30` read `0.000` for every rung *and* every naive floor. Ground truth: 294 fraud
merchants, `drift_onset_at` min 30, median 108.5, max 217; the validation window opens on
day 240. d7 needs onset >= 233 and d14 needs onset >= 226, so zero merchants were eligible;
d30 was reachable by 4 of 294 before the fold split and censoring filter took their cut.
This is an arithmetic impossibility in the split geometry, not a modelling deficit. **No
method in this survey — not CUSUM, not an e-detector, not a multi-stream mixture rule —
could have moved those numbers in cycle 3.** The fix is the one cycle 4 already makes:
widen the onset window to days 30-364 so that ~14 onsets land in validation (240-299) and
~13 in test (300-364). QCD contributes nothing to this failure except the observation that
it was never a modelling failure. Say so plainly in the writeup, because reporting an
impossibility as a score is a worse bug than the score.

**Failure 2 — every learned rung loses on money to a size ranking.** `volume_rank` alerts
the K largest merchants by pre-window GMV, the same K merchants every day (week-over-week
alert Jaccard exactly 1.000), and scores 0.6016 normalised savings against the best learned
rung's 0.4348 — while ranking at PR-AUC 0.217 against the learned rung's 0.836 and
precision@K 0.864. **This is the failure QCD addresses, and it is the only one.** A Jaccard
of 1.000 is the signature of a detector whose statistic is a function of *level* and not of
*time*: it emits the same alert set on the day before an onset and the day after. Every
method in this survey has, as its explicit objective function, expected delay from the onset
subject to a false-alarm constraint. That objective *cannot* be optimised by a time-invariant
alert set. So QCD is the literature that attacks the failure at its root, and Jaccard is the
mechanism check that says whether a candidate has done so.

**But be honest about the limit.** The metric that `volume_rank` wins is *savings*, a
cost-weighted allocation metric, not a delay metric. Under the project's loss model, money
lost accrues at a rate roughly proportional to merchant size, so ranking by size buys
exposure-weighted expected loss directly. QCD buys timing. These are orthogonal factors of
the same product, and the eventual winning rung is almost certainly `evidence-of-change x
exposure`, not one replacing the other. This survey names the change factor. It does not
claim that factor alone beats 0.6016, and §8 argues seriously that it will not.

---

## 2. The map

### 2A. Classical parametric QCD — the optimality results, and what they actually promise

Four procedures and three optimality criteria. Being precise about which is which matters,
because the criteria differ and the folklore blurs them.

| Procedure | Statistic (log form) | Minimises | Subject to | Optimality status |
|---|---|---|---|---|
| Page / CUSUM | `C_t = max(0, C_{t-1} + LLR_t)` | Lorden's worst-case delay `sup_v ess sup E_v[(T-v)^+ given F_v]` | `ARL_0 = E_inf[T] >= gamma` | **Exactly** optimal for every gamma>1 (Moustakides 1986); asymptotically optimal (Lorden 1971) |
| Shiryaev-Roberts (SR) | `R_t = (1 + R_{t-1}) * Lambda_t` | Integral / stationary average delay | `ARL_0 >= gamma` | Exactly optimal in the multi-cyclic (stationary-regime) sense |
| Shiryaev-Roberts-Pollak (SRP) | SR with random head start from the quasi-stationary law | Pollak's `sup_v E_v[T-v given T>v]` | `ARL_0 >= gamma` | Third-order asymptotically optimal (Pollak 1985); **NOT exactly optimal** — counterexample in Polunchenko & Tartakovsky (2010), where SR-r with a *deterministic* head start is strictly better |
| Shiryaev (Bayesian) | posterior odds of change | `E[(T-v)^+]` under a geometric prior on v | `P(false alarm) <= alpha` | Exactly optimal in the Bayesian formulation |

Read the "Minimises" column carefully. **Lorden's criterion (CUSUM) is a double worst case**
— worst change time, worst pre-change history. **Pollak's criterion (SRP / SR-r) is a
conditional average over histories.** Both are "detection delay under a false-alarm
constraint"; neither is classification accuracy. That is the property the spec asked me to
weight, and every row of this table has it.

**Assumptions, all four:** observations i.i.d. before the change, i.i.d. after it, and both
densities `f0`, `f1` *known*. The statistic is a likelihood ratio; if you get the likelihood
wrong, you are running a valid recursion on the wrong number.

**Where this breaks on Rakshak data.** Daily per-merchant counts have measured Fano factor
12.25-12.37 against Poisson's 1.0. A Poisson LLR would be off by an order of magnitude in
its variance and would fire on ordinary Tuesdays. A Gaussian LLR with a plug-in variance is
better but still assumes a fixed, known post-change mean shift, which the generator does not
provide. Additionally: daily counts are weekday-seasonal and platform-confounded, so the
"i.i.d. pre-change" premise fails at the panel level, not just the stream level. And the
`ARL_0 >= gamma` constraint — the thing every threshold in this table is calibrated to — is
**irrelevant here**, because the capacity layer already fixes the alert rate at exactly K per
day. See §2C; this is the single most consequential structural observation in the survey.

**Partial repair inside the family:** GLR-CUSUM / window-limited GLR (Lai) replaces the
known post-change parameter with a maximised one over a sliding window, which removes the
"known `f1`" assumption at O(window) cost per stream per day. It does not remove the "known
parametric family" assumption, which is the one that hurts at Fano 12.3.

### 2B. Non-parametric and distribution-free QCD — the branch that survives overdispersion

Five sub-families, in rough order of how well they fit a 4-core CPU budget:

1. **Rank / normal-score / empirical-quantile CUSUM.** Replace the LLR increment with a
   monotone transform of the observation's *rank* within a reference set. Under the no-change
   null the rank is uniform by construction, so the increment has a known mean whatever the
   raw distribution is — overdispersion, zero-inflation and weekday seasonality all become
   irrelevant if the reference set is drawn correctly. Cost: O(1) per stream per day given a
   per-day ranking. This is the cheapest way to buy distribution-freeness and it is what §4
   recommends.
2. **e-detectors / betting-based sequential change detection** (Shin, Ramdas & Rinaldo,
   arXiv:2203.03532). An e-detector is a sum of e-processes started at each candidate change
   time; it yields **non-asymptotic** average-run-length bounds *without* knowing either
   distribution, plus near-optimal delay bounds in nonparametric classes (sub-Gaussian,
   sub-exponential). This is the theoretically correct answer to "the likelihood is
   misspecified": it replaces the likelihood ratio with a wealth process that is a
   supermartingale under *any* null in the class, and classical CUSUM is recovered as a
   special case. Cost is the catch: the naive form is O(t) per stream (one e-process per
   candidate changepoint), tamed to O(log t) by geometric partitioning of the post-change
   parameter space. Non-partitioned variants exist (arXiv:2607.28322) — **UNVERIFIED** beyond
   title and abstract listing; I did not read it.
3. **Conformal test martingales** (Vovk et al., PMLR v152, 2021; Volkhonskiy et al.,
   arXiv:1706.03415, inductive conformal martingales for CPD). Distribution-free under an
   exchangeability null, betting on conformal p-values. Needs a per-stream calibration set;
   with a 30-day frozen warmup baseline per merchant, that exists. Closely related to (2) —
   e-detectors and conformal martingales are two dialects of the same idea.
4. **Kernel / MMD change detection.** Scan-B (Li, Xie, Dai & Song, arXiv:1507.01279) keeps B
   reference blocks and computes an online MMD with a change-of-measure tail approximation for
   threshold setting; NEWMA (arXiv:1805.08061) gets O(1) per-step cost via random features and
   two EWMAs at different rates; graph / nearest-neighbour tests (Chen 2019,
   doi:10.1214/18-AOS1718) are fully nonparametric and multivariate. All of these are built
   for *multivariate* streams where the change is in a joint distribution. On a 20k-40k
   merchant panel with a per-merchant reference block, the per-day cost is B^2 kernel
   evaluations per merchant — a 10 ms/merchant p99 budget makes this tight but not impossible;
   the bigger objection is that they detect *any* distributional change, not a directional
   increase, and fraud is directional.
5. **Classic distribution-free monitoring** (nonparametric CUSUM on a bounded score with a
   known sign; open-end retrospective-CUSUM monitoring, arXiv:2007.08369). Mature, boring, and
   effectively what (1) is.

**Where each breaks here:** every one of (1)-(5) needs a clean pre-change reference. The
project's frozen 30-day warmup baseline supplies it, and the freeze is already a deliberate
anti-slow-ramp (R2) decision. What none of (1)-(5) solves on its own is the platform-wide
confounder: a festival spike moves *every* merchant's reference comparison at once, and a
per-stream distribution-free detector will happily fire on all of them simultaneously. That is
§2D's problem, not this section's.

### 2C. Multi-stream QCD — the heart of it, and the structural surprise

The real problem is not N independent detectors. It is: *which* of N merchants changed, under
one global budget. The literature:

- **Mei (2010), Biometrika 97(2):419-433, doi:10.1093/biomet/asq010** — SUM-CUSUM: add the
  local CUSUM statistics across streams and threshold the sum. Asymptotically minimises
  detection delay for *every* combination of affected streams subject to a global false alarm
  constraint. This is the foundational scalable scheme.
- **Xie & Siegmund (2013), Ann. Statist. 41(2):670-692, doi:10.1214/13-AOS1094** — a mixture
  procedure: local generalised-LR statistics combined under an assumed affected fraction `p0`,
  no spatial structure assumed. Better than SUM when the affected set is sparse and the
  post-change mean is unknown.
- **Chan (2017), Ann. Statist. 45(6):2736-2763, doi:10.1214/17-AOS1546** — the sparsity
  taxonomy. Three detection domains as the affected fraction shrinks: immediate detection;
  delay growing logarithmically in N with a sparse-mixture constant; and finally Lorden's
  classical delay regime. Directly relevant: **Rakshak's affected fraction is tiny** (~14 fresh
  onsets among 20k-40k merchants, ~4e-4), which puts it deep in Chan's third and hardest
  domain — where the multi-stream gain over a single-stream detector is smallest.
- **Fellouris & Sokolov** — second-order asymptotically optimal generalised and mixture-based
  rules (cited in the Tartakovsky and Halme reviews; **primary source UNVERIFIED**, I read it
  only through those reviews).
- **Top-r and hard-thresholded SUM (SUM-shrinkage)** — sum only the local statistics that are
  "large", by hard threshold or by top-r selection. Retains asymptotic optimality and is the
  scalable practical form (Liu, Zhang & Mei; surveyed in the reviews below).
- **Tartakovsky & Spivak, arXiv:2305.07834**, and **Tartakovsky et al., arXiv:1807.08971** —
  multistream QCD in general non-i.i.d. models, plus applications.
- **Halme & Koivunen, "Multi-stream Quickest Change Detection: Foundations and Recent
  Advances", arXiv:2604.18008 / Entropy 28(5):566, doi:10.3390/e28050566** — the current
  review; read this one first if only one is read.
- **Identification under an error-rate budget:** Halme & Koivunen, "Optimal Multi-Stream
  Quickest Detection with False Discovery Rate Control", Asilomar 2023, pp. 877-881
  (**UNVERIFIED page range** — taken from a secondary listing, not the proceedings); and the
  online-FDR machinery it builds on: Javanmard & Montanari, arXiv:1502.06197 (LOND, LORD),
  LORD++ / SAFFRON and their positive-dependence guarantees.
- **Shrinkage across streams:** Halme, Veeravalli & Koivunen, "Quickest Change Detection for
  Multiple Data Streams Using the James-Stein Estimator", arXiv:2404.05486, IEEE Trans. Inform.
  Theory 71(10):7802-7814 (2025). Pools information across the panel to estimate unknown
  post-change means — a principled cousin of the cohort construct, though it shrinks
  post-change parameters rather than removing a common component.

**The structural surprise, and it is the most useful thing in this file.** Every rule above —
SUM, mixture, top-r, FDR-controlled — is an *aggregation-plus-threshold* device: combine local
statistics into one global number, compare to a threshold calibrated to `ARL_0 >= gamma` or
`FDR <= alpha`, raise one alarm. **Rakshak needs none of that.** The decision-policy seam
already performs the aggregation (rank merchants, take the largest) and the capacity layer
already fixes the alert rate (exactly K per day, always, no threshold to calibrate). What
Rakshak needs from this literature is precisely the *local statistic* that every one of these
rules is built out of, and nothing else.

Two consequences:

1. **Adopting "top-r thresholding of local CUSUM statistics" is a zero-line change**, because
   the existing rung interface plus capacity layer *is* top-r. The implementation cost of the
   multi-stream branch is nil beyond the local statistic.
2. **Almost all of the optimality theory is void here.** Those theorems are statements about
   thresholds under a false-alarm constraint. With a hard capacity K there is no threshold and
   no false-alarm constraint — only an *ordering*. The theory tells you the local statistic is
   the right object to rank by; it does not tell you the ranking is optimal. Nobody has proved
   anything about hard-capacity QCD (§7).

**Online FDR (LORD / SAFFRON) — do not adopt.** It solves a different constraint (expected
proportion of false alerts) than the one Rakshak has (hard headcount). Running LORD *inside* a
fixed-K budget would either under-spend the budget (analysts idle) or be overridden by it (the
alpha-wealth accounting becomes decorative). Note it, reject it, move on.

### 2D. Panel change detection with a shared nuisance signal — is the cohort residual principled?

**Short answer: yes, and the literature also names its failure mode.**

The construct in question: a merchant's drift z-score minus the leave-one-out cohort median
z-score. A platform-wide festival spike, gateway outage or fee change moves every merchant in
the cohort, moves the cohort median with them, and cancels in the residual.

The literature's principled version is **factor adjustment**: model the panel as
`X_it = (common component) + (idiosyncratic component)`, estimate and remove the common
component, then detect changes in the idiosyncratic part.

- **Barigozzi, Cho & Fryzlewicz (2018)**, "Simultaneous multiple change-point and factor
  analysis for high-dimensional time series", *J. Econometrics* 206(1):187-225,
  arXiv:1612.06928. They estimate change-point number and locations *and* attribute each to the
  common or the idiosyncratic component. They report that factor analysis prior to change-point
  detection **improves detectability**, which is the direct theoretical endorsement of the
  cohort residual. They also report a **spillover effect**: substantial breaks in idiosyncratic
  components get mis-attributed to the common component. That is precisely the risk in a
  leave-one-out *mean*; using the **median** bounds it (50% breakdown point), so the project's
  existing choice of median is the right one and should be defended in the writeup on those
  grounds rather than on intuition.
- **Cho (2016)**, "Change-point detection in panel data via double CUSUM statistic", *Electron.
  J. Statist.* 10(2), arXiv:1611.08631 — panel CUSUM that aggregates cross-sectionally while
  remaining sensitive to sparse cross-sectional changes.
- **Expectation-based scan statistics** (Neill and co-authors) and the **Farrington
  quasi-Poisson** outbreak-detection family from public-health surveillance. This is the closest
  *applied* analogue: fit a baseline expectation per unit from history including shared
  seasonality and trend, then monitor the residual, with the dispersion parameter estimated
  rather than assumed (quasi-Poisson / negative binomial), explicitly because count surveillance
  data are overdispersed. Tango's point that *prospective* monitoring requires
  **expectation-based** rather than population-based scan statistics — baseline parameters
  estimated on past data believed free of anomalies — is exactly the frozen-warmup design the
  project already has.

**Verdict:** the cohort residual is a crude, online, one-factor, robust version of
factor-adjusted panel change detection. It has a name in the literature, a demonstrated benefit
(improved detectability), a named failure mode (spillover), and a mitigation the project already
applies (median, not mean). **What the literature does not have is a sequential, one-pass,
O(1)-state version** — factorcpt-style methods are retrospective. §4's recommendation goes one
step further and makes the factor removal *free* by replacing the residual arithmetic with a
within-day cross-sectional rank, which removes *any* monotone platform-wide shock, not just an
additive one.

---

## 3. Candidates table

Ranked. "Labels" counts labels the method itself consumes, not the incumbent model's.

| # | Method | What its objective actually minimises | Labels needed | Assumption violated here (honestly) | numpy/scipy, 4 cores? | Licence | Expected effect on d7/d14/d30 at n≈14 |
|---|---|---|---|---|---|---|---|
| 1 | **Rank-CUSUM on the within-day cross-sectional rank of the cohort-residual score**, ranked top-K | Lorden worst-case delay (Page recursion), on a rank-transformed increment | **0** for the detector; 3 fitted params (intercept + 2 coefs) to re-emit a calibrated probability | Increment independence across days (serial correlation in a merchant's own score is real); rank uniformity assumes a few hundred+ active merchants/day | Yes. One `argsort` per day + O(1) per merchant. 8 B state | numpy BSD-3, scipy BSD-3 (already pinned) | d30: plausible move; d14: marginal; d7: expect 0-2 merchants, descriptive only |
| 2 | Window-limited GLR-CUSUM (Lai) on residual z | Lorden delay with unknown post-change parameter | 0 | Known parametric family; Gaussian and Poisson both misspecified at Fano 12.3 | Yes, O(W) per merchant/day, W≈30 → ~40k×30 flops/day, trivial | none needed | Similar to #1, higher variance, more knobs |
| 3 | e-detector / betting CUSUM (Shin-Ramdas-Rinaldo) on a bounded transform of the residual | Delay subject to a **non-asymptotic** ARL bound with no distributional assumption | 0 | Needs the observation bounded or sub-exponential — requires clipping the residual, which discards the extreme-value information fraud lives in | Yes with geometric partitioning; O(log t) processes per merchant | Paper n/a; `confseq` reference impl is **MIT** (C++ core + Python), but adding it is discouraged — reimplement in ~40 lines | Comparable to #1; the ARL guarantee it buys is exactly what the capacity layer makes worthless |
| 4 | Shiryaev-Roberts / SR-r on the same increment | Stationary-regime average delay (SR); Pollak's conditional delay (SR-r) | 0 | Same as #1, plus needs a head-start `r` tuned to an (unavailable) ARL target | Yes, `R_t=(1+R_{t-1})Λ_t`, one float | none needed | Indistinguishable from #1 at n≈14. Multiplicative form overflows without a log parameterisation |
| 5 | Shiryaev (Bayesian) with geometric onset prior | Expected delay under a prior on the onset time | 0 | The prior would be read off the generator config — that is peeking at the DGP | Yes | none needed | Would look good and mean nothing. **Reject on validity, not on cost** |
| 6 | Scan-B kernel/MMD per merchant (Li-Xie-Dai-Song) | Delay via an MMD statistic with an ARL-controlled threshold | 0 | Detects *any* distributional change including benign ones; fraud is directional and this throws the direction away | Borderline: B² kernel evals/merchant/day; B=5, d=20 is survivable at 40k merchants but eats the 10 ms p99 | none needed (pure numpy) | Unlikely to beat #1; strictly more compute and more knobs |
| 7 | Conformal test martingale per merchant (Vovk) | Power against an exchangeability null; delay only implicitly | 0 | Exchangeability null is false — weekday seasonality alone breaks it unless de-seasonalised first | Yes | none needed | Similar to #3, weaker delay theory |
| 8 | NEWMA (two EWMAs at different rates on random features) | Model-free online CPD; **no delay optimality claimed** | 0 | Fails the spec's explicit weighting instruction: latency is not its objective | Yes, O(1) | none checked; irrelevant, reimplement | Would probably work; theory-light, so ranked below #1-#4 |
| 9 | NN / graph sequential test (Chen 2019) | Nonparametric multivariate delay | 0 | Requires a per-merchant reference sample and O(reference) distance computations per day | Marginal at 40k merchants daily | none needed | Not worth the compute at n≈14 |
| 10 | Factor-adjusted panel segmentation (`factorcpt` style) | Retrospective change-point localisation, **not delay** | 0 | **Retrospective**, not sequential — wrong regime entirely; also R | No (R) | CRAN, GPL-family — **UNVERIFIED, and disqualifying if GPL** | n/a — cannot produce a per-day score |
| 11 | Online FDR (LORD / LORD++ / SAFFRON) as the decision layer | Expected false-discovery proportion, **not delay** | 0 | Controls an error *rate*; Rakshak has a hard headcount K. Mutually exclusive constraints | Yes | R `onlineFDR` is Bioconductor, licence **UNVERIFIED**; trivial to reimplement anyway | None — would be overridden by the capacity layer |
| 12 | Global multi-stream rules (Mei SUM-CUSUM; Xie-Siegmund mixture; Chan) | Global delay: "has *any* stream changed" | 0 | Emits **one global alarm**, not a per-merchant score. Wrong output type for the rung interface | Yes but pointless | none needed | n/a — but their *local* statistics are exactly #1. Adopt the part, not the whole |
| 13 | `ruptures` offline CPD library | Offline segmentation cost minimisation | 0 | Offline; sees the future; would leak across the eval boundary | Yes | **BSD-2-Clause** (verified at github.com/deepcharles/ruptures) | n/a |
| 14 | `river` drift detectors (ADWIN, Page-Hinkley) | ADWIN: window comparison; PH: CUSUM on the mean | 0 | Single-stream, no panel confounder handling; PH is #1 without the rank transform | Yes | **BSD-3-Clause**, v0.26.1 (verified at pypi.org/project/river) | Adds a dependency for ~15 lines of numpy. Reject on the ladder, not on licence |
| 15 | Deep / learned-kernel CPD (KL-CPD, Chang et al., arXiv:1901.06077, and successors) | Test power via a learned kernel; delay only implicitly | Many | Requires `torch`; needs far more positives than 134 | **GATED** — violates the no-autograd decision | Reference impl is PyTorch; licence not checked because the gate binds first | **GATED, visible and revisitable.** Not silently dropped |

---

## 4. First place, unambiguous

### **Per-merchant Page/CUSUM on the within-day cross-sectional rank of the cohort-residual score, with the capacity layer performing top-K selection.**

Call it `rank_cusum` in the rung register.

**Why it wins.** It is the only candidate that satisfies all six binding constraints at once
while having detection delay as its literal objective. Its objective is Lorden's worst-case
delay and the recursion it uses is *exactly* optimal for that criterion at every false-alarm
level (Moustakides 1986) — not "classification accuracy with latency inherited as a side
effect", which is what the spec told me to weight against. The rank transform makes the
increment distribution-free, so the measured Fano factor of 12.25-12.37 — which would wreck any
Poisson or Gaussian likelihood ratio — becomes irrelevant, because under the no-change null a
within-day rank is uniform whatever the count distribution is. The *within-day cross-sectional*
rank is simultaneously the confounder guard: a festival spike, a gateway outage or a fee change
is a monotone shock applied to the whole panel on one day, and a rank is invariant to any
monotone shock, so the detector cannot fire on it — a strictly stronger guarantee than the
additive cancellation the cohort-residual subtraction provides, and it is free. The multi-stream
literature says the right global rule is top-r thresholding of local CUSUM statistics (Mei 2010
and its descendants), and the existing decision-policy seam plus the capacity budget K *already
is* top-r, so the entire multi-stream apparatus is adopted at zero implementation cost and no
threshold ever has to be calibrated. It consumes **zero labels** for the detector itself and
three fitted parameters to satisfy the `Decision.score ∈ [0,1]` contract, which matters more
than anything else on a fold with 134 trainable positives and labels arriving 66-141 days late.
It costs 8 bytes of state per merchant against a 4 KB budget, one `argsort` per day plus O(1)
arithmetic per merchant, adds no dependency, needs no autograd and no GPU, and reaches the
capacity layer through the existing seam without touching the hash-locked evaluation package.

### How it computes a per-merchant, per-day score

Run once per day `t`, over the set `A_t` of merchants active that day.

**Step 0 — input channel.** `s[m,t]` = the incumbent rung's per-merchant-per-day score
(calibrated probability in [0,1], from the Rung-3 cohort-residual model). The CUSUM is a
*wrapper*: it consumes no new features and no new labels. A fully label-free variant uses the
raw cohort-residual drift z-score in place of `s`; run it as a third arm (see §8).

**Step 1 — cross-sectional rank (the confounder guard and the distribution-free step).** Rank
within the merchant's cohort if the cohort has >= 30 members (reuse the existing 30-member
backoff chain from T-142), else platform-wide:

```
u[m,t] = (rank of s[m,t] among {s[j,t] : j in cohort(m)} - 0.5) / n_cohort_t
```

`u` lies in (0,1) and is uniform under no change. `np.argsort` twice, or `scipy.stats.rankdata`.

**Step 2 — normal scores.** `x[m,t] = scipy.stats.norm.ppf(u[m,t])`, clipped to ±4 so a single
day cannot dominate the accumulator. Mean 0, variance ≈ 1 under the null, by construction,
regardless of overdispersion.

**Step 3 — Page recursion (the whole method).**

```
C[m,t] = min(C_max, max(0.0, C[m,t-1] + x[m,t] - k))
```

- `k` is the reference value: half the standardised shift you most want to detect fastest.
  Start `k = 0.25` (targets a 0.5σ shift). This is the only tuning knob that matters; grid
  `k ∈ {0.10, 0.25, 0.50}` on the *training* fold only.
- `C_max = 20.0`. Without the cap, a merchant that drifted 200 days ago pins the top-K forever
  and the alert set becomes static again — reintroducing exactly the `volume_rank` pathology the
  method exists to defeat.
- `C[m,·]` initialises at 0 after the merchant's 30-day frozen warmup.
- One `float64` per merchant: **8 bytes** of `MerchantState`, declared as such in the
  `FeatureSpec.state_bytes` budget.

**Step 4 — emit through the existing interface.** `Decision.score` must be a calibrated
probability in [0,1] and a raw CUSUM statistic is not one. Fit, on the training fold only, a
two-feature logistic regression:

```
score[m,t] = sigmoid(a + b * logit(s[m,t]) + c * C[m,t])
```

Three parameters against 134 positives — inside the label budget with room to spare, and it
keeps both the level information (which `volume_rank` monetises) and the change information
(which is the point). `sklearn.linear_model.LogisticRegression`, BSD-3, already pinned.

**Step 5 — top-K.** Nothing to do. The capacity layer ranks by `score` and takes K. That is
Mei's top-r rule, already implemented, already tested.

**Reason codes.** `Decision.reason_codes` needs 3 entries for non-PASS. Emit `cusum_run_length`
(days since `C` last hit 0), `cusum_level` (`C[m,t]`), and the top `pred_contrib` feature from
the incumbent model. The run length is a genuinely new and human-legible artefact — "this
merchant has been drifting for 11 consecutive days" is something no static ranking can say.

**Dual-runner parity.** The recursion is inherently sequential, so `batch()` must replay it day
by day over the panel — exact, but the 1e-9 parity assertion will catch any off-by-one at the
warmup boundary. Expect that to be the one bug.

---

## 5. The numeric adoption gate, pre-registered

Fix this before any rung code exists. Roughly 14 evaluable onsets in the validation fold means a
single detection rate carries ≈ ±13 pp standard error, so **two independent proportions
differing by less than ~25 pp are not separable.** The gate is written to route around that
where it can and to admit defeat where it cannot.

**G1 — PRIMARY (paired, because independent proportions have no power at n=14).** On the
validation fold, for each evaluable merchant, record the binary indicator "detected within 30
days of `drift_onset_at`" under (i) `rank_cusum` and (ii) the best incumbent rung by savings.
Let `b` = merchants caught by `rank_cusum` and missed by the incumbent, `c` = the reverse.
**Adopt only if the exact one-sided binomial (McNemar) test on the `b+c` discordant pairs gives
`p < 0.05`.** Concretely that needs `(b >= 6, c = 0)` or `(b >= 7, c <= 1)`. Pairing is what
makes n=14 usable at all; the unpaired version of this comparison is unpowered by construction
and must not be reported as if it were a result.

**G2 — MECHANISM (full power; this is the gate that will actually decide the cycle).** Median
week-over-week alert Jaccard on validation must fall in **[0.30, 0.85]**. `volume_rank` scores
exactly 1.000. Above 0.85 means the method did not produce a time-varying alert set and the
whole premise failed; below 0.30 means it is churning and the analyst queue is noise. This gate
is computed over the entire 20k-40k merchant alert stream, not over 14 merchants, so it has real
power and a clean verdict.

**G3 — GUARDRAIL (must not regress).** Normalised savings on validation >= `incumbent_best -
0.02`. `rank_cusum` is **not** required to beat `volume_rank`'s 0.6016 to be adopted — it is
required not to lose ground against the best learned rung's 0.4348 while buying latency. If it
beats 0.6016, say so loudly; do not expect it (§8).

**G4 — RANKING SANITY.** PR-AUC on validation >= 0.75 (incumbent 0.836). The rank transform
throws away level information and could plausibly cost ranking quality; a drop below 0.75 means
the wrapper broke the ranker and the `c` coefficient in Step 4 needs re-examining.

**G5 — NULL RESULT, pre-declared.** If G2 and G3 pass but G1 does not reach `p < 0.05`, record
the outcome as **INSEPARABLE ON LATENCY AT n=14** in `LIMITATIONS.md`, adopt or reject on
G2/G3 alone, and **do not report the d30 point difference as a finding.** This is the most
likely outcome and pre-declaring it is the whole point of pre-registration.

**Descriptive only, never a gate:** `detection_rate_d7` and `detection_rate_d14`. With ~14
merchants and a 7-day horizon these will be 0, 1 or 2 merchants for every arm. Report them; do
not decide on them.

---

## 6. ADR stub

**ADR-0xx — Adopt rank-CUSUM as the cycle-4 latency rung**

**Status:** Proposed (cycle 4, pre-registration)

**Context.** Cycle 3 reported `detection_rate_d7/d14/d30 = 0.000` for every rung and every
floor; investigation showed the metric was arithmetically unreachable given the onset window
(days 30-240, realised max 217) and the validation window opening at day 240. Separately, every
learned rung lost on normalised savings (best 0.4348) to a static size ranking (`volume_rank`,
0.6016) whose week-over-week alert Jaccard is exactly 1.000 — a detector with no time dependence
at all. Cycle 4 widens the onset window to days 30-364, making latency measurable for the first
time (~14 validation onsets, ~13 test). Constraints: no autograd, no GPU, 4 CPU cores, ~10
ms/merchant p99, permissive licences only, ~134 trainable positives, labels delayed 66-141 days
with 15% never arriving, and a hash-locked evaluation package that must not be edited.

**Decision.** Add one rung, `rank_cusum`: a per-merchant Page/CUSUM recursion
(`C_t = min(C_max, max(0, C_{t-1} + x_t - k))`) run on the normal-score transform of the
merchant's **within-day, within-cohort rank** of the incumbent Rung-3 cohort-residual score,
with `k = 0.25` and `C_max = 20`. The statistic is folded back into the required calibrated
probability by a three-parameter logistic blend of `logit(incumbent score)` and `C_t`, fitted on
the training fold. Top-K selection is left entirely to the existing capacity layer, which
already implements the multi-stream literature's top-r rule. No new dependency; numpy, scipy and
scikit-learn only, all already pinned, all BSD-3.

**Consequences.**

- *Positive:* detection delay becomes the explicit objective of a rung for the first time. The
  rank transform makes the detector distribution-free, sidestepping the Fano-12.3 overdispersion
  that would misspecify any Poisson or Gaussian likelihood ratio. Within-day cross-sectional
  ranking is invariant to *any* monotone platform-wide shock, a strictly stronger confounder
  guard than the additive cohort residual. 8 bytes of state per merchant. A new human-legible
  reason code (`cusum_run_length`). Zero labels consumed by the detector.
- *Negative:* the recursion is sequential, so the offline `batch()` runner must replay it and the
  1e-9 dual-runner parity test becomes the highest-risk test in the ticket. The rank transform
  discards absolute level, which is the information `volume_rank` monetises — the logistic blend
  is the mitigation and may not be enough (§8). One new tuning knob (`k`), gridded on the
  training fold only. The classical ARL / false-alarm optimality theory does not transfer to a
  hard-capacity regime, so the adoption argument rests on the gate in §5, not on a theorem.
- *Risk accepted:* at ~14 evaluable onsets the primary latency gate may return INSEPARABLE. G5
  pre-declares that outcome so the cycle cannot be rescued by post-hoc metric selection.

**Alternatives rejected.**

- *Poisson / Gaussian CUSUM or GLR-CUSUM on raw counts* — likelihood misspecified at Fano
  12.25-12.37; would fire on ordinary weekday variation.
- *e-detectors / betting CUSUM* — theoretically the strongest answer to misspecification and the
  only family with non-asymptotic ARL guarantees, but the guarantee it buys is an ARL bound, and
  the capacity layer makes ARL irrelevant. Kept as the documented upgrade path if the rank
  transform proves too lossy.
- *Shiryaev Bayesian with a geometric onset prior* — rejected on validity: the prior would be
  read from the generator's own config.
- *Scan-B / MMD / NN-graph detectors* — detect any distributional change, not a directional
  increase; more compute, more knobs, no gain expected at n≈14.
- *Global multi-stream rules (SUM-CUSUM, Xie-Siegmund mixture)* — emit one global alarm, not a
  per-merchant score; wrong output type. Their local statistics are adopted; their aggregation is
  not, because the capacity layer already does it.
- *Online FDR (LORD / SAFFRON)* — controls an error rate; the operating constraint is a hard
  headcount. Mutually exclusive.
- *`ruptures`* (BSD-2-Clause) — offline, would leak future information across the eval boundary.
- *`river` drift detectors* (BSD-3-Clause, v0.26.1) — a dependency for ~15 lines of numpy, with
  no panel confounder handling.
- *Deep / learned-kernel CPD (KL-CPD and successors)* — **GATED** on the no-autograd
  architectural decision, and separately implausible with 134 positives. Recorded here so the
  gate stays visible and revisitable rather than silently dropped.

---

## 7. Where the literature is thin

1. **Hard per-period capacity has no theory.** Every optimality result in QCD is stated under
   `ARL_0 >= gamma` (frequentist), `PFA <= alpha` (Bayesian) or `FDR <= alpha` (multi-stream
   identification). Rakshak's constraint is "exactly K alerts per day, forever". I found no paper
   minimising expected detection delay subject to a fixed per-period *selection budget*. Top-r
   thresholding (Mei's family) is the closest object, but it is introduced as a *computational*
   device for scalability, not as the operating constraint, and its optimality is still proved
   against an ARL constraint. **This is a real gap and it is the gap Rakshak sits in.**
2. **Exposure-weighted delay does not exist.** Nothing found on minimising *cost-weighted* delay
   where the loss rate after the change is proportional to a stream-specific exposure (merchant
   GMV). Every delay criterion in §2A counts days, not rupees. This is exactly why `volume_rank`
   beats a better ranker on money, and the absence of a citation here is not the absence of a
   problem — it is the absence of a solution. Anyone extending this project has a publishable
   question sitting here.
3. **Overdispersed count QCD is under-served.** Statistical process control for counts is
   Poisson-INAR, negative-binomial-GARMA, or Winsorised-Poisson CUSUM — all parametric, all
   assuming the dispersion is estimable and stable. Distribution-free *delay optimality* for
   panels at Fano ≈ 12 is not something I found stated anywhere. Rank-based methods sidestep the
   problem rather than solving it, and that distinction should be recorded honestly.
4. **Delayed, censored and never-arriving labels have no counterpart.** QCD treats the changepoint
   as a latent time to be detected. Rakshak's ground truth arrives 66-141 days after the fact via
   `Exp(21) + Uniform(45,120)`, and 15% of it never arrives at all. There is no sequential-detection
   literature on *evaluating* delay under that censoring; the eval harness's censoring filter has
   no theoretical justification to point at.
5. **Sequential factor adjustment is missing.** Factor-adjusted panel change detection
   (Barigozzi-Cho-Fryzlewicz) is retrospective. No one appears to have published an online,
   one-pass, O(1)-per-stream-state version. The cohort residual and the cross-sectional rank are
   both ad hoc substitutes; they work, but they are engineering, not theory.
6. **Slow-ramp adversaries (R2).** Almost all delay theory is for abrupt i.i.d. changes. Work on
   transient and intermittent changes exists (e.g. arXiv:2210.17342, intermittent changes of
   unknown duration) but is thin, and none of it addresses an adversary deliberately ramping
   slowly enough to walk a rolling baseline — the attack the frozen-warmup design anticipates.

---

## 8. Contrarian view — the case against rank-CUSUM

The best argument against my own first place is that it deliberately destroys the information the
scoring metric actually pays for. The savings metric is exposure-weighted: money lost accrues
roughly in proportion to merchant size, so `volume_rank`'s 0.6016 may not be a fluke of a
stationary window at all — it may be the metric correctly rewarding a detector that allocates all
K reviews to the merchants where a caught fraud is worth the most, and accepting a PR-AUC of
0.217 as the price. A within-day cross-sectional rank is *scale-free by construction*: a
₹2 crore/month merchant drifting 0.6σ and a ₹40,000/month merchant drifting 0.6σ produce identical
statistics, and the capacity layer, ranking by that statistic, will happily spend reviews on the
small one. The most likely single outcome of cycle 4 is therefore a **split verdict**: d30
improves (unmeasurably, at n=14, per G5) while savings falls — which, weighed against a
full-population money metric with real power on one side and a 14-merchant latency metric with
±13 pp noise on the other, reads unambiguously as "the new rung lost". The three-parameter
logistic blend in Step 4 is the mitigation, but it is a *thin* one: it re-imports level
information only through the incumbent score, and the incumbent score is itself size-normalised.

There is a second, sharper possibility: **QCD may be buying nothing here at all, because the
generator's fraud is legible in volume alone.** If a merchant's fraud onset raises its GMV or
transaction count by a large multiple within a day or two, a same-day residual z-score already
flags it and Page's accumulation adds nothing — accumulation is only valuable when the
per-observation signal-to-noise is *small*, which is the regime CUSUM was designed for and may
simply not be this one. If the drift is large and abrupt, a plain threshold on today's residual
is the better detector and CUSUM only adds latency (it must accumulate past `k` before it rises)
and staleness (`C` stays high after the merchant reverts). Under that hypothesis `volume_rank`
beats the learned rungs *and* would beat rank-CUSUM, and the cycle-3 result was never about
stationarity — it was about exposure weighting, and the "the window contained no onsets" story is
a comfortable explanation for an uncomfortable metric.

**Two mitigations, both cheap, and I would insist on both.** First, run a **third arm**:
`residual_z_today` — the same cross-sectionally-ranked residual with *no* accumulation
(`C_t = x_t`). If it matches rank-CUSUM on d30, the accumulation bought nothing and the result is
attributable to the rank/residual transform alone; that is a cleaner and more honest finding than
a CUSUM win, and it costs one extra rung row. Second, log the **exposure-weighted** decomposition
of the savings metric per arm, so a split verdict is diagnosable rather than merely
disappointing — and note explicitly that fixing it is a *decision-policy* change (cost-weighted
top-K), not a QCD change, and belongs in a different survey.

---

## 9. References

**Classical QCD**

- Page, E. S. (1954). "Continuous inspection schemes." *Biometrika* 41(1-2):100-115.
  doi:[10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100)
- Lorden, G. (1971). "Procedures for reacting to a change in distribution." *Ann. Math. Statist.*
  42(6):1897-1908. doi:[10.1214/aoms/1177693055](https://doi.org/10.1214/aoms/1177693055)
- Pollak, M. (1985). "Optimal detection of a change in distribution." *Ann. Statist.*
  13(1):206-227. doi:[10.1214/aos/1176346587](https://doi.org/10.1214/aos/1176346587)
- Moustakides, G. V. (1986). "Optimal stopping times for detecting changes in distributions."
  *Ann. Statist.* 14(4):1379-1387.
  doi:[10.1214/aos/1176350164](https://doi.org/10.1214/aos/1176350164)
- Polunchenko, A. S. & Tartakovsky, A. G. (2010). "On optimality of the Shiryaev-Roberts procedure
  for detecting a change in distribution." *Ann. Statist.* 38(6):3445-3457.
  doi:[10.1214/09-AOS775](https://doi.org/10.1214/09-AOS775); arXiv:0904.3370
- Xie, L., Zou, S., Xie, Y. & Veeravalli, V. V. (2021). "Sequential (quickest) change detection:
  classical results and new directions." *IEEE J. Sel. Areas Inform. Theory* 2(2):494-514.
  doi:[10.1109/JSAIT.2021.3072962](https://doi.org/10.1109/JSAIT.2021.3072962); arXiv:2104.04186
  — **the single best entry point to §2A.**
- Shiryaev (1963) and Roberts (1966), the origin papers for the Shiryaev-Roberts statistic: cited
  throughout the sources above; **primary bibliographic details UNVERIFIED here** — I read them
  only through Polunchenko & Tartakovsky (2010) and Xie et al. (2021).
- Lai, T. L., window-limited GLR-CUSUM for unknown post-change parameters: cited in Xie et al.
  (2021) §III. **Primary source UNVERIFIED.**

**Non-parametric / distribution-free**

- Shin, J., Ramdas, A. & Rinaldo, A. (2023). "E-detectors: a nonparametric framework for
  sequential change detection." arXiv:[2203.03532](https://arxiv.org/abs/2203.03532); also
  *New England J. Statistics in Data Science*.
- "Non-partitioned e-detectors for nonparametric sequential change detection."
  arXiv:[2607.28322](https://arxiv.org/abs/2607.28322) — **UNVERIFIED beyond title/abstract.**
- Vovk, V. et al. (2021). "Retrain or not retrain: conformal test martingales for change-point
  detection." *PMLR* v152; arXiv:[2102.10439](https://arxiv.org/abs/2102.10439)
- Volkhonskiy, D. et al. (2017). "Inductive conformal martingales for change-point detection."
  arXiv:[1706.03415](https://arxiv.org/abs/1706.03415)
- Li, S., Xie, Y., Dai, H. & Song, L. "Scan B-statistic for kernel change-point detection."
  arXiv:[1507.01279](https://arxiv.org/abs/1507.01279); partial results in *NIPS 2015*
  ("M-statistic for kernel change-point detection").
- "NEWMA: a new method for scalable model-free online change-point detection."
  arXiv:[1805.08061](https://arxiv.org/abs/1805.08061) — **author list UNVERIFIED**; I read only
  the title and abstract.
- Chen, H. (2019). "Sequential change-point detection based on nearest neighbors." *Ann. Statist.*
  47(3):1381-1407. doi:[10.1214/18-AOS1718](https://doi.org/10.1214/18-AOS1718)
- "Open-end nonparametric sequential change-point detection based on the retrospective CUSUM
  statistic." arXiv:[2007.08369](https://arxiv.org/abs/2007.08369) — **authors UNVERIFIED.**

**Multi-stream**

- Mei, Y. (2010). "Efficient scalable schemes for monitoring a large number of data streams."
  *Biometrika* 97(2):419-433. doi:[10.1093/biomet/asq010](https://doi.org/10.1093/biomet/asq010)
- Xie, Y. & Siegmund, D. (2013). "Sequential multi-sensor change-point detection." *Ann. Statist.*
  41(2):670-692. doi:[10.1214/13-AOS1094](https://doi.org/10.1214/13-AOS1094)
- Chan, H. P. (2017). "Optimal sequential detection in multi-stream data." *Ann. Statist.*
  45(6):2736-2763. doi:[10.1214/17-AOS1546](https://doi.org/10.1214/17-AOS1546); arXiv:1506.08504
- Halme, T. & Koivunen, V. (2026). "Multi-stream quickest change detection: foundations and recent
  advances." *Entropy* 28(5):566. doi:[10.3390/e28050566](https://doi.org/10.3390/e28050566);
  arXiv:[2604.18008](https://arxiv.org/abs/2604.18008) — **the current review.**
- Tartakovsky, A. G. & Spivak, V. (2023). "Quickest changepoint detection in general multistream
  stochastic models: recent results, applications and future challenges."
  arXiv:[2305.07834](https://arxiv.org/abs/2305.07834)
- Tartakovsky, A. G. et al. (2018). "Asymptotically optimal quickest change detection in
  multistream data — Part 1: general stochastic models."
  arXiv:[1807.08971](https://arxiv.org/abs/1807.08971)
- Halme, T., Veeravalli, V. V. & Koivunen, V. (2025). "Quickest change detection for multiple data
  streams using the James-Stein estimator." *IEEE Trans. Inform. Theory* 71(10):7802-7814;
  arXiv:[2404.05486](https://arxiv.org/abs/2404.05486)
- Halme, T. & Koivunen, V. (2023). "Optimal multi-stream quickest detection with false discovery
  rate control." *Asilomar Conf. Signals, Systems & Computers*, pp. 877-881 — **page range
  UNVERIFIED** (secondary listing only).
- Liu, K., Zhang, R. & Mei, Y. "Scalable SUM-shrinkage schemes for distributed monitoring of
  large-scale data streams." *Statistica Sinica* — **volume/pages UNVERIFIED**; PDF seen at
  stat.uga.edu.
- Fellouris, G. & Sokolov, G., second-order asymptotic optimality in multi-sensor sequential change
  detection — **primary source UNVERIFIED**; read only through the Tartakovsky and Halme reviews.
- Javanmard, A. & Montanari, A. (2015). "On online control of false discovery rate."
  arXiv:[1502.06197](https://arxiv.org/abs/1502.06197) (LOND and LORD).
- Robertson, D. S. et al. "onlineFDR: an R package to control the false discovery rate for growing
  data repositories." *Bioinformatics* 35(20):4196-4199 — R package; **licence UNVERIFIED and out
  of stack.**

**Panel / shared nuisance signal**

- Barigozzi, M., Cho, H. & Fryzlewicz, P. (2018). "Simultaneous multiple change-point and factor
  analysis for high-dimensional time series." *J. Econometrics* 206(1):187-225;
  arXiv:[1612.06928](https://arxiv.org/abs/1612.06928) — source of both the "factor analysis
  improves detectability" result and the spillover caveat.
- Cho, H. (2016). "Change-point detection in panel data via double CUSUM statistic." *Electron. J.
  Statist.* 10(2); arXiv:[1611.08631](https://arxiv.org/abs/1611.08631)
- Neill, D. B. (2005/2006), expectation-based Poisson scan statistic; and Farrington, C. P. et al.
  (1996), quasi-Poisson outbreak detection with overdispersion. Both surveyed in "Outbreak
  detection algorithms based on generalized linear model: a review with new practical examples",
  PMC10576884 — **primary sources UNVERIFIED**, read through that review and the `scanstatistics`
  package documentation.

**Gated / rejected on architecture**

- Chang, W.-C., Li, C.-L., Yang, Y. & Póczos, B. (2019). "Kernel change-point detection with
  auxiliary deep generative models." *ICLR 2019*;
  arXiv:[1901.06077](https://arxiv.org/abs/1901.06077) — **GATED (requires PyTorch).**

**Libraries — licences verified at source, 2026-09-01**

| Library | Licence | Version | Verified at | Verdict |
|---|---|---|---|---|
| numpy | BSD-3-Clause | >=2, pinned | already in stack | use |
| scipy | BSD-3-Clause | pinned | already in stack | use (`stats.rankdata`, `stats.norm.ppf`) |
| scikit-learn | BSD-3-Clause | pinned | already in stack | use (`LogisticRegression`) |
| `ruptures` | **BSD-2-Clause** | — | github.com/deepcharles/ruptures | permissive but **offline-only**; reject on regime |
| `river` | **BSD-3-Clause** | 0.26.1 (2026-08-21) | pypi.org/project/river | permissive; reject on the ladder (~15 lines of numpy) |
| `confseq` | **MIT** | — | github.com/gostevehoward/confseq | permissive; C++ core; only if e-detectors are adopted later |
| `factorcpt`, `npcp`, `cpm`, `onlineFDR` | **UNVERIFIED** (CRAN/Bioconductor, GPL-family likely) | — | not checked | out of stack (R); do not adopt |
| PyTorch-based CPD (`klcpd` etc.) | not checked | — | — | **GATED** by the no-autograd decision; the gate binds before the licence question |
