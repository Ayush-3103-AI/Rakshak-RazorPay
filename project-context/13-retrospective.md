# 13 — Retrospective: what we built, how, what it returned, and where it diverged

**Written 2026-08-29, after T-0011 rendered K2.** Complete context in one file: the hypothesis, the
method, every measured result, and — the section that matters most — the places where what we got
did not match what we expected.

> **Sequence-layer metrics are measured on synthetic merchant streams with injected typologies;
> the generator is in this repo. The decision layer is additionally validated on BAF (Feedzai,
> NeurIPS 2022), a public benchmark derived from real bank data.**

**Read §5 before quoting any number from §4.** Several of them mean something different from what
they appear to mean, and the difference is the substance of this project.

---

## 1. What we set out to do

Razorpay's **Vulcan** scores every *transaction* in milliseconds. Razorpay's **Bumblebee** reviews
every *merchant* once, at onboarding. Nothing watches a merchant that was already cleared drift
from good to bad over the following weeks — so bust-outs, laundering endpoints, category drift and
refund collusion surface only when chargebacks land 45–120 days later.

**Rakshak** runs a per-merchant Hidden Markov Model over the transaction stream, maintaining a
belief over latent risk states that updates with each transaction, and converts that belief into
**pass / review / hold** using per-merchant cost economics under a fixed analyst-hour budget.

**The pre-registered claim** (`00-charter.md` §2, amended and dated 2026-08-28 *before* any swept
number existed):

> Rakshak beats a static velocity/refund-ratio rule engine by **≥20% relative** on the Bahnsen
> savings score at the cited central cost asymmetry, at equal analyst-hour budget, on a
> temporally-and-group-split held-out set of unseen merchants — with the relative improvement
> reported across the full plausible asymmetry range and the boundary at which the claim fails
> stated explicitly.

**That claim does not hold.** See §4.3.

---

## 2. What we built, and with which algorithms

### 2.1 Generator — the evaluation artifact

Synthetic merchant streams, 500 merchants, 771,900 transactions in ~5 s. Five typologies injected
with known transition timestamps: **bust-out**, **laundering endpoint**, **category drift**,
**refund collusion**, and an adversarial **slow ramp** built specifically to be hard.

It is an evaluation artifact, not a fraud toolkit (FR-006). It exists so detection can be tested
against ground truth that does not exist in any public dataset.

### 2.2 Features

- **Within-merchant standardisation (FR-007).** Each merchant is z-scored against *itself*, not
  against the population. A ₹300-AOV D2C brand and a ₹80,000-AOV electronics seller have nothing
  in common except that each has a stable baseline; deviation from *self* is the signal. This was
  identified in Phase 2 as the single most important modelling decision in the project.
- **Graph-derived scalars (FR-008, ADR-0002).** Payer-set entropy, repeat-payer ratio, payer-set
  Jaccard drift, Herfindahl concentration. These stand in for a graph neural network, which was
  rejected: GPU-only, and evaluating a GNN on a graph our own generator wrote would be circular.

### 2.3 Sequence model — hand-written HMM (ADR-0001)

Forward–backward, Viterbi and Baum-Welch, **written by hand in log-space numpy**. `hmmlearn` was
rejected — limited-maintenance mode, and it cannot express the hierarchical priors the design
called for. Four latent states: HEALTHY, RAMP, FRAUD, DORMANT.

Fitted by **label-informed weighted-likelihood partially-supervised EM** (ADR-0009): γ is clamped
to known states on labelled *training-split* windows only, and rare labelled states are up-weighted
in the M-step. This replaced unsupervised Baum-Welch after K1 fired — see §4.1.

**The Viterbi path is the product**, not a diagnostic. It is what lets a held merchant be told
which state they entered, on which day, and which three emissions drove it.

### 2.4 Decision layer — Bayes Minimum Risk under capacity (ADR-0005)

Three actions — **PASS / REVIEW / HOLD** — chosen as the argmin of expected cost given the
posterior and that merchant's own cost parameters, then allocated under a hard analyst-hour budget
with the binding constraint reported rather than inferred (FR-017).

Cost primitives (`07-math.md` §5, corrected at T-0017/T-0007a):

- `V_m = g · v_m · ℓ_m` — expected **lifetime gross margin**. An earlier definition used one
  window's MDR revenue, and a second error was found inside the first: `MDR_RATE = 0.02` is a
  *price*, not a margin. The platform's own margin is ~10 bps of TPV, not 200 bps.
- `L_m = r_cb · (1 + φ) · G^bad_m` — **realised** loss, not gross turnover while bad.

Both definitions were wrong before T-0017, which is why every `savings` figure predating it is
unquotable.

### 2.5 Baselines

Static rule engine (the floor that must be beaten), **LightGBM** (the incumbent), and **uniform
random** (the absolute floor). The random row turned out to be the most important one in the
project — §5.4.

### 2.6 Evaluation

Temporal **and** merchant-group split, enforced in code rather than by convention. The `test`
window (days 210–269) is locked by `eval.splits.load_split`, which refuses it without an unlock
ticket; it was opened exactly once, at T-0011. Perfect-foresight **knapsack oracle** as a ceiling,
plus an unconstrained hindsight oracle. Metrics: Bahnsen savings, PR-AUC, precision@K, Brier,
detection lag. **ROC-AUC and accuracy are prohibited as headline metrics** — at 20% prevalence
ROC-AUC flatters everything and "predict healthy" beats most models on accuracy.

### 2.7 Decided in, then cut

| | Why it was cut | Consequence |
|---|---|---|
| **T-0008** empirical-Bayes shrinkage | fourth in the cut list on 2026-08-28 | **Became load-bearing later.** BMR consumes raw scores as posteriors, so miscalibration moves the *decision*, not just the ranking |
| **T-0009** NSGA-II frontier | schedule | Its own obligation — beat a grid search or be deleted — is undischarged |
| **T-0010** BOCPD | schedule | **No sequence-aware baseline other than the HMM was ever measured** |

Cut, not deleted. Every one is absent from the ablation table **with a stated reason**, never zero
and never silently missing.

---

## 3. How we worked

The method is a deliverable in its own right, because Track 02 is scored on measurement discipline
more than on model sophistication.

**One ticket per session, ordered by risk retirement rather than comfort.** The board was
deliberately front-loaded with whatever was most likely to kill the project.

**Pre-registration.** The charter's headline claim was amended to be explicitly conditional on the
cost asymmetry on **2026-08-28, before T-0007b's sweep ran**. Amending it afterwards would have
read as an excuse. The same discipline produced a pre-registered RAMP-recall ≥ 0.35 bar that
**failed at 0.234 and shipped as a strict `xfail`**.

**Kill criteria checked, not admired.** K1 (state recovery) fired at T-0004 and was answered with
an oracle ceiling rather than patched around. K2 (beat the rule engine) fired at T-0011 and is
reported as a failure.

**Spec errors raised, not coded around.** When a ticket revealed a spec defect, the rule was to
stop and raise it. This was followed except once — T-0007b re-scoped an invariant in code before
raising it — and that exception is recorded in the ticket's own amendment block rather than
smoothed over.

**Parallel agents on file-disjoint scopes.** T-0007b and T-0015 ran simultaneously against
non-overlapping files, with the shared board/state/logbook reserved to the orchestrator.

**Two-axis review before every commit** — a Standards pass against the repo's documented rules and
a Spec pass against the originating ticket, run as independent reviewers so neither could mask the
other. This caught three defects the test suite did not (§5.9).

**Determinism as a hard requirement.** Every script takes `--seed`; every artifact is verified
byte-identical across runs.

---

## 4. Results

### 4.1 K1 — latent state recovery. Gate failed.

FR-013 required four-way state ARI > 0.5.

| | value |
|---|---|
| Unsupervised ARI | **0.091** |
| Label-informed ARI | **0.319** |
| **Oracle-parameterised ceiling** | **0.404** |
| Gate | 0.5 |

**The load-bearing number is the ceiling.** With HMM parameters read straight off ground truth —
the fully-supervised MLE estimator — recovery reaches only 0.404. **The gate was unreachable by any
correctly-implemented HMM.** Two real bugs were found and fixed en route and **both moved ARI
down**, which is what established the gap was not debuggable.

Root cause: **RAMP sits 1.19σ from HEALTHY**, which holds ~90% of window mass. RAMP is the
early-warning state, so the gate failed precisely on the product premise.

FR-013 was amended to AMI + per-state recall + binary PR-AUC + detection lag, **with the ARI and
the ceiling retained and reported permanently** — those are what make the amendment credible rather
than convenient.

### 4.2 Validate split (100 merchants, 20% prevalence, K = 5)

| model | savings | PR-AUC | precision@5 | Brier | bad flagged |
|---|---|---|---|---|---|
| random | +0.6929 | 0.1651 | 0.0000 | 0.3589 | — |
| rules | +0.6980 | 0.5377 | 0.8000 | 0.1319 | 0.45 |
| gbdt | +0.7392 | **0.6778** | **1.0000** | **0.1242** | 0.50 |
| hmm | **+0.7464** | 0.4994 | 0.6000 | 0.3149 | **0.65** |

### 4.3 Test window — K2's verdict. **FAIL.**

Days 210–269, unlocked exactly once at T-0011. Every threshold and configuration was fixed on
`train`/`validate` before this file existed, and **nothing was changed after reading it**.

| model | savings | savings − random | PR-AUC | precision@5 | Brier | median lag | bad flagged |
|---|---|---|---|---|---|---|---|
| random | **0.5365** | 0.0000 | 0.2449 | 0.2000 | 0.3069 | n/a | 0.00 |
| rules | 0.4889 | −0.0475 | 0.5547 | 1.0000 | 0.1358 | 5.0 d | 0.65 |
| gbdt | 0.5069 | −0.0296 | **0.6523** | 1.0000 | **0.1453** | 10.0 d | 0.65 |
| hmm | 0.5176 | −0.0188 | 0.3347 | 0.4000 | 0.4321 | 11.0 d | **0.75** |

| K2 | |
|---|---|
| `hmm` vs `rules`, relative margin | **5.9%** |
| Pre-registered bar (NFR-001) | **20%** |
| **Verdict** | **FAIL** |

### 4.4 Ablations (test window, frozen configuration, nothing selected here)

| component | setting | savings | Δ savings | PR-AUC | Δ PR-AUC |
|---|---|---|---|---|---|
| **HMM** | on (shipping) | 0.5176 | ref | 0.3347 | ref |
| **HMM** | **off** — same pipeline, LightGBM scorer | 0.5069 | **−0.0107** | 0.6523 | **+0.3176** |
| graph features | off — HMM refitted | 0.4170 | −0.1006 | 0.2957 | −0.0390 |
| graph features | off — LightGBM refitted | 0.5116 | +0.0047 | 0.5306 | −0.1217 |
| within-merchant standardisation | off — HMM refitted | 0.4858 | −0.0318 | 0.2872 | −0.0475 |
| within-merchant standardisation | off — LightGBM refitted | 0.5068 | −0.0001 | 0.6556 | +0.0032 |

### 4.5 Cost-asymmetry sweep

Range **2.5 – 530.3** on `validate`, **0.7 – 146.9** on `test`, derived from
`config.COST_PRIMITIVE_RANGES` with no literal endpoint. The HMM's margin over `rules` **crosses
zero between asymmetry 18.5 and 36.2** and loses by **−220.5%** at the bottom of the range. The
unflattering half ships.

### 4.6 BAF — real-data validation of the decision layer

1,000,000 rows, BAF's own temporal split, month 7 reported (96,843 applications, 1.47% prevalence).

| model | savings | PR-AUC | Brier | held |
|---|---|---|---|---|
| random | −28.2169 | 0.0143 | 0.3340 | 4,033 |
| credit_risk_score | −5.2810 | 0.0403 | 0.3200 | 469 |
| gbdt | **+0.0294** | **0.2179** | **0.0129** | 1 |

`gbdt` is the only positive model at **all ten** swept asymmetries.

### 4.7 Calibration gap — generator vs real data

| marginal | empirical | generator | divergence |
|---|---|---|---|
| `daily_count_fano_factor` | 12.25 | 1.00 | **×16.0 — structural** |
| `txns_per_active_day_mean` | 80.09 | 2.47 | ×32.5 |
| `refund_rate` | 0.1714 | 0.0220 | ×7.8 |
| `amount_log_sd` | 1.256 | 0.647 | ×1.9 |
| `new_payer_frac` | 0.1324 | 0.755 | ×0.18 |

**5 of 8 ratio-scale marginals diverge ≥1.9×.**

---

## 5. Where expectation and result diverged

This is the section the project exists to be able to write.

### 5.1 We expected a debuggable estimation gap. We got an unclosable representation gap.

**Expected:** the HMM recovers four latent states well enough to clear ARI 0.5; a miss means a bug.

**Got:** ARI 0.091, and an **oracle ceiling of 0.404** proving no correct HMM could clear the gate
on these emissions. Two genuine bugs were found and fixed and **both moved the number down.**

**Why it diverged:** the states overlap. RAMP sits 1.19σ from HEALTHY. The problem was in the
*data's separability*, not in the estimator — and the only reason we know that is the oracle. **A
project without an oracle ceiling would have spent four days debugging a correct implementation.**

### 5.2 We expected supervision to help the early-warning state. It hurt it.

**Expected:** partial labels lift every state, especially the rare ones.

**Got:** every headline metric roughly doubled — ARI 0.134 → 0.319, AMI 0.102 → 0.218, binary
PR-AUC 0.109 → 0.327 — while **RAMP recall fell 0.328 → 0.234**.

**Why it diverged:** labels help rare **separable** states, not rare **overlapping** ones. **The
configuration that wins overall is the one that goes blind on the state the product exists to
catch.** A pre-registered ≥0.35 bar was recorded before measuring, failed at 0.234, and ships as a
strict `xfail`.

### 5.3 We expected to beat the rule engine by ≥20%. We beat it by 5.9%.

**Expected:** the charter's pre-registered claim.

**Got:** 5.9% relative on the held-out test window. **K2 FAIL.**

**Why it diverged:** the HMM's advantage is *coverage* — it flags 0.75 of truly-bad merchants
against `rules`' 0.65 — but it is worse at *ranking* and much worse at *calibration*. Under a
cost-optimal policy that converts to a small savings edge, nowhere near 20%.

### 5.4 We expected savings to measure detection. It largely measures the cost matrix.

**Expected:** a savings lead over the baselines means the model is finding fraud.

**Got, on validate:** `random` +0.6929 against `rules` +0.6980 — a gap of **0.0051** — while
ranking at PR-AUC 0.1651, i.e. at prevalence.

**Got, on test:** **`random` scores 0.5365 and beats `rules`, `gbdt` and `hmm` outright.** The K2
margin is therefore a comparison between two models that **both sit below a uniform random score
on the primary metric.**

**Why it diverged:** when `c_fp` is small relative to `L_m`, a random score still lands most
merchants on the correct side of a merchant-specific threshold. This is `07-math.md` §6's AP-06
guard arriving as a measurement instead of a warning. **No savings figure in this repo may be
quoted without PR-AUC beside it and without subtracting the random floor.**

### 5.5 We expected the random floor to indict the metric. Real data indicted the generator instead.

**Expected:** §5.4 means the savings score is weak.

**Got:** on BAF, at a realistic **1.47%** prevalence, `random` scores **−28.2169** — catastrophic,
not competitive.

**Why it diverged:** the generator's **20% merchant fraud rate**, chosen so each typology has
enough merchants for a per-class metric. At 20% a random policy hits enough true positives to look
competent; at 1.5% it cannot. The AP-06 warning stands, but **its severity on synthetic data is
substantially an artefact of a prevalence we inflated on purpose.** This is the first time real
data talked back to a synthetic result, and it is the whole argument for the hybrid data strategy.

### 5.6 We expected early detection. The model is *late*, and the earlier number was an artefact.

**Expected:** a −1.0 day median lag meant detection *before* the labelled onset. The open question
was whether that was legitimate early warning or generator leakage.

**Got:** **neither.** It was **window-start attribution** — `gbdt` and `hmm` credited a flag to the
first day of the 7-day window that raised it, awarding up to 6 days of earliness they never had,
while `rules` had always reported a window-*end* day. One column was printing two conventions.

Corrected to window-end for both models together: `rules` **+5.0 d**, `gbdt` **+10.0 d**, `hmm`
**+11.0 d**.

**The correction reverses the reading.** The window-based models are **later** than the rule
engine, not earlier. A merchant-clustered permutation test over pre-onset windows cleared leakage
separately (largest effect 0.159, p = 0.310), and the HMM's `flag_day` is provably forward-only.
**No claim of the form "Rakshak detects N days before the fraud starts" is available to this
repo.**

### 5.7 We expected the HMM to be the value-add. The ablation says it costs more than it adds.

**Expected:** removing the sequence layer degrades results.

**Got:** replacing the HMM with LightGBM in the same pipeline moves savings by **−0.0107** and
improves PR-AUC by **+0.3176**.

**Why it diverged:** the HMM's contribution to the headline metric is within noise of nothing,
while its ranking deficit is large and consistent. **What the HMM uniquely provides is the Viterbi
path — an explanation, not accuracy.** That is a real product claim and it is now the *only* one
the measurements support.

A secondary surprise: the graph features are **load-bearing for the HMM** (−0.1006 savings without
them) and **worth nothing to LightGBM** (+0.0047).

### 5.8 We expected the cost asymmetry to be 400–600. It is 47.5, then 13.1, then 61,368.

**Expected:** `07-math.md` §5's commentary band of 400–600 INR of FP cost per INR 100 of fraud
loss, with an orientation estimate of ~280.

**Got:** **47.5** on validate, **13.1** on test, **61,368** on BAF.

**Why it diverged:** three separate reasons, each worth recording.

- The **band measures something else** — declined baskets at checkout, not held settlements costing
  the platform its own ~10 bps margin. They were never the same asymmetry.
- The **§5 orientation estimate of ~280 was itself wrong**: it assumed `V_m` rises 1.5× when it
  actually falls 4.67×. Chasing that 6× surprise found a real latent bug — `V_m` grew with split
  length, so `test` would have read 29% higher than `validate` for no reason. **The pre-registered
  prediction is what surfaced it.**
- **Validate vs test differ by population, not by tuning.** `L_m` is a stock accumulated over
  history; `V_m` is rate-derived. The test window loads 60 more days, so the denominator grows.
- **BAF's 61,368 is a unit assumption**, not a property of BAF: credit limits of 190–2000 against
  absolute INR support and review costs.

**The 400–600 check was demoted from a gate to a reported cross-check before any of this**,
specifically because `07-math.md` §5 as written instructed the project to tune parameters until the
ratio came into range — the identical practice the repo forbids for the generator, and worse,
because savings is the headline metric.

### 5.9 We expected the test suite to protect determinism and lint. It did neither, twice.

**Expected:** green tests and clean lint mean the invariants hold.

**Got:**

- **`src/rakshak/data/` had never been linted.** `extend-exclude = ["results", "data"]` matched the
  *source package* as well as the git-ignored directory. **871 lines were invisible and nine real
  errors were hiding.** Every "ruff clean" claim before 2026-08-29 covered only what ruff happened
  to look at.
- **`baf.py` fitted LightGBM without the three flags that make NFR-003 hold** — `deterministic`,
  `force_row_wise`, `num_threads=1` — and its numbers were already in a results file. **The full
  suite was green throughout.** Determinism was asserted for the harness, not for every
  artifact-producing path.
- **`data/profile.py` accepted `--seed`, never read it, and stamped a literal `--seed 42` into two
  committed artifacts.** Running `--seed 7` produced a file claiming 42 — a provenance lie in the
  artifact whose only job is provenance. The determinism test passed the whole time: it verified
  the artifact was *reproducible*, never that the provenance line was *true*.

All three were caught by **review**, not by tests.

### 5.10 We expected the board to track the project. It tracked the repo.

**Expected:** seventeen tickets covering the work meant the work was covered.

**Got:** `00-charter.md:83` states the output is *"public repo + 5-minute video + architecture doc.
**All three are graded.**"* The repo had seventeen tickets in forensic detail. **Neither of the
other two graded artifacts had a ticket at all** — the video had two days allocated, no script, no
shot list and no owner.

**Why it diverged:** completeness was being measured against the ticket list rather than against
the charter. Found on 2026-08-29 with three build days left; T-0018–T-0021 opened in response.

### 5.11 We expected the calibration gap to be closable. One divergence is structural.

**Expected:** a measured gap means a parameter swap fixes it.

**Got:** the generator's `daily_count_fano_factor` is **1.0 by construction** — it draws daily
counts from `rng.poisson`, and a Poisson process has variance equal to its mean — against a real
**12.25**.

**Why it diverged:** **no value of any generator constant produces over-dispersion.** Closing that
marginal requires replacing the emission process, which invalidates the K1 analysis, the 0.404
oracle ceiling and every baseline row. What looked like a scheduling question was a scope question.

A second limit sits behind it: the empirical side is **n = 1 merchant** — a UK B2B gift-ware
wholesaler trading in GBP, closed Saturdays.

---

## 6. What the divergences add up to

**The hypothesis did not survive contact with its own measurements.** The HMM does not beat the
rule engine by 20%; it beats it by 5.9%, and both sit below a random floor on the primary metric.
Removing the HMM entirely costs 0.0107 savings and *improves* ranking by 0.3176.

**What survives is narrower and true:**

- The HMM produces a **Viterbi path** — a merchant-readable account of which state was entered,
  when, and which emissions drove it. No baseline produces this. It is the answer to the panel's
  third question, *"can I explain the decision to the merchant when they call and shout?"*, and it
  is the one thing the measurements still support.
- The **decision layer works**, and BAF validates it on real-bank-derived data: BMR takes the
  economically correct action under an extreme cost asymmetry, the capacity constraint binds and is
  reported, and the savings score orders models consistently at every swept asymmetry.
- **The measurement apparatus is the strongest artifact in the repo** — the oracle ceiling that
  proved K1 unreachable, the pre-registration that made a conditional result honest, the random
  floor that exposed AP-06, the ablation that showed the HMM's own contribution, and the lag
  correction that removed an early-detection claim we would otherwise have made on camera.

`00-charter.md` §3's kill criterion anticipated exactly this: *"do not tune to win. Report the
negative result, pivot the narrative to explainability and the cost frontier, and say so on
camera."* That is what happens next.

---

## 7. What is left

| Ticket | State |
|---|---|
| **T-0013** Explainability + README | ready — all blockers closed |
| **T-0018** Architecture doc + diagram | ready — graded, previously unowned |
| **T-0019** Video script + shot list + edit | ready to draft; numbers after T-0013 |
| **T-0020** Release hygiene | ready — LICENSE, `09-interfaces.md`, drop `pymoo` |
| **T-0021** Verify `make eval` on a clean checkout | blocked by T-0020 |
| **T-0014** Read-only results viewer | blocked by T-0013 |
| **T-0016** Generator recalibration | conditional; structural divergence means it cannot fully succeed as scoped |

**Known open before freeze:** `make eval` has never run on a clean checkout — `make` is not
installed on the build machine and the Makefile has shipped unexercised since T-0001, while both
the README's provenance claim and the video script depend on it working.
