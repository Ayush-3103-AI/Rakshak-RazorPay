# Rakshak G3 — Architecture

**What this document is.** A description of how a merchant's transactions become a decision
somebody can defend on a phone call, for the tree that is actually submitted (`v3/`, the third
generation). It is written for a risk-operations or engineering reader who will not open the
code, and it is the third graded artifact alongside the README and the panel.

**What this document is not.** It contains **no measured numbers**. Not one, deliberately: the
design has to be legible whether the headline claim held or failed, and in this project the
pre-registered gate **failed on both of its conjuncts** and the test split was never opened. The
verdict, every metric and the provenance of each one live in
[`../../README.md`](../../README.md), [`results_v2.md`](results_v2.md),
[`results/cost_sweep.md`](results/cost_sweep.md) and — above all —
[`../LIMITATIONS.md`](../LIMITATIONS.md). Read those before you believe anything about how well
this works. This document tells you only what was built, why it has the shape it has, and what
is deliberately missing from it.

**Predecessor.** [`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) is the same document
for **G2**, the tree at the repository root. G2's four-layer shape and its ADRs are still worth
reading — G3 is the answer to G2's own falsification, not a rewrite for its own sake — but where
the two disagree about what exists today, this file is the one describing the shipped system.

---

## 1. The gap, in three sentences

Razorpay's **Vulcan** scores every *transaction*, in milliseconds. Razorpay's **Bumblebee**
reviews every *merchant*, once, at onboarding. Nothing watches a merchant who was already
cleared drift from good to bad over the following weeks, so bust-outs, laundering endpoints,
category drift and refund collusion surface only when the chargebacks land 45–120 days later.

Rakshak occupies that hole and only that hole. It does not score transactions (Vulcan owns
that), it does not gate onboarding (Bumblebee owns that), and it does not touch RTO/COD returns
(Thirdwatch owns that). Those are non-goals in `project-context/00-charter-v2.md` and reopening
any of them is a re-scope, not an addition.

---

## 2. What G3 changed about the *question*

G2 built a model and then measured it. Its own measurements falsified it — and worse, they
could not say whether the model or the data was at fault, because the generator had never been
validated against anything. **G3 inverts the order.** The generator, the split engine, the
metric suite, the floors and the cost model are built, tested and **hashed into a lock file
before the first model is written.** Only then does a ladder of policies race against them.

Everything below follows from that inversion. It is why the eval package is sealed and cannot
be edited to fix a defect it contains; why a rung that loses is a row in the results table
rather than a deletion; and why the interesting parts of this architecture are the **walls**
(§5) rather than the model.

---

## 3. The shape, in one picture

```
  configs/scenario_v2.yaml ──┐        (hashed into the lock, with the generator source)
                             │
  ┌──────────────────────────▼───────────────────────────────────┐
  │  GENERATOR          src/rakshak/generator/                   │
  │  personas × typologies × confounders                         │
  │  NB / Hawkes arrivals · delayed chargeback labels            │
  │  EVALUATION ARTIFACT, NOT A FRAUD TOOLKIT                    │
  └──────────────┬──────────────────────────┬────────────────────┘
                 │ transaction stream       │ ground_truth + labels
                 │ (parquet)                │ ── QUARANTINED (§5.1, §5.2)
                 ▼                          ▼
  ┌──────────────────────────┐   ┌──────────────────────────────┐
  │ EVENT STORE  store.py    │   │ eval/splits.py               │
  │ duckdb over parquet      │   │ the ONLY door to the label   │
  │ every read takes as_of   │   │ table, and it takes an as_of │
  └──────────────┬───────────┘   └──────────────────────────────┘
                 │ events where event_time <= as_of
                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  FEATURE LAYER      src/rakshak/features/                    │
  │  ONE FeatureSpec  →  TWO runners, asserted equal to 1e-9     │
  │      .update(state, event)   O(1), bounded state — production│
  │      .batch(frame)           polars over history — training  │
  │  Tier 1 · Tier 2 · cohort residual · (payer, day) capsules   │
  └──────────────┬───────────────────────────────────────────────┘
                 │ FeatureVector(merchant, as_of) → models/dataset.py
                 ▼                                  materialises the panel
  ┌──────────────────────────────────────────────────────────────┐
  │  THE LADDER         src/rakshak/models/                      │
  │  Rung 0 floors → 1 rules → 2 LGBM → 3 +cohort → 4 cost-in-fit│
  │  → 5 MIL → 6 conformal → 7 HSMM → 8 TPP → 9 rank-CUSUM       │
  │  Every rung is a Scorer: (split, rng) -> scored frame         │
  └──────────────┬───────────────────────────────────────────────┘
                 │ score per merchant-day
                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  DECISION LAYER     eval/capacity.py                         │
  │  expected cost of PASS / REVIEW / HOLD per merchant          │
  │  rank by benefit-of-intervening, take top K, rest PASS       │
  │  exposure arm: declared GMV  vs  realised GMV  (A/B)         │
  └──────────────┬───────────────────────────────────────────────┘
                 │ action per merchant-day
                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  FROZEN EVAL HARNESS   eval/  ·  EVAL-LOCK-CYCLE4.json       │
  │  temporal + merchant-group + label-availability splits        │
  │  four floors · perfect-foresight oracle · PR-AUC · savings   │
  │  · P@K · ECE · stability · TTD · per-typology recall         │
  │  open_count for the test split — a one-way door              │
  └──────────────┬───────────────────────────────────────────────┘
                 │ EvalResult rows
                 ▼
  ┌──────────────────────────┐        ┌──────────────────────────┐
  │ eval/report.py           │        │ artifacts/build.py       │
  │ docs/results_v2.md       │        │ artifacts/*.json         │
  │ + .parquet               │        │ schema-versioned         │
  └──────────────────────────┘        └────────────┬─────────────┘
                                                   │ the panel's ONLY data source
                                                   ▼
                                      ┌──────────────────────────┐
                                      │ dashboard/  React + Vite │
                                      │ static build, NO BACKEND │
                                      └──────────────────────────┘
```

---

## 4. The layers

### 4.1 Generator — `src/rakshak/generator/`

Personas (the legitimate business shapes) × risk typologies (the ways a cleared merchant goes
bad) × confounders (the platform-wide events that *look* like a merchant going bad and are
not). Daily counts are overdispersed rather than Poisson, and arrival times within a day come
from a marked point process, because a Poisson stream is separable by a variance test alone and
would make the whole problem easier than it is.

Two properties are load-bearing:

**Labels arrive late, on purpose.** A chargeback lands 45–120 days after the transaction that
caused it, so a merchant that turns fraudulent is *not labelled* for months. `label_available_at`
is carried on every label and enforced downstream (§5.2). Almost every published result in this
area quietly assumes labels are available at decision time; this one cannot.

**It is an evaluation artifact, not a fraud toolkit.** The track is strictly defense-only. The
typologies exist so that detection can be *measured* against them, they live only in this
package, and nothing in `features/` or `models/` may import it (§5.1). No real Razorpay data,
API or internal system is touched anywhere in this repo.

The generator's source and `configs/scenario_v2.yaml` are both hashed into the lock, so "which
data produced this number" is answerable from the results file alone.

### 4.2 Event store — `src/rakshak/store.py`

Parquet on disk, duckdb in front of it. **Every read takes an `as_of` and there is no method
that returns "all events".** That is a signature-level decision, not a convention: the
alternative is a point-in-time filter written correctly at eleven call sites and forgotten at
the twelfth. Labels are deliberately *absent* from this module — a second door into the label
table is the door nobody guards.

### 4.3 Feature layer — `src/rakshak/features/`

**One definition, two runners, parity asserted in CI.** Each feature is written once as an O(1)
fold over a bounded per-merchant state (what production would run) and once as a polars
expression over the whole history (what training runs), and `tests/parity/` asserts the two
agree to 1e-9 at every epoch for every merchant.

This is the single most important interface in the repo, and it exists to prevent one specific
failure: a feature that is easy to compute over a full history and impossible to compute from a
stream. A system built out of those demos beautifully and cannot be deployed. It is also the
gap between G2's claim and G2's reality, which is why G3 pays for it structurally.

Four kinds of feature:

- **Tier 1** — cheap level statistics, computed for every merchant every day. Whether a *level*
  moved.
- **Tier 2** — histogram and divergence statistics against the merchant's own frozen baseline.
  Whether a *shape* moved. A merchant can hold count and GMV exactly constant while its
  ticket-size, instrument mix and hour-of-day distributions all change, and that merchant is
  invisible to every Tier-1 feature.
- **Cohort residual** — the layer the whole generation was built to test. Every drift feature
  gets a companion equal to the merchant's own z-score minus the median z-score of its cohort,
  excluding itself. When a festival lifts GMV platform-wide every z rises together and every
  residual stays near zero; when one merchant moves alone the residual survives. This is the
  named open problem in post-onboarding monitoring — separating adversarial drift from natural
  platform drift — and it is isolated here so it can be tested as a single variable.
- **Capsules** — the same rows reshaped into a bag of (payer, day) instances, for the rungs that
  need what a daily aggregate throws away. They are read through the same store, not a second
  read path.

The per-merchant state carries a declared byte budget (NFR-04) checked **at import**, so a state
budget is discovered at startup rather than during an outage. **That budget is currently
exceeded and is not met** — see `../LIMITATIONS.md` for the measurement and the trade it forces.
It is reported as a failed non-functional requirement rather than raised to fit.

### 4.4 The inference cascade — `src/rakshak/features/cascade.py`

Three stages, and the arithmetic is the whole argument for them:

| Stage | Runs on | Features | Budget |
|---|---|---|---|
| 0 — screen | every merchant, every day | Tier 1 only | ≤ 0.5 ms per merchant-epoch |
| 1 — score | the top 10% from stage 0 | Tier 1 + Tier 2 + cohort | ≤ 10 ms per merchant-epoch |
| 2 — explain | non-`PASS` decisions only | reason codes, HSMM narrative | ≤ 50 ms per decision |

A single-stage design fits the daily sweep budget on today's population. What it does not do is
*stay* fitting as the population grows or the feature set widens, and the point of the cascade
is that the expensive work is proportional to the alerts rather than to the merchants. These are
declared budgets; whether each is met, and on which platform, is in `../LIMITATIONS.md`.

### 4.5 The ladder — `src/rakshak/models/`

Every policy on the ladder — including the floors — implements the same `Scorer` contract and
goes through the identical decision layer and metric suite. A floor that is only ever a column
is a floor nobody can inspect.

| Rung | What | Why it is on the ladder |
|---|---|---|
| **0** | `all_pass`, `all_hold`, `random_at_k`, `volume_rank` | The four floors. A model that beats nothing is not a result. `volume_rank` — rank by size, no learning — is the hardest of them, and it turns out to be an exposure estimator in disguise. |
| **1** | Static rule engine, thresholds declared not fitted | The incumbent a real deployment replaces. Deliberately *not* the bar — moving the goalposts back to the rule engine would be the dishonest move. |
| **2** | LightGBM on windowed aggregates | **The bar.** No post-hoc calibrator and no class weighting, so the reported probability is the honest one. |
| **3** | Rung 2 **plus** the cohort-residual columns | The single-variable test of §4.3's hypothesis. The trainer refuses to run if the two column sets differ by anything other than the registered residual columns. |
| **4** | The instance-dependent cost, inside the training objective | Tests the literature's claim that pricing inside the fit beats pricing after it. Implemented as instance weighting, so the output stays a probability the decision layer can consume. |
| **5 / 5b** | MIL pooling over payer capsules; gated attention | Instance-level evidence, aggregated. |
| **6** | Mondrian conformal risk control over the three actions | A distribution-free bound on the false-hold rate — the strongest decision-layer claim available. |
| **7 / 7b** | HSMM with negative-binomial emissions; segmented onset | **Explainer only.** See §5.3. |
| **8 / 8b** | Hawkes/NB temporal point process; neural intensity | Continuous-time anomaly detection as a hypothesis test. |
| **9** | Page/CUSUM on the within-day cross-sectional rank | Changepoint detection on rank rather than on level. |

`configs/rung_roster.yaml` is the machine-readable roster: every rung with its status
(`planned` / `built` / `scored` / `cut` / `deferred` / `conditional` / `UNVERIFIED`), its module,
its ticket and the citation the status was derived from. **It carries no scores by construction** —
scores live in `artifacts/ladder.json`, keyed by the same id — because a rung that was cut is
invisible to a results table built from scored rows, and "cut" has to be sayable. Which rungs
were adopted, and which were not, is a results question and is answered in the README and
`LIMITATIONS.md`, not here.

### 4.6 The decision layer — `src/rakshak/eval/capacity.py`

Analyst capacity is the binding operational constraint, and a metric that ignores it is
decoration. Per day, given calibrated scores and a capacity K:

1. compute the expected cost of each action for each merchant;
2. rank by `cost(PASS) − min(cost(REVIEW), cost(HOLD))` — the benefit of intervening, not the
   probability of fraud;
3. take the top K, each with its own cost-minimising action;
4. everything else `PASS`es.

A wrong K changes the *ranking* of rungs and not merely their scores, so every
capacity-constrained number names the K it was computed at.

**The exposure arm.** `models/decision_realised_exposure.py` prices the decision on the exposure
a merchant *realised* rather than the one it *declared* at onboarding. It carries no rung number
deliberately — it is registered as a controlled A/B over the *whole* ladder, not as a competing
rung, because numbering it would have put two different things under one heading. Four lines of
arithmetic; the reason it exists is the whole of its value, and that reason is in
`LIMITATIONS.md` rather than here.

### 4.7 The eval harness and the lock — `src/rakshak/eval/`

The claim this package has to make checkable **by someone who does not trust us**:

> "The test split was opened once, after every model was final, against a harness whose code and
> configuration were hashed before any model existed."

Four mechanisms, none of which is a matter of discipline:

- **Splits** — temporal **and** merchant-group-disjoint **and** label-availability-aware,
  applied simultaneously. Any one alone leaks. Merchants are hashed to folds, so no merchant id
  appears in two splits; training at decision time `t` may use only labels already available at
  `t`.
- **The lock** — `EVAL-LOCK*.json` records the sha256 of every module that computes a number,
  plus the generator source and the scenario config, plus the commit it was frozen at. If one
  changes, every eval **refuses to run**: a result against a different harness is not comparable
  to a result against this one, and the harness says so rather than quietly producing a number
  that looks like the old one. Locks supersede forward and the chain is verifiable.
- **The open counter** — the test split opens exactly once, and `open_count` records how many
  times it has. It is a one-way door and the counter is rendered on the panel.
- **The floors and the oracle** — four floors priced on every row, and a perfect-foresight
  oracle that converts every result into a *gap to what was achievable*, so that "we captured
  this fraction of the achievable savings" replaces a savings score that means nothing to
  anyone.

The metric suite scores the **merchant-day**, not the merchant, and the positive class is
`(merchant is fraud) AND (day ≥ drift onset)` — a fraud merchant's days *before* it drifts are
legitimate days, and an alert on one is a false positive rather than an early catch.

**A defect inside the lock is named, not fixed.** Editing a locked module changes its hash and
voids the pre-registration, so where the sealed harness contains something wrong — a comparison
that is not like-for-like, a docstring that outlived its constant — it is written down in
`LIMITATIONS.md` and measured around, not silently corrected. This is the single most
counter-intuitive rule in the repo and it is the one that makes the lock mean anything.

### 4.8 Artifacts and the panel — `src/rakshak/artifacts/`, `dashboard/`

`make artifacts` emits schema-versioned JSON into `artifacts/`, and that directory is **the
panel's only data source. No backend, ever.**

- The emitter is a **pure function of its input files' contents** — no clock, no `git` subprocess,
  no network, no writes into `data/`. Regenerating from the same committed results is
  byte-identical, and that is an acceptance criterion with a test behind it, because the two
  cheapest ways to break it are a `generated_at` stamp and a `git rev-parse`.
- The **contract** and the **emitter** are separate modules, so the validator can be run over a
  file nobody in this process wrote. That is the only way to be sure the loader's check and the
  emitter's check are the same check.
- Every artifact carries `schema_version` on its envelope and the loader rejects a mismatch **by
  name and reason**. A missing artifact renders a named absence, never a blank chart standing in
  for a number nobody measured.
- The dashboard is a static React/Vite build. Its test suite mounts the whole panel against the
  **real committed artifacts**, and the Pages deployment depends on that test passing — so an
  artifact regenerated into a shape the panel cannot read fails loudly instead of publishing a
  blank page with every check green.

---

## 5. The four walls

The interesting part of this architecture is what it structurally prevents. Each of these is
enforced by a test, not by a convention.

### 5.1 Ground truth cannot reach a feature or a model

`persona_id`, `risk_typology_id`, `drift_onset_at` and everything in the ground-truth table are
radioactive. Nothing in `features/` or `models/` may import `generator` or read those fields, and
`tests/gates/test_g4_no_leakage.py` enforces it with an AST scan over the source rather than a
grep. The same file carries a **meta-test that the scanner catches a synthetic offender**, because
a leak detector that silently stopped detecting would look exactly like a clean tree.

### 5.2 Labels have exactly one door, and it takes an `as_of`

`eval/splits.available_labels(as_of)` is the only path to the label table, and it applies the
`label_available_at <= as_of` gate. That this is the *sole* path is itself enforced, by
`tests/gates/test_label_access.py`: it scans for a literal label-parquet path, for label SQL and
for a parquet reader named after the label table anywhere outside the door, and it checks that
docstrings are not being mistaken for code while it does so. With a 45–120 day chargeback delay, a harness that hands a model labels it could not have
had is not measuring detection — it is measuring hindsight.

### 5.3 An explainer may not become a scorer

An **explainer** says *why* a merchant-day looks the way it does. A **scorer** says how risky it
is, and its number enters the cost layer, the capacity selector and every committed metric.
`explain/registry.py` keeps them different roles.

The reason is specific rather than stylistic. Rung 7's HSMM runs at Stage 2 only, on merchants a
cheaper rung already promoted. Its state posterior is a superb *narrative* — "this merchant moved
into a high-refund regime on day 143" — and a terrible headline score, because it has only ever
seen the promoted subset. So the registered explainer has no `predict` method: the registry
accepts it and the scoring path cannot reach it.

### 5.4 The test split is a one-way door with one guard

Every pipeline stage — `gen`, `features`, `train`, `eval`, `report` — enters through `cli.py`
rather than into a module, and every path there that scores anything calls
`require_unlocked_or_refuse(split)` and then `verify_lock()` before a single row is read. The unlock environment variable is checked in exactly one place and reached from exactly
one place. `make eval` refuses the test split unless it is set, and it is not set anywhere in
this repository.

---

## 6. What is deliberately absent

**Rejected by design — do not introduce them.** No GPU, anywhere. No torch, no transformers, no
GNN library, no `hmmlearn`, no cloud SDK, no pandas. The CPU-only constraint is not a budget
excuse: it forces the same discipline Bumblebee already applies — deterministic cheap work first,
expensive inference only on the residual — and that discipline is what makes the cascade in §4.4
real rather than decorative. Where the no-autograd constraint was later reversed by an explicit
lead decision, the reversal is recorded as an ADR with the evidence that argued *against* it, not
quietly applied.

**Absent because it was measured and lost.** Several rungs on §4.5's ladder were built, scored on
real data and **not adopted**. They stay in the tree, in the roster and in the results table,
because Prime Directive 6 is that a rung which loses is a finding rather than an embarrassment.
Which ones, and by how much, is in the README and `LIMITATIONS.md`.

**Absent because the door stayed shut.** There is no test-split number in this project. The
pre-registered gate that governs the one-way door was not met, so it was not opened, and no
amount of "the result would probably have held" is allowed to stand in for the number that was
never taken.

**Not built, and named as not built.** BAF (Feedzai's Bank Account Fraud, NeurIPS 2022) is the
project's only external anchor, and it is **not vendored in this tree** — some gates skip for
that reason, and every G3 number is therefore synthetic-only. That is stated in the README's
second paragraph rather than in a footnote.

---

## 7. Why this shape, for the person who has to answer the phone

A Head of Risk Ops asks three questions of anything like this, and the architecture is arranged
around the third.

1. *Will it catch what I am missing?* — the ladder and the floors (§4.5, §4.7).
2. *What will it cost me in analyst hours and in angry merchants?* — the capacity-constrained
   decision layer, which is a hard budget rather than a threshold (§4.6).
3. *Can I explain the decision to the merchant when they call and shout?* — the explainer path
   (§5.3): reason codes on every non-`PASS`, and a temporal narrative naming the day the
   merchant's behaviour changed. That is an *intrinsic and temporal* kind of explanation rather
   than a post-hoc static one. It is not a claim that nobody else in the field answers the
   question — deployed systems answer it with post-hoc attribution and narration — only that
   this answers it differently.

---

## 8. Reading order

| | |
|---|---|
| [`../../README.md`](../../README.md) | The argument, the results, and the live panel. Start here. |
| [`../LIMITATIONS.md`](../LIMITATIONS.md) | Every failure, with the number. The longest document in the repo, deliberately. |
| this file | What was built and why it has this shape. |
| [`../project-context/STATE.md`](../project-context/STATE.md) | The resume point: what is open, what is closed. |
| [`PRE-REGISTRATION-CYCLE4-2026-09-01.md`](PRE-REGISTRATION-CYCLE4-2026-09-01.md) | The claims, written before the run that tested them. |
| [`results_v2.md`](results_v2.md) · [`results/cost_sweep.md`](results/cost_sweep.md) | The tables, regenerable by `make report`. |
| [`../configs/rung_roster.yaml`](../configs/rung_roster.yaml) | Every rung with its status and the citation behind it. |
| [`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) | The same document for **G2**, the previous generation. |
