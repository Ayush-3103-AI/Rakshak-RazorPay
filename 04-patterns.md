<!-- HEAD
FILE:     04-patterns.md
PHASE:    1d — UNDERSTAND
UPDATED:  2026-08-28
STATUS:   gated
SUMMARY:  Nine patterns distilled from the landscape, each terminating in an ADR or an FR —
          deterministic-rules-first, within-merchant normalisation, named-state audit trails,
          oracle ceilings, reporting the failure typology, cost-relative metrics. Seven
          anti-patterns with failure signatures so Phase 5 can recognise them early: random
          splits, ROC-AUC on imbalanced data, tuning on test, resume-driven architecture,
          burying the synthetic caveat, single-number reporting, unmapped README numbers.
OPEN:     none
-->

# 04 — Patterns

Every entry terminates in an ADR or a requirement. A pattern that changes no decision is trivia, and trivia costs every future session tokens to skip past.

---

## Patterns — what the top-percentile efforts did

### P-01 — Deterministic rules first, expensive inference only on the residual
**Evidence:** Razorpay's own Bumblebee Analyzer runs deterministic rules first and invokes the LLM only for interpretive tasks. Independent of cost, it made their system faster, cheaper, and more auditable.
**Applies here because:** our constraint is CPU-only, which forces the same discipline. Cheap deterministic features and closed-form HMM inference do the work; nothing expensive in the hot path.
**Terminates in:** ADR-0001, and the architecture diagram — *arriving independently at Razorpay's own architecture is a strong signal, so say so explicitly.*

### P-02 — Normalise within the entity, not across the population
**Evidence:** the 2008-era cardholder HMMs compared against global norms and drowned in false positives. Rieke et al. (2017) modelled each store terminal against itself.
**Applies here because:** a ₹300-AOV D2C brand and a ₹80,000-AOV electronics seller have nothing in common except that each has its own stable baseline. Deviation from *self* is the signal.
**Terminates in:** FR-007. **This is the single most important modelling decision in the project.**

### P-03 — Make the model's internal state the audit trail
**Evidence:** Razorpay documented that merchants couldn't see why funds were held, and rebuilt the case-management UI to fix it — but the underlying decision remains opaque.
**Applies here because:** Viterbi gives a state path for free. No post-hoc SHAP, no surrogate model — the explanation *is* the inference.
**Terminates in:** FR-014, and it is the reason the HMM was chosen over BOCPD.

### P-04 — Report gap-to-oracle, not an unanchored absolute
**Evidence:** framework doctrine; standard in operations research and rare in ML submissions.
**Applies here because:** on synthetic data we know which merchants go bad, so the perfect-foresight knapsack allocation of K review-hours is exactly computable.
**Terminates in:** FR-018. **Almost no student submission will have an oracle. This is disproportionate signal for one afternoon of work.**

### P-05 — Name the better method you couldn't build
**Evidence:** the GNN literature is unambiguous that graphs win on ring-structured fraud. Pretending otherwise would be caught.
**Applies here because:** stating "the correct long-term answer is a heterogeneous GNN; I approximated it with graph-derived scalars because of GPU and data constraints" converts a limitation into evidence of judgement.
**Terminates in:** ADR-0002 and the video script.

### P-06 — Use a named, citable metric rather than inventing one
**Evidence:** Bahnsen's savings score is the literature standard for example-dependent cost-sensitive fraud.
**Applies here because:** an invented metric invites the question "why this formula?" A cited one moves the conversation on.
**Terminates in:** `07-math.md §6`, FR-016.

### P-07 — Sequence tickets by risk retirement, not by comfort
**Evidence:** framework doctrine; confirmed by the shape of the risk register — A-002 can kill the project and must be attacked first.
**Applies here because:** the pleasant work (NSGA frontier, pretty figures) is also the droppable work. Doing it first would defer the discovery that kills the project until there is no time to pivot.
**Terminates in:** `11-tickets/BOARD.md` ordering.

### P-08 — Decide the scope cuts while calm
**Evidence:** universal; the reason it is in the grill output as D-17.
**Applies here because:** the Monday-night version of the builder will make worse choices than the Friday version. The cut order is already frozen in `STATE.md`.
**Terminates in:** `STATE.md §Countdown`.

### P-09 — Translate every headline number into the stakeholder's unit
**Evidence:** `02-stakeholders.md` — the risk-ops lead thinks in alert volume and analyst hours; the ML reviewer thinks in PR-AUC and splits.
**Applies here because:** the same result must appear twice, in two vocabularies, or half the panel discounts it.
**Terminates in:** FR-019, README structure, video script.

---

## Anti-patterns — with failure signatures

Phase 5 must be able to recognise these early.

### AP-01 — The random split
**Signature:** `train_test_split(X, y, random_state=42)` anywhere near this dataset. Suspiciously high scores. The same merchant ID appearing in both train and test.
**Why fatal:** fraud detection evaluated on a random split is the standard self-deception in this domain, and an ML reviewer checks for it in the first two minutes.
**Defence:** `eval/splits.py` is the only module allowed to produce splits, and it enforces both temporal and merchant-group separation. **Write a test that fails if any merchant ID appears in both sides.**

### AP-02 — ROC-AUC on a 1% positive rate
**Signature:** a reported AUROC above 0.95 with no PR curve alongside it.
**Why fatal:** ROC-AUC flatters heavily imbalanced problems. Quoting it as the headline reads as either naive or evasive.
**Defence:** PR-AUC is the headline. ROC-AUC may appear in a secondary table only.

### AP-03 — Tuning on the test set
**Signature:** "I tried a few thresholds and picked the best." Any hyperparameter chosen after looking at test performance.
**Why fatal:** invalidates every number.
**Defence:** three-way split. Validation window (month 7) is the only place thresholds and K are chosen. Test (months 8–9) is touched exactly once, at the end.

### AP-04 — Resume-driven architecture
**Signature:** a component whose removal wouldn't change any number. An RL agent with no reward. A GA over a space small enough to enumerate.
**Why fatal:** a panel member with the relevant background will ask "what does this actually buy you?" and there will be no answer.
**Defence:** every sophisticated component must have an **ablation row** proving it earns its place. If NSGA-II doesn't beat the uncoupled grid search, say so and keep the grid search.

### AP-05 — Burying the synthetic-data caveat
**Signature:** the word "synthetic" appearing once, in a footnote, on page 3.
**Why fatal:** a reviewer who discovers it themselves discounts everything. A builder who states it first is trusted.
**Defence:** it goes in the README's first paragraph, in every results table caption, and spoken aloud in the video.

### AP-06 — The single number
**Signature:** "0.94 precision." No baseline, no operating point, no cost, no failure case.
**Why fatal:** it answers none of the three questions the Head of Risk Operations actually has.
**Defence:** every headline number ships with a baseline comparison, a stated operating point, and the FP cost at that point.

### AP-07 — README numbers no script produces
**Signature:** a results table that was hand-edited during writing.
**Why fatal:** the engineering reviewer clones the repo, runs `make eval`, and gets different numbers. Instant credibility collapse.
**Defence:** `make eval` writes `results/*.md` and the README **includes** those files rather than restating them. A test asserts the README's numbers match the generated tables.

---

## The pattern that is also the pitch

P-02 + P-03 together are the project's whole argument: **normalise the merchant against itself, and let the model's own latent state be the explanation.** One gives detection quality that the 2008 literature couldn't reach; the other gives the merchant an answer when they call and shout. Everything else in the repo is scaffolding around those two sentences.
