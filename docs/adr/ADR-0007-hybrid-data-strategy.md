# ADR-0007 — Hybrid data: own generator for sequences, public data for the decision layer

**Status:** Accepted — decision taken in Phase 2 (pre-execution); **executed 2026-08-29 (T-0012).**
Written retrospectively on 2026-08-29 from `01-understanding.md` D-13, `03-landscape.md`
(dataset table), FR-001, FR-021, and `11-tickets/T-0015.md` / `T-0012.md`.
**Supersedes:** none.
**Related:** ADR-0002 (the same circularity objection, applied to graphs).

## Context

The project needs per-merchant transaction **sequences** with **merchant-level risk labels** and
known transition timestamps. A survey of the public landscape establishes the constraint this
whole ADR exists to record:

> **No public merchant-sequence dataset with merchant-level risk labels exists.**

Recorded at `06-requirements.md:28`. This is a hard boundary, not a search that ran out of time.

What public fraud data *is*: transaction-level (ULB, IEEE-CIS) or application-level (BAF). What
it can legitimately supply: **marginals and base rates** — amount distributions, inter-arrival
times, seasonality, refund and chargeback rates, payer structure. What it cannot supply:
**sequences or labels.**

Two failure modes bracket the decision. Train on synthetic data and report it as a real-world
result — dishonest. Refuse to build because no real data exists — no submission.

## Options considered

**(a) Public data only.** No merchant sequences exist, so there is no sequence layer. The project
collapses into a transaction-level classifier, which is what Razorpay's Vulcan already is.

**(b) Own generator only.** Every number then measures how well the model learned our own
assumptions. Circular by construction, and `03-landscape.md`'s dataset table says so in those
words.

**(c) Hybrid.** The own generator trains and evaluates the **sequence** layer; a public benchmark
independently validates the **decision / cost** layer on realistic imbalance and real temporal
drift. The claim splits at the layer boundary and each half is defensible on its own terms.

## Decision

(c). BAF (Feedzai, NeurIPS 2022) validates the decision layer — ~1M rows, 1.10% fraud prevalence
in the Base variant, 8 months of temporal drift, CTGAN-synthesised from an anonymised **real**
bank dataset. The generator owns the sequence layer.

**`CLAUDE.md` fixes the wording verbatim**, and the wording is itself the deliverable:

> *"Sequence-layer metrics are measured on synthetic merchant streams with injected typologies;
> the generator is in this repo. The decision layer is additionally validated on BAF (Feedzai,
> NeurIPS 2022), a public benchmark derived from real bank data."*

## Consequences

* **The second half of that sentence is now backed — 2026-08-29.** BAF was fetched
  (558 MB, SHA-256 in `data/external/baf.manifest.json`) and T-0012 ran the decision layer
  against the Base variant on BAF's native temporal split. `results/baf_validation.md` is the
  evidence; the apologetic parenthetical is gone from `results/summary.md` and the mandated
  sentence now stands unqualified. FR-021 is met.
* **What it validates is narrower than the sentence sounds, and the results file says so.**
  BMR takes the economically correct action under an extreme cost asymmetry, the capacity
  constraint binds on real data and is reported, and the savings score orders the three models
  identically at every swept asymmetry. It does **not** validate the balanced regime where
  REVIEW and HOLD genuinely trade off — no available public dataset puts real money on both
  sides of that trade.
* **BAF's granularity is narrower than the plan assumed.** It is bank **account-opening
  applications** — no amount, no timestamp, no payer, no merchant. Adequate for decision-layer
  validation; it can inform **none** of the generator's marginals. Any framing implying that BAF
  grounds the generator is wrong.
* **T-0015 executed the marginals half against a different dataset, and still should have.**
  BAF was unreachable when T-0015 ran (no Kaggle token yet), so the calibration profile was
  built from Online Retail II (UCI 502, CC BY 4.0) — real invoices, and **n = 1 merchant**, a
  UK B2B gift-ware wholesaler trading in GBP. Now that BAF is downloaded this looks like it
  could be revisited; it cannot. BAF carries no amount, no timestamp and no payer, so it can
  inform none of the profile's marginals regardless of availability. `results/calibration_gap.md`
  publishes the per-marginal divergence and states which marginals the profile cannot inform.
* **Licence terms bind what may be committed.** BAF is CC BY-NC-SA 4.0 — usable inside a
  git-ignored `data/`, **not vendorable** into this MIT-licensed repo. Note the trap T-0015
  found: BAF's GitHub `LICENSE` file is Apache-2.0 and covers the *code*, not the data.
* **Nothing under `data/` is committed except manifests.** Provenance travels as
  `data/external/*.manifest.json` — source URL, retrieval date, SHA-256, row count, licence, and
  the subsample seed if subsampled.
* **This ADR does not license a search for the dataset that does not exist.** T-0015 states the
  boundary explicitly so that neither it nor any future ticket drifts into hunting for one.
