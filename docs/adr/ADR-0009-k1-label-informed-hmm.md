# ADR-0009 — Response to K1: label-informed HMM estimation and a re-specified FR-013 metric suite

**Status:** Accepted — 2026-08-28 (T-0004b). **Renumbered from ADR-0005 on 2026-08-29** — see
"Numbering" below. Promoted from the `PROPOSED` stub in
`project-context/12-lit-survey-k1.md` and reconciled here against what T-0004b actually measured.
**Amends:** FR-013 (`06-requirements.md`).
**Related:** ADR-0001 (owning the estimator is what made this implementable).

## Numbering

The stub was drafted as `ADR-0005` inside the K1 literature survey on 2026-08-28. **That number
was already taken** — FR-015, FR-017 and `07-math.md` §7 had cited ADR-0005 for the three-action
policy since Phase 2. This decision takes **0009**; the policy keeps 0005. Citations were
updated in place on 2026-08-29 with a note, and no dated amendment block was rewritten to conceal
the collision.

## Context

FR-013 required four-way latent-state recovery at **ARI > 0.5**. T-0004 measured **0.091**.

The load-bearing number is not the failure — it is the ceiling. An **oracle-parameterised** HMM,
with parameters read straight off ground truth, reaches only **0.378** (0.404 after T-0003b,
0.381 on the validate group). That is the fully-supervised MLE estimator. **The gate is
unreachable by any correctly-implemented HMM on these emissions.** Two genuine bugs were found
and fixed en route and **both moved ARI down**, which is what established the gap was not
debuggable.

The cause is per-state overlap. **RAMP sits 1.19σ from HEALTHY**, which holds ~90% of window
mass. RAMP is the early-warning state — so the gate failed precisely on the product premise.

The literature split the failure into a **closable estimation gap** and an **unclosable
representation gap**, and established that ARI is the wrong index for a 90 / 6.4 / 3.4 / 2.2
reference partition (Romano, Vinh, Bailey & Verspoor, JMLR 17, 2016 — ARI for balanced
references, AMI for unbalanced ones with small clusters).

## Options considered

**A. Keep FR-013 as written and report the failed gate.** Honest, and hands a competitor the
headline.

**B. Collapse RAMP into HEALTHY and score three-state recovery.** Deletes the early-warning
claim — the product premise — to make a metric pass.

**C. Build an HSMM with explicit RAMP duration.** 14+ hours; `hsmmlearn` is **GPLv3** (excluded
by `CLAUDE.md`'s licence constraint), `pyhsmm` is unmaintained and Py3.7. Ruiz-Suarez et al.
(2021) indicate overlap, not duration, is the binding constraint here.

**D. Promote BOCPD from baseline to primary.** Adams & MacKay define changepoints as **abrupt**
parameter variations; RAMP is gradual by construction. Wrong tool for the target state.

**E. Replace the HMM with a linear-chain CRF.** 8 hours plus a new dependency, and it abandons
the locked generative framing that gives Viterbi its explanation path.

**F. Label-informed (partially-supervised, weighted-likelihood) estimation inside the existing
hand-written HMM, plus a re-specified metric suite.**

## Decision

**F, with A retained inside it.** The 0.091 four-way ARI and the oracle ceiling are reported
**permanently** alongside every new number. That is what makes the amendment credible rather than
convenient.

1. **Amend FR-013's metric suite** to AMI (primary four-way index), ARI (**retained**), per-state
   recall and its macro-average, binary non-healthy PR-AUC at base rate, and per-typology
   detection lag — each reported with its oracle ceiling.
2. **Replace unsupervised Baum-Welch with weighted-likelihood partially-supervised EM**: clamp
   γ to known states on labelled **training-split** windows only; up-weight rare labelled states
   in the M-step.
3. ~~Handle DORMANT by a deterministic rule on `sparse` and fit K=3 on the remainder.~~
   **REFUTED at T-0004b.** Not shipped.
4. ~~Dirichlet transition prior with a sticky self-transition term and emission variance floors.~~
   **Measured null at T-0004b** — ARI change of literally 0.0000. Not shipped.
5. Keep BOCPD as the T-0010 baseline with a pre-registered prediction. **T-0010 was subsequently
   cut**, so no sequence-aware baseline other than the HMM was ever measured; T-0011 reports that
   question as open.

**Shipping configuration is items 1 + 2 only.**

## Rationale

Elworthy (ANLP 1994) and Merialdo (CL 1994) establish that unsupervised Baum-Welch degrades
accuracy when labels exist. Li, Zhou & Wang (2024) show EM's contraction radius approaches 1
under rare-event mixtures and that partial labels fix it. Sidrow et al. (PLOS ONE 2025) give the
weighted-likelihood recipe and report improved accuracy *and* interpretability. Ruiz-Suarez et
al. (2021) find misspecification largely benign except under high state-distribution overlap,
which demotes emission-family and duration remedies. ~12 developer-hours, no new dependency,
CPU-only, no change to Viterbi-path explainability.

## Consequences

* **It worked, on the headline metrics.** ARI 0.134 → 0.319, AMI 0.102 → 0.218, binary PR-AUC
  0.109 → 0.327 — about 85% of the way to the ceiling and **never above it**.
* **And it made the thing we care about worse.** **RAMP recall fell 0.328 → 0.234** while every
  headline metric roughly doubled. Labels help rare *separable* states, not rare *overlapping*
  ones. **The configuration that wins overall is the one that goes blind on the state the project
  exists to catch.** Decision taken: ship item 2 as primary and report both configurations side
  by side, with the RAMP regression stated prominently. This must reach the video.
* **A pre-registered bar failed and was kept.** The survey recorded RAMP-recall ≥ 0.35 *before*
  measuring; it came in at 0.234 and is committed as a strict `xfail`. It stays.
* **The sequence layer is now label-informed on SYNTHETIC generator labels.** This is a stronger
  limitation than the plain synthetic-data caveat and must be stated verbatim in the README, the
  video and every results table. The video must not imply the sequence layer is unsupervised.
* **T-0006b's fit regime is not T-0004b's and the numbers are not comparable.** T-0004b was
  transductive (all merchants, labels restricted, 38% of windows labelled); a harness scorer
  cannot be, so T-0006b fits on `train` alone where ~96% of windows carry labels. T-0004b's
  ARI/AMI figures must not be quoted against T-0006b's row.
* **FR-013's original 0.5 threshold is retired as unreachable** on these emissions, with the
  oracle ceiling as the evidence. **Do not remove or bury either the ARI or the ceiling.**

## Revisit triggers

* **Label-informed four-way ARI materially exceeds the oracle ceiling** → suspect leakage; audit
  `eval/splits.py` before believing it. Do not celebrate.
* **Any real (non-synthetic) merchant stream becomes available** → re-measure RAMP separation.
  The 1.19σ figure is a property of the generator, not of fraud.
* **A build window ≥ 3 days AND a permissively-licensed, maintained HSMM library** → revisit
  option C.
