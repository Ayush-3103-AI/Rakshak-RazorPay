# ADR-0006 — Per-merchant cost parameters are shrunk to their segment, in closed form

**Status:** Accepted (decision) — **but NOT IMPLEMENTED. T-0008 was cut on 2026-08-28.**
Decision taken in Phase 2; **written retrospectively on 2026-08-29** from `07-math.md` §4,
FR-011, FR-016 and `11-tickets/T-0008.md`.
**Supersedes:** none.
**Related:** ADR-0005 (the policy that consumes the calibrated posterior this ADR was to produce).

## Context

The decision layer is example-dependent: every merchant gets their own threshold, derived from
their own `L_m` and `c_fp(m)`. That immediately raises the cold-start objection, and it is the
first thing a Head of Risk Operations will ask — **"what does your system do on a merchant with
eleven transactions?"**

A per-merchant MLE on eleven transactions is noise. A single global parameter throws away the
economics that make the approach worth having. The standard answer is partial pooling.

For a per-merchant parameter `theta_m` with segment-level hyperparameters estimated across
merchants:

```
theta_hat_m = w_m * theta_MLE_m + (1 - w_m) * theta_segment

w_m = var_between / (var_between + var_within / n_m)
```

James–Stein / partial-pooling form. `var_between` from the variance of segment members' MLEs;
`var_within` from within-merchant sampling variance.

Its properties are exactly the answer to the objection:

* `n_m` large ⇒ `w_m` → 1 — high-volume merchants converge to their own economics.
* `n_m` small ⇒ `w_m` → 0 — new merchants inherit their segment's.
* Continuous in between — **no arbitrary cliff at a minimum-volume gate.**

FR-011 supplies the segment definition this requires: MCC × AOV-band, with at least 20 merchants
per segment in training.

## Options considered

**(a) Per-merchant MLE only.** Noise-dominated on low-volume merchants, which are most of them.

**(b) One global parameter.** Discards the example-dependence that is the point of the cost layer.

**(c) A minimum-volume gate** — MLE above N transactions, segment default below. Introduces a
cliff, and merchants will sit either side of it with near-identical economics and different
treatment.

**(d) Full Bayesian hierarchical model via MCMC (PyMC / NumPyro).** Correct, and disqualified by
compute: MCMC on CPU against NFR-004's 15-minute `make eval` budget.

**(e) Empirical-Bayes shrinkage, closed form.** Same shape as (d)'s posterior mean, no sampler,
a handful of lines of numpy.

## Decision

(e). Closed form, **no MCMC, no PyMC** — recorded in `CLAUDE.md`'s stack table as a locked
constraint, not a preference.

## Consequences

* **It was not built.** T-0008 was cut in the 2026-08-28 re-plan; it sat fourth in the cut list
  and lost its slot to T-0006b (the HMM had no scorer at all), T-0007a/b (the cost matrix was
  definitionally wrong) and T-0012 (the repo made a BAF-validation claim it could not back).
* **The cut became load-bearing later, in a way the re-plan did not foresee.** ADR-0005's BMR
  policy consumes each model's raw score as if it were a calibrated `P(bad)`. Under the earlier
  top-K placeholder that only cost the HMM its Brier gap — a ranking policy does not care about
  calibration. **Under BMR, miscalibration moves the argmin, not merely the ordering.** The HMM's
  Brier is 0.3149 against `gbdt`'s 0.1242, so this is not hypothetical.
* **Therefore `savings` and `Brier` are coupled in the current results** in a way they would not
  be in a calibrated system, and `results/summary.md` says so. This is the strongest single
  argument for reinstating T-0008 if a day appears.
* **The cold-start answer is currently unbuilt.** The demo artifact §4 describes — animating one
  merchant's threshold migrating from its segment default to its own economics as volume
  accumulates — does not exist. Do not describe it as though it does.
* **FR-016 cites this ADR and arguably should not.** FR-016 is Bayes Minimum Risk, whose
  authorities are Elkan (2001) and Bahnsen (2015); the shrinkage this ADR covers is a separate
  concern that belongs to FR-011. Minor mis-citation, recorded rather than silently corrected.

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
