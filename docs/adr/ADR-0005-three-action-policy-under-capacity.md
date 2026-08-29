# ADR-0005 — Three actions (pass / review / hold) under a hard review-capacity constraint

**Status:** Accepted — decision taken in Phase 2 (pre-execution); **implemented 2026-08-29 at
T-0007b.** Written retrospectively on 2026-08-29 from FR-015, FR-017, `07-math.md` §§6–7 and
`11-tickets/T-0007b.md`.
**Supersedes:** none.
**Related:** ADR-0008 (how the capacity figure is expressed), ADR-0004 (how the thresholds were
to be optimised — cut), ADR-0006 (the calibration this policy assumes — cut).

> ### Numbering collision, resolved 2026-08-29
>
> **`ADR-0005` was booked twice.** This decision — the three-action policy — is the original
> holder: FR-015, FR-017 and `07-math.md` §7 have cited ADR-0005 for it since Phase 2, before
> execution began. On 2026-08-28 the K1 literature survey
> (`project-context/12-lit-survey-k1.md`) drafted a *different* ADR under the same number, for
> label-informed HMM estimation, and `06-requirements.md`'s FR-013 amendment block cited it.
>
> **Resolution: this file keeps 0005** (earlier claim, more citations, all in frozen spec
> documents). **The K1 response is renumbered to ADR-0009.** The renumbering is noted in place
> wherever the old number appears; no dated amendment block was rewritten to hide it.

## Context

Most fraud systems are binary: allow or block. FR-015 requires three actions, and the third one
is the product.

The economics are asymmetric and **example-dependent** — they differ per merchant. Holding a
good merchant's settlements costs their expected lifetime gross margin plus support handling
(`c_fp(m)`, see `07-math.md` §5 as corrected by T-0017). Passing a bad merchant costs realised
chargeback loss (`L_m`). Between them sits **review**: it costs analyst time (`tau` hours at
`WAGE_ANALYST_INR_PER_HOUR`), it is imperfect (`P_ANALYST_MISS`), and **it is scarce**.

Scarcity is the part that most submissions omit. An unconstrained policy that reviews everything
suspicious is not a policy; it is a wish. FR-017 makes the analyst-hour budget a hard constraint
and requires the binding constraint to be **reported**, not silently dropped.

## Options considered

**(a) Binary allow/block at a tuned threshold.** Discards review, and with it the only action
that can correct a false positive before it reaches the merchant. Also discards the explanation
path — a held merchant who calls to shout has no case to be examined.

**(b) Three actions, ranked by score, top-K reviewed.** Simple, and it is what
`harness.budget_policy` did as a placeholder. It is a **ranking** policy, not a cost policy: it
ignores that merchants differ in `L_m` and `c_fp(m)` by orders of magnitude, and it penalises a
well-covering but badly-calibrated model twice — once for ranking, once for calibration.

**(c) Three actions by Bayes Minimum Risk**, choosing the argmin of expected cost per merchant
given the posterior and that merchant's own cost parameters, then allocating scarce review
capacity by expected regret.

## Decision

(c). `src/rakshak/decision/policy.py` returns exactly one of `{PASS, REVIEW, HOLD}` **plus the
expected cost of each alternative** (FR-015), allocates REVIEW under the analyst-hour budget, and
reports which constraint binds per model alongside the number of reviews unconstrained BMR
*wanted* (FR-017). Savings is scored per Bahnsen et al. (2016).

## Consequences

* **Replacing the top-K placeholder reversed the model ordering on savings** — the HMM moved from
  −0.3625 to +0.7464, above both baselines, while PR-AUC and Brier did not move. That is an
  explanation of the placeholder's double penalty, **not** a model improvement, and `STATE.md`
  records it as such.
* **It exposed a measurement that outranks it.** Under BMR, `random` scores **+0.6929** savings
  against `rules`' **+0.6980** while ranking at PR-AUC 0.1651 — this split's prevalence. The cost
  matrix earns almost all of the savings *level*, not detection. Savings may never be quoted
  without PR-AUC beside it, and T-0011 must report it relative to the `random` floor.
* **It assumes a calibrated posterior it does not have.** BMR consumes each model's raw score as
  `P(bad)`. ADR-0006's shrinkage was cut, so **no recalibration happens anywhere in this repo**.
  Under a rank-only policy miscalibration cost only the Brier gap; under BMR it moves the argmin.
  This is the single strongest argument for reinstating T-0008.
* **It broke an invariant, correctly.** `review_knapsack_oracle` is a ceiling over the
  review-only action class; this policy can HOLD, so the ceiling never bounded it. T-0007a
  predicted this in `tests/test_cost.py`'s header before it happened. The invariant is now scoped
  per action class — see the amendment block in `11-tickets/T-0007b.md`.
* **The thresholds are not optimised.** ADR-0004's NSGA-II frontier was cut, so thresholds come
  from the closed-form BMR boundary rather than from a searched frontier.
