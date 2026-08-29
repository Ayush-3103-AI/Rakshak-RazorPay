# BAF validation of the decision layer (T-0012, FR-021)

> **The decision layer is additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark derived from real bank data.** This file is the evidence for that sentence. Until it existed, `CLAUDE.md` mandated the sentence and the repo could not back it.

## What this validates, and what it does not

**BAF is bank account-opening applications.** No amount, no timestamp, no payer, no merchant, **no sequences**. So the HMM cannot run here and is not run here. Nothing in this file speaks to the sequence layer. Verbatim, per `CLAUDE.md`: *Sequence-layer metrics are measured on synthetic merchant streams with injected typologies; the generator is in this repo.*

What **is** exercised is the decision layer on real data: Bayes Minimum Risk over the cost matrix, the analyst-hour capacity constraint, and the savings score, against a real label distribution with real temporal drift.

| Field | Value |
|---|---|
| Produced by | `python -m rakshak.eval.baf --seed 42` |
| Dataset | BAF Base variant, `data/external/baf.manifest.json` |
| Licence | CC BY-NC-SA 4.0 — git-ignored, **not vendored** |
| Rows | 1,000,000 (full) |
| Split | BAF's **native** months — train 0-5 (794,989), validate 6 (108,168), **test 7 (96,843)** |
| Test-month prevalence | 0.0147 |
| Review capacity B | 387.37 h |
| Native asymmetry | 61368.0 INR FP cost per INR 100 loss |
| Swept range | 5497.4 - 519634.4, central 61368.0 |

## Two assumptions, stated because neither is derivable from BAF

**1. `proposed_credit_limit` stands in for exposure.** BAF records no realised loss and no customer lifetime value. `L = limit * r_cb * (1 + phi)` and `V = limit * g * lifetime`. Both are linear in the same column, so `L/V` is constant and **the only source of per-application threshold variation is the flat `c_support` term.** That is weaker example-dependence than the synthetic layer, where volume and lifetime move independently. Do not describe this as validating example-dependent costing; it validates the policy and the constraint.

**2. BAF's monetary unit is treated as the cost layer's monetary unit.** Absolute scale matters because `c_support` and `c_review` are absolute. **This is why the table below is reported across the whole swept asymmetry range and no single-point savings figure is quoted as the result.**

## Read this before the tables: the cost matrix sits in an extreme corner

**The native asymmetry reads 61,368 against the synthetic split's 47.5, and the swept range (5,497 - 519,634) never reaches 47.5 at any point.** So no row below is measured in the operating regime the rest of the project reports on. That is a consequence of assumption 2 above, not a property of BAF: BAF's credit limits run 190-2000 in its own units while `COST_SUPPORT_INR` and `COST_REVIEW_INR` are absolute INR constants sized for merchants doing lakhs of monthly volume, so `c_fp` dwarfs `L` for every application.

In that corner the economically correct policy is to hold almost nobody. The tables below should be read as **"does the decision layer do the right thing when false positives are overwhelmingly expensive?"** — not as a validation of the review-versus-hold trade-off at the project's own asymmetry.

## Test month (month 7) — the reported window

| model | savings | PR-AUC | precision@K | Brier | reviewed | held | capacity binds |
|---|---|---|---|---|---|---|---|
| random | -28.2169 | 0.0143 | 0.0137 | 0.3340 | 5781 | 4033 | capacity (wanted 6615) |
| credit_risk_score | -5.2810 | 0.0403 | 0.0560 | 0.3200 | 5781 | 469 | capacity (wanted 13038) |
| gbdt | +0.0294 | 0.2179 | 0.1436 | 0.0129 | 139 | 1 | none (wanted 139) |

**Read the `random` row first.** `results/summary.md` established on the synthetic split that the cost matrix, not detection, earns most of the savings *level*. The same discipline applies here: any margin quoted off the savings column is a claim about the model only to the extent it exceeds the `random` row.

## Savings across the swept asymmetry

| asymmetry | random | credit_risk_score | gbdt |
|---|---|---|---|
| 5497.4 | -17.3762 | -14.9088 | +0.0802 |
| 9707.4 | -21.6179 | -18.3907 | +0.0429 |
| 17141.3 | -24.8588 | -17.6498 | +0.0353 |
| 30268.2 | -27.0110 | -11.8878 | +0.0323 |
| 53447.7 | -27.8974 | -6.1511 | +0.0306 |
| 61368.0 | -28.2169 | -5.2810 | +0.0294 |
| 94378.1 | -28.9427 | -3.6772 | +0.0288 |
| 166653.1 | -30.5865 | -2.7835 | +0.0288 |
| 294276.5 | -30.2144 | -2.5612 | +0.0288 |
| 519634.4 | -31.1542 | -2.5536 | +0.0288 |

The range is derived from `config.COST_PRIMITIVE_RANGES`, exactly as in `results/sensitivity.md`. Nothing here is narrowed because part of it is unflattering.

## What the numbers actually say

**BMR does the economically correct thing under the corner described above.** `gbdt` holds 1 applications out of 96,843 and stays positive; `random` holds 4,033 and is destroyed. That is the policy behaving correctly under the costs it was given.

**So this file validates:**

* BMR takes the economically correct action under an extreme cost asymmetry;
* the analyst-hour capacity constraint binds on real data and is reported;
* the savings score orders the three models identically at **every** swept asymmetry, across two orders of magnitude - the ordering is not an artefact of one cost matrix.

**It does not validate** the balanced regime where REVIEW and HOLD genuinely trade off against each other. No public dataset available to this project puts real money on both sides of that trade, and this one does not either.

### The cross-check that cuts back at the synthetic split

`results/summary.md` reported that on the synthetic split `random` scored +0.6929 savings against `rules`' +0.6980 - within 0.0051 - and concluded the cost matrix, not detection, was earning the savings level (AP-06). **On BAF, at a realistic 1.47% prevalence, `random` is catastrophically negative at -28.2169.**

That points at the synthetic split's **20% merchant fraud rate**, not at the savings metric. At 20% prevalence a random policy lands on enough true positives to look competent; at 1.5% it cannot. The AP-06 warning stands - savings must never be quoted without PR-AUC beside it - but its severity on the synthetic split is substantially an artefact of a prevalence the generator inflated on purpose, for per-typology sample size. **T-0011 should say both things.**
