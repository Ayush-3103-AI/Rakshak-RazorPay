# Rakshak — evaluation summary

> **Sequence-layer metrics are measured on synthetic merchant streams with injected typologies; the generator is in this repo.** The decision layer is additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark derived from real bank data. (BAF validation lands in T-0012 and is absent below.)

## Provenance

| Field | Value |
|---|---|
| Produced by | `python -m rakshak.eval.harness --seed 42` |
| Seed | 42 |
| Split reported | `validate` (days 180-209) |
| Data horizon | day 0 = 2026-01-01, 500 merchants |
| Review budget K | 5 merchants (0.40 h / 0.067 h per review) |
| Capacity rule | 4.0 analyst-hours per 1000 merchants under watch, scaled to this split's 100 merchants (ADR-0008) |
| Bad states | DORMANT, FRAUD, RAMP |

The test window (days 210-269) is **not** touched by this run. 06-requirements.md §3 reserves it for T-0011/T-0013.

## Prevalence — read every precision number against this

- **20 of 100 merchants in this split are truly bad (20.0%).**
- The generator's `FRAUD_MERCHANT_RATE = 0.20` is far above real-world merchant-fraud prevalence. It was chosen so each of the five typologies has enough merchants for a per-class metric, not because it is realistic.
- A precision of P here corresponds to a lift of P / 0.20 over random selection. Quote the lift, not the precision, when comparing against a real base rate.

## Splits — merchant counts (NFR-002: disjoint merchant IDs, enforced in code)

| typology | train | validate | test |
|---|---|---|---|
| BUST_OUT | 12 | 4 | 4 |
| CATEGORY_DRIFT | 12 | 4 | 4 |
| LAUNDERING_ENDPOINT | 12 | 4 | 4 |
| NONE | 240 | 80 | 80 |
| REFUND_COLLUSION | 12 | 4 | 4 |
| SLOW_RAMP | 12 | 4 | 4 |
| TOTAL | 300 | 100 | 100 |

- `train`: days 0-179 (6 month(s))
- `validate`: days 180-209 (1 month(s))
- `test`: days 210-269 (2 month(s))

## Ceilings — perfect foresight

| ceiling | reviewed | held | hours used | loss averted (INR) | savings |
|---|---|---|---|---|---|
| oracle (review knapsack, perfect foresight) | 5 | 0 | 0.34 | 384,497 | 0.3169 |
| oracle (perfect hindsight, unconstrained) | 0 | 20 | 0.00 | 569,845 | 0.8262 |

> **The review budget binds.** K = 5 review slots (0.40 analyst-hours at 0.067 h per review) against 100 merchants, 20 of them truly bad. Even with perfect foresight the knapsack oracle can only reach 5 of them, leaving 15 bad merchants unreviewed for want of analyst hours. K covers 5% of the book against a 20% prevalence, so `precision@5` has real headroom and the baselines can separate on it. Capacity is now expressed per 1000 merchants (ADR-0008, T-0003b); before that it was an absolute 40 h = 597 slots and bound nothing.

## Cost-matrix cross-check (07-math.md §5, as amended by T-0017)

> Indian payments commentary estimates INR 400-600 lost to falsely declined legitimate orders for every INR 100 saved by preventing fraud.

| quantity | value |
|---|---|
| Total false-positive cost, all healthy merchants held (INR) | 301,019 |
| Total fraud loss, all bad merchants passed (INR) | 633,162 |
| INR of FP cost per INR 100 of fraud loss | 47.5 |
| 07-math.md §5 cross-check (commentary, not a gate) | 400 - 600 |
| Divergence from the band | 47.5 vs 400-600 — reported, not closed |

**T-0017 demoted this row from a gate to a cross-check, and T-0007a corrected the two definitions underneath it.** `V_m` is now expected *lifetime* gross margin (`g * v_m * l_m`, with `g` the platform's own ~10 bps of TPV, not the merchant-facing 2% MDR — a price is not a margin), and `L_m` is *realised* loss (`r_cb * (1 + phi) * G_bad`), not gross turnover during a bad state. Turnover is not loss: a bust-out processing INR 10,00,000 with INR 50,000 charged back cost the acquirer INR 50,000 plus fees. The previous definitions were wrong by roughly 15x on `L_m` and 3x net on `V_m`, in opposite directions — they partly cancelled, which is why no sanity check on this ratio alone could ever have found them. Every constant carries a citation or an explicit ASSUMPTION tag and a range in `config.py`. **The divergence from 400-600 is stated, not tuned away**: the commentary band measures declined baskets at checkout, this ratio measures held settlements costing the platform its own margin. They were never the same asymmetry, and closing the gap by choosing constants that land in the band is the practice T-0016 forbids for the generator.

## Models

All rows share the same analyst-hour budget. Actions come from the cost-optimal three-action policy in `decision/policy.py` (T-0007b): Bayes Minimum Risk over {PASS, REVIEW, HOLD} under the cost matrix, then the capacity constraint. It replaced `harness.budget_policy`, the top-K placeholder that never held anyone.

**No verdict is rendered here (T-0006 is plumbing).** These are the baseline rows only; the comparison that decides anything happens at T-0011 on the test window, with the sequence layer present.

**`gbdt` caveat.** LightGBM early-stops its iteration count on this same `validate` split, as 06-requirements.md §3 directs ("all hyperparameters and thresholds chosen on the validation window"). Its row here is therefore mildly optimistic while the harness reports validate; it is clean at T-0011, where the reported window is `test` and validate is only the early-stopping set. `rules` has no fitted quantity at all and `random` has none either, so neither carries this caveat.

**`hmm` is the proposal.** The row comes from `models/hmm_score.py` (T-0006b): a per-merchant belief over four latent states, fitted on the training split alone with T-0004b's shipping configuration, scored by the **forward-only filtered posterior** so that neither `score` nor `flag_day` uses information from after the window it reports on. A truncation test proves this, and carries a negative control that runs the same assertion against the smoothed posterior and requires it to fail, so the proof cannot be vacuous.

**`savings` became readable at T-0007a** and these are the first cost numbers in the project worth reading. Two caveats bind them. First, this is the `validate` split — the `test` window is reserved (06-requirements.md §3) and **no verdict is rendered here**; K2 renders at T-0011. Second, T-0007b's BMR policy consumes each model's raw score as a posterior — **T-0008's empirical-Bayes calibration was cut in the 2026-08-28 re-plan and no recalibration happens anywhere in this repo.** A miscalibrated score moves the argmin, not merely the ranking, so `savings` and `Brier` are coupled here in a way they would not be in a calibrated system. Read PR-AUC, precision@K, Brier and median lag alongside `savings`, not instead of it. The sensitivity of every `savings` figure to the cost asymmetry is in `results/sensitivity.md` (FR-020).

**Both median-lag cells reading -1.0 is a definitional artefact, not leakage.** A flag is attributed to the *start* day of the 7-day window that produced it, so a merchant going bad on day 192 detected from the window opening day 189 records lag -3. `validate` holds only four whole windows, so every flag lands on one of four days. T-0011 must state this or move both models to window-end attribution together.

| model | savings | gap to knapsack oracle | gap to hindsight oracle | PR-AUC | precision@5 | Brier | median lag (days) | flagged frac | reviewed | held | hours | capacity binds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.6929 | -1.1862 | 0.1614 | 0.1651 | 0.0000 | 0.3589 | n/a | 0.00 | 5 | 11 | 0.34 | capacity (wanted 15) |
| rules | 0.6980 | -1.2023 | 0.1552 | 0.5377 | 0.8000 | 0.1319 | 3.0 | 0.45 | 5 | 8 | 0.34 | capacity (wanted 16) |
| gbdt | 0.7392 | -1.3324 | 0.1053 | 0.6778 | 1.0000 | 0.1242 | -1.0 | 0.50 | 5 | 12 | 0.34 | capacity (wanted 10) |
| hmm | 0.7464 | -1.3550 | 0.0966 | 0.4994 | 0.6000 | 0.3149 | -1.0 | 0.65 | 5 | 12 | 0.34 | capacity (wanted 6) |

**`gap to knapsack oracle` is now a category error and is printed only because deleting it would hide the reason.** The review-knapsack ceiling is the best *review-only, <= K* allocation; T-0007b's policy may HOLD, and nothing bounds a holding policy by a review-only ceiling. T-0007a wrote this down before it bit (`tests/test_cost.py` header: *"nothing forces it above hold-everything"*). Read `gap to hindsight oracle` instead, and read `results/sensitivity.md` for the asymmetry below which the knapsack ceiling stops clearing hold-everything altogether.

`capacity binds` reports FR-017's binding constraint per model, with the number of reviews unconstrained BMR *wanted* beside it. A model whose unconstrained demand is below K is not being limited by analyst hours at all, and its row must not be read as a capacity result.

### The `random` row is the most important number in this table

**Under the BMR policy, `random` scores 0.6929 savings against `rules`' 0.6980 — a gap of 0.0051 — while ranking at PR-AUC 0.1651, i.e. at this split's prevalence.** Nothing about the model produced that; the cost matrix did. A uniform random score still lands most merchants on the correct side of a merchant-specific threshold when `c_fp` is small relative to `L_m`, so most of the savings on this split is attributable to the decision layer's cost arithmetic rather than to detection. This is 07-math.md §6's AP-06 guard arriving as a measurement rather than as a warning: **the savings score is manipulable through the cost matrix and must never be quoted without PR-AUC beside it.** Any headline of the form "Rakshak saves X%" that does not subtract the random floor is not a claim about the model. T-0011 must report savings *relative to the `random` row*, not in absolute terms.

This also changed the ordering. Under T-0006's top-K placeholder the HMM sat below both baselines on savings; under BMR it sits above them. STATE.md predicted the mechanism before the policy existed — a well-covering but badly-calibrated model was penalised twice by a rank-only policy. **That is an explanation, not a verdict.** The verdict is T-0011's, on the test window, and the `random` row above says how much of any margin is the cost matrix.

### Models absent from this run

| model | status |
|---|---|
| bocpd | **ABSENT** — T-0006 — changepoint baseline |

These rows are missing, not zero. No headline claim can be made from this run until they land.

## Metrics deliberately not reported

**ROC-AUC and raw accuracy are prohibited as headline metrics** (06-requirements.md §3) and are not implemented in `rakshak.eval.metrics`. At 20% prevalence ROC-AUC flatters every model and "predict healthy" beats most models on accuracy.

Median detection lag reads `n/a` for any model that does not return a `flag_day`; a single per-merchant score has no time at which it fired.

