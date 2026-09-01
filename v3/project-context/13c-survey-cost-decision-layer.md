<!-- HEAD
FILE:     project-context/13c-survey-cost-decision-layer.md
PHASE:    survey (cycle 4)
UPDATED:  2026-09-01
STATUS:   ready-for-pre-registration — first place is unambiguous and source-verified
SUMMARY:  The floor-fail is not a ranking failure and not primarily a calibration failure.
          Rung 2 beats `volume_rank` on BOTH precision@K (0.864 vs 0.571) and recall@K
          (0.314 vs 0.195) at the same K and still loses on rupees. Source reading shows
          why: `true_loss_amount_inr = loss_fraction x post-onset REALISED captured GMV`,
          the `volume_rank` floor ranks on realised captured GMV, and the decision layer
          is handed `p_declared_monthly_gmv` — a static self-declaration — as its
          `exposure_inr`. The floor gets a near-sufficient statistic for the objective's
          magnitude term; the model gets a self-reported proxy for it. First place is a
          realised-exposure decision-policy wrapper on the existing seam.
OPEN:     (1) The exact split between the exposure mechanism and the floor/rung action
          mismatch (floors are REVIEW-only at 250 INR/error; rungs may HOLD at 8250
          INR/error) is not settled without running the three-cell diagnostic in section 1.
          (2) The ladder artifact is single-seed (`n_seeds: 1, seeds: [42]`); with ~41
          fraud merchants and a heavy-tailed loss, the effective sample size for savings
          is the number of LARGE-loss frauds, plausibly 5-10. Every savings number below,
          including the 0.6017 floor, is weaker evidence than its four decimal places
          suggest. (3) Whether the fix degenerates into `volume_rank` is a real risk and
          is why the adoption gate carries an anti-degeneracy clause.
-->

# Survey — the cost and decision layer

## 0. What this file is, and the one thing to read if you read nothing else

Three parallel surveys feed cycle 4. This is the one that owns the question *why does a
detector at PR-AUC 0.836 lose money to a size ranking at PR-AUC 0.217*.

The answer is not in the papers. It is in `src/rakshak/generator/engine.py:536` and
`src/rakshak/cli.py:920`, and the papers explain why it was invisible. The literature's
contribution here is diagnostic vocabulary, not a method to import.

**The one-line version.** Savings is denominated in
`true_loss_amount_inr = loss_fraction x post-onset captured GMV`. The `volume_rank` floor
is scored by ranking on *realised captured GMV*. The learned rungs' decision layer is
scored by ranking on `p_hat x p_declared_monthly_gmv` — the merchant's *self-declared
monthly GMV at onboarding*. Both are "expected-value ranking". Only one of them is using
the exposure variable the objective is actually written in.

---

## 1. The explanation, first

### 1.1 The fact that kills the ranking hypothesis

Pull the numbers that matter out of `artifacts/ladder.json` (VALIDATION, seed 42,
K = 15/day, 3,036 merchants, 181,980 rows, ~60 days, prevalence 0.01352, oracle 0.9087):

| model | PR-AUC | ECE | precision@K | recall@K | alerts/day | savings |
|---|---|---|---|---|---|---|
| `volume_rank` | 0.2169 | 0.4866 | **0.5714** | **0.1951** | 15.0 | **0.6017** |
| Rung 2 (LightGBM) | 0.8357 | 0.0079 | **0.8637** | **0.3142** | 15.0 | **0.4131** |
| Rung 3 (+cohort) | 0.8551 | 0.0080 | 0.8559 | 0.3114 | 15.0 | 0.4331 |
| Rung 4 (cost-in-loss) | 0.8534 | 0.0433 | 0.3903 | 0.1402 | 15.0 | 0.4348 |
| Rung 5 (MIL) | 0.7176 | 0.1630 | 0.9362 | 0.0179 | **1.0** | 0.0250 |
| Rung 6 (conformal) | 0.8319 | 0.0086 | 0.9167 | 0.3333 | **8.0** | 0.2439 |

Read the second row against the first. **At the same budget, Rung 2 alerts on more frauds
(recall 0.314 vs 0.195) and wastes fewer slots (precision 0.864 vs 0.571), and still loses
1,900 basis points of savings.**

That is arithmetically impossible under any hypothesis whose causal variable is *ranking
quality*. A ranking explanation has to predict that the loser catches fewer frauds, or
burns more slots, or both. Neither holds. Rung 2 wins the count contest decisively and
loses the rupee contest decisively.

Only two families of explanation survive that:

- **(a) It catches the wrong frauds.** The frauds Rung 2 finds are worth systematically
  fewer rupees than the ones `volume_rank` finds. Savings is loss-weighted; count recall
  is not.
- **(b) Its errors cost more per error.** The two are not being scored under the same
  action policy.

Both turn out to be true, and both are verifiable from source without running anything.

### 1.2 Mechanism M1 — exposure misspecification (the dominant one)

Three lines of the tree, all outside the locked eval modules:

1. `src/rakshak/generator/engine.py:536` —
   `true_loss = max(loss_fraction * post_onset_gmv, min_true_loss_inr)`, and
   `generator/config.py:240` states it plainly: *"true_loss_amount_inr = loss_fraction x
   post-onset captured GMV"*. **The objective's magnitude term is realised captured GMV.**
2. `src/rakshak/cli.py:304-325` `_observed_volume()` — *"Captured, non-refunded GMV per
   merchant up to `cutoff_day`. The volume the `volume_rank` floor ranks on."*
   **The floor ranks on realised captured GMV.** Same quantity, one integration window
   earlier.
3. `src/rakshak/cli.py:688, 920` — `exposure_inr = rows.column("p_declared_monthly_gmv")`,
   and that array is what is handed to `select_actions(...)`. `p_declared_monthly_gmv`
   is defined in `features/tier1.py:1620` as `profile.declared_monthly_gmv`: a **static,
   self-reported onboarding declaration.** **The decision layer ranks on
   `p_hat x declared GMV`.**

The locked decision layer is already doing textbook expected-value ranking. `capacity.py`
`expected_costs()` gives, for the REVIEW branch,

```
benefit(REVIEW) = cost_pass - cost_review = p_catch * p * exposure - review_cost
                = 0.80 * p_hat * exposure_inr - 250
```

which *is* `P(fraud) x exposure`, exactly the move the brief asks whether the project is
making. It is making it. It is making it against the wrong `exposure`.

So the competition is not "smart model vs dumb baseline". It is:

- floor: rank on `E[loss magnitude]`, estimated well, with `p = const`;
- rung: rank on `p_hat x E[loss magnitude]`, with the magnitude estimated by a
  self-declaration.

If `declared_monthly_gmv` and realised captured GMV disagree over orders of magnitude —
and `features/tier1.py:700` builds an entire feature (`v_declared_ratio`) out of the fact
that they disagree, calling it *"the promise-versus-reality gap ... the clearest single
argument for the post-onboarding surveillance position"* — then multiplying a good `p_hat`
by a bad magnitude estimate produces a *worse* rupee ranking than the good magnitude
estimate alone. The `p_hat` term is not adding signal fast enough to pay for the noise the
exposure term injects.

That is the whole story, and it explains the specific shape of the failure: Rung 2's extra
count-recall is spent on merchants whose *declared* GMV is large, which is a different set
from the merchants whose *realised* GMV is large.

**Back-of-envelope on the loss-weighted recall.** For a REVIEW-only policy,
`savings = (p_catch * L_caught - review_cost * n_alerts) / B` where `B` is the sum of
fraud losses. `volume_rank` is REVIEW-only by construction (see M2), with ~900 alerts over
the window, so `250 * 900 = 225,000 INR` of review cost. Unless `B` is tiny, that term is
small, and `0.6017 ~= 0.80 * LWR` gives **loss-weighted recall ~ 0.75 for `volume_rank`
at 19.5% count recall**. Rung 2's mixed policy makes the same inversion approximate, but
the ceiling `0.4131 / 0.80` puts its LWR near **0.5 at 31.4% count recall**. The floor
catches three quarters of the money with one fifth of the cases; the model catches half
the money with a third of the cases. The loss distribution is that concentrated, and
`volume_rank` is a direct estimator of where the concentration is.

### 1.3 Mechanism M2 — the floor and the rungs are not scored under the same action policy

This one is subtle and matters for how the FLOOR-FAIL verdict should be *reported*, not
for whether the model is bad.

- `metrics.py:393-394` — floors are computed by `savings_of_ranking(...)`, whose signature
  carries `action: Action = Action.REVIEW`. **Every floor alert is a REVIEW.** By
  `row_cost`, a REVIEW on an honest merchant costs `review_cost_inr = 250`.
- `metrics.py:605` — the rung's savings is `savings_of_actions(output.action[keep], ...)`,
  and `output.action` comes from `select_actions`, which emits HOLD whenever
  `cost_hold < cost_review` and the `ActionPolicy` gate permits. **A HOLD on an honest
  merchant costs `false_hold_cost_inr + review_cost_inr = 8,250`** — 33x a false REVIEW.

`savings_of_ranking`'s own docstring says *"the only thing that differs between them is
the score vector. That is what makes a FLOOR-FAIL attributable to the ranking rather than
to two different decision policies."* That is true floor-against-floor. It is **not** true
floor-against-rung, because the rung does not go through `savings_of_ranking`.

Neither function is wrong; `capacity.py` and `metrics.py` are both correct and both stay
byte-identical. But the comparison that produces the word FLOOR-FAIL is not action-matched,
and the honest report of that is a sentence in the writeup, not an edit to the harness.

Note the sign of the effect is not obvious a priori, which is why it needs measuring
rather than asserting. `cost_hold` is **flat in exposure** (`250 + (1-p) * 8000`, capped at
8,250) while the *benefit* of HOLD over REVIEW on a true fraud scales as
`(1 - p_catch) * loss = 0.2 * loss`. On a very large true fraud, HOLD is nearly
free-rolling. On an honest merchant it is a flat 8,250. So HOLD is a high-variance bet that
pays off in proportion to exposure and loses a constant — which is exactly the bet that goes
wrong when your exposure estimate is the self-declared one (M1 again).

**Rung 6 is the confirming experiment, run by accident.** `ConformalHold` rewrites
HOLD -> REVIEW to control the realised false-HOLD rate. PR-AUC barely moved (0.8551 ->
0.8319) and precision@K went *up* (0.856 -> 0.917), and savings **halved** (0.4331 ->
0.2439). Softening a HOLD on a large true fraud converts a cost of 250 into a cost of
`250 + 0.2 * loss`. If savings were driven by ranking, that swap would be nearly free. It
was not free. **Savings is dominated by the action taken on a handful of the
largest-exposure merchants.** That is the same claim as M1, arriving from the opposite
direction.

### 1.4 Mechanism M3 — capacity under-spend

`alerts_per_day` is 1.0 for Rung 5 and 8.0 for Rung 6 against K = 15. The `worth_it` gate
in `select_actions` (`benefit > 0`, comment: *"Capacity is a ceiling, never a quota"*) is
correct policy in principle. But `benefit` is computed with the misspecified
`exposure_inr`, so it zeroes out precisely the merchants whose declared GMV is small and
whose realised GMV is large — the merchants who carry the loss. Rung 5 leaves 93% of the
analyst budget unspent on a problem where a review costs 250 and can avert 0.8 x a
seven-figure loss. That is not conservatism, it is M1 propagating into the gate.

This is a corollary of M1 rather than an independent mechanism, but it is separately
reportable and separately checkable.

### 1.5 Mechanism M4 — rank/calibration blindness, and why ECE does not exonerate Rung 2

The brief asks for a plain statement. Here it is, in two parts.

**Part one: the rank metrics are structurally incapable of seeing any of M1-M3.** PR-AUC
and ROC-AUC are invariant to every strictly increasing transform of the score, so they
are invariant to the entire arithmetic of `expected_costs` — the multiplication by
exposure, the review-cost subtraction, the HOLD/REVIEW branch, the `worth_it` gate. This
is the classical point in Elkan (2001) and it is made precise by Hernández-Orallo, Flach
and Ferri (2012), who show that a performance metric is only interpretable once you name
the *threshold choice method* that turns scores into decisions, and that different
threshold-choice methods make different metrics the right ones. Rakshak's threshold choice
method is "top-K by expected net benefit under a capacity constraint", and PR-AUC is not
the loss that method minimises. **So the survey's answer to "could the floor-fail be a
failure rank metrics are blind to?" is: yes, and it definitely is one — but the blind spot
that matters here is exposure misspecification, not calibration.**

**Part two: calibration is nonetheless not exonerated by the reported ECE.** Rung 2's ECE
is 0.0079 at 1.35% prevalence, which looks excellent. It is a *marginal* statistic: it
averages the calibration gap over confidence bins pooled across the whole population. The
quantity the decision layer actually needs is `P(fraud | features)` *conditional on
exposure stratum*, because the score is multiplied by exposure before it is ranked. A model
can be perfectly calibrated marginally and badly miscalibrated within every size decile,
with the errors cancelling. Perez-Lebel, Le Morvan and Varoquaux (ICLR 2023,
arXiv:2210.16315) name this the **grouping loss** and show it is exactly the part of the
proper-scoring-rule decomposition that calibration metrics cannot see: samples sharing a
confidence score but having different true posteriors. Their follow-up (arXiv:2503.18025)
converts that decomposition into **decision regret** for cost-sensitive binary decisions
and finds regimes where recalibration recovers most of the regret and regimes where the
grouping loss dominates and recalibration does nothing.

So: report ECE-within-exposure-decile, not just ECE. If it is flat, calibration is
genuinely fine and M1 is the whole story. If it slopes with size, M1 and M4 are the same
bug wearing two hats. Either way **ECE 0.0079 is not a defence, because it is not the
statistic the decision rule consumes.**

A second, cheaper calibration observation: the HOLD gate requires `score >= 0.90`. A
well-calibrated probability at 1.35% prevalence reaches 0.90 rarely. `volume_rank`'s ECE is
0.4866 — its "score" is raw GMV, wildly out of [0,1] — which is irrelevant for it because
the floor path never calls `expected_costs`. But it means any comparison of ECE across the
floor row and the rung rows is meaningless, and the ladder currently prints them in the
same column.

### 1.6 Mechanisms M5 and M6 — the two hypotheses the evidence demotes

**M5, the stationary-window hypothesis** ("`volume_rank` won because there were no onsets
in the scored window, and a static ranking is correct for a stationary regime"). Plausible
a priori, and cycle 4 fixes the window regardless because time-to-detection needs it. But
it does not survive contact with section 1.1: it predicts that `volume_rank` beats the
model on *detection*, and the model beats `volume_rank` on detection, at both precision and
recall. M5 also cannot explain Rung 6's collapse, which happened at fixed scores. **Demote
to secondary.** Concretely: M1 is stationarity-independent, so moving the onsets into the
scored window will *not* on its own close the gap, and cycle 4 should not be allowed to
claim it will.

**M6, the generator-legibility confound** ("fraud is too legible in transaction volume
alone"). Real, stated, and not fixed in cycle 4. But it argues the wrong way here: if fraud
were legible in volume, the volume-fed LightGBM should *win*, and on rank metrics it does
(0.836 vs 0.217). The rungs win the legibility contest and lose the money contest.
M6 remains a genuine limitation on external validity — it means a `volume_rank` floor is
unusually strong on *this* generator and might be weak on Razorpay's real book — but it is
not the mechanism of the floor-fail. **Keep in LIMITATIONS.md; do not treat as the cause.**

### 1.7 The single diagnostic that distinguishes them

One run, three cells, all of them calls to already-locked functions with different
arguments. **Nothing in the eval package changes; only what is passed to it changes.**

Hold Rung 2's score vector fixed. Then:

| cell | exposure passed to the decision layer | action policy | what a big move means |
|---|---|---|---|
| **A** (status quo) | `p_declared_monthly_gmv` | `select_actions` (may HOLD) | baseline = 0.4131 |
| **B** | `v_declared_ratio x p_declared_monthly_gmv` (= realised trailing-30d GMV) | `select_actions` | **M1 confirmed** |
| **C** | `p_declared_monthly_gmv` | `savings_of_ranking(..., action=REVIEW)` — action-matched to the floor | **M2 confirmed** |

Plus one scalar, computed once: Spearman correlation on fraud rows of
`log(true_loss_amount_inr)` against `log(realised pre-window GMV)` versus against
`log(p_declared_monthly_gmv)`. If the first is materially higher, M1 is proven directly
rather than inferred; the generator source says it must be, but measuring it costs nothing
and turns an argument into a number.

Then add the calibration cell: ECE and Brier computed *within exposure decile* for Rung 2.
Flat -> M4 is not live. Sloped -> M1 and M4 are the same defect.

**Cell B needs no new feature and no new data.** `v_declared_ratio` is already in the
registry (`features/tier1.py:700`), already point-in-time (trailing 30 days ending at the
scoring day), already leakage-gated, and is *defined* as trailing-30d GMV divided by
declared monthly GMV. Multiplying it back by `p_declared_monthly_gmv` recovers realised
trailing-30d GMV exactly. Two existing feature columns, one multiply.

---

## 2. The map — four literatures, and what each is actually good for here

### 2.1 Cost-sensitive learning and decision theory

**Elkan (2001)** is the foundation and its result is the one that most directly indicts
Rung 4. Elkan shows that for a two-class problem with cost matrix entries `C(i,j)`, the
optimal decision is to predict the class minimising expected cost, and that the *only*
thing costs do to a probabilistic classifier is move a threshold: the optimal threshold is
`p* = (C(1,0) - C(0,0)) / (C(1,0) - C(0,0) + C(0,1) - C(1,1))`. Everything else —
resampling, example reweighting — is a device for getting a base learner to land its
threshold in the right place. He proves the rescaling theorem that says how to change the
negative-class proportion to make a cost-insensitive learner cost-optimal.

The corollary is blunt: **if your decision layer already computes expected cost from a
probability and a cost matrix, putting cost into the training loss is redundant at best.**
Rakshak's `expected_costs` does exactly that. Rung 4 put cost in the loss and moved savings
by 0.0017 — 4 parts in 10,000 — while *destroying* precision@K (0.856 -> 0.390) and
recall@K (0.311 -> 0.140) and multiplying ECE by 5.4x (0.0080 -> 0.0433). That is not a
null result; it is the predicted result. Rung 4 damaged the probability estimate to
re-encode information the decision layer already had.

**Sheng and Ling (AAAI 2006)** sharpen it empirically: their "Thresholding" method searches
the threshold by cross-validation and beats both MetaCost and the direct theoretical
threshold, and their stated reason is that *thresholding does not require accurate
probabilities — an accurate ranking plus a well-chosen threshold is enough*. Note that this
is a claim about a **fixed, exposure-free threshold**. It does not transfer to Rakshak,
because Rakshak's operating point is not a threshold on `p`; it is a top-K on
`p x exposure`. When the operating point involves a product, ranking-only is no longer
sufficient — the scale of `p` matters relative to the scale of `exposure`. This is worth
saying explicitly in the writeup, because the "ranking is enough" folklore is what makes
PR-AUC feel like a sufficient statistic.

**Kang and Mussmann (arXiv:2605.03135)** is the most recent and the most directly on point
for Rung 4. Surveying instance-level cost evaluation, they find that *"incorporating costs
into training via loss weighting, sampling strategies, or regression yields inconsistent
benefits"*, with gains appearing only when costs are predictable from the input features.
Rakshak's cost is `loss_fraction x post-onset GMV` and `loss_fraction` is drawn per typology
in [0.35, 0.80] — i.e. a large part of the per-instance cost is *not* predictable from the
features at all. Rung 4 was attempting something the literature says does not reliably work
in exactly this regime.

**Zadrozny and Elkan (KDD 2001)** is the paper for the thing Rakshak has that most
cost-sensitive papers do not: costs that are themselves *estimated*. Their point is that
probabilities and costs are not independent, and that naive plug-in estimators of
example-dependent cost are biased. Rakshak plugs in `p_declared_monthly_gmv` as its cost
scale without any bias analysis. Section 1.2 is a special case of their warning.

**Höppner, Baesens, Verbeke and Verdonck (EJOR 297(1), 2022; arXiv:2005.02488)** is the
closest applied analogue: instance-dependent cost matrices for transfer fraud, with an
**instance-dependent threshold** derived from the cost matrix rather than a global one.
Their `cslogit`/`csboost` are R packages, so not directly importable, but the
instance-dependent-threshold idea is the right frame for the HOLD gate: Rakshak's
`hold_score_threshold = 0.90` is a *global* threshold sitting on top of an
*instance-dependent* cost structure, which is theoretically the wrong shape. **Bahnsen,
Aouada and Ottersten**'s example-dependent cost-sensitive trees and the `savings` measure
Rakshak's `savings_of_actions` is modelled on come from the same line.

### 2.2 Calibration

The relevant results, in the order they bear on this project:

- **Rank metrics are calibration-blind by construction.** AUC and PR-AUC are invariant
  under every strictly increasing transform of the score. Cube the scores; the ROC curve is
  unchanged and every expected-cost decision changes. This is not a subtlety, it is the
  definition.
- **Proper scoring rules see both, and that is their weakness as a diagnostic.** Brier and
  log-loss confound discrimination and calibration; the decomposition into calibration loss
  + grouping loss + irreducible loss is what separates them.
- **ECE is a marginal statistic and can hide the failure that matters.** Perez-Lebel et al.
  (ICLR 2023) show that a perfectly calibrated model can still assign the same confidence to
  samples with very different true posteriors — the grouping loss — and give an estimator
  for it. Their 2025 follow-up (arXiv:2503.18025) converts that decomposition into decision
  regret and finds regimes where recalibration recovers most of the regret and regimes where
  it recovers none. That paper is the single best template for how to *report* Rakshak's
  cycle-4 result, whichever way it lands.
- **Method choice under 1.35% prevalence.** Platt/sigmoid scaling assumes a specific
  parametric shape and is known to be poor when the score distribution is heavily skewed —
  which it always is under extreme imbalance. **Isotonic regression** (Zadrozny and Elkan
  2002) is non-parametric and assumes only monotonicity, but is data-hungry and will
  overfit the tail with ~134 trainable positives; it must be cross-fitted, and its step
  function will produce large ties at the top, which interacts badly with a top-K selector.
  **Beta calibration** (Kull, Silva Filho and Flach, AISTATS 2017 / EJS 11(2):5052-5080) is
  the middle path: a three-parameter family on [0,1] that contains the identity map, so it
  cannot make an already-calibrated model worse the way Platt can, and it needs far fewer
  positives than isotonic. **Venn-ABERS** (Vovk and Petej, UAI 2014, arXiv:1211.0025) gives
  a *validity guarantee* under exchangeability rather than a point estimate, at the cost of
  returning an interval.
- **Calibration under a budget** is thin as a named literature; the closest usable frame is
  Hernández-Orallo, Flach and Ferri's threshold-choice machinery (JMLR 13, 2012;
  arXiv:1112.2640), which formalises the fact that "which metric is right" is downstream of
  "how do scores become decisions". Rakshak's threshold-choice method is a rate-constrained
  top-K, and the corresponding right metric is a rate-constrained expected loss — which is
  what `savings` already is. **The metric is correct. Do not touch it.**

**The plain statement the brief asks for.** Could the whole floor-fail be a calibration
failure that rank metrics are structurally blind to? *It is definitely a failure that rank
metrics are structurally blind to. It is probably not, in the main, a calibration failure.*
The blind spot is in the *exposure* factor of `p x exposure`, not in the `p` factor. The
tell is section 1.1: a calibration story has to explain how a model that finds more frauds
and burns fewer slots finds *cheaper* frauds, and monotone recalibration of `p` cannot
change which frauds get found unless it changes the ranking, which by definition it does
not. Calibration returns as a *second-order* effect through grouping loss within exposure
strata (M4), and that is worth measuring, but it is not the headline.

### 2.3 Budgeted allocation, expected value and the knapsack framing

The brief expects this to be the most important section. It is the most important section
to get *right*, and the right answer is deflationary.

**The objective is modular, so greedy top-K is exactly optimal — no approximation.** With a
budget of K reviews per day, each merchant consuming exactly one slot, and total cost
additive over merchants, the selection problem is
`max sum_i b_i x_i  s.t. sum_i x_i <= K, x_i in {0,1}` with `b_i` the per-merchant net
benefit of intervening. That is a modular objective under a cardinality constraint, whose
exact optimum is "sort by `b_i`, take the top K with `b_i > 0`". `select_actions` already
does precisely this, `worth_it` gate and all. **This is a solved problem in the codebase
and there is no headroom in the allocation algorithm.**

Consequently:

- **Submodular selection is not applicable.** The `(1 - 1/e)` greedy guarantee of Nemhauser,
  Wolsey and Fisher (1978) is for *monotone submodular* objectives, where the marginal value
  of an item falls as the selected set grows — diversity, coverage, sensor placement. Savings
  here has no such interaction: catching merchant A does not reduce the value of catching
  merchant B. Importing submodular machinery would add a dependency and a worse guarantee
  than the exact one already in place. **Reject.**
- **Knapsack (0/1, non-uniform weights) is also not applicable to the selector**, because
  every review costs exactly one slot. It *is* applicable to the oracle, which is why
  `eval/oracle.py:57` contains a DP knapsack — the oracle is allowed non-uniform weights.
  That code is locked and correct and is not the lever.
- **What *is* the lever is the estimate of `b_i`**, which is
  `p_catch * p_hat * exposure - review_cost` and is only as good as `exposure`. The
  budgeted-allocation literature is unanimous that the hard part of a knapsack is the value
  estimate, not the solver.

The genuinely useful pieces of this literature are therefore about *what to rank by*, not
*how to select*:

- **The expected-value framework** (Provost and Fawcett, *Data Science for Business*, 2013)
  is the canonical statement that the operational quantity is `P(class) x value(class)` and
  that profit curves, not ROC curves, are the deliverable when a budget binds. It is a
  textbook, not a paper, and it is the right citation for "rank by expected value" because
  the idea predates and outlives any single paper.
- **Amount-aware operational ranking** (Hartatik, *Journal of Computing Theories and
  Applications* 4(1):202-229, 2026, DOI 10.62411/jcta.16260) is a recent, directly analogous
  study: under a budget, ranking that incorporates transaction amount concentrates
  fraudulent losses in a compact top segment, and a Budget-Weighted Capture Rate at 2/4/6%
  budgets is the reported metric. Their finding — losses concentrate in the top-ranked
  amount-aware segment — is the same phenomenon as Rakshak's `volume_rank` reaching ~0.75
  loss-weighted recall at 19.5% count recall.
- **Learning to defer under workload constraints** (Alves et al., arXiv:2403.06906,
  Feedzai) is the closest thing in the literature to Rakshak's actual problem statement:
  cost-sensitive assignment of cases to a *capacity-limited* human review queue, solved by
  constraint programming over the whole batch. Worth naming as prior art for the framing;
  the CP solver is unnecessary here for the modularity reason above.

**Why a size-correlated score wins under a budget.** The mechanism is not mysterious and
does not need a paper, but it should be stated numerically in the writeup. When exposure is
heavy-tailed over three orders of magnitude, the sum of the top 1% of losses can be a
majority of the sum of all losses — the standard fat-tail concentration result from the
actuarial literature (the largest 5% of losses being worth most of the total is the
textbook case). Savings normalises by the sum of all fraud losses, so savings *is*, to first
order, loss-weighted recall. Under a hard K, an estimator of the loss magnitude is therefore
directly an estimator of the objective, and one that is *right about magnitude and wrong
about who is a fraudster* can beat one that is *right about who is a fraudster and wrong
about magnitude* — because the magnitude term has three orders of magnitude of dynamic
range and the probability term, post-calibration at 1.35% prevalence, has far less.

That is the single sentence that explains the whole cycle-3 table.

### 2.4 Trivial-but-strong baselines, and alert stability

**On trivial baselines.** The strongest and most-cited version of this argument is Wu and
Keogh, *Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the
Illusion of Progress* (IEEE TKDE 35(3):2421-2429, 2023; arXiv:2009.13807), which shows that
popular benchmarks contain trivial anomalies and that apparent progress is largely
illusory. Kim et al. (AAAI 2022) showed the companion result that the point-adjustment
protocol awards near-perfect F1 to *random* anomaly scores. The methodological lesson
Rakshak has already internalised — mandatory floors on every savings row, `beats_all_floors`
as a hard field — is exactly the corrective these papers argue for, and cycle 3's floor-fail
verdict is the machinery *working*. **The correct read of the cycle-3 table is "the floor
regime caught something", not "the floor regime is embarrassing".** That framing belongs in
the writeup.

**On alert stability.** `volume_rank` has week-over-week alert Jaccard of exactly **1.000**,
and this is true by construction, not by luck: `_observed_volume` is cut at a single
`cutoff_day` before the window opens and never recomputed, so the ranking is frozen for the
entire scored period. It alerts on the same 15 merchants every day for ~60 days.

The literature is genuinely split, and the split is along a line Rakshak can use:

- **Stability is a virtue in the workflow literature.** SOC and AML operations research is
  dominated by alert fatigue: industry-reported false-positive rates in AML transaction
  monitoring run 90-95%, and the operational complaint is churn and duplication, not
  staleness. Investigation continuity — the same analyst holding the same case across days —
  is a real efficiency.
- **Stability of exactly 1.000 is the tell of a degenerate detector.** An alert set that
  never changes cannot, by construction, detect an *onset*. It has zero time-to-detection
  resolution: a merchant either was always alerted or never was. This is the precise sense
  in which `volume_rank` is not a detector at all — it is a static watchlist that happens to
  score well on a loss-weighted metric because the loss is concentrated on the watchlist.
- **The honest way to report it** is to publish Jaccard beside savings on every row, always,
  and to name the degenerate case explicitly: *"`volume_rank` achieves 0.6017 savings with
  week-over-week alert Jaccard 1.000 and undefined time-to-detection. It is a watchlist, not
  a detector. Any rung that approaches its savings by approaching its Jaccard has
  re-derived the watchlist."* Rakshak already computes `alert_jaccard_wow`. It should be
  promoted from a reported metric to a **gate condition** (see section 5).

### 2.5 Decision-focused learning — and the autograd wall

This is the family the brief most wants adjudicated, so the adjudication is explicit.

**Smart "Predict, then Optimize"** (Elmachtoub and Grigas, arXiv:1710.08005, *Management
Science* 68(1):9-26, 2022) makes exactly the right conceptual point: minimising prediction
error does not minimise decision regret, so train against a loss that measures decision
quality. The SPO loss is neither convex nor continuous, so they introduce **SPO+**, a convex
surrogate with a Fisher-consistency guarantee. The **decision-focused learning survey**
(Mandi, Kotary, Berden, Mulamba, Bucarey, Guns and Fioretto, JAIR 80:1623-1701, 2024, DOI
10.1613/jair.1.15320) is the canonical map of the whole family, with a benchmark suite.

**And essentially all of it is written for autograd.** The mechanism of DFL is
differentiating through (or around) an argmax: SPO+ subgradients, perturbed optimisers,
blackbox differentiation, cvxpylayers. The reference implementation, **PyEPO** (Tang and
Khalil, arXiv:2206.14234), is MIT-licensed and at 2.2.7 (July 2026) but is described as *"A
PyTorch/JAX-based End-to-End Predict-then-Optimize Tool"*, with `pytorch` and `jax` as
extras. Under the standing no-autograd decision this is **GATED, not dropped**: flag it,
keep it revisitable, and note precisely what reopening it would cost (a torch dependency, a
GPU-free training loop on 4 cores, and a re-argued architectural decision).

**But there is a much more important reason DFL is the wrong answer here, and it is not the
licence or the autograd wall.** DFL exists to solve the case where the downstream optimiser
is complex and the prediction feeds it *through a non-trivial argmax* — shortest path,
scheduling, portfolio construction — so that the map from prediction error to decision regret
is opaque. Rakshak's downstream optimiser is `sort and take the top K`. The map from
prediction to decision is a *sort*. There is no argmax to differentiate through that is not
already transparent. And critically: **SPO+ would train the model to make better decisions
under the exposure vector it is given.** It cannot fix a wrong exposure vector; it would
learn to compensate for one, which is strictly worse than fixing it. Applying DFL here would
be spending the cycle's entire budget and a rejected dependency to paper over a two-column
multiplication.

The one genuinely usable idea from this family, and it costs nothing: **SPO's evaluation
discipline.** Report *decision regret against the oracle* (`gap_to_oracle`, already on every
row) as the primary number rather than a prediction-quality metric. Rakshak already does
this. Good.

---

## 3. Candidates, ranked

`labels required` is measured against the ~134 trainable positive merchants at the day-239
boundary. "Edits locked eval package?" must be NO for every row that is not rejected.

| # | intervention | mechanism addressed | edits locked eval pkg? | needs autograd? | labels required vs 134 | licence | expected effect on savings vs the 0.6017 floor |
|---|---|---|---|---|---|---|---|
| **1** | **Realised-exposure decision policy** — a `DecisionPolicy` wrapper that rewrites `DecisionRequest.exposure_inr` to realised trailing-30d GMV (`v_declared_ratio x p_declared_monthly_gmv`) and forwards to `CapacityTopK` | **M1** (and M3 as a corollary, since the `worth_it` gate uses the same quantity) | **NO** — wraps `CapacityTopK` via the T-0118 seam; `capacity.py` and `metrics.py` byte-identical | **No** | **0** — no training, no labels, pure decision-time arithmetic | none added (numpy only) | **Large.** Replaces a self-declared magnitude with the realised one the objective is written in. Plausibly 0.55-0.75; the honest prior is "somewhere between M1-is-everything and M1-is-nothing", which is what makes it worth pre-registering |
| **2** | **Exposure-stratified recalibration** — cross-fitted isotonic or beta calibration of `p_hat` *within exposure decile*, so grouping loss inside the multiplication is reduced | **M4** | NO — recalibration happens upstream of the seam, on the score vector | No | ~134, cross-fitted; **isotonic is marginal at this count** and beta calibration is the safer parameterisation | scikit-learn 1.9.0, **BSD-3-Clause** (verified at PyPI) — `IsotonicRegression`, `CalibratedClassifierCV` already available | Moderate and *conditional on cell B*. If ECE-by-decile is flat this does nothing. Stack it on #1, do not substitute it |
| **3** | **Instance-dependent HOLD threshold** — replace `hold_score_threshold = 0.90` with the per-merchant break-even implied by the cost matrix (Höppner et al.'s instance-dependent threshold), inside the same wrapper | M2 | NO — `ActionPolicy` is a dataclass *passed into* `select_actions`; a wrapper can supply its own without editing `capacity.py` | No | 0 | none added | Small-to-moderate, high variance. HOLD is a flat-cost/scaling-benefit bet; getting the gate right matters most on the largest merchants, which is where the power is worst |
| **4** | **Action-matched floor reporting** — additionally report every rung under `savings_of_ranking(..., action=REVIEW)` so the floor comparison is like-for-like | M2 (reporting, not performance) | **NO** — `savings_of_ranking` is a locked *public* function; calling it more often is not editing it | No | 0 | none added | **Zero effect on savings by construction.** It is an honesty deliverable, not an improvement. Ship it anyway |
| **5** | Cost-sensitive resampling / class-weighting of the LightGBM objective (Elkan rescaling; Zadrozny-Langford-Abe cost-proportionate weighting) | none of M1-M4 | NO | No | 134 | LightGBM 4.7.0, **MIT** (verified at github.com/microsoft/LightGBM) | **~Zero.** This is Rung 4, already run, moved savings 0.0017 and cost 47 points of precision@K. Elkan's theorem predicts this. Do not re-run it in a new costume |
| **6** | Venn-ABERS calibration for guaranteed-valid probabilities | M4, with a validity guarantee | NO | No | 134 (needs a calibration split, which halves an already-thin positive count) | `venn-abers`, **MIT** (verified at github.com/ip200/venn-abers) | Small. Adds a dependency and an interval-valued score the top-K selector would have to collapse anyway. Not worth it at n=134 |
| **7** | Learning-to-rank on a loss-weighted relevance target (LambdaMART with amount-graded labels, per Hartatik 2026) | M1, indirectly — bakes magnitude into the ranker rather than the policy | NO | No | 134; graded relevance needs the *magnitude* per positive, which is fine (exposure is a feature, not a label) | LightGBM 4.7.0 MIT — `lambdarank` objective is built in | Moderate, but it is #1's job done in a less direct and less falsifiable way. If #1 fails, this is the natural second attempt |
| **8** | **SPO+ / decision-focused end-to-end training** | claims all of them; addresses none of the actual ones | NO | **YES — GATED** | 134 | PyEPO 2.2.7, MIT (verified at PyPI); requires `torch` or `jax` | **GATED and not recommended even if ungated.** The downstream optimiser is a sort; there is no opaque argmax to differentiate through, and SPO+ would learn to compensate for a wrong exposure rather than fix it |
| **9** | Constraint-programming batch assignment (DeCCaF-style, Alves et al. 2024) | budgeted allocation | NO | No | 134 + expert-decision data Rakshak does not have | code at github.com/feedzai/deccaf, licence **UNVERIFIED** | **~Zero.** The objective is modular; exact greedy already achieves the optimum. A CP solver cannot beat an exact algorithm |
| **10** | Submodular / diversity-aware selection under cardinality | budgeted allocation | NO | No | 0 | n/a | **Negative.** Replaces an exact optimum with a `(1 - 1/e)` approximation for an interaction structure that does not exist here |
| **11** | Change the savings metric so the comparison is fair | — | **YES** | — | — | — | **Rejected on sight.** Byte-identical `eval_module_sha256` between cycles 3 and 4 is a *deliverable*. See section 3.1 |

### 3.1 The recommendation this survey explicitly declines to make

Sections 1.2 and 1.3 identify two respects in which the cycle-3 floor comparison is not
apples-to-apples: the floor is handed a better exposure estimate than the decision layer,
and the floor is scored under a cheaper action policy. A reviewer will feel the pull toward
"so fix the metric".

**Do not.** The brief is right and the reason is worth stating in the writeup rather than
only in this file. `eval_module_sha256` staying byte-identical across cycles 3 and 4 is what
makes the sentence *"we only moved the data"* checkable by someone who does not trust us.
A metric that gets adjusted when it delivers an unwelcome verdict is not a metric. Both
observations are handled correctly as **reporting** (candidate #4) and as **policy**
(candidate #1) — neither requires a byte of `metrics.py` or `capacity.py` to change, and the
fact that they do not is itself evidence the harness was designed properly.

---

## 4. First place, unambiguous

### The realised-exposure decision policy. Rung 8, `realised_exposure(capacity_topk)`.

**Why it wins, in one paragraph.** Every other candidate improves the estimate of `p` or the
machinery that consumes `p x exposure`. This one fixes `exposure`, and `exposure` is the term
carrying three orders of magnitude of dynamic range in an objective that is, to first order,
loss-weighted recall. The generator writes the objective in realised captured GMV
(`engine.py:536`); the floor that beats us ranks on realised captured GMV
(`cli.py:_observed_volume`); the decision layer ranks on a static self-declaration
(`cli.py:920`). The gap between the model and the floor is therefore not a gap in modelling
skill — the model already wins precision@K 0.864 to 0.571 and recall@K 0.314 to 0.195 — it is
a gap in which variable the money is counted in. The intervention requires no training, no
labels, no new feature, no new dependency, no autograd, and touches no locked file; it is a
multiplication of two columns already in the feature registry, injected through a seam built
for exactly this in T-0118. It is also *maximally falsifiable*: if savings does not move, M1
is dead, the cycle has bought a real negative result, and candidates #2 and #7 are next in a
clearly defined order. No other candidate on the list is cheap enough to be worth running
*and* informative enough to be worth reporting if it fails.

**Concrete sketch — enough to implement without reading a paper.**

New file, `src/rakshak/models/rung8_exposure.py`. Nothing else moves. The shape is copied
from `rung6_conformal.ConformalHold`, which already establishes the wrapper precedent.

```
Class RealisedExposure, frozen dataclass, two fields:
    inner: DecisionPolicy          # normally DEFAULT_DECISION
    realised_gmv: np.ndarray       # one entry per scored row, aligned to request.score

  name -> "realised_exposure(" + inner.name + ")"

  decide(request: DecisionRequest) -> np.ndarray:
      # Rebuild the request with the exposure the objective is denominated in.
      # dataclasses.replace on the frozen DecisionRequest; every other field is
      # forwarded untouched, so K, CostParams and the HOLD policy are unchanged.
      swapped = replace(request, exposure_inr = self.realised_gmv)
      return self.inner.decide(swapped)
```

`realised_gmv` is assembled in the runner (`src/rakshak/score_rung8.py`, sitting at the same
composition-root level as `score_rung6.py`) as, elementwise,

```
realised_gmv = features["v_declared_ratio"] * features["p_declared_monthly_gmv"]
```

Both columns are already in the registry and already leakage-gated; `v_declared_ratio` is
*defined* (`features/tier1.py:700-720`) as trailing-30d captured GMV divided by declared
monthly GMV, so the product is trailing-30d captured GMV exactly. It is point-in-time by
construction — the trailing window ends at the scoring day — and it never touches `Truth`,
so the Prime Directive 3 AST gate over `models/` is satisfied and the
`_observed_volume`/`Truth.volume` quarantine is not breached. Guard the zero case: where
`p_declared_monthly_gmv <= 0` the ratio feature is defined to return 0.0, so fall back to
`p_declared_monthly_gmv` there rather than emitting a zero exposure that the `worth_it` gate
would silently drop.

**Three things the implementer must get right.**

1. **Report it under its own name and never merge its rows with the default's.**
   `sweep_cost_asymmetry` already takes `decision: DecisionPolicy` and defaults to
   `DEFAULT_DECISION`, and its docstring already says a rung supplying its own policy is
   *"reported under its `name`, never merged with the default's rows"*. Rungs 0-4 keep the
   numbers they have. This is the pre-registration §3 no-rescoring rule, and it is already
   enforced by the seam's design.
2. **Flag one contract wrinkle honestly.** `DecisionPolicy`'s docstring says a wrapper *"may
   only ever soften an action (HOLD -> REVIEW -> PASS)"*, and that clause exists because the
   rung-6 wrapper post-processes the action array, where promotion could breach K. A wrapper
   that rewrites the *request* and forwards is a different shape: it cannot breach K, because
   it delegates to the same `top_k_by_day` selection it always did, but it can promote. The
   invariant that actually matters — at most K non-PASS actions per day — holds by
   construction. Write that in the docstring and add it to the existing
   `tests/unit/test_capacity_seam.py` assertions rather than quietly violating a documented
   contract.
3. **Run the three-cell diagnostic (section 1.7) in the same run**, so the artifact records
   *why* the number moved, not just that it did. Cell C in particular costs one extra call to
   an already-locked function.

Expected cost: one file of roughly 60 lines, one runner, two tests, no dependency change, no
retraining. Well inside a cycle. p99 scoring latency is unaffected — the wrapper adds one
elementwise multiply over the row array.

---

## 5. The pre-registered adoption gate

Fixed now, before the code exists. Rung 8 is adopted if and only if **all four** clauses hold
on VALIDATION at K = 15/day.

**G1 — Beat the floor by a real margin.**
`savings(rung8) >= 0.7017`, i.e. **at least +0.10 absolute over the `volume_rank` floor of
0.6017.**

*Defence of 0.10.* Three independent arguments converge near it. (i) *Headroom.* The oracle
is 0.9087 and the floor is 0.6017, so the whole remaining prize is 0.307. A margin of 0.10
claims about a third of the available headroom — substantial enough to be worth a cycle,
modest enough to be achievable by an arithmetic fix. (ii) *Noise.* With ~41 fraud merchants
in the window and a heavy-tailed loss, a single large-loss merchant moving in or out of the
top-K can shift savings by several points; the effective sample size for a *loss-weighted*
metric is the number of large-loss frauds, plausibly 5-10, not 41. A margin below ~0.05 is
not distinguishable from a reshuffle. 0.10 is roughly 2x that, which is the smallest margin
this design can defend. (iii) *Asymmetry of the claim.* The claim being made is strong —
"the decision layer, not the ranker, is where the money was lost" — and a strong claim
should carry a margin that a hostile reviewer cannot attribute to luck.

**G2 — Stable across the cost sweep.** G1 holds at **at least 4 of the 5**
`ASYMMETRY_RATIOS` (0.01, 0.1, 1.0, 10.0, 100.0). A ranking that flips across the sweep is
itself a finding, and `capacity.py`'s own comment says so; a win at one guessed ratio is not
a win.

**G3 — Stable across seeds.** G1 holds at **at least 4 of the 5** locked seeds
(42, 43, 44, 45, 46). The current ladder artifact is `n_seeds: 1`. **This clause is not
optional and is the single largest threat to the result**, because savings is exactly the
metric a heavy tail destabilises. If the five-seed run cannot be afforded, the honest report
is a single-seed result *labelled as such*, and G1 is not satisfied.

**G4 — Anti-degeneracy, two parts. Both must hold.**
- `alert_jaccard_wow < 0.95` — the policy must not collapse into the frozen watchlist it is
  trying to beat. `volume_rank` sits at exactly 1.000. Rung 2 sits at 0.2875. A Rung 8 that
  reaches 0.70 savings at Jaccard 0.99 has re-derived `volume_rank` with extra steps and must
  be reported as such, not as a win.
- `alerts_per_day >= 0.9 * K` — no silent capacity under-spend. Rung 5 (1.0/15) and Rung 6
  (8.0/15) both left budget on the table and both cratered. Capacity is a ceiling, not a
  quota, but a policy declining half the budget is making an implicit claim that must be
  visible.

**Reported alongside the gate, not gating (measurements, not thresholds, because the power
is not there to gate on them):** `precision_at_k`, `recall_at_k`, `gap_to_oracle`, `ece`
**and ECE-within-exposure-decile**, `ttd_median_days`, `detection_rate_d7/d14/d30`, and the
three diagnostic cells A/B/C. The brief is explicit that the latency family has ~14 evaluable
merchants at ±13 pp standard error; **no TTD or detection-rate number may appear in the
gate**, and none does.

**What a failure means, pre-committed.** If G1 fails but cell B moved savings materially
(say by more than +0.05), M1 is real but partial, and candidate #2 (exposure-stratified
recalibration) stacks on top for cycle 5. If cell B did not move savings at all, M1 is dead,
the cycle has bought a genuine negative result about a widely-believed hypothesis, and the
honest cycle-4 report is section 8's contrarian view.

---

## 6. ADR stub

**ADR-C4-03 — Adopt a realised-exposure decision policy as Rung 8; decline decision-focused
learning and decline any change to the savings metric.**

**Status.** Proposed, pending the cycle-4 pre-registration.

**Context.** On cycle-3 validation every learned rung is FLOOR-FAIL on savings against
`volume_rank` (0.6017). The natural reading — that the ranker is inadequate — is refuted by
the ladder artifact itself: at the same K, Rung 2 achieves precision@K 0.864 against the
floor's 0.571 and recall@K 0.314 against 0.195, and still loses 0.19 of savings. Source
reading locates the cause. `true_loss_amount_inr = loss_fraction x post-onset realised
captured GMV` (`generator/engine.py:536`); the `volume_rank` floor ranks on realised captured
GMV (`cli.py:_observed_volume`); the decision layer is handed `p_declared_monthly_gmv`, a
static self-declaration, as its `exposure_inr` (`cli.py:920`). The locked decision layer
already performs correct expected-value ranking (`benefit = p_catch * p * exposure -
review_cost`) — against the wrong exposure variable. A second, smaller confound: the floors
are scored REVIEW-only at 250 INR per error while the rungs may HOLD at 8,250 INR per error,
so the floor comparison is not action-matched.

**Decision.** Add Rung 8: a `DecisionPolicy` wrapper (`realised_exposure(capacity_topk)`)
that rewrites `DecisionRequest.exposure_inr` to realised trailing-30d captured GMV, computed
as the product of two existing registry features (`v_declared_ratio x
p_declared_monthly_gmv`), and forwards to `CapacityTopK`. Reach the capacity layer only
through the T-0118 seam. Change no locked eval module. Additionally report every rung under
the action-matched `savings_of_ranking(..., action=REVIEW)` so the floor comparison is
like-for-like. Pre-register the four-clause gate in section 5 before writing the code.

**Consequences.**
- *Positive.* Zero new dependencies, zero autograd, zero new features, zero training, zero
  additional labels against a budget of 134 positives. `eval_module_sha256` stays
  byte-identical, so *"we only moved the data"* remains verifiable. The intervention is
  maximally falsifiable: one number decides it, and its failure is informative.
- *Negative.* The policy will move Rung 8's alert set toward `volume_rank`'s, so
  `alert_jaccard_wow` will rise and time-to-detection may degrade. G4 exists to catch the
  degenerate limit, but a partial move toward the watchlist is a real cost that must be
  reported, not explained away.
- *Negative.* A request-rewriting wrapper honours the K bound by construction but not the
  `DecisionPolicy` docstring's "may only ever soften" clause. The docstring and the seam test
  must be updated to distinguish request-transform wrappers from action-transform wrappers.
- *Negative.* If G1 passes, the honest headline is "the decision layer was misconfigured",
  which is a less flattering finding than "we built a better detector" and must be written up
  that way regardless.

**Alternatives rejected.**
- *Decision-focused learning / SPO+ (PyEPO, MIT, 2.2.7).* Requires torch or jax — GATED under
  the standing no-autograd decision, flagged rather than dropped. Rejected on merit
  independently of the gate: the downstream optimiser is a sort, so there is no opaque argmax
  to differentiate through, and SPO+ would train the model to compensate for a wrong exposure
  vector rather than fix it.
- *More cost in the training loss.* This is Rung 4. It moved savings by 0.0017 while halving
  precision@K. Elkan (2001) predicts it: for a probabilistic model with an explicit
  expected-cost decision rule, costs act only through the operating point.
- *Submodular or knapsack selection.* The objective is modular under a cardinality
  constraint; exact greedy top-K is already the optimum. A `(1 - 1/e)` guarantee would be a
  downgrade.
- *Changing the savings metric so the floor comparison is fair.* Rejected on sight. Both
  unfairnesses are addressable as reporting and as policy. A hash-locked metric that gets
  adjusted when its verdict is unwelcome is not a metric.
- *Recalibration alone (isotonic / beta / Venn-ABERS).* Not rejected — deferred. It addresses
  M4, which the evidence puts second, and it is a natural stack on top of Rung 8 once the
  ECE-by-exposure-decile diagnostic says whether there is anything there.

---

## 7. Where the literature is thin

Named honestly, because a pre-registration that overstates its evidential base is worse than
one that admits gaps.

1. **"Calibration under a budget" is not a literature.** There is a large calibration
   literature and a large budgeted-selection literature and almost nothing joining them. The
   specific question — *given that only the top K scores will ever be acted on, where should
   calibration effort be spent, and is tail calibration the only calibration that matters?* —
   has no canonical treatment I could find. Rakshak is doing something the field has not
   written down.
2. **Calibration of a product `p x exposure` is unaddressed.** Every calibration result I
   found concerns `P(y|x)` in isolation. The decision-relevant object here is a product with
   a heavy-tailed, *separately estimated* second factor, and the propagation of calibration
   error through that product — especially when the two factors are correlated, which they
   are — is not covered. Zadrozny and Elkan (2001) come closest by warning that costs and
   probabilities are not independent, but they do not analyse the heavy-tailed case.
3. **Alert stability has no metric standard.** Week-over-week Jaccard is a reasonable
   invention, but the SOC/AML literature I found is overwhelmingly practitioner-facing
   (vendor blogs, industry false-positive-rate surveys) rather than methodological. There is
   no accepted answer to "what alert churn *should* a healthy detector have", so Rakshak's G4
   threshold of 0.95 is an engineering judgement, not a literature-backed number, and should
   be labelled as such.
4. **Instance-dependent thresholds under a capacity constraint.** Höppner et al. (2022)
   derive an instance-dependent threshold; the budgeted-selection literature derives a
   capacity rule. Nobody I found does both simultaneously, which is precisely Rakshak's HOLD
   gate (a global 0.90 threshold on top of an instance-dependent cost structure under a hard
   K). Candidate #3 is therefore an *extrapolation*, and its expected effect in the table is
   a guess.
5. **Statistical power for loss-weighted metrics.** Confidence intervals for a
   heavy-tail-weighted recall at small n is a hard problem the ML evaluation literature
   largely ignores; the actuarial literature has the tools (extreme value theory, tail index
   estimation) but not the framing. Rakshak's savings numbers deserve a bootstrap interval
   and currently do not have one. **This is arguably a more valuable cycle-4 deliverable than
   any new rung**, and it is cheap.
6. **Cost-asymmetry ratios in Indian payments.** The three-orders-of-magnitude gap between
   the measured 47.5 / 13.1 / 61,368 and the literature band of 400-600 is unexplained and
   nothing I searched resolves it. The five-point sweep is the right response to an
   unresolved parameter, and it should keep being described that way rather than as
   robustness theatre.

---

## 8. Contrarian view — the case against everything above

Taken seriously, in three escalating forms.

### 8.1 "You have found a bug, not a research result."

The strongest objection. If M1 is right, cycle 4's headline is *"we passed the wrong column
into a function"*, and the correct write-up is a one-paragraph defect note, not a Rung 8, not
a survey, and certainly not a pre-registered gate with four clauses. Dressing a configuration
fix as a research contribution is exactly the illusion-of-progress failure Wu and Keogh
document. A reviewer who spots that `p_declared_monthly_gmv` versus realised GMV is a
two-line change will not be impressed by the ceremony around it.

**Partial concession.** The ceremony *is* disproportionate to the fix. The defence is that
the pre-registration exists to stop the fix being oversold if it works and quietly buried if
it does not, and the survey exists because three competing mechanisms were live and only
source reading separated them. But the writeup should lead with "the decision layer was
denominated in the wrong currency", not with "we adopted a novel cost-aware decision policy".
If the fix works, say it was a bug. If it does not, section 8.3 applies.

### 8.2 "Just deploy the size ranking." — the strongest version

Put at full strength, because the weak version is easy to dismiss.

`volume_rank` costs nothing to build, nothing to maintain, nothing to serve, has no training
data requirement, no model risk, no drift, no retraining schedule, and no explainability
problem — an analyst can be told *"these are our fifteen biggest merchants"* and will
understand it immediately. It has zero p99 latency. It cannot be attacked by an adversary who
cannot grow their own GMV, which is the most expensive attack there is. It delivers 0.6017
savings against an oracle ceiling of 0.9087 — **66% of perfect information, from a
`SELECT ... ORDER BY sum(amount) DESC LIMIT 15`.** Its week-over-week Jaccard of 1.000, which
this survey calls a degeneracy, is from an operations standpoint a *feature*: the same
analyst holds the same merchant relationship for months, builds context, and never suffers
alert churn. In an industry where AML false-positive rates run 90-95%, a stable 57%-precision
watchlist is not embarrassing, it is enviable.

Against that, Rung 2 requires 49 features, a feature store with per-merchant state, a
training pipeline, a calibration pipeline, a leakage gate, an AST gate, a drift monitor, and
134 labelled positives that took a generator to manufacture — and delivers **0.4131**.

**The honest counter, and it is narrow.** `volume_rank` has *undefined* time-to-detection. It
cannot detect an onset, because its alert set does not change; a merchant who turns fraudulent
on day 250 is either already on the list (detected at t=0, meaninglessly) or will never be on
it. Rakshak's stated purpose is detecting drift into fraud, and a frozen watchlist does not do
that at any price. It is also unfalsifiable as a *detector*: it would score 0.6017 on a window
containing no fraud at all, because it is measuring where the money is, not where the crime is.
And it is uninsurable against a shift in the loss distribution — the moment `loss_fraction`
decorrelates from size, it goes to zero with no warning.

But note what that counter concedes: **the case for a learned model here rests on
time-to-detection, which is the metric the brief says is power-starved at ~14 evaluable
merchants and ±13 pp standard error.** The strongest argument for the whole project rests on
the weakest-powered number in the harness. That should be stated in the writeup, out loud.

### 8.3 "The honest cycle-4 result is that `volume_rank` wins on this generator."

The possibility the brief explicitly raises, and it deserves to be planned for rather than
argued against.

Suppose Rung 8 lands at 0.62 — above the floor, below G1. Suppose the exposure swap moves
savings by +0.10 and lands at 0.51, still under the floor. Suppose cells A/B/C all move a
little and none of them dominates. Every one of those is a live outcome.

In any of them, the correct report is: **on this generator, with this cost matrix, at this K,
a static size ranking is a very strong policy, and the learned ladder does not beat it on
loss-weighted savings.** That is publishable, useful, and considerably more honest than most
of what the fraud-detection literature reports. It is the same species of result as Wu and
Keogh's, and Kim et al.'s near-perfect F1 from random scores — a demonstration that the
evaluation regime has teeth. Rakshak has the rarer thing: a floor regime that *caught* its
own ladder, and a project willing to print the number.

The generator criticism (M6) belongs in that report as a limitation with a stated direction:
`true_loss = loss_fraction x post-onset GMV` makes loss mechanically proportional to size,
which *guarantees* a size ranking is a good loss estimator. Cycle 4 does not fix that, and it
is possible that no amount of decision-layer work can beat a size ranking on a generator that
defines loss as proportional to size. **Naming that clearly is worth more than beating the
floor by 0.10.** If the survey has one recommendation beyond Rung 8, it is: whatever the
number does, write the generator's loss definition and the floor's ranking variable next to
each other in LIMITATIONS.md, so the next reader sees in ten seconds what took this survey a
source trace to find.

---

## 9. References

Verification status is per-item. **VERIFIED** = identifier or licence confirmed at
arXiv/DOI/publisher/PyPI/GitHub during this survey. **UNVERIFIED** = title, authors and venue
are confidently correct but the exact identifier was not confirmed at source this session;
treat the identifier, not the work, as provisional.

**Cost-sensitive learning and decision theory**

1. Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI-01, 973-978.
   dblp:conf/ijcai/Elkan01; author PDF at `cseweb.ucsd.edu/~elkan/rescale.pdf`;
   ACM DL 10.5555/1642194.1642224. **VERIFIED.**
2. Zadrozny, B. and Elkan, C. (2001). *Learning and Making Decisions When Costs and
   Probabilities Are Both Unknown.* KDD '01, 204-213. DOI 10.1145/502512.502540. **VERIFIED.**
3. Domingos, P. (1999). *MetaCost: A General Method for Making Classifiers Cost-Sensitive.*
   KDD '99. Title, author and venue **VERIFIED** via Semantic Scholar; DOI **UNVERIFIED.**
4. Zadrozny, B., Langford, J. and Abe, N. (2003). *Cost-Sensitive Learning by
   Cost-Proportionate Example Weighting.* ICDM 2003. Title/authors/venue **VERIFIED**;
   DOI **UNVERIFIED.**
5. Sheng, V. S. and Ling, C. X. (2006). *Thresholding for Making Classifiers Cost-Sensitive.*
   AAAI-06, vol. 1, 476-481. ACM DL 10.5555/1597538.1597615; author PDF at
   `csd.uwo.ca/~xling/papers/AAAI06a.pdf`. **VERIFIED.**
6. Höppner, S., Baesens, B., Verbeke, W. and Verdonck, T. (2022). *Instance-dependent
   cost-sensitive learning for detecting transfer fraud.* European Journal of Operational
   Research 297(1), 291-300. arXiv:2005.02488. **VERIFIED.** Code (R):
   `github.com/SebastiaanHoppner/CostSensitiveLearning` — licence **UNVERIFIED.**
7. Bahnsen, A. C., Aouada, D., Stojanovic, A. and Ottersten, B. (2016). *Feature engineering
   strategies for credit card fraud detection.* Expert Systems with Applications 51, 134-142.
   DOI 10.1016/j.eswa.2015.12.030. **VERIFIED.**
8. Bahnsen, A. C., Aouada, D. and Ottersten, B. *Ensemble of Example-Dependent Cost-Sensitive
   Decision Trees.* Title/authors **VERIFIED** via Semantic Scholar; arXiv id and venue
   **UNVERIFIED.**
9. Kang, K. and Mussmann, S. (2026). *Instance-Level Costs for Nuanced Classifier
   Evaluation.* arXiv:2605.03135. **VERIFIED** — source of the finding that cost-aware
   training (loss weighting, sampling, regression) yields inconsistent benefits.

**Calibration, proper scoring rules and grouping loss**

10. Hernández-Orallo, J., Flach, P. and Ferri, C. (2012). *A Unified View of Performance
    Metrics: Translating Threshold Choice into Expected Classification Loss.* JMLR 13,
    2813-2869. `jmlr.org/papers/v13/hernandez-orallo12a.html`. **VERIFIED.**
11. Hernández-Orallo, J., Flach, P. and Ferri, C. (2011). *Threshold Choice Methods: the
    Missing Link.* arXiv:1112.2640. **VERIFIED.**
12. Kull, M., Silva Filho, T. M. and Flach, P. (2017). *Beta calibration: a well-founded and
    easily implemented improvement on logistic calibration for binary classifiers.* AISTATS
    2017, PMLR 54:623-631. **VERIFIED.** Extended: *Beyond sigmoids: how to obtain
    well-calibrated probabilities from binary classifiers with beta calibration.* Electronic
    Journal of Statistics 11(2), 5052-5080. DOI 10.1214/17-EJS1338SI. **VERIFIED.**
13. Perez-Lebel, A., Le Morvan, M. and Varoquaux, G. (2023). *Beyond calibration: estimating
    the grouping loss of modern neural networks.* ICLR 2023. arXiv:2210.16315. Code:
    `github.com/aperezlebel/beyond_calibration`. **VERIFIED.**
14. Perez-Lebel, A., Varoquaux, G., Koyejo, S., Doutreligne, M. and Le Morvan, M. (2025).
    *Decision from Suboptimal Classifiers: Excess Risk Pre- and Post-Calibration.*
    arXiv:2503.18025. **VERIFIED** — the decision-regret decomposition into miscalibration
    and grouping loss, and the two regimes.
15. Silva Filho, T., Song, H., Perello-Nieto, M., Santos-Rodriguez, R., Kull, M. and Flach, P.
    (2023). *Classifier Calibration: A survey on how to assess and improve predicted class
    probabilities.* Machine Learning. arXiv:2112.10327. **VERIFIED (arXiv).**
16. Vovk, V. and Petej, I. (2014). *Venn-Abers Predictors.* UAI 2014. arXiv:1211.0025.
    **VERIFIED.**
17. Zadrozny, B. and Elkan, C. (2002). *Transforming Classifier Scores into Accurate
    Multiclass Probability Estimates.* KDD '02 — canonical source for isotonic calibration.
    **UNVERIFIED** at source this session.
18. Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to
    Regularized Likelihood Methods.* Advances in Large Margin Classifiers. **UNVERIFIED** at
    source this session.
19. Niculescu-Mizil, A. and Caruana, R. (2005). *Predicting Good Probabilities with
    Supervised Learning.* ICML 2005. **UNVERIFIED** at source this session.

**Budgeted allocation, expected value, selection**

20. Provost, F. and Fawcett, T. (2013). *Data Science for Business.* O'Reilly.
    ISBN 978-1-4493-6132-7. **VERIFIED** (publisher page) — the expected-value framework and
    profit curves.
21. Nemhauser, G. L., Wolsey, L. A. and Fisher, M. L. (1978). *An analysis of approximations
    for maximizing submodular set functions - I.* Mathematical Programming 14, 265-294. The
    `(1 - 1/e)` greedy guarantee under a cardinality constraint, and its tightness (with
    Nemhauser and Wolsey). Result **VERIFIED** via multiple secondary sources; DOI
    **UNVERIFIED.** *Cited here to explain why it does NOT apply.*
22. Hartatik (2026). *Beyond Binary Fraud Detection: Amount-Aware Operational Ranking for
    Transaction Risk Prioritization.* Journal of Computing Theories and Applications 4(1),
    202-229. DOI 10.62411/jcta.16260. **VERIFIED** — amount-aware ranking, Budget-Weighted
    Capture Rate, loss concentration in the top-ranked segment.
23. Alves, J. V., Leitão, D., Jesus, S., Sampaio, M. O. P., Liébana, J., Saleiro, P.,
    Figueiredo, M. A. T. and Bizarro, P. (2024). *Cost-Sensitive Learning to Defer to
    Multiple Experts with Workload Constraints.* arXiv:2403.06906. Code:
    `github.com/feedzai/deccaf` — licence **UNVERIFIED.** Paper **VERIFIED.**

**Decision-focused learning (all GATED under the no-autograd decision)**

24. Elmachtoub, A. N. and Grigas, P. (2022). *Smart "Predict, then Optimize".* Management
    Science 68(1), 9-26. arXiv:1710.08005. **VERIFIED.**
25. Mandi, J., Kotary, J., Berden, S., Mulamba, M., Bucarey, V., Guns, T. and Fioretto, F.
    (2024). *Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future
    Opportunities.* JAIR 80, 1623-1701. DOI 10.1613/jair.1.15320. Benchmark code:
    `github.com/PredOpt/predopt-benchmarks`. **VERIFIED.**
26. Tang, B. and Khalil, E. B. (2022). *PyEPO: A PyTorch-based End-to-End Predict-then-
    Optimize Library for Linear and Integer Programming.* arXiv:2206.14234. **VERIFIED.**
    PyPI `pyepo` 2.2.7 (2 July 2026), **MIT License — VERIFIED at PyPI.** Requires `torch` or
    `jax` extras — **GATED.**

**Evaluation integrity and trivial baselines**

27. Wu, R. and Keogh, E. (2023). *Current Time Series Anomaly Detection Benchmarks are Flawed
    and are Creating the Illusion of Progress.* IEEE TKDE 35(3), 2421-2429. arXiv:2009.13807.
    **VERIFIED.**
28. Kim, S., Choi, K., Choi, H.-S., Lee, B. and Yoon, S. (2022). *Towards a Rigorous
    Evaluation of Time-Series Anomaly Detection.* AAAI 2022 — the result that point-adjusted
    F1 rewards random anomaly scores. Title/authors/venue **VERIFIED** via secondary sources;
    arXiv id **UNVERIFIED.**
29. *Aligning Evaluation with Clinical Priorities: Calibration, Label Shift, and Error Costs.*
    arXiv:2506.14540 (NeurIPS 2025). **VERIFIED (arXiv + NeurIPS proceedings PDF);** author
    list **UNVERIFIED.**

**Library licences and versions (checked at source, 1 September 2026)**

| library | version | licence | checked at | verdict |
|---|---|---|---|---|
| scikit-learn | 1.9.0 (2 Jun 2026) | **BSD-3-Clause** | pypi.org/project/scikit-learn | permissive, already pinned, OK |
| LightGBM | 4.7.0 (18 Jul 2026) | **MIT** | github.com/microsoft/LightGBM (LICENSE) | permissive, already pinned, OK |
| PyEPO | 2.2.7 (2 Jul 2026) | **MIT** | pypi.org/project/pyepo | licence OK, **GATED on torch/jax** |
| venn-abers | — | **MIT** | github.com/ip200/venn-abers | licence OK; **not recommended** (n=134) |
| numpy / scipy / polars / duckdb | as pinned | BSD-3 / BSD-3 / MIT / MIT | not re-checked this session — **UNVERIFIED**, but already in the pinned set and not newly proposed | n/a |
| `costcla` (Bahnsen) | — | **UNVERIFIED** | not checked | **do not adopt without a licence check** |
| `cslogit` / `csboost` | — | **UNVERIFIED** | github.com/SebastiaanHoppner | R-only; **reject** |

**No new dependency is required by the first-place recommendation.**

---

## 10. Source references in this repository (for the implementer)

All paths relative to `v3/`.

| what | where | why it matters |
|---|---|---|
| true loss definition | `src/rakshak/generator/engine.py:536`, `generator/config.py:240` | `true_loss = loss_fraction x post-onset captured GMV` — the objective's magnitude term |
| `volume_rank` ranking variable | `src/rakshak/cli.py:304-325` (`_observed_volume`) | realised captured GMV, cut once at `cutoff_day` — hence Jaccard exactly 1.000 |
| decision-layer exposure | `src/rakshak/cli.py:688, 920, 956` | `p_declared_monthly_gmv` — the defect |
| the EV ranking that is already correct | `src/rakshak/eval/capacity.py` `expected_costs`, `select_actions` | `benefit = p_catch * p * exposure - review_cost`; LOCKED, do not touch |
| floors are REVIEW-only | `src/rakshak/eval/metrics.py:325-343, 391-394` | `savings_of_ranking(..., action=Action.REVIEW)`; the action mismatch |
| rung savings uses rung actions | `src/rakshak/eval/metrics.py:605` | `savings_of_actions(output.action[keep], ...)` — may include HOLD at 8,250 |
| the seam to wrap | `src/rakshak/eval/capacity.py` `DecisionPolicy`, `DecisionRequest`, `CapacityTopK`, `DEFAULT_DECISION` | T-0118; built for exactly this |
| the wrapper precedent | `src/rakshak/models/rung6_conformal.py:205-228` | copy this shape |
| the feature that recovers realised GMV | `src/rakshak/features/tier1.py:700-750` (`v_declared_ratio`) and `:1620` (`p_declared_monthly_gmv`) | their product is trailing-30d captured GMV, point-in-time, leakage-gated |
| the locked module list | `src/rakshak/eval/lock.py:134`, `EVAL-LOCK.json` | `splits.py`, `metrics.py`, `oracle.py`, `capacity.py`, `lock.py`; `eval_module_sha256` is the enforced hash |
| the cycle-3 numbers quoted here | `artifacts/ladder.json` (`payload.rungs`, split VALIDATION, seed 42, K=15) | note `n_seeds: 1` — see gate clause G3 |
