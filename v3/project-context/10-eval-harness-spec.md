<!-- HEAD
FILE:     10-eval-harness-spec.md
PHASE:    2 — SPECIFY
UPDATED:  2026-08-31
STATUS:   active — freeze target is end of Block 5
SUMMARY:  Specification for the v2 evaluation harness: three-way split (temporal +
          merchant-group + label-availability), the metric suite with mandatory floors,
          the perfect-foresight oracle, the capacity-constrained decision layer, the
          cost-asymmetry sweep, and the EVAL-LOCK protocol that makes "opened once" a
          verifiable claim rather than an assertion. Implements FR-020..FR-026.
OPEN:     analyst capacity K is [ASSUMED] at 50/day per 10k merchants (charter §10.4).
-->

# 10 — Eval Harness v2

## The claim this file has to make defensible

> "The test split was opened once, after every model was final, against a harness whose
> code and configuration were hashed before any model existed."

Everything below exists to make that sentence checkable by someone who does not trust
you. That is the actual product here. The v1 build already demonstrated that this
discipline is the project's strongest differentiator; v2 makes it mechanically enforced
rather than behaviourally maintained.

---

## 1. Splits (FR-020)

Three constraints applied simultaneously. Any one of them alone leaks.

**Temporal.** Days 0–119 train, 120–149 validation, 150–179 test. No feature at time t
uses any event with `event_time > t`.

**Merchant-group disjoint.** Merchants are hashed to folds; no `merchant_id` appears in
more than one split. Without this, a model memorises merchant identity and reports
inflated numbers on the same merchants it trained on.

**Label-availability aware.** This is the one almost everyone gets wrong. Training at
decision time `t` may use only labels with `label_available_at <= t`. Because the
chargeback delay is 45–120 days, a merchant that turned fraudulent on day 100 is *not
labelled* during a day-120 training run. That is the real operating condition, and a
harness that hands the model day-100 labels on day 120 is measuring a system that cannot
exist.

Implement exactly once:

```python
def available_labels(as_of: datetime) -> pl.LazyFrame:
    """Labels the system is permitted to know at as_of. The ONLY way to read labels."""
```

Every training and evaluation path goes through it. `tests/unit/test_splits.py` asserts
no other module imports the label parquet directly.

**Censoring.** Merchants with `is_censored=true` are excluded from label-based metrics
and reported separately as a coverage number. Silently dropping them inflates
prevalence; silently treating them as negatives deflates recall. Report the count.

---

## 2. Metric suite

### Headline

- **PR-AUC** — the ranking metric. Reported at the declared prevalence, always, in the
  same row as the prevalence value (FR-021).

### Mandatory floors, on every row (FR-021)

`all_pass`, `all_hold`, `random_at_K`, `volume_rank`. If a rung does not beat all four
on a metric, that metric is reported as **FLOOR-FAIL** in bold. v1 discovered `random`
winning on savings at 20% prevalence; the correction is to make that discovery
automatic and unavoidable rather than to drop the metric.

### Cost

**Bahnsen-style savings**, with instance-dependent costs:

```
cost(decision, truth) =
  PASS  & fraud   →  true_loss_amount_inr
  PASS  & good    →  0
  REVIEW& fraud   →  review_cost + (1 − p_catch) × true_loss_amount_inr
  REVIEW& good    →  review_cost
  HOLD  & fraud   →  review_cost
  HOLD  & good    →  false_hold_cost_inr + review_cost
savings = 1 − total_cost / cost_of_all_pass
```

**Cost-asymmetry sweep is required, not optional.** v1 measured the asymmetry at 47.5 /
13.1 / 61,368 against a literature band of 400–600. Three orders of magnitude of spread
means the ratio cannot be assumed. `make report` produces the rung ranking at
`false_hold_cost / fraud_loss` ∈ {0.01, 0.1, 1, 10, 100}. **A ranking that is stable
across the sweep is a far stronger claim than a win at one guessed ratio** — and if the
ranking flips, that is itself the finding, and a more interesting one.

### Operational

- **precision@K / recall@K** at analyst capacity K.
- **alerts_per_day** — must be ≤ K by construction; assert it.
- **gap_to_oracle** — `(oracle_savings − rung_savings) / oracle_savings`. Report this
  rather than an unanchored absolute, so a 0.6 savings is legible as "72% of achievable"
  rather than as a number with no scale.
- **alert_jaccard_wow** (FR-024) — week-over-week Jaccard of the alert set restricted to
  non-drifting merchants. Target ≥ 0.60 (NFR-09). An alert list that churns weekly is
  unusable by an ops team no matter how good its PR-AUC, and this is the metric that
  makes that visible.

### Latency (FR-022) — the metric v1 did not have

```
TTD(m) = first as_of where action(m, as_of) ∈ {REVIEW, HOLD}  −  drift_onset_at(m)
```

Report **median TTD** over uncensored positives, and **detection rate at day 7 / 14 /
30**. Merchants never detected are right-censored — use the day-30 detection rate as the
headline rather than a mean that silently drops them.

TTD is an equal-standing win condition in charter §2 because it is the operationally
meaningful quantity: days of loss prevented. Two models with identical PR-AUC and a
5-day TTD difference are not equivalent systems.

### Per-typology recall — required output

Recall broken out by R1–R9. A single aggregate lets easy R1 hide hard R2 and R7. The
v1 slow-ramp failure is the reason this is a required column: if R2 recall is near zero
again, that must be visible on the front page of the results, not discoverable by
someone who digs.

### Calibration

**ECE** with 10 equal-mass bins, plus a reliability curve saved to `docs/figures/`. The
cost-aware decision layer takes probabilities as input, so an uncalibrated score makes
the entire cost calculation meaningless.

---

## 3. Perfect-foresight oracle

Given true labels and `true_loss_amount_inr`, for each day select the K merchants
maximising prevented loss. With uniform review cost this is a top-K selection; with the
budget expressed in analyst-hours it is a knapsack — implement the knapsack, it is
twenty lines and it generalises.

The oracle is a **ceiling, not a target**. Its purpose is to convert every result into
gap-to-oracle so that "we captured 68% of the achievable savings" replaces "our savings
score was 0.41", which means nothing to anyone.

Sanity assertion: `oracle_savings >= any_rung_savings`. If a rung beats the oracle, the
rung is leaking. This assertion has caught real bugs and should run on every eval.

---

## 4. Capacity-constrained decision layer (FR-034)

Per epoch, given calibrated scores and capacity K:

1. Compute expected cost of each action per merchant using the cost matrix in §2.
2. Rank merchants by `cost(PASS) − min(cost(REVIEW), cost(HOLD))` — the benefit of
   intervening.
3. Take the top K under the budget; assign each its own cost-minimising action.
4. Everything else gets `PASS`.

Two properties to test: alerts never exceed K, and the selection is stable under small
score perturbations (feeds `alert_jaccard_wow`).

`HOLD` is reserved for scores above a high threshold **and** an expected loss above a
configured floor — you do not freeze a merchant over ₹4,000 of expected exposure. Both
thresholds live in config.

---

## 5. Reason codes (FR-033)

For every non-`PASS` decision, take LightGBM `pred_contrib=True`, select the top 3
features by absolute contribution, and render each through the feature's
`human_template` from the registry:

> "GMV is 4.2σ above this merchant's own norm, while comparable merchants are flat."
> "Auth failure rate rose 3.1σ; 78% of attempts are under ₹10."
> "Payout requests increased from weekly to daily."

This is the audit trail. It costs nothing — `pred_contrib` is native to LightGBM and
adds microseconds — and it is strictly more actionable to a disputing merchant than "you
entered state 3." The v1 Viterbi path was the right *instinct* about explainability
attached to the wrong model; this keeps the instinct and drops the dependency.

Note the second clause of the first example: the cohort comparison is *in the
explanation*. "You went up and your peers did not" is the sentence that survives a
merchant dispute.

---

## 6. EVAL-LOCK protocol (FR-025)

`EVAL-LOCK.json`, written once by T-133 in Block 5, committed, never hand-edited.

```json
{
  "created_at": "2026-09-01T...Z",
  "scenario_config_sha256": "...",
  "eval_module_sha256": "...",
  "generator_module_sha256": "...",
  "seeds": [42, 43, 44, 45, 46],
  "split_boundaries": {"train": [0,119], "val": [120,149], "test": [150,179]},
  "metrics": ["pr_auc", "savings", "ttd_median_days", "precision_at_k", "..."],
  "declared_adoption_margins": {
    "relative_pr_auc": 0.10,
    "ttd_days": 3.0,
    "note": "from 00-charter-v2.md §2, declared before any v2 model existed"
  },
  "open_count": 0,
  "open_log": []
}
```

`make eval` behaviour:

1. Recompute all three hashes. **Mismatch → hard fail.** If the eval code changed after
   the lock, results against it are not comparable and the harness says so.
2. If `--split test` and `RAKSHAK_UNLOCK` is unset → refuse with a message naming this
   file.
3. On an authorised test run: append `{timestamp, git_sha, rungs_scored}` to `open_log`,
   increment `open_count`, write, and remind the operator to commit.

`open_count` appears in every results table and in the writeup. A committed counter
sitting at 1 is a claim anyone can verify from the git history. That verifiability is
worth more than the number it reports.

---

## 7. The G5 figure

Gate G5 (confounder null) produces the single most valuable artifact of the sprint:

**X axis** simulation day. **Y axis** alert rate. **Shaded bands** the six confounder
windows. **Two lines:** Rung 2 (raw features) and Rung 3 (cohort residuals). Run with
`prevalence = 0`, so every alert is by construction a false positive.

If the hypothesis holds, the raw line spikes inside every shaded band and the residual
line stays flat. That is the whole thesis of v2 in one image, it is legible in three
seconds, and it is the strongest candidate for the video.

If the lines are the same, charter K-1 has fired. Publish the figure anyway — a clean
falsification of a well-motivated hypothesis, on a harness frozen in advance, is a real
result and reads far better than silence.

---

## 8. Report output

`make report` → `docs/results_v2.md` + `docs/results_v2.parquet` + `docs/figures/`.

Front page, in this order:
1. Provenance header — git SHA, `eval_lock_sha`, `open_count`, prevalence, K, seeds.
2. Main table: one row per rung, all four floors, gap-to-oracle, TTD, latency.
3. Per-typology recall matrix.
4. Cost-asymmetry sweep — rung ranking at each ratio.
5. The G5 figure.
6. Ablations: cohort-residual on/off (the K-1 test), leave-one-family-out, T1 vs T1+T2,
   graph-features re-test under the corrected generator.
7. `LIMITATIONS.md` link, with every failed rung and its number.

Anything that did not work goes on this page, not in an appendix. Reporting honest
failure was assessed as a credibility feature in v1 and it remains one.
