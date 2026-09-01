# Pre-registration: cycle 3 adds three metric names, and nothing else

**Written and committed BEFORE any Rung 5, 6, 7 or 8 code exists, before any of the three
metrics named below is implemented, and before `EVAL-LOCK-CYCLE3.json` is written.** Its
whole value is that it predates all of them. Check its position in the git history: if it
does not sit before the commit that adds `false_hold_coverage` to `src/rakshak/eval/metrics.py`,
it is worthless and should be read as such.

**Date:** 2026-09-01 (drafted 2026-08-31, committed at the first opportunity) · **Author:** lead · **Supersedes:** nothing. Adds a third cycle lock.
**Tree at time of writing:** `687d0e7b26c653c7781d6d33f183792e46ba79c0`

---

## 1. Why this is happening

T-0118 (GitHub #52) carries this acceptance criterion:

> `EVAL-LOCK.json`'s metric list includes all three names before #44 runs

It cannot be satisfied. #44 (T-0111, the lock ticket) is closed. Both locks that exist are
sealed:

| File | `open_count` | Contains the three names? |
|---|---|---|
| `EVAL-LOCK.json` (cycle 1) | 0 | no |
| `EVAL-LOCK-CYCLE2.json` (cycle 2) | 0 | no |

The three metrics that Rungs 6, 7 and 8 exist to be judged on —
`false_hold_coverage`, `onset_localisation_error`, `tpp_rescaled_ks` — are in neither
list. Under Prime Directive 5 a rung is adopted only against margins declared before the
run, and under the lock's own contract a metric not named before sealing cannot be claimed
afterward. So Rungs 6/7/8 currently have no admissible headline metric.

**The defect is procedural and it is checkable without any model output.** It is a
comparison of two committed JSON files against one ticket's text. At the time of writing:

- `src/rakshak/models/` contains `rung0_floors.py`, `rung1_rules.py`, `rung2_lgbm.py`,
  `rung3_cohort.py`, `rung4_cost.py` — and nothing above Rung 4.
- `grep -rn 'false_hold_coverage\|onset_localisation_error\|tpp_rescaled_ks' src/ tests/`
  returns nothing.

No number produced by any of these three metrics has ever been computed, by anyone, on any
split. There is therefore no result this pre-registration could have been written to
accommodate. **This is the distinction that matters and it is the only defence:** the
cycle-2 re-freeze had to argue that a label-arithmetic defect was derivable from the config
alone; this amendment does not even need that argument, because the thing being declared
has never been measured.

---

## 2. What changes

**Additive only. Three strings enter a metric list.**

| # | Change | Justification independent of any result |
|---|---|---|
| 1 | `false_hold_coverage` enters the metric list | Rung 6 is conformal risk control. Its claim *is* that realised false-HOLD rate respects nominal alpha per Mondrian stratum. A conformal rung scored without a coverage metric is scored on nothing it claims. |
| 2 | `onset_localisation_error` enters the metric list | Rung 7 (HSMM) claims to say *when* drift began, not merely that it did. Signed days between estimated change-point and `drift_onset_at`, as a distribution with median and IQR. |
| 3 | `tpp_rescaled_ks` enters the metric list | Rung 8 is a temporal point process. Time-rescaling KS is the standard goodness-of-fit for one; without it the rung has no admissible fit criterion. |

## 3. What explicitly does NOT change

Carried across from `EVAL-LOCK-CYCLE2.json` **unchanged**, and not renegotiated:

- **The declared adoption margins** — `>=10%` relative PR-AUC OR `>=3` days median TTD.
- **The five seeds** — 42, 43, 44, 45, 46.
- **Population geometry** — 20,000 merchants x 365 days, onsets confined to [30, 240].
- **Split boundaries** — 0-239 train / 240-299 val / 300-364 test; 60/15/25 merchant fold.
- **Capacity** — K = 50 reviews/day per 10,000 merchants, as a rate.
- **Prevalence** — the realistic ~1.5%.
- **Cost asymmetry ratios** — 0.01, 0.1, 1.0, 10.0, 100.0.
- **Every metric already in the cycle-2 list.** None is removed. None is redefined.

**No existing rung is rescored and no committed number moves.** Rungs 0-4 are judged on the
cycle-2 lock exactly as before. The three new names are inert for them.

## 4. The ordering this commits to, and why it is not the obvious one

`eval_module_sha256` is the **only enforced hash**. Implementing the three metrics edits
`src/rakshak/eval/metrics.py`, which changes that hash. So the lock cannot be written first
and the metrics added after — that would break the lock on the very next commit, which is
precisely the failure mode `EVAL-LOCK.json`'s own `enforcement_note` warns about ("a lock
that is routinely overridden is not a lock").

The committed sequence is therefore:

1. **This document.** (no code)
2. **T-0118** — the three metrics, the decision-policy seam in `eval/capacity.py`, and the
   explainer registration surface. Eval-side only. **No rung attached.**
3. **`EVAL-LOCK-CYCLE3.json`** — written once, hashing the eval modules as they stand after
   step 2, `open_count: 0`, superseding `EVAL-LOCK-CYCLE2.json`.
4. **Only then**, Rungs 5/6/7/8 (T-0120, T-0121, T-0123, T-0125).

If any Rung 5-8 code lands before step 3, this pre-registration has failed and the rung
must be reported as post-lock and ineligible for adoption.

## 5. What would falsify this, and the commitment

The three metrics are being declared, not promised a value. Specifically:

- If `false_hold_coverage` shows realised coverage violating nominal alpha, **that violation
  is reported** — T-0118's own acceptance criterion requires the test to report a violation
  rather than clamp it.
- If `onset_localisation_error` shows Rung 7 cannot localise onset better than the trivial
  guess, that is a negative result and Rung 7 is dropped from the scoring path.
- If `tpp_rescaled_ks` rejects, Rung 8's intensity is misspecified and is reported as such.
- If a rung fails to clear the carried adoption margin, it is **not** adopted, and no metric
  in this document may be used to argue it should be. These metrics describe rungs; they do
  not override Prime Directive 5.

Per Prime Directive 6, each such outcome goes in `LIMITATIONS.md` with its number.

## 6. Scope limit

This amendment authorises **three metric names**. It authorises no change to the generator,
no change to the splits, no change to the margins, no rescoring of any existing rung, and no
reopening of the test split. The test split still opens exactly once, in T-0116, after every
rung is final — and T-0116's GitHub dependencies were corrected on 2026-08-31 to record
T-0112, T-0113 and T-0114 as blockers, which had been missing.
