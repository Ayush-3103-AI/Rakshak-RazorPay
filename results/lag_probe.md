# Rakshak — the detection-lag probe

> **Sequence-layer metrics are measured on synthetic merchant streams with injected typologies; the generator is in this repo.** The decision layer is additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark derived from real bank data. Nothing on this page is measured on BAF: BAF is account-opening applications with no sequences, so it has no detection lag.

## Verdict, up front

**The -1.0 day median detection lag that `results/summary.md` reported for `gbdt` and `hmm` before T-0011 was a reporting artefact of window-start attribution. It was not early warning and it was not generator leakage. It has been corrected at source and `summary.md` no longer prints it.**

A flag was being credited to the *first* day of the seven-day window whose evidence raised it. That window contains up to six days of post-onset behaviour, so the model was given credit for days it had not yet seen. Attributing the flag to the window's **last** day — the first day on which the model could actually have fired — moves every window-based lag by exactly `WINDOW_DAYS - 1` = **+6 days**, and the negative lags disappear.

**Can the repo claim "Rakshak detects N days before the fraud starts"? No.** Under the attribution this document recommends, no model detects before onset. The honest claim is about *how soon after* onset a merchant is flagged, and about how many bad merchants are flagged at all — both of which are in the tables below. Any "detects before the fraud starts" line must be struck from the README, the video and the pitch.

**Shipped: window-END attribution, applied to `gbdt` and `hmm` together.** `rules` already reported a window-end day and was not shifted a second time. Before T-0011 `summary.md` printed both conventions in one column — that is the defect this probe found while confirming the one it was sent to confirm, and it is now fixed at source in `models/gbdt.py` and `models/hmm_score.py` rather than in the reporting layer, so no future caller can reintroduce it.

## Provenance

| Field | Value |
|---|---|
| Produced by | `python -m rakshak.eval.lag_probe --seed 42` |
| Seed | 42 |
| Splits reported | `validate` (days 180-209), `test` (days 210-269) |
| Window length | `WINDOW_DAYS` = 7 days |
| Test window | unlocked by `load_split("test", unlock_test="T-0011")` |
| Truth | `Split.transition_day` — the generator's first day in any of `config.BAD_STATES` |

## 1. What the generator does before onset — read, not measured

`generator/generate.py` was read directly rather than probed, and the claim in the T-0011 amendment holds:

- `_ramp(days, start, stop, lo, hi)` fills `out[:start]` with `lo` and only then writes `out[start:stop] = linspace(lo, hi, ..., endpoint=False)`. Every day strictly before `start` is the unmodified baseline value — and because `endpoint=False` makes `out[start] == lo`, the onset day itself is still unmodified. The glide begins the day *after* onset.
- All five injectors write only through `_ramp` or through an explicit `[onset:]` / `[mid:]` / `[glide:]` slice: `_inject_bust_out`, `_inject_laundering`, `_inject_category_drift`, `_inject_refund_collusion`, `_inject_slow_ramp`. `_inject_category_drift` multiplies the whole `amount_mult` array, but by a `_ramp` that is exactly 1.0 before onset, and assigns the whole `hour_shift` array from a `_ramp` that is exactly 0.0 before onset.
- `p.state` is likewise only ever assigned from `onset` forward, so the label and the signal start on the same day.

**No injector writes signal ahead of the labelled onset.** That is a reading of the generator, not a measurement of it; section 3 is the measurement.

## 2. Lag under both attributions

Every model in `MODEL_REGISTRY` is scored through the harness's own `_model_rng` / `_normalise`, so the **SHIPPED** column reproduces exactly what `results/summary.md` and `results/verdict.md` print. The **SUPERSEDED** column reconstructs what those files printed before T-0011, by subtracting `WINDOW_ATTRIBUTION_OFFSET_DAYS` = 6 days from the window-based models' flag days. `rules` never used that convention — it has always reported the last day of its own trailing evidence — so its superseded cell is `n/a` rather than a number it never produced. Subtracting from it would manufacture a convention and double-count the correction.

### `validate` — days 180-209

| model | flag_day means | median lag, SUPERSEDED window-START (days) | median lag, SHIPPED window-END (days) | delta (days) | flagged frac | n bad | n behind the median | distinct flag days used |
|---|---|---|---|---|---|---|---|---|
| random | none — returns no flag_day | n/a | (n/a)* | n/a* | 0.00 | 20 | 0 | 0 |
| rules | decision day (last day of its own trailing evidence) | n/a | (3.0)* | n/a* | 0.45 | 20 | 9 | 6 |
| gbdt | last day of a 7-day window | -1.0 | 5.0 | 6.0 | 0.50 | 20 | 10 | 4 |
| hmm | last day of a 7-day window | -1.0 | 5.0 | 6.0 | 0.65 | 20 | 13 | 4 |

\* `rules` is day-resolved: it evaluates trailing counters ending on the decision day **inclusive**, so its `flag_day` is already the last day of the evidence that fired it. The window-end offset does not apply to it and its shifted cell is printed in brackets to show what double-shifting it would have produced, and is not a number this repo reports anywhere. **That `rules` was already correct is itself the finding: before T-0011 `summary.md` printed its end-attributed lag in the same column as `gbdt`'s and `hmm`'s start-attributed lags, so the table compared two conventions without saying so. Both window-based scorers were moved to match it; `rules` was not touched.**

#### Quantisation — how precise can this median possibly be?

- A window-based scorer on `validate` can only flag on one of **4 distinct days**: 182, 189, 196, 203. Every `gbdt` and `hmm` flag lands on that 7-day grid, so every lag is a grid day minus an onset day.
- `rules` can flag on any of the **30 days** in the window, so its lag is not quantised the same way and its column is not directly comparable at one-day resolution.
- `rules`: the median is computed over **9 of 20** truly-bad merchants, whose flags land on **6 distinct day(s)**.
- `gbdt`: the median is computed over **10 of 20** truly-bad merchants, whose flags land on **4 distinct day(s)**.
- `hmm`: the median is computed over **13 of 20** truly-bad merchants, whose flags land on **4 distinct day(s)**.
- **A median over a handful of merchants on a small discrete grid is not a precise quantity.** It moves in whole grid steps, it has no meaningful sub-day resolution, and a single merchant changing windows can move it by 7 days. Read it as "which window", not as "how many days".

### `test` — days 210-269

| model | flag_day means | median lag, SUPERSEDED window-START (days) | median lag, SHIPPED window-END (days) | delta (days) | flagged frac | n bad | n behind the median | distinct flag days used |
|---|---|---|---|---|---|---|---|---|
| random | none — returns no flag_day | n/a | (n/a)* | n/a* | 0.00 | 20 | 0 | 0 |
| rules | decision day (last day of its own trailing evidence) | n/a | (5.0)* | n/a* | 0.65 | 20 | 13 | 11 |
| gbdt | last day of a 7-day window | 4.0 | 10.0 | 6.0 | 0.65 | 20 | 13 | 4 |
| hmm | last day of a 7-day window | 5.0 | 11.0 | 6.0 | 0.75 | 20 | 15 | 8 |

\* `rules` is day-resolved: it evaluates trailing counters ending on the decision day **inclusive**, so its `flag_day` is already the last day of the evidence that fired it. The window-end offset does not apply to it and its shifted cell is printed in brackets to show what double-shifting it would have produced, and is not a number this repo reports anywhere. **That `rules` was already correct is itself the finding: before T-0011 `summary.md` printed its end-attributed lag in the same column as `gbdt`'s and `hmm`'s start-attributed lags, so the table compared two conventions without saying so. Both window-based scorers were moved to match it; `rules` was not touched.**

#### Quantisation — how precise can this median possibly be?

- A window-based scorer on `test` can only flag on one of **8 distinct days**: 210, 217, 224, 231, 238, 245, 252, 259. Every `gbdt` and `hmm` flag lands on that 7-day grid, so every lag is a grid day minus an onset day.
- `rules` can flag on any of the **60 days** in the window, so its lag is not quantised the same way and its column is not directly comparable at one-day resolution.
- `rules`: the median is computed over **13 of 20** truly-bad merchants, whose flags land on **11 distinct day(s)**.
- `gbdt`: the median is computed over **13 of 20** truly-bad merchants, whose flags land on **4 distinct day(s)**.
- `hmm`: the median is computed over **15 of 20** truly-bad merchants, whose flags land on **8 distinct day(s)**.
- **A median over a handful of merchants on a small discrete grid is not a precise quantity.** It moves in whole grid steps, it has no meaningful sub-day resolution, and a single merchant changing windows can move it by 7 days. Read it as "which window", not as "how many days".

## 3. Pre-onset separability — the leakage check, run either way

For merchants that do go bad, do the emission features in windows lying **entirely before** the labelled onset already separate them from merchants that never go bad? If they do not, the negative lag is aliasing and is cleared. If they do, this is a leakage investigation and must be treated as one.

Features come through the existing path — `models.gbdt.build_window_matrix` with the segment map fitted on `train`, and `decision_mask` to keep only whole windows inside the split's decision window — so these are byte-identical to the vectors `gbdt` and `hmm` consume. The statistic is the rank-based common-language effect size (Mann-Whitney AUC): 0.5 is no separation, and no new dependency was added for it.

### `validate`

- Positive group: **24 windows** from **13 of 20** truly-bad merchants — every decision window ending at or before that merchant's labelled onset day.
- Negative group: **320 windows** from **80** merchants that never go bad in this split.
- Onsets are drawn from the first weeks of each split (`generator.onset_window`), so **the pre-onset windows sit at the start of the decision window while the control windows span all of it.** Any drift in a feature across the window would separate the two groups with no leakage whatsoever. The permutation below holds the positive group's window days fixed, which removes that confound exactly.

| emission feature | AUC (pre-onset vs never-bad) | \|AUC - 0.5\| | naive z |
|---|---|---|---|
| method_entropy | 0.673 | 0.173 | 2.8 |
| hour_entropy | 0.653 | 0.153 | 2.5 |
| refund_ratio | 0.369 | 0.131 | -2.1 |
| log_amount_var | 0.401 | 0.099 | -1.6 |
| sparse | 0.420 | 0.080 | -1.3 |
| repeat_payer_ratio | 0.440 | 0.060 | -1.0 |
| new_payer_ratio | 0.549 | 0.049 | 0.8 |
| chargeback_lag_days | 0.545 | 0.045 | 0.7 |
| log_amount_mean | 0.545 | 0.045 | 0.7 |
| chargeback_ratio | 0.541 | 0.041 | 0.7 |
| log_velocity | 0.536 | 0.036 | 0.6 |
| payer_jaccard_prev | 0.534 | 0.034 | 0.6 |
| payer_herfindahl | 0.473 | 0.027 | -0.4 |
| payer_entropy | 0.508 | 0.008 | 0.1 |

Largest effect **|AUC - 0.5| = 0.173**, against a merchant-clustered permutation null (n = 499) whose 95th percentile is **0.222** — **p = 0.164**.

### `test`

- Positive group: **24 windows** from **13 of 20** truly-bad merchants — every decision window ending at or before that merchant's labelled onset day.
- Negative group: **640 windows** from **80** merchants that never go bad in this split.
- Onsets are drawn from the first weeks of each split (`generator.onset_window`), so **the pre-onset windows sit at the start of the decision window while the control windows span all of it.** Any drift in a feature across the window would separate the two groups with no leakage whatsoever. The permutation below holds the positive group's window days fixed, which removes that confound exactly.

| emission feature | AUC (pre-onset vs never-bad) | \|AUC - 0.5\| | naive z |
|---|---|---|---|
| repeat_payer_ratio | 0.341 | 0.159 | -2.6 |
| new_payer_ratio | 0.636 | 0.136 | 2.3 |
| hour_entropy | 0.585 | 0.085 | 1.4 |
| refund_ratio | 0.448 | 0.052 | -0.9 |
| sparse | 0.453 | 0.047 | -0.8 |
| chargeback_ratio | 0.540 | 0.040 | 0.7 |
| method_entropy | 0.535 | 0.035 | 0.6 |
| payer_jaccard_prev | 0.533 | 0.033 | 0.6 |
| log_velocity | 0.489 | 0.011 | -0.2 |
| chargeback_lag_days | 0.491 | 0.009 | -0.2 |
| payer_herfindahl | 0.493 | 0.007 | -0.1 |
| log_amount_var | 0.505 | 0.005 | 0.1 |
| log_amount_mean | 0.495 | 0.005 | -0.1 |
| payer_entropy | 0.503 | 0.003 | 0.0 |

Largest effect **|AUC - 0.5| = 0.159**, against a merchant-clustered permutation null (n = 499) whose 95th percentile is **0.215** — **p = 0.310**.

### Result

The statistic that decides this is **the largest per-feature |AUC - 0.5|, against a merchant-clustered permutation null.** The naive *z* beside each AUC is a diagnostic and not the test: it treats windows from one merchant as independent, it ignores that the reported number is the maximum of 14 features, and it ignores that pre-onset windows sit earlier in the split than control windows do. The permutation controls all three at once — it keeps each positive merchant's window count **and its exact window days**, and permutes only which merchants are the positives.

| split | largest \|AUC - 0.5\| | null 95th pct | p |
|---|---|---|---|
| validate | 0.173 | 0.222 | 0.164 |
| test | 0.159 | 0.215 | 0.310 |

**Pre-onset windows are NOT separable from never-bad merchants.** The largest observed effect (0.173) sits inside what merchant-level relabelling produces by chance at these sample sizes, on both splits. The generator is not telegraphing typologies before the state path records them, which is what reading `_ramp` predicted. **The -1.0 is aliasing, and `summary.md`'s existing numbers are cleared of the leakage suspicion.**

Two further signs it is noise rather than signal, both visible in the tables above. The features that come closest to separating are **not the same features on the two splits and not in the same direction** — a generator leak would show the same mechanism twice, since it is the same generator. And the positive group is only 24 windows drawn from a dozen merchants, because onsets are placed early in each split; there is very little pre-onset material to look at, and this document does not pretend otherwise.

## 4. What this changes

- **Window-END attribution is shipped for `gbdt` and `hmm`.** The corrected medians are the SHIPPED column of the tables in section 2, and they are what `summary.md` and `verdict.md` now print.
- **Both models moved together.** Moving one alone would have made the two rows incomparable, which is precisely the defect this probe found in the `rules` column.
- **Strike any "detects before the fraud starts" claim.** It was an artefact of crediting a model with a window it had not finished observing.
- **Report the quantisation with the median, every time.** On a 7-day grid over a handful of flagged merchants the median is a coarse ordinal, not a measured duration.
- **The leakage suspicion is retired, with a measurement behind it** rather than only a reading of the generator source.

