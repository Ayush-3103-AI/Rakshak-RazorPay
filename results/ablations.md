# Rakshak — ablation table (FR-018)

> **Sequence-layer metrics are measured on synthetic merchant streams with injected typologies; the generator is in this repo.** The decision layer is additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark derived from real bank data.

Everything on this page is the synthetic split. BAF has no sequences, so no ablation of the sequence layer could have been run there (`results/baf_validation.md`).

## Provenance

| Field | Value |
|---|---|
| Produced by | `python -m rakshak.eval.ablations --seed 42` |
| Seed | 42 |
| Split reported | `test` (days 210-269), unlocked with `unlock_test="T-0011"` |
| Population | 100 merchants, 20 truly bad (20.0% prevalence) |
| Review budget K | 5 merchants (0.40 analyst-hours at 0.067 h per review) |
| Cost basis for the INR column | Cost_l = INR 555,961 on this split |
| Fits performed | 6 (3 configurations x 2 models) |

### No configuration was selected on `test`

The shipping configuration was fixed at **T-0004b on `validate`** — four latent states, the items 1+2 partially-supervised fit, the full FR-008/FR-009 emission vector, FR-007 within-merchant standardisation — and has not moved since. **These rows are a report, not a search.** Each one re-runs that frozen configuration with a single component removed, at one seed, and every row that ran is printed whether it flatters the component or not. Nothing on this page was chosen because it looked better here; if it had been, the test window would no longer be a held-out window and every number in the README would inherit the problem.

Both models refit for every ablation. The segment map is fitted on the **training** population alone and passed into the held-out build in all six fits, so the leakage guard in `eval/splits.py` holds unchanged across the variants. The variant is part of the memoisation key in `models/gbdt.py` and `models/hmm_score.py`, so a variant fit can never be served to the shipping path.

## The table

Headline metric is `savings`. **PR-AUC is printed beside every savings number** and must be read with it — see the AP-06 note below. FR-019's two vocabularies: the ML columns are PR-AUC / precision@K / Brier, the operational columns are INR saved, analyst-hours consumed and merchants held per 1000.

| component | setting | savings | d savings | PR-AUC | d PR-AUC | precision@5 | d prec | Brier | d Brier | INR saved | analyst-h | held /1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HMM (the proposal) | **on** — shipping | 0.5176 | reference | 0.3347 | reference | 0.4000 | reference | 0.4321 | reference | 287,772 | 0.27 | 150 |
| HMM (the proposal) | **off** — same pipeline, LightGBM scorer | 0.5069 | -0.0107 | 0.6523 | +0.3176 | 1.0000 | +0.6000 | 0.1453 | -0.2868 | 281,805 | 0.34 | 130 |
| graph features (FR-008) | **on** — HMM | 0.5176 | reference | 0.3347 | reference | 0.4000 | reference | 0.4321 | reference | 287,772 | 0.27 | 150 |
| graph features (FR-008) | **off** — HMM refitted | 0.4170 | -0.1006 | 0.2957 | -0.0390 | 0.4000 | 0 (no change) | 0.4177 | -0.0144 | 231,829 | 0.34 | 130 |
| graph features (FR-008) | **on** — LightGBM | 0.5069 | reference | 0.6523 | reference | 1.0000 | reference | 0.1453 | reference | 281,805 | 0.34 | 130 |
| graph features (FR-008) | **off** — LightGBM refitted | 0.5116 | +0.0047 | 0.5306 | -0.1217 | 0.8000 | -0.2000 | 0.1415 | -0.0038 | 284,422 | 0.34 | 140 |
| within-merchant standardisation (FR-007) | **on** — HMM | 0.5176 | reference | 0.3347 | reference | 0.4000 | reference | 0.4321 | reference | 287,772 | 0.27 | 150 |
| within-merchant standardisation (FR-007) | **off** — HMM refitted | 0.4858 | -0.0318 | 0.2872 | -0.0475 | 0.4000 | 0 (no change) | 0.6050 | +0.1729 | 270,089 | 0.20 | 140 |
| within-merchant standardisation (FR-007) | **on** — LightGBM | 0.5069 | reference | 0.6523 | reference | 1.0000 | reference | 0.1453 | reference | 281,805 | 0.34 | 130 |
| within-merchant standardisation (FR-007) | **off** — LightGBM refitted | 0.5068 | -0.0001 | 0.6556 | +0.0032 | 1.0000 | 0 (no change) | 0.1371 | -0.0081 | 281,774 | 0.34 | 130 |
| empirical-Bayes shrinkage (ADR-0006) | on / off | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** |
| NSGA-II vs. grid search (ADR-0004) | frontier vs. grid | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** |

`d X` is the row's value minus its reference row's. For `Brier`, lower is better, so a **negative** delta is an improvement; for the other three, higher is better. `INR saved` is `savings x Cost_l` and therefore carries every caveat `savings` carries. `held /1000` is FR-019's operational quantity: merchants placed on HOLD per 1000 under watch.

## Context rows — how much of `savings` is the cost matrix

Not ablations. Printed because a savings delta cannot be read without knowing what a model that ranks at chance earns from the same cost matrix. **No verdict is rendered here**; K2 is rendered elsewhere in T-0011.

| model | savings | PR-AUC | precision@5 | Brier | INR saved | held /1000 |
|---|---|---|---|---|---|---|
| random | 0.5365 | 0.2449 | 0.2000 | 0.3069 | 298,248 | 140 |
| rules | 0.4889 | 0.5547 | 1.0000 | 0.1358 | 271,836 | 120 |

**AP-06, as a measurement, and it is worse on `test` than it was on `validate`.** `random` scores **0.5365** savings here while ranking at PR-AUC 0.2449 — chance, at this split's 20% prevalence. The shipping configuration scores 0.5176 and `rules` scores 0.4889. **Savings net of the random floor is -0.0188 for the shipping configuration and -0.0475 for `rules`** — on this split a uniform random score, spent through the same Bayes-Minimum-Risk policy, out-saves every fitted model in the table above.

Nothing about any model produced that; the cost matrix did. When `c_fp` is small relative to `L_m`, a merchant-specific threshold puts most merchants on the correct side of the decision whatever the score is, so `savings` measures the cost arithmetic far more than it measures detection. On `validate` (T-0007b) the same effect left `random` 0.0051 behind `rules`; on `test` it puts `random` ahead of everything. **This is not a result about the models and it must not be read as one — it is the strongest evidence in the repo that an absolute savings figure is not a claim about detection.** Whoever renders K2 must report savings relative to this floor, and must state that on `test` that relative figure is negative for every model here. On the ranking metrics — PR-AUC, precision@K, Brier — `random` sits where it belongs, at the bottom.

## What the table actually says

**1. Removing the HMM costs nothing on savings and buys a great deal on every ranking metric.** Against the shipping configuration, the LightGBM row moves savings by -0.0107 (degrades), PR-AUC by +0.3176 (improves), precision@5 by +0.6000 (improves) and Brier by -0.2868 (improves). The incumbent ranks and calibrates far better on this window. That is A-005's question arriving with an answer, and it is reported here rather than tuned: **the verdict clause of K2 is rendered elsewhere in T-0011, but nothing in this row supports a claim that the sequence layer earns its place on ranking quality.** What the HMM has is the Viterbi path an analyst can read, and that is an explainability argument, not a metric argument.

**2. The graph scalars are not decoration — for either model.** Dropping the four FR-008 columns moves the HMM's PR-AUC by -0.0390 (degrades) and its savings by -0.1006 (degrades); it moves LightGBM's PR-AUC by -0.1217 (degrades) and its precision@5 by -0.2000 (degrades). **ADR-0002's substitution of four CPU scalars for a graph neural network is carrying real signal on this generator.** It does not follow that a GNN was unnecessary, and it does not follow that the scalars would carry the same signal on a real payer graph — the generator wrote the payer process these features read.

**3. Within-merchant standardisation is load-bearing for the HMM and close to decoration for LightGBM.** Turning FR-007 off moves the HMM's PR-AUC by -0.0475 (degrades) and its Brier by +0.1729 (degrades); it moves LightGBM's savings by -0.0001 (degrades), its PR-AUC by +0.0032 (improves) and its precision@5 by +0.0000 (does not move). **This is FR-018's own test landing on a component: for LightGBM, on this split, at this seed, within-merchant standardisation is very nearly a component whose removal changes no number.** The asymmetry is mechanical and was predictable — see the standardisation section below — but it is printed rather than smoothed over, and it means the `gbdt` baseline in `results/summary.md` would be about as strong without P-02 as with it. P-02 earns its place through the pooled Gaussian HMM, not through the incumbent.

## What each row means

### HMM on / off — the construction, stated plainly

**"HMM off" here is the `gbdt` path: the identical feature pipeline, the identical segment map, the identical decision-window mask and the identical BMR policy, scored by LightGBM over windowed aggregates instead of by a filtered latent-state posterior.** `models/hmm_score.py` takes its design matrix from `models/gbdt.py::build_window_matrix` precisely so the two see byte-identical inputs.

**That is not the same experiment as switching the sequence layer off in place**, and the difference matters enough to state rather than bury. Removing the HMM leaves no scorer at all; something has to score the merchants. What this row measures is therefore *HMM versus the incumbent discriminative model on the same features*, which is A-005's question, not *the marginal value of sequence structure*. The cleaner experiment — a sequence-aware model that is not this HMM — was BOCPD, and BOCPD was cut. See the note below.

### Graph features on / off — this is an ADR-0002 result

The four scalars removed are `payer_entropy`, `repeat_payer_ratio`, `payer_jaccard_prev` and `payer_herfindahl` (`features/windows.py::BASE_FEATURES`). **They exist because ADR-0002 rejected a graph neural network** — GPU-bound, circular to evaluate on a synthetic graph, infeasible solo in four days — and chose these CPU-computable scalars as the stand-in. The emission vector goes from 14 features to 10 and both models are refitted.

So this row is not a feature-selection curiosity. It is the only evidence in the repo about whether ADR-0002's substitution bought anything, and the delta must be read as an ADR-0002 result in either direction: a delta of ~0 would have meant **the GNN stand-in is decoration on this generator**, and a large delta means the substitution carries signal *that the generator put in the payer process*. Neither reading licenses a claim about what a GNN would have done on real data — this repo has no evidence about that either way.

### Within-merchant standardisation on / off — cross-merchant comparability

`features/standardise.py::standardise_panel` expresses every emission as a deviation from *that merchant's own* burn-in norm (FR-007, P-02). With it off, the raw per-window aggregates go straight into both models with no location, no scale, no segment shrinkage and no Z_CLIP winsoriser.

`features/windows.py`'s module docstring says what that costs: *"Nothing here is comparable across merchants yet: a grocer's velocity and a jeweller's velocity live three orders of magnitude apart."* One pooled Gaussian HMM over raw features is therefore modelling merchant identity — size, category, ticket scale — rather than merchant drift, and it is exactly the 2008-era cardholder-HMM failure mode where the jeweller is flagged for being a jeweller. LightGBM is far less exposed: it splits on thresholds per feature and can carve out scale bands on its own, so the two models are **not** expected to lose the same amount here, and a small LightGBM delta is not evidence that standardisation is decoration for the HMM.

## Rows that were never measured

FR-018 names five components. Two of them cannot be measured because the tickets that would have built them were cut in the 2026-08-28 re-plan. **They are printed as `not measured`, never as zero and never omitted** — a blank row and a zero row make opposite claims, and only one of them is true here.

| row | why it is absent | what is undischarged |
|---|---|---|
| empirical-Bayes shrinkage on / off | **T-0008 was cut.** ADR-0006 records the decision and its status line says it was never built. | No recalibration happens anywhere in this repo, and the BMR policy in `decision/policy.py` consumes each model's raw score **as if it were a calibrated posterior**. Under a rank-only policy miscalibration would only cost the Brier column; under BMR it moves the argmin. Every `savings` number on this page inherits that. |
| NSGA-II vs. grid search | **T-0009 was cut.** ADR-0004 chose NSGA-II over NSGA-III and made the grid-search comparison a *mandatory* ablation. | No Pareto frontier exists, so the obligation ADR-0004 wrote down is **undischarged**. `pymoo` is still declared as a dependency in `pyproject.toml` for work that did not happen; it should be removed or explicitly justified before freeze. |

### No sequence-aware baseline other than the HMM was measured

**T-0010 (BOCPD, Adams & MacKay 2007) was cut in the same re-plan.** It was the only planned model that was sequence-aware without being this HMM, so with it gone the question **"is any margin here from sequence modelling, or from the HMM specifically?"** has no experiment behind it in this repo.

**That question is left open, and it is stated as open.** The HMM-on/off row above compares a sequence model against a non-sequence model, which cannot separate the two hypotheses: any margin it shows is consistent with *sequence structure helps* and equally consistent with *this particular HMM happens to suit this generator*. Nothing in the README or the video may claim the former on the strength of that row. Closing it needs a second sequence-aware baseline, and none was built.

## Limits of this page

- **One seed.** Every row is a single fit at one seed; no repeat-seed variance is reported, so a small delta is not distinguishable from fit noise. Treat any delta below the seed-to-seed spread — which this repo has not measured — as unresolved rather than as zero.
- **100 merchants, 20 of them bad.** Precision@5 moves in steps of 0.20, so its deltas are coarse by construction.
- **The prevalence is not real.** `FRAUD_MERCHANT_RATE` is 0.20, chosen for per-typology sample size. `results/baf_validation.md` shows what a 1.47% prevalence does to the same decision layer.
- **The generator is ours.** `results/calibration_gap.md` measures the divergence from the one public real-merchant dataset that survived the licence gate; 5 of 8 ratio-scale marginals diverge by 1.9x or more. Every delta above is a delta on our own assumptions.

