# Rakshak — recall by fraud typology (FR-005, CLAUDE.md non-negotiable 1)

> **Sequence-layer metrics are measured on synthetic merchant streams with injected typologies; the generator is in this repo.** The decision layer is additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark derived from real bank data.

BAF has no typologies and no sequences, so nothing on this page could have been measured there. Every number here is the synthetic split.

## Provenance

| Field | Value |
|---|---|
| Produced by | `python -m rakshak.eval.typology --seed 42` |
| Seed | 42 |
| Split reported | `test` (days 210-269), unlocked with ticket T-0013 |
| Population | 100 merchants, 20 truly bad |
| Review budget K | 5 merchants |

**Nothing here selects anything.** Every configuration was frozen at T-0004b on `validate`; this module re-partitions the actions `harness.evaluate_model` already chose at T-0011. No number in `results/verdict.md` can move because of this file, and no row here can feed back into a decision.

## Read the sample size before the recall

Truly-bad merchants per typology on this split: `BUST_OUT` n=4, `CATEGORY_DRIFT` n=4, `LAUNDERING_ENDPOINT` n=4, `REFUND_COLLUSION` n=4, `SLOW_RAMP` n=4.

**Every cell below is a proportion over about four merchants.** Recall is therefore quantised to {0, 0.25, 0.50, 0.75, 1.00} and the 95% Wilson interval spans most of the unit interval at every one of those points. These rows carry information about *direction* and almost none about *magnitude*. They are published with their intervals rather than omitted, because the alternative to an honestly-underpowered table is not a better table — it is a submission that quietly never reports the typology it promised to fail on.

The intervals are Wilson score intervals, not normal approximations: at 0 and 4 successes out of 4 the normal interval has zero width, which would be a lie in exactly the cells that matter most.

## Recall by typology — `acted on` (policy chose REVIEW or HOLD)

The operational definition: the merchant reached an analyst or had settlement held. Subject to FR-017's capacity constraint, and defined for every model.

| typology | n | `random` | `rules` | `gbdt` | `hmm` |
|---|---|---|---|---|---|
| `BUST_OUT` | 4 | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) |
| `CATEGORY_DRIFT` | 4 | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) |
| `LAUNDERING_ENDPOINT` | 4 | 1.00 (4/4, 95% CI 0.51-1.00) | 0.75 (3/4, 95% CI 0.30-0.95) | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) |
| `REFUND_COLLUSION` | 4 | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) |
| `SLOW_RAMP` **(adversarial)** | 4 | 0.75 (3/4, 95% CI 0.30-0.95) | 0.50 (2/4, 95% CI 0.15-0.85) | 0.50 (2/4, 95% CI 0.15-0.85) | 0.75 (3/4, 95% CI 0.30-0.95) |

## Recall by typology — `flagged` (the model's own score crossed its threshold)

Unconstrained by capacity. `random` returns no `flag_day` — a single per-merchant score has no time at which it fired — so its column is `n/a` rather than zero.

| typology | n | `random` | `rules` | `gbdt` | `hmm` |
|---|---|---|---|---|---|
| `BUST_OUT` | 4 | n/a | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) |
| `CATEGORY_DRIFT` | 4 | n/a | 0.50 (2/4, 95% CI 0.15-0.85) | 0.75 (3/4, 95% CI 0.30-0.95) | 1.00 (4/4, 95% CI 0.51-1.00) |
| `LAUNDERING_ENDPOINT` | 4 | n/a | 0.50 (2/4, 95% CI 0.15-0.85) | 0.50 (2/4, 95% CI 0.15-0.85) | 0.25 (1/4, 95% CI 0.05-0.70) |
| `REFUND_COLLUSION` | 4 | n/a | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) | 1.00 (4/4, 95% CI 0.51-1.00) |
| `SLOW_RAMP` **(adversarial)** | 4 | n/a | 0.25 (1/4, 95% CI 0.05-0.70) | 0.00 (0/4, 95% CI 0.00-0.49) | 0.50 (2/4, 95% CI 0.15-0.85) |

## The typologies

| typology | what the generator injects |
|---|---|
| `BUST_OUT` | legitimate history, then a hard volume ramp, then the account vanishes |
| `CATEGORY_DRIFT` | a silent shift of ticket size and time-of-day profile |
| `LAUNDERING_ENDPOINT` | normal tickets, abnormal payer graph — many payers, no repeats |
| `REFUND_COLLUSION` | merchant and a small payer set extract value through refunds |
| `SLOW_RAMP` | **ADVERSARIAL (FR-005)** — a monotone, changepoint-free drift built to defeat exactly the changepoint logic this project is made of |

## `SLOW_RAMP` — the row this table exists for

`SLOW_RAMP` is FR-005's adversarial typology: a monotone, changepoint-free drift, built deliberately to defeat changepoint and state-transition logic. **It exists so that this project has somewhere honest to fail**, and `CLAUDE.md` forbids tuning it away.

| model | recall on `SLOW_RAMP` | recall on the other four typologies | delta |
|---|---|---|---|
| `random` | 0.75 | 1.00 | -0.25 |
| `rules` | 0.50 | 0.94 | -0.44 |
| `gbdt` | 0.50 | 1.00 | -0.50 |
| `hmm` | 0.75 | 1.00 | -0.25 |

**A delta at or above zero here is not evidence that the adversarial typology was solved.** With four merchants in the `SLOW_RAMP` cell and sixteen in the comparison cell, this difference is not separable from sampling noise at any conventional level, and the intervals in the tables above show it directly. The row is published in whichever direction it falls; it is not evidence in either.

## What this table does not establish

1. **Nothing about magnitude.** Four merchants per cell. See the interval on every number.
2. **Nothing about real fraud.** These are the generator's own five caricatures, injected by this repo and detected by this repo. `results/calibration_gap.md` measures how far the generator's marginals sit from a real transaction stream; its typology *dynamics* are calibrated against nothing, because no public merchant-sequence dataset with merchant-level risk labels exists (`06-requirements.md:28`, ADR-0007).
3. **Nothing about the pooled verdict.** K2 was rendered at T-0011 on pooled savings and PR-AUC and is unchanged by this decomposition. See `results/verdict.md`.

