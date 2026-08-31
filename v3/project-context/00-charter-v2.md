<!-- HEAD
FILE:     00-charter-v2.md
PHASE:    0 — CHARTER (v2 cycle)
UPDATED:  2026-08-31
STATUS:   active
SUMMARY:  Falsifiable charter for the Rakshak v2 rebuild. Fixes the generator and the
          evaluation harness first, then re-races the model ladder against a re-frozen
          bar. Carries the v1 result forward unchanged; v2 is a separate lock. Defines
          the success metric, the kill criteria, and the 48-hour scope cut.
OPEN:     four [ASSUMED] fields flagged inline — confirm before Block 1.
-->

# 00 — Charter: Rakshak v2

**Cycle:** v2 · **Window:** 48 hours from 31 Aug 2026 · **Team:** solo

---

## 1. Problem, one sentence

A merchant is risk-assessed once at onboarding and never again as an entity, so the
weeks in which a cleared merchant drifts into bust-out, card-testing, or transaction
laundering are watched only one transaction at a time — and a transaction scorer cannot
see a pattern that is only visible at the merchant-week level.

**Who hurts, and what it costs them.** The risk-ops analyst, who finds out at
chargeback time — 45 to 120 days late, by which point the loss is realised and
unrecoverable. And the good merchant frozen by a blunt rule, who churns.

---

## 2. Success metric

> **v2 succeeds if, on the v2 frozen harness at 1.5% merchant-level prevalence, the
> best adopted rung beats the Rung-2 LightGBM incumbent by ≥10% relative PR-AUC OR
> reduces median time-to-detection by ≥3 days at equal alerts-per-analyst-day — while
> holding p99 full-pipeline scoring latency ≤10 ms per merchant on one CPU core.**

Three things to notice about that sentence, because they are deliberate:

- **The bar is LightGBM, not the rule engine.** LightGBM beat v1 by 0.3176 PR-AUC.
  Moving the goalposts back to the rule engine would be the dishonest move and a panel
  will spot it.
- **Time-to-detection is an equal-standing win condition.** PR-AUC is a ranking metric
  and it hides latency. The operational question is *how many days of loss did we
  prevent*, and the generator knows `drift_onset_at`, so this is measurable.
- **The latency term is inside the success metric, not in a footnote.** A model that
  wins on PR-AUC and cannot be served is not a win.

**Declared before the run, per Prime Directive 5.** These margins are written here, in
git, before any v2 model exists. They are not adjustable after results are seen.

---

## 3. Kill criteria

| # | Observation | Correct response |
|---|---|---|
| K-1 | Rung 3 (cohort-residual) fails to beat Rung 2 by ≥5% relative PR-AUC | The cohort-residual hypothesis for separating adversarial from platform drift is **dead**. Report it with the number. Do not add features to rescue it. |
| K-2 | No rung beats Rung 2 by the §2 margin | Conclude that merchant-level drift detection is a **tabular problem with a temporal feature set**. Drop the ladder, ship the harness and the generator as the contribution. This is a legitimate outcome. |
| K-3 | Generator gate G2 (baseline transfer to BAF) stays RED after one repair attempt | The generator is fiction. Stop building models on it, report the gap, and fall back to evaluating on BAF alone. |
| K-4 | Gate G5 (confounder null test) cannot be made green for any rung | The system cannot distinguish platform drift from fraud, which is the whole premise. Report as the central negative finding. |
| K-5 | `make all` does not pass from a clean clone by end of Block 6 | Stop feature work immediately and fix reproducibility. An irreproducible repo scores zero regardless of its contents. |

K-2 and K-4 are the ones that would hurt. Both are still worth reporting, and both are
more interesting than a marginal win.

---

## 4. Hard constraints

| Constraint | Value | Note |
|---|---|---|
| Build window | 48 hours | Freeze extended from 1 Sep to 3 Sep [ASSUMED — confirm] |
| Compute | 1 laptop, CPU only, 4 cores | No GPU exists. This is a design input, not a limitation to apologise for. |
| Team | solo | One session at a time; parallel lanes only where `schemas.py` already exists |
| Real data | none | BAF (Feedzai, NeurIPS 2022) is the only external anchor |
| Language/stack | Python 3.11, polars/duckdb/lightgbm | Locked in `CLAUDE.md` §Tech Stack |
| Licensing | MIT / BSD / Apache only | Excludes TabPFN 2.5 and 3 (non-commercial), permits TabPFN v2 with attribution — not used this sprint |
| Eval integrity | v1 lock untouched; v2 gets its own | Non-negotiable. See §6. |

---

## 5. Project type and doctrine

**Type:** hybrid — research (evaluation methodology) + software (a servable pipeline).

**Doctrine hook:** the Ayudh build doctrine applies where it fits — determinism as a
tested invariant, external-tool parity gates with GREEN/RED verdicts, provenance as
load-bearing, and named failure modes rather than a generic "limitations" paragraph.
Where Ayudh doctrine and the generic project framework conflict, Ayudh wins.

---

## 6. The eval-integrity rule

This is the single most defensible property of the project and the 2-day extension puts
it at risk, so it is written as a charter clause rather than left to discipline.

1. **The v1 harness is closed forever.** Its results are reported as measured. No v1
   number is recomputed, corrected, or improved. The K2 gate miss (5.9% against a 20%
   bar) stands in the retrospective as written.
2. **The v2 harness is a new artifact with a new lock.** It is built and hashed into
   `EVAL-LOCK.json` in Block 5, **before any v2 model exists**.
3. **The v2 test split is opened exactly once**, in Block 6, after all rungs are trained
   on train+validation only. The open counter is in the lock file and is committed.
4. **Both results are reported side by side.** "v1 hypothesis was falsified; here is the
   diagnosis, here is the re-frozen v2 harness, here is what changed" is a stronger
   story than a single clean number, and it is true.

---

## 7. What v2 changes, and why

| v1 property | v2 change | Because |
|---|---|---|
| 20% evaluation prevalence | 1.47% (BAF-native), declared in every artifact | Savings-style metrics degenerate at inflated prevalence — v1's own AP-06 finding |
| Poisson-ish arrivals | Negative-binomial / Hawkes, Fano calibrated to 12.25 | Measured overdispersion; Poisson emissions produced a jittery Viterbi path |
| No platform-level events | Explicit confounder layer P1–P6 + gate G5 | Separating adversarial drift from natural platform drift is the open problem; you cannot test for it without generating it |
| Instant labels | 45–120 day delay, censoring, label-availability-aware splits | The chargeback window is the defining constraint of the domain |
| HMM in the scoring path | LightGBM scores; state model demoted to explanation-only, deferred | The ablation said so. Do not remove the thing that works to protect a hypothesis. |
| Batch-only features | Every feature dual-runner, online + offline, parity-tested | Data arrives as a stream. A feature that cannot be maintained incrementally is not deployable. |
| PR-AUC alone | + time-to-detection, precision@K, alert stability, mandatory floors | v1 discovered `random` winning on savings. The remedy is reporting floors always, not hiding the metric. |

---

## 8. Scope cut, in cut order

Cut from the bottom when time runs out. This order is decided now, when it is cheap,
rather than at hour 40 when it is emotional.

**Non-negotiable (if these do not ship, the sprint failed)**
1. Generator v2 with personas, typologies, confounders, delayed labels
2. Dual-runner feature framework with parity test
3. Frozen v2 eval harness with floors and time-to-detection
4. Rungs 0–2 (floors, rules, LightGBM incumbent)

**Should ship**
5. Cohort-residual layer + Rung 3 — this is the sprint's actual hypothesis
6. Gates G1–G5 all green
7. Perf budget assertions

**Cut first, in this order**
8. Rung 4 (cost-in-the-loss / `csboost`-style objective)
9. T3 sketch/graph features
10. Rung 6 (conformal risk control) — highest-value deferral, spec it in Future Work
11. T2 divergence features beyond the two cheapest

**Already deferred, spec only, do not start**
MIL (Rung 5), HSMM-NB explanation layer (Rung 7), neural TPP, GRU, any GNN.

---

## 9. Provisional cast

| Role | Skill | Status |
|---|---|---|
| Simulation/package architecture | `ayudh-sim-architect` | [have] |
| Eval harness, CI gates, property tests | `ayudh-test-engineer` | [have] |
| Data pipeline, point-in-time correctness, leakage | `quant-data` | [have] |
| Signal design, ablation rigour, overfitting discipline | `quant-alpha-research` | [have] |
| Capacity-constrained decision layer, cost asymmetry | `quant-portfolio-risk` | [have] |
| Ticketing / WBS | `elite-project-manager` | [have] |
| Agent handoff docs | `md-architect`, `agent-prd-generator` | [have] |

No gaps worth building a skill for inside a 48-hour window.

---

## 10. Open [ASSUMED] fields — confirm before Block 1

1. **Freeze extended to 3 Sep, submission still 5 Sep.** [ASSUMED]
2. **v2 is a new package in the same repo** (`src/rakshak/`, v1 preserved under
   `src/rakshak_v1/` or a git tag), not a separate repository. [ASSUMED — a tag is
   cleaner and costs nothing; recommend `git tag v1-frozen` before Block 1.]
3. **Merchant population for the v2 scenario: 10,000 merchants × 180 days.** [ASSUMED —
   large enough for 1.5% prevalence to yield ~150 positives, small enough to generate in
   under 3 minutes.]
4. **Analyst capacity K = 50 reviews/day per 10,000 merchants** (0.5%). [ASSUMED —
   drives every capacity-constrained metric; a wrong K changes the ranking of rungs, so
   confirm or state it as a swept parameter.]

---

```
━━━ GATE: PHASE 0 — CHARTER (v2) ━━━
PRODUCED:   00-charter-v2.md, 06-requirements-v2.md, 07-feature-register.md,
            08-generator-v2-spec.md, 09-interfaces.md, 10-eval-harness-spec.md,
            11-tickets/BOARD.md, CLAUDE.md, STATE.md

KEY CALLS:  1. Bar is LightGBM, not the rule engine — expensive to reverse because it
               determines whether v2 can claim a win at all.
            2. v2 gets a new lock; v1 numbers immutable.
            3. Generator is fixed before any model — consumes the first half of the
               window and cannot be reordered without invalidating every comparison.
            4. Cohort-residual is the sprint's single testable hypothesis, with K-1 as
               its explicit kill criterion.

UNRESOLVED: The four [ASSUMED] fields in §10. K in particular is load-bearing.

RISK:       Highest unretired assumption is that a corrected generator plus a
            cohort-residual feature layer is enough to beat LightGBM-alone. If K-1
            fires, the sprint's contribution reduces to the harness and generator —
            which is still shippable, and §8 is ordered so that outcome still leaves a
            complete artifact.

NEXT:       Block 1 — foundation (T-100..T-102). Estimated load: light.
```
