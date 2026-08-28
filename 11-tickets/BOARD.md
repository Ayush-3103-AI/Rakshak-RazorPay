# 11-tickets/BOARD.md — Rakshak Ticket Board

Ordered by **risk retirement, not comfort** (CLAUDE.md). Do not reorder toward the easy ones.
One ticket per session. Read the ticket file, build, test, log in `LOGBOOK.md`, update `STATE.md`, stop.

## DAG

**Revised 2026-08-28 after T-0006.** See "Revision" at the bottom for why. "Blocked by" is
the gating edge; the Day column carries sequencing, which is ordered by risk retirement.

### Done

| ID | Title | State |
|---|---|---|
| [T-0001](T-0001.md) | Repo scaffold + determinism guard | done |
| [T-0002](T-0002.md) | Spike: hand-written HMM core + minimal recovery proof | done |
| [T-0003](T-0003.md) | Full synthetic generator — 4 typologies + adversarial slow-ramp | done |
| T-0003b | Fix-up: onset schedule, config consolidation, capacity scaling | done |
| [T-0004](T-0004.md) | Feature layer + full-scale HMM recovery confirmation | done, **K1 gate failed** |
| T-0004b | K1 remediation: FR-013 amendment + label-weighted HMM | done, gate still failing |
| [T-0005](T-0005.md) | Eval harness: splits, metrics, oracle | done |
| [T-0006](T-0006.md) | Baselines: rule engine, LightGBM, random | done |

### Remaining

| ID | Title | Day | Risk | Cuttable | Blocked by |
|---|---|---|---|---|---|
| [T-0017](T-0017.md) | Spec reconciliation + pre-registration | Sat 29 | low to build, **high if skipped** | **no — it is the pre-registration** | — |
| [T-0006b](T-0006b.md) | **HMM scorer:** filtered posterior → risk + `flag_day` | Sat 29 | **highest remaining in the project** | **no — spine. This is the proposal.** | — |
| [T-0007a](T-0007a.md) | Cost redefinition + oracle-dominance invariant | Sat 29 | **high — primary metric unreadable until this lands** | **no** | T-0017 |
| [T-0007b](T-0007b.md) | BMR policy + capacity + cost-asymmetry sweep | Sun 30 | medium | **no** | T-0007a |
| [T-0015](T-0015.md) | Public data + calibration profile + **gap diff** | Sun 30 | low tech, high scope | **no — it is the T-0016 decision gate** | — |
| [T-0012](T-0012.md) | BAF validation | Sun 30 | low to build, **provenance-critical** | **no — promoted to MUST** | T-0015, T-0007b |
| [T-0011](T-0011.md) | Verdict: ablations, sweep boundary, lag probe | Mon 31 | **A-005 verdict, K2** | **no** | T-0006b, T-0007b |
| [T-0013](T-0013.md) | Explainability + README | **Tue 1 Sep** | differentiator — **the artifact the panel reads** | **no** | T-0011, T-0012, T-0015 |
| [T-0016](T-0016.md) | Generator recalibration | **conditional** | **high — invalidates every existing number** | **gated on T-0015's diff; expected to be cut** | T-0015 |
| [T-0014](T-0014.md) | Read-only results viewer | **Tue 2 – Wed 3 Sep, video window** | low — cannot affect a number | yes, falls back to matplotlib | T-0013 |

### Published to GitHub Issues — 2026-08-28

The ticket files in this directory remain the **source of truth**; the issues are a thin
mirror carrying the blocking edges and acceptance criteria. If they diverge, the file wins.
Repo: `Ayush-3103-AI/Rakshak-RazorPay`.

| Ticket | Issue | Blocked by | Label |
|---|---|---|---|
| T-0017 | [#1](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/1) | — | `ready-for-agent` |
| T-0006b | [#2](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/2) | — | `ready-for-agent` |
| T-0015 | [#3](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/3) | — | `ready-for-agent` |
| T-0007a | [#4](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/4) | #1 | `blocked` |
| T-0007b | [#5](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/5) | #4 | `blocked` |
| T-0012 | [#6](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/6) | #3, #5 | `blocked` |
| T-0011 | [#7](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/7) | #2, #5 | `blocked` |
| T-0013 | [#8](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/8) | #7, #6, #3 | `blocked` |
| T-0016 | [#9](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/9) | #3 | `blocked` `conditional` |
| T-0014 | [#10](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/10) | #8 | `blocked` |

Three tickets are unblocked and can start immediately: **#1, #2, #3**. Move a ticket's label
from `blocked` to `ready-for-agent` when its blockers close.

T-0008, T-0009 and T-0010 have **no issues** — they were cut, not deferred.

### Cut 2026-08-28

| ID | Title | Consequence recorded at |
|---|---|---|
| [T-0008](T-0008.md) | Empirical-Bayes shrinkage | T-0011 — shrinkage on/off ablation row struck, marked **not measured** |
| [T-0009](T-0009.md) | NSGA-II multi-objective frontier | T-0011 — NSGA vs grid-search row struck, marked **not measured** |
| [T-0010](T-0010.md) | BOCPD changepoint baseline | T-0011 — **no sequence-aware baseline other than the HMM was measured; that question is reported as open** |

Cut, not deleted. Their rows are absent from the ablation table with a stated reason, never
zero and never silently missing.

## Reading this table

- **Spine (T-0001–T-0006):** not cuttable at any budget. This is the sequence in CLAUDE.md's countdown that must hold or the project DESCENDs to Phase 2 (LightGBM + cost layer only, no HMM).
- **Cut order**, if Monday arrives at 60% (last cut first, i.e. cut T-0014 before ever touching T-0007):
  `T-0014 → T-0010 → T-0009 → T-0008`. **T-0007 was removed from the cut list at T-0006** — see note below.
- **Kill criteria checked here:**
  - **K1** (HMM can't recover states) — checked at T-0002 (spike) and re-confirmed at T-0004 (full scale). Sat EOD deadline.
  - **K2** (Rakshak doesn't beat the rule engine) — verdict renders at T-0011. Do not tune to win; report it.
  - **K3** (`make eval` > 15 min) — watched from T-0005 onward, cut typologies 4→2 if violated.
  - **K4** (deadline moves earlier) — freeze at whatever ticket is complete.

## Countdown — revised 2026-08-28

`CLAUDE.md` and the previous version of this table both called the freeze **"Mon 1 Sep."**
**1 Sep 2026 is a Tuesday.** Corrected here; T-0017 fixes `CLAUDE.md`.

| Date | Tickets | Milestone |
|---|---|---|
| Fri 28 Aug | T-0001 → T-0006, T-0003b, T-0004b | done. K1 fired and was answered. |
| Sat 29 Aug | T-0017 → T-0006b → T-0007a | **The proposal gets scored for the first time.** `savings` becomes readable. |
| Sun 30 Aug | T-0007b → T-0015 → T-0012 | Sweep machinery, real data procured, decision layer validated on BAF. |
| Mon 31 Aug | T-0011 | **K2's verdict**, with its cost-asymmetry boundary. |
| **Tue 1 Sep** | T-0013, then **code freeze** | README numbers final, every one with provenance. |
| Tue 2 – Wed 3 Sep | T-0014 | Read-only viewer + video. Outside the build window. |

**There is no float.** If anything slips, T-0011 compresses before T-0013 does — T-0013 is
the differentiator and the artifact the panel actually reads.

---|---|---|
| Sat 29 Aug | T-0001 → T-0004 | HMM must recover known states by EOD. |
| Sun 30 Aug | T-0005 → T-0008 | Baselines + eval harness + cost layer. |
| Mon 31 Aug – 1 Sep | T-0009 → T-0014 | Frontier, ablations, BAF validation, README. |
| Mon 1 Sep EOD | — | **Code freeze.** `make eval` green. README numbers final. |


---

## Board corrections — 2026-08-28 (added during the T-0001..T-0006 execution session)

**K1 fired at T-0004 and was answered, not patched around.** Four-way state ARI came in at
0.091 against FR-013's 0.5 gate. The load-bearing number is the oracle-parameterised ceiling
of 0.378 (0.404 after T-0003b): with HMM parameters read straight off ground truth, the gate
is unreachable by any correctly-implemented HMM. The cause is per-state overlap — RAMP sits
1.07σ from HEALTHY, which holds ~90% of windows, and RAMP is the early-warning state the
product premise depends on. A literature survey (`project-context/12-lit-survey-k1.md`) split
the failure into a closable estimation gap and an unclosable representation gap, and the user
ratified the FR-013 amendment to AMI + per-state recall, with ARI and the oracle ceiling
retained and reported permanently. T-0004b implements the response.

**T-0007 is no longer safely cuttable.** At T-0006 the `savings` metric came back negative on
every row *including both perfect-foresight oracles*, because the provisional cost matrix
fails `07-math.md` §5's own sanity check (13.4 INR of false-positive cost per 100 INR of fraud
loss, against a stated 400–600 target). Savings is the project's primary metric. Until T-0007
repairs the cost matrix, that metric is unreadable and every downstream ticket is scored on
secondaries. T-0007 is now a prerequisite for a credible results table, not a nice-to-have.

**T-0003b was inserted** to fix an evaluation-validity defect found at T-0005: the generator's
typology onsets (days 67–187) did not compose with the frozen split, leaving zero in-window
transitions in the test window. The headline claim would have been measured on "spot an
already-bad merchant" rather than "catch a merchant drifting". The frozen artefact is the split
spec, so the generator's onset schedule was widened instead. `06-requirements.md` §3 untouched.

**Open question raised at T-0006, owned by T-0011.** LightGBM's median detection lag is −1.0
days: it flags *before* the labelled transition. Either legitimate early warning via RAMP, or
the generator telegraphing typologies before the state path records them. T-0011 must compute
lag against first entry into any bad state, and run a pre-onset separability probe. If features
separate before the labelled onset, that is generator leakage and it gets fixed and reported.


## T-0015 / T-0016 — added 2026-08-28, deliberately unscheduled

Raised after T-0006. They attack the project's deepest limitation: **every number measured to
date comes from a benchmark this repo wrote itself.** T-0015 procures public data and distils it
into a committed calibration profile; T-0016 re-grounds the generator on it and fixes two
measured sampling defects — the 20% merchant fraud rate, and the minority-mass starvation behind
T-0004b's 0.021-vs-0.091 per-segment result.

They are unscheduled rather than slotted because **T-0016 invalidates every existing number** and
forces a full re-measurement, and because T-0007 (cost matrix) currently blocks the primary
metric and must come first. If they cannot land before the Mon 1 Sep freeze, ship with the
calibration gap documented as a limitation — a partially recalibrated generator is worse than an
honestly uncalibrated one.

Hard constraint carried into both, from `06-requirements.md:28` / ADR-0007: **no public
merchant-sequence dataset with merchant-level risk labels exists.** Public data can supply
marginals and base rates, never sequences or labels. Neither ticket may drift into hunting for
one.


---

## Revision — 2026-08-28, after T-0006

The board was re-planned against the execution process the project is meant to follow:
**hypothesis → set the oracle → procure real data → ground the synthetic generator on it →
code → train → eval harness → test fairly → results.**

Steps 5–8 had been executed correctly and honestly. Steps 2–4 had not been executed at all,
and one item had fallen out of the DAG entirely. Five findings drove the revision.

### 1. The proposal had no scoring path, and no ticket owned building one

`MODEL_REGISTRY` held `random`, `rules`, `gbdt`. `hmm` sat in `EXPECTED_MODELS` marked
ABSENT and attributed to *"T-0004/T-0008."* T-0004 built features and recovery, not scoring.
**T-0008 does not contain the string `hmm`** — it is shrinkage, and it sat 4th in the cut
list. T-0007 never mentioned it. T-0011 line 9 *assumed* the HMM ablation rows existed.

So the charter's hypothesis had no producible subject: K2 would have rendered a verdict at
T-0011 on a model nobody had built a scorer for. **T-0006b** closes this and is now the
highest-risk unbuilt ticket in the project — ahead of the cost matrix, ahead of calibration.
K1 already fired once on this model; a second failure must surface Saturday, not Tuesday.

### 2. The oracle was invalid, and nothing tested that it was a ceiling

The perfect-foresight knapsack scores **−0.678** against hold-everything's **+0.573**. A
ceiling beaten by a trivial policy is not a ceiling. **T-0007a** adds the oracle-dominance
invariant — an assertion that the ceiling weakly dominates every scored policy, trivial
ones included. That single check would have caught this at T-0005 rather than T-0006.

### 3. T-0007 as written instructed the project to tune a parameter until a check passed

`07-math.md §5` said: *"Our cost matrix should reproduce roughly this ratio... If it does
not, the parameters are wrong — check this in T-0007."* Followed literally, that means
adjusting `L_m` and `V_m` until 13.4 becomes ~500 — the identical practice T-0016 forbids
for the generator, and worse here because `savings` is the **headline** metric.

The actual defect is **definitional, not calibrational**, and identifiable without ever
looking at 400–600: `c_fp` charges one window's MDR for a churn that costs **lifetime**
margin, and `L_m` counts **gross turnover** as realised fraud loss. T-0017 rewrites both
definitions and demotes 400–600 from a gate to a reported cross-check.

### 4. The headline claim was about to become conditional after the fact

T-0007b sweeps savings across the FP:fraud asymmetry. If the win turns out to be
conditional and `00-charter.md §2` is amended *afterwards*, it reads as an excuse. **T-0017
amends §2 first**, dated, before the sweep runs — the same discipline that made the K1 story
credible, where the RAMP-recall ≥ 0.35 bar was pre-registered, failed at 0.234, and was
committed as a strict `xfail`.

### 5. Real data: one slot, and T-0012 won it

T-0015 and T-0016 were unscheduled. The revision splits them at the evidence boundary:

- **T-0015 runs** (Sun 30). It now also produces `results/calibration_gap.md`, the
  per-marginal diff between the empirical profile and the generator's hand-chosen values.
- **T-0016 becomes conditional on that diff** and is expected to be cut. Publishing a
  measured calibration gap is a stronger README limitation than a half-recalibrated
  generator measured on a distribution nobody can characterise — T-0016's own argument.
- **T-0012 was promoted SHOULD → MUST** and took the last contested slot. `CLAUDE.md`
  mandates a verbatim sentence claiming BAF validation that the repo cannot currently
  back; `results/summary.md` already prints it with a parenthetical apology. Cutting T-0012
  would mean editing the project's own honesty statement downward to fit a schedule. And
  with one slot, **validating on real data beats simulating real data better.**

### Also resolved

- **The dashboard self-contradiction.** `00-charter.md §7` forbade a production UI while
  T-0014 built one. T-0017 narrows §7 rather than reversing it, and T-0014 moves to the
  **video window (2–3 Sep)** as a read-only viewer over frozen artifacts — zero build days,
  cannot touch a number, falls back to matplotlib if it does not come together. It computes
  nothing; a viewer that calculates is a second implementation that can disagree with the
  README.
- **The −1.0 day lag probe narrowed.** `_ramp()` returns `lo` for every pre-onset day and
  every injector writes through it, so no injector telegraphs. With `WINDOW_DAYS = 7`, a
  window straddling onset holds up to 6 post-onset days and window-start-vs-onset gives
  exactly −1. T-0011's probe now confirms **window aliasing** first and only escalates to a
  leakage investigation if that fails to explain it.
- **The freeze date.** 1 Sep 2026 is a Tuesday, not a Monday.

### Unchanged, deliberately

RAMP separability, SLOW_RAMP and ramp amplitude stay untouched. The ARI figure and the
0.404 oracle ceiling stay reported permanently. Both strict `xfail`s stay. Nothing is tuned
to make a gate pass — including the 400–600 cross-check.
