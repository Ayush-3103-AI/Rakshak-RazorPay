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
| [T-0017](T-0017.md) | Spec reconciliation + pre-registration | done — 2026-08-28. §2/§7 amended pre-registration, `07-math.md` §5 redefined, FR-020 re-aimed, FR-021 → MUST |
| [T-0006b](T-0006b.md) | HMM scorer: filtered posterior → risk + `flag_day` | done — 2026-08-28. `hmm` in `MODEL_REGISTRY`; forward-only `flag_day` proven by a truncation test with a negative control. **No verdict rendered — see K2 note below.** |
| [T-0007a](T-0007a.md) | Cost redefinition + oracle-dominance invariant | done — 2026-08-28. `L_m`/`V_m` redefined, `MDR_RATE` deleted, invariant wired as a harness precondition. **`savings` is readable for the first time.** |
| [T-0007b](T-0007b.md) | BMR policy + capacity + cost-asymmetry sweep | done — 2026-08-29. `budget_policy` deleted; BMR is the scored policy; sweep range 2.5–530.3 derived from `COST_PRIMITIVE_RANGES`. **HMM margin over `rules` crosses zero between 18.5 and 36.2 — the losing half ships.** No verdict rendered. |
| [T-0015](T-0015.md) | Public data + calibration profile + **gap diff** | done — 2026-08-29. Online Retail II (CC BY 4.0) procured, hashed, manifested. `calibration_profile.json` + `calibration_gap.md` committed. **Gate fired: recommends CUT on T-0016.** |
| [T-0022a](T-0022a.md) | Shock-capable generator | done — 2026-08-30. `--shock-day` / `--shock-magnitude` on `generator.generate`, writing to `data/synthetic_shock/`; the CLI refuses to write shocked data into `data/synthetic/`. Emission-only: `_apply_shock` never touches the state path. **The shocked population is NOT a paired counterfactual of the frozen one — onsets re-roll. T-0022c must compare shock-day vs control windows inside the shocked dataset.** |
| [T-0012](T-0012.md) | BAF validation | done — 2026-08-29. BAF Base fetched (558 MB, SHA-256 manifested), decision layer run on BAF's **native** month split, month 7 reported. **FR-021 met — `CLAUDE.md`'s mandated sentence is backed and the apologetic parenthetical is gone.** |

### Remaining

| ID | Title | Day | Risk | Cuttable | Blocked by |
|---|---|---|---|---|---|
| ~~[T-0011](T-0011.md)~~ | **DONE 2026-08-29 — K2 rendered: FAIL.** Verdict, ablations, sweep boundary, lag probe | ~~Mon 31~~ | **A-005 verdict, K2 — fired** | **no** | closed |
| [T-0018](T-0018.md) | Architecture doc + diagram | Sat 29 - Sun 30 | **graded deliverable, was unowned** | **no** | — |
| [T-0020](T-0020.md) | Release hygiene: LICENSE, `09-interfaces.md`, drop `pymoo` | Sat 29 | low effort, panel-visible | **no** | — |
| [T-0021](T-0021.md) | Verify `make eval` on a clean checkout | Sun 30, re-run after freeze | **can fail and generate work** | **no** | T-0020 |
| [T-0019](T-0019.md) | Video: script, shot list, edit checklist | draft Sun 30 - Mon 31; cut Wed 2 - Thu 3 | **graded deliverable, was unowned** | **no** | — to draft; T-0013 for numbers |
| [T-0013](T-0013.md) | Explainability + README | **Tue 1 Sep** | differentiator — **the artifact the panel reads** | **no** | T-0011, T-0012, T-0015 |
| [T-0016](T-0016.md) | Generator recalibration | **conditional — KEPT, gate answered** | **high — see the scoping note below; one divergence is structural** | **NOT cut — user decision 2026-08-29** | T-0015 — closed |
| [T-0014](T-0014.md) | Read-only results viewer | **Wed 2 – Thu 3 Sep, video window** | low — cannot affect a number | yes, falls back to matplotlib | T-0013 |
| ~~[T-0022a](T-0022a.md)~~ | **DONE 2026-08-30.** Shock-capable generator (`data/synthetic_shock/`) | ~~Sun 30 eve~~ | low, additive only | ~~yes~~ | closed |
| ~~[T-0022b](T-0022b.md)~~ | **DONE 2026-08-30.** Harness data-path seam: `harness.run(transactions_path=, state_paths_path=)` + `load_split(...)` overrides, carried to `gbdt.fit` / `hmm_score.fit` through an active-dataset context manager. **Not "low, mechanical" — the ticket's Build list missed four readers and would have shipped models trained on `data/synthetic/` while scoring the shock set. See the dated amendment in T-0022b.md.** | ~~Sun 30 eve~~ | low, mechanical | ~~yes~~ | closed |
| [T-0022c](T-0022c.md) | Black-swan report (`results/blackswan.md`) | Mon 31 | medium — has a pre-agreed fallback | **yes — ranked below every ticket above** | T-0022a, T-0022b |
| [T-0023](T-0023.md) | Drift-detection literature survey (doc only) | Sun 30 eve → Mon 31 | low, doc only | **yes — ranked below every ticket above** | — |

### Published to GitHub Issues

The ticket files in this directory remain the **source of truth**; the issues are a thin
mirror carrying status, blocking edges and outcomes. If they diverge, the file wins.
Repo: `Ayush-3103-AI/Rakshak-RazorPay`. **All 26 tickets mirrored as of 2026-08-29** — every
ticket in this directory, including the fix-ups (T-0003b, T-0004b) and the cut/superseded
ones (T-0007, T-0008, T-0009, T-0010), now has an issue reflecting its current status.

| Ticket | Issue | State | Blocked by |
|---|---|---|---|
| T-0001 | [#15](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/15) | closed — done | — |
| T-0002 | [#16](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/16) | closed — done | #15 |
| T-0003 | [#17](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/17) | closed — done | #16 |
| T-0003b | [#18](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/18) | closed — done | #17 |
| T-0004 | [#19](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/19) | closed — done, **K1 fired** | #18 |
| T-0004b | [#20](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/20) | closed — done, gate still fails | #19 |
| T-0005 | [#21](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/21) | closed — done | #20 |
| T-0006 | [#22](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/22) | closed — done | #21 |
| T-0007 | [#23](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/23) | closed — superseded | #22 |
| T-0008 | [#24](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/24) | closed — **cut** | — |
| T-0009 | [#25](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/25) | closed — **cut** | — |
| T-0010 | [#26](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/26) | closed — **cut** | — |
| T-0017 | [#1](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/1) | closed — done | — |
| T-0006b | [#2](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/2) | closed — done | — |
| T-0015 | [#3](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/3) | closed — done | — |
| T-0007a | [#4](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/4) | closed — done | #1 |
| T-0007b | [#5](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/5) | closed — done | #4 |
| T-0012 | [#6](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/6) | closed — done, **FR-021 met** | #3, #5 |
| T-0011 | [#7](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/7) | closed — done, **K2 FAIL** | #2, #5 |
| T-0013 | [#8](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/8) | open — `ready-for-agent`, every blocker closed | #7, #6, #3 |
| T-0016 | [#9](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/9) | open — `conditional`, **kept, not cut** | #3 (closed) |
| T-0014 | [#10](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/10) | open — `blocked` | #8 |
| T-0020 | [#11](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/11) | open — `ready-for-agent` | — |
| T-0018 | [#12](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/12) | open — `ready-for-agent` | — |
| T-0021 | [#13](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/13) | open — `blocked` | #11, re-run after #8 |
| T-0019 | [#14](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/14) | open — `ready-for-agent` to draft | — to draft; #8 for numbers |
| T-0022a | [#27](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/27) | closed — done 2026-08-30 | — |
| T-0022b | [#28](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/28) | closed — done 2026-08-30 | — |
| T-0023 | [#29](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/29) | open — `ready-for-agent`, **ranked below everything above** | — |
| T-0022c | [#30](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/issues/30) | open — `blocked` | #27, #28 |

**Open and actionable now: #8, #9 (conditional), #11, #12, #14, #28, #29.** #10, #13
wait on #8/#11; #30 now waits on #28 alone (**#27 closed 2026-08-30**). **#28, #29, #30 rank
below every other open issue on this board** — see "T-0022a/b/c, T-0023 — added 2026-08-30"
above.

Originally three tickets were unblocked: **#1, #2, #3**. Move a ticket's label
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

## T-0022a/b/c, T-0023 — added 2026-08-30, additive-only, ranked below the whole backlog

Opened from a grilling session on Sun 30 Aug evening, after K2 had already fired FAIL at
T-0011. Two gaps the project owner raised: (1) the central cost claim — *"blunt global
thresholds freeze honest merchants"* — has never been measured inside this repo, because
the generator has no population-wide/shared shock, only fully-independent per-merchant
processes; (2) the locked stack has not been re-checked against current prior art on the
actual problem (post-onboarding drift detection) since the narrow K1 metric survey.

Full spec: `project-context/14-spec-blackswan-and-drift-survey.md`.

- **T-0022a/b/c** — a black-swan shock stress test, split into three tracer-bullet slices
  (generator shock injection / harness data-path seam / integration + report) so each is
  independently demoable. T-0022c carries a pre-agreed fallback (shock-only-on-test-window,
  no retrain) because this repo has no model-persistence machinery, so "reuse the
  already-fitted T-0011 models" is not actually a lighter option. **Additive only — a new,
  separate dataset (`data/synthetic_shock/`) and a new results file
  (`results/blackswan.md`). Cannot touch or invalidate K1, K2, the ablations, or BAF.**
- **T-0023** — a written literature survey re-checking every ADR (0001-0009) against
  current prior art on post-onboarding merchant drift detection. Doc only. Explicitly does
  **not** extend to implementing or benchmarking a new model in the ablation table — that
  was proposed and rejected in the grilling session as too risky for a one-night,
  honest-metrics-required build.

**All four rank below every existing ticket on this board, including T-0014.** If the
Tue 1 Sep freeze is at risk, these are cut, or degraded to their fallbacks, before a single
hour is taken from T-0013, T-0018, T-0020, T-0021, or T-0019.

## Reading this table

- **Spine (T-0001–T-0006):** not cuttable at any budget. This is the sequence in CLAUDE.md's countdown that must hold or the project DESCENDs to Phase 2 (LightGBM + cost layer only, no HMM).
- **Cut order**, if Monday arrives at 60% (first listed cut first):
  `T-0023 → T-0022c → ~~T-0022b~~ → ~~T-0022a~~ → T-0014 → T-0010 → T-0009 → T-0008`.
  **T-0022a and T-0022b landed 2026-08-30 and are no longer cuttable**; it is additive and touches no
  committed number, so cutting the rest of the T-0022 chain now simply leaves a shock-capable
  generator with no report, which is a stated outcome rather than dead code. **T-0007 was removed from the cut list at T-0006** — see note below.
- **Kill criteria checked here:**
  - **K1** (HMM can't recover states) — checked at T-0002 (spike) and re-confirmed at T-0004 (full scale). Sat EOD deadline.
  - **K2** (Rakshak doesn't beat the rule engine) — **FIRED 2026-08-29 at T-0011. FAIL: +5.9% relative against a >=20% bar, and the claim holds at no swept asymmetry.** Nothing was tuned. The response is now live: report the negative result, pivot the narrative to explainability and the cost frontier, say so on camera. `results/verdict.md`.
  - **K3** (`make eval` > 15 min) — **checked 2026-08-29: 261 s end to end** with all four eval modules chained. Not violated; no typologies cut.
  - **K4** (deadline moves earlier) — freeze at whatever ticket is complete.

## Countdown — revised 2026-08-28

Both `CLAUDE.md` and the previous version of this table dated the freeze to a **Monday,**
1 Sep 2026. **1 Sep 2026 is a Tuesday.** Corrected here on 2026-08-28; T-0017 corrected
`CLAUDE.md` and `00-charter.md` the same day. The same off-by-one had propagated into the
video window, which is **Wed 2 – Thu 3 Sep**, review **Fri 4 Sep**, submit **Sat 5 Sep**.

| Date | Tickets | Milestone |
|---|---|---|
| Fri 28 Aug | T-0001 → T-0006, T-0003b, T-0004b | done. K1 fired and was answered. |
| Sat 29 Aug | T-0017 → T-0006b → T-0007a | **The proposal gets scored for the first time.** `savings` becomes readable. |
| Sun 30 Aug | **all done early on Sat 29, T-0011 included.** Remaining: **T-0018**, **T-0020**, **T-0021**, T-0019 draft | Sweep machinery, real data, BAF validation and the verdict all landed ahead of plan. |
| ~~Mon 31 Aug~~ | ~~T-0011~~ — **done 2026-08-29** | **K2 fired: FAIL.** There is no boundary asymmetry; the claim holds nowhere in the range. |
| **Tue 1 Sep** | T-0013, then **code freeze** | README numbers final, every one with provenance. **It must carry the FAIL.** |
| Wed 2 – Thu 3 Sep | T-0014, **T-0019 (record + cut)** | Read-only viewer + video. Outside the build window. |

**There is now roughly two days of float**, because Saturday absorbed both Sunday's and Monday's
tickets. It exists because the schedule was beaten, not because it loosened. T-0013 is the
differentiator and the artifact the panel actually reads — spend the float there, or on T-0018
and T-0020, **not on T-0016**.

**What the float must not be spent on: making K2 pass.** The verdict is rendered and
`00-charter.md` §3 forbids tuning to win. Reopening T-0008 (calibration) is defensible *only* as
an honest attempt at a known gap — BMR consumes raw scores as posteriors and `hmm`'s Brier is
0.4321 — and only if its result is reported whichever way it falls, including if it closes the
gap and the verdict changes. Anything framed as "get to 20%" is out.

*(A stale duplicate of the pre-revision countdown sat here and was removed by T-0017 on
2026-08-28: it still routed through the cut tickets T-0008–T-0010 and still dated the freeze to
a Monday. The revised countdown above is the only one.)*


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
metric and must come first. If they cannot land before the Tue 1 Sep freeze, ship with the
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
- **The freeze date.** 1 Sep 2026 is a Tuesday, not a Monday. T-0017 corrected it in
  `CLAUDE.md` and `00-charter.md` on 2026-08-28, and found the same off-by-one in the video
  window (Wed 2 – Thu 3 Sep, review Fri 4 Sep, submit Sat 5 Sep).

### Unchanged, deliberately

RAMP separability, SLOW_RAMP and ramp amplitude stay untouched. The ARI figure and the
0.404 oracle ceiling stay reported permanently. Both strict `xfail`s stay. Nothing is tuned
to make a gate pass — including the 400–600 cross-check.


---

## Session close — 2026-08-29 (T-0007b + T-0015, run in parallel)

Both tickets met every `Done when` clause. Full `pytest` exit 0 with the two strict `xfail`s
intact, `ruff` clean, `make eval` 16.3 s against NFR-004's 15-minute budget (K3 comfortable).

**T-0016's gate has fired and the recommendation is CUT.** Not because the divergence is large
but because one divergence is **structural**: the generator's `daily_count_fano_factor` is 1.0
*by construction* (`rng.poisson` ⇒ variance = mean) against a real 12.25. No parameter value
closes it; only a different emission process does, which would invalidate K1, the 0.404 oracle
ceiling and every baseline row. That is not the cheap parameter swap the *divergence small*
branch assumed. The decision is the user's and is recorded at the top of `STATE.md`.

**T-0012 is now the critical path and it is blocked on a Kaggle credential, not on code.** BAF
is Kaggle-only; the downloader is built and raises rather than fabricating. FR-021 is a promoted
MUST with no data behind it until a token exists.

### Four spec defects raised, not patched around

1. **T-0007b re-scoped T-0007a's oracle-dominance invariant in code.** Defensible — T-0007a
   predicted it in writing in `tests/test_cost.py`'s header before the session — and disclosed in
   three artifacts, with no constant moved and no sweep point dropped. But `CLAUDE.md` says raise,
   don't patch. Needs a dated amendment to `T-0007b.md` or a revert.
2. **T-0015's `Done when` ("nothing under `data/` is committed") contradicts its own build
   section** (a committed manifest at `data/external/*.manifest.json`). Code took the
   manifest-committed reading; amend the clause to "no dataset payload".
3. **FR-020 requires `sensitivity.md` as a table AND a figure.** Tables complete, **figure has no
   owner** — T-0010 owned figures and was cut. Assign or strike; silently unmet is not an option.
4. **ADR-0001 … ADR-0007 are cited across the repo and none exist as files.** Only ADR-0008 does.
   Same class as the missing `09-interfaces.md`.

### The finding that outranks the rest

**Under the BMR policy `random` scores +0.6929 savings against `rules`' +0.6980, while ranking at
PR-AUC 0.1651 — this split's prevalence.** The cost matrix earns almost all of the savings level,
not detection. `07-math.md` §6's AP-06 guard has arrived as a measurement. **T-0011 must report
savings relative to the `random` floor, never in absolute terms**, and no headline may quote
savings without PR-AUC beside it. The BMR policy also reversed the HMM/baseline ordering on
savings — that is an explanation of the earlier top-K penalty, **not** a model improvement:
PR-AUC and Brier did not move.


---

## T-0016 kept, and the scoping note that must travel with it — 2026-08-29

The user declined T-0015's cut recommendation. **T-0016 stays.** The gate has still fired and
its finding still binds, because it changes what T-0016 *can* achieve rather than whether it
runs.

**One divergence is structural, not parametric.** `daily_count_fano_factor` is **1.0 by
construction** in the generator (`rng.poisson` ⇒ variance = mean) against a measured **12.25**.
No generator constant closes it. Whoever executes T-0016 must pick a branch and say which:

- **(a) parameter swap only** — closes the four genuinely parametric marginals (`refund_rate`
  x7.81, `new_payer_frac` x0.18, `amount_log_sd` x1.94, `top_decile_payer_share` x1.89), leaves
  the Fano factor documented as an unclosed structural gap, re-measures. Fits the schedule.
- **(b) emission-process replacement** — negative binomial or latent intensity. Closes it
  properly and **invalidates every number in the repo, K1 and the 0.404 oracle ceiling
  included.** Not startable after Sun 30 Aug without losing the freeze.

Both branches carry the **n = 1** caveat: the empirical side is one UK B2B gift-ware wholesaler
in GBP, closed Saturdays. `results/calibration_gap.md` marks the non-comparable marginals
(currency, category, weekday shape) and those must not be recalibrated against.

## FR-020's figure — assigned, not struck — 2026-08-29

T-0010 owned `results/figures/` and was cut, orphaning FR-020's "table AND a figure" clause.
**Assigned to T-0007b.** `src/rakshak/eval/figures.py` → `results/figures/sensitivity.png`,
three panels covering FR-020(a), (b), (d), drawn from `results/sensitivity.csv` so the figure
computes nothing of its own. `make figures` redraws without refitting a model.

## ADRs written — 2026-08-29

`docs/adr/` now holds all eight plus an index. **Written retrospectively and dated as such**, from
the spec documents that did exist. **ADR-0005's double-booking is resolved:** the three-action
policy keeps 0005, the K1 response is renumbered **ADR-0009**.

Three record decisions that were never built and carry that in their status line — **ADR-0004**
(NSGA-II, T-0009 cut), **ADR-0006** (shrinkage, T-0008 cut), **ADR-0007** (the BAF half,
Kaggle-gated). No README or video line may cite these as shipped.

## Still open before freeze

- **`09-interfaces.md` does not exist** but tickets name it as the source of the `Scorer`
  contract. The contract actually lives in `eval/harness.py`'s module docstring. Write the file or
  stop pointing at it.
- **`pymoo` is declared in `pyproject.toml`** for T-0009, which was cut. Remove it or justify it.


---

## T-0012 closed — 2026-08-29. FR-021 is met.

BAF Base (1,000,000 rows) fetched from Kaggle, hashed, manifested. The decision layer ran against
it on **BAF's own temporal split** — train months 0-5, early-stop on 6, **month 7 reported**
(96,843 applications, 1.47% prevalence). Full run: **16 seconds**. `results/baf_validation.md`.

| model | savings | PR-AUC | precision@K | Brier |
|---|---|---|---|---|
| random | -28.2169 | 0.0143 | 0.0137 | 0.3340 |
| credit_risk_score | -5.2810 | 0.0403 | 0.0560 | 0.3200 |
| gbdt | **+0.0294** | **0.2179** | **0.1436** | **0.0129** |

`gbdt` is the only positive model, **at every one of the ten swept asymmetries** — the ordering
is not an artefact of one cost matrix.

**But no swept point reaches the operating regime this project reports on.** BAF's native
asymmetry is **61,368** and its swept range is **5,497-519,634**, against the synthetic split's
**47.5**. That is the unit assumption (credit limits of 190-2000 against absolute INR support and
review costs), not a property of BAF. In that corner the correct policy is to hold almost nobody
and BMR does exactly that. **The review-versus-hold trade-off at this project's own asymmetry is
validated by no public dataset available to it.**

### The finding that cuts back at the synthetic split

On the synthetic split `random` scored **+0.6929** against `rules`' **+0.6980** — within 0.0051
— which is why `results/summary.md` says the cost matrix, not detection, earns the savings level
(AP-06). **On BAF at 1.47% prevalence `random` scores −28.2169.**

That points at the generator's **20% merchant fraud rate**, not at the savings metric. At 20%
prevalence a random policy hits enough true positives to look competent; at 1.5% it cannot. The
AP-06 warning stands — savings must never be quoted without PR-AUC beside it — but its severity
on the synthetic split is substantially an artefact of a prevalence the generator inflated on
purpose for per-typology sample size. **T-0011 must state both halves.**

### What T-0012 does NOT validate, stated in the results file

BAF is account-opening applications with **no sequences**, so the HMM cannot run there and does
not. And the native asymmetry reads **61,368** against the synthetic split's 47.5 — that is the
unit assumption (BAF credit limits of 190-2000 against absolute INR support and review costs),
not a property of BAF. In that corner the correct policy is to hold almost nobody and BMR does
exactly that. **The balanced regime where REVIEW and HOLD genuinely trade off is not validated by
any public dataset available to this project.**

### Two defects fixed en route

- **Kaggle changed its token format.** The T-0015 downloader spoke only the legacy
  `kaggle.json` username/key Basic auth; Kaggle now mints opaque `KGAT_` bearer tokens.
  `data.download.kaggle_auth` now handles both, preferring bearer.
- **`src/rakshak/data/` had never been linted.** `pyproject.toml`'s `extend-exclude` read
  `["results", "data"]`, and unanchored `"data"` matched the source package as well as the
  git-ignored data directory. Now `["/results", "/data"]`. Nine real lint errors were hiding.


---

## Board audit — 2026-08-29. Two of the three graded artifacts had no ticket.

T-0007b, T-0015 and T-0012 all closed on **Sat 29**, a day ahead of the Sun 30 plan. The slack
prompted an audit of what remained, which found that the board was tracking the repo well and the
**submission** badly.

`00-charter.md:83` — *"**Output:** public repo + 5-minute video + architecture doc. **All three
are graded.**"* T-0013 owned the README. T-0014 owned the results viewer. **Nothing owned the
architecture doc, and nothing owned the video** — two days were allocated to a deliverable with no
script, no shot list and no owner. Four tickets were opened.

| Ticket | What it closes |
|---|---|
| **T-0018** | The architecture doc — graded, previously unowned. Describes design, not results, so it does not wait on T-0011. |
| **T-0019** | The video — graded, previously unowned. Script and shot list draftable now; numbers land from committed artifacts after T-0013; recording and the edit checklist are in the video window. |
| **T-0020** | Three defects visible to anyone who opens a public repo: **no `LICENSE` file** while `pyproject.toml` declares MIT; **`09-interfaces.md` cited by tickets but absent**; **`pymoo` declared for T-0009, which was cut.** |
| **T-0021** | **`make eval` has never run on a clean checkout.** `make` is not installed on the build machine and the `Makefile` has shipped unexercised since T-0001. T-0019's script asserts the numbers regenerate; this is what makes that true. |

**T-0021 is deliberately separate from T-0020.** Every item in T-0020 is a two-minute edit that
cannot fail. T-0021 can fail, and if it does it changes what the submission is allowed to claim.
Bundling a real risk inside a chore list is how it gets skipped.

### Accepted as debt, not ticketed

- **The cost matrix has two homes**, `eval/metrics.py` and `decision/cost.py`, pinned equal by a
  test. T-0007a's logbook assigned the migration to T-0007b; T-0007b's spec never asked for it and
  it was correctly not done half-way. Cosmetic, and risky to touch this close to freeze.
- **ADR-0003's POMDP slide** — folded into T-0019 rather than given its own ticket.

### Revised sequence

| Day | Tickets |
|---|---|
| **Sat 29** (done) | T-0017, T-0006b, T-0007a, T-0007b, T-0015, T-0012 + the FR-020 figure and the nine ADRs |
| **Sat 29 → Sun 30** | **T-0020**, **T-0018** — neither blocked, neither touches a number |
| **Sun 30** | **T-0011** (K2's verdict), **T-0021**, T-0019 draft |
| **Mon 31** | float — the day the original plan spent on T-0011 |
| **Tue 1 Sep** | T-0013, then **freeze**. T-0021 re-runs on the frozen commit. |
| **Wed 2 – Thu 3 Sep** | T-0014 + T-0019 record and cut |

**There is now one day of float where there was none.** It exists because Saturday absorbed
Sunday's tickets — not because the schedule loosened. Spend it on T-0011's verdict or on
reinstating T-0008, not on T-0016.
