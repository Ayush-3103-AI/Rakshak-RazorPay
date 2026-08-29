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

### Remaining

| ID | Title | Day | Risk | Cuttable | Blocked by |
|---|---|---|---|---|---|
| [T-0012](T-0012.md) | BAF validation | Sun 30 | low to build, **provenance-critical** | **no — promoted to MUST** | T-0015, T-0007b |
| [T-0011](T-0011.md) | Verdict: ablations, sweep boundary, lag probe | Mon 31 | **A-005 verdict, K2** | **no** | T-0006b, T-0007b |
| [T-0013](T-0013.md) | Explainability + README | **Tue 1 Sep** | differentiator — **the artifact the panel reads** | **no** | T-0011, T-0012, T-0015 |
| [T-0016](T-0016.md) | Generator recalibration | **conditional — KEPT, gate answered** | **high — see the scoping note below; one divergence is structural** | **NOT cut — user decision 2026-08-29** | T-0015 — closed |
| [T-0014](T-0014.md) | Read-only results viewer | **Wed 2 – Thu 3 Sep, video window** | low — cannot affect a number | yes, falls back to matplotlib | T-0013 |

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

Both `CLAUDE.md` and the previous version of this table dated the freeze to a **Monday,**
1 Sep 2026. **1 Sep 2026 is a Tuesday.** Corrected here on 2026-08-28; T-0017 corrected
`CLAUDE.md` and `00-charter.md` the same day. The same off-by-one had propagated into the
video window, which is **Wed 2 – Thu 3 Sep**, review **Fri 4 Sep**, submit **Sat 5 Sep**.

| Date | Tickets | Milestone |
|---|---|---|
| Fri 28 Aug | T-0001 → T-0006, T-0003b, T-0004b | done. K1 fired and was answered. |
| Sat 29 Aug | T-0017 → T-0006b → T-0007a | **The proposal gets scored for the first time.** `savings` becomes readable. |
| Sun 30 Aug | T-0007b → T-0015 → T-0012 | Sweep machinery, real data procured, decision layer validated on BAF. |
| Mon 31 Aug | T-0011 | **K2's verdict**, with its cost-asymmetry boundary. |
| **Tue 1 Sep** | T-0013, then **code freeze** | README numbers final, every one with provenance. |
| Wed 2 – Thu 3 Sep | T-0014 | Read-only viewer + video. Outside the build window. |

**There is no float.** If anything slips, T-0011 compresses before T-0013 does — T-0013 is
the differentiator and the artifact the panel actually reads.

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

## Still open before freeze

**ADR-0001 … ADR-0007 are cited across the repo and none exist as files.** Only ADR-0008 does.
Same class as the missing `09-interfaces.md`. Write them or stop citing them.
