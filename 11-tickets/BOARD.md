# 11-tickets/BOARD.md — Rakshak Ticket Board

Ordered by **risk retirement, not comfort** (CLAUDE.md). Do not reorder toward the easy ones.
One ticket per session. Read the ticket file, build, test, log in `LOGBOOK.md`, update `STATE.md`, stop.

## DAG

| ID | Title | Day | Risk | Cuttable | Depends on |
|---|---|---|---|---|---|
| [T-0001](T-0001.md) | Repo scaffold + determinism guard | Sat | low | **no — spine** | — |
| [T-0002](T-0002.md) | Spike: hand-written HMM core + minimal recovery proof | Sat | **A-002, highest in project** | **no — spine** | T-0001 |
| [T-0003](T-0003.md) | Full synthetic generator — 4 typologies + adversarial slow-ramp | Sat | medium | **no — spine** | T-0001 |
| [T-0004](T-0004.md) | Feature layer + full-scale HMM recovery confirmation | Sat | **A-002, confirmation** | **no — spine** | T-0002, T-0003 |
| [T-0005](T-0005.md) | Eval harness: splits, metrics, oracle | Sun | low | **no — spine** | T-0003 |
| [T-0006](T-0006.md) | Baselines: rule engine, LightGBM, random | Sun | **A-005 setup** | **no — spine** | T-0004, T-0005 |
| [T-0007](T-0007.md) | Cost layer: Bayes Minimum Risk + policy + capacity constraint | Sun | medium | yes — cut 5th (last) | T-0006 |
| [T-0008](T-0008.md) | Empirical-Bayes shrinkage | Sun | low | yes — cut 4th | T-0004, T-0007 |
| [T-0009](T-0009.md) | NSGA-II multi-objective frontier | Mon | **A-006** | yes — cut 3rd | T-0007 |
| [T-0010](T-0010.md) | BOCPD changepoint baseline | Mon | low | yes — cut 2nd | T-0005, T-0007 |
| [T-0011](T-0011.md) | Ablation table + sensitivity analysis | Mon | **A-005 verdict, K2** | no in spirit | T-0006, T-0007, T-0008, T-0009, T-0010 |
| [T-0012](T-0012.md) | BAF validation | Mon | low | SHOULD (FR-021) | T-0007 |
| [T-0013](T-0013.md) | Explainability + README | Mon | differentiator | no in spirit | T-0004, T-0011 |
| [T-0014](T-0014.md) | Static results dashboard | Mon | low | yes — cut 1st (first) | T-0011, T-0012, T-0013 |

## Reading this table

- **Spine (T-0001–T-0006):** not cuttable at any budget. This is the sequence in CLAUDE.md's countdown that must hold or the project DESCENDs to Phase 2 (LightGBM + cost layer only, no HMM).
- **Cut order**, if Monday arrives at 60% (last cut first, i.e. cut T-0014 before ever touching T-0007):
  `T-0014 → T-0010 → T-0009 → T-0008 → T-0007`
- **Kill criteria checked here:**
  - **K1** (HMM can't recover states) — checked at T-0002 (spike) and re-confirmed at T-0004 (full scale). Sat EOD deadline.
  - **K2** (Rakshak doesn't beat the rule engine) — verdict renders at T-0011. Do not tune to win; report it.
  - **K3** (`make eval` > 15 min) — watched from T-0005 onward, cut typologies 4→2 if violated.
  - **K4** (deadline moves earlier) — freeze at whatever ticket is complete.

## Countdown (from STATE.md)

| Date | Tickets | Milestone |
|---|---|---|
| Sat 29 Aug | T-0001 → T-0004 | HMM must recover known states by EOD. |
| Sun 30 Aug | T-0005 → T-0008 | Baselines + eval harness + cost layer. |
| Mon 31 Aug – 1 Sep | T-0009 → T-0014 | Frontier, ablations, BAF validation, README. |
| Mon 1 Sep EOD | — | **Code freeze.** `make eval` green. README numbers final. |
