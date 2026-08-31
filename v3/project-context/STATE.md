<!-- HEAD
FILE:     STATE.md
PHASE:    resume point
UPDATED:  2026-08-31
STATUS:   live — update at the end of EVERY session
SUMMARY:  The resume point for Rakshak v2. Read this first, every session, before anything
          else. It names the current block, the next ticket, and the exact files that ticket
          needs. Nothing else should be loaded speculatively.
OPEN:     none. The four [ASSUMED] fields in 00-charter-v2.md §10 were confirmed 2026-08-31.
-->

# STATE — Rakshak v2

**Cycle:** v2 · **Phase:** 5 (execute) · **Block 1 COMPLETE**
**Block:** 2/4/5 running in parallel lanes · **Window closes:** 3 Sep 2026 (freeze), submission 5 Sep

---

## Where things stand

**Block 1 is done. T-100, T-101, T-102 are all green.** The repo scaffolds, the environment
is pinned and locked, `schemas.py` is written and **frozen**, the duckdb event store enforces
point-in-time by construction, and the `FeatureSpec` dual-runner framework plus its parity
harness are in place and proven against three deliberately broken features.

Gate at the end of Block 1: `ruff` clean, `mypy --strict` clean, **70 tests pass**.

v1 is complete and frozen at git tag `v1-frozen` (311d588). Its results are immutable — the
HMM lost to LightGBM by 0.3176 PR-AUC, the pipeline cleared the rule engine by 5.9% against a
20% gate, and those numbers stand as reported. v2 is a separate cycle with its own harness and
its own lock.

---

## §10 assumptions — CONFIRMED 2026-08-31

All four confirmed at their stated defaults. They are no longer open.

1. Freeze extended to **3 Sep**, submission **5 Sep**.
2. v2 lives in its own tree at `ver-2/v3/`, with v1 preserved at tag `v1-frozen`.
3. Scenario population **10,000 merchants × 180 days**.
4. Analyst capacity **K = 50 reviews/day per 10,000 merchants** (0.5%). Load-bearing: a wrong
   K changes the ranking of rungs, so every capacity-constrained metric names it.

---

## Next action

Block 1 opened the parallel lanes. `schemas.py` and `spec.py` freeze the boundary, so:

| Lane | Tickets | Depends on | Status |
|---|---|---|---|
| **A — generator** | T-110 → T-116 | T-102 ✅ | **running** |
| **C — eval harness** | T-130 → T-133 | T-101 ✅ | **running** |
| **B — features** | T-120 → T-121, T-122 | T-102 ✅, T-112 | starts when T-112 lands |

Lane C running early is *preferable*, not merely permitted: a harness frozen before the
generator is finished is even harder to accuse of hindsight.

---

## The three things that must not slip

1. **EVAL-LOCK is written in Block 5, before any model trains** (T-133). It is a one-way door
   and it is the project's strongest claim.
2. **The test split opens exactly once**, in T-151, after every rung is final.
3. **`make all` passes from a clean clone** by T-152. v1's reproducibility gap was the
   highest-scoring demo-day risk; charter K-5 makes it a stop-work condition.

---

## Live risk register

| Risk | Status | Retired by |
|---|---|---|
| Generator is fiction; models measure the generator | OPEN | T-116 (gates G1, G2) |
| Cohort-residual hypothesis is wrong (charter K-1) | OPEN | T-142 |
| Cannot separate platform drift from fraud (K-4) | OPEN | T-116 (G5), T-151 |
| Features are not online-computable | **framework retired** | T-102 ✅; per-feature at T-120 |
| `make eval` does not reproduce on clean clone (K-5) | PARTIAL | T-100 CI job ✅, T-152 |
| 48 hours is not enough for Lanes A–E | OPEN | scope cut order, charter §8 |

The scope cut order in charter §8 is decided now precisely so that hour-40 decisions are
mechanical rather than emotional. Cut from the bottom.

---

## Carried out of Block 1 — read before touching the store or a timestamp

**The tz-aware-UTC-nanosecond convention is not self-enforcing.** Two bugs in T-101 came from
assuming it was, and Linux CI would have caught neither:

- duckdb renders `TIMESTAMPTZ` in the *session* timezone. `EventStore.__init__` now pins
  `SET TimeZone='UTC'`, and `_as_contract_dtypes()` casts every returned timestamp column back
  to `Datetime("ns","UTC")` at the one point duckdb hands data back. **Do not add a new duckdb
  read path that bypasses it.**
- Windows has no system tz database. `tzdata` is a pinned dependency for that reason; do not
  remove it because "nothing imports it".

**`make all` is currently `lint test`, and each lane appends its own stage** — T-110 appends
`gen`, T-116 appends `gates`, T-120 appends `features parity`, T-150 appends `perf`. Add your
stage in your ticket. Leaving it to T-152 means the clean-clone job spends the whole sprint
testing a subset of what exists.

---

## Session log pointer

`docs/logbook/T-*.md` — one file per ticket: built / surprised / broke. The surprise field is
the one that matters; Phase 6 mines it, and surprises are where the project's model of the
world was wrong.

---

## Deferred (do not start)

MIL (Rung 5), conformal risk control (Rung 6), HSMM-NB explanation layer (Rung 7), neural TPP
(Rung 8), any GNN, any UI. All specified in `14-lit-survey-v2.md` and
`11-tickets/BOARD.md` §Deferred, for T-0018 §Future Work.
