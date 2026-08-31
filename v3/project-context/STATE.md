<!-- HEAD
FILE:     STATE.md
PHASE:    resume point
UPDATED:  2026-08-31
STATUS:   live — update at the end of EVERY session
SUMMARY:  The resume point for Rakshak v2. Read this first, every session, before anything
          else. It names the current block, the next ticket, and the exact files that ticket
          needs. Nothing else should be loaded speculatively.
OPEN:     charter §10 confirmed. Two spec defects raised by Lane A are unresolved — see below.
-->

# STATE — Rakshak v2

**Cycle:** v2 · **Phase:** 5 (execute) · **Blocks 1–5 COMPLETE**
**Next:** Lane D (model rungs) · **Window closes:** 3 Sep 2026 (freeze), submission 5 Sep

---

## Where things stand

**Sixteen tickets are done: T-100…T-102, T-110…T-116, T-120…T-122, T-130…T-133.**
Three lanes ran in parallel and are merged.

Gate across the whole tree: `ruff` clean, `mypy --strict` clean (27 source files),
**402 passed, 2 skipped, 2 xfailed.**

- **The generator is fixed and frozen-in-practice.** 10,000 merchants × 180 days →
  14.82M transactions in 53 s, 366 MB parquet. Realised population Fano **12.371** against
  a 12.25 ± 1.0 target.
- **The eval harness is FROZEN.** `EVAL-LOCK.json` is committed at `0a7172c` with
  `open_count: 0`, frozen at tree `b4bb2ab`, **before any v2 model exists**. That ordering
  is the whole claim.
- **The feature layer is built.** 28 features, both runners each, max parity difference
  1.871e-11.

v1 is complete and frozen at tag `v1-frozen`. Its numbers are immutable.

---

## §10 assumptions — CONFIRMED 2026-08-31

Freeze **3 Sep**, submission **5 Sep** · v2 in its own tree at `ver-2/v3/`, v1 at tag
`v1-frozen` · **10,000 merchants × 180 days** · **K = 50 reviews/day per 10,000 merchants**.

---

## Next action — Lane D, and it is serial

| Ticket | Depends on | Status |
|---|---|---|
| **T-140** Rung 0 floors + Rung 1 rules | T-133 ✅ | **next** |
| **T-141** Rung 2 LightGBM — the bar | T-140, T-120 ✅ | after |
| **T-142** Rung 3 + cohort residual ⚠️ **the K-1 test** | T-141, T-121 ✅ | after |
| **T-143** Rung 4 cost-in-loss | T-142 | P2, cut first |
| **T-150** perf budgets | T-142 | parallel with T-143 |
| **T-151** ⚠️ **OPEN THE TEST SPLIT — ONE-WAY DOOR** | T-150, every rung final | **not delegated** |
| **T-152** report, LIMITATIONS, clean clone | T-151 | last |

**T-142 is where charter K-1 gets its verdict.** Not T-121 — that clause was amended
(see BOARD.md, dated) because it was arithmetically unreachable. If the Rung 3 vs Rung 2
validation delta is under 5% relative, **K-1 has fired: write it in `LIMITATIONS.md` with
the number and do NOT add features to rescue it.** A clean falsification is the result.

---

## The three things that must not slip

1. ~~EVAL-LOCK written before any model trains~~ — **done, T-133, committed `0a7172c`.**
2. **The test split opens exactly once**, in T-151, after every rung is final.
3. **`make all` passes from a clean clone** by T-152. Currently
   `all: lint parity gen gates test`; `features` and `perf` still to be appended.

---

## Carry-forwards — items with an owner named, none of them optional

**1. `cli.py` must call BOTH `require_unlocked_or_refuse(split)` and `verify_lock()` before
any scoring path. Owner: Lane D.** Lane C built and tested the primitive but could not wire
it — `cli.py` is Lane A's file and only has a `gen` subcommand today. The primitive refuses
on anything but the literal string `"1"`: `"true"`, `"yes"`, `"TRUE"` all refuse. **This is
the guard on the one-way door and nothing else guards it.**

**2. `configs/scenario_v2.yaml` is missing two blocks the harness already depends on.**
- `p_catch` is used by the §2 cost matrix, absent from §8 and from the YAML. Standing in as
  a `CostParams` default of **0.80**. Belongs under `costs:`.
- §4's HOLD thresholds are spec'd as config with no `decision:` block. Defaults **0.90 /
  ₹25,000** named on `ActionPolicy`.
Both are code defaults impersonating config, which is the "no magic numbers in `src/`" rule
being bent. Fix when Lane D touches the decision path.

**3. Two unresolved spec defects in `08-generator-v2-spec.md`, raised by Lane A, not patched
around:**
- §8's `P6_macro: {amplitude: 0.15}` is incompatible with §7's `mean |z| > 1.0` gate. At
  Fano 12.25 a 15% relative shift is |z| ≈ 0.1. §4's own table already states effects in
  sigma, so confounder magnitudes were made sigma-valued. **§8 should say sigma, or §7's
  threshold should drop.**
- §4 never says what window the sigma is read over, **and it changes the answer by 20×**.
  Each event is currently read over its own duration. §4 should say so explicitly.

**4. `verify_lock(strict=True)` should now be called from T-116**, since the generator has
landed. Only `eval_module_sha256` is a hard fail today; strict promotes all three.

---

## Live risk register

| Risk | Status | Retired by |
|---|---|---|
| Generator is fiction; models measure the generator | **PARTIAL — G2 absent, not green** | needs BAF vendored |
| Cohort-residual hypothesis is wrong (charter K-1) | **OPEN — and G5 says the premise may not hold** | T-142, then T-151 |
| Cannot separate platform drift from fraud (K-4) | **OPEN — raw detector passed G5 too** | T-151 vs Rungs 2/3 |
| Features are not online-computable | **RETIRED** | T-102 ✅, T-120 ✅, G4b ✅ on real data |
| `make eval` does not reproduce on clean clone (K-5) | PARTIAL | CI job ✅, T-152 |
| 48 hours is not enough for Lanes A–E | OPEN | scope cut order, charter §8 |

**Read `LIMITATIONS.md` §5 and §6 before writing any claim about the gates.** The external
anchor is absent rather than green, and G5 currently passes for the raw detector too — so
the demo premise the project was built around does not hold on this generator as it stands.

---

## Carried out of Block 1 — read before touching the store or a timestamp

The tz-aware-UTC-nanosecond convention is not self-enforcing. duckdb renders `TIMESTAMPTZ`
in the session timezone (`EventStore.__init__` pins `SET TimeZone='UTC'`, and
`_as_contract_dtypes()` coerces every returned column — **do not add a duckdb read path that
bypasses it**), and Windows has no system tz database, which is why `tzdata` is pinned.

---

## Session log pointer

`docs/logbook/T-*.md` — one file per ticket: built / surprised / broke. The surprise field is
the one that matters. The three most valuable so far:

- **T-111** — drawing counts at a flat `F_nb` over a *composed* intensity gave a realised
  Fano of 15.11, because the variance of the intensity adds to the count variance. The unit
  tests could not have caught it; they isolate the process at constant intensity. G1 did.
- **T-113** — the R2 assertion took four attempts and three of them produced a number that
  looked like a finding about R2 and was a finding about overdispersion (8.46σ → 0.66σ). At
  Fano 12.25, almost every naive statistic measures the arrival process.
- **T-120** — parity stayed green while every baseline was empty. **Parity says two runners
  agree; it never says they agree about something meaningful.**

---

## Deferred (do not start)

MIL (Rung 5), conformal risk control (Rung 6), HSMM-NB (Rung 7), neural TPP (Rung 8), any
GNN, any UI.
