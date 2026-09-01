<!-- HEAD
FILE:     docs/LOGBOOK.md
PHASE:    5 — EXECUTE
STATUS:   append-only. Never edit a past entry.
SUMMARY:  One entry per ticket. The "Surprised me" field is the most valuable thing in
          this repo — it is where the project's model of the world was wrong, and it is
          what the retrospective and the panel round are built from. Record surprises
          even when they are embarrassing. Especially then.
-->

# LOGBOOK — Rakshak v2

Entry template. Copy it, fill it, append. Do not edit entries above.

```
## T-0000 | <title>
Date:        2026-09-0X  ·  Session:  N  ·  Duration:  Xh
Status:      DONE | BLOCKED | PARTIAL (split into T-000Xa/b)

Built:       <what exists now that did not before, by file>
Verified:    <the test that was run, and its output>
Surprised:   <what you expected vs what happened. Blank is a smell — something always
              surprises. If nothing did, you probably did not look.>
Broke:       <what failed, and how it was fixed. Include dead ends.>
Decided:     <any choice made that was not in the spec, and why. If it contradicts a
              spec section, name the section — that is a DESCEND candidate.>
Numbers:     <any measurement taken: timings, sizes, metric values on VALIDATION only>
Next:        <the ticket that follows, named but NOT started>
```

---

## Standing rules

- One entry per ticket, written before the session ends, not batched later.
- Numbers from the **validation** split only until T-151. Any test-split number appearing
  in this file before T-151 is an integrity breach and must be reported, not deleted.
- A BLOCKED entry stops the sprint. Write what is blocking and what would unblock it.
- If a ticket is split, log the split and why — oversized tickets are a planning signal
  worth harvesting.

---

<!-- entries below this line -->

## T-0112 | Rung 0 floors and Rung 1 rule engine on the new harness — BLOCKED
Date:        2026-09-01  ·  Session:  1  ·  Duration:  ~2h
Status:      BLOCKED — two blockers upstream of the ticket, both found by running things

Built:       `src/rakshak/models/dataset.py` — `_merchant_blocks_batched()`, replacing the
             single whole-stream `collect()` in `materialise()`. Memory restructuring only.
             `tests/unit/test_dataset_batching.py` — 3 cases, the equivalence guard.

Verified:    - `uv run pytest tests/unit/test_rungs_0_1.py tests/unit/test_cli_guard.py
               tests/unit/test_lock.py tests/unit/test_dataset_batching.py` → **60 passed
               in 11.34s**. The rung 0/1 machinery, the one-way-door guard and the real-lock
               guard are all green as they stand.
             - Batched vs single-pass `materialise()` on REAL generator output
               (`data/smoke`, 300 merchants, 442,114 events, 150 epochs): identical row
               hash `8429050030b2dd80…`, identical rows (26,040), identical
               `events_replayed`. 152.7s single-pass vs 142.5s batched.
             - `ruff check` and `mypy` clean on both touched files.
             - `EVAL-LOCK.json` and `EVAL-LOCK-CYCLE2.json` byte-identical to HEAD,
               `open_count: 0`, `open_log: []`. `RAKSHAK_UNLOCK` never set. The test split
               was not read.

Broke:       **BLOCKER 1 — `make gen` cannot run at the pre-registered T-0101 geometry on
             this machine.** `uv run python -m rakshak.cli gen --config
             configs/scenario_v2.yaml --seed 42` died with
             `memory allocation of 987455792 bytes failed`. 20,000 x 365 is ~62.8M
             transactions; the generator is fully vectorised over all of them at once.
             Measured peak private commit: **11.59 GB at 2,500 merchants (7.74M txns)** and
             **18.58–21.88 GB at 5,000 (15.72M txns)**, i.e. ~0.88 GB per million
             transactions plus a ~4.8 GB floor → **~60 GB extrapolated at 20,000**. This box
             has 16 GB RAM and an auto-managed 29.7–36.5 GB pagefile: commit limit 44.7 GB
             now, ~64 GB at Windows' 3x-RAM auto maximum. No admin rights to raise it.
             Not tuned around: the geometry is pre-registered (RE-FREEZE Amendment 4) and
             shrinking it to fit would be the exact move the charter forbids.

             **BLOCKER 2 — `tests/gates/` was never carried forward to the 365-day window,
             so the gates cannot be green.** `./make.ps1 gates` → **2 failed, 18 passed,
             4 skipped in 732.79s**. Both failures are `IndexError` in
             `gates_report.daily_counts`, from a hardcoded 180 while
             `gates_report.scenario()` overrides only `n_merchants` and `prevalence` and
             therefore runs at the manifest's 365 days:
               - `test_g1_marginal_parity.py:69` `daily_counts(gate_data, GATE_MERCHANTS, 180)`
                 → `IndexError: index 248 is out of bounds for axis 1 with size 180`.
                 **G1a produced no Fano number at all** — this is not a calibration finding.
               - `test_g5_confounder_null.py:47` `N_DAYS = 180` → `IndexError: index 229 …`.
             Two more of the same class do NOT raise and silently truncate instead:
             `test_g1_marginal_parity.py:101` and `test_g2_baseline_transfer.py:163`
             (`.filter(pl.col("window") < 180 // window_days)`) drop days 180-364, and
             `test_g5_confounder_null.py:119` `busy[window.start_day:window.end_day]` on a
             length-180 array silently fails to mark the confounders Amendment 4 moved to
             days 197 / 243 / 290 / 308 — the ones that put a discrete confounder in val
             and test. Not fixed here: `tests/gates/` is not this lane's file, and a gate
             edited by the lane it is meant to gate is not a gate.
             GREEN as reported: G3, G3b, G4, G4b (28 features x 36 merchants, 293,148 real
             events). SKIP: G1b, G1c, G1d, G2 — BAF is not vendored, pre-existing.

Decided:     Did not score any rung. `make gates` is RED, and the standing instruction is
             that no model is scored before it is green. `data/v2/` still holds cycle-1
             output (`run_summary.json`: 10,000 x 180) and was left untouched rather than
             half-overwritten.

Numbers:     gen peak commit 11.59 GB @ 2,500 · 18.58/21.88 GB @ 5,000 · ~60 GB projected
             @ 20,000 (all 365 days). Gates 732.79s. Feature replay 152.7s for 300
             merchants x 150 epochs x 442k events → the real job (20,000 x 300 epochs,
             ~50M events) extrapolates to **3.5-5 hours** single-core, once a dataset exists.
             No validation-split metric was computed. No test-split anything.

Next:        T-0112 is unchanged and un-started. It needs, in order: (a) a decision from the
             lead on BLOCKER 1 — run `gen` on a >=64 GB box, or have the generator lane make
             `_build_transaction_frame` memory-feasible (four `.tolist()` calls on 63M-element
             object arrays for instrument/status/mcc/decline_code, then a full-frame sort;
             categoricals plus a chunked write would cut peak by several fold); (b) the
             generator lane to lift the five 180-day literals out of `tests/gates/`.

---

## T-0112b | Gates carried forward to the 365-day window — BLOCKER 2 cleared
Date:        2026-09-01  ·  Session:  2  ·  Duration:  ~15m of work, 692s of gates
Status:      DONE — T-0112's BLOCKER 2 is discharged. BLOCKER 1 still open (see T-0112).

Built:       `tests/gates/gates_report.py` — `GATE_DAYS` resolved once from
             `load_scenario(CONFIG_PATH).population.n_days` (= 365); `daily_counts()` now
             defaults BOTH dimensions from the manifest; new `complete_window_counts()`
             holds the complete-window filter that G1 and G2 each restated.
             Call sites de-hardcoded in `test_g1_marginal_parity.py`,
             `test_g2_baseline_transfer.py`, `test_g5_confounder_null.py`.
             `grep -rn "180" tests/gates/*.py` now matches only an explanatory comment.

Verified:    `./make.ps1 gates` → **20 passed, 4 skipped in 692.04s** (was 2 failed, 18
             passed, 4 skipped in 732.79s). Both `IndexError`s gone. `ruff` clean.
             Blocking gates GREEN: G3 determinism (sha256 4154c262fc55de32… identical),
             G3b no-global-rng (0 bare `np.random.*`), G4 no-leakage (0 forbidden refs
             across 16 files), G4b point-in-time (28 features × 36 merchants, 293,148
             real events, agreeing online vs offline at every epoch).
             4 SKIPs all "BAF dataset not present" — pre-existing, charter K-5.

Surprised:   **Three things.**
             1. **G1a's Fano had never been produced.** It crashed before this fix, so the
                generator has never reported one. It is **13.040** against target 12.25
                ± 1.0 — GREEN with **0.21 of margin left**. That is a near-miss, not a
                comfortable pass, and nothing downstream should lean on it as if it were.
             2. **G5 came back RED** — raw **+7.07pp**, cohort-residual **+2.70pp**,
                against +2pp allowed. Quiet-day rate 0.0050 on both = nominal, so the
                calibration assertion passed and this is a *detector* finding, not a
                broken measurement. One confounder is responsible: P1 festival on
                `txn_count`.
             3. **This reverses T-116's headline.** T-116 reported G5 green for the *raw*
                detector too and framed it as the most interesting result — the setup for
                "if the two lines stay close, K-1 has fired, publish the falsification."
                The lines are **not** close: the residual cuts P1a 7.07 → 2.70pp (−62%)
                and flips P1b(308) RED → GREEN. That is directional *support* for the
                cohort-residual hypothesis, not a falsification.

Broke:       The silent half was worse than the crashing half. At `N_DAYS = 180`,
             **five of nine confounder windows were marked zero days** (P1b 308, P2b 197,
             P2c 290, P4 182-212, P5 243-253) and the threshold was fitted on **126 of
             365 days**. `assert quiet.size > 40` passed at 126, so nothing caught it.
             A patch fixing only `daily_counts` while leaving `N_DAYS = 180` would have
             calibrated on the wrong 126 days and *then* crashed at P4 — which is why the
             fix is one manifest-sourced constant and not five corrected literals.
             **Retraction:** an earlier claim in this lane that post-180 days were counted
             as *quiet* and contaminated the calibration quantile was WRONG and was
             withdrawn with the measurement. `quiet` is drawn from `np.arange(N_DAYS)`
             too, so those days were absent from BOTH sets, not misfiled into one.

Decided:     Nothing tuned. `FANO_TOLERANCE`, the 12.25 target, `EXCESS_ALLOWED = 0.02`,
             `BASELINE_DAYS`, nominal FPR, geometry, confounder days, splits and the
             generator were all left exactly as pre-registered. A RED gate is a finding.
             **Charter K-4 has NOT fired.** Its wording is "cannot be made green for any
             **rung**", and G5 here runs against a single-feature z-threshold proxy, not a
             rung. T-0116 decides K-4. G5 records rather than asserts (T-116), so the
             suite is green while the gate reports RED — by design.

Numbers:     G1a Fano 13.040 (target 12.25 ±1.0). G5 raw +7.07pp / cohort-residual
             +2.70pp (allowed +2pp), quiet-day rate 0.0050 both. Quiet 265 / busy 89
             across all nine windows (was 126 / 43). Gates 692.04s.
             **NOT comparable to T-116's +1.27pp/+0.72pp** — different geometry
             (pre-Amendment-4) and a threshold fitted on 126 of 180 days. Any before/after
             citing both would be wrong. Validation split only; test split never read.

Next:        T-0112 proper, once BLOCKER 1 (generator memory) clears.

---

## T-0119 | Point-in-time payer capsule accessor
Date:        2026-09-01  ·  Session:  2  ·  Duration:  ~18m
Status:      DONE — GitHub #53. No rung attached.

Built:       `capsules_as_of(store, merchant_id, as_of)` was **already complete and 13/13
             green** from a prior lane; it was not rewritten. `CAPSULE_SCHEMA` in
             `schemas.py` pins column order. What was added is the test that was missing:
             `test_a_capsule_is_identical_whether_or_not_the_future_is_on_disk[7|20|33]`
             (truncation against a physically truncated store, digest equality, asserting
             the cutoff actually removed rows so it cannot pass vacuously) and
             `test_the_truncation_test_is_not_vacuous` (the negative control).

Verified:    `ruff` + `mypy` clean. **17 passed in 9.07s.** G4 AST leakage scan GREEN,
             0 forbidden references across 16 files.

Surprised:   **The pre-existing suite could not catch its own failure mode.** All 13
             original point-in-time tests inspect only the OUTPUT rows, so an accessor
             that consults day 40 to compute a value stamped onto day 20 passes every one
             of them. Mutation-tested rather than assumed: a mutant computing
             `device_payers` from a prefix a year past `as_of` — emitting no future row,
             touching no timestamp — fails the 4 new tests and **PASSES three of the
             originals**. That leak would have inflated every rung above it invisibly.
             Second surprise: `payer_is_new` is lookahead-safe for free
             (`event_date == min(event_date)`; extending a prefix forward cannot lower a
             minimum), while the innocuous-looking device counter is the real leak channel.

Broke:       Nothing. The prior lane's accessor was sound; only its test coverage was not.

Decided:     **KNOWN CEILING, recorded not hidden — this needs a lead decision at T-0120.**
             The accessor is NOT bounded. Eleven of thirteen columns are per-(merchant,
             day, payer) and incremental trivially. `payer_is_new` and
             `device_shared_payers` need insert-only per-merchant / per-device sets that
             grow with distinct payers and devices. These are **the same two quantities
             T-122 already cut from the T2 register** (`g_payer_hhi`,
             `g_device_reuse_rate`) as too large for NFR-04's 4 KB. Resolution taken:
             capsules are not register features, carry no `MerchantState`, and are not
             measured against NFR-04 — but this module does scan the whole visible prefix
             per call. Upgrade path (per-merchant HLL, bounded LRU of hot devices) is in
             the module docstring **with the note that a sketch CHANGES THE NUMBERS**, so
             it belongs to the Rung 5 decision and must not be smuggled into the accessor.
             Net: Rung 5's servability question now lives in this module rather than
             having been solved by it.

Numbers:     17 tests, 9.07s. 13-wide Float64 vector, one `store.query_events` call.
             No metric computed on any split. Test split never read.

Next:        T-0120 (Rung 5 MIL) — but NOT before `EVAL-LOCK-CYCLE3.json` is written,
             per the cycle-3 pre-registration's ordering. Lock is the lead's, after T-0118.

Carry-forward (defect found, deliberately NOT fixed in this lane):
             `EventStore.epoch_bounds()` in `store.py` reads min/max `event_date` over the
             ENTIRE parquet with no `as_of` bound — a genuine lookahead surface on the
             store. `capsules_as_of` does not call it, so nothing is contaminated today.
             **Every other caller is unaudited.** Needs a ticket.
