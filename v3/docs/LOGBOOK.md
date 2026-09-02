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

---

## T-0112a | Generator: stream the transaction table — BLOCKER 1 cleared
Date:        2026-09-01  ·  Session:  2  ·  Duration:  ~1h
Status:      DONE — `make gen` runs at the pre-registered geometry on this box.

Built:       `_build_transaction_frame` → `_transaction_blocks()`, yielding
             merchant-contiguous ~2M-row blocks. `transactions` is a `cached_property`
             that concatenates for tests/gates; `write()` streams blocks through
             pyarrow's `ParquetWriter` (polars cannot append); `sha256()` folds block
             row-hashes. `row_counts` added so a row count needs no materialisation.
             `_append_children`: 14 `np.concatenate` calls → one in-place field-by-field
             append. `del` of ~12 spent per-transaction arrays in `generate()`.
             `cli.py` one line: `getattr(data, name).height` → `data.row_counts[name]`.

Verified:    **61,715,987 transactions, 193.8s wall, peak private commit 9.005 GB**
             (was: crash at `memory allocation of 987455792 bytes failed`).
             Identity before/after at 300 and 2,500 merchants and at prevalence 0.0
             (the other Hawkes branch): identical `content_sha256`, row count, row
             order, schema, `DataFrame.equals=True`. Row hashes cd38c342…, a90e1f45…,
             19c381e8… on both sides. ruff + mypy clean. `tests/unit` failure set
             identical before and after, proven by reverting both files and re-running.

Surprised:   **The handed-down diagnosis was the smaller half.** The four `.tolist()`
             calls are real but minor. The dominant cost is that the finished frame
             CANNOT EXIST: at 61.7M rows polars' string-view columns come to ~19 GB on
             their own (`device_hash`/`ip_hash` are 32 B/row each), costed at ~12.4 GB
             best case even with five dtype changes. **No amount of categorical-ing
             reaches 14 GB. Streaming was the only door.**
             Second: `_append_children` was quietly the single worst line — two complete
             copies of a 5.6 GB mark stream, 11.3 GB at full scale. In-place is a
             *smaller* diff and a 5.7 GB saving.
             Third: the `del`s bought nothing at small n (3.51 → 3.56 GB at 2,500) and
             matter only at full scale where `generate()`'s own transient binds.

Broke:       `refunds.parent` is a global index while inherited marks are looked up
             block-locally — found by running, fixed with `searchsorted` onto the block
             plus a runtime check that a refund's capture never leaves its block.

Decided:     Did NOT chase parquet byte-identity. File bytes differ (pyarrow row-group
             layout vs polars); `GeneratedData.sha256()`'s own docstring says file bytes
             are the wrong comparison, and that hash is what gate G3 compares.

Numbers:     Peak private commit, same in-process instrument both sides — 300: 1.446 →
             1.428 GB · 2,500: 6.624 → 3.593 GB · 5,000: OOM → 4.658 GB · 10,000: →
             5.888 GB · **20,000 × 365: CRASHED → 9.005 GB**.
             `content_sha256: 1ef7e1802349e0bba6f2efbeba2cd667e1d8dc6aa56f6e006603cac03f59741a`

Next:        Feature panel, then rungs. `data/v2/features.parquet` from cycle 1 is STALE
             and must be rebuilt. Old dataset backed up to `data/_v2_cycle1_backup/`.

---

## T-0118 | Cycle-3 metrics, the decision seam, the explainer registry
Date:        2026-09-01  ·  Session:  2  ·  Duration:  ~45m
Status:      DONE — GitHub #52. Step 2 of the pre-registration's ordering. No rung, no lock.

Built:       `false_hold_coverage` (P(HOLD | not a fraud-day) per Mondrian stratum, strict
             `realised > alpha`, **no slack/shrinkage/clamp**, no aggregate row, `nan` on a
             zero denominator), `onset_localisation_error` (signed days, `n_unlocalised`
             counted never imputed, quartile convention stated), `tpp_rescaled_ks` (takes
             compensator increments directly, so it holds no opinion on Rung 8's
             parametric form and could be declared before that rung exists).
             `DecisionPolicy` protocol + `CapacityTopK` + `DEFAULT_DECISION`;
             `sweep_cost_asymmetry` routed through it. `explain/registry.py` whose
             `register()` REFUSES anything satisfying `Scorer`.

Verified:    ruff + mypy clean, 35 new tests pass. `DEFAULT_DECISION.decide(...)` asserted
             `np.array_equal` to pre-refactor `select_actions(...)` across all five locked
             seeds, plus row-list equality on the whole sweep.

Surprised:   T-0118's golden criterion names a byte-identical `summary.md` and **nothing in
             this tree writes one** — `eval/report.py` has no such path. Substituted the
             thing that number is made of: identical actions imply identical savings,
             floors, P@K and TTD by construction.

Decided:     `DecisionRequest` carries no `y`/`loss`/`truth`/`onset` field and a test asserts
             those names are absent — **Prime Directive 3 enforced by shape, not by
             discipline.** A wrapper may only soften HOLD→REVIEW→PASS; promoting breaches K.

Numbers:     35 tests, 6.83s. No metric computed on any split.

Next:        `EVAL-LOCK-CYCLE3.json` — the lead's step 3. Blocked, see below.

---

## T-0126b | Lock resolution by chain, and two defects the seal exposed
Date:        2026-09-01  ·  Session:  2  ·  Duration:  ~1h
Status:      DONE for the code. **THE SEAL ITSELF IS STILL HELD** — see Broke.

Built:       `resolve_authoritative(root)` + `LOCK_GLOB` + `BrokenLockChainError` in
             `eval/lock.py`. `load_lock`, `read_open_count`, `record_open` and
             `verify_lock` default to it; `write_lock` deliberately does not (it writes a
             NEW lock, and resolving there would make it refuse itself).
             `write_lock` now emits `cycle`/`supersedes`/`pre_registration` and takes
             `boundaries`. `cli.py`'s seven pinned uses now resolve.

Verified:    Both sealed locks verified against the committed tree at `5167abb` BEFORE
             T-0118 landed — `f15be39e…` on both, recorded in
             `docs/LOCK-VERIFICATION-2026-09-01.md`, because after T-0118 the working tree
             stops matching and never matches again.

Surprised:   **Two defects, both found by sealing cycle 3 and READING what came out
             rather than trusting it. The lock produced was inspected, found wrong, and
             deleted before it was ever committed.**
             1. `write_lock` sourced `split_boundaries` from `DEFAULT_BOUNDARIES`, whose
                dataclass defaults are STILL the 180-day window — T-0101 moved the geometry
                by passing explicit tuples and deliberately not editing `eval/splits.py`.
                The sealed lock read train (0,119) / val (120,149) / test (150,179) against
                a pre-registration carrying 0-239 / 240-299 / 300-364. **A lock certifying
                a window the harness never scores is worse than no lock.** Cycle 2 has the
                right values only because they were typed in by hand.
             2. `cli.py` pinned a lock filename — first cycle 1, then cycle 2. Both edits
                were right on the day and both went stale at the next freeze *silently*,
                because a superseded lock is still a valid file that parses. One of the
                seven call sites is `record_open`, **the one-way door**: opening the test
                split against a pinned name increments a lock that no longer governs,
                leaving the authoritative lock reading `open_count: 0` after the split had
                been opened, with every check still reporting it true.

Broke:        Sealing makes cycle 3 authoritative, and `artifacts/build.py`'s
             `_results_provenance()` then refuses the whole ladder — it compares each row's
             `eval_lock_sha` against the AUTHORITATIVE lock, and cycle 3 is the first cycle
             where `eval_module_sha256` actually moves. **That refusal contradicts the
             pre-registration it enforces:** §3 says rungs 0-4 stay judged on cycle 2 and no
             committed number moves. The check must become chain-aware — legitimate if a row
             matches SOME lock in the chain, recording which cycle, still a hard refusal if
             it matches none. With the artefacts lane; the seal follows it.

Decided:     Did not weaken enforcement. The mismatch still raises, now against the lock
             that governs, and the message names the resolved file rather than always
             saying `EVAL-LOCK.json`. The risk carried deliberately: sealing a new cycle
             now clears drift, so it is visible IN the chain (`supersedes`,
             `pre_registration`, `open_count`, `open_log`) rather than hidden by it.

Next:        Seal cycle 3 once the ladder's provenance is chain-aware.

---

## T-0126c | G5 series dump — #62 unblocked
Date:        2026-09-01  ·  Session:  2  ·  Duration:  ~25m
Status:      DONE. `g5_confounder_null` goes MISSING → PRESENT.

Built:       `tests/gates/test_g5_confounder_null.py` writes `data/v2/gates/g5_series.json`.
             `role` is DERIVED — one constant `DETECTOR_FEATURE = "txn_count"`, and
             `role = "adversarial" if window.feature == DETECTOR_FEATURE else "control"`.

Verified:    Numbers unchanged and verdict still RED, matching the T-0112b entry exactly.
             `make artifacts` twice → identical sha256 `98ec8df3`.

Surprised:   **The G5 leg is 5 seconds, not slow** — the 692s in T-0112b is the whole gates
             suite. And **seven of nine windows were already GREEN**, every control within
             ±0.5pp of nominal: the null holds everywhere the detector is not looking, which
             is what makes P1 a clean finding rather than noise. P6 (sinusoid, `txn_count`)
             is GREEN at −0.23pp while P1 (festival, same feature) is +7.07pp — amplitude
             and abruptness, not "volume events" as a class.

Decided:     `alert_rate_by_day[0..28]` is `null`, not `0.0` — those days have no trailing
             baseline so z is undefined; `0.0` would draw "the detector was quiet", a
             different and false claim.

Broke:       Nothing. **Carried forward unresolved:** `build_g5`'s error message calls the
             window range INCLUSIVE while the generator emits and the measurement uses
             HALF-OPEN `[start, end)`. Harmless today (max `end_day` 313 < 365) but #62 will
             render every P2 band two days wide if it believes the message.

Numbers:     9 windows, 2 series, 365 days. P2's role is the genuinely arguable one and is
             recorded as such; P6's window is 11-34 against `BASELINE_DAYS` 28, so only
             **5 of 23 days are measured** and `days_measured` is in every row to say so.

Next:        #62 can be built against the artefact.

---

## T-0121b | The P2 cohort-residual test has not run since the 365-day move
Date:        2026-09-01  ·  Session:  2  ·  Duration:  ~20m
Status:      DONE for the fixture. **The assertion is RED and left as the ticket words it.**

Built:       `_p2_population()` scales the splits with the window (train 0-65, val 66-81,
             test 82-99), keeping the real proportions. `N_DAYS` is a named constant so the
             window and the day filter cannot drift apart again.

Verified:    The fixture generates. The test executes for the first time since T-0101.

Surprised:   **It was not failing, it was not RUNNING.** The config validator requires
             `test_end_day == n_days - 1`; the fixture shrank the population to 100 days and
             left the 365-day boundaries behind, so it raised `ConfigError` before generating
             anything. It has been an ERROR in every suite run since the migration, and an
             ERROR reads as infrastructure noise rather than as a dead K-1 assertion.

Numbers:     P2 day — median z=+1.311, median r=−0.000, mean|z|=4.247, mean|r|=4.215,
             alert@3 raw=0.4290 res=0.3677. Null day — median z=−0.462, median r=+0.000,
             mean|z|=0.844, mean|r|=0.646, alert@3 raw=0.0400 res=0.0477.
             3,000 merchants, prevalence 0, confounders on, seed 11, validation side only.

Decided:     `assert res_alert < raw_alert * 0.85` LEFT UNCHANGED and red. The layer cancels
             common-mode exactly — median residual 0.000 to three decimals on both days —
             but the alert rate falls **14.3% against a 15% bar**, missing by 0.7pp.
             Relaxing 0.85 → 0.86 is the move this project forbids and the move the gates
             lane was told not to make when G5 came back RED.
             **This corroborates G5 independently.** G5: residual cuts P1a +7.07 → +2.70pp
             (−62%), still over the +2pp allowance. Here: −14.3% against −15%. Two separate
             confounders agree that **the cohort residual works materially and
             under-delivers against every bar declared for it.** That is a finding about the
             central hypothesis, not a test-maintenance problem.

Next:        Lead's call whether to apply the sibling test's `xfail(strict=True)`-with-the-
             measurement idiom. It reclassifies K-1 evidence, so it was not done here.

---

## PERF | `features --workers`, and a bad number of mine
Date:        2026-09-01  ·  Session:  2  ·  Duration:  ~50m
Status:      DONE. Landed defaulting to serial; NOT used for the run in flight.

Built:       `_replay_chunk` / `_replay_results` in `models/dataset.py`, yielding in TASK
             order. `workers<=1` is `map()` in-process, no pool, no pickling, byte-identical
             to before. Above that, `ProcessPoolExecutor` on a spawn context with a bounded
             window of `2*workers` futures so buffered results cannot become a second copy
             of the cube. `--workers` on the CLI, default 1.

Verified:    800 real merchants, full 300-epoch horizon: workers 1/4/8 → 676.1s / 288.2s /
             193.4s, **all three sha256 `47cb4d50…`**, identical rows, order, schema,
             `events_replayed`, `rows_by_split`, `state_bytes_p99`.
             `tests/unit/test_dataset_parallel.py` enforces it with `PARALLEL_CHUNK`
             monkeypatched to 7 so completion order ≠ task order.

Surprised:   **The 8.3-hour estimate was wrong and it was mine.** Extrapolated from the
             log's first sample, 1,000 merchants in 1,503s — taken while the box was
             thrashing. Steady state is **~196s per 1,000 merchants (~12,000 events/s)**,
             about 8x faster. Proved accidentally: when a test suite holding ~6.5 GB exited,
             available RAM jumped 1,219 → 7,715 MB and the live job's rate recovered in the
             same window. **On this box the panel's throughput is governed by free RAM, not
             by cores** — so the benchmark cost the live run (~1,100s) more than parallelism
             would have saved on its remainder. Keep other jobs off this box while
             `features` runs.
             Second: per-chunk scan is ~0.2s. The stream is merchant-major so row-group
             pruning makes chunk scans nearly free — the opposite of the expected I/O wall.

Decided:     The cohort residual is NOT parallelised — `assign_cohorts`/`residual_matrix`
             need every merchant present on a day. It stays in the parent, after the cube is
             whole. Parallelism is scoped to the replay, which is where the time is.
             Did not restart the live run: it was at 9,000/20,000, and restarting would
             replay ~50M events from zero to at best tie the remaining time while discarding
             finished work and swapping a proven path for a 90-minute-old one, two days
             before a freeze.

Numbers:     3.5x at 8 workers, sub-linear and measured with the live job competing. Peak
             footprint serial 1,633 MB vs 8 workers 1,976 MB — only 21% more, because the
             smaller chunk offsets the worker count.

Next:        Use `--workers` for the next materialisation: a re-run, a second seed, cycle 3.

---

## PERF-corr | Correction to the PERF entry's throughput number
Date:        2026-09-01  ·  Session:  2
Status:      CORRECTION. The entry above stands as written (append-only); this supersedes
             its throughput figure only. Everything else in it is unaffected.

Wrong:       "Steady state is ~196s per 1,000 merchants (~12,000 events/s), about 8x
             faster [than 1,503s]." That was generalised from the merchants 3000-7000
             stretch. It is a ~5x anomaly against every other bucket, with identical event
             counts per bucket, and it has no explanation — recorded as unexplained rather
             than given a story.

Right:       Uncontended rate is **~1,000-1,150s per 1,000 merchants (~2,400 events/s)**,
             measured on a quiet box with only the materialiser running:

               8000   5761s
               9000   7060s   (+1299)  contended by a concurrent pytest run
              10000   8204s   (+1144)  clean
              11000   9211s   (+1007)  clean

             Full run ≈ **5h20m**, not 8.3h and not 40m. ETA ~15:30 on 2026-09-01.

What this   The contention finding SURVIVES but is demoted: 1299s against ~1075s is a real
does to     ~20% penalty from running a test suite alongside, not the dominant term it was
the story:  claimed to be. "Throughput is governed by free RAM, not cores" is still the
             right operational rule — available physical was 3,011 MB with only the
             materialiser resident — but it explains a fifth of the variance, not all of it.

Consequence: The parallel path is NOT usable at full geometry today. Projected peak for
             20,000 merchants at 8 workers is ~3.2 GB against 3,011 MB available. It has
             only ever executed at 800 merchants. An OOM would cost the whole run rather
             than the ~70 minutes a restart would save, and the freeze is 2026-09-03 with
             ~36 hours of slack. Serial finishes. `--workers 6` next time, not 8 — the
             parent's cube alone is 1.2 GB at full geometry and worker count is the cheap
             thing to trade away.

Surprised:   Three estimates of the same quantity in one session — 8.3h, 40m, 5h20m — two
             of them wrong and both wrong because a rate was extrapolated from one window
             of a log without checking whether that window was representative. The
             equivalence proof, which was measured rather than extrapolated, needed no
             correction.

---

## PANEL | The v3 feature panel, and NFR-04 measured at 2.37x its budget
Date:        2026-09-01  ·  Session:  2  ·  Duration:  4h38m wall
Status:      DONE — the panel exists. **NFR-04 IS VIOLATED and is reported, not fixed.**

Built:       `data/v2/features.parquet`, 264 MB, materialised from the streamed 20,000 ×
             365 dataset at the pre-registered geometry.

Verified:    3,070,800 rows · 20,000 merchants · 300 epochs · `last_day: 299` ·
             50,334,951 events replayed · 16,694.1s · 28 base + 21 residual features.
             `rows_by_split: {train: 2,888,640, val: 182,160}` — **no test rows.** The
             materialiser defaults `last_day` to the last VALIDATION day and refuses the
             test split from this entry point, so days 300-364 were never replayed.
             `RAKSHAK_UNLOCK` never set; `open_count` still 0 on all three locks.

Surprised:   **`state_bytes_p99 = 9,716 B` against NFR-04's 4,096 B budget.**

             | | bytes |
             |---|---|
             | NFR-04 budget (`STATE_BYTES_BUDGET`) | 4,096 |
             | **Declared** total, summed over the register | 3,968 |
             | **Measured** packed p99 over 20,000 merchants | **9,716** |

             The import-time guard in `features/registry.py` passes, because it checks the
             sum of each feature's **declared** `state_bytes` — 3,968, comfortably under.
             `MerchantState.nbytes()` returns `len(pack(self))`, the actual packed wire
             format, and its own docstring names this exact scenario: *"the registry checks
             declared state_bytes at import so the budget fails at startup; this measures
             what the declaration promised, so a feature that quietly outgrows its
             declaration is caught in tests/perf/."*

             **The mechanism the code anticipated has fired.** The declarations were
             honest; the features outgrew them, and nothing between import and here
             compares the two. A guard that validates a declaration rather than a
             measurement passes precisely when the declaration is the thing that is wrong.

Broke:       Nothing crashed. That is the point — this is a silent 2.37x overrun that no
             gate, no import guard and no unit test caught, surfacing only because
             `materialise()` happens to record a p99 in its summary.

Decided:     **Not fixed, not tuned, not shrunk.** Under Prime Directive 5 a rung is adopted
             only if it beats the previous rung by the declared margin **AND meets the
             compute NFRs**. So this is not a footnote to the ladder — it is an open
             adoption question hanging over EVERY rung scored on this panel, before a single
             rung has been scored. Shrinking a feature's state now, after the panel exists
             and before any rung is scored, would change the numbers every rung is about to
             be judged on. That is the lead's call with a re-materialisation, not a patch.

             It also converges with two findings already logged:
             - **T-0119**: `capsules_as_of` is unbounded — `payer_is_new` and
               `device_shared_payers` need per-merchant/per-device sets growing with
               distinct payers and devices.
             - **T-0120**: those are the same two quantities T-122 cut from the T2 register
               as too large for NFR-04, and Rung 5 is therefore not servable under NFR-04
               as it stands.
             Three independent routes to the same conclusion: **the state budget is the
             binding constraint on this design, and the register is already over it.**

Numbers:     panel 3,070,800 × 49 features · 50,334,951 events · 16,694.1s (4h38m) ·
             264,159,658 bytes · `state_bytes_p99` **9,716 B** vs 4,096 B budget (2.37x) ·
             declared 3,968 B. Validation split only; no test-split number exists.

Next:        Rungs 0-4 rescored on this panel, then T-0114 (the K-1 test). The NFR-04
             overrun needs a lead decision BEFORE adoption is claimed for any rung — the
             margin and the NFR are two separate clauses of Prime Directive 5 and both bind.

---

## T-0112/0113/0114 | The ladder scored. K-1 FIRED, and no rung beats a volume ranker.
Date:        2026-09-01  ·  Session:  2  ·  Duration:  ~1h30m
Status:      DONE. Two findings, both negative, both reported per Prime Directive 6.

Built:       Rungs 0-4 trained and scored on VALIDATION, all five locked seeds for every
             trained rung. `data/v2/models/rung{2,3,4}_seed{42..46}` and 18 result rows in
             `data/v2/eval/`. Harness verified before and after: `verify_lock` zero drift,
             authoritative lock `EVAL-LOCK-CYCLE3.json`, `eval_module_sha256` c009e38d,
             `open_count: 0` on both checks, `RAKSHAK_UNLOCK` never set, no `--split test`
             ever passed, no `*_test_*` file anywhere in `data/v2/eval/`.

## FINDING 1 — charter K-1 has fired. The cohort-residual hypothesis is dead.

             Rung 3 (cohort residual) vs Rung 2 (LightGBM incumbent), relative PR-AUC:

               seed 42   0.83404 -> 0.85781   +2.850%
               seed 43   0.84296 -> 0.85203   +1.075%
               seed 44   0.85105 -> 0.86556   +1.706%
               seed 45   0.82865 -> 0.85510   +3.192%
               seed 46   0.83570 -> 0.84281   +0.851%

             **Mean +1.93% relative. Range +0.85% to +3.19%. All five seeds under the
             declared 5% bar, none even reaching half of it.**

             Charter K-1: *"Rung 3 fails to beat Rung 2 by >=5% relative PR-AUC -> the
             cohort-residual hypothesis for separating adversarial from platform drift is
             dead. Report it with the number. Do not add features to rescue it."*
             No features were added. No retuning was done. One `HParams`, one seed threaded
             identically across Rungs 2/3/4, as FR-031 requires so the delta is attributable
             to the residual columns and nothing else. Feature sets verified to differ by
             exactly the 21 `r_*` residual columns and nothing more.

             Savings agrees: 0.4179 -> 0.4318, +3.3% relative. A small, real, economically
             minor effect that does not change the verdict.

             **THREE INDEPENDENT MEASUREMENTS NOW CONVERGE ON THE SAME ANSWER**, taken on
             different data by different lanes on different days:
               - Gate G5: the residual cuts the worst confounder window 62% (+7.07pp ->
                 +2.70pp) and still misses the +2pp allowance. RED.
               - T-0121's revived P2 test: the residual cuts the alert rate 14.3% against a
                 15% bar. Misses by 0.7pp. RED, left red.
               - K-1 here: +1.93% against a 5% bar.
             The residual mechanism works — it cancels common-mode exactly, median residual
             0.000 to three decimals — and it under-delivers against every bar declared for
             it. That is a coherent, publishable falsification and it is the figure the
             charter asked for.

## FINDING 2 — no trained rung beats a trivial volume ranker on savings. None of them.

             | rung | PR-AUC | savings | P@K | ECE | beats floors |
             |---|---|---|---|---|---|
             | volume_rank (FLOOR) | 0.2169 | **0.6017** | 0.5714 | 0.4866 | — |
             | rung1 rules | 0.2843 | 0.1147 | 0.0606 | 0.0184 | NO |
             | rung2 lgbm | 0.8385 | 0.4179 | 0.8574 | 0.0079 | NO |
             | rung3 cohort | 0.8547 | 0.4318 | 0.8519 | 0.0080 | NO |
             | rung4 cost | 0.8542 | 0.4347 | 0.3889 | 0.0435 | NO |

             `floor_fail: ['volume_rank']` on **every rung, every seed**. Rank-by-GMV
             captures more of the exposure-weighted savings pool than any trained model,
             while losing to them massively on ranking (0.22 vs 0.83-0.87), on precision@K,
             and on calibration (ECE 0.487 vs 0.008). Seed spread is tight (Rung 2 savings
             0.406-0.435) and the floor is identical across every row because it is computed
             on the same truth and population. This is not noise.

             T-0113's own acceptance criterion says "beats all four floors on validation —
             if it does not, that is the finding." It does not. That is the finding.

## FINDING 3 — Rung 4 (cost-in-loss) makes it worse, and that was the fix for Finding 2.

             Cost-reweighting was the designed answer to a savings gap. It moved savings
             0.4179 -> 0.4347, nowhere near the 0.6016 floor, and it cost:
               - ECE 0.0079 -> 0.0435, five to six times worse
               - precision@K 0.857 -> 0.389 and recall@K 0.310 -> 0.140, both roughly halved
               - **median TTD to infinity on all five seeds** — more than half of true
                 positives are never alerted at all, where Rungs 2/3 have finite medians
                 under the identical window
             It concentrates alerts on a few high-loss merchants at the expense of broad
             recall, without closing the gap to the heuristic it was built to beat.

Surprised:   The volume-rank result. A one-line heuristic — sort merchants by GMV — out-earns
             a tuned LightGBM, a cohort-residual model and a cost-aware model on the metric
             the project cares most about, on every seed, while being catastrophically
             worse on every other metric. The savings pool is exposure-weighted, so a model
             that ranks by probability of fraud rather than by expected loss leaves most of
             the money on the table. Rung 4 was supposed to be exactly that correction and
             it did not work.

Broke:       Nothing. Three lock checks clean, door never opened.

Decided:     **No adoption claim arises from this run at all** — not because of NFR-04, but
             because no rung clears the volume_rank savings floor. The NFR-04 overrun
             (state_bytes_p99 9,716 B vs 4,096 B, 2.37x, carried in every result row) would
             have made any adoption conditional anyway; it is moot here.

Numbers:     Capacity K=15, 3,036 val merchants, 182,160 rows, prevalence 0.0135.
             Validation split only. No test-split number exists anywhere in this repo.

## TTD, detection rates and recall-by-typology are NOT interpretable on this split

             Verified against `ground_truth.parquet` rather than assumed: **all 294 labelled
             onsets fall in [30, 217], median 108** — strictly before the validation window
             [240, 299] opens. `time_to_detection` computes `alert_day - onset_day` and an
             alert cannot occur before day 240, so TTD is floored near `240 - onset` (>=132
             for the median merchant) no matter how fast a model would have reacted. Only
             4 of 294 merchants have onset >= 210, so a "detected within 30 days of onset"
             hit is geometrically possible for a handful at most — which is why
             `detection_rate_d7/d14/d30` read 0.000 for **every** rung including the floors.
             `recall_by_typology` reads 0.0 across every typology for the same reason.

             **This is window/onset geometry, not model responsiveness, and it must not be
             reported as the latter.** The one exception is Rung 4's median TTD going to
             infinity: that IS a genuine within-run signal about recall collapsing below
             50%, because Rungs 2/3 have finite medians under the identical window.

Next:        T-0116 (open the test split, once) and T-0117 (report). Both findings above are
             validation-side and stand independent of the test split. The report's headline
             is now written for it: the central hypothesis is falsified, and the ladder's own
             floor beats the ladder.

---

## FIX | Two red tests from the rung 5/6/7 lane, and a property test that was testing the repo
Date:        2026-09-01  ·  Session:  2
Status:      DONE. Both agent-introduced failures closed. Two PRE-EXISTING reds left red,
             deliberately — see below.

Built:       `src/rakshak/score_rung7.py`. Rung 7's runner moved out of `explain/`, which it
             should never have entered: it fits the HSMM, so it imports
             `rakshak.models.rung7_hsmm`, and
             `test_explain_registry.py::test_the_explain_package_does_not_import_the_models_package`
             refuses that from anywhere under `explain/`. The file was split along the line
             the wall already draws — explainer behind it, runner outside it, beside
             `score_rung5.py` and `score_rung6.py` where the eval side is allowed to see
             both halves. `explain/hsmm_onset.py` keeps `HsmmOnsetExplainer` and
             `first_change_point`, which need a **decoder**, not the HSMM class, and now
             take one structurally as a `Decoder` Protocol. `STATE_NAMES` is injected by the
             runner rather than imported, for the same reason.

Verified:    No number moved. The estimator, the fit pool, the pool RNG seed (20260901) and
             the artifact schema are byte-for-byte what they were; the refactor is a move
             plus a Protocol. `ruff` and `mypy --strict` clean over 47 source files.

Surprised:   **`test_a_row_scored_under_a_superseded_cycle_is_recorded_not_refused` was not
             testing the property in its own name.** It read the LIVE `data/v2/eval/` and
             asserted every row came back `cycles == [1, 2]` — which silently encoded
             "nothing in this repo has been scored under cycle 3 yet". True the day it was
             written; false the moment Rungs 2-6 were rescored under cycle 3. **So the test
             failed for the repo doing exactly what cycle 3 was sealed to let it do.** The
             assertion was about the directory's contents wearing the costume of an
             assertion about the chain.

             Same family as T-120's parity finding and T-0121b's dead fixture, and the third
             instance this cycle: a check whose subject quietly became something other than
             what its name says. The fix is a controlled results dir, like its two siblings
             already used, plus a separate live-dir test asserting the weaker thing the live
             dir IS entitled to assert — that no committed row has drifted off the chain
             entirely, whichever cycle it belongs to.

Decided:     **Two pre-existing reds NOT touched, and neither is mine to close.**
             1. `test_cohort.py::test_what_the_cohort_residual_actually_does_under_p2` —
                red before this lane started, documented in T-0121b, missing a 15% bar by
                0.7pp. Relaxing 0.85 -> 0.86 is the move this project forbids.
             2. `tests/parity/test_tier2_parity.py` — two ERRORS, not failures: the fixture
                shrinks the population to 45 days and leaves the 365-day split boundaries
                behind, so `ConfigError` fires before anything runs. **It has not executed
                since the geometry moved.** Identical to T-0121b's defect. Repairing the
                fixture may expose a real parity failure, which is a finding needing the
                lead two days from freeze — raised, not patched around.

Numbers:     `tests/unit` + `tests/parity`: 2 failures remaining, both pre-existing, 0 from
             this lane. Wall tests green (30 passed).

Next:        The parity fixture (defect 2 above) needs an owner before T-0116.

---

## POST-CYCLE-4 | The sweep that was never run, and the half-gate it left behind

Built:       `scripts/cost_sweep.py` — `sweep_cost_asymmetry` over the ladder's committed
             decisions, five seeds, five declared ratios, both exposure arms, nothing
             refitted and no locked file touched. Three tables, because one moves two
             things at once: A (rungs on the actions they take), B (every policy as a raw
             REVIEW-only ranking) and C (arm B through the decision layer with HOLD made
             unreachable via `hold_expected_loss_floor_inr = inf`). `cycle4_verdict.py`
             §3a; `LIMITATIONS.md` §10; `eval/report.py` §4 reading the artefact;
             `tests/unit/test_cost_sweep.py`; roster corrected against the ladder.

Surprised:   **Three things, in ascending order of how much they should have been caught
             earlier.**

             1. `sweep_cost_asymmetry` has been in the tree, unit-tested, since T-132 with
                no artefact and no results section. `docs/results_v2.md` §4 said "**The
                sweep was not run**" — correctly — and would have gone on saying it after
                the sweep ran, in a graded artefact, with nothing failing. That is v1's
                `results/ablations.md:94` defect exactly, a second time, in the same repo,
                after it was written up as a lesson. The fix is not the paragraph; it is
                that the section now reads the artefact and cannot assert its absence.

             2. **`PRE-REGISTRATION-CYCLE4` §5 condition 1 is a conjunction and only one
                conjunct was ever computed.** "≥ 4/5 seeds AND ≥ 4/5 sweep ratios" — the
                ratio half had no input, so `cycle4_verdict.py` silently evaluated the
                gate on half its terms. Nobody noticed because the half that WAS computed
                failed, and a failed gate stops the reader. Had it passed on seeds, the
                test split could have opened on a gate that was never fully checked. The
                general form is worth keeping: **a conjunction reported as one verdict
                hides which conjuncts were evaluated.**

             3. **Table B inverted the hypothesis it was built to test.** The expectation
                going in was that the 33× floor/rung pricing asymmetry flatters the rungs.
                Priced identically — every policy as a raw REVIEW-only ranking — every rung
                lands between +0.2349 and +0.2589 against `volume_rank`'s +0.5240, and the
                best rupee-ranker among them is **Rung 1, the rule engine**. The LightGBM
                rungs rank fraud at PR-AUC 0.73–0.77 against Rung 1's 0.30 and capture
                fewer rupees. So the asymmetry does not flatter the rungs into a win; the
                decision layer carries them, and the asymmetry is worth ~45% of that.

Broke:       Nothing, and that was checked rather than assumed: `docs/results_v2.parquet`
             is byte-identical, `git diff` on `results_v2.md` is confined to §4, and
             `report.py` is not in `EVAL_MODULES` so `eval_module_sha256` is untouched.
             One duplication was unavoidable — `sweep_cost_asymmetry` does not export the
             swept `CostParams` and the floor side has to rebuild them; adding an exported
             helper to `capacity.py` WOULD have moved the lock hash. Four duplicated lines,
             pinned by `test_cost_sweep.py::test_swept_matches_the_sweeps_own_cost_params`
             at every declared ratio.

Numbers:     Rung 4 arm B **+0.5853 → +0.6001** across ratios 0.01–100, best rung at every
             one, above the `volume_rank` floor (+0.5240) at 5/5. Margin decomposition at
             ratio 0.01: **+0.0740** as scored, **+0.0403** with HOLD unreachable (still
             5/5 above the floor), **−0.2892** as a raw ranking. Shipped cost matrix sits
             at ratio **0.154** — inside the grid. Gate: **0/5 ratios ≥ 0.7017**, FAIL,
             matching the seed half. `ruff` and `mypy --strict` clean on 49 source files.

Next:        #50 (T-0117), the report. Unblocked — #49 closed as superseded. Remaining:
             v1 beside v3 as the trajectory, per-seed spread beside every pooled headline,
             clean-clone `make all`. Recommended cut: #58/#59/#65/#66.
