<!-- HEAD
FILE:     11-tickets/BOARD.md
PHASE:    4 — TICKET
UPDATED:  2026-08-31
STATUS:   active
SUMMARY:  The dependency DAG, critical path, 48-hour block schedule, and the full ticket
          set for the Rakshak v2 sprint. Tickets are sequenced by risk retirement, not by
          comfort. One ticket per Claude Code session. Each ticket's Context line names
          the exact spec sections to load — load nothing else.
OPEN:     none. Confirm charter §10 assumptions before starting T-100.
-->

# 11 — Ticket Board (v2 sprint)

## How to run this board

One ticket per Claude Code session. Every session:

1. `cat project-context/STATE.md`
2. Read the ticket below. Load **only** the files its `Context:` line names.
3. Build. Write the test in `Test:`. Run it.
4. Check `Done when:`. If it does not pass, the ticket is not done — do not mark it
   complete with "mostly working".
5. Append to `docs/LOGBOOK.md`: built / surprised / broke.
6. Update `STATE.md` and this board's status column. Commit.
7. **Stop.** Name the next ticket. Do not start it.

A ticket that cannot fit in one session is a phase — split it and say so.

---

## Critical path

```
T-100 ─► T-101 ─► T-102 ─┬─► T-110 ─► T-111 ─► T-112 ─► T-113 ─► T-114 ─► T-115 ─► T-116
                         │                                                          │
                         └─► T-120 ─► T-121 ────────────────────────────────────────┤
                                                                                    ▼
                                    T-130 ─► T-131 ─► T-132 ─► T-133 [FREEZE] ─► T-140
                                                                                    │
                                                    T-141 ─► T-142 ─► T-143 ◄───────┘
                                                                        │
                                                                        ▼
                                                              T-150 ─► T-151 [UNLOCK]
```

**Hard ordering constraint:** T-133 (EVAL-LOCK written) must complete before T-141
(first trained model). T-151 is the only ticket permitted to open the test split.

**Parallel lanes** open after T-102, because `schemas.py` and `spec.py` freeze the
boundary: Lane A (generator, T-110..T-116) and Lane B (features, T-120..T-124) can
proceed in separate sessions. Lane C (harness, T-130..T-133) needs only `schemas.py`, so
it can also run in parallel — and running it early is *preferable*, because a harness
frozen before the generator is finished is even harder to accuse of hindsight.

---

## 48-hour block schedule

| Block | Hours | Tickets | Gate to pass before moving on |
|---|---|---|---|
| **1** | 4 | T-100, T-101, T-102 | `make test` green; `09-interfaces.md` frozen |
| **2** | 6 | T-110, T-111, T-112 | G3 determinism green |
| **3** | 5 | T-113, T-114, T-115, T-116 | **G1–G5 all green** |
| **4** | 6 | T-120, T-121, T-122 | `make parity` green (NFR-08) |
| **5** | 4 | T-130, T-131, T-132, T-133 | **EVAL-LOCK written and committed** |
| **6** | 5 | T-140, T-141, T-142, T-143 | all rungs trained on train+val only |
| **7** | 4 | T-150, T-151, T-152 | `make all` from clean clone; report generated |
| — | 4 | buffer | do not spend this early |

Blocks 2–3 and 4 can overlap across sessions if you are running Lane A and Lane B
alternately. The schedule assumes you will not.

---

## Status

Updated at the end of every session.

| Ticket | Lane | Status |
|---|---|---|
| T-100 | foundation | ✅ **DONE** 2026-08-31 — scaffold, pins, Makefile + `make.ps1`, CI clean-clone job, `v1-frozen` tag |
| T-101 | foundation | ✅ **DONE** 2026-08-31 — `schemas.py` + `store.py`. **09-interfaces.md FROZEN.** Two tz bugs found by the property suite, see logbook |
| T-102 | foundation | ✅ **DONE** 2026-08-31 — `FeatureSpec` dual runner, `MerchantState`, registry + NFR-04 budget, `assert_parity` harness |
| T-110 … T-116 | A generator | ◐ running |
| T-120 … T-122 | B features | ☐ blocked on T-112 |
| T-130 … T-133 | C harness | ◐ running |
| T-140 … T-143 | D rungs | ☐ blocked on T-133 (FREEZE) |
| T-150 … T-152 | E hardening | ☐ not started |

**Block 1 gate passed:** `ruff` clean, `mypy --strict` clean, 70 tests pass.

---

# Lane 0 — Foundation

```
T-100 | Scaffold the repo, pin the environment, and wire the Makefile
  Retires:    "the v1 make eval reproducibility gap" — the highest-scoring risk
  Depends on: none
  Lane:       critical-path
  Context:    CLAUDE.md §Tech Stack, §Project Structure, §Development Commands
  Budget:     light (<30K)
  Files:      pyproject.toml, Makefile, .github/workflows/ci.yml, src/rakshak/__init__.py,
              tests/conftest.py, .gitignore, docs/LOGBOOK.md
  Done when:  `uv sync && make lint && make test` passes on a fresh clone with zero
              source modules beyond __init__; CI has a `clean-clone` job that runs
              `git clone . /tmp/x && cd /tmp/x && uv sync && make all`
  Test:       tests/unit/test_smoke.py — imports rakshak, asserts __version__
  Rollback:   delete the directory; nothing depends on it yet
  Note:       `git tag v1-frozen` on the current HEAD before the first commit of this
              ticket. v1 must remain retrievable and untouched.
```

```
T-101 | Implement schemas.py and the duckdb event store with point-in-time queries
  Retires:    "modules will exchange ad-hoc dicts and drift apart"
  Depends on: T-100
  Lane:       critical-path
  Context:    09-interfaces.md (entire file — it is the ticket)
  Budget:     medium (<60K)
  Files:      src/rakshak/schemas.py (new), src/rakshak/store.py (new),
              tests/unit/test_schemas.py (new)
  Done when:  every table and enum in 09-interfaces.md exists as a dataclass/StrEnum;
              store.query_events(merchant_id, as_of) provably returns zero rows with
              event_time > as_of
  Test:       tests/unit/test_point_in_time.py — property test with hypothesis over
              random as_of values, asserting no future row ever appears
  Rollback:   single file revert
  Note:       FREEZE 09-interfaces.md at the end of this ticket. Later changes are a
              DESCEND, not an edit.
```

```
T-102 | Build the FeatureSpec dual-runner framework and the parity test harness
  Retires:    "features will be written batch-only and will not be servable" — the
              assumption that decides whether this system is real
  Depends on: T-101
  Lane:       critical-path
  Context:    09-interfaces.md §7, §8; 07-feature-register.md §"Two rules"
  Budget:     medium (<60K)
  Files:      src/rakshak/features/spec.py (new), src/rakshak/features/state.py (new),
              src/rakshak/features/registry.py (new), tests/parity/conftest.py (new)
  Done when:  a trivial reference feature (daily txn count) implements update() and
              batch(); the parity harness replays a synthetic stream and asserts
              agreement ≤1e-9 at every epoch; registry sums declared state_bytes and
              raises at import if the total exceeds 4096
  Test:       tests/parity/test_reference_feature.py
  Rollback:   revert; Lane A is unaffected
```

---

# Lane A — Generator (critical path)

```
T-110 | Persona and typology config dataclasses + scenario YAML loader
  Retires:    "magic numbers will end up in src/"
  Depends on: T-102
  Lane:       critical-path
  Context:    08-generator-v2-spec.md §2, §3, §8
  Budget:     light (<30K)
  Files:      src/rakshak/generator/config.py (new), configs/scenario_v2.yaml (new),
              tests/unit/test_config.py (new)
  Done when:  scenario_v2.yaml loads into typed dataclasses; persona shares validate to
              1.0 ± 1e-9; an invalid config raises with a message naming the field
  Test:       tests/unit/test_config.py
  Rollback:   single file revert
```

```
T-111 | Marked point process: NB daily counts with Fano calibration + Hawkes overlay
  Retires:    "Poisson-ish arrivals misspecified the emissions" — a named v1 cause
  Depends on: T-110
  Lane:       critical-path
  Context:    08-generator-v2-spec.md §1; 06-requirements-v2.md FR-003
  Budget:     medium (<60K)
  Files:      src/rakshak/generator/arrivals.py (new), tests/unit/test_arrivals.py (new)
  Done when:  realised population Fano factor = target ± 1.0 at target ∈ {1, 5, 12.25};
              Hawkes overlay measurably raises short-lag autocorrelation; every function
              takes rng: np.random.Generator
  Test:       tests/unit/test_arrivals.py — Fano assertion across three targets
  Rollback:   single file revert
```

```
T-112 | Implement legitimate personas L1–L8
  Retires:    "the negatives are too easy, so false-positive cost is fictional"
  Depends on: T-111
  Lane:       critical-path
  Context:    08-generator-v2-spec.md §2; 07-feature-register.md (hard-negative notes)
  Budget:     medium (<60K)
  Files:      src/rakshak/generator/personas.py (new), src/rakshak/generator/engine.py (new),
              tests/unit/test_personas.py (new)
  Done when:  all 8 personas emit streams; L3 growth is separable from R1 only by
              convexity (assert: linear fit R² > 0.9 for L3 GMV, < 0.7 for R1);
              L5 inter-arrival CV < 0.3; L8 refund rate > 0.15
  Test:       tests/unit/test_personas.py — one signature assertion per persona
  Rollback:   revert personas.py; engine keeps a single default persona
```

```
T-113 | Implement risk typologies R1–R9 with drift_onset_at and true_loss_amount
  Retires:    "fraud is one undifferentiated class, so per-typology failure is invisible"
  Depends on: T-112
  Lane:       critical-path
  Context:    08-generator-v2-spec.md §3; 09-interfaces.md §5
  Budget:     medium (<60K)
  Files:      src/rakshak/generator/typologies.py (new), tests/unit/test_typologies.py (new)
  Done when:  all 9 typologies emit; every fraud merchant has drift_onset_at >
              onboarded_at and a positive true_loss_amount_inr; R2's weekly change stays
              under 1σ of its own baseline (assert it — that is what makes R2 hard)
  Test:       tests/unit/test_typologies.py — R2 slow-ramp σ assertion is the key one
  Rollback:   single file revert
```

```
T-114 | Implement the platform confounder layer P1–P6
  Retires:    "we cannot test adversarial-vs-natural drift separation" — the sprint's
              central hypothesis is untestable without this
  Depends on: T-113
  Lane:       critical-path
  Context:    08-generator-v2-spec.md §4
  Budget:     medium (<60K)
  Files:      src/rakshak/generator/confounders.py (new), tests/unit/test_confounders.py (new)
  Done when:  confounders apply as an independent multiplicative layer that persona and
              typology code cannot see; with prevalence=0 and confounders on, the
              population mean |z| for the affected feature exceeds 1.0 inside every
              event window; persona_sensitivity varies effect size across personas
  Test:       tests/unit/test_confounders.py
  Rollback:   set confounders.enabled=false; the rest of the generator is unaffected
```

```
T-115 | Delayed, noisy, and censored label emission
  Retires:    "instant labels made v1 measure a system that cannot exist"
  Depends on: T-113
  Lane:       parallel-A
  Context:    08-generator-v2-spec.md §6; 09-interfaces.md §4
  Budget:     light (<30K)
  Files:      src/rakshak/generator/labels.py (new), tests/unit/test_labels.py (new)
  Done when:  all four label states are produced; the invariant
              label_available_at > label_event_at >= drift_onset_at holds on every row;
              unreported_rate and spurious_chargeback_rate are honoured within sampling
              error over 10k merchants
  Test:       tests/unit/test_labels.py — hypothesis property test on the invariant
  Rollback:   single file revert
```

```
T-116 | Implement parity gates G1–G5 as tests
  Retires:    "the generator is fiction and we would not know" — charter K-3, K-4
  Depends on: T-114, T-115
  Lane:       critical-path
  Context:    08-generator-v2-spec.md §7; 10-eval-harness-spec.md §7
  Budget:     heavy (<90K)
  Files:      tests/gates/test_g1_marginal_parity.py, test_g2_baseline_transfer.py,
              test_g3_determinism.py, test_g4_no_leakage.py, test_g5_confounder_null.py
              (all new); src/rakshak/eval/baf_adapter.py (new)
  Done when:  `make gates` runs all five and prints a GREEN/RED verdict per gate; G3 and
              G4 are green; G1, G2, G5 report their statistic even if red
  Test:       the gates are the test
  Rollback:   none — this ticket only adds tests
  Note:       G5 needs a trained model, so on first pass run it against Rung 1 (rules).
              Re-run it in T-151 against Rungs 2 and 3 to produce the headline figure.
  Warning:    this is the largest ticket on the board. If it does not fit one session,
              split G1/G2 (external anchor) from G3/G4/G5 (internal) and log the split.
```

---

# Lane B — Features

```
T-120 | Implement the Tier-1 feature set (F1, F2 partial, F5, F6, F7, F8, F9)
  Retires:    "the drift signal is not actually computable online"
  Depends on: T-102, T-112
  Lane:       parallel-B
  Context:    07-feature-register.md §F1, F2, F5, F6, F7, F8, F9; 09-interfaces.md §7, §8
  Budget:     heavy (<90K)
  Files:      src/rakshak/features/tier1.py (new), tests/parity/test_tier1_parity.py (new)
  Done when:  all 23 T1 features implement update() and batch(); parity ≤1e-9 for every
              one; total declared state_bytes for T1 < 1024
  Test:       tests/parity/test_tier1_parity.py — parametrised over the whole registry
  Rollback:   revert tier1.py; Rung 1 rules can run on a subset
  Warning:    23 features is a lot for one session. If needed, split by family group
              (F1+F2 / F5+F6 / F7+F8+F9) into T-120a/b/c and log the split.
```

```
T-121 | Implement cohort assignment, EB shrinkage, and the cohort-residual layer
  Retires:    the sprint's central hypothesis — charter K-1 fires here
  Depends on: T-120
  Lane:       critical-path
  Context:    07-feature-register.md §"The cohort-residual layer"; 06-requirements-v2.md
              FR-012, FR-013
  Budget:     medium (<60K)
  Files:      src/rakshak/features/cohort.py (new), tests/unit/test_cohort.py (new),
              tests/parity/test_cohort_parity.py (new)
  Done when:  cohorts assign with the 30-member backoff chain; leave-one-out median is
              computed without an O(N²) recompute; residuals exist for all flagged T1
              features; and under prevalence=0 with confounder P2 active, ALL THREE of:
                (a) the common mode is removed — |median residual| < 0.05 on both the P2
                    day and a null day, while median raw z > 1.0 on the P2 day;
                (b) the alert rate at |·| > 3 falls to < 85% of the raw-z alert rate;
                (c) it does NOT fall to the null-day level — residual alert rate stays
                    > 5x the null-day residual alert rate. This clause is deliberately
                    an admission: the residual is a PARTIAL defence against P2 and G5
                    can still be red on a single feature. A criterion that could only
                    ever report success is not a criterion.
  Test:       tests/unit/test_cohort.py — the three-part P2 assertion is the ticket

  AMENDED 2026-08-31, after T-121 measured the original clause and found it unreachable.
  The original read "mean |residual| < 0.25 while mean |raw z| > 1.0". A z-score has unit
  scale by definition and E|N(0,1)| = 0.798, so subtracting a cohort median cannot bring
  the mean absolute value below ~0.6 unless every merchant in a cohort agrees to within
  0.3σ. Measured on a day with NO confounder at all, mean |residual| was already 0.558.
  A perfect residual layer fails that clause and so does a useless one — it measured
  nothing. The original assertion is preserved verbatim in the test as
  xfail(strict=True) with the arithmetic in its reason; it was not deleted and it was not
  weakened to go green.
  **K-1's verdict is NOT rendered here.** It is rendered at T-142, on the Rung 3 vs Rung 2
  validation delta, which is where the board always put it. This ticket proves the
  mechanism works; T-142 decides whether the mechanism is worth anything.
  Rollback:   Rung 3 degenerates to Rung 2; nothing else breaks
```

```
T-122 | Implement Tier-2 features (histogram/divergence/sketch based)
  Retires:    "the expensive tier does not earn its compute"
  Depends on: T-120
  Lane:       parallel-B
  Context:    07-feature-register.md §F2, F3, F4 (T2 rows only)
  Budget:     medium (<60K)
  Files:      src/rakshak/features/tier2.py (new), tests/parity/test_tier2_parity.py (new)
  Done when:  9 T2 features implement both runners with parity ≤1e-9; the 32-bin log
              histogram and HLL sketches stay inside their declared state_bytes
  Test:       tests/parity/test_tier2_parity.py
  Rollback:   cut to T1 only; the cascade degrades to a single stage
  Note:       P2 in the cut order. If Block 4 runs long, cut this and log it.
```

---

# Lane C — Eval harness (freeze before models)

```
T-130 | Split engine: temporal + merchant-group + label-availability
  Retires:    "the harness silently hands the model labels it could not have had"
  Depends on: T-101
  Lane:       critical-path
  Context:    10-eval-harness-spec.md §1; 06-requirements-v2.md FR-020
  Budget:     medium (<60K)
  Files:      src/rakshak/eval/splits.py (new), tests/unit/test_splits.py (new)
  Done when:  available_labels(as_of) is the only path to the label table (asserted by
              an AST scan); no merchant_id spans two splits; censored merchants are
              excluded and counted
  Test:       tests/unit/test_splits.py + tests/gates/test_label_access.py
  Rollback:   single file revert
```

```
T-131 | Metric suite: PR-AUC, savings + four floors, TTD, P@K, ECE, stability,
        per-typology recall
  Retires:    "PR-AUC alone hides latency and floor failures" — v1's AP-06
  Depends on: T-130
  Lane:       critical-path
  Context:    10-eval-harness-spec.md §2; 09-interfaces.md §11
  Budget:     heavy (<90K)
  Files:      src/rakshak/eval/metrics.py (new), tests/unit/test_metrics.py (new)
  Done when:  every field in the EvalResult schema is produced; savings rows always
              carry all four floors; a rung failing a floor is flagged FLOOR-FAIL;
              TTD handles censoring
  Test:       tests/unit/test_metrics.py — includes a synthetic case where random beats
              the model, asserting FLOOR-FAIL fires
  Rollback:   single file revert
```

```
T-132 | Oracle knapsack + capacity-constrained action selector + cost-asymmetry sweep
  Retires:    "results are unanchored absolutes"
  Depends on: T-131
  Lane:       critical-path
  Context:    10-eval-harness-spec.md §3, §4; 08-generator-v2-spec.md §8 (costs block)
  Budget:     medium (<60K)
  Files:      src/rakshak/eval/oracle.py (new), src/rakshak/eval/capacity.py (new),
              tests/unit/test_oracle.py, tests/unit/test_capacity.py (new)
  Done when:  oracle_savings >= any rung's savings (asserted); alerts_per_day <= K
              always; the sweep produces a ranking at all five asymmetry ratios
  Test:       tests/unit/test_oracle.py — the "rung beats oracle ⇒ leakage" assertion
  Rollback:   single file revert
```

```
T-133 | Write and commit EVAL-LOCK.json  ⚠️ FREEZE POINT
  Retires:    "we cannot prove the harness predated the models"
  Depends on: T-132
  Lane:       critical-path
  Context:    10-eval-harness-spec.md §6; 00-charter-v2.md §6
  Budget:     light (<30K)
  Files:      src/rakshak/eval/lock.py (new), EVAL-LOCK.json (new),
              tests/unit/test_lock.py (new)
  Done when:  the lock is written and COMMITTED with open_count=0; `make eval
              --split test` refuses without RAKSHAK_UNLOCK=1; a modified eval module
              causes a hash-mismatch hard fail
  Test:       tests/unit/test_lock.py
  Rollback:   NONE. This is a one-way door and it is meant to be. Do not proceed until
              the metric suite is complete — you cannot add a metric after the freeze
              without invalidating the claim.
```

---

# Lane D — Model rungs

```
T-140 | Rung 0 floors and Rung 1 static rule engine
  Retires:    "the sophisticated method might not beat the dumb one"
  Depends on: T-133
  Lane:       critical-path
  Context:    06-requirements-v2.md §A, FR-030; 10-eval-harness-spec.md §2
  Budget:     light (<30K)
  Files:      src/rakshak/models/rung0_floors.py, rung1_rules.py (new),
              tests/unit/test_rungs_0_1.py (new)
  Done when:  all four floors and the rule engine score on the VALIDATION split and
              produce complete EvalResult rows
  Test:       tests/unit/test_rungs_0_1.py
  Rollback:   single file revert
```

```
T-141 | Rung 2 — LightGBM on windowed aggregates (the incumbent, the bar)
  Retires:    "we do not know what the real bar is on the corrected generator"
  Depends on: T-140, T-120
  Lane:       critical-path
  Context:    06-requirements-v2.md FR-030, §D; 07-feature-register.md (T1 rows)
  Budget:     medium (<60K)
  Files:      src/rakshak/models/rung2_lgbm.py (new), tests/unit/test_rung2.py (new)
  Done when:  trains on train+val ONLY, in ≤20 min on 4 cores; model ≤20 MB; produces
              calibrated scores (ECE reported); beats all four floors on validation
  Test:       tests/unit/test_rung2.py + tests/perf/test_train_budget.py
  Rollback:   single file revert
  Warning:    do not touch the test split. Tune on validation. This is the ticket where
              the temptation is highest.
```

```
T-142 | Rung 3 — LightGBM + cohort-residual features  ⚠️ the K-1 test
  Retires:    the sprint's hypothesis. This ticket either validates v2 or kills it.
  Depends on: T-141, T-121
  Lane:       critical-path
  Context:    06-requirements-v2.md FR-031; 07-feature-register.md §cohort-residual
  Budget:     medium (<60K)
  Files:      src/rakshak/models/rung3_cohort.py (new), tests/unit/test_rung3.py (new)
  Done when:  identical to Rung 2 in every respect except the added residual features —
              same hyperparameters, same seed, same splits — so the delta is
              attributable to one variable; validation delta reported
  Test:       tests/unit/test_rung3.py asserts the feature sets differ by exactly the
              residual columns
  Rollback:   Rung 2 remains the best rung; log K-1 as fired
  Note:       if the validation delta is under 5% relative, charter K-1 has fired. Write
              it in LIMITATIONS.md with the number and do NOT add features to rescue it.
              A clean falsification is the result.
```

```
T-143 | Rung 4 — instance-dependent cost inside the training objective   [P2, cut first]
  Depends on: T-142
  Lane:       parallel-D
  Context:    06-requirements-v2.md FR-032; 08-generator-v2-spec.md §8 costs
  Budget:     medium (<60K)
  Files:      src/rakshak/models/rung4_cost.py (new)
  Done when:  a custom LightGBM objective weights each instance by its true cost;
              savings improves over Rung 3 on validation, or the negative result is
              logged
  Rollback:   cut entirely; it is the first thing to drop
```

---

# Lane E — Hardening, unlock, report

```
T-150 | Perf budget assertions as CI gates
  Retires:    "the latency NFRs were measured once and hoped for"
  Depends on: T-142
  Lane:       critical-path
  Context:    06-requirements-v2.md §C (NFR-01..NFR-06, NFR-10)
  Budget:     medium (<60K)
  Files:      tests/perf/*.py (new), src/rakshak/features/cascade.py (new)
  Done when:  every NFR with a number is an assertion that fails CI when violated; the
              three-stage cascade is implemented and NFR-03 passes
  Test:       `make perf`
  Rollback:   cascade off → single-stage; NFR-03 will fail and that gets logged
```

```
T-151 | Open the test split. Once.  ⚠️ ONE-WAY DOOR
  Retires:    nothing. This is the measurement.
  Depends on: T-150, and every rung final
  Lane:       critical-path
  Context:    10-eval-harness-spec.md §6, §7; 00-charter-v2.md §2, §3
  Budget:     light (<30K)
  Files:      EVAL-LOCK.json (open_count 0→1), docs/results_v2.parquet
  Done when:  every rung is scored on test in a SINGLE run; open_count is 1 and
              committed; the G5 figure is regenerated against Rungs 2 and 3
  Test:       tests/unit/test_lock.py asserts open_count == 1
  Rollback:   NONE. After this ticket, no model may be modified. If a rung looks bad,
              that is the finding — write it down.
  Checklist before running:
    ☐ every rung trained on train+val only
    ☐ `make gates` green
    ☐ `make parity` green
    ☐ `make perf` green
    ☐ adoption margins in EVAL-LOCK unchanged since Block 5
    ☐ working tree clean and committed
```

```
T-152 | Generate the report, write LIMITATIONS.md, verify clean-clone reproducibility
  Retires:    charter K-5 — the disqualification risk
  Depends on: T-151
  Lane:       critical-path
  Context:    10-eval-harness-spec.md §8; 00-charter-v2.md §3
  Budget:     medium (<60K)
  Files:      src/rakshak/eval/report.py (new), docs/results_v2.md, LIMITATIONS.md,
              README.md
  Done when:  `git clone` to a fresh directory, `uv sync`, `make all` — passes end to
              end; the report front page carries the provenance header; LIMITATIONS.md
              names every failed rung, every fired kill criterion, and every cut feature
              with its number
  Test:       CI `clean-clone` job green
  Rollback:   none
```

---

## Deferred — do not start (spec only, for T-0018 §Future Work)

| Rung | What | Why deferred |
|---|---|---|
| 5 | Temporal attention-MIL over (payer, day) capsules | Right reformulation, 2+ days of work; spec'd in `14-lit-survey-v2.md` §Family A |
| 6 | Mondrian conformal risk control on the 3-action decision | Highest-value deferral — a distribution-free bound on the false-hold rate is the single strongest decision-layer claim available. Spec it, do not build it. |
| 7 | HSMM with negative-binomial emissions, explanation-only | The honest repair of the v1 HMM. Scored on explanation quality, never on PR-AUC. |
| 8 | Neural TPP + goodness-of-fit statistic | Continuous-time anomaly detection as a hypothesis test |
| — | Delayed-label drift detection under an upper-bounded window | The genuinely novel research direction. Named as unstudied in the standing research agenda for this domain. This is where the actual contribution lives, and it is an *evaluation* contribution — which is where this project has already shown it is strongest. |
