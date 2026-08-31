# Rakshak v2 — Post-Onboarding Merchant Risk Sentinel

> This file is auto-loaded into every Claude Code session. Read it fully. Then read
> **only** the `project-context/` sections your ticket names. Never load the whole
> context folder — it will not fit and it is not needed.

---

## Project Overview

Razorpay approves a merchant once, at onboarding (Bumblebee). It scores every
transaction in real time (Vulcan). Nothing watches the **merchant** in the weeks
after approval, while its behaviour drifts from the profile it was approved on.

Rakshak is the sentinel for that gap. Every day, for every cleared merchant, it emits
one of three actions — `PASS`, `REVIEW`, `HOLD` — under a hard analyst-capacity
budget, with a merchant-readable justification attached to every non-`PASS`.

**v2 exists because v1's measurements falsified v1's hypothesis.** A hand-written HMM
sequence layer lost to plain LightGBM on windowed aggregates by 0.3176 PR-AUC, and the
whole pipeline cleared the rule engine by only 5.9% against a self-imposed 20% gate.
Diagnosis (see `project-context/14-lit-survey-v2.md`): the evaluation ran at 20%
prevalence instead of the real ~1.5%, the generator assumed Poisson arrivals when real
counts are overdispersed (measured Fano 12.25), and the task was formulated as latent-
state inference when the labels are merchant-level.

v2 fixes the **generator and the harness first**, then re-races models against a
re-frozen bar. Until the generator is right, every model comparison is measuring the
generator.

---

## The Prime Directives

These override everything, including a ticket that appears to ask otherwise.

1. **The eval harness is frozen before any v2 model is written.** Once
   `EVAL-LOCK.json` exists, the test split is opened exactly once, at the end. If you
   find yourself tempted to peek to debug a model, you are about to destroy the most
   valuable property of this project. Debug on the validation split.
2. **v1 results are immutable.** Never edit, re-run, or "correct" any number in the v1
   retrospective. v2 is a separate harness with a separate lock. Both get reported.
3. **Ground-truth fields are radioactive.** `persona_id`, `risk_typology_id`,
   `drift_onset_at`, and anything in the `ground_truth` table must never be importable
   from `src/rakshak/features/` or `src/rakshak/models/`. A CI test enforces this.
4. **Every feature must be computable online.** If a feature cannot be maintained
   incrementally from a bounded per-merchant state object, it does not go in the
   register. This is not a performance preference — it is what makes the system real.
5. **A rung is adopted only if it beats the previous rung on the frozen eval by the
   margin declared before the run, AND meets the compute NFRs.** Otherwise it is
   reported as a negative result and dropped from the scoring path.
6. **Report the failure.** A rung that loses is a finding, not an embarrassment. Write
   it in `LIMITATIONS.md` with the number.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │  GENERATOR v2  (src/rakshak/generator/)   │
                    │  personas × typologies × confounders      │
                    │  NB/Hawkes arrivals · delayed labels      │
                    └───────────────┬──────────────────────────┘
                                    │ Transaction stream (parquet)
                                    │ + ground_truth (quarantined)
                                    ▼
   BAF (external anchor) ──►  ┌──────────────────────┐
                              │  EVENT STORE (duckdb)│  point-in-time queries only
                              └──────────┬───────────┘
                                         │
                    ┌────────────────────▼─────────────────────┐
                    │  FEATURE LAYER (src/rakshak/features/)   │
                    │  ONE FeatureSpec → two runners:          │
                    │    .batch(frame)   offline               │
                    │    .update(state,e) online               │
                    │  parity asserted to 1e-9 in CI           │
                    │  T1 cheap · T2 divergence · T3 graph     │
                    │  + COHORT-RESIDUAL layer                 │
                    └────────────────────┬─────────────────────┘
                                         │ FeatureVector(merchant, as_of)
                    ┌────────────────────▼─────────────────────┐
                    │  MODEL RUNGS (src/rakshak/models/)       │
                    │  0 floors → 1 rules → 2 LGBM →           │
                    │  3 +cohort → 4 cost-in-loss              │
                    └────────────────────┬─────────────────────┘
                                         │ score + pred_contrib reason codes
                    ┌────────────────────▼─────────────────────┐
                    │  DECISION LAYER (src/rakshak/eval/)      │
                    │  cost-aware PASS/REVIEW/HOLD under        │
                    │  analyst capacity K                       │
                    └────────────────────┬─────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │  FROZEN EVAL HARNESS  (EVAL-LOCK.json)   │
                    │  temporal + merchant-group + label-avail  │
                    │  PR-AUC · savings+floors · TTD · P@K      │
                    └──────────────────────────────────────────┘
```

The three-stage **inference cascade** is what buys the compute NFR:

| Stage | Runs on | Features | Budget |
|---|---|---|---|
| 0 — screen | every merchant, every day | T1 only (~14) | ≤ 0.5 ms |
| 1 — score | top 10% from stage 0 | T1 + T2 + cohort | ≤ 10 ms |
| 2 — explain | non-`PASS` decisions only | `pred_contrib` reason codes | ≤ 50 ms |

---

## Tech Stack & Versions

CPU-only. No GPU anywhere. No torch, no transformers, no cloud services.

| Layer | Choice | Pin | Why |
|---|---|---|---|
| Language | Python | 3.11 | polars/duckdb/lightgbm all stable here |
| Env/deps | `uv` | latest | lockfile reproducibility; `pip` is not used |
| Offline dataframes | polars | `>=1.0,<2` | lazy, columnar, out-of-core; **pandas is not used** |
| Online state | numpy | `>=2.0,<3` | plain arrays and scalars, no DataFrame in the hot path |
| Event store | duckdb | `>=1.0` | parquet-native, serverless, point-in-time SQL |
| Model | lightgbm | `>=4.3,<5` | won the v1 ablation; native `pred_contrib` |
| Stats | scipy, scikit-learn | current | KS/Wasserstein, PR-AUC, calibration |
| Tests | pytest, hypothesis | `>=8`, `>=6` | invariants + property tests |
| Lint/type | ruff, mypy | current | `ruff check` and `mypy --strict` on `src/` |
| Multi-objective | pymoo | `0.6.2` | carried from v1; optional in this sprint |
| Conformal (stretch) | crepes **or** mapie | current | Rung 6 only, cut first if time runs out |

**Explicitly rejected for this sprint** — do not introduce them: torch, transformers,
any GNN library, `hmmlearn`, TabPFN (license: v2 only, and it needs a GPU to be quick),
river (delayed labels make online learning unsound here), pandas, any cloud SDK.

---

## Project Structure

```
rakshak/
├── CLAUDE.md                  ← this file
├── Makefile                   ← every workflow is a make target
├── pyproject.toml             ← uv-managed, all pins here
├── EVAL-LOCK.json             ← written once by T-133; never hand-edited
├── LIMITATIONS.md             ← honest failures, with numbers
├── project-context/           ← specs. Load only the sections your ticket names.
│   ├── STATE.md               ← resume point. Read first, every session.
│   ├── 00-charter-v2.md
│   ├── 06-requirements-v2.md
│   ├── 07-feature-register.md
│   ├── 08-generator-v2-spec.md
│   ├── 09-interfaces.md
│   ├── 10-eval-harness-spec.md
│   └── 11-tickets/BOARD.md
├── src/rakshak/
│   ├── schemas.py             ← ALL dataclasses/enums. Single source of truth.
│   ├── generator/
│   │   ├── config.py          ← persona & typology parameter dataclasses
│   │   ├── arrivals.py        ← NB / Hawkes marked point process
│   │   ├── personas.py        ← L1–L8 legitimate merchant behaviour
│   │   ├── typologies.py      ← R1–R9 risk behaviour + drift_onset_at
│   │   ├── confounders.py     ← P1–P6 platform-wide events
│   │   ├── labels.py          ← delayed/noisy/censored label emission
│   │   └── engine.py          ← orchestrator; only public entry point
│   ├── features/
│   │   ├── spec.py            ← FeatureSpec: .batch() + .update() dual runner
│   │   ├── state.py           ← MerchantState, bounded to 4 KB
│   │   ├── registry.py        ← the register, as code
│   │   ├── tier1.py tier2.py tier3.py
│   │   └── cohort.py          ← cohort assignment + residual layer
│   ├── eval/
│   │   ├── splits.py          ← temporal + merchant-group + label-availability
│   │   ├── metrics.py         ← PR-AUC, savings+floors, TTD, P@K, ECE, stability
│   │   ├── oracle.py          ← perfect-foresight knapsack ceiling
│   │   ├── capacity.py        ← cost-aware action selection under K
│   │   ├── lock.py            ← EVAL-LOCK write/verify, open counter
│   │   └── report.py          ← results table generator
│   ├── models/
│   │   ├── rung0_floors.py rung1_rules.py rung2_lgbm.py
│   │   ├── rung3_cohort.py rung4_cost.py
│   └── cli.py                 ← typer CLI; make targets call this
├── tests/
│   ├── unit/                  ← per-module
│   ├── parity/                ← online vs offline feature agreement (NFR-08)
│   ├── gates/                 ← G1–G5 generator parity gates
│   └── perf/                  ← latency/memory budgets as assertions
├── configs/                   ← YAML scenario manifests, hashed into EVAL-LOCK
└── data/                      ← gitignored. Regenerable from seed + config.
```

---

## Development Commands

```bash
uv sync                       # install exact pinned env
make gen                      # generate the v2 dataset from configs/scenario_v2.yaml
make features                 # materialise FeatureVectors for the generated stream
make gates                    # run G1–G5 generator parity gates. MUST be green.
make parity                   # online vs offline feature agreement (NFR-08)
make perf                     # latency/memory budget assertions (NFR-01..05)
make train RUNG=2             # train one rung on train+val only
make eval RUNG=2              # score a rung. Refuses to run if EVAL-LOCK is unopened
                              #   unless RAKSHAK_UNLOCK=1 is set. Increments the counter.
make report                   # regenerate the results table into docs/
make test                     # pytest, all suites
make lint                     # ruff check + mypy --strict src/
make all                      # gen → features → gates → parity → perf → test
```

**`make all` must pass from a clean `git clone` on a fresh env.** The v1 build's single
biggest disqualification risk was `make eval` not reproducing on a clean checkout. A CI
job does exactly this on every push. Do not let it go red.

---

## Key Domain Concepts

- **Merchant** — the entity being watched. Already approved. The unit of decision.
- **Epoch** — one decision point. Daily. Every merchant gets an action every epoch.
- **Drift** — movement of a merchant's behaviour away from *its own* post-onboarding
  baseline. Always relative, never absolute. A ₹2L ticket is normal for jewellery and
  alarming for a kirana.
- **Drift onset (`drift_onset_at`)** — generator ground truth: the timestamp the
  typology began deviating. Used **only** to compute time-to-detection in eval.
- **Cohort** — `(mcc_group, gmv_decile, vintage_bucket)`, backing off to `mcc_group`
  then global when fewer than 30 members. Cold-start uses empirical-Bayes shrinkage
  toward the cohort prior (carried from v1 ADR).
- **Cohort residual** — a merchant's drift z-score minus the leave-one-out cohort
  median of the same z-score. This is how adversarial drift is separated from platform
  drift: when the whole platform moves, the residual stays near zero.
- **Confounder** — a platform-wide event (festival spike, gateway outage, fee change)
  that moves everyone's features without any fraud occurring. The generator emits these
  deliberately. Gate G5 tests that the detector does not alert on them.
- **Rung** — a numbered model in the ladder. Each must beat the one below it.
- **Typology** — a named fraud pattern (R1–R9), not a generic "fraud" label.
- **Capacity K** — analyst reviews available per day. The binding operational
  constraint. Metrics that ignore K are decoration.

---

## Coding Conventions & Patterns

- **One feature definition, two runners.** Every feature is a `FeatureSpec` subclass in
  `src/rakshak/features/` implementing both `batch(frame) -> Series` and
  `update(state, event) -> float`. Never write a feature in only one form; the parity
  test in `tests/parity/` will fail and it should.
- **All types live in `schemas.py`.** No ad-hoc dicts crossing a module boundary. If
  two modules exchange data, the shape is a dataclass in `schemas.py`.
- **Seeds are explicit and threaded.** Every stochastic function takes `rng:
  np.random.Generator` as an argument. There is no module-level global RNG and no bare
  `np.random.*` call anywhere in `src/`.
- **Point-in-time by construction.** Any query against the event store filters on
  `event_time <= as_of` AND, for labels, `label_available_at <= as_of`. There is a
  helper for this in `eval/splits.py` — use it rather than writing the filter yourself.
- **Money is float64 in whole rupees.** Not Decimal, not paise-integers. Document it
  once and be consistent.
- **Timestamps are UTC, tz-aware, nanosecond.** No naive datetimes cross a boundary.
- Config lives in `configs/*.yaml`. No magic numbers in `src/`. If you need a constant,
  it goes in the config with a name and a comment.
- Reason codes are generated from LightGBM `pred_contrib=True` — top 3 features by
  absolute contribution, mapped to human strings via `features/registry.py`. No
  separate SHAP dependency.

---

## Testing Strategy

| Suite | Path | Gate |
|---|---|---|
| Unit | `tests/unit/` | ≥80% line coverage on generator, features, eval |
| Parity | `tests/parity/` | max abs diff online vs offline ≤ 1e-9, every feature |
| Gates | `tests/gates/` | G1–G5 all green before any model trains |
| Perf | `tests/perf/` | NFR-01..05 asserted, not measured-and-hoped |
| Determinism | `tests/unit/test_determinism.py` | same seed → identical output SHA256 |
| Leakage | `tests/gates/test_no_ground_truth_import.py` | AST-scan `features/` and `models/` for forbidden symbols |

Property tests (hypothesis) belong on the generator invariants: amounts positive,
refunds never exceed the original, `label_available_at > label_event_at`, arrival times
monotonic per merchant.

---

## Known Constraints & Non-Goals

**Constraints**
- Solo developer. ~2 days of build time. Scope is cut aggressively — see BOARD.md.
- CPU-only laptop. No GPU is available and none will be.
- No real Razorpay data. BAF (Feedzai, NeurIPS 2022) is the only external anchor.

**Non-goals for this sprint — do not build these**
- Multiple Instance Learning / attention pooling (Rung 5) — deferred, spec'd only.
- HSMM with negative-binomial emissions (Rung 7) — deferred, spec'd only.
- Neural temporal point processes, GRUs, transformers — deferred.
- Any GNN. The synthetic-graph circularity objection from v1's ADR still stands.
- A dashboard or UI of any kind.
- Online/incremental *learning*. The features are online; the model is retrained in
  batch. Delayed labels make anything else unsound.

---

## Active Work & Current Focus

Read `project-context/STATE.md` at the start of every session — it is the resume point
and it is the only file guaranteed current. `project-context/11-tickets/BOARD.md` holds
the DAG, the critical path, and the 48-hour block schedule.

Current phase: **Block 1 — foundation**. Nothing has been built yet.

---

## Agent Behavioral Rules

### Always
- Read `project-context/STATE.md` first, then only the context sections your ticket's
  `Context:` line names. Announce which files you loaded.
- Thread `rng: np.random.Generator` through every stochastic call.
- Write the test named in the ticket's `Test:` field **before** marking the ticket done,
  and run it.
- Run `make lint && make test` before declaring any ticket complete.
- Append a `LOGBOOK.md` entry after every ticket: what was built, what surprised you,
  what broke. The surprises are the most valuable thing in the repo.
- Update `STATE.md` and `BOARD.md` status at the end of every session, then stop.

### Never
- Open the test split. `make eval` without `RAKSHAK_UNLOCK=1` must refuse, and you must
  not set that variable until Block 6.
- Import `persona_id`, `risk_typology_id`, `drift_onset_at`, or anything from the
  `ground_truth` table inside `src/rakshak/features/` or `src/rakshak/models/`.
- Add a feature that cannot be maintained incrementally in `MerchantState`.
- Use `pandas`, bare `np.random.*`, or a module-level RNG.
- Introduce a dependency that is not already pinned in `pyproject.toml` without an
  explicit line in the LOGBOOK justifying it.
- Tune a model after seeing test-split results. Validation only.
- Change a v1 number, anywhere, for any reason.
- Start the next ticket in the same session. Finish, log, stop.

### Before starting any task
- `cat project-context/STATE.md`
- Read the ticket in `11-tickets/BOARD.md` and load only its named `Context:` sections.
- Confirm the ticket's `Depends on:` tickets are marked DONE on the board. If not, say
  so and stop rather than building on an unfinished dependency.

### When something in the spec is wrong
Stop. Do not patch around it in code. Say which spec section is wrong and what you
believe it should say, and wait. A spec error worked around in code becomes permanent
and undocumented, and it will surface in the panel round as a question you cannot
answer.
