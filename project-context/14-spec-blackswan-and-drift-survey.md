<!-- HEAD
FILE:     14-spec-blackswan-and-drift-survey.md
PHASE:    5 — EXECUTE (spec for new, additive-only work opened mid-phase)
UPDATED:  2026-08-30
STATUS:   agreed, not yet built
SUMMARY:  Spec for two additive work items opened after K2's FAIL verdict (T-0011):
          (1) a population-wide "black-swan" shock stress test, additive, never touching
          any frozen number; (2) a written literature survey re-checking the locked stack
          against prior art on post-onboarding merchant drift detection, doc-only, no new
          model in the ablation table. Both rank below T-0013/T-0018/T-0020/T-0021/T-0019
          in the pre-declared cut order — see 11-tickets/BOARD.md.
-->

# 14 — Spec: black-swan shock stress test + drift-detection literature survey

Produced from a grilling session on 2026-08-30 (Sun 30 Aug, evening), the day before the
Tue 1 Sep EOD freeze. Read `CLAUDE.md` and `STATE.md` before touching either ticket this
spec produces — the non-negotiables below are not restated generically, they are applied
specifically to this new work.

## Problem Statement

K2 fired FAIL at T-0011 (`STATE.md`, "T-0011 — the verdict"): the HMM does not beat the
`rules` baseline at any swept cost asymmetry. The project's response, per `00-charter.md`
§3, is to report the negative result and pivot the narrative to explainability and the
cost frontier — not to tune the model.

Two gaps remain in that pivoted narrative, both raised by the project owner tonight:

1. The central cost claim this project exists to make — *"blunt global thresholds freeze
   honest merchants"* — has never actually been demonstrated inside this repo. The
   generator (`src/rakshak/generator/generate.py`) models every merchant as a fully
   independent stochastic process. There is no scenario anywhere in `results/` showing
   what happens to any model's false-positive rate when something hits many merchants at
   once — a demand surge, an MDR change, a payment-rail outage. Without that scenario, the
   project's own thesis is asserted, not measured.
2. The stack (hand-written HMM, LightGBM, Bayes Minimum Risk, 9 ADRs) was locked before
   and during the build. Aside from the K1 literature survey
   (`project-context/12-lit-survey-k1.md`), which addressed a narrow metric-choice
   question (ARI vs. AMI for a 90/6.4/3.4/2.2 state split), nothing has re-checked the
   locked stack against current prior art specifically on the problem this project claims
   to solve — post-onboarding merchant drift detection sitting between Vulcan-style
   transaction scoring and Bumblebee-style onboarding scoring.

## Solution

Two additive tickets, both explicitly ranked **below** the existing open tickets
(T-0013, T-0018, T-0020, T-0021, T-0019) in the pre-declared cut order recorded in
`11-tickets/BOARD.md`, each individually degradable to a cheaper fallback rather than an
all-or-nothing build:

- **T-0022** — black-swan shock stress test. Primary: a new, separate dataset with a
  population-wide shock injected across train/validate/test, `rules`/`gbdt`/`hmm` refit on
  it via the existing harness pointed at a new data directory, reported in a new
  `results/blackswan.md`. Fallback (if the refit doesn't land in time): keep training
  exactly as the frozen T-0011 run (same `data/synthetic/`), and only inject the shock into
  a copy of the **test-window** transactions before feature computation — this needs no
  model-persistence machinery (none exists in this repo today) and still answers "how do
  already-learned models react to a shock they never trained on."
- **T-0023** — literature survey, written document only. No new model is implemented or
  added to the ablation table from this survey before freeze; if the survey turns up a
  compelling method, it becomes the headline item of a "future work" section, not a
  same-night addition.

Both feed a short, honest paragraph each into T-0013's limitations/future-work section.
Neither can touch, replace, or be spun as improving K2's verdict.

## User Stories

1. As the panel member reading this repo, I want to see the project's own central cost
   claim ("blunt thresholds freeze honest merchants") actually tested against a scenario
   the current models were never designed for, so that I can judge whether the claim is
   asserted or measured.
2. As the panel member, I want an honest report of what happens to Rakshak's own models
   under a shared shock — including if they panic just as badly as a blunt threshold would
   — so that the negative result (if any) is reported with the same discipline as K2's FAIL.
3. As Ayush assembling T-0013's README, I want a `results/blackswan.md` artifact produced
   the same way every other results file is (fixed seed, regenerable, no hand-edited
   numbers), so that this new claim carries the same provenance guarantee as every other
   number in the repo.
4. As Ayush, I want the black-swan dataset to live in its own directory
   (`data/synthetic_shock/`) and never overwrite `data/synthetic/`, so that K1's ARI
   ceiling, K2's verdict, the ablation table, and the BAF validation cannot be silently
   invalidated by a change made under time pressure the night before freeze.
5. As a future engineer maintaining Rakshak, I want the shock-injection code to reuse the
   existing `Segment`/typology/state-path machinery in `generator/generate.py` rather than
   a parallel generator module, so that there is exactly one place that defines how a
   merchant's transaction stream is produced.
6. As a future engineer, I want `eval.splits.load_split` and `eval.harness.run` to accept
   optional data-path overrides rather than a second harness implementation existing, so
   that a bug fixed in the scoring logic is fixed for both the primary and the shock track
   at once.
7. As Ayush, I want the literature survey to explicitly compare prior art against the 9
   existing ADRs, so that if the survey disagrees with a locked decision (no GNN, no
   transformers, no RL, hand-written HMM over `hmmlearn`, NSGA-II not III), that
   disagreement is stated and dated rather than silently contradicted.
8. As the panel member, I want the survey to be honest about what changed and what didn't
   since ADR-0001..0009 were written, so that a dated addendum — not a rewritten ADR — is
   the record of any reconsideration, matching the discipline already used for the FR-013
   amendment.
9. As Ayush, I want a pre-declared fallback for T-0022 (shock-only-on-test-window if the
   full refit stalls) and a pre-declared cut order across both tickets relative to the
   existing backlog, so that no decision about what to drop happens under pressure at
   11pm tomorrow.
10. As the panel member, I want any degraded merchant recall or elevated false-positive
    number produced by either ticket to be reported exactly as measured, including if it
    is unflattering to Rakshak, so that these two tickets are held to the same honesty bar
    as K1's RAMP-recall regression and K2's FAIL verdict.

## Implementation Decisions

### T-0022 — black-swan shock stress test

- **Generator seam**: `generate.py`'s CLI (via `rakshak.cli.base_parser`) gains
  `--shock-day` (repeatable, day-of-horizon integers) and `--shock-magnitude` (a
  multiplier applied to per-merchant daily transaction volume and mean amount on shock
  days, applied identically across every merchant regardless of typology or segment).
  Output goes to a new `SYNTHETIC_SHOCK_DIR` (`data/synthetic_shock/`), mirroring
  `SYNTHETIC_DIR`'s two-file layout (`transactions.parquet`, `state_paths.parquet`).
  `data/synthetic/` is never written to by this path.
- **No new latent state.** The shock is a shared multiplicative perturbation on the
  emission process, not a new HMM state — HEALTHY merchants stay HEALTHY in
  `state_paths.parquet` through a shock day. This is the entire point of the test: the
  ground truth says nothing changed, so any model that flags a shocked-but-healthy
  merchant is by definition a false positive.
- **Same split geometry.** `SPLIT_DAY_BOUNDS` and `MERCHANT_GROUP_CYCLE` from
  `config.py` are reused unmodified — this is a new dataset scored the same way, not a
  new split design (changing split geometry is a DESCEND per `config.py`'s own docstring,
  and this ticket does not need to touch it).
- **Harness seam**: `eval.splits.load_split()` gains optional `transactions_path=` /
  `state_paths_path=` parameters, defaulting to the current `TRANSACTIONS_PARQUET` /
  `STATE_PATHS_PARQUET` constants so every existing call site is unaffected.
  `eval.harness.run()` (which already accepts a `results_dir` override) gains a matching
  optional pass-through. A new thin entry point (a `--dataset shock` flag on the existing
  `harness.main`, or a two-line wrapper script — implementer's call, but it must call
  `run()`, not reimplement it) points both overrides at the shock directory and at
  `results/blackswan/`.
- **Primary metric**: for each of `rules`/`gbdt`/`hmm`, flagged-fraction (and, where the
  model produces one, review/hold rate) on shock-day windows vs. a matched set of
  non-shock control windows from the same merchants, reported as a delta with the
  underlying counts shown — not just the delta, so a reader can see the base rate it's
  computed from (AP-06 discipline, same as the `random`-floor requirement everywhere
  else in this repo).
- **Fallback definition (revised from the grilling session's "score already-fitted T-0011
  artifacts")**: this repo does not persist fitted model objects anywhere — every
  `evaluate_model` call fits fresh from the `Split` it's handed. A literal "run the exact
  T-0011 model" fallback would require adding model persistence, which is new
  infrastructure, not a lighter fallback. The equivalent fallback that needs no new
  infrastructure: keep `split.train` / `split.validate` exactly as the frozen
  `data/synthetic/` run (so nothing about how the models were fit changes), and inject the
  shock only into the transactions underlying `split.test`'s feature windows before
  scoring. This is arguably a better-motivated experiment than the full-retrain primary
  path (it measures how models already deployed react to a shock they never saw), and it
  is the cheaper build. If the full refit is at risk of not landing by the agreed cutoff,
  downgrade to this rather than cutting the ticket.
- **Reporting**: `results/blackswan.md`, written by a renderer following the existing
  `render_summary`/`render_sensitivity` pattern in `eval/harness.py` / `decision/policy.py`
  — same table style, same provenance line (seed, split, dataset path) at the top.

### T-0023 — literature survey

- **Scope**: prior art specifically on post-onboarding merchant/account risk drift
  detection — sequence models over merchant transaction streams, changepoint detection
  for behavioral drift, bust-out detection, laundering-endpoint / mule-account detection,
  refund-collusion detection — evaluated against this project's hard constraints (CPU
  only, solo, 4-day original build window, now ~1 day of extension).
- **Explicit comparison targets**: every locked stack choice and every rejected
  alternative in `CLAUDE.md`'s stack table and `docs/adr/ADR-0001` through `ADR-0009`.
  For each, the survey states one of: *reaffirmed* (literature still supports the
  decision), *reaffirmed with a caveat* (supports it, but a newer method exists that
  wasn't feasible under this project's constraints — name it), or *reconsidered*
  (literature suggests the decision should change — this requires a dated ADR addendum,
  not a silent rewrite, mirroring the FR-013 amendment's discipline).
- **Output**: `project-context/15-lit-survey-drift-detection.md` (numbered after this
  spec; `12-lit-survey-k1.md` and `13-retrospective.md` already occupy 12 and 13 in this
  directory), following that file's format (question, sources checked, verdict, what it
  changed). Any *reconsidered* finding also gets a dated addendum block appended to the
  relevant `docs/adr/ADR-000X-*.md` file, in that file's own house format.
- **No code output.** This ticket does not touch `src/`, `MODEL_REGISTRY`, or
  `results/ablations.md`. A short paragraph summarizing the verdict is handed to T-0013
  for its limitations/future-work section.

## Testing Decisions

- **T-0022** is testable the same way T-0006b and T-0007a were: a determinism test (same
  seed + same `--shock-day`/`--shock-magnitude` → identical output hash, mirroring
  `generate.py`'s existing determinism test for the unshocked path); a ground-truth
  invariant test asserting no merchant's `state_paths.parquet` entry changes state across
  a shock day purely because of the shock (the shock must be visible only in
  `transactions.parquet`, never in ground truth — this is what makes a flag on a shock day
  provably a false positive rather than a correct catch); and a harness-level test
  asserting a run against `data/synthetic/` (unchanged path) produces byte-identical
  `results/summary.md` before and after this ticket lands, so the seam addition is proven
  not to have touched the primary path. Prior art: `tests/test_cost.py`'s
  oracle-dominance invariant test, `tests/test_hmm_recovery_fullscale.py`'s burn-in
  assertion — both are "assert a structural invariant holds," the same shape as the new
  no-state-change-under-shock test.
- **T-0023** is a document; its "test" is the same as `12-lit-survey-k1.md`'s — every
  claim about the current stack cites the specific ADR or CLAUDE.md line it agrees or
  disagrees with, checkable by a reader, not asserted from memory.

## Out of Scope

- Replacing or regrounding the emission process (`daily_count_fano_factor` etc.) — that
  is T-0016, already scoped and kept conditional, and is a different problem (matching a
  real marginal distribution) from this one (testing shared-shock robustness).
- Any change to `SPLIT_DAY_BOUNDS`, `MERCHANT_GROUP_CYCLE`, or any number in
  `results/verdict.md`, `results/ablations.md`, `results/baf_validation.md`, or
  `results/lag_probe.md`. If building T-0022 or T-0023 requires touching any of those
  files or the frozen split logic, stop and raise it — do not patch around it.
- Building and benchmarking a new candidate model surfaced by the survey. This was
  explicitly proposed and explicitly rejected during the grilling session as too risky
  for a one-night, honest-metrics-required build.
- Any change to `00-charter.md`'s pre-registered NFR-001 ≥20% bar, or to the K2 verdict's
  wording in `results/verdict.md`. Both tickets are orthogonal probes, not a rebuttal.

## Further Notes

- Both tickets rank below T-0013, T-0018, T-0020, T-0021, and T-0019 in the cut order
  agreed in the grilling session and recorded in `11-tickets/BOARD.md`. If time runs out:
  cut T-0023's ADR-addendum step first (ship the survey doc without it, flagged as
  incomplete), then cut T-0022 down to its fallback, then cut T-0022 entirely before ever
  touching T-0013 or the four already-open tickets.
- Neither ticket may be reported as strengthening or rescuing K2's FAIL verdict. If
  T-0022 shows Rakshak's own models are just as prone to shock-induced false positives as
  a blunt global threshold, that is exactly as reportable as the finding itself — and per
  `CLAUDE.md` non-negotiable #1, it must be reported that way, not around.
- `results/blackswan/` and `project-context/15-lit-survey-drift-detection.md` are new
  paths; add `results/blackswan/` to whatever `make eval` / `make.ps1` step already
  chains the existing four eval modules only if T-0022 lands as a `make eval` target —
  if it's kept as a manually-invoked side track (recommended, since it is explicitly not
  part of the frozen NFR-004 15-minute budget), state that plainly in `results/blackswan.md`
  and in `09-interfaces.md`/T-0020 so nobody assumes it's part of the regenerable core.
