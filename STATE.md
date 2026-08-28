# STATE — Rakshak

PHASE:        5 — EXECUTE, in progress
LAST SESSION: 2026-08-28 — T-0001 through T-0006 built, plus T-0003b and T-0004b. **K1 fired at T-0004 and was answered, not patched around.** Suite: 101 passed, 2 xfailed. `ruff` clean. Session closed with a **board re-plan** — see "Re-plan" below.
NEXT ACTION:  Execute **T-0017** (spec reconciliation + pre-registration). Documentation only, no code, ~1-2 h. It gates T-0007a and it must land **before** any further number is measured.

## Load for next session
- `CLAUDE.md`
- `11-tickets/BOARD.md` — read the "Revision — 2026-08-28" section at the bottom
- `11-tickets/T-0017.md`

Nothing else. T-0017 touches only `.md` files.

## Re-plan — 2026-08-28, after T-0006

The board was re-sequenced against the intended execution process: **hypothesis → set the
oracle → procure real data → ground the generator on it → code → train → eval harness →
test fairly → results.** Steps 5-8 had been executed correctly. Steps 2-4 had not been
executed at all, and one item had fallen out of the DAG entirely.

**The finding that outranks everything else: the proposal had no scoring path.**
`MODEL_REGISTRY` holds `random`, `rules`, `gbdt`. `hmm` is marked ABSENT and attributed to
"T-0004/T-0008" — but T-0004 built features and recovery, and **T-0008 does not contain the
string `hmm`** (it is shrinkage, and it sat 4th in the cut list). Nothing on the board built
the model the charter's hypothesis is about. **T-0006b** now owns it and is the highest
unretired risk in the project.

New sequence, no float:

| Day | Tickets |
|---|---|
| Sat 29 | T-0017 → T-0006b → T-0007a |
| Sun 30 | T-0007b → T-0015 → T-0012 |
| Mon 31 | T-0011 |
| **Tue 1 Sep** | T-0013, then **freeze** |
| 2-3 Sep | T-0014 (read-only viewer) + video |

**Cut:** T-0008, T-0009, T-0010. **Conditional:** T-0016, gated on T-0015's calibration-gap
diff and expected to be cut. **Promoted to MUST:** T-0012 — `CLAUDE.md` mandates a verbatim
sentence claiming BAF validation that the repo cannot currently back.

Full reasoning is in `11-tickets/BOARD.md` under "Revision — 2026-08-28". Do not re-litigate
it from this summary.

## What is built and green

| Ticket | State | Note |
|---|---|---|
| T-0001 | done | scaffold, `config.py`, `--seed` convention, Makefile (+ `make.ps1`; `make` is not installed on this machine) |
| T-0002 | done | hand-written HMM, log-space. Toy-fixture ARI 0.963, Viterbi == brute force |
| T-0003 | done | generator, 5 typologies, 771,900 rows in ~5 s |
| T-0003b | done | **inserted.** Onset schedule fixed, config consolidated, capacity now binds |
| T-0004 | done, **gate failed** | feature layer passes FR-007 exactly (8e-15). FR-013 gate fails — K1 |
| T-0004b | done, **gate still failing** | K1 remediation. ARI 0.147 → 0.319 against a 0.5 gate |
| T-0005 | done | splits/metrics/oracle. Leakage guard + test-split lock |
| T-0006 | done | rules, LightGBM, random baselines. No verdict rendered — that is T-0011 |

## The K1 story — read this before touching the sequence layer

FR-013 required four-way latent-state ARI > 0.5. T-0004 measured **0.091**. The load-bearing
number is the **oracle-parameterised ceiling of 0.378** (0.404 after T-0003b, 0.381 on the
validate group): with HMM parameters read straight off ground truth, the gate is unreachable by
any correctly-implemented HMM. Two real bugs were found and fixed en route and **both moved ARI
down** — the gap is not debuggable.

Root cause is per-state overlap. RAMP sits **1.19σ** from HEALTHY, which holds ~90% of windows.
RAMP is the early-warning state, so the gate failed precisely on the product premise.

A literature survey (`project-context/12-lit-survey-k1.md`) split the failure into a closable
estimation gap and an unclosable representation gap, and established that ARI is the wrong index
for a 90/6.4/3.4/2.2 reference (Romano et al., JMLR 17, 2016). **The user ratified amending
FR-013 to AMI + per-state recall + binary PR-AUC + detection lag, with ARI AND the oracle ceiling
retained and reported permanently.** The amendment block is in `06-requirements.md` at FR-013,
dated, citing the source, and stating it was made after the gate failed. Do not remove or bury
the ARI or the ceiling — they are what make the amendment credible rather than convenient.

T-0004b then implemented the remediation. Result: **partially-supervised fitting works**
(ARI 0.134 → 0.319, AMI 0.102 → 0.218, binary PR-AUC 0.109 → 0.327, ~85% of the way to the
ceiling and never above it). The DORMANT-rule item was **refuted**; the EM-guard item was a
**measured null** (ARI change of literally 0.0000). Shipping config is items 1+2 only.

### The unflattering finding that must reach the video

**Supervision made RAMP recall WORSE — 0.328 → 0.234 — while doubling every headline metric.**
Labels help rare *separable* states, not rare *overlapping* ones. The configuration that wins
overall is the one that goes blind on the state the project exists to catch.
**Decision taken: ship item 2 as primary and report both configurations side by side, with the
RAMP regression stated prominently.** Do not quietly ship the better-looking number.

The survey's pre-registered RAMP-recall ≥ 0.35 bar was recorded before measuring and **failed at
0.234**. It is committed as a second `xfail(strict=True)`. Leave it there.

## Why T-0007 was split into T-0007a / T-0007b

At T-0006 the `savings` metric came back **negative on every row, including both
perfect-foresight oracles** — the knapsack oracle scores -0.678 against hold-everything's
+0.573. A ceiling beaten by a trivial policy is not a ceiling.

The re-plan found the cause is **definitional, not calibrational**: `c_fp` charges one
window's MDR for a churn that costs **lifetime** margin, and `L_m` counts **gross turnover**
as realised fraud loss. Both are wrong independently of the 400-600 target ratio.

`07-math.md §5` as written instructed the project to adjust parameters until the ratio came
into range — the identical practice T-0016 forbids for the generator, and worse here because
`savings` is the headline metric. T-0017 rewrites the definitions and demotes 400-600 from a
gate to a **reported cross-check**. T-0007a implements the corrected definitions and adds an
**oracle-dominance invariant**; T-0007b builds the policy, the capacity constraint, and the
cost-asymmetry sweep that `00-charter.md §2` now requires.

## Baseline numbers (validate, 100 merchants, 20 bad, 20% prevalence, K=5)

| model | savings | PR-AUC | precision@5 | Brier | median lag |
|---|---|---|---|---|---|
| random | -3.2714 | 0.1651 | 0.0000 | 0.3589 | n/a |
| rules | -1.4472 | 0.5377 | 0.8000 | 0.1319 | 3.0 d |
| gbdt | -2.3424 | 0.6778 | 1.0000 | 0.1242 | **-1.0 d** |

No verdict is rendered yet. That is T-0011's job and it must not be pre-empted.

## Open questions and risks

- **Does the HMM beat LightGBM at all?** Unmeasured — no scorer existed until T-0006b.
  This is now the project's largest unretired risk. `gbdt` already reports PR-AUC 0.678 and
  precision@5 of 1.000 on validate; the proposal has to clear that.
- **Is the win conditional on the cost asymmetry?** T-0007b sweeps it, T-0011 states the
  boundary. `00-charter.md §2` is amended by T-0017 **before** the sweep runs, so a
  conditional result is pre-registered rather than a post-hoc caveat.
- **gbdt flags BEFORE the labelled transition (-1.0 d).** Narrowed by reading the generator:
  `_ramp()` returns `lo` for every pre-onset day and every injector writes through it or
  through an explicit `[onset:]` slice, so no injector telegraphs. `WINDOW_DAYS = 7`, so a
  window straddling onset holds up to 6 post-onset days and window-start-vs-onset yields
  exactly -1. **T-0011 confirms window aliasing first**, and only escalates to a leakage
  investigation if that fails to explain it.
- **Every number is still measured on a generator this repo wrote.** T-0015 makes the gap
  measured rather than merely admitted, via `results/calibration_gap.md`. T-0016 would close
  it and is expected to be cut; if so, the gap ships documented.
- **Full `pytest` now takes 2-4 minutes** (the full-scale HMM fit), and T-0006b adds a second
  fit. Watch it against NFR-004's 15-minute `make eval` budget (K3). Subagents running the
  full suite have been killed by a 600 s no-output watchdog — run targeted files during
  development.
- **`make` is not installed on this machine.** The Makefile ships unexercised; `make.ps1` is
  the local shim. Do not claim `make eval` is green on camera until it runs on a Linux
  checkout.
- **ADR-0005's consequences** should record that T-0004b refuted the DORMANT-rule approach.

### Closed by the 2026-08-28 re-plan

- ~~The repo contradicts itself on the dashboard.~~ T-0017 narrows `00-charter.md §7` rather
  than reversing it; T-0014 moves to the video window (2-3 Sep) as a read-only viewer over
  frozen artifacts, costing zero build days and unable to affect any number.
- ~~Cost matrix blocking the primary metric with no owner for the tuning problem.~~ Owned by
  T-0017 (definitions) and T-0007a (implementation + oracle-dominance invariant).
- ~~The freeze date.~~ 1 Sep 2026 is a **Tuesday**. Corrected in `BOARD.md`; T-0017 fixes
  `CLAUDE.md`.
