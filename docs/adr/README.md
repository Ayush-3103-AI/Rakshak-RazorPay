# Architecture Decision Records

One file per decision. Format follows ADR-0008, the first one written: **Context → Options
considered → Decision → Consequences**, with a status line naming the ticket that took or
executed the decision.

## Provenance — read this before citing one

**ADR-0001 through ADR-0007 and ADR-0009 were written on 2026-08-29, after the decisions they
record.** They were reconstructed from the documents that *did* exist at the time — `CLAUDE.md`'s
stack and rejection tables, `01-understanding.md`, `03-landscape.md`, `04-patterns.md`,
`06-requirements.md`, `07-math.md`, the ticket files and `project-context/12-lit-survey-k1.md`.

**They are not backdated.** Each carries the date it was written and the sources it was
reconstructed from. The decisions are contemporaneous with the phase named in the status line;
only the record is retrospective.

Why they were missing: every one of these numbers was cited across the spec documents from Phase
2 onward, but only ADR-0008 was ever written to disk. The gap was found in a code review on
2026-08-29 and closed the same day, rather than being left as a README citing decision records
that do not exist.

## Index

| ADR | Decision | Status |
|---|---|---|
| [0001](ADR-0001-hand-written-hmm.md) | Hand-written HMM in numpy; `hmmlearn` rejected | Accepted, **implemented** |
| [0002](ADR-0002-no-graph-models.md) | No GNN, no sequence transformer; graph signal approximated by scalars | Accepted, **implemented** (FR-008) |
| [0003](ADR-0003-no-reinforcement-learning.md) | RL rejected for the build, retained as a POMDP pitch slide | Accepted, **by design not built** |
| [0004](ADR-0004-nsga-ii-not-nsga-iii.md) | NSGA-II not NSGA-III, plus a mandatory grid-search ablation | Accepted, **NOT implemented — T-0009 cut** |
| [0005](ADR-0005-three-action-policy-under-capacity.md) | Three actions under a hard review-capacity constraint | Accepted, **implemented** (T-0007b) |
| [0006](ADR-0006-empirical-bayes-shrinkage.md) | Closed-form empirical-Bayes shrinkage of per-merchant cost parameters | Accepted, **NOT implemented — T-0008 cut** |
| [0007](ADR-0007-hybrid-data-strategy.md) | Hybrid data: own generator for sequences, public benchmark for the decision layer | Accepted, **implemented** (T-0012) |
| [0008](ADR-0008-review-capacity-scaling.md) | Review capacity expressed per 1000 merchants, not absolute | Accepted, **implemented** (T-0003b) |
| [0009](ADR-0009-k1-label-informed-hmm.md) | K1 response: label-informed HMM estimation, FR-013 metric suite re-specified | Accepted, **implemented** (T-0004b) |

## Two that are cited but not built

Do not let a README or the video cite these as though they shipped:

* **ADR-0004** — no multi-objective frontier exists. `pymoo` was removed from `pyproject.toml`
  (T-0020); ADR-0004's grid-search obligation is still undischarged and should be closed or
  explicitly re-justified before freeze.
* **ADR-0006** — no calibration happens anywhere in this repo, which is load-bearing because
  ADR-0005's BMR policy consumes raw scores as posteriors.

## The ADR-0005 collision

`ADR-0005` was booked twice: for the three-action policy (Phase 2, cited by FR-015, FR-017 and
`07-math.md` §7) and for the K1 response (2026-08-28, drafted inside
`project-context/12-lit-survey-k1.md` and cited by FR-013's amendment block).

**Resolved 2026-08-29:** the policy keeps **0005** on the earlier claim; the K1 response is
renumbered to **0009**. Citations were updated in place with a note. Entries in `LOGBOOK.md` are
left as written — it is append-only, and rewriting a log to match a later renumbering would be
worse than the inconsistency.
