# ADR-0003 — Reinforcement learning is rejected for the build and retained as a pitch slide

**Status:** Accepted — decision taken in Phase 2 (pre-execution). **Written retrospectively on
2026-08-29** from `CLAUDE.md`'s rejection table and `01-understanding.md` D-06.
**Supersedes:** none.

## Context

"Merchant risk decisioning under a capacity constraint" reads like a sequential decision problem,
and a reviewer will think of RL within a minute of hearing it. The framing is genuinely apt —
this **is** a POMDP: the latent risk state is unobserved, the belief is the sufficient statistic,
actions are pass/review/hold, and review capacity is a budget constraint across the horizon.

Two facts kill it as build work.

1. **There is no reward signal inside the build window.** Ground truth arrives when chargebacks
   land — **45 to 120 days** after the transactions that caused them. An agent trained during a
   4-day build has nothing to learn from.
2. **Training on our own generator learns our own assumptions.** The only environment available
   is the simulator this repo wrote. An RL agent would converge on the generator's transition
   model, and its measured return would be a statement about the generator, not about fraud. This
   is the same circularity objection ADR-0002 raises against GNNs, and it is worse here because
   the agent trains *inside* the simulator rather than merely being scored on it.

## Options considered

**(a) Offline RL / batch-constrained Q-learning on generated episodes.** Removes the online
interaction problem, not the circularity one. Still learns the generator.

**(b) Bandit formulation over review allocation.** Cheaper, but the delayed-reward objection is
unchanged and it discards the belief state, which is the part of the project that has value.

**(c) Bayes Minimum Risk over the HMM's belief — a one-step-optimal decision rule.** Closed form,
no training, and it is the myopic-optimal action given the posterior and the cost matrix. Under
delayed and lagging ground truth, the horizon RL would optimise over is not observable anyway.

**(d) Formulate the POMDP explicitly and present it as considered-and-rejected.**

## Decision

(c) for the build — see ADR-0005 — **plus (d) for the pitch.** The POMDP formulation goes on a
slide with its state, action, observation and reward definitions, followed by the two reasons it
was not trained. `CLAUDE.md` records this as *"considered and rejected with reasons."*

## Consequences

* **The policy is myopic and it must be described as myopic.** BMR optimises the current
  decision, not a trajectory. Claiming sequential optimality would be false.
* **A real deployment could revisit this** once 45–120 days of realised outcomes accumulate.
  That is the honest condition under which the rejection would be reopened, and it should be
  stated on the slide rather than implied.
* **Anticipating the reviewer's question is the deliverable.** A panel member who asks "why not
  RL?" and gets a formulated POMDP plus two disqualifying facts is a better outcome than one who
  asks and gets a shrug — and a considerably better outcome than a trained agent whose returns
  measure our own simulator.
