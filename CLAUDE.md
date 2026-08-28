# CLAUDE.md — Rakshak

> Read this file first, every session. Then read `project-context/STATE.md`. Then read **only** the files the current ticket names. Nothing else.

---

## What this is

**Rakshak** — a post-onboarding merchant risk sentinel with cost-optimal, capacity-constrained decisioning.

Submission for the **Razorpay AI Buildathon 2026, Track 02 (AI Risk Manager)**. Solo builder. Deliverables are a public repo, a 5-minute pitch video, and architecture documentation. Selection is by panel review of what was built and how honestly it was measured.

**One-paragraph version:** Razorpay's Vulcan scores every *transaction* in milliseconds. Razorpay's Bumblebee reviews every *merchant* once at onboarding. Nothing watches a merchant that was already cleared drift from good to bad over the following weeks — so bust-outs, laundering endpoints, category drift and refund collusion surface only when chargebacks land 45–120 days later. Rakshak runs a per-merchant Hidden Markov Model over the transaction stream, updating a belief over latent risk states with each new transaction, then converts that belief into pass / review / hold using per-merchant cost economics under a fixed analyst-hour budget.

---

## The three non-negotiables

Everything else in this repo is negotiable. These are not.

### 1. Honest metrics or nothing

The track's published bar is *"honest metrics including false-positive cost."* This project is judged on measurement discipline more than on model sophistication.

- Every number in the README and the video must be regenerable by `make eval` with fixed seeds.
- Never report a metric on data the model has seen. Temporal split AND merchant-group split, both enforced in code, not by convention.
- Report where the model fails. The slow-ramp evader typology exists specifically so that we can report degraded recall on it. **Do not tune it away. Do not hide it.**
- If a baseline beats the HMM, report that the baseline beat the HMM.

### 2. Defense-only, always

The track states: *"Strictly defense-only: anything offense-capable is disqualified."*

- Never write code that generates evasive fraud patterns for any purpose other than the declared adversarial test in the eval harness.
- The generator produces fraud typologies to **test detection**. It is not a fraud toolkit. Keep it in `src/rakshak/generator/`, documented as an evaluation artifact, and say so in the README.
- No code that probes, enumerates, or exploits any real payment system. All Razorpay API usage, if any, is test-mode only.

### 3. Synthetic data must be labelled synthetic, everywhere

The sequence layer trains on data we generated. That is a real limitation and it must be stated in the README, in the video, and in every results table — not buried.

Correct framing, use it verbatim: *"Sequence-layer metrics are measured on synthetic merchant streams with injected typologies; the generator is in this repo. The decision layer is additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark derived from real bank data."*

---

## Hard constraints

| Constraint | Value |
|---|---|
| **Compute** | CPU only. No GPU, not even Colab. If a method needs a GPU, it is out of scope. |
| **Team** | Solo. No parallel lanes that assume a second person. |
| **Build window** | Sat 29 Aug – Mon 1 Sep 2026, end of day. Hard stop. |
| **Video** | Tue 2 Sep – Wed 3 Sep. Review Thu 4 Sep. Submit Fri 5 Sep. |
| **Real data** | None available. Public datasets + own generator only. |
| **Runtime** | Full `make eval` must complete in under 15 minutes on a laptop. |
| **Dependencies** | Permissive licences only (MIT / BSD / Apache-2.0). No GPL/AGPL. |

---

## Stack — locked

| Layer | Choice | Note |
|---|---|---|
| Sequence model | **Hand-written HMM** in numpy | Forward-backward, Viterbi, Baum-Welch. **Do not use `hmmlearn`** — it is in limited-maintenance mode, last release Oct 2024, and cannot express our hierarchical priors. Writing it ourselves is also the clearest proof of mathematical depth in the repo. |
| Changepoint baseline | Hand-written BOCPD | Adams & MacKay 2007, truncated run-length posterior |
| Discriminative baseline | LightGBM | The incumbent that must be beaten |
| Cost layer | Bayes Minimum Risk + savings score | Bahnsen (2015), Elkan (2001) |
| Calibration | Empirical-Bayes shrinkage, closed form | No MCMC, no PyMC |
| Multi-objective | pymoo **NSGA-II** ≥0.6.2 | **NSGA-II, not NSGA-III.** Three objectives is not many-objective. |
| Data | Own generator + BAF (Feedzai, NeurIPS 2022) | |
| Language | Python ≥3.11 | |

### Explicitly rejected — do not reintroduce

| Rejected | Why | Where it's documented |
|---|---|---|
| Graph neural networks | GPU required; synthetic graph evaluation is circular; infeasible solo in 4 days. **Approximate with graph-derived scalar features instead** (payer entropy, repeat-payer ratio, Jaccard drift, Herfindahl concentration). | ADR-0002 |
| Sequence transformers | A payments-fraud vendor's own research team (NICE Actimize, arXiv:2605.21490) reports parity with feature engineering. Not worth 4 days on CPU. | ADR-0002 |
| Reinforcement learning | No reward signal — ground truth lags 45–120 days, and training on our own generator means learning our assumptions, not fraud. Framed in the pitch as *considered and rejected with reasons*, plus a POMDP formulation slide. | ADR-0003 |
| `hmmlearn` | Limited-maintenance mode; cannot express hierarchical priors. | ADR-0001 |
| NSGA-III | Designed for 4+ objectives. We have 3. | ADR-0004 |
| Deepfake / KYC document analysis | Needs vision models and GPU. Out of scope. | 03-landscape.md |
| RTO / COD return fraud | Razorpay's Thirdwatch has owned this since 2019. Most crowded idea in the submission pool. | 03-landscape.md |

---

## Working agreements for Claude Code

**Session shape.** One ticket per session. Read `STATE.md`, read the ticket, read only the context sections the ticket names, build, test, log, update `STATE.md` and `BOARD.md`, stop. Do not start the next ticket in the same session.

**Ticket order is by risk retirement, not by comfort.** The DAG in `11-tickets/BOARD.md` is deliberately front-loaded with the tickets most likely to kill the project. Do not reorder toward the easy ones.

**When a ticket reveals a spec error, stop and raise it.** Do not patch around it in code. Offer a DESCEND to Phase 2. A spec error worked around silently becomes permanent and undocumented.

**Never mark a ticket done that doesn't pass its `Done when` clause.** "Mostly working" is not done.

**Write the `SURPRISE` field in `LOGBOOK.md` honestly, especially when it's unflattering.** It is the highest-value field in the repo and it feeds the video's credibility.

**Determinism is a hard requirement.** Every script takes `--seed`. Global seed is set in `src/rakshak/config.py`. If a result is not reproducible, it does not go in the README.

**Every number in the README carries provenance** — which script produced it, which seed, which split, which date.

**Do not add dependencies without checking the licence** and recording the addition in `LOGBOOK.md`.

**Prefer boring, readable code.** A panel member will read this repo. Clever one-liners cost more than they save. Type hints on every public function. Docstrings that state units.

---

## Repo layout (target)

```
rakshak/
├── CLAUDE.md                      ← this file
├── README.md                      ← the panel reads this. Results + provenance.
├── Makefile                       ← make setup / make eval / make figures / make test
├── pyproject.toml
├── project-context/               ← the project's memory. Read STATE.md every session.
├── src/rakshak/
│   ├── config.py                  ← seeds, paths, cost defaults, all constants
│   ├── generator/                 ← synthetic merchant streams + 4 typologies
│   ├── features/                  ← emission feature engineering
│   ├── models/
│   │   ├── hmm.py                 ← hand-written. forward, backward, viterbi, baum_welch
│   │   ├── bocpd.py               ← changepoint baseline
│   │   ├── gbdt.py                ← LightGBM baseline
│   │   └── rules.py               ← static rule engine (the floor)
│   ├── decision/
│   │   ├── cost.py                ← Bayes minimum risk, savings score
│   │   ├── shrinkage.py           ← empirical-Bayes per-merchant calibration
│   │   └── policy.py              ← 3-action policy under capacity constraint
│   ├── optimize/nsga.py           ← pymoo NSGA-II frontier + grid-search baseline
│   ├── eval/
│   │   ├── splits.py              ← temporal + merchant-group split. LEAKAGE GUARD LIVES HERE.
│   │   ├── metrics.py             ← PR-AUC, precision@K, Brier, detection lag, savings
│   │   ├── oracle.py              ← perfect-foresight knapsack ceiling
│   │   └── harness.py             ← runs everything, writes results/
│   └── explain/reasons.py         ← Viterbi path → merchant-readable reason string
├── tests/
├── results/                       ← generated. tables + figures. git-tracked.
└── data/                          ← generated + downloaded. git-ignored except manifests.
```

---

## The audience

Write everything for one reader: a **Razorpay Head of Risk Operations sitting on the panel.** They have seen a thousand fraud demos. They do not care about your architecture diagram. They care about three things:

1. Does it catch things we currently miss, and by how much?
2. How many honest merchants does it freeze that shouldn't be frozen, and what does that cost?
3. Can I explain the decision to the merchant when they call and shout?

Question 3 is the one nobody else in the submission pool will answer. It is why the HMM's Viterbi path is the centrepiece and not an afterthought.

---

## Context files — read on demand only

| File | Read when |
|---|---|
| `project-context/STATE.md` | **Every session. Always.** |
| `00-charter.md` | Questioning scope or success criteria |
| `01-understanding.md` | Questioning a settled decision |
| `03-landscape.md` | Questioning a rejected alternative |
| `06-requirements.md` | Any ticket — read only the FR/NFR sections it names |
| `07-math.md` | Implementing HMM, cost layer, shrinkage, or NSGA objectives |
| `08-pseudocode.md` | Implementing an algorithm |
| `09-interfaces.md` | Touching a module boundary or a schema |
| `10-done.md` | Writing the README, the demo, or checking acceptance |
| `11-tickets/BOARD.md` | Choosing the next ticket |
| `LOGBOOK.md` | Never read in full. Append only. |
