<!-- HEAD
FILE:     03-landscape.md
PHASE:    1c — UNDERSTAND (lit-survey output)
UPDATED:  2026-08-28
STATUS:   gated
SUMMARY:  Razorpay's existing risk stack (Vulcan, Bumblebee, Thirdwatch, ACS Risk Engine,
          Capital dedupe) mapped so we build into the gap, not on top of it. Five method
          families surveyed: HMM (chosen, with the 2008 false-positive failure explained),
          BOCPD (baseline), GNN (rejected, GPU), transformers (rejected, vendor reports
          parity), GBDT (the incumbent baseline). Failure archaeology: why cardholder-level
          HMMs failed and why merchant-level does not inherit it. Datasets: BAF, FiFAR.
          Library health: hmmlearn dying, pymoo healthy at 0.6.2.
OPEN:     none — stop rule reached; further sources stopped changing ADRs
-->

# 03 — Landscape

All claims below are sourced. A claim in the context folder without provenance is a rumour with formatting.

---

## Part 1 — What Razorpay already built

This section exists so we never pitch something they shipped.

### Vulcan — the transaction-level foundation model (Aug 2026)
- India's first transformer-based AI foundation model for payments; built with NVIDIA and AWS. — razorpay.com/foundation-model, press.aboutamazon.com (Aug 2026)
- Trained on ~3 trillion data points across ~4 billion payments; ~3,000 signals per transaction; proprietary architecture and data. — BW Disrupt, Crowdfund Insider (Aug 2026)
- Beta results claimed: 8–10% improvement in payment success rates; 8× more international card fraud detected; 5× more fraudulent or disputed transactions identified; 1–2 lakh additional monthly purchases via checkout personalisation. — Harshil Mathur, X, Aug 2026
- Architectural point is *structure, not scale*: it collapses routing, fraud, and risk scoring — previously separate models — into one shared layer. — techtimes.com, 18 Aug 2026
- **Unit of analysis: the transaction, in real time.** This is the gap we exploit.

### Bumblebee / "Agentic Risk" — onboarding merchant website review (Dec 2025)
Source: dev.to/razorpaytech, `meet-bumblebee-agentic-ai-flagging-risky-merchants-in-under-90-seconds`

- Risk ops was doing 10,000–12,000 manual website reviews/month at ~4 min each = 700–800 human hours/month, with inconsistent verdicts between agents.
- Their third-party explicit-content screening service produced ~50 alerts/month at **under 10% precision**.
- Architecture: Planner → parallel Fetchers (each pruning locally) → Analyzer that **runs deterministic rules first, invoking the LLM only for interpretive tasks.**
- Results: token usage −60%, latency 35s → 8–12s, success rate 88% → 99%+.
- **Their stated future work is our project:** *"adding predictive agents that don't just evaluate merchant risk at onboarding but continuously monitor for behavioral changes."*

**Two things to steal:** (a) deterministic-rules-first is the architecture we independently arrive at — say so; (b) the 700–800 analyst-hours figure is our capacity constant.

### Risk Case Management redesign (Feb 2026)
Source: engineering.razorpay.com, `our-obsession-with-merchant-experience-breaking-the-risk-review-black-box`

- Documented failures: generic email templates asking for broad document categories rather than the specific proof tied to the risk flag; and coupled state logic where "Needs Clarification" was a sub-status of "Funds on Hold," so agents could not request information without freezing settlements.
- Claimed results: 50% reduction in investigation time, 40% fewer merchant interactions per case.
- **They fixed the interface. The decision quality underneath is still the bottleneck.** That is where our reason-string output lands.

### Others
- **Thirdwatch** (acq. 2019) — COD/RTO fraud scoring. Mature. Do not compete.
- **ACS + Risk Engine** — risk-based authentication on cards.
- **Razorpay Capital dedupe** — fuzzy matching on PAN/phone/GSTIN against blocklists for lending underwriting.

### The commercial wound
Razorpay's public review profile splits sharply: strong on developer experience, weak once an account is flagged. Sudden account limitations and fund withholding during compliance reviews dominate negative reviews. — xflowpay.com review, Aug 2026; Trustpilot. **Every false positive is a churned merchant plus a public post.** This is the business case for the cost layer.

---

## Part 2 — Method families

### HMM — chosen
- **Canonical:** Srivastava et al., *Credit Card Fraud Detection Using Hidden Markov Model*, IEEE TDSC 2008. HMM trained on cardholder normal behaviour; flag when a transaction is not accepted with sufficiently high probability.
- **The known criticism, which we must answer:** a survey (arXiv:1611.06439) records that these models used profiles of the last ~10 transactions and that **"HMM produces high false positive rate."**
- **Why it does not transfer to us — three structural fixes:**
  1. **Unit.** Those were per-cardholder with ~10 observations — hopelessly under-identified transition matrices. We are per-merchant with hundreds to thousands.
  2. **No cost layer.** They thresholded likelihood directly. "High false-positive rate" is a statement about an uncalibrated threshold, not the model. Our entire second stage exists to fix this.
  3. **No pooling.** Each cardholder fit in isolation. We shrink toward a segment prior.
- **The supporting precedent:** Rieke et al., *Sequential fraud detection for prepaid cards using hidden Markov model divergence*, Expert Systems with Applications, 2017 — explicitly models **store terminals rather than customers**, i.e. our unit of analysis. Its framing of the problem is our thesis: *a sequence of new transactions may be anomalous while any single transaction within it is valid.* **Put that sentence in the video.**
- Maturity: `Consensus` as a method, `dated` in this specific application. Fit: **9/10**.

### BOCPD — baseline
- Adams & MacKay, arXiv:0710.3742 (2007). Maintains a posterior over run length — observations since the last changepoint — updated by message-passing recursion.
- Nearly our architecture already exists in an adjacent domain: arXiv:2510.09619 builds a risk-calibrated BOCPD detector with a cost-sensitive decision rule, motivated by the observation that classical unsupervised methods *"assume static distributions and do not account for cost asymmetry."*
- Practical note: truncate the run-length posterior at max length L for bounded update cost (see CALIBURN, arXiv:2605.24696).
- **Why not primary:** gives "something changed 12 days ago," not "merchant entered the bust-out state." Named states are the business differentiator.
- Maturity: `Consensus`. Fit: **8/10**.

### GNN — rejected, signal stolen
- A 2026 survey (Springer, IWINAC) reports GNNs outperforming XGBoost by **12–25% AUROC** on fraud rings via relational modelling, with production architectures at <100ms latency and 10K+ TPS.
- Our typology #2 (laundering endpoint) **is** a graph problem and a GNN would probably win on it.
- **Rejected because:** GPU requirement; we would have to synthesise the graph ourselves, making any win circular; infeasible solo in 4 days.
- **Mitigation:** graph-derived scalar features as HMM emissions — payer-set entropy, repeat-payer ratio, payer-set Jaccard drift, Herfindahl concentration on payer volume. O(n) on CPU.
- **Say the rejection out loud in the video.** Naming the better method you couldn't build is a strength signal.
- Fit: **3/10** under our constraints.

### Sequence transformers — rejected
- NICE Actimize (a production fraud vendor's own research team), *Temporal Contrastive Transformer for Financial Crime Detection*, arXiv:2605.21490 (2026). Their honest finding: **"achieving performance comparable to a strong feature-engineered baseline is itself a meaningful outcome... while not yet production-ready."**
- A vendor's own scientists reporting parity with feature engineering is the most efficient possible justification for not spending 4 laptop-days on this.
- Maturity: `Emerging`, honestly self-reported as unproven. Fit: **2/10**.

### GBDT on windowed aggregates — the incumbent baseline
- Not exotic, and that is the point. This is what a competent team would actually ship. **If we cannot beat it, that is a finding, not a bug.**
- Fit as baseline: **10/10**. Fit as the project: 7/10.

---

## Part 3 — Cost-sensitive decisioning

- **Elkan (2001), IJCAI** — *The Foundations of Cost-Sensitive Learning.* The theorem that the optimal decision threshold is a function of the cost matrix. This is the mathematical justification for the entire second stage.
- **Bahnsen (2015), PhD, Univ. Luxembourg** — *Example-Dependent Cost-Sensitive Classification.* Motivating case is exactly ours: *"failing to detect a fraudulent transaction may have an economic impact from a few to thousands of Euros depending on the particular transaction and cardholder."*
- **Savings score** (Bahnsen et al. 2016, restated in arXiv:2005.02488): cost of using the algorithm relative to the cost of using no algorithm, where the no-algorithm cost is the minimum of always-negative and always-positive. **Use this named metric rather than inventing one.**

## Part 4 — Capacity-constrained deferral

This is the finding that most changes the project's positioning.

- **Learning to Defer** under capacity constraints is an *acknowledged open problem*, not a solved one. The L2D survey (arXiv:2206.13202) states: *"in the presence of capacity constraints the best decision-maker may not be the optimal choice... current L2D methods would simply assign them to every instance, disregarding capacity constraints... we believe this to be a blind spot and an opportunity for future research."*
- **FiFAR** (arXiv:2312.13218) — a public fraud dataset built specifically for this: 50 synthetic analysts with varied bias and feature dependence, plus *"a realistic definition of human work capacity constraints, an aspect of L2D systems that is often overlooked."*
- **Implication for the pitch:** our Q14 design choice sits on a recognised research frontier. Frame it honestly — *"the literature calls capacity management a blind spot; I implemented a practical version for merchant risk"* — not as novel research.

## Part 5 — Datasets

| Dataset | What it is | Use here | Caveat |
|---|---|---|---|
| **BAF** (Feedzai, NeurIPS 2022, `github.com/feedzai/bank-account-fraud`) | 6 synthetic tabular variants generated by CTGAN from an anonymised real bank account-opening fraud dataset; ~1M rows; 1.10% fraud prevalence in Base; 8 months of temporal drift; differential-privacy noise applied | **Validate the decision/cost layer** on realistic imbalance and real temporal dynamics | Account-opening, not merchant time series. Cannot train the HMM. |
| **FiFAR** (arXiv:2312.13218) | 50 synthetic analysts + explicit capacity constraints | Reference for the capacity framing; read the datasheet even if unused | Synthetic experts |
| **Own generator** | Merchant streams with 4 injected typologies + known transition timestamps | Train and evaluate the sequence layer | Circular by construction. Must be stated everywhere. |
| IEEE-CIS, PaySim, ULB | Considered | Not used — no merchant-level sequence structure | — |

## Part 6 — Library health (verified at source, 28 Aug 2026)

| Library | Version | Health | Verdict |
|---|---|---|---|
| `hmmlearn` | 0.3.3, released **31 Oct 2024** | README states *"under limited-maintenance mode"*; third-party analysis rates maintenance **Inactive**, no PyPI release in 12 months | **Do not use.** Hand-write the HMM. |
| `pymoo` | **0.6.2, released 27 Jun 2026** — restored CMA-ES/NumPy 2.x compatibility, added repo-wide type-checking and a behaviour-regression suite | Healthy, actively maintained | **Use.** NSGA-II. |
| `lightgbm` | 4.x | Healthy | Use. |
| `scikit-learn` | 1.5+ | Healthy | Use for metrics and splits only. |

## Part 7 — Failure archaeology

What killed prior attempts, and how we avoid each.

| Failure | Where observed | Our defence |
|---|---|---|
| **Cardholder HMMs drowned in false positives** | arXiv:1611.06439 survey | Merchant-level unit + cost layer + within-merchant standardisation |
| **Rigid rule engines break when fraudsters adapt** | Razorpay's own Bumblebee post | Latent-state model adapts to per-merchant baseline; rule engine kept as the baseline to beat |
| **Third-party screening at <10% precision** | Razorpay's own Bumblebee post | Precision reported at a stated operating point, with the FP cost attached |
| **Random splits inflate fraud results** | Standard in the literature; the most common self-deception in this domain | Temporal + merchant-group split enforced in `eval/splits.py`, tested |
| **Single-agent LLM pipelines hit token limits** | Bumblebee Phase 2 retrospective | We are not building an LLM pipeline. Non-issue, but informs the deterministic-first architecture. |
| **Freezing the eval after seeing results** | Framework doctrine | Eval frozen in `06-requirements.md` and `10-done.md` before any model is written |
| **Merchants cannot find out why funds are held** | Razorpay's own Feb 2026 post + public reviews | Viterbi path → reason string is a graded requirement (FR-014), not a nice-to-have |

## Stop rule

Reached. The last four sources confirmed existing ADRs without changing any. Further search would be budget set on fire.

## Cast note (not a skill)

A "payments-risk domain expert" role was invoked repeatedly during planning but fails the two-of-three test — the knowledge is captured here and would not transfer to a second project. Handled inline. **Recorded so it is not rebuilt.**
