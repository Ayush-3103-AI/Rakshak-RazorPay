# ADR-0002 — No GNN and no sequence transformer; approximate the graph signal with scalars

**Status:** Accepted — decision taken in Phase 2 (pre-execution). **Written retrospectively on
2026-08-29** from `CLAUDE.md`'s rejection table, `03-landscape.md` (GNN and transformer
sections), `01-understanding.md` D-07, and `04-patterns.md` P-05.
**Supersedes:** none.

## Context

Two model families would plausibly beat the chosen architecture and both are rejected. Recording
*why* is the point: `04-patterns.md` P-05 makes "name the better method you could not build" a
deliberate credibility move, not an omission to be hidden.

**Graph neural networks.** A 2026 Springer/IWINAC survey reports GNNs outperforming XGBoost by
**12–25% AUROC** on fraud rings, with production architectures under 100 ms at 10K+ TPS. Typology
#2 in this repo (laundering endpoint) **is** a graph problem, and a GNN would probably win on it.

**Sequence transformers.** Razorpay itself has a transformer-based payments foundation model
(Aug 2026). But NICE Actimize — a production fraud vendor's own research team — reports in
*Temporal Contrastive Transformer for Financial Crime Detection* (arXiv:2605.21490, 2026) that
they achieved *"performance comparable to a strong feature-engineered baseline… while not yet
production-ready."* That is a vendor publishing parity against feature engineering.

Against both: **CPU only, no GPU, not even Colab** (`CLAUDE.md` hard constraints), solo, four
build days.

## Options considered

**(a) Build a heterogeneous GNN.** Needs a GPU we do not have. Worse, the evaluation would be
**circular**: the only merchant×payer graph available is the one this repo's generator writes, so
a GNN would be scored on how well it learned our own graph assumptions. A win would prove nothing.

**(b) Build a sequence transformer.** Same GPU problem, and the strongest available evidence —
from a vendor with production data we do not have — says the expected gain over feature
engineering is parity.

**(c) Approximate the graph signal with graph-derived scalar features on CPU.** Payer-set entropy,
repeat-payer ratio, payer-set Jaccard similarity against the previous window, and Herfindahl
concentration on payer volume. These are computable per merchant per window in closed form.

## Decision

(c), and **say (a) out loud**. FR-008 implements the four scalars; T-0004 built them. The video
script and README state that the correct long-term answer is a heterogeneous GNN and that it was
approximated under stated constraints.

## Consequences

* **The laundering-endpoint typology is detected with a weaker instrument than the literature
  recommends**, and any per-typology result for it must be read in that light.
* **The scalars are per-merchant, not cross-merchant.** The generator scopes payers to a single
  merchant by design, so no cross-merchant collusion structure exists to detect. This is stated
  in `results/calibration_gap.md` as one of the marginals the empirical profile cannot inform.
* **The rejection is a pitch asset, not an apology.** Stating the constraint and the
  approximation converts a limitation into evidence of judgement (P-05). Framing it as anything
  other than a deliberate trade would be a mistake.
* **Neither family may be reintroduced during the build window.** If a GPU appeared tomorrow the
  circularity objection to (a) would still stand — it is an evaluation-validity problem, not a
  compute problem.
