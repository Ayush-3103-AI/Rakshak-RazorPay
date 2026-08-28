<!-- HEAD
FILE:     02-stakeholders.md
PHASE:    1b — UNDERSTAND
UPDATED:  2026-08-28
STATUS:   gated
SUMMARY:  Five stakeholders on the panel side, each with the metric they actually measure and
          the thing that makes them say no. The binding audience is the Head of Risk Ops, who
          measures alert precision, analyst hours, and merchant-complaint volume — not AUROC.
          The ML reviewer's veto is leakage. The eng reviewer's veto is an unreproducible repo.
          Vocabulary table at the end is the wording to reuse verbatim in README and video.
OPEN:     none
-->

# 02 — Stakeholders

The "users" of this project are the people deciding whether to hire. Write for them.

| Stakeholder | Role in decision | What they actually measure | Their vocabulary | What makes them say no |
|---|---|---|---|---|
| **Head of Risk Operations** | The binding audience. If they lean forward, you're hired. | Alert precision, false-positive cost in ₹, analyst hours consumed, merchant-complaint volume, time-to-detection | "alert volume", "review queue", "funds on hold", "merchant escalation", "chargeback ratio", "VAMP threshold" | A demo that ignores the cost of being wrong. A model with no operating point. Anything that would add to their queue without saying by how much. |
| **ML / DS reviewer** | Technical veto | Split methodology, calibration, ablations, whether baselines were honestly run | "leakage", "held-out", "PR-AUC vs ROC-AUC", "class imbalance", "temporal split", "ablation" | **Leakage.** A random split. ROC-AUC quoted on a 1% positive rate. A single number with no baseline. Synthetic data presented as if it were real. |
| **Engineering reviewer** | Technical veto | Can they clone it and run it? Is the code readable? Does it do what the README claims? | "reproducible", "one command", "dependencies", "seeds", "CI" | A repo that doesn't run. Numbers in the README that no script produces. A dead dependency. |
| **Founder / senior leader** (if present) | Judgement call on the person | Does this candidate understand our business? Did they read what we built? | "Vulcan", "Bumblebee", "merchant experience", "success rate", "our scale" | Pitching something Razorpay already shipped. Generic fintech-fraud framing with the company name swapped in. |
| **The merchant** (absent, represented) | The party the system acts on | Whether they can find out why their money is held, and what would release it | "why is my settlement on hold", "nobody told me what document they need" | A model that produces a score with no reason attached. This is the stakeholder everyone else's submission will forget. |

## The one column that matters

**"What they measure."** Two failure modes kill technically-fine projects here:

1. Reporting AUROC to a risk-ops lead, who thinks in alert volume and analyst hours. Translate every result into their unit before it goes on camera.
2. Reporting rupees saved to an ML reviewer without showing the split methodology, which reads as either naive or evasive.

The README and the video must carry **both translations of every headline number.**

## Vocabulary to reuse verbatim

Reuse this wording. It is drawn from Razorpay's own published material and from the track brief; using their words is the cheapest possible signal that the homework was done.

| Use this | Not this |
|---|---|
| "funds on hold" | "account suspension" |
| "merchant review queue" | "alert backlog" |
| "analyst hours" | "operational cost" |
| "false-positive cost" | "type I error" |
| "the merchant can see why" | "model interpretability" |
| "risk case" | "incident" |
| "post-onboarding" | "downstream" |
| "one class of loss" | "fraud vertical" |
| "measured precision and recall on a held-out test set" | "strong performance" |

## Anti-persona

The reviewer who has seen forty submissions that all say *"we used AI to detect fraud, achieving 97% accuracy."* Every design choice in this project should be legible as a deliberate refusal to be that submission — the temporal split, the reported failure typology, the cost frontier instead of a single threshold, the baseline that might win.
