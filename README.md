# Rakshak — post-onboarding merchant risk sentinel

> This README is a stub. T-0013 writes the real one, with results and provenance.
> The section below is required by FR-006 and was drafted in T-0003.

## Scope and Safety

**Defense only.** Rakshak is a detector. Every component in this repository exists to identify
risk on a merchant portfolio a payment aggregator already owns, and to price the cost of acting
on it. There is no component that probes, enumerates, or exploits any payment system. Any
Razorpay API usage, if it appears at all, is test-mode only.

**The synthetic generator is an evaluation artifact, not a fraud toolkit.**
`src/rakshak/generator/` produces labelled synthetic merchant transaction streams so that
detection can be *measured* against ground truth we control. It writes rows to a local parquet
file. It has no payment-system client, no credential handling, and makes no network calls.

Its five "typologies" are coarse statistical caricatures pitched at the level a risk analyst
would sketch on a whiteboard — a volume ramp, a flat payer graph, a ticket-size shift, a refund
rate. They encode no operational tradecraft and produce nothing usable against a live system.
Read as an attack recipe, the module says only that fraud changes a merchant's statistics, which
is the premise of every fraud-detection paper ever published. The four realistic typologies plus
the deliberately-hard fifth (`SLOW_RAMP`) exist to be *reported against*, including where
Rakshak fails to catch them.

**Synthetic data is labelled synthetic, everywhere.** Sequence-layer metrics are measured on
synthetic merchant streams with injected typologies; the generator is in this repo. The decision
layer is additionally validated on BAF (Feedzai, NeurIPS 2022), a public benchmark derived from
real bank data.

**Explicitly out of scope:** deepfake or KYC document analysis, RTO/COD return fraud, and any
capability that could be repurposed offensively. See `03-landscape.md` for why.

### Regenerating the synthetic dataset

```
python -m rakshak.generator --seed 42
```

Writes `data/synthetic/transactions.parquet` and `data/synthetic/state_paths.parquet`
(git-ignored). At the defaults — 500 merchants, 270 days, seed 42 — this is 747,006
transactions in about 3.5 seconds on a laptop CPU.

### How synthetic is the synthetic data? — the measured answer

`results/calibration_gap.md` (produced by `python -m rakshak.data.profile --seed 42`, T-0015)
compares eleven marginals of the generator against the same marginals measured on **Online
Retail II** (UCI 502, CC BY 4.0) — a real transaction stream, downloaded, hashed and
manifested at `data/external/online_retail_ii.manifest.json`.

**Of the eight ratio-scale marginals, five diverge by 1.9x or more and four by 5x or more;
the widest is 32.5x.** One divergence is structural rather than parametric: real daily
transaction counts are over-dispersed (Fano factor 12.3) while the generator draws them from
a Poisson process, whose Fano factor is 1.0 by construction. No choice of rate can close that.

**What the profile cannot inform, and what therefore remains entirely synthetic and entirely
uncalibrated:** latent risk states and their transitions, merchant-level fraud labels and
prevalence, the chargeback rate, the cross-merchant payer graph, MCC/category structure and
absolute ticket size, and every typology dynamic (bust-out, laundering, category drift, refund
collusion, slow ramp). `06-requirements.md:28` and ADR-0007: no public merchant-sequence
dataset with merchant-level risk labels exists, so these cannot be calibrated against anything.
The marginals above are the generator's surface statistics; its sequence structure and its
labels — the things the HMM is actually scored on — are calibrated against nothing.
