"""External-dataset procurement and the empirical calibration profile (T-0015).

Nothing in here trains or scores a model. It exists to answer one question the rest of
the repo cannot answer about itself: **how far do the generator's hand-chosen marginals
sit from marginals measured on real transaction data?**

`06-requirements.md:28` and ADR-0007 already establish that no public merchant-sequence
dataset with merchant-level risk labels exists. Public data therefore supplies marginals
and rates only — never sequence structure and never merchant-level labels. Those remain
entirely synthetic and `results/calibration_gap.md` says so in terms.
"""
