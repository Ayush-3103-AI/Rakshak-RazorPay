# G1-G5 on cycle-4 data — 2026-09-02

`uv run pytest tests/gates -q`, 24 tests: 20 passed, 4 skipped (BAF absent).
The GREEN/RED words below are the *gate report*, not test outcomes — G5 reports its
excess as a finding rather than failing the suite.

> **RESOLVED 2026-09-02 - the observation below is retired, and the caution was
> correct.** The controlled comparison has now been run
> (`scripts/g5_cycle_comparison.py`): the cycle-3 and cycle-4 configs produce
> **bit-identical** G5 numbers (raw +7.07pp, residual +2.70pp under both), so there
> is no cycle-3 -> cycle-4 effect here at all. The real discrepancy is between
> `LIMITATIONS.md` section 6 and the gate as it currently stands, and it is written
> up as **section 6a**, where the conclusion - not just the numbers - inverts. What
> follows is what was originally observed, and why it was not claimed.

## The one thing that moved, and it is not yet a claim

On cycle-4 data the RAW detector is RED at **+7.07pp** worst-window excess while the
cohort-residual detector is RED at **+2.70pp** — the residual cuts the worst window by
62%. `LIMITATIONS.md` §6 records the opposite for cycle 3: *"G5 is green for the RAW
detector too — the demo premise does not hold."*

**This is NOT yet evidence that §6 is overturned, and must not be reported as such.**
G5 runs at `prevalence=0`, so no typology is assigned and the onset-window change cannot
act on it directly — but it does move the RNG stream, so this is a different random
realisation as well as a different config. Establishing the effect needs G5 re-run under
the cycle-3 config with the identical reporting, which has not been done. Until it is,
this is an observation with a plausible mechanism (40,000 merchants makes cohorts larger
and the residual a better instrument) and no controlled comparison behind it.

`tests/gates/gates_report.py::CONFIG_PATH` is a hardcoded constant, so the comparison
requires swapping `configs/scenario_v2.yaml`, not an env var.

## Full report

```
.ssss...................                                                 [100%]
============================== warnings summary ===============================
tests/gates/test_g5_confounder_null.py::test_g5_confounder_null
  D:\Projects\razorpay-project\ver-2\v3\.venv\Lib\site-packages\numpy\lib\_nanfunctions_impl.py:1215: RuntimeWarning: All-NaN slice encountered
    return fnb._ureduce(a, func=_nanmedian, keepdims=keepdims,

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

==============================================================================
GENERATOR PARITY GATES - 08-generator-v2-spec.md section 7
==============================================================================
G1a fano                   GREEN  realised Fano = 13.141 vs target 12.25 (+/- 1.0)
                                  measured over 1200 merchants x 365 observed days, on the composed intensity (persona x typology x confounder)
G1b baf-parity             SKIP   BAF dataset not present
                                  set RAKSHAK_BAF_PATH (e.g. at data/external/baf.zip) to enable. The external anchor is the only real-derived data this project has; a SKIP here is a materially weaker claim than a GREEN and is reported as such.
G1c baf-overdispersion     SKIP   BAF dataset not present
                                  this is the single highest-value external check available to the project � whether real count data rejects the Poisson assumption v1 made � and without the anchor it cannot be made. Set RAKSHAK_BAF_PATH to enable.
G1d baf-prevalence         SKIP   BAF dataset not present
                                  the real fraud base rate and the real temporal drift are the two things BAF supplies without needing an analogue argument. Set RAKSHAK_BAF_PATH.
G2 baseline-transfer       SKIP   BAF dataset not present
                                  the gate cannot be evaluated at all on this machine; charter K-3 is unaddressed. Set RAKSHAK_BAF_PATH to enable. Note that when it IS present the gate reports NOT WELL-POSED � see the module docstring.
G3 determinism             GREEN  sha256 2c798b1840b590bc... == 2c798b1840b590bc...
                                  400 merchants, seed 1234, all five tables
G3b no-global-rng          GREEN  0 bare np.random.* call(s) in src/rakshak/generator/
                                  every stochastic function takes rng
G4 no-leakage              GREEN  0 forbidden reference(s) across 20 file(s) in features/models
                                  quarantined: ['GroundTruth', 'drift_onset_at', 'is_unreported', 'persona_id', 'risk_typology_id', 'true_loss_amount_inr']
G4b point-in-time          GREEN  28 features x 36 merchants (281,762 real events) agree online vs offline at every epoch
                                  recomputed from real generator output, not a synthetic stream - T-120 found a warmup bug that survived the synthetic check because both runners were wrong
G5 cohort-residual P1      RED    alert rate 0.0320 vs nominal 0.0050 (excess +2.70pp, allowed +2pp)
                                  days 93-98, feature txn_count
G5 cohort-residual P1      GREEN  alert rate 0.0162 vs nominal 0.0050 (excess +1.12pp, allowed +2pp)
                                  days 308-313, feature txn_count
G5 cohort-residual P2      GREEN  alert rate 0.0000 vs nominal 0.0050 (excess -0.50pp, allowed +2pp)
                                  days 57-58, feature auth_fail_rate
G5 cohort-residual P2      GREEN  alert rate 0.0025 vs nominal 0.0050 (excess -0.25pp, allowed +2pp)
                                  days 197-198, feature auth_fail_rate
G5 cohort-residual P2      GREEN  alert rate 0.0058 vs nominal 0.0050 (excess +0.08pp, allowed +2pp)
                                  days 290-291, feature auth_fail_rate
G5 cohort-residual P3      GREEN  alert rate 0.0029 vs nominal 0.0050 (excess -0.21pp, allowed +2pp)
                                  days 122-136, feature instrument_mix
G5 cohort-residual P4      GREEN  alert rate 0.0071 vs nominal 0.0050 (excess +0.21pp, allowed +2pp)
                                  days 182-212, feature new_instrument_share
G5 cohort-residual P5      GREEN  alert rate 0.0045 vs nominal 0.0050 (excess -0.05pp, allowed +2pp)
                                  days 243-253, feature cnp_share
G5 cohort-residual P6      GREEN  alert rate 0.0025 vs nominal 0.0050 (excess -0.25pp, allowed +2pp)
                                  days 11-34, feature txn_count
G5 cohort-residual SUMMARY RED    worst window excess +2.70pp; quiet-day rate 0.0050
                                  prevalence=0, so every alert here is a false positive by construction
G5 raw P1                  RED    alert rate 0.0757 vs nominal 0.0050 (excess +7.07pp, allowed +2pp)
                                  days 93-98, feature txn_count
G5 raw P1                  RED    alert rate 0.0320 vs nominal 0.0050 (excess +2.70pp, allowed +2pp)
                                  days 308-313, feature txn_count
G5 raw P2                  GREEN  alert rate 0.0000 vs nominal 0.0050 (excess -0.50pp, allowed +2pp)
                                  days 57-58, feature auth_fail_rate
G5 raw P2                  GREEN  alert rate 0.0025 vs nominal 0.0050 (excess -0.25pp, allowed +2pp)
                                  days 197-198, feature auth_fail_rate
G5 raw P2                  GREEN  alert rate 0.0058 vs nominal 0.0050 (excess +0.08pp, allowed +2pp)
                                  days 290-291, feature auth_fail_rate
G5 raw P3                  GREEN  alert rate 0.0024 vs nominal 0.0050 (excess -0.26pp, allowed +2pp)
                                  days 122-136, feature instrument_mix
G5 raw P4                  GREEN  alert rate 0.0075 vs nominal 0.0050 (excess +0.25pp, allowed +2pp)
                                  days 182-212, feature new_instrument_share
G5 raw P5                  GREEN  alert rate 0.0039 vs nominal 0.0050 (excess -0.11pp, allowed +2pp)
                                  days 243-253, feature cnp_share
G5 raw P6                  GREEN  alert rate 0.0027 vs nominal 0.0050 (excess -0.23pp, allowed +2pp)
                                  days 11-34, feature txn_count
G5 raw SUMMARY             RED    worst window excess +7.07pp; quiet-day rate 0.0050
                                  prevalence=0, so every alert here is a false positive by construction
==============================================================================

```
