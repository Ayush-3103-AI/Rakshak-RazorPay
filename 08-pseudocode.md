<!-- HEAD
FILE:     08-pseudocode.md
PHASE:    2 — SPECIFY
UPDATED:  2026-08-28
STATUS:   gated
SUMMARY:  Pseudocode for the six algorithms where choice matters: the leakage-guarded splitter,
          the typology generator, within-merchant feature standardisation, the HMM (log-space
          forward/Viterbi/Baum-Welch with degeneracy handling), the three-action cost policy,
          and the capacity-constrained NSGA-II frontier with its grid-search baseline. Each
          carries typed I/O, complexity, and failure/numerical-stability notes. CRUD and
          plumbing are deliberately not pseudocoded.
OPEN:     none
-->

# 08 — Pseudocode

Only algorithms where *choice* matters. Plumbing is not pseudocoded; that's ceremony.

---

## A. `eval/splits.py` — the leakage guard

**This module is the single point of truth for splits. Nothing else in the repo may call `train_test_split`.**

```
FUNCTION make_splits(transactions: DataFrame, cfg: Config) -> Splits
  INPUT   transactions with columns [merchant_id, timestamp, ...]
  OUTPUT  Splits(train, val, test) — each a DataFrame + a merchant_id set

  # 1. Temporal boundaries — fixed in config, never derived from results
  t_train_end = cfg.month_boundary(6)
  t_val_end   = cfg.month_boundary(7)

  # 2. Merchant-group partition — disjoint by construction, seeded
  merchants   = sorted(unique(transactions.merchant_id))      # sorted → deterministic
  rng         = Generator(cfg.seed)
  shuffled    = rng.permutation(merchants)
  m_train, m_val, m_test = split_proportions(shuffled, [0.6, 0.15, 0.25])

  # 3. Intersect BOTH constraints
  train = transactions[ (ts <= t_train_end)              & (merchant_id in m_train) ]
  val   = transactions[ (t_train_end < ts <= t_val_end)  & (merchant_id in m_val)   ]
  test  = transactions[ (ts >  t_val_end)                & (merchant_id in m_test)  ]

  # 4. ASSERT, do not trust
  ASSERT m_train ∩ m_val == ∅
  ASSERT m_train ∩ m_test == ∅
  ASSERT m_val  ∩ m_test == ∅
  ASSERT max(train.ts) < min(val.ts) < min(test.ts)

  RETURN Splits(train, val, test)
```

**Complexity:** O(N log N).
**Failure notes:** if any assertion fails, raise — never warn. A warning gets ignored at 2 a.m. on Monday. If a split ends up empty (too few merchants), raise with the counts in the message rather than silently proceeding.
**Why it is written this way:** AP-01 is the single most likely cause of an inflated, indefensible result.

---

## B. `generator/` — merchant streams with injected typologies

```
FUNCTION generate(cfg: GenConfig, seed: int) -> (transactions, ground_truth)

  rng = Generator(seed)
  FOR each merchant m in 1..M:
      segment  = sample_segment(rng)                    # MCC × AOV band
      profile  = MerchantProfile(
                   aov_mu     = lognormal per segment,
                   volume_mu  = lognormal per segment,
                   payer_pool = size ~ f(volume),
                   repeat_rate, refund_rate, hour_profile, method_mix )

      typology = sample_typology(rng, cfg.prevalence)   # HEALTHY | one of 5

      IF typology == HEALTHY:
          state_path = [0] * T
      ELSE:
          t_switch   = rng.integers(cfg.burn_in + 10, T - 10)   # after burn-in, before end
          state_path = [0]*t_switch + [state_of(typology)]*(T - t_switch)
          RECORD ground_truth(m, typology, t_switch, timestamp_of(t_switch))

      FOR t in 1..T:
          params = apply_typology(profile, typology, t, t_switch)
          EMIT transactions for window t from params

  RETURN transactions, ground_truth
```

**Typology transforms** — each must be separably detectable by at least one emission (FR-004):

| Typology | Transform after `t_switch` |
|---|---|
| `BUST_OUT` | volume × ramp(t) rising to 5–10×; AOV × 0.3–0.6; new-payer ratio → ~0.95; hard stop after 3–6 windows |
| `LAUNDERING_ENDPOINT` | volume × 2–4; AOV distribution *unchanged* (this is what makes it hard); payer pool → effectively unbounded; repeat-payer ratio → ~0; refund rate → ~0; hour entropy rises |
| `CATEGORY_DRIFT` | AOV shifts to a different segment's distribution; method mix shifts; hour profile shifts; volume roughly constant |
| `REFUND_COLLUSION` | refund rate 0.05 → 0.35; refunds concentrated in a payer subset of size 5–20; chargeback lag shortens |
| `SLOW_RAMP` (adversarial) | same terminal state as `BUST_OUT` but ramp spread over 40–60 windows, each step below any plausible single-step detection threshold |

**Failure notes:** clamp all sampled rates to [0,1]; ensure `t_switch + min_post_window <= T` or the transition is unobservable and the merchant is silently unlabelable; assert every non-healthy merchant has exactly one recorded transition.
**Safety (FR-006):** module docstring states this is an evaluation artifact. Parameters are calibrated for *detectability testing*, not for evasion. `SLOW_RAMP` exists solely to be reported as a failure mode.

---

## C. `features/` — window features and within-merchant standardisation

```
FUNCTION build_emissions(txns, cfg) -> ndarray[M, T, D]

  FOR each merchant m:
      windows = resample(txns[m], freq=cfg.window)          # default 7 days

      FOR each window w:
          raw[m,w] = {
            log_aov_mean, log_aov_var, velocity,
            refund_ratio, chargeback_ratio, chargeback_lag_mean,
            hour_entropy, method_mix_entropy,
            # graph-derived scalars (ADR-0002 mitigation)
            payer_entropy       = shannon(payer volume shares),
            repeat_payer_ratio  = |payers seen before| / |payers|,
            payer_jaccard_prev  = |P_w ∩ P_{w-1}| / |P_w ∪ P_{w-1}|,
            payer_herfindahl    = Σ (share_i)²,
            new_payer_ratio,
            # optional Vulcan proxy (FR-010)
            vulcan_mean, vulcan_p95        # omitted + logged if column absent
          }

      # within-merchant standardisation — FR-007, math §3
      burn = raw[m, 1 : cfg.burn_in]
      ASSERT burn window ends strictly before the evaluation window   # leakage guard
      n_m  = count(burn)
      w    = n_m / (n_m + cfg.n0)
      mu   = w * mean(burn)  + (1-w) * segment_mean[segment(m)]
      sd   = w * std(burn)   + (1-w) * segment_std[segment(m)]
      emissions[m] = (raw[m] - mu) / (sd + eps)

  RETURN emissions
```

**Complexity:** O(N) over transactions plus O(M·T·|payers|) for the graph scalars. On 5,000 merchants × 100 windows this is seconds, not minutes.
**Failure notes:** a window with zero transactions → forward-fill the previous window's features and set a `sparse` flag emission; do not drop it, or the sequence indices desynchronise from ground truth. Zero-variance features in the burn-in → `sd` falls back entirely to the segment value.

---

## D. `models/hmm.py` — hand-written, log space throughout

```
CLASS HMM(K: int, D: int)
  FIELDS  log_pi[K], log_A[K,K], mu[K,D], var[K,D]      # diagonal covariance only

  FUNCTION _log_emission(X[T,D]) -> [T,K]
      # diagonal Gaussian log-pdf, vectorised over t and k
      RETURN -0.5 * Σ_d [ log(2π var[k,d]) + (X[t,d]-mu[k,d])² / var[k,d] ]

  FUNCTION forward(X) -> (alpha[T,K], loglik)
      alpha[0] = log_pi + log_emission[0]
      FOR t in 1..T-1:
          alpha[t] = log_emission[t] + logsumexp(alpha[t-1][:,None] + log_A, axis=0)
      RETURN alpha, logsumexp(alpha[T-1])

  FUNCTION filter_online(x_new, alpha_prev) -> alpha_new
      # O(K²). THIS is the "update belief with each transaction" behaviour.
      RETURN log_emission(x_new) + logsumexp(alpha_prev[:,None] + log_A, axis=0)

  FUNCTION viterbi(X) -> path[T]
      delta[0] = log_pi + log_emission[0]
      FOR t in 1..T-1:
          scores   = delta[t-1][:,None] + log_A
          psi[t]   = argmax(scores, axis=0)
          delta[t] = log_emission[t] + max(scores, axis=0)
      backtrace from argmax(delta[T-1])
      RETURN path

  FUNCTION fit(sequences: list, max_iter=100, tol=1e-4)
      initialise: k-means on pooled emissions → mu; var = pooled variance;
                  A = 0.9 diagonal + uniform off-diagonal; pi = uniform
      REPEAT:
          # E-step over ALL sequences (pooled across the segment)
          accumulate gamma, xi via forward + backward
          # M-step
          log_A  = normalise(Σ xi, axis=1)
          mu     = weighted_mean(X, gamma)
          var    = weighted_var(X, gamma, mu) + REG          # REG = 1e-6
          log_pi = normalise(Σ gamma[0])
          # degeneracy handling — REQUIRED
          FOR k where occupancy(k) < 1.0:
              reinitialise mu[k] = global_mean + rng.normal(scale=0.1)
              var[k] = global_var
          floor log_A at log(1e-8); renormalise rows
      UNTIL Δloglik / |loglik| < tol OR max_iter
```

**Complexity:** forward/backward/Viterbi O(TK²); Baum-Welch O(I·M·T·K²). With K=4, T=100, M=5000, I=50 this is ~4×10⁸ flops — seconds in vectorised numpy.
**Failure notes, all mandatory:**
- `logsumexp` must subtract the max before exponentiating, or long sequences underflow to −inf.
- Without the variance regulariser, a low-occupancy state collapses to a delta function and the log-likelihood diverges to +inf. This *will* happen without `REG`.
- Baum-Welch is only guaranteed to reach a local optimum. Run 5 restarts with different seeds and keep the best validation log-likelihood.
- Assert monotonic log-likelihood increase in tests. A decrease means an M-step bug, always.
- `fit` on an empty sequence list → raise, don't return an untrained model.

---

## E. `decision/policy.py` — three-action cost-minimising policy

```
FUNCTION decide(gamma_t[K], merchant: MerchantParams, cfg) -> Decision

  # 1. Collapse latent states into a fraud posterior
  p_fraud = Σ_{k ∈ cfg.bad_states} gamma_t[k]

  # 2. Merchant-specific costs — empirical-Bayes shrunk (math §4)
  L    = merchant.expected_loss
  c_fp = merchant.churn_prob * merchant.ltv + cfg.c_support
  c_rev = cfg.tau * cfg.analyst_wage

  # 3. Expected cost of each action  (math §5 matrix)
  E_pass   = p_fraud * L
  E_review = c_rev + p_fraud * cfg.p_miss * L
  E_hold   = p_fraud * cfg.rho * L + (1 - p_fraud) * c_fp

  action = argmin({PASS: E_pass, REVIEW: E_review, HOLD: E_hold})

  RETURN Decision(action, p_fraud, {E_pass, E_review, E_hold}, reason=None)
```

**Complexity:** O(K) per merchant.
**Failure notes:** if `p_fraud` is NaN (degenerate posterior), default to REVIEW and log it — never silently PASS. If two expected costs tie within 1e-9, prefer the less severe action; document the tiebreak.
**Sanity check for T-0007:** at typical merchant parameters, the implied ratio of false-positive cost to fraud-prevention benefit should land near the ₹400–600 per ₹100 figure from `03-landscape.md`. If it doesn't, the parameters in `07-math.md §5` are wrong.

---

## F. `optimize/nsga.py` — capacity-constrained frontier + the baseline that justifies it

```
FUNCTION grid_baseline(val_data, cfg) -> ParetoSet
  # Uncoupled: each segment optimised independently, budget split pro-rata by volume
  FOR each segment s:
      FOR (θ_rev, θ_hold) in grid(cfg.grid_resolution):
          evaluate f1, f2, f3 on segment s
  combine → pareto_filter
  RETURN frontier_grid

FUNCTION nsga_frontier(val_data, cfg) -> ParetoSet
  # Coupled: all segments jointly, ONE global budget — this is what justifies the GA
  problem = Problem(
      n_var  = 2 * S,                                  # (θ_rev, θ_hold) per segment
      n_obj  = 3,                                      # f1 loss, f2 fp-cost, f3 hours
      n_ieq  = 1,                                      # f3 - B <= 0
      xl = 0.0, xu = 1.0,
      evaluate = λ x: simulate_policy(x, val_data, cfg) )

  algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)
  res = minimize(problem, algorithm, ('n_gen', 200), seed=cfg.seed)
  RETURN pareto_filter(res.F, res.X)

FUNCTION justify_ga(frontier_nsga, frontier_grid) -> bool
  # ADR-0004 obligation. If this returns False, DELETE the GA and ship the grid search.
  RETURN hypervolume(frontier_nsga) > hypervolume(frontier_grid)
```

**Complexity:** each evaluation is O(M); 100 pop × 200 gen = 20,000 evaluations. Precompute per-merchant posteriors once outside the loop or this becomes the runtime bottleneck and NFR-004 fails.
**Failure notes:** if the constraint is infeasible for every individual (budget too small for any useful policy), pymoo returns all-infeasible — detect this and report the minimum feasible budget rather than an empty frontier. Cache `simulate_policy` on the threshold vector; NSGA re-evaluates duplicates.
**The important line is `justify_ga`.** It is the difference between a measured result and decoration (AP-04).
