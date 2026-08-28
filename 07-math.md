<!-- HEAD
FILE:     07-math.md
PHASE:    2 — SPECIFY
UPDATED:  2026-08-28
STATUS:   gated
SUMMARY:  Seven sections: HMM (forward/backward/Viterbi/Baum-Welch, all log-space), BOCPD
          run-length recursion, within-merchant standardisation, empirical-Bayes shrinkage,
          the cost matrix with sourced provisional values in INR, Bayes Minimum Risk plus the
          Bahnsen savings score, and the NSGA-II objective vector with the capacity constraint.
          Every symbol carries a unit. Validity ranges and numerical-stability notes included.
          §5 cost values are PROVISIONAL — FR-020 sensitivity analysis is what defends them.
OPEN:     Cost values provisional. K (state count) set empirically by BIC sweep in T-0004.
-->

# 07 — Math

An equation without units is a bug that hasn't happened yet.

---

## 0. Symbol table

| Symbol | Meaning | Unit | Typical range |
|---|---|---|---|
| $m$ | merchant index | — | 1 … M |
| $M$ | number of merchants | count | 2,000 – 10,000 |
| $t$ | window index within a merchant | window | 1 … T_m |
| $T_m$ | number of windows for merchant m | count | 20 – 400 |
| $K$ | number of latent states | count | 2 – 6, provisionally 4 |
| $D$ | emission dimensionality | count | 13 – 16 |
| $z_t$ | latent risk state at window t | categorical | {1..K} |
| $\mathbf{x}_t$ | emission vector at window t | standardised, dimensionless | roughly N(0,1) per dim |
| $A$ | transition matrix, $A_{ij} = P(z_t=j \mid z_{t-1}=i)$ | probability | rows sum to 1 |
| $\pi$ | initial state distribution | probability | sums to 1 |
| $\mu_k, \Sigma_k$ | emission mean and covariance for state k | dimensionless | — |
| $\gamma_t(k)$ | posterior $P(z_t = k \mid \mathbf{x}_{1:t})$ (filtered) | probability | [0,1] |
| $C_a$ | cost of taking action a | INR | see §5 |
| $B$ | review budget per period | analyst-hours | 400 – 800 |
| $\tau$ | review duration per merchant | hours | 0.067 (≈4 min) |
| $L_m$ | expected loss if merchant m is fraudulent and passed | INR | AOV-dependent |
| $V_m$ | merchant lifetime value | INR | AOV × volume × margin |

---

## 1. Hidden Markov Model

### Model
$$P(\mathbf{x}_{1:T}, z_{1:T}) = \pi_{z_1} \prod_{t=2}^{T} A_{z_{t-1} z_t} \prod_{t=1}^{T} \mathcal{N}(\mathbf{x}_t \mid \mu_{z_t}, \Sigma_{z_t})$$

Gaussian emissions with diagonal covariance. Diagonal, not full: with $D \approx 15$ and per-state sample counts in the hundreds, a full covariance is under-determined and Baum-Welch will produce singular matrices.

### Forward recursion (log space — mandatory)
$$\alpha_1(k) = \log \pi_k + \log \mathcal{N}(\mathbf{x}_1 \mid \mu_k, \Sigma_k)$$
$$\alpha_t(k) = \log \mathcal{N}(\mathbf{x}_t \mid \mu_k, \Sigma_k) + \operatorname{logsumexp}_{j} \left[ \alpha_{t-1}(j) + \log A_{jk} \right]$$

Sequence log-likelihood: $\log P(\mathbf{x}_{1:T}) = \operatorname{logsumexp}_k \alpha_T(k)$

**This recursion is the online belief update.** It is what the design originally proposed RL for; it is closed form, provably optimal under the model, and runs in $O(K^2)$ per new observation.

Filtered posterior: $\gamma_t(k) = \exp\left[\alpha_t(k) - \operatorname{logsumexp}_j \alpha_t(j)\right]$

### Backward recursion
$$\beta_T(k) = 0, \qquad \beta_t(k) = \operatorname{logsumexp}_{j}\left[\log A_{kj} + \log \mathcal{N}(\mathbf{x}_{t+1} \mid \mu_j, \Sigma_j) + \beta_{t+1}(j)\right]$$

### Viterbi — the audit trail
$$\delta_1(k) = \log \pi_k + \log \mathcal{N}(\mathbf{x}_1 \mid \mu_k,\Sigma_k), \qquad \delta_t(k) = \log \mathcal{N}(\mathbf{x}_t \mid \mu_k,\Sigma_k) + \max_j \left[\delta_{t-1}(j) + \log A_{jk}\right]$$

with backpointers $\psi_t(k) = \arg\max_j [\delta_{t-1}(j) + \log A_{jk}]$, then backtrace from $\arg\max_k \delta_T(k)$.

**The MAP path $z^*_{1:T}$ is the explanation artifact of FR-014.** The transition index is where the reason string points.

### Baum-Welch (EM)
E-step: $\xi_t(i,j) \propto \exp[\alpha_t(i) + \log A_{ij} + \log \mathcal{N}(\mathbf{x}_{t+1}\mid\mu_j,\Sigma_j) + \beta_{t+1}(j)]$, normalised over $(i,j)$.

M-step, pooled across all merchants in a segment:
$$\hat{A}_{ij} = \frac{\sum_m \sum_t \xi^{(m)}_t(i,j)}{\sum_m \sum_t \sum_{j'} \xi^{(m)}_t(i,j')}, \qquad \hat\mu_k = \frac{\sum_m \sum_t \gamma^{(m)}_t(k)\,\mathbf{x}^{(m)}_t}{\sum_m \sum_t \gamma^{(m)}_t(k)}$$

**Numerical stability notes — implement these, they are not optional:**
- All quantities in log space; use a stable `logsumexp` with max-subtraction.
- Add $\epsilon = 10^{-6}$ to the diagonal of $\Sigma_k$ every M-step, or a low-occupancy state will collapse to a singular covariance.
- Floor $A_{ij}$ at $10^{-8}$ and renormalise, so no transition becomes permanently impossible.
- If a state's total occupancy $\sum_t \gamma_t(k) < 1$, reinitialise that state from the global mean plus noise rather than letting EM die.
- Stop on relative log-likelihood improvement $< 10^{-4}$ or 100 iterations, whichever first.

**Validity range:** requires $T_m \gtrsim 10 K$ per merchant for stable per-merchant inference. Below that, rely entirely on the pooled segment fit. This is exactly the failure that sank the 2008 cardholder HMMs, which had $T \approx 10$ with $K \ge 2$.

### Model selection
$$\mathrm{BIC}(K) = -2\log \hat{L} + p(K)\log N, \qquad p(K) = \underbrace{K-1}_{\pi} + \underbrace{K(K-1)}_{A} + \underbrace{2KD}_{\mu,\,\mathrm{diag}\Sigma}$$
Sweep $K \in \{2,\dots,6\}$ on the **validation window only**. Report the curve; do not assert $K=4$.

---

## 2. BOCPD (baseline)

Run-length posterior with hazard $H(r) = 1/\lambda$:
$$P(r_t \mid \mathbf{x}_{1:t}) \propto \sum_{r_{t-1}} P(r_t \mid r_{t-1})\, P(\mathbf{x}_t \mid r_{t-1}, \mathbf{x}^{(r)})\, P(r_{t-1}\mid \mathbf{x}_{1:t-1})$$

Growth: $r_t = r_{t-1}+1$ with probability $1-H$. Changepoint: $r_t = 0$ with probability $H$.

Truncate the run-length posterior at $L = 90$ windows for bounded cost and constant memory.
Conjugate Normal-Inverse-Gamma predictive per dimension, independence assumed across dimensions.
Alarm when $P(r_t < 5) > \theta$.

**Why it is the baseline and not the model:** it answers "something changed ~12 days ago." It cannot answer "the merchant entered the bust-out state," which is what FR-014 requires.

---

## 3. Within-merchant standardisation (FR-007)

For merchant $m$, feature $d$, using only the burn-in window $t \in [1, t_0]$:
$$\tilde{x}^{(m)}_{t,d} = \frac{x^{(m)}_{t,d} - \hat\mu^{(m)}_d}{\hat\sigma^{(m)}_d + \epsilon}$$

with location and scale themselves shrunk toward the segment $s(m)$ for short histories:
$$\hat\mu^{(m)}_d = w_m \bar{x}^{(m)}_d + (1-w_m)\,\bar{x}^{(s)}_d, \qquad w_m = \frac{n_m}{n_m + n_0}$$

$n_m$ = merchant's burn-in observation count; $n_0$ = shrinkage constant, provisionally 30.

**Why this is the most important equation in the file:** it makes the model measure deviation from the merchant's *own* norm rather than from a population norm. A ₹300-AOV merchant and a ₹80,000-AOV merchant become directly comparable in emission space, which is what makes one pooled HMM legitimate across a heterogeneous population.

**Burn-in must be strictly before the evaluation window** or this leaks. Enforce in `eval/splits.py`.

---

## 4. Empirical-Bayes shrinkage of cost parameters (ADR-0006)

For a per-merchant parameter $\theta_m$ (e.g. fraud base rate, or the FP-cost multiplier), with segment-level hyperparameters estimated across merchants:
$$\hat\theta_m = w_m \hat\theta^{\text{MLE}}_m + (1-w_m)\,\hat\theta_{s(m)}, \qquad w_m = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{between}} + \sigma^2_{\text{within}}/n_m}$$

James–Stein / partial-pooling form. $\sigma^2_{\text{between}}$ from the variance of segment members' MLEs; $\sigma^2_{\text{within}}$ from the within-merchant sampling variance.

Properties that make this the right answer to the cold-start objection:
- $n_m \to \infty \Rightarrow w_m \to 1$: high-volume merchants converge to their own economics.
- $n_m \to 0 \Rightarrow w_m \to 0$: new merchants inherit their segment's.
- Continuous in between — no arbitrary cliff at a minimum-volume gate.

**Demo artifact:** animate one merchant's threshold migrating from segment default to its own economics as volume accumulates.

---

## 5. Cost matrix — PROVISIONAL

⚠️ **These are assumptions. FR-020's sensitivity analysis is what makes them defensible. State them as assumptions in the README and on camera.**

Per merchant $m$ per decision period:

| | True: healthy | True: fraudulent |
|---|---|---|
| **PASS** | 0 | $L_m$ — full fraud loss |
| **REVIEW** | $c_{\text{rev}}$ | $c_{\text{rev}} + p_{\text{miss}} L_m$ |
| **HOLD** | $c_{\text{fp}}(m)$ — churn cost | $\rho L_m$ — residual leakage |

$$c_{\text{rev}} = \tau \cdot w_{\text{analyst}} \qquad c_{\text{fp}}(m) = P(\text{churn}\mid\text{hold}) \cdot V_m + c_{\text{support}}$$

| Parameter | Provisional value | Source / basis |
|---|---|---|
| $\tau$ | 0.067 h (≈4 min) | Razorpay Engineering, Dec 2025 — stated per-review time |
| $w_{\text{analyst}}$ | ₹600/h | Assumption. Indian risk-ops fully-loaded cost. **Flag as assumption.** |
| $c_{\text{rev}}$ | ≈ ₹40 | Derived |
| $P(\text{churn}\mid\text{hold})$ | 0.35 | Assumption informed by the public-review pattern. **Flag as assumption.** |
| $c_{\text{support}}$ | ₹500 | Assumption — escalation handling |
| $p_{\text{miss}}$ | 0.15 | Analyst miss rate. Assumption. |
| $\rho$ | 0.10 | Residual leakage before a hold takes effect |
| $L_m$ | AOV × ramp multiplier × exposure window | Typology-dependent, from the generator |

**The asymmetry that motivates the whole project:** Indian payments commentary estimates ₹400–600 lost to falsely declined legitimate orders for every ₹100 saved by preventing fraud. Our cost matrix should reproduce roughly this ratio at typical merchant parameters. **If it does not, the parameters are wrong — check this in T-0007.**

---

## 6. Bayes Minimum Risk and the savings score

### Optimal action
Given calibrated $p_m = P(\text{fraudulent} \mid \mathbf{x}^{(m)}_{1:t})$ from the HMM posterior over "bad" states:
$$a^*_m = \arg\min_{a \in \{\text{PASS},\text{REVIEW},\text{HOLD}\}} \left[ p_m \cdot C_a(\text{fraud}) + (1-p_m) \cdot C_a(\text{healthy}) \right]$$

Elkan (2001): the optimal threshold is a function of the cost matrix, not a hyperparameter to tune. **This is the mathematical justification for the entire decision stage.** In the binary case the PASS/HOLD boundary sits at $p^*_m = c_{\text{fp}}(m) / (L_m + c_{\text{fp}}(m) - \rho L_m)$, which is explicitly merchant-dependent.

### Total cost and savings (Bahnsen et al. 2016)
$$\text{Cost}(f) = \sum_m \left[ y_m\,C_{a_m}(\text{fraud}) + (1-y_m)\,C_{a_m}(\text{healthy}) \right]$$
$$\text{Cost}_{\ell} = \min\big(\text{Cost}(\text{all PASS}),\ \text{Cost}(\text{all HOLD})\big)$$
$$\boxed{\;\text{Savings}(f) = \frac{\text{Cost}_{\ell} - \text{Cost}(f)}{\text{Cost}_{\ell}}\;}$$

Use this **named, citable** metric. An invented formula invites "why this formula?"; a cited one moves the conversation on.

**Guard (AP-06):** the savings score is manipulable via the cost matrix. Always report PR-AUC alongside it.

---

## 7. Policy optimisation (ADR-0004, ADR-0005)

Decision variables: per-segment threshold pairs $(\theta^{\text{rev}}_s, \theta^{\text{hold}}_s)$ for $s = 1..S$, shrunk to per-merchant thresholds by §4.

**Objectives (all minimised):**
$$f_1 = \mathbb{E}[\text{fraud loss}] = \sum_m (1-\mathbb{1}[a_m \neq \text{PASS}])\, y_m L_m + \sum_m \mathbb{1}[a_m=\text{HOLD}]\, y_m \rho L_m$$
$$f_2 = \mathbb{E}[\text{false-positive cost}] = \sum_m \mathbb{1}[a_m = \text{HOLD}](1-y_m)\, c_{\text{fp}}(m)$$
$$f_3 = \text{review hours} = \tau \sum_m \mathbb{1}[a_m = \text{REVIEW}]$$

**Constraint:** $f_3 \le B$

Three objectives, so **NSGA-II** (ADR-0004). Reference directions and NSGA-III are for four or more.

**The obligation ADR-0004 creates:** implement the uncoupled per-segment grid search as a baseline. If NSGA-II's coupled solution does not dominate it in hypervolume, NSGA-II is decoration — remove it and keep the grid search. This is the ablation that converts a GA from a resume line into a measured result.

### Oracle ceiling
Given full hindsight $y_m$ and true transition times, allocate $B$ hours to maximise loss averted:
$$\max_{S \subseteq \mathcal{M}} \sum_{m \in S} y_m L_m \quad \text{s.t.} \quad \tau|S| \le B$$
With unit review cost this is exactly solvable by sorting on $y_m L_m$ descending and taking the top $\lfloor B/\tau \rfloor$. **Report gap-to-oracle, not unanchored absolutes.**

---

## 8. Detection lag

$$\text{lag}_m = t^{\text{first flag}}_m - t^{\text{true transition}}_m \quad \text{[windows]} \times \text{window length [days]}$$

Report the median over merchants that were both truly bad and eventually flagged, with the flagged fraction stated alongside — median lag over a small flagged subset is meaningless without it.

**Why it earns its place:** lag maps directly to money. Loss exposure grows roughly linearly in lag during a bust-out ramp, so "we detect 11 days earlier" is convertible to rupees, and rupees is the risk-ops lead's unit (P-09).
