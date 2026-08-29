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
          §5 was rewritten by T-0017 (2026-08-28): V_m is lifetime gross margin, L_m is
          realised loss, every primitive carries a citation and a range, and the 400–600
          asymmetry is a reported cross-check rather than a gate.
OPEN:     Six §5 primitives remain marked ASSUMPTION with ranges; FR-020 sweeps them.
          K (state count) set empirically by BIC sweep in T-0004.
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
| $L_m$ | **realised** loss if merchant m is fraudulent and passed | INR | $r_{cb}(1+\varphi)G^{bad}_m$, §5 |
| $V_m$ | merchant expected **lifetime gross margin** | INR | $g \cdot v_m \cdot \ell_m$, §5 |

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

## 5. Cost matrix — sourced, with ranges

> **AMENDMENT — two definitional fixes + cross-check demotion · dated 2026-08-28 · ticket T-0017.**
> This section previously (a) defined $V_m$ as one decision window's MDR revenue, (b) defined
> $L_m$ as gross volume transacted while in a bad state, and (c) closed with the instruction
> *"Our cost matrix should reproduce roughly this ratio at typical merchant parameters. **If it
> does not, the parameters are wrong — check this in T-0007.**"* All three are replaced below.
> **The two definitional fixes are justified by the definitions measuring the wrong quantity,
> and by nothing else.** They were written before the resulting ratio was computed, and the
> ratio is not permitted to justify them retroactively. The (c) instruction is retired because,
> followed literally, it means tuning parameters until a check passes — the identical practice
> `T-0016` forbids for the generator, and worse here because `savings` is the headline metric.

⚠️ **Several primitives below are still assumptions. They are marked `ASSUMPTION` with an
explicit range. FR-020's sensitivity analysis is what makes the headline claim defensible, not
the central values. State them as assumptions in the README and on camera.**

Per merchant $m$ per decision period:

| | True: healthy | True: fraudulent |
|---|---|---|
| **PASS** | 0 | $L_m$ — realised fraud loss |
| **REVIEW** | $c_{\text{rev}}$ | $c_{\text{rev}} + p_{\text{miss}} L_m$ |
| **HOLD** | $c_{\text{fp}}(m)$ — churn cost | $\rho L_m$ — residual leakage |

$$c_{\text{rev}} = \tau \cdot w_{\text{analyst}}$$
$$c_{\text{fp}}(m) = P(\text{churn}\mid\text{hold}) \cdot V_m + c_{\text{support}}$$

### Definitional fix 1 — $V_m$ is expected **lifetime** gross margin, not one window's revenue

$$\boxed{\;V_m \;=\; g \cdot v_m \cdot \ell_m\;}$$

| Symbol | Meaning | Unit |
|---|---|---|
| $g$ | platform gross margin per rupee of processed volume | dimensionless |
| $v_m$ | merchant's expected **monthly** gross processed volume | INR / month |
| $\ell_m$ | expected **remaining** merchant lifetime | months |

**Why the old definition was wrong, independent of any ratio.** A merchant who is held and
churns does not cost the platform one 30-day window's margin. They cost every rupee of margin
the platform would have earned from them for the rest of their life on the platform. The
previous form, $V_m = \texttt{MDR_RATE} \times$ window volume (`config.py:174`), is the margin
on a single window — it is a *revenue rate*, and $c_{\text{fp}}$ needs a *stock*. Multiplying a
rate by a lifetime is what converts one into the other.

**A second error sits inside the first: `MDR_RATE = 0.02` is a price, not a margin.** 2% is what
the merchant *pays*. Almost all of it leaves again as issuer interchange, scheme fees and GST.
The platform's own gross margin on a rupee of TPV is roughly **10 basis points**, not 200 — see
$g$ below. Using the merchant-facing MDR as the platform's gross margin overstates $V_m$ by
about 20×. Both errors were present at once and they pull in opposite directions, which is part
of why neither was visible in the aggregate.

### Definitional fix 2 — $L_m$ is realised loss, not turnover

$$\boxed{\;L_m \;=\; r_{\text{cb}} \cdot (1 + \varphi) \cdot G^{\text{bad}}_m\;}$$

| Symbol | Meaning | Unit |
|---|---|---|
| $G^{\text{bad}}_m$ | gross volume transacted by $m$ while in a bad state | INR |
| $r_{\text{cb}}$ | **realisation rate** — fraction of that volume returning as chargeback, confirmed-fraud write-off or unrecovered negative balance | dimensionless |
| $\varphi$ | ancillary loading — scheme dispute fees, representment handling, monitoring-programme penalties | dimensionless |

**Why the old definition was wrong, independent of any ratio.** The previous form counted the
merchant's **gross turnover while bad** as the loss. Turnover is not loss. A bust-out merchant
who processes ₹10,00,000 and has ₹50,000 charged back has cost the acquirer ₹50,000 plus fees,
not ₹10,00,000 — the remainder settled against genuine purchases or was recovered from the
merchant's own reserve. Charging full turnover inflates $L_m$ by more than an order of magnitude
and makes *every* policy score negative savings, both perfect-foresight oracles included, which
is exactly what T-0006 observed (knapsack oracle −0.678 against hold-everything's +0.573). A
ceiling beaten by a trivial policy is a symptom of a mis-specified loss, not of a bad oracle.

### Primitives — one citation and one range each

Source classes: **[S]** sourced, cited below · **[D]** derived from other rows · **[A]**
`ASSUMPTION` — no public source found, range stated.

| Parameter | Central | Plausible range | Class | Source / basis |
|---|---|---|---|---|
| $\tau$ | 0.067 h (≈4 min) | 0.05 – 0.12 h | [S] | Razorpay Engineering, Dec 2025 — stated per-review time. Cross-checks against the 700–800 analyst-hours/month figure in `00-charter.md §1`. |
| $w_{\text{analyst}}$ | ₹600 / h | ₹300 – ₹700 / h | [S+A] | PayScale India *Fraud Analyst* ₹4.19 L/yr, Glassdoor India ₹4.48 L/yr (2026) → ₹210–225/h at 2,000 h/yr. The fully-loaded multiplier of 1.5–1.8× (benefits, seat, QA, supervision) is **[A]**. The shipping default is deliberately held at the **upper** end of the band: an expensive analyst makes REVIEW look costly, which is conservative *against* Rakshak's own capacity story. |
| $c_{\text{rev}}$ | ≈ ₹40 | ₹15 – ₹84 | [D] | $\tau \cdot w_{\text{analyst}}$. |
| $g$ | 0.0010 (10 bps of TPV) | 0.0008 – 0.0015 | [S] | Razorpay FY24: revenue ≈ ₹2,501 Cr against annualised TPV ≈ US$180 bn → take rate ≈ **0.27% of TPV**; gross profit ₹906 Cr FY24 (₹1,277 Cr FY25) → gross margin ≈ **36% of revenue**. $g \approx 0.0027 \times 0.36$. Secondary sources reporting company disclosures, not an audited filing — flagged as such. |
| $v_m$ | per merchant | — | [D] | From the generator's merchant volume profile. Not a constant. |
| $\ell_m$ | 30 months | 18 – 48 months | [A] | **No public disclosure of Indian payment-aggregator merchant retention exists.** The range brackets a 2.1–5.6% monthly churn rate. This is the least-defensible number in the file and FR-020 must sweep it. |
| $P(\text{churn}\mid\text{hold})$ | 0.35 | 0.15 – 0.60 | [A] | Informed by the public-review pattern — frozen settlements are the most common Razorpay complaint (`00-charter.md §1`). A pattern, not a measurement. |
| $c_{\text{support}}$ | ₹500 | ₹200 – ₹1,500 | [A] | Escalation handling on a held merchant. Loosely bounded above by published dispute-handling fees — Visa VAMP's excessive tier charges US$8 ≈ ₹700 per dispute from 1 Apr 2026. |
| $r_{\text{cb}}$ | 0.05 | 0.02 – 0.20 | [A], bracketed by [S] | **Floor anchors, both cited.** Nilson puts *all-merchant* card fraud at **6.43¢ per US$100** (0.064% of volume, 2024) — a population floor far below any individual bad merchant. Card-scheme monitoring programmes define where a merchant becomes formally abnormal: Mastercard ECM **1.5%** of transactions, HECM **3.0%**; Visa VAMP "excessive" **1.5%** from 1 Apr 2026. A merchant Rakshak exists to catch is by construction above those, so the range starts at 2%. **Ceiling anchor:** a terminal bust-out window can approach total dispute, hence 20%. The central 0.05 sits just above the scheme-excessive boundary. **It was chosen from these anchors before the resulting FP-to-loss ratio was computed.** |
| $\varphi$ | 0.35 | 0.20 – 0.50 | [A], bracketed by [S] | LexisNexis Risk Solutions *True Cost of Fraud* puts total cost at **3.84× face value in India (2021)** and **3.95× in APAC (2023)** — but that multiplier includes internal labour and recovery effort, which this cost matrix already charges separately via $c_{\text{rev}}$ and $c_{\text{support}}$. Double-counting it would be an error, so $\varphi$ covers only scheme dispute fees, representment handling and monitoring penalties. The LexisNexis figure is therefore a **hard upper bound**, not the value. |
| $p_{\text{miss}}$ | 0.15 | 0.05 – 0.30 | [A] | Analyst miss rate on a reviewed-and-cleared merchant. No public source. |
| $\rho$ | 0.10 | 0.05 – 0.25 | [A] | Residual leakage between the hold decision and settlement actually stopping. |
| $L_m$ | per merchant | — | [D] | $r_{\text{cb}}(1+\varphi)\,G^{\text{bad}}_m$; typology-dependent through $G^{\text{bad}}_m$. |
| $V_m$ | per merchant | — | [D] | $g \, v_m \, \ell_m$. |

**Citations — all retrieved 2026-08-28. Re-verify before the video.**

1. Nilson Report, *Card Fraud Losses Worldwide — 2024*: 6.43¢ per US$100 (down from 6.58¢),
   US$33.41 bn total. <https://nilsonreport.com/articles/card-fraud-losses-worldwide-2024/>
2. Card-scheme monitoring thresholds — Mastercard ECM 1.5% / 100 cases, HECM 3.0% / 300 cases;
   Visa VAMP replaced VDMP and VFMP on 1 Apr 2025, and its "excessive" tier drops to 1.5% with
   a US$8-per-dispute fee on 1 Apr 2026.
   <https://www.chargeflow.io/blog/chargeback-threshold-limits> ·
   <https://solidgate.com/blog/monitoring-programs/>
3. LexisNexis Risk Solutions, *True Cost of Fraud — Asia Pacific*: S$3.95 per S$1 (2023 study,
   released Apr 2024); India US$3.84 per US$1 (2021 study).
   <https://risk.lexisnexis.com/global/en/about-us/press-room/press-release/20240429-tcof-apac>
4. Razorpay published pricing — 2% + GST domestic standard, 3% premium/international.
   <https://razorpay.com/blog/razorpay-payment-gateway-pricing-explained/>
5. Razorpay FY24 financials — revenue ₹2,501 Cr, gross profit ₹906 Cr (FY25 ₹1,277 Cr),
   annualised TPV ≈ US$180 bn.
   <https://entrackr.com/fintrackr/razorpay-payment-gateway-biz-crosses-rs-2000-cr-revenue-in-fy24-pat-soars-5x-7370662>
   · <https://inc42.com/company/razorpay/financials/>
6. Analyst compensation — PayScale India *Fraud Analyst* ₹4.19 L/yr; Glassdoor India ₹4.48 L/yr
   (2026). <https://www.payscale.com/research/IN/Job=Fraud_Analyst/Salary> ·
   <https://www.glassdoor.co.in/Salaries/fraud-analyst-salary-SRCH_KO0,13.htm>
7. RBI *Annual Report* — card/internet and digital-payment fraud counts and amounts, FY24–FY26.
   Used as a directional check that digital-payment fraud in India is material and volatile
   year on year, **not** as a per-merchant rate.
8. Bahnsen, Aouada & Ottersten (2015) *Example-Dependent Cost-Sensitive Decision Trees*;
   (2016) *Feature Engineering Strategies for Credit Card Fraud Detection* — the savings score
   and the example-dependent cost-matrix formulation used in §6.
9. Elkan (2001) *The Foundations of Cost-Sensitive Learning* — the optimal threshold is a
   function of the cost matrix, not a hyperparameter.

### The 400–600 asymmetry — a reported cross-check, **not a gate**

Indian payments commentary estimates **₹400–600 lost to falsely declined legitimate orders for
every ₹100 saved by preventing fraud** (`00-charter.md §1`). It is the asymmetry that motivates
the project.

**It is commentary. It is not a measurement and it carries no per-primitive provenance.** The
band is therefore demoted from a gate to a **cross-check that is computed and reported, never
closed.**

The obligation, binding on T-0007a and T-0011:

1. **Compute** the FP-cost-per-₹100-of-fraud-loss ratio that the cited primitives above produce,
   at central values and across the stated ranges.
2. **Report it** in `results/` and in the README, with its inputs.
3. **State any divergence from 400–600 rather than closing it.** If the sourced primitives
   produce 280, the repo says "our sourced primitives produce 280 against a commentary figure of
   400–600, and here is why the two differ." It does **not** move a primitive to reach 400.
4. Any primitive that changes after today changes because its **source** changed, and the change
   is recorded in `LOGBOOK.md` with the new source. Never because of the ratio.

For orientation only: under the *old* definitions the ratio was **13.4 per ₹100** (T-0006). Both
fixes move it upward by construction — $L_m$ falls by roughly
$1/[r_{\text{cb}}(1+\varphi)] \approx 15\times$ and $V_m$ rises by roughly
$g\,\ell_m / \texttt{MDR_RATE} \approx 1.5\times$ — so a central figure in the low hundreds is
expected, with the stated ranges spanning roughly 70 to 700. **That expectation is written down
here so that a measured value far from it is visible as a surprise rather than absorbed
silently. It is not a target and nothing may be tuned toward it.**

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
