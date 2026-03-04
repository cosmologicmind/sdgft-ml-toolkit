# Observable Derivation Chain

SDGFT derives all 37 observables through a strict hierarchical chain. Information flows **downward only** — no backward dependencies. This DAG (directed acyclic graph) structure is what the GNN surrogate mirrors.

## Level Map

```
Level 0:  Inputs        (Δ, δ_g, φ)
              │
Level 1:  Constants      sin²30° = 1/4
              │
Level 2:  Dimension      D*, D*_fp, n, n_fp
              │
Level 3:  Gravity        αM, αB, η_slip
              │
Level 4:  Inflation      Nₑ, ns, r, ε, η, βiso
              │
Level 5:  Cosmology      Ωb, Ωc, ΩDE, Ωm, wDE, ηB, S₈
              │
Level 6:  Particles      α⁻¹em, αs, sin²θW, mH, mμ/me, mτ/mμ,
                          Ngen, θ₁₂, θ₂₃, θ₁₃, |Vus|, |Vub|
```

---

## Level 2: Effective Dimension (4 observables)

These set the stage for everything downstream.

| Observable | Symbol | Formula | Axiom Value |
|------------|--------|---------|-------------|
| Tree-level dimension | D*_tree | 3 − sin²30° + δ_g = 3 − 1/4 + δ_g | 67/24 ≈ 2.7917 |
| Fixed-point dimension | D*_fp | D* = Δ^{−1/D*} · φ · Δ^{Δ·δ_g} (iterated) | 2.7970 |
| Tree-level f(R) exponent | n_tree | D*_tree / 2 | 67/48 ≈ 1.3958 |
| Fixed-point f(R) exponent | n_fp | D*_fp / 2 | 1.3985 |

```python
# In ParametricForward:
d_star_tree = 3.0 - 0.25 + delta_g                   # D*_tree
d_star_fp = fixed_point_iteration(delta, delta_g, phi) # D*_fp
n_tree = d_star_tree / 2                              # n_tree
n_fp = d_star_fp / 2                                  # n_fp
```

The fixed-point iteration converges in ~15 iterations to machine precision:

```
D_{k+1} = Δ^{-1/D_k} · φ · Δ^{Δ·δ_g}
```

---

## Level 3: Modified Gravity (4 observables)

The f(R) gravity framework introduces modified gravity parameters.

| Observable | Symbol | Formula | Axiom Value |
|------------|--------|---------|-------------|
| Running Planck mass rate | α_M | (n−1)/(2n−1) | 0.2210 |
| Braiding | α_B | −αM/2 | −0.1105 |
| Gravitational slip (survey) | η_slip(k/aH=10) | (1+2Bx²)/(1+4Bx²), B=n−1 | 0.9720 |
| Gravitational slip (horizon) | η_slip(k/aH=1) | same with x=1 | 0.7175 |

```python
alpha_m = (n - 1) / (2*n - 1)
alpha_b = -alpha_m / 2
eta_slip = (1 + 2*b*x**2) / (1 + 4*b*x**2)   # b = n-1, x = k/(aH)
```

The gravitational slip η measures the ratio of the two metric potentials Φ/Ψ. At different scales:
- **Horizon scale** (k/aH = 1): significant deviation from GR (η = 0.72)
- **Survey scale** (k/aH = 10): small deviation (η = 0.97)
- **Sub-horizon** (k/aH → ∞): approaches GR (η → 1/2)

---

## Level 4: Inflation (6 observables)

All inflationary observables derive from the f(R) exponent n and the number of e-folds Nₑ.

| Observable | Symbol | Formula | Axiom Value | Experiment |
|------------|--------|---------|-------------|------------|
| E-folds | N_e | (D*/Δ)·ln[(D*−2−δ_g)/(Δ·δ_g)] | 59.95 | 55–65 required |
| Spectral index | n_s | 1 − 2(2n−1)/[Nₑ(2n−1)+n] | 0.9671 | 0.9649 ± 0.0042 |
| Tensor-to-scalar ratio | r | 48(2n−1)²/[Nₑ(2n−1)+n]² | 0.0135 | < 0.036 |
| Isocurvature fraction | β_iso | 1/36 | 0.0278 | < 0.038 |
| Slow-roll ε | ε_SR | (2n−1)(n−2)²/[Nₑ(2n−1)+n]² | ~10⁻⁴ | — |
| Slow-roll η | η_SR | (2n−1)(2n²−7n+4)/[…]² − (2n−1)/[…] | ~−0.016 | — |

```python
n_e = (d_star / delta) * math.log((d_star - 2 - delta_g) / (delta * delta_g))
nbar = 2*n - 1
denom = n_e * nbar + n
n_s = 1 - 2*nbar / denom
r = 48 * nbar**2 / denom**2
epsilon_sr = nbar * (n - 2)**2 / denom**2
eta_sr = nbar*(2*n**2 - 7*n + 4)/denom**2 - nbar/denom
beta_iso = 1/36  # geometric constant: (1/6)²
```

**Key insight**: The spectral index formula for f(R)^n gravity has the same functional form as the standard Starobinsky R² model — but with n ≈ 1.4 instead of n = 2, yielding slightly different predictions.

---

## Level 5: Cosmology (7 observables)

The energy budget of the universe is a **combinatorial partition** of the 24-cell's 2304 = 96 × 24 total states.

| Observable | Symbol | Formula | Axiom Value | Experiment |
|------------|--------|---------|-------------|------------|
| Baryon density | Ω_b | (Δ/4)(1−δ_g) | 0.0499 | 0.0493 ± 0.0020 |
| CDM density | Ω_c | 6Δ² | 0.260 | 0.265 ± 0.007 |
| Dark energy density | Ω_DE | 1 − Ωb − Ωc | 0.690 | 0.6847 ± 0.0073 |
| Total matter | Ω_m | Ωb + Ωc | 0.310 | 0.3153 ± 0.0073 |
| DE equation of state | w_DE | −D*/3 | −0.932 | −1.03 ± 0.03 ⚠️ |
| Baryon asymmetry | η_B | δ_g⁶(1−δg)/8 | 6.27 × 10⁻¹⁰ | 6.143 ± 0.190 × 10⁻¹⁰ |
| Clustering amplitude | S₈ | σ₈·√(Ωm/0.3), σ₈=0.775 | 0.788 | 0.832 ± 0.013 |

```python
omega_b = (delta / 4) * (1 - delta_g)
omega_c = 6 * delta**2
omega_de = 1 - omega_b - omega_c    # flatness by construction
omega_m = omega_b + omega_c
w_de = -d_star / 3
eta_b = delta_g**6 * (1 - delta_g) / 8
sigma_8 = 0.775  # MCMC anchor
s_8 = sigma_8 * math.sqrt(omega_m / 0.3)
```

### The Energy Partition

The fractions can be expressed as exact integers over 2304:

| Component | Formula | Fraction of 2304 | Decimal |
|-----------|---------|-------------------|---------|
| Baryonic matter | 115/2304 | 115 states | 0.0499 |
| Dark matter | 600/2304 | 600 states | 0.260 |
| Dark energy | 1589/2304 | 1589 states | 0.690 |
| **Total** | **2304/2304** | **2304** | **1.000** |

Cosmic flatness (Ωtot = 1) is not a dynamical fine-tuning — it is a **combinatorial identity**.

### ⚠️ The w_DE Tension

The main tension: w_DE = −D*/3 ≈ −0.932 vs. observed −1.03 ± 0.03 (~3σ). This is SDGFT's most important testable prediction — DESI/Euclid will improve the measurement to σ(w₀) < 0.01, definitively confirming or excluding the predicted deviation from −1.

---

## Level 6: Particle Physics (10 observables)

### Electroweak Couplings

| Observable | Symbol | Formula | Axiom Value | Experiment |
|------------|--------|---------|-------------|------------|
| Fine structure const.⁻¹ | α_em⁻¹ | 2π(D*)³ + δ_g·D* | 136.82 | 137.036 |
| Strong coupling | α_s | √2/12 | 0.1179 | 0.1180 ± 0.0009 |
| Weak mixing angle | sin²θ_W | 1/9 + γ_EW | 0.2312 | 0.23122 ± 0.00003 |

```python
alpha_em_inv = 2*pi*d_star**3 + delta_g*d_star  # tree-level
alpha_s = sqrt(2) / 12                           # geometric constant
sin2_theta_w = 1/9 + gamma_ew                    # gamma_ew = 0.12011 (RG)
```

Note: γ_EW = 0.12011 is the only external numerical anchor in SDGFT (an electroweak RG correction). The tree-level value sin²θ_W = 1/9 ≈ 0.1111 receives a 108% RG correction to reach 0.2312.

### Mass Ratios & Higgs

| Observable | Symbol | Formula | Axiom Value | Experiment |
|------------|--------|---------|-------------|------------|
| Muon/electron mass | m_μ/m_e | 3/(2α_em) + 1 + Δ | 206.76 | 206.768 |
| Tau/muon mass | m_τ/m_μ | 6D* | 16.75 | 16.817 |
| Higgs quartic | λ | Δ/φ | 0.1287 | 0.1291 ± 0.0020 |
| Higgs mass | m_H | √(2λ)·v_H | 125.30 | 125.25 ± 0.17 |
| Generations | N_gen | max{n : φⁿ < Δ/δ_g} | **3** (exact) | 3 |

```python
mu_e = 3/(2*alpha_em) + 1 + delta
tau_mu = 6 * d_star
lambda_geo = delta / phi
higgs_mass = sqrt(2 * lambda_geo) * v_higgs      # v_higgs = 246.22 GeV
n_gen = max{n : phi^n < 5}                        # phi¹=1.62, phi²=2.62, phi³=4.24 < 5, phi⁴=6.85 > 5 → 3
```

**The generation formula**: Why exactly 3 families? The Fibonacci stability condition asks: how many powers of φ fit below Δ/δ_g = 5? Since φ³ = 4.236 < 5 but φ⁴ = 6.854 > 5, the answer is exactly 3. This purely geometric argument replaces the unexplained "3 generations" of the Standard Model.

---

## Level 6: Neutrinos (3 observables)

Starting from tri-bimaximal mixing and applying geometric corrections:

| Observable | Symbol | Formula | Axiom Value | Experiment |
|------------|--------|---------|-------------|------------|
| Solar angle | θ₁₂ | arctan(1/√2)·(1−δ_g) | 33.7° | 33.44 ± 0.77° |
| Atmospheric angle | θ₂₃ | 45°·(1+Δ/√6) | 48.8° | 49.2 ± 1.0° |
| Reactor angle | θ₁₃ | arcsin(Δ/√2) | 8.47° | 8.57 ± 0.12° |

```python
theta_12 = degrees(atan(1/sqrt(2)) * (1 - delta_g))         # TBM + lattice correction
theta_23 = 45 * (1 + delta / sqrt(6))                       # maximal + conflict correction
theta_13 = degrees(asin(delta / sqrt(2)))                    # conflict-generated reactor angle
```

The neutrino mixing angles start from the **tri-bimaximal mixing** (TBM) pattern and receive perturbative corrections proportional to Δ and δ_g. The reactor angle θ₁₃ — zero in TBM and discovered to be non-zero at Daya Bay (2012) — is predicted as arcsin(Δ/√2), directly proportional to the conflict parameter.

---

## Level 6: CKM Matrix (3 observables)

| Observable | Symbol | Formula | Axiom Value | Experiment |
|------------|--------|---------|-------------|------------|
| |V_us| | — | √Ω_b | 0.2234 | 0.2243 ± 0.0005 |
| |V_ub| | — | Δ^φ · δ_g · exp(δ_g·ln(τ_e)/φ²) | 0.00383 | 0.00382 ± 0.00020 |
| Quark hierarchy | m_c/m_u | exp(2π) | 535.5 | ~550 ± 100 |

```python
v_us = sqrt(omega_b)
v_ub = delta**phi * delta_g * exp(delta_g * log(tau_e_ratio) / phi**2)
quark_hierarchy = exp(2*pi)  # ≈ 535.5
```

The CKM-cosmology connection |V_us| = √Ω_b is one of SDGFT's most striking crossdomain predictions — it links a particle physics quantity (quark mixing) to a cosmological one (baryon density) through the same geometric origin.

---

## Complete Observable List (37 entries)

The ordered list matches the GNN's 37-node DAG:

| # | Key | Level | Domain |
|---|-----|-------|--------|
| 0 | d_star_tree | 2 | Dimension |
| 1 | d_star_fp | 2 | Dimension |
| 2 | n_tree | 2 | Dimension |
| 3 | n_fp | 2 | Dimension |
| 4 | alpha_m_tree | 3 | Gravity |
| 5 | alpha_b_tree | 3 | Gravity |
| 6 | eta_slip_survey | 3 | Gravity |
| 7 | eta_slip_horizon | 3 | Gravity |
| 8 | n_efolds_fp | 4 | Inflation |
| 9 | n_s | 4 | Inflation |
| 10 | r_tensor | 4 | Inflation |
| 11 | beta_iso | 4 | Inflation |
| 12 | epsilon_sr | 4 | Inflation |
| 13 | eta_sr | 4 | Inflation |
| 14 | omega_b | 5 | Cosmology |
| 15 | omega_c | 5 | Cosmology |
| 16 | omega_de | 5 | Cosmology |
| 17 | omega_m | 5 | Cosmology |
| 18 | w_de_fp | 5 | Cosmology |
| 19 | eta_b | 5 | Cosmology |
| 20 | s_8 | 5 | Cosmology |
| 21 | alpha_em_inv_tree | 6 | Particle |
| 22 | alpha_em_inv_fp | 6 | Particle |
| 23 | alpha_em_tree | 6 | Particle |
| 24 | alpha_s | 6 | Particle |
| 25 | sin2_theta_w | 6 | Particle |
| 26 | mu_e_ratio | 6 | Particle |
| 27 | tau_mu_ratio_tree | 6 | Particle |
| 28 | lambda_geo | 6 | Particle |
| 29 | higgs_mass | 6 | Particle |
| 30 | n_generations | 6 | Particle |
| 31 | theta_12 | 6 | Neutrino |
| 32 | theta_23 | 6 | Neutrino |
| 33 | theta_13 | 6 | Neutrino |
| 34 | v_us | 6 | CKM |
| 35 | v_ub | 6 | CKM |
| 36 | quark_hierarchy | 6 | CKM |

---

**Previous:** [← Dimensional Flow](03_dimensional_flow.md) | **Next:** [Experimental Validation →](05_experimental_validation.md)
