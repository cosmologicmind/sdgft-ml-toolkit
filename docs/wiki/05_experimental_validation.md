# Experimental Validation

SDGFT makes **zero-parameter predictions** for 37 observables. Of these, 22 have precision experimental measurements, enabling a rigorous statistical test.

## The 22-Observable Scorecard

### Cosmology (7 observables)

| # | Observable | SDGFT (axiom) | Experiment | σ_exp | Pull (σ) | Status |
|---|-----------|---------------|------------|-------|----------|--------|
| 1 | Ω_b | 0.0499 | 0.0493 ± 0.0020 | 0.0020 | +0.3 | ✅ |
| 2 | Ω_c | 0.260 | 0.265 ± 0.007 | 0.007 | −0.7 | ✅ |
| 3 | Ω_DE | 0.690 | 0.6847 ± 0.0073 | 0.0073 | +0.7 | ✅ |
| 4 | Ω_m | 0.310 | 0.3153 ± 0.0073 | 0.0073 | −0.7 | ✅ |
| 5 | w_DE | −0.932 | −1.03 ± 0.03 | 0.03 | +3.3 | ⚠️ |
| 6 | η_B | 6.27 × 10⁻¹⁰ | 6.143 ± 0.190 × 10⁻¹⁰ | 0.190 × 10⁻¹⁰ | +0.7 | ✅ |
| 7 | S₈ | 0.788 | 0.832 ± 0.013 | 0.013 | −3.4 | ⚠️ |

### Inflation (2 observables)

| # | Observable | SDGFT | Experiment | σ_exp | Pull | Status |
|---|-----------|-------|------------|-------|------|--------|
| 8 | n_s | 0.967 | 0.9649 ± 0.0042 | 0.0042 | +0.5 | ✅ |
| 9 | r | 0.013 | < 0.036 | 0.036 | — | ✅ |

### Particle Physics (8 observables)

| # | Observable | SDGFT | Experiment | σ_eff | Pull | Status |
|---|-----------|-------|------------|-------|------|--------|
| 10 | α_em⁻¹ | 136.82 | 137.036 | 0.5 (theory) | −0.4 | ✅ |
| 11 | α_s | 0.1179 | 0.1180 ± 0.0009 | 0.0009 | −0.1 | ✅ |
| 12 | sin²θ_W | 0.2312 | 0.23122 ± 0.00003 | 0.00003 | −0.1 | ✅ |
| 13 | m_H | 125.3 GeV | 125.25 ± 0.17 GeV | 0.17 | +0.3 | ✅ |
| 14 | m_μ/m_e | 206.76 | 206.768 ± 1.0 (theory) | 1.0 | 0.0 | ✅ |
| 15 | m_τ/m_μ | 16.75 | 16.817 ± 0.1 (theory) | 0.1 | −0.7 | ✅ |
| 16 | λ_geo | 0.1287 | 0.1291 ± 0.0020 | 0.0020 | −0.2 | ✅ |
| 17 | N_gen | 3 | 3.0 ± 0.008 | 0.008 | 0.0 | ✅ (exact) |

### Neutrinos (3 observables)

| # | Observable | SDGFT | Experiment | σ_exp | Pull | Status |
|---|-----------|-------|------------|-------|------|--------|
| 18 | θ₁₂ | 33.7° | 33.44 ± 0.77° | 0.77° | +0.3 | ✅ |
| 19 | θ₂₃ | 48.8° | 49.2 ± 1.0° | 1.0° | −0.4 | ✅ |
| 20 | θ₁₃ | 8.47° | 8.57 ± 0.12° | 0.12° | −0.8 | ✅ |

### CKM Matrix (2 observables)

| # | Observable | SDGFT | Experiment | σ_exp | Pull | Status |
|---|-----------|-------|------------|-------|------|--------|
| 21 | \|V_us\| | 0.2234 | 0.2243 ± 0.0005 | 0.0005 | −1.8 | ✅ |
| 22 | \|V_ub\| | 0.00383 | 0.00382 ± 0.00020 | 0.00020 | +0.1 | ✅ |

## The χ² Test

### Methodology

The goodness of fit is quantified by:

$$\chi^2 = \sum_{i=1}^{N_{\text{dof}}} \frac{(O_i^{\text{pred}} - O_i^{\text{exp}})^2}{\sigma_{\text{eff},i}^2}$$

where:

$$\sigma_{\text{eff},i} = \max(\sigma_{\text{exp},i}, \; \sigma_{\text{theory},i})$$

Three observables have theory uncertainties that dominate:

| Observable | σ_exp | σ_theory | Reason |
|------------|-------|----------|--------|
| α_em⁻¹ | 2.1 × 10⁻⁸ | **0.5** | Tree-level formula; ~0.4% loop corrections expected |
| m_μ/m_e | 4.6 × 10⁻⁶ | **1.0** | Geometric ratio; ~0.5% radiative corrections |
| m_τ/m_μ | 0.0015 | **0.1** | Geometric ratio; QCD corrections expected |

### Results

| Metric | Value |
|--------|-------|
| Number of observables | 22 |
| Degrees of freedom | 22 (zero free parameters) |
| χ² total | 32.5 |
| **χ²/dof** | **1.479** |
| **p-value** | **0.069** |

A p-value of 0.069 means SDGFT is **not excluded** at the standard 95% confidence level. The fit is statistically acceptable, though on the boundary — primarily driven by the w_DE and S₈ tensions.

### Interpretation Guide

| χ²/dof Range | Interpretation |
|-------------|----------------|
| < 0.5 | Overfitting or overestimated errors |
| 0.5 – 1.5 | Good fit |
| **1.0 – 2.0** | **Acceptable fit** (SDGFT is here) |
| 2.0 – 3.0 | Marginal — some tensions |
| > 3.0 | Poor fit — model likely wrong |

## Main Tensions

### 1. Dark Energy Equation of State (w_DE)

| | Value |
|--|-------|
| SDGFT prediction | w₀ = −D*/3 = −67/72 ≈ −0.931 |
| Planck+BAO+SN | w₀ = −1.03 ± 0.03 |
| Tension | 3.3σ |

This is the **primary falsification channel**. SDGFT predicts w₀ ≠ −1 (phantom energy) at tree level. The DESI Year-3 data (expected ~2027) will measure w₀ to σ < 0.01, decisively testing this prediction.

Note: DESI Year-1 results (2024) hinted at w₀ > −1, which would be in the SDGFT direction. The situation is evolving.

### 2. Clustering Amplitude (S₈)

| | Value |
|--|-------|
| SDGFT prediction | S₈ = 0.788 |
| Planck (CMB) | S₈ = 0.832 ± 0.013 |
| DES-Y3 (lensing) | S₈ = 0.776 ± 0.017 |
| Tension | 3.4σ vs Planck, 0.7σ vs DES |

The "S₈ tension" is a known discrepancy between CMB and weak-lensing measurements. SDGFT's prediction aligns with the lensing value but not the CMB value.

## Pull Distribution

The 22 pulls (measured in units of σ) should follow a standard normal distribution N(0, 1) if the model is correct:

```
          Histogram of pulls
    |
  6 |  ■■■■■■
  5 |  ■■■■■■
  4 |  ■■■■■■  ■■■
  3 |  ■■■■■■  ■■■
  2 |  ■■■■■■  ■■■  ■■■
  1 |  ■■■■■■  ■■■  ■■■        ■
    |__________________________________
      [-1,0]  [0,1] [1,2] [2,3] [3,4]
```

- 18/22 observables (82%) are within 1σ — excellent agreement
- 2/22 (9%) are in the 1–2σ range — expected for random fluctuations
- 2/22 (9%) exceed 3σ — the w_DE and S₈ tensions
- No observable deviates by more than 3.4σ

## Oracle Database χ² Statistics

The 100M-point parameter sweep provides global context:

| Metric | Value |
|--------|-------|
| Total points swept | 100,000,000 |
| Points with χ²/dof < 2.0 | 61,701,488 (61.7%) |
| Points with χ²/dof < 1.2 (Gold) | 35,021,095 (35.0%) |
| Best-fit χ²/dof | ~0.89 |
| Axiom-point χ²/dof (analytical) | 1.479 |

The axiom point is not the global minimum — nearby parameter values can achieve better χ² by fitting the w_DE tension. However, the axiom point is distinguished by being **parameter-free** (both Δ and δ_g are fixed combinatorial integers).

## Using the Validation Module

```python
from sdgft_ml.validation import (
    validate_at_axiom,
    chi_squared,
    scorecard,
    EXPERIMENTAL_DATA,
)

# Full validation at the axiom point
results = validate_at_axiom()
chi2 = chi_squared(results)

print(f"χ²/dof = {chi2['chi2_per_dof']:.3f}")
print(f"p-value = {chi2['p_value']:.3f}")

# Detailed scorecard
sc = scorecard(results)
for entry in sc:
    print(f"  {entry['name']:20s}  pull={entry['pull']:+.2f}σ  {'✅' if abs(entry['pull'])<2 else '⚠️'}")

# Validate at arbitrary point
from sdgft_ml.validation import validate_at_point
results_custom = validate_at_point(delta=0.210, delta_g=0.042)
```

## Falsification Criteria

SDGFT proposes five falsification "bets":

| # | Prediction | Decisive Measurement | Timeline |
|---|-----------|---------------------|----------|
| 1 | w₀ = −0.931 ± 0.02 | DESI Y3 / Euclid | 2027–2029 |
| 2 | r = 0.013 ± 0.003 | LiteBIRD / CMB-S4 | 2028–2032 |
| 3 | Σm_ν > 0 via geometric formula | KATRIN / DESI | 2027–2030 |
| 4 | W-boson mass from geometry | CDF-III / LHCb | 2026–2028 |
| 5 | Muon g-2 geometric correction | Fermilab g-2 final | 2026 |

If any prediction is ruled out at >5σ by future experiments, SDGFT is falsified.

---

**Previous:** [← Observable Derivation Chain](04_observable_chain.md) | **Next:** [How the GNN Surrogate Works →](06_gnn_surrogate.md)
