# Atomic Physics & QED

SDGFT modifies atomic-scale observables through the geometric anomalous dimension $\gamma_\text{geo}^2 = \delta^4 / D^{*2}$.

---

## Lamb Shift

The hydrogen 2S₁/₂–2P₁/₂ Lamb shift receives a geometric correction:

$$L_\text{geo} = \frac{\Delta}{\Delta + \delta} \cdot \gamma_\text{geo}^2 \cdot R_\infty c$$

where $\Delta/(\Delta+\delta) = 5/6$ is the **projection factor** from the cone angle.

| Quantity | Tree-level | Fixed-point | Observed (CODATA) |
|----------|-----------|-------------|-------------------|
| $L_\text{geo}$ (MHz) | 1060.3 | 1056.1 | 1057.845 ± 0.009 |
| Deviation | +0.23% | −0.16% | — |

### Usage

```python
from sdgft_ml.physics import atomic

L_tree = atomic.lamb_shift_tree()      # ≈ 1060.3 MHz
L_fp   = atomic.lamb_shift_fp()        # ≈ 1056.1 MHz
d_star = atomic.d_star_from_lamb_shift(1057.845)  # invert → D*
```

### Rydberg Correction

The Rydberg constant also shifts:

$$\frac{\delta R_\infty}{R_\infty} = \frac{D^* - 3}{3} \cdot \alpha^2$$

```python
corr = atomic.rydberg_geo_correction()  # O(10⁻⁶)
```

---

## Anomalous Magnetic Moment (g − 2)

SDGFT predicts a **mass-dependent geometric correction** to $(g-2)/2$:

$$\Delta a_\ell = \frac{\alpha}{\pi} \cdot \gamma_\text{geo}^2 \cdot \ln\!\left(\frac{m_\ell}{m_e}\right)$$

| Lepton | $m_\ell/m_e$ | $\Delta a_\ell$ | Experimental $(g-2)/2$ |
|--------|-------------|----------------|----------------------|
| Electron | 1 | 0 (exact) | $0.001\,159\,652\,180\,59$ |
| Muon | 206.768 | $\sim 2 \times 10^{-10}$ | $116\,592\,059 \times 10^{-11}$ |
| Tau | 3477.48 | $\sim 3 \times 10^{-10}$ | not yet measured |

The electron gets **exactly zero** SDGFT correction because $\ln(m_e/m_e) = 0$. The muon correction partially explains the Fermilab g−2 anomaly.

### Usage

```python
from sdgft_ml.physics import qed

p_e  = qed.predict_electron()   # G2Prediction dataclass
p_mu = qed.predict_muon()       # includes sigma_vs_exp
p_tau = qed.predict_tau()

print(f"Muon SDGFT shift: {p_mu.delta_a_geo:.2e}")
print(f"Muon tension: {p_mu.sigma_vs_exp:.1f}σ")
```

The `G2Prediction` dataclass provides `.a_sdgft` (SM + SDGFT total), `.sigma_vs_exp`, and `.fraction_of_anomaly`.

---

## Notebook

See **[NB 06 — Atomic Physics & QED](../../notebooks/06_atomic_qed.ipynb)** for interactive exploration.

---

[← API Reference](10_api_reference.md) · [Neutrino Physics →](12_neutrino_physics.md) · [Wiki Home](README.md)
