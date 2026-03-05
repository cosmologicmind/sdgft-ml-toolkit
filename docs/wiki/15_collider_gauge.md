# Collider Signatures & Gauge Unification

SDGFT produces testable deviations at colliders and unifies gauge couplings through the 24-cell / D₄ root system.

---

## Part A — Collider Signatures

### Modified Running Couplings

At scale $\sqrt{s}$ the strong coupling runs as:

$$\alpha_s(\sqrt{s}) = \alpha_s(M_Z)\left[1 + \frac{7\alpha_s(M_Z)}{2\pi}\ln\frac{\sqrt{s}}{M_Z} + \gamma_\text{geo}\right]$$

with $\gamma_\text{geo} = 1/1608 \approx 6.2 \times 10^{-4}$.

```python
from sdgft_ml.physics import collider

run = collider.sdgft_modified_running(sqrt_s_gev=1000.0)
print(f"α_s(1 TeV) = {run['alpha_s']:.6f}")
print(f"Δα_s/α_s  = {run['delta_alpha_over_alpha']:.2e}")
```

### Drell-Yan Cross-Section Ratio

$$R_{DY} = \frac{\sigma_\text{SDGFT}}{\sigma_\text{SM}} = 1 + \gamma_\text{geo}\,\ln\frac{s}{M_Z^2}$$

```python
dy = collider.drell_yan_ratio(sqrt_s_gev=3000.0)
print(f"R_DY(3 TeV) = {dy['R_DY']:.6f}")
```

### Graviton Exchange

Virtual graviton exchange in extra (fractal) dimensions produces contact-interaction signals:

$$\sigma_\text{grav} \propto \frac{s^2}{M_D^8} \cdot \gamma_\text{geo}$$

where $M_D \approx D^* \cdot M_\text{Pl} / (8\pi)$ is the effective scale.

```python
grav = collider.graviton_exchange(sqrt_s_gev=3000.0)
print(f"σ_graviton / σ_SM = {grav['sigma_ratio']:.2e}")
print(f"M_D = {grav['M_D_gev']:.3e} GeV")
```

### Kaluza-Klein Spectrum

KK excitation masses from the SDGFT spectral dimension:

$$m_n = n \cdot \frac{2\pi}{L_\text{extra}}, \quad L = \frac{D^* - 4}{M_\text{Pl}}$$

```python
kk = collider.kk_spectrum(n_max=5)
for mode in kk['modes']:
    print(f"n={mode['n']}  m={mode['m_gev']:.3e} GeV")
```

### Higgs Coupling Modification

$$\kappa_H = 1 + \frac{\gamma_\text{geo}}{2} \approx 1.000311$$

```python
h = collider.higgs_modification()
print(f"κ_H = {h['kappa_H']:.6f}")  # ~1.000311
```

### BSM Reach Summary

```python
reach = collider.bsm_reach_summary()
for ch in reach:
    print(f"{ch['channel']:20s}  deviation={ch['deviation']:.2e}  "
          f"discoverable={ch['discoverable_at']}")
```

| Channel | Deviation | Discoverable at |
|---------|----------|----------------|
| Drell-Yan ratio | ~10⁻³ | FCC-hh |
| Graviton exchange | ~10⁻³⁰ | Not accessible |
| KK modes | ~10¹⁸ GeV | Not accessible |
| Higgs κ | ~3×10⁻⁴ | FCC-ee |
| Dijet angular | ~6×10⁻⁴ | FCC-hh |

---

## Part B — Gauge Unification

### The 24-Cell and D₄

SDGFT derives gauge structure from the **24-cell** (the unique self-dual regular 4-polytope) with 24 vertices fitting the **D₄** root system.

```python
from sdgft_ml.physics import gauge_groups as gg

verts = gg.twentyfour_cell_vertices()
print(f"{len(verts)} vertices")        # 24

roots = gg.d4_roots()
print(f"{len(roots)} roots")           # 24
print(f"Length of each root: {roots[0] @ roots[0]:.0f}")  # 2
```

### Cartan Matrix

$$A_{ij} = \frac{2 \langle \alpha_i, \alpha_j \rangle}{\langle \alpha_j, \alpha_j \rangle}$$

```python
C = gg.cartan_matrix_d4()
# [[ 2, -1,  0,  0],
#  [-1,  2, -1, -1],
#  [ 0, -1,  2,  0],
#  [ 0, -1,  0,  2]]
```

### SM Decomposition

$$D_4 \;\longrightarrow\; SU(3) \times SU(2) \times U(1)$$

Rank 4 → 2 + 1 + 1 = 4 ✓

```python
dec = gg.sm_decomposition()
for g in dec['groups']:
    print(f"{g['name']}: rank {g['rank']}, dim {g['dim']}")
# SU(3): rank 2, dim 8
# SU(2): rank 1, dim 3
# U(1) : rank 1, dim 1
```

### Triality

D₄ has a unique **triality** automorphism (order 3) permuting the vector and two spinor representations. SDGFT links this to the three SM generations:

```python
tri = gg.triality_map()
print(tri['order'])  # 3
print(tri['maps'])   # vector ↔ spinor+ ↔ spinor-
```

### Weinberg Angle Boundary Condition

$$\sin^2 \theta_W \big|_{M_\text{GUT}} = \frac{3}{8} = 0.375$$

```python
wa = gg.weinberg_angle_boundary()
print(f"sin²θ_W(GUT) = {wa['sin2_theta_W_gut']}")  # 3/8
```

### RG Flow & Unification

```python
from sdgft_ml.physics import rg_running as rg

trajectory = rg.coupling_trajectory(mu_min=91.0, mu_max=1e16, n_points=200)
print(f"{len(trajectory)} points from {trajectory[0]['mu_gev']:.0f} to {trajectory[-1]['mu_gev']:.2e} GeV")

uni = rg.unification_scale()
print(f"M_GUT ≈ {uni['mu_unification_gev']:.2e} GeV")
print(f"α_GUT ≈ {uni['alpha_unified']:.4f}")
```

---

## Notebooks

- **[NB 09 — Collider Signatures](../../notebooks/09_collider_signatures.ipynb)**
- **[NB 10 — Gauge Unification & D₄](../../notebooks/10_gauge_unification.ipynb)**

---

[← Galaxy Rotation](14_galaxy_rotation.md) · [Wiki Home](README.md)
