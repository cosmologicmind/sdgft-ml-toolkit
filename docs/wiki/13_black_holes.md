# Black Holes & Compact Stars

SDGFT replaces the classical singularity with a finite-curvature core controlled by the running Newton coupling $G(k)$.

---

## Running Newton Coupling

$$G(k) = \frac{G_N}{1 + (k/k_P)^2}$$

where $k_P = \sqrt{1/G_N}$ is the Planck momentum. At trans-Planckian momenta $G(k) \to 0$, softening the UV.

```python
from sdgft_ml.physics import black_holes as bh

# Running G at half-Planck scale
G_half = bh.running_G(0.5)   # k/k_P = 0.5
print(f"G(k_P/2) / G_N = {G_half:.4f}")  # ≈ 0.80
```

## Modified Schwarzschild Radius

$$r_s(M) = \frac{2\,G(M)\,M}{c^2}$$

For stellar-mass black holes the correction is negligible; near the Planck scale it prevents collapse to a point.

## Hawking Temperature

$$T_H = \frac{\hbar\,c^3}{8\pi\,G\,M\,k_B} \cdot \left(1 - e^{-M/M_P}\right)$$

The exponential cutoff ensures $T_H \to 0$ as $M \to 0$ — a finite maximum temperature exists:

```python
result = bh.hawking_temperature(1e-8)  # solar masses
print(f"T = {result['T_K']:.3e} K")
print(f"T_max = {result['T_max_K']:.3e} K")
```

| Quantity | Formula | Significance |
|----------|---------|-------------|
| $T_\text{max}$ | $\sim M_P c^2 / k_B$ | No thermal divergence |
| Kretschner bound | $K \leq k_P^4$ | Finite curvature everywhere |

## Quasi-Normal Mode Corrections

$$\omega_\text{QNM} = \omega_\text{GR} \left(1 + \gamma_\text{geo}\right)$$

with $\gamma_\text{geo} = \delta^2/D^* = 1/1608 \approx 6.2 \times 10^{-4}$.

```python
qnm = bh.qnm_correction(omega_gr=0.37, n=0)
print(f"ω_SDGFT = {qnm['omega_sdgft']:.6f}")   # 0.37 × (1 + 1/1608)
print(f"Δf/f = {qnm['delta_f_over_f']:.2e}")    # ≈ 6.2e-4
```

LISA/Einstein Telescope may reach 10⁻⁴ fractional precision — sufficient to test this.

## Bekenstein-Hawking Entropy

$$S = \frac{A}{4\,G(k)\,\ell_P^2} = \frac{4\pi\,r_s^2}{4\,G(k)\,\ell_P^2}$$

Logarithmic corrections arise naturally from the running $G(k)$.

## TOV Integration

For compact-star structure, the module integrates the Tolman-Oppenheimer-Volkoff equation with a polytropic EOS ($P = K\rho^\Gamma$), using the running $G(r)$:

```python
tov = bh.tov_solve(rho_c=5e14, K=5e-3, Gamma=2.0, dr=100.0)
print(f"R = {tov['R_km']:.1f} km")
print(f"M = {tov['M_msun']:.4f} M☉")
```

---

## Notebook

See **[NB 08 — Black Holes & Compact Objects](../../notebooks/08_black_holes.ipynb)** for interactive exploration.

---

[← Neutrino Physics](12_neutrino_physics.md) · [Galaxy Rotation →](14_galaxy_rotation.md) · [Wiki Home](README.md)
