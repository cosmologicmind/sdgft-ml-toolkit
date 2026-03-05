# Galaxy Rotation Curves

SDGFT explains flat rotation curves without dark matter, via a scale-dependent effective Newton coupling $G_\text{eff}(r)$.

---

## Freeman Thin-Disk Model

The baryonic surface density follows a Freeman exponential:

$$\Sigma(r) = \Sigma_0 \, e^{-r/r_d}$$

where $r_d$ is the disk scale-length and $\Sigma_0$ the central surface density.

```python
from sdgft_ml.physics import galaxy

Sigma = galaxy.freeman_disk(r_kpc=5.0, Sigma0=800.0, r_d=3.0)
print(f"Σ(5 kpc) = {Sigma:.1f} M☉/pc²")
```

## Effective Newton Coupling

The enhancement over $G_N$ is controlled by the SDGFT screening function:

$$G_\text{eff}(r) = G_N \left[1 + \epsilon \left(1 - e^{-r/r_\text{screen}}\right)\right]$$

where $\epsilon = \delta/(4\pi) \approx 0.0033$ and $r_\text{screen}$ is the screening radius (typically a few kpc).

| Parameter | Value | Source |
|-----------|-------|--------|
| $\epsilon$ | $\delta/(4\pi)$ ≈ 0.0033 | SDGFT axioms |
| $r_\text{screen}$ | tuned per galaxy | ~ kpc scale |
| Enhancement at $r \gg r_\text{screen}$ | $1 + \epsilon$ | ~0.3% |

### Candidate ε Values

```python
candidates = galaxy.epsilon_candidates()
# [('delta/(4*pi)', 0.00332), ('delta**2/(2*pi)', 0.000276), ('gamma_geo/(2*pi)', 9.9e-5)]
```

## Rotation Curve

The circular velocity combines the Newtonian disk contribution with the SDGFT enhancement:

$$v(r)^2 = v_\text{disk}^2(r) \cdot \frac{G_\text{eff}(r)}{G_N}$$

```python
curve = galaxy.rotation_curve(
    r_array_kpc=[1, 3, 5, 8, 12, 20],
    Sigma0=800.0,
    r_d=3.0,
    epsilon=0.0033,
    r_screen=2.0,
)
for pt in curve:
    print(f"r = {pt['r_kpc']:5.1f} kpc  v = {pt['v_kms']:.1f} km/s")
```

### NGC 3198 Example

NGC 3198 is a classic test galaxy with well-measured HI rotation.

```python
ngc3198 = galaxy.rotation_curve(
    r_array_kpc=[2, 5, 8, 12, 18, 25],
    Sigma0=900.0, r_d=3.2,
    epsilon=0.0033, r_screen=2.5,
)
```

## SPARC Data

The toolkit ships SPARC photometric data (Lelli+ 2016) in `data/sparc/`:

```python
import pandas as pd
sparc = pd.read_csv("data/sparc/sparc_photometric.mrt", comment="#", sep="\\s+")
```

## Screening Function

At small radii the modification is exponentially suppressed:

$$\text{screening}(r) = 1 - e^{-r/r_s}$$

ensuring Solar System gravity is untouched.

```python
s = galaxy.screening(r_kpc=0.01, r_screen=2.0)
# ≈ 0.005 — negligible in the inner kpc
```

## Baryonic Tully-Fisher

SDGFT predicts the slope of the baryonic Tully-Fisher relation:

$$M_b \propto v_\text{flat}^b, \quad b = \frac{91}{24} \approx 3.792$$

| Quantity | SDGFT | Observed | Tension |
|----------|-------|----------|---------|
| BTFR slope $b$ | 3.792 | 3.85 ± 0.09 | ~0.6σ |

```python
tf = galaxy.tully_fisher(v_flat=180.0)
print(f"M_b = {tf['M_baryonic']:.3e} M☉")
print(f"slope b = {tf['b_slope']:.3f}")
```

---

## Notebook

See **[NB 09 — Collider Signatures](../../notebooks/09_collider_signatures.ipynb)** (galaxy section) and **NB 07** for interactive exploration.

---

[← Black Holes](13_black_holes.md) · [Collider & Gauge →](15_collider_gauge.md) · [Wiki Home](README.md)
