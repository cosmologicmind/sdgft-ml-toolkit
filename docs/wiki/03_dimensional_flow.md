# Dimensional Flow

One of SDGFT's most distinctive features is that **spacetime dimension is not an integer** — it runs continuously with scale, interpolating between D* = 2 at the Planck scale and D* ≈ 2.797 at cosmological distances.

## The Effective Dimension D*

### Tree-Level Derivation

$$D^* = 3 - \underbrace{\sin^2(30°)}_{\text{spin projection}} + \underbrace{\delta_g}_{\text{lattice tension}} = 3 - \frac{1}{4} + \frac{1}{24} = \frac{67}{24} \approx 2.7917$$

Or equivalently, using the axiom constraint Δ + δ_g = 1/4:

$$D^* = 3 - \Delta = 3 - \frac{5}{24} = \frac{67}{24}$$

The dimension is a **rational number**: exactly 67/24.

### Self-Consistent Fixed-Point Equation

The tree-level value can also be obtained as the attracting fixed point of:

$$D^* = \Delta^{-1/D^*} \cdot \phi \cdot \Delta^{\Delta \cdot \delta_g}$$

Starting from any positive seed D₀ > 0, iterating this equation converges to D*_fp ≈ 2.797, agreeing with 67/24 to 0.18%. This self-referential structure — the dimension determines its own value — is characteristic of quantum-gravitational systems.

### The f(R) Exponent

The effective dimension directly determines the gravitational action:

$$n_{f(R)} = \frac{D^*}{2} = \frac{67}{48} \approx 1.396$$

The unique dimensionless action in D* dimensions is:

$$S = \int d^4x \sqrt{-g} \; R^{D^*/2} = \int d^4x \sqrt{-g} \; R^{67/48}$$

This belongs to the well-studied class of f(R) theories and crucially predicts **αT = 0** (gravitational waves travel at the speed of light), consistent with the LIGO/Virgo constraint from GW170817.

## The Dimensional Beta-Function

D* is not constant — it **runs with scale** according to a renormalization-group-like beta-function:

$$\beta(D^*) = \frac{dD^*}{dN} = \frac{\Delta}{D^*} \left( D^*_{\text{IR}} - D^* \right)$$

where N is the number of e-folds (logarithmic scale evolution). This has the structure of a **Newton cooling law**: D* relaxes exponentially toward its IR fixed point.

### Analytic Solution

$$D^*(N) = D^*_{\text{IR}} - \left( D^*_{\text{IR}} - D^*_{\text{UV}} \right) \cdot \exp\left( -\frac{\Delta}{D^*_{\text{IR}}} \, N \right)$$

With boundary conditions:
- UV (Planck scale): D*_UV = 2 — the dimension of the quantum foam
- IR (Hubble horizon): D*_IR ≈ 67/24 ≈ 2.797 — the cosmological fixed point

### Dimensional Running Table

| Scale | r | D*(r) | Physical Regime | Consequences |
|-------|---|-------|-----------------|-------------|
| Planck | 1.6 × 10⁻³⁵ m | 2.0 | Quantum foam | Gravity is power-counting renormalizable |
| GUT | 10⁻³² m | ~2.1 | Grand unification | Coupling unification candidates |
| Electroweak | 10⁻¹⁸ m | ~2.95 | Standard Model | SM coupling corrections kick in |
| Nuclear | 10⁻¹⁵ m | ~2.75 | QCD confinement | Strong coupling transition |
| Galactic | 10²⁰ m | ~2.84 | Galaxy dynamics | Dark matter mimicry: running G(r) |
| Hubble | 10²⁶ m | 2.797 | Cosmological | IR attractor; dark energy domination |

## Inflation as Dimensional Relaxation

In standard cosmology, inflation is driven by a hypothetical scalar field (the inflaton). In SDGFT, **inflation is dimensional relaxation** — no new fields required.

### The Number of E-Folds

$$N_e = \frac{D^*}{\Delta} \cdot \ln\left[ \frac{D^* - 2 - \delta_g}{\Delta \cdot \delta_g} \right]$$

At the axiom point: **Nₑ ≈ 59.95 e-folds**, consistent with the CMB requirement of 55–65.

### Slow-Roll Parameters

The dimensional flow maps directly to slow-roll parameters:

$$\epsilon_{\text{SR}} = \frac{(2n-1)(n-2)^2}{[N_e(2n-1) + n]^2}$$

$$\eta_{\text{SR}} = \frac{(2n-1)(2n^2-7n+4)}{[N_e(2n-1)+n]^2} - \frac{(2n-1)}{[N_e(2n-1)+n]}$$

where n = D*/2.

### The Spectral Index and Tensor Ratio

Two of SDGFT's sharpest predictions:

$$n_s = 1 - \frac{2(2n-1)}{N_e(2n-1) + n} \approx 0.967$$

$$r = \frac{48(2n-1)^2}{[N_e(2n-1) + n]^2} \approx 0.013$$

| Observable | SDGFT | Experimental | Status |
|------------|-------|-------------|--------|
| n_s | 0.967 | 0.9649 ± 0.0042 (Planck) | ✅ 0.5σ |
| r | 0.013 | < 0.036 (BICEP/Keck) | ✅ Well below bound |

The tensor-to-scalar ratio r ≈ 0.013 is a precise prediction testable by CMB-S4 and LiteBIRD within this decade.

## UV Fixed Point: Gravity Renormalizability

At the Planck scale, D*_UV = 2. In two dimensions:

$$[G_N] = \text{mass}^{D-2} = \text{mass}^0 = \text{dimensionless}$$

Newton's constant becomes dimensionless, making gravity **power-counting renormalizable**. This resolves the hierarchy problem without supersymmetry — gravity is non-renormalizable only because we observe it at D* ≈ 2.8, not because it is fundamentally sick.

This UV limit matches:
- **Causal Dynamical Triangulations (CDT)**: measured D_s = 1.80 ± 0.25 at short distances
- **Asymptotic Safety**: the Reuter fixed point requires D_s ≈ 2
- **Hořava–Lifshitz gravity**: engineering dimension → 2 at UV

## The Running Gravitational Coupling

The scale-dependent dimension induces a scale-dependent gravitational constant:

$$G(r) = G_N \left[ 1 + \varepsilon(M) \ln\left(\frac{r}{r_{\text{ref}}}\right) \right]$$

where ε depends on the mass within radius r. This logarithmic running:

1. **Reproduces flat rotation curves** at galactic scales without dark matter particles
2. Predicts a characteristic **transition radius** r_trans ~ 1 kpc between Newtonian and modified regimes
3. Is compatible with the Tully–Fisher relation with slope b_TF = D* + 1 ≈ 3.79 (observed: 3.85 ± 0.09)

## Connection to the ML Toolkit

The dimensional flow equations are computed analytically by `ParametricForward`:

```python
from sdgft_ml.data import ParametricForward

fwd = ParametricForward(delta=5/24, delta_g=1/24)

# Tree-level dimension
print(f"D*_tree = {fwd.d_star_tree:.4f}")   # 2.7917

# Fixed-point dimension
print(f"D*_fp = {fwd.d_star_fp:.4f}")        # 2.7970

# E-folds
print(f"N_e = {fwd.e_folds():.1f}")          # 59.9

# Spectral index
print(f"n_s = {fwd.spectral_index():.4f}")   # 0.9671
```

The GNN surrogate reproduces these with R² > 0.9995 in <1 ms per evaluation, enabling the 100M-point Oracle sweep that took ~6 hours (vs. ~40 hours analytically).

---

**Previous:** [← The Three Parameters](02_parameters.md) | **Next:** [Observable Derivation Chain →](04_observable_chain.md)
