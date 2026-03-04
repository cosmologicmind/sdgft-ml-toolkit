# The Three Parameters

SDGFT has **two fundamental axiom parameters** and **one algebraically derived constant**. Together, they form the 3-dimensional input space (Δ, δ_g, φ) that feeds into all 37 observable formulas.

## Parameter 1: Lattice Tension δ_g

$$\boxed{\delta_g = \frac{1}{24} \approx 0.04167}$$

### Definition

The irreducible geometric quantum — one vertex out of the 24-cell's 24 vertices. It represents the minimal resolvable fractional perturbation that a single vertex can impart on the lattice.

### Physical Meaning

δ_g encodes **Planck-scale discreteness**: the resolution limit of the geometric vacuum. Think of it as the geometric "pixel size" — below this scale, the lattice structure cannot be further resolved.

### Where It Appears

| Context | Formula | Meaning |
|---------|---------|---------|
| Effective dimension | D* = 3 − sin²30° + δ_g | Lattice correction to dimension |
| Bosonic string dimension | D_crit = 2 + 1/(2δg) = 26 | String critical dimension |
| Dedekind eta function | η(τ) = q^{1/24} ∏(1−qⁿ) | Modular form factor |
| Baryon asymmetry | ηB = δ_g⁶(1−δ_g)/8 | Baryogenesis from geometry |
| Inflationary e-folds | Nₑ = (D*/Δ)·ln[(D*−2−δ_g)/(Δ·δ_g)] | UV→IR transition duration |
| Ramanujan regularization | 1+2+3+⋯ = −1/12 = −2δ_g | Zeta regularization constant |

### Connection to Gravity

δ_g directly governs the strength of gravitational corrections:
- It is the additive lattice tension in the effective dimension D*
- The running gravitational coupling G(r) = Gₙ[1 + ε(M)·ln(r/r_ref)] has ε ∝ δ_g
- At galactic scales, this running reproduces flat rotation curves without dark matter particles

### Range in the Oracle Database

The Oracle parameter sweep scans δ_g ∈ [0.040, 0.043], centered on the axiom value 1/24 ≈ 0.04167.

---

## Parameter 2: Fibonacci–Lattice Conflict Δ

$$\boxed{\Delta = \frac{F_5}{24} = \frac{5}{24} \approx 0.20833}$$

### Definition

The irreducible mismatch when a Fibonacci-spiral packing is imposed on the crystallographic D₄ lattice. F₅ = 5 is the fifth Fibonacci number, and out of 24 vertex modes, 5 are "frustrated" — they cannot be accommodated without remainder.

### Why F₅ = 5?

The Fibonacci numbers less than 24 are: 1, 1, 2, 3, 5, 8, 13, 21. Among these:
- F₆ = 8 divides 24 cleanly (24/8 = 3), producing a trivial subgroup
- F₅ = 5 produces an **irreducible fraction** 5/24, maximizing frustration
- Higher Fibonacci numbers (13, 21) give near-integer fractions with minimal tension

F₅ = 5 is uniquely selected as the **largest Fibonacci number that generates non-trivial lattice frustration**.

### Physical Meaning

At the topological level, Δ encodes:

| Role | How |
|------|-----|
| **Matter content** | 5 frustrated vertices → 5 visible degrees of freedom |
| **Matter–antimatter asymmetry** | The surplus of frustrated modes → baryon asymmetry |
| **Dimensional reduction** | D* = 3 − Δ = 3 − 5/24 = 67/24 |
| **Inflationary index** | Δ controls the slow-roll parameters |
| **Coupling constants** | Feeds into α_em, α_s, sin²θ_W |

### Where It Appears

| Context | Formula | Value |
|---------|---------|-------|
| Effective dimension | D* = 3 − sin²30° + δ_g = 3 − Δ | 67/24 |
| Baryon density | Ωb = (Δ/4)(1 − δ_g) | 0.0499 |
| CDM density | Ωc = 6Δ² | 0.260 |
| Spectral index | ns = 1 − 2(2n−1)/[Nₑ(2n−1)+n] | 0.967 |
| Higgs quartic | λ = Δ/φ | 0.1287 |
| Reactor angle | θ₁₃ = arcsin(Δ/√2) | 8.47° |
| CKM |Vub| | \|Vub\| = Δ^φ · δ_g · exp(…) | 0.00382 |
| Quark mixing | \|Vus\| = √Ωb | 0.2234 |

### Range in the Oracle Database

The Oracle sweep scans Δ ∈ [0.200, 0.220], centered on the axiom value 5/24 ≈ 0.20833.

---

## Derived Constant: The Golden Ratio φ

$$\boxed{\phi = \frac{1 + \sqrt{5}}{2} \approx 1.61803}$$

### How It Arises

φ is **not an independent input**. It is an algebraic consequence of the ratio:

$$\frac{\Delta}{\delta_g} = \frac{5/24}{1/24} = 5$$

Since 5 is a Fibonacci number, φ enters as the dominant eigenvalue of the Fibonacci recurrence:

$$\phi = \frac{1 + \sqrt{\Delta/\delta_g}}{2} = \frac{1 + \sqrt{5}}{2}$$

### Physical Meaning

φ represents the **dominant eigenvalue of the geometric transfer matrix** — the Fibonacci resonance amplification factor. Just as φ governs:
- The spacing of sunflower seeds (Fibonacci spirals)
- The ratio of successive Fibonacci numbers
- The irrational rotation number of the golden torus

In SDGFT, it governs:
- The stability condition for fermion generations: max{n : φⁿ < Δ/δ_g} = 3
- The Higgs quartic coupling: λ = Δ/φ
- The icosahedral symmetry underlying the lattice frustration
- The fixed-point iteration for D*

### Where It Appears

| Context | Formula | Value |
|---------|---------|-------|
| Fermion generations | Ngen = max{n : φⁿ < Δ/δ_g} | 3 |
| Higgs quartic coupling | λ = Δ/φ | 0.1287 |
| Higgs mass | mH = √(2λ) · v_H = √(2Δ/φ) · 246.22 | 125.3 GeV |
| Fixed-point dimension | D* = Δ^{−1/D*} · φ · Δ^{Δ·δ_g} | 2.797 |
| CKM |Vub| | \|Vub\| = Δ^φ · δ_g · exp(…) | 0.00382 |

### Why φ Is Treated as a Third ML Input

Though φ is algebraically fixed by Δ/δ_g = 5, the ML models treat it as a third input dimension (Δ, δ_g, φ) because:

1. When scanning the parameter landscape, φ can be varied independently to test sensitivity
2. The GNN learns the functional relationships rather than assuming them
3. It allows the CVAE inverter to recover all three values from observables
4. Off-axiom exploration may reveal φ-dependent structure

In the Oracle Database, φ is fixed at (1+√5)/2 for all 100M points.

---

## The Axiom Closure Condition

$$\Delta + \delta_g = \sin^2(30°) = \frac{1}{4}$$

This is the **master identity** of SDGFT. Geometrically, it means the cone half-angle is exactly 30° — the largest non-trivial common divisor of 90° (orthogonality) and 360° (full rotation):

$$\frac{90°}{30°} = 3 \quad \text{(spatial dimensions)}, \qquad \frac{360°}{30°} = 12 = \frac{24}{2} \quad \text{(half the kissing number)}$$

### Consequences of the Closure Condition

1. **Fermion spin**: S = sin(30°) = 1/2 — fermions are "tilted" by 30° to fit through the lattice
2. **Boson spin**: S = sin(90°) = 1 — bosons propagate along symmetry axes
3. **Cosmic flatness**: Ωtot = Ωb + Ωc + ΩDE = 2304/2304 = 1 (exact, by combinatorics)
4. **Dark energy UV boundary**: cos²(30°) = 3/4 sets the starting condition for ΩDE

---

## Parameter Space Summary

| Parameter | Symbol | Axiom Value | Oracle Range | Type |
|-----------|--------|-------------|-------------|------|
| Fibonacci conflict | Δ | 5/24 ≈ 0.20833 | [0.200, 0.220] | Fundamental |
| Lattice tension | δ_g | 1/24 ≈ 0.04167 | [0.040, 0.043] | Fundamental |
| Golden ratio | φ | (1+√5)/2 ≈ 1.61803 | Fixed | Derived |

The **axiom point** (Δ = 5/24, δ_g = 1/24, φ = golden ratio) is where all predictions are parameter-free. The ML toolkit explores the neighborhood of this point to study sensitivity, degeneracies, and the structure of the fitness landscape.

---

**Previous:** [← What is SDGFT?](01_what_is_sdgft.md) | **Next:** [Dimensional Flow →](03_dimensional_flow.md)
