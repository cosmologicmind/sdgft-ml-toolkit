# What is SDGFT?

## One-Sentence Summary

**Six-Dimensional Geometric Field Theory (SDGFT)** is a zero-free-parameter framework that derives all fundamental constants — from the Higgs mass to dark energy — from two topological integers on a 4D polytope.

## The Central Idea

SDGFT starts from a single mathematical object: the **24-cell** {3, 4, 3}, the unique self-dual regular polytope in four Euclidean dimensions. All physics follows from the interplay between:

- The **crystallographic order** of the 24-cell lattice (the D₄ root lattice)
- The **irrational growth law** of the Fibonacci sequence (golden ratio φ)

This tension — rational lattice versus irrational spiral — generates a *dimensional mismatch* that, when propagated through a strict derivation chain, yields ~37 measurable predictions spanning particle physics, cosmology, inflation, and gravity.

> *The universe is not particles in spacetime — the universe **is** spacetime with non-trivial topology.*

## The 24-Cell Polytope

| Property | Value |
|----------|-------|
| Vertices | 24 |
| Edges | 96 |
| Faces | 96 (triangular) |
| Cells | 24 (octahedral) |
| Symmetry group | F₄, order 1152 |
| Root lattice | D₄ (kissing number κ₄ = 24) |

The 24-cell is singled out by three exceptional properties:

1. **Unique self-duality** — isomorphic to its own dual (vertex ↔ cell correspondence). This is the origin of wave-particle duality in SDGFT.
2. **Densest 4D lattice packing** — the D₄ lattice achieves the highest known sphere-packing density in 4 dimensions.
3. **Triality** — the D₄ Dynkin diagram has an order-3 outer automorphism, unique among simple Lie algebras. This maps to exactly three fermion generations.

The 24 vertices decompose into:
- **Class I** (8 vertices): permutations of (±1, 0, 0, 0) — a 16-cell
- **Class II** (16 vertices): all (±½, ±½, ±½, ±½) — a tesseract

These can be identified with the 24 unit quaternions forming the **binary tetrahedral group**.

## The Six-Cone Geometry

The "six-dimensional" label does **not** mean 6 spatial dimensions in the Kaluza–Klein sense. It refers to **six independent geometric field channels**: six cones aligned with ±x̂, ±ŷ, ±ẑ (three Cartesian half-axes and their negatives), each with half-opening angle θ_max = 30°.

The connection: **24 = 6 × 4** — four vertices per cone. When the 24-cell is projected from ℝ⁴ to ℝ³, it maps onto this 6-cone system.

```
         +z
          | /  (cone: 4 vertices within 30° of +z)
          |/
  --------+--------→ +x
         /|
        / |
       /  
      +y
```

Each cone covers a solid angle of ½(1 − cos 30°) ≈ 0.067 of 4π sr. All six together subtend ≈ 40.2% of the full sphere — the "geometric coverage" that determines the dark energy fraction.

## The Two Axioms

### Axiom I — Lattice Tension

$$\delta_g = \frac{1}{N_{\text{vert}}} = \frac{1}{24} \approx 0.04167$$

One vertex out of 24: the minimal resolvable fractional perturbation by a single vertex on the lattice. This is the **Planck-scale quantum of geometry**.

### Axiom II — Fibonacci–Lattice Conflict

$$\Delta = \frac{F_5}{24} = \frac{5}{24} \approx 0.20833$$

Five frustrated vertices out of 24: the irreducible mismatch when a Fibonacci-spiral packing is imposed on the D₄ lattice. F₅ = 5 is the largest Fibonacci number that produces a non-trivial fraction of 24.

### The Closure Condition

$$\Delta + \delta_g = \frac{5}{24} + \frac{1}{24} = \frac{6}{24} = \frac{1}{4} = \sin^2(30°)$$

This is the **geometric closure condition**: the sum of the two axiom parameters equals the squared sine of the cone half-angle. It simultaneously:

- Defines the half-opening angle θ_max = 30°
- Generates spin from geometry: fermion spin S = sin(30°) = ½, boson spin S = sin(90°) = 1
- Sets the complementary fraction cos²(30°) = ¾ as the UV boundary condition for dark energy
- Makes cosmic flatness a combinatorial identity (Ωtot = 1 exactly)

## What Does SDGFT Predict?

From just these two integers (1 and 5) and the number 24, SDGFT derives:

| Domain | # Predictions | Examples |
|--------|---------------|----------|
| Cosmology | 7 | Ωb, Ωc, ΩDE, Ωm, wDE, ηB, S₈ |
| Inflation | 6 | ns, r, Ne, εSR, ηSR, βiso |
| Particle physics | 10 | α⁻¹em, αs, sin²θW, mH, mμ/me, mτ/mμ, Ngen, λgeo |
| Neutrinos | 3 | θ₁₂, θ₂₃, θ₁₃ |
| CKM matrix | 3 | |Vus|, |Vub|, quark hierarchy |
| Gravity | 4 | αM, αB, ηslip(survey), ηslip(horizon) |
| Dimensional flow | 4 | D*, D*fp, n, nfp |
| **Total** | **37** | |

The global fit against 22 experimentally measured observables yields **χ²/dof = 1.479** (p = 0.069) — a statistically acceptable fit using zero free parameters.

## How is This Different from Standard Model + ΛCDM?

| Aspect | SM + ΛCDM | SDGFT |
|--------|-----------|-------|
| Free parameters | 19 (SM) + 6 (ΛCDM) = **25** | **0** (two combinatorial axioms) |
| Dark matter | BSM particle (undetected) | Running G(r) from dimensional flow |
| Dark energy | Cosmological constant (fine-tuned to 10⁻¹²²) | Geometric: ΩDE = 1 − Ωb − Ωc = 0.690 |
| Inflation | Inflaton field (unidentified) | Dimensional relaxation D* : 2 → 2.797 |
| Higgs mass | Free parameter (125.25 GeV) | mH = √(2Δ/φ) · vH = 125.3 GeV |
| Three generations | Unexplained | Fibonacci stability: max{n : φⁿ < Δ/δg} = 3 |
| Hierarchy problem | Unresolved | Natural: D*UV = 2 makes gravity renormalizable |

## Key Tension

The main tension with data is the **dark energy equation of state**: SDGFT predicts w₀ = −D*/3 = −67/72 ≈ −0.931, while Planck+BAO+SN measures w₀ = −1.03 ± 0.03 (~3σ tension). This is the most important testable prediction — upcoming DESI and Euclid surveys will resolve it.

---

**Next:** [The Three Parameters →](02_parameters.md)
