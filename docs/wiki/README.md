# SDGFT-ML Wiki

**Comprehensive reference for the Six-Dimensional Geometric Field Theory ML Toolkit**

> **Oracle Database DOI:** [10.5281/zenodo.18863347](https://doi.org/10.5281/zenodo.18863347) — 61.7M parameter points, 5.1 GB Parquet

## Contents

### Theory

1. [**What is SDGFT?**](01_what_is_sdgft.md) — Overview of the theory, its axioms, and what it predicts
2. [**The Three Parameters**](02_parameters.md) — Physical meaning of Δ, δ_g, and φ
3. [**Dimensional Flow**](03_dimensional_flow.md) — How D* runs with scale, the beta-function, and spectral dimension
4. [**Observable Derivation Chain**](04_observable_chain.md) — All 37 formulas, level by level, from geometry to measurement
5. [**Experimental Validation**](05_experimental_validation.md) — 22 precision tests, χ² methodology, tensions, and predictions

### Machine Learning

6. [**How the GNN Surrogate Works**](06_gnn_surrogate.md) — Architecture, training, DAG structure, and ensemble logic
7. [**How the CVAE Inverter Works**](07_cvae_inverter.md) — Inverse-problem solver: observables → parameters
8. [**The Oracle Database**](08_oracle_database.md) — 100M-point parameter sweep, quality filtering, and query patterns

### Physics Modules

11. [**Atomic Physics & QED**](11_atomic_qed.md) — Lamb shift, anomalous magnetic moment, Rydberg correction
12. [**Neutrino Physics**](12_neutrino_physics.md) — Mass sum rule, PMNS mixing, oscillation predictions, DUNE/T2K/JUNO
13. [**Black Holes & Compact Stars**](13_black_holes.md) — Running G, Hawking temperature, QNM corrections, TOV integration
14. [**Galaxy Rotation Curves**](14_galaxy_rotation.md) — Freeman disk, G_eff enhancement, SPARC data, Tully-Fisher
15. [**Collider & Gauge Unification**](15_collider_gauge.md) — Drell-Yan, graviton exchange, KK spectrum, D₄ → SM decomposition

### Using the Toolkit

9. [**Installation & Setup**](09_installation.md) — Environment, dependencies, data files, first run
10. [**API Reference**](10_api_reference.md) — SDGFTPredictor, OracleDB, ParametricForward, validation functions

---

*All pages are self-contained and cross-linked. Start with [What is SDGFT?](01_what_is_sdgft.md) for a conceptual tour, or jump to [How the GNN Surrogate Works](06_gnn_surrogate.md) for the ML specifics, or explore the [Physics Modules](11_atomic_qed.md) for SDGFT predictions.*
