# The Oracle Database

## What Is It?

The Oracle Database is a **100-million-point pre-computed exploration** of the SDGFT parameter space. For every point on a dense (Δ, δ_g) grid, the GNN surrogate predicted all 37 observables, and each prediction was scored against 21 experimental measurements via χ².

The result: a complete map of "where SDGFT agrees with experiment" — stored as two Parquet files totaling 5.1 GB.

> **Download:** The data files are archived on Zenodo with DOI [10.5281/zenodo.18863347](https://doi.org/10.5281/zenodo.18863347).
> ```bash
> wget -P data/ https://zenodo.org/records/18863347/files/oracle_db.parquet
> wget -P data/ https://zenodo.org/records/18863347/files/oracle_gold.parquet
> ```

## Generation Pipeline

```
Step 1: Define grid
    Δ ∈ [0.200, 0.220]  (10,000 steps)
    δ_g ∈ [0.040, 0.043] (10,000 steps)
    → 100,000,000 parameter pairs

Step 2: GNN prediction
    For each (Δ, δ_g): predict 37 observables via ensemble
    → ~6 hours on GPU (vs. ~40 hours analytical)

Step 3: χ² scoring
    Compare 21 observables against experiment
    (η_B excluded due to unit mismatch)
    → total_chi2, chi2_per_dof, n_tensions

Step 4: Quality filtering
    Discard points with χ²/dof ≥ 2.0
    → 61.7M points kept (61.7%)

Step 5: Gold Standard extraction
    Points with χ²/dof < 1.2
    → 35.0M points (35.0%)

Step 6: Export to Parquet
    oracle_db.parquet:   61.7M rows, 44 columns (3.2 GB)
    oracle_gold.parquet: 35.0M rows, 44 columns (1.9 GB)
```

## File Details

| File | Rows | Size | Filter |
|------|------|------|--------|
| `data/oracle_db.parquet` | 61,701,488 | 3.2 GB | χ²/dof < 2.0 |
| `data/oracle_gold.parquet` | 35,021,095 | 1.9 GB | χ²/dof < 1.2 |

Both files share the same 44-column schema.

## Column Schema (44 Columns)

### Parameters (2)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `delta` | float32 | [0.200, 0.220] | Fibonacci conflict Δ |
| `delta_g` | float32 | [0.040, 0.043] | Lattice tension δ_g |

### Observables (37)

Grouped by physics domain (see [Observable Derivation Chain](04_observable_chain.md) for formulas):

**Dimension (4):** `d_star_tree`, `d_star_fp`, `n_tree`, `n_fp`

**Gravity (4):** `alpha_m_tree`, `alpha_b_tree`, `eta_slip_survey`, `eta_slip_horizon`

**Inflation (6):** `n_efolds_fp`, `n_s`, `r_tensor`, `beta_iso`, `epsilon_sr`, `eta_sr`

**Cosmology (7):** `omega_b`, `omega_c`, `omega_de`, `omega_m`, `w_de_fp`, `eta_b`, `s_8`

**Particle physics (10):** `alpha_em_inv_tree`, `alpha_em_inv_fp`, `alpha_em_tree`, `alpha_s`, `sin2_theta_w`, `mu_e_ratio`, `tau_mu_ratio_tree`, `lambda_geo`, `higgs_mass`, `n_generations`

**Neutrinos (3):** `theta_12`, `theta_23`, `theta_13`

**CKM (3):** `v_us`, `v_ub`, `quark_hierarchy`

### Metadata (5)

| Column | Type | Description |
|--------|------|-------------|
| `total_chi2` | float32 | Σ(pred − exp)² / σ² over 21 observables |
| `chi2_per_dof` | float32 | total_chi2 / 21 |
| `n_tensions` | int8 | Count of observables with \|pull\| > 2σ |
| `gold_standard` | bool | True if χ²/dof < 1.2 |
| `desi_w_match` | bool | True if w_DE matches DESI range (−0.997 ± 0.025) |

## χ² Computation

For each of the 21 scored observables:

$$\chi^2_i = \frac{(O_i^{\text{GNN}} - O_i^{\text{exp}})^2}{\sigma_{\text{eff},i}^2}$$

where:

$$\sigma_{\text{eff},i} = \max\left(\sigma_{\text{exp},i}, \; 0.01 \times |O_i^{\text{GNN}}|\right)$$

The 1% surrogate uncertainty floor accounts for GNN prediction error. For observables with theory uncertainties (α_em⁻¹, mμ/me, mτ/mμ), the larger σ is used.

**Excluded observable**: η_B (baryon asymmetry) — the surrogate predicts ~10⁻⁷ vs. the true ~10⁻¹⁰ due to a unit/scale mismatch in the training pipeline.

## Grid Structure

```
δ_g ↑
0.043 ┌───────────────────────────────┐
      │                               │
      │         100M grid points      │
      │         (10,000 × 10,000)     │
      │                               │
      │           ★ Axiom point       │
      │         (5/24, 1/24)          │
      │                               │
0.040 └───────────────────────────────┘
      0.200                     0.220 → Δ
```

Grid spacing:
- ΔΔ = (0.220 − 0.200) / 10,000 = 2 × 10⁻⁶
- Δδ_g = (0.043 − 0.040) / 10,000 = 3 × 10⁻⁷

This is fine enough to resolve any local structure in the χ² landscape.

## Query Patterns

### Using the OracleDB API

```python
from sdgft_ml.inference import OracleDB

db = OracleDB()  # lazy loads parquet on first query

# Best-fit points
top10 = db.best_fit(n=10)

# Filter by observable range
higgs_match = db.filter_observable("higgs_mass", 125.0, 125.5)

# Compound query (pandas expression)
result = db.query("higgs_mass > 125 and n_s < 0.97 and gold_standard == True")

# Gold Standard subset
gold = db.gold_standard()

# Parameter range statistics
print(db.parameter_range())

# Summary
print(db.summary())
```

### Using Pandas Directly

```python
import pandas as pd

# Load full database
db = pd.read_parquet("data/oracle_db.parquet")

# Column-selective loading (saves memory)
cols = ["delta", "delta_g", "higgs_mass", "n_s", "chi2_per_dof"]
db = pd.read_parquet("data/oracle_db.parquet", columns=cols)

# Standard pandas operations
best = db.nsmallest(100, "chi2_per_dof")
correlation = db[["higgs_mass", "n_s", "omega_de"]].corr()
```

### Using DuckDB (SQL, Zero-Copy)

```python
import duckdb

conn = duckdb.connect()

# SQL queries directly on Parquet files — no loading into memory
result = conn.sql("""
    SELECT delta, delta_g, higgs_mass, n_s, chi2_per_dof
    FROM 'data/oracle_db.parquet'
    WHERE higgs_mass BETWEEN 125.0 AND 125.5
      AND chi2_per_dof < 1.0
    ORDER BY chi2_per_dof
    LIMIT 20
""").df()

# Aggregations
stats = conn.sql("""
    SELECT
        ROUND(delta, 4) as delta_bin,
        COUNT(*) as n_points,
        AVG(chi2_per_dof) as mean_chi2,
        MIN(chi2_per_dof) as best_chi2
    FROM 'data/oracle_db.parquet'
    GROUP BY delta_bin
    ORDER BY delta_bin
""").df()
```

DuckDB is recommended for analytical queries on the full 61.7M rows — it pushes predicates into the Parquet reader for memory-efficient processing.

## The χ² Landscape

The landscape has a distinctive structure:

```
χ²/dof    Landscape cross-section (at fixed δ_g ≈ 0.0417)

  3.0 ┤  ╲                              ╱
      │   ╲                            ╱
  2.0 ┤ ── ╲────── cutoff ────────── ╱ ── (oracle_db)
      │     ╲                      ╱
  1.2 ┤ ──── ╲─── Gold ─────── ╱ ─────── (oracle_gold)
      │       ╲              ╱
  1.0 ┤        ╲            ╱
      │         ╲    ★    ╱   ← axiom point (1.479)
  0.8 ┤          ╰───╮──╯   ← global minimum (~0.89)
      │               │
      └───────────────┴──────────────────→ Δ
        0.200   0.205   0.210   0.215   0.220
```

Key features:
- **Broad valley**: χ²/dof < 1.2 spans Δ ∈ [0.204, 0.214] — a 5% relative width
- **Steep walls**: χ² rises sharply beyond the valley
- **Axiom point slightly off-minimum**: χ²/dof = 1.479 vs. best ~0.89 — the axiom point trades optimality for parameter-freeness
- **δ_g sensitivity**: The landscape is ~10× more sensitive to δ_g than to Δ

## Use Cases

### 1. Finding the Best-Fit Parameters

```python
top = db.best_fit(n=1)
print(f"Best (Δ, δ_g) = ({top.iloc[0]['delta']:.5f}, {top.iloc[0]['delta_g']:.5f})")
print(f"χ²/dof = {top.iloc[0]['chi2_per_dof']:.3f}")
```

### 2. Observable Correlations

Which observables co-vary across the parameter space?

```python
sample = db._data.sample(500_000)  # random subset
corr = sample[observable_columns].corr()
```

Strong correlations reveal shared parameter dependencies (e.g., Ωb and |Vus| both depend on Δ through √Ωb = |Vus|).

### 3. Sensitivity Analysis

How sensitive is each observable to parameter changes?

```python
# Fix δ_g at axiom, vary Δ
delta_scan = db.query("abs(delta_g - 0.04167) < 0.0001")
for obs in ['higgs_mass', 'n_s', 'omega_de']:
    sensitivity = delta_scan[obs].std() / delta_scan[obs].mean()
    print(f"  {obs}: {100*sensitivity:.1f}% variation over scan range")
```

### 4. DESI Compatibility Check

```python
desi_compatible = db.query("desi_w_match == True")
print(f"DESI-compatible points: {len(desi_compatible):,} ({100*len(desi_compatible)/len(db._data):.1f}%)")
```

## Data Provenance

The Oracle Database is generated by the GNN surrogate (not by analytical formulas). This means:
- **Speed**: 100M points in ~6 hours (impossible analytically in reasonable time)
- **Accuracy**: R² > 0.9995 vs. analytical; <0.1% mean absolute error
- **Limitation**: η_B is unreliable (known surrogate artifact — excluded from χ²)

For individual-point verification, use `ParametricForward` (analytical) or `SDGFTPredictor.predict_with_uncertainty()` (ensemble with uncertainty).

---

**Previous:** [← How the CVAE Inverter Works](07_cvae_inverter.md) | **Next:** [Installation & Setup →](09_installation.md)
