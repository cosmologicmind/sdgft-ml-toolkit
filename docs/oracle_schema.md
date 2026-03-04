# Oracle Database Schema

The Oracle Database contains **61.7 million** parameter-space points, each evaluated
through the GNN surrogate ensemble and scored against 21 experimental measurements
(η_B excluded due to unit mismatch in the surrogate).

## Files

| File | Rows | Size | Description |
|------|------|------|-------------|
| `oracle_db.parquet` | 61,701,488 | 3.4 GB | All points with χ²/dof < 2.0 |
| `oracle_gold.parquet` | 35,021,095 | 1.9 GB | Gold Standard: χ²/dof < 1.2 |

**Grid specification:** Δ ∈ [0.200, 0.220] × δ_g ∈ [0.040, 0.043], 10,000 × 10,000 = 100M total points evaluated. Points with χ²/dof ≥ 2.0 were discarded.

## Columns (44 total)

### Parameters (2 columns)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `delta` | float32 | [0.200, 0.220] | Fibonacci-lattice conflict parameter Δ |
| `delta_g` | float32 | [0.040, 0.043] | Lattice tension parameter δ_g |

### Observables — Dimension (4 columns)

| Column | Type | Formula | Description |
|--------|------|---------|-------------|
| `d_star_tree` | float32 | 3 − sin²30° + δ_g | Tree-level effective dimension D* |
| `d_star_fp` | float32 | Fixed-point iteration | RG fixed-point effective dimension |
| `n_tree` | float32 | D*/2 | Tree-level f(R) exponent |
| `n_fp` | float32 | D*_fp/2 | Fixed-point f(R) exponent |

### Observables — Gravity (4 columns)

| Column | Type | Exp. Value | σ | Source |
|--------|------|-----------|---|--------|
| `alpha_m_tree` | float32 | — | — | Modified gravity: α_M = (n−1)/(2n−1) |
| `alpha_b_tree` | float32 | — | — | Modified gravity: α_B = −α_M/2 |
| `eta_slip_survey` | float32 | — | — | Gravitational slip at k/(aH) = 10 |
| `eta_slip_horizon` | float32 | — | — | Gravitational slip at k/(aH) = 1 |

### Observables — Inflation (6 columns)

| Column | Type | Exp. Value | σ | Source |
|--------|------|-----------|---|--------|
| `n_efolds_fp` | float32 | — | — | Number of e-folds N_e |
| `n_s` | float32 | 0.9649 | 0.0042 | Planck 2018 |
| `r_tensor` | float32 | < 0.036 | 0.036 | BICEP/Keck 2021 |
| `beta_iso` | float32 | — | — | Isocurvature fraction (= 1/36) |
| `epsilon_sr` | float32 | — | — | Slow-roll ε |
| `eta_sr` | float32 | — | — | Slow-roll η |

### Observables — Cosmology (7 columns)

| Column | Type | Exp. Value | σ | Source |
|--------|------|-----------|---|--------|
| `omega_b` | float32 | 0.0493 | 0.0020 | Planck 2018 |
| `omega_c` | float32 | 0.265 | 0.007 | Planck 2018 |
| `omega_de` | float32 | 0.6847 | 0.0073 | Planck 2018 |
| `omega_m` | float32 | 0.3153 | 0.0073 | Planck 2018 |
| `w_de_fp` | float32 | −1.03 | 0.03 | Planck+BAO+SN |
| `eta_b` | float32 | 6.143e-10 | 0.19e-10 | Planck+BBN (excluded from χ²) |
| `s_8` | float32 | 0.832 | 0.013 | Planck 2018 |

### Observables — Particle Physics (10 columns)

| Column | Type | Exp. Value | σ | Source |
|--------|------|-----------|---|--------|
| `alpha_em_inv_tree` | float32 | 137.036 | 0.5 (theory) | PDG 2024 CODATA |
| `alpha_em_inv_fp` | float32 | — | — | RG fixed-point α_em⁻¹ |
| `alpha_em_tree` | float32 | — | — | α_em = 1/α_em⁻¹ |
| `alpha_s` | float32 | 0.1180 | 0.0009 | PDG 2024 |
| `sin2_theta_w` | float32 | 0.23122 | 0.00003 | PDG 2024 |
| `mu_e_ratio` | float32 | 206.768 | 1.0 (theory) | PDG 2024 CODATA |
| `tau_mu_ratio_tree` | float32 | 16.817 | 0.1 (theory) | PDG 2024 |
| `lambda_geo` | float32 | 0.1291 | 0.0020 | Derived from m_H |
| `higgs_mass` | float32 | 125.25 GeV | 0.17 | PDG 2024 |
| `n_generations` | float32 | 3.0 | 0.008 | LEP Z width |

### Observables — Neutrinos (3 columns)

| Column | Type | Exp. Value | σ | Source |
|--------|------|-----------|---|--------|
| `theta_12` | float32 | 33.44° | 0.77° | NuFIT 5.3 |
| `theta_23` | float32 | 49.2° | 1.0° | NuFIT 5.3 |
| `theta_13` | float32 | 8.57° | 0.12° | NuFIT 5.3 |

### Observables — CKM (3 columns)

| Column | Type | Exp. Value | σ | Source |
|--------|------|-----------|---|--------|
| `v_us` | float32 | 0.2243 | 0.0005 | PDG 2024 |
| `v_ub` | float32 | 0.00382 | 0.00020 | PDG 2024 |
| `quark_hierarchy` | float32 | — | — | exp(2π) ≈ 535.5 |

### Metadata (5 columns)

| Column | Type | Description |
|--------|------|-------------|
| `total_chi2` | float32 | Sum of (pred − exp)² / σ² across 21 observables |
| `chi2_per_dof` | float32 | total_chi2 / 21 |
| `n_tensions` | int8 | Number of observables with |pull| > 2σ |
| `gold_standard` | bool | True if chi2_per_dof < 1.2 |
| `desi_w_match` | bool | True if w_DE matches DESI (−0.997 ± 0.025) |

## χ² Computation Details

- **Surrogate uncertainty floor**: 1% relative error added in quadrature: `σ_eff = max(σ_exp, 0.01 × |prediction|)`
- **Excluded observable**: `eta_b` (baryon asymmetry) — surrogate predicts ~9×10⁻⁷ vs experimental 6.1×10⁻¹⁰ due to unit mismatch
- **Theory uncertainties**: For `alpha_em_inv_tree` (0.5), `mu_e_ratio` (1.0), `tau_mu_ratio_tree` (0.1), the larger of experimental or theory σ is used

## Loading Examples

### Pandas
```python
import pandas as pd
db = pd.read_parquet("data/oracle_db.parquet")
gold = pd.read_parquet("data/oracle_gold.parquet")
```

### Column-selective (memory-efficient)
```python
cols = ["delta", "delta_g", "higgs_mass", "n_s", "total_chi2"]
db = pd.read_parquet("data/oracle_db.parquet", columns=cols)
```

### DuckDB (zero-copy, SQL queries)
```python
import duckdb
conn = duckdb.connect()
result = conn.sql("""
    SELECT delta, delta_g, higgs_mass, total_chi2
    FROM 'data/oracle_db.parquet'
    WHERE higgs_mass BETWEEN 125.0 AND 125.5
    ORDER BY total_chi2
    LIMIT 10
""").df()
```
