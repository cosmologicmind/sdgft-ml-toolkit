# SDGFT-ML-Toolkit

**Inference, querying & exploration toolkit for Six-Dimensional Geometric Field Theory (SDGFT)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18863347.svg)](https://doi.org/10.5281/zenodo.18863347)

## What is SDGFT?

Six-Dimensional Geometric Field Theory (SDGFT) is a parameter-free unified theory that derives all fundamental constants — from the Higgs mass to dark energy — from two geometric quantities on a 6D Fibonacci lattice: the conflict parameter Δ and the lattice tension δ_g. At the axiom point (Δ = 5/24, δ_g = 1/24), SDGFT predicts 37 observables spanning cosmology, particle physics, gravity, and inflation with zero free parameters.

## What's in this Toolkit?

| Component | Description |
|-----------|-------------|
| **GNN Ensemble** | 5-member GATv2 graph neural network trained on the SDGFT computation DAG. Maps (Δ, δ_g) → 37 observables in microseconds. |
| **Oracle Database** | 100M pre-computed parameter points (61.7M kept, 35M Gold Standard) stored in Parquet with χ² scores against 22 experiments. |
| **CVAE Inverter** | Conditional variational autoencoder solving the inverse problem: observables → (Δ, δ_g, φ). |
| **Theory Engine** | Self-contained `ParametricForward` — all 37 SDGFT formulas in pure Python, no external dependencies. |
| **6 Notebooks** | From quickstart to frontier physics predictions, each fully self-contained. |

## Installation

```bash
git clone https://github.com/cosmologicmind/sdgft-ml-toolkit.git
cd sdgft-ml-toolkit
pip install -e ".[jupyter]"
nbstripout --install   # auto-strips notebook outputs before every commit

# Download Oracle Database from Zenodo (~5.1 GB, with progress bar)
python data/download_oracle.py

# Download SPARC galaxy database (required for rotation-curve training/validation)
python data/download_sparc.py
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1, PyTorch Geometric ≥ 2.4

## Quick Start

### Predict observables

```python
from sdgft_ml.inference import SDGFTPredictor

predictor = SDGFTPredictor()          # loads 5-member ensemble
result = predictor.predict()           # axiom point (Δ=5/24, δ_g=1/24)

print(f"Higgs mass:   {result['higgs_mass']:.2f} GeV")   # → 125.63 GeV
print(f"sin²θ_W:      {result['sin2_theta_w']:.5f}")      # → 0.23122
print(f"n_s:          {result['n_s']:.4f}")                # → 0.9664

# With uncertainty
detailed = predictor.predict_with_uncertainty(0.210, 0.042)
for obs in ["higgs_mass", "n_s", "omega_de"]:
    d = detailed[obs]
    print(f"  {obs}: {d['mean']:.4f} ± {d['std']:.4f}")
```

### Query the Oracle Database

```python
from sdgft_ml.inference import OracleDB

db = OracleDB()                                    # lazy-loads 3.4 GB Parquet
print(db.summary())

# Best-fit points
top10 = db.best_fit(n=10)

# Filter by observable
higgs = db.filter_observable("higgs_mass", 125.0, 125.5)

# SQL-like queries
matches = db.query("higgs_mass > 125 and n_s < 0.97 and gold_standard == True")

# Gold Standard subset (35M points, χ²/dof < 1.2)
gold = db.gold_standard()
```

### Exact theory predictions (no ML)

```python
from sdgft_ml.data import ParametricForward

fwd = ParametricForward(delta=5/24, delta_g=1/24)
obs = fwd.compute_all()

print(f"Ω_DE = {obs['omega_de']:.4f}")     # → 0.6847
print(f"α_s  = {obs['alpha_s']:.4f}")       # → 0.1178
print(f"r    = {obs['r_tensor']:.4f}")       # → 0.0135
```

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 00 | [Quickstart](notebooks/00_quickstart.ipynb) | 5-minute intro: load model, predict, query Oracle |
| 01 | [Oracle Queries](notebooks/01_oracle_queries.ipynb) | Deep dive into the 61.7M-point database with pandas & DuckDB |
| 02 | [Parameter Landscape](notebooks/02_parameter_landscape.ipynb) | χ² heatmaps, sensitivity maps, correlation analysis |
| 03 | [Experimental Validation](notebooks/03_experimental_validation.ipynb) | 22-observable scorecard against PDG/Planck/NuFIT |
| 04 | [Predictions & Frontier](notebooks/04_predictions_frontier.ipynb) | W-boson mass, muon g-2, dark radiation, gravitational waves |
| 05 | [Inverse Problem](notebooks/05_inverse_problem.ipynb) | CVAE parameter recovery from observables |

## Project Structure

```
sdgft-ml-toolkit/
├── src/sdgft_ml/
│   ├── data/               # ParametricForward, DAG builder
│   ├── models/             # SurrogateGNN, InverterCVAE
│   ├── inference/          # SDGFTPredictor, OracleDB (high-level API)
│   └── validation/         # 22 experimental measurements + scorecard
├── checkpoints/
│   ├── ensemble/           # 5 × GATv2 (best_model.pt + norms.npz)
│   └── inverter/           # CVAE (best_inverter.pt)
├── data/
│   ├── oracle_db.parquet   # 61.7M rows, 44 columns (3.4 GB)  ⬇ Zenodo
│   ├── oracle_gold.parquet # 35M rows, Gold Standard (1.9 GB)  ⬇ Zenodo
│   └── oracle_landscape.png
├── notebooks/              # 6 self-contained notebooks
├── docs/                   # Architecture, schema, experimental data reference
└── tests/
```

## Model Architecture

The GNN surrogate mirrors the SDGFT computation DAG:

- **Input**: 3 parameters (Δ, δ_g, φ)
- **Graph**: 37 nodes (observables) connected by 45+ directed edges (dependency links)
- **Architecture**: GATv2Conv — 128-dim embeddings, 8 attention heads, 6 message-passing layers
- **Parameters**: ~1.3M per model, 5 ensemble members
- **Training**: 30K Latin-hypercube samples, hybrid loss (MSE + log-cosh + relative error)
- **Fidelity**: R² = 0.9995 round-trip, val_loss < 1e-4

## Oracle Database Schema

The Parquet files contain 44 columns:

| Group | Columns |
|-------|---------|
| Parameters | `delta`, `delta_g` |
| Observables (37) | `d_star_tree`, `d_star_fp`, `n_tree`, `n_fp`, `alpha_m_tree`, `alpha_b_tree`, `eta_slip_survey`, `eta_slip_horizon`, `n_efolds_fp`, `n_s`, `r_tensor`, `beta_iso`, `epsilon_sr`, `eta_sr`, `omega_b`, `omega_c`, `omega_de`, `omega_m`, `w_de_fp`, `eta_b`, `s_8`, `alpha_em_inv_tree`, `alpha_em_inv_fp`, `alpha_em_tree`, `alpha_s`, `sin2_theta_w`, `mu_e_ratio`, `tau_mu_ratio_tree`, `lambda_geo`, `higgs_mass`, `n_generations`, `theta_12`, `theta_23`, `theta_13`, `v_us`, `v_ub`, `quark_hierarchy` |
| Metadata | `total_chi2`, `chi2_per_dof`, `n_tensions`, `gold_standard`, `desi_w_match` |

See [docs/oracle_schema.md](docs/oracle_schema.md) for the full column reference with units and experimental values.

## Experimental Data Sources

- **PDG 2024**: α_em, α_s, sin²θ_W, m_H, m_μ/m_e, m_τ/m_μ, |V_us|, |V_ub|, N_gen
- **Planck 2018** (TT,TE,EE+lowE+lensing): Ω_b, Ω_c, Ω_m, Ω_DE, S_8, η_B, n_s, w_DE
- **NuFIT 5.3**: θ₁₂, θ₂₃, θ₁₃
- **BICEP/Keck 2021**: r (tensor-to-scalar ratio upper limit)

Full reference: [docs/experimental_data.md](docs/experimental_data.md)

## Data (Oracle Database)

The two Parquet data files (5.1 GB total) are hosted on Zenodo:

> **DOI:** [10.5281/zenodo.18863347](https://doi.org/10.5281/zenodo.18863347)

```bash
# Download into data/
python data/download_oracle.py
```

## Citation

If you use this software or the Oracle Database, please cite:

```bibtex
@software{besemer2026sdgft_ml_toolkit,
  author  = {Besemer, David A.},
  title   = {{SDGFT-ML-Toolkit}: Inference \& Exploration for Six-Dimensional Geometric Field Theory},
  year    = {2026},
  url     = {https://github.com/cosmologicmind/sdgft-ml-toolkit},
  version = {1.0.0},
}

@dataset{besemer2026sdgft_oracle,
  author  = {Besemer, David A.},
  title   = {{SDGFT Oracle Database}: High-Resolution Parameter-Observable Lattice},
  year    = {2026},
  doi     = {10.5281/zenodo.18863347},
  publisher = {Zenodo},
}
```

## License

MIT — see [LICENSE](LICENSE).
