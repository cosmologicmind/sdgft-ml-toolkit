# Model Architecture

## Overview

The SDGFT-ML surrogate replaces the analytical SDGFT computation chain with a
differentiable, GPU-accelerated graph neural network. The GNN topology mirrors the
physical dependency structure: each observable is a node, each causal dependency
is a directed edge.

## GNN Surrogate (GATv2)

```
Input: (Δ, δ_g, φ) — 3 scalar parameters
  ↓
Parameter Encoder: MLP 3 → 128 → 128 → (37 × 128)
  ↓ (one 128-d embedding per node)
GATv2Conv Layer 1:  128 → 16×8 heads → 128  + LayerNorm + SiLU + Residual
GATv2Conv Layer 2:  128 → 16×8 heads → 128  + LayerNorm + SiLU + Residual
GATv2Conv Layer 3:  128 → 16×8 heads → 128  + LayerNorm + SiLU + Residual
GATv2Conv Layer 4:  128 → 16×8 heads → 128  + LayerNorm + SiLU + Residual
GATv2Conv Layer 5:  128 → 16×8 heads → 128  + LayerNorm + SiLU + Residual
GATv2Conv Layer 6:  128 → 16×8 heads → 128  + LayerNorm + SiLU + Residual
  ↓
Node Decoder: MLP 128 → 128 → 1  (per node)
  ↓
Output: 37 scalar predictions (one per observable)
```

**Total parameters:** ~1.3M per model

## Ensemble

5 independently trained models (different random seeds), combined at inference:

- **Mean**: ensemble average as point estimate
- **Std**: ensemble spread as epistemic uncertainty
- Predictions are de-normalized using per-member `norms.npz` (mean/std arrays)

## DAG Structure

The 37 observable nodes are organized in a dependency DAG with ~45 directed edges:

```
Level 0-1: Parameters (Δ, δ_g, φ)
    ↓
Level 2: Dimension
    d_star_tree, d_star_fp, n_tree, n_fp
    ↓
Level 3: Gravity
    alpha_m_tree, alpha_b_tree, eta_slip_survey, eta_slip_horizon
    ↓
Level 4: Inflation
    n_efolds_fp, n_s, r_tensor, beta_iso, epsilon_sr, eta_sr
    ↓
Level 5-6: Cosmology + Particle Physics + Neutrinos + CKM
    omega_b, omega_c, omega_de, omega_m, w_de_fp, eta_b, s_8,
    alpha_em_inv_tree/fp, alpha_em_tree, alpha_s, sin2_theta_w,
    mu_e_ratio, tau_mu_ratio_tree, lambda_geo, higgs_mass,
    n_generations, theta_12/23/13, v_us, v_ub, quark_hierarchy
```

Information flows **downward** — dimension feeds into gravity, gravity into
inflation, and so on. The GATv2 attention mechanism learns the relative
importance of each edge dynamically.

## Normalization

All observables span vastly different scales (from 10⁻¹⁰ for η_B to 10² for
α_em⁻¹). Training uses per-observable z-score normalization:

```
normalized = (raw - mean) / max(std, 0.001)
```

The `norms.npz` files in each checkpoint store these arrays.
At inference, predictions are de-normalized:

```
predicted = raw_output × std + mean
```

## CVAE Inverter

The inverse problem (observables → parameters) is solved by a conditional
variational autoencoder:

```
Encoder: 36 observables → 128 → 128 → 128 → (μ, log σ²) in ℝ¹⁶
    ↓ reparameterization trick
Decoder: z ∈ ℝ¹⁶ → 128 → 128 → 128 → 3 → sigmoid scaling
    ↓
Output: (Δ, δ_g, φ) in valid ranges
```

Parameter ranges: Δ ∈ [0.01, 0.45], δ_g ∈ [0.001, 0.20], φ ∈ [1.0, 2.0]

## Training Details

- **Data**: 30,000 Latin-hypercube samples over Δ ∈ [0.05, 0.40], δ_g ∈ [0.01, 0.08]
- **Loss**: Hybrid — 0.7 × MSE + 0.3 × log-cosh + relative error term
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Schedule**: Cosine annealing with warm restarts (T₀=20, T_mult=2)
- **Per-observable weights**: Sensitivity-based (from Fisher information)
- **Epochs**: 200 per member
- **Validation split**: 15%
- **Best model**: Selected by lowest validation loss
