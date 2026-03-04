# How the GNN Surrogate Works

## The Problem

Evaluating all 37 SDGFT observables analytically via `ParametricForward` takes ~2 ms per point. This sounds fast, but for a 100M-point parameter sweep, that is ~56 CPU-hours. The GNN surrogate reduces this to **~50 μs per point on GPU** — a 40× speedup that made the Oracle Database feasible.

## Why a Graph Neural Network?

The 37 SDGFT observables are not independent: they form a **directed acyclic graph** (DAG) where early observables (dimension) feed into later ones (gravity → inflation → cosmology → particles). A standard MLP would ignore this structure. The GNN encodes it directly:

- Each **node** = one observable
- Each **directed edge** = one causal dependency (e.g., D* → n_s)
- **Message passing** = information flows along physical causation

This inductive bias ensures the surrogate respects the physics, even in the untrained corners of parameter space.

## Architecture: GATv2Conv

The model uses **Graph Attention Network v2** (GATv2, Brody et al. 2022), which learns dynamic attention weights on each edge — effectively learning *how much* each upstream observable matters for each downstream one.

### Full Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│                 INPUT: (Δ, δ_g, φ)               │
│                   3 scalar parameters             │
└─────────────────────┬────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│            PARAMETER ENCODER (MLP)                │
│  3 → 128 → 128 → (37 × 128)                     │
│  Produces one 128-d embedding per DAG node        │
└─────────────────────┬────────────────────────────┘
                      │  37 node embeddings, each ∈ ℝ¹²⁸
                      ▼
┌──────────────────────────────────────────────────┐
│         GATv2Conv × 6 LAYERS                      │
│                                                   │
│  Per layer:                                       │
│    1. Multi-head attention: 8 heads × 16 dim      │
│       → 128-dim output                            │
│    2. LayerNorm                                   │
│    3. SiLU activation                             │
│    4. Residual connection (x + layer(x))          │
│                                                   │
│  Message passing follows the DAG edges:           │
│    D* → n → αM, αB → Nₑ → ns, r → ...          │
└─────────────────────┬────────────────────────────┘
                      │  37 refined embeddings, each ∈ ℝ¹²⁸
                      ▼
┌──────────────────────────────────────────────────┐
│           NODE DECODER (per node MLP)             │
│  128 → 128 → 1                                   │
│  Single scalar prediction per observable          │
└─────────────────────┬────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│          OUTPUT: 37 normalized predictions         │
│  De-normalized using per-member norms.npz         │
│  (z-score: predicted = raw × std + mean)          │
└──────────────────────────────────────────────────┘
```

### GATv2 Attention Mechanism

For each edge (i → j) in the DAG, the attention coefficient is:

$$\alpha_{ij} = \frac{\exp\left(\mathbf{a}^T \, \text{LeakyReLU}\left(\mathbf{W}\left[\mathbf{h}_i \| \mathbf{h}_j\right]\right)\right)}{\sum_{k \in \mathcal{N}(j)} \exp\left(\mathbf{a}^T \, \text{LeakyReLU}\left(\mathbf{W}\left[\mathbf{h}_k \| \mathbf{h}_j\right]\right)\right)}$$

where ‖ denotes concatenation, **W** is a learnable weight matrix, and **a** is the attention vector. The key improvement of GATv2 over GATv1 is that the attention function is **universally expressive** — it can attend to any function of the input features, not just static rankings.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden dimension | 128 | Balance between capacity and speed |
| Attention heads | 8 | Multi-view message aggregation |
| Per-head dimension | 16 | 128/8 = 16 per head |
| GATv2 layers | 6 | Matches the 6 DAG levels |
| Dropout | 0.1 | Regularization during training |
| Activation | SiLU (swish) | Smooth, non-monotonic; outperforms ReLU on physics tasks |
| Normalization | LayerNorm | Stabilizes training with varied observable scales |
| Skip connections | Residual (additive) | Gradient flow across layers |
| Parameters/model | **~1.3M** | Modest model; 5 members = 6.5M total |

## The DAG Structure

37 nodes connected by ~45 directed edges:

```
d_star_tree ────→ n_tree ────→ alpha_m_tree
     │                              │
     ├─→ alpha_em_inv_tree          ├─→ alpha_b_tree
     ├─→ tau_mu_ratio_tree          │
     ├─→ mu_e_ratio                 │
     │                              ▼
d_star_fp ──────→ n_fp ──→ n_efolds_fp ──→ n_s
                                    │         │
                                    │         ├─→ r_tensor
                                    │         ├─→ epsilon_sr
                                    │         └─→ eta_sr
                                    │
delta ──→ omega_b ──→ omega_m ──→ s_8
     │         │
     ├─→ omega_c ──→ omega_de
     ├─→ theta_13
     ├─→ theta_23
     ├─→ lambda_geo ──→ higgs_mass
     │
delta_g ──→ eta_b
       ──→ theta_12
       ──→ n_generations
```

The edges are defined in `dag_builder.py` and form the adjacency matrix for the GATv2 layers.

## Ensemble: 5 Independent Members

Instead of one model, we train **5 models** with different random seeds. At inference:

$$\hat{y}_{\text{ensemble}} = \frac{1}{5} \sum_{m=1}^{5} \hat{y}_m$$

$$\sigma_{\text{epistemic}} = \text{std}\left(\hat{y}_1, \ldots, \hat{y}_5\right)$$

The ensemble provides:
- **Better accuracy**: averaging reduces individual model biases
- **Uncertainty estimates**: spread quantifies epistemic (model) uncertainty
- **Robustness**: prediction is stable even if one member has a local error

## Normalization

Observables span vastly different scales:

| Observable | Scale |
|-----------|-------|
| η_B | ~10⁻¹⁰ |
| ε_SR | ~10⁻⁴ |
| α_s | ~10⁻¹ |
| α_em⁻¹ | ~10² |
| quark_hierarchy | ~10² |

Without normalization, the loss function would be dominated by large-scale observables. The solution: **per-observable z-score normalization**:

$$\tilde{y}_i = \frac{y_i - \mu_i}{\max(\sigma_i, \, 0.001)}$$

Each ensemble member stores its own mean/std arrays in `norms.npz`. At inference:

$$\hat{y}_i = \tilde{y}_i^{\text{raw}} \times \sigma_i + \mu_i$$

## Training Details

| Aspect | Value |
|--------|-------|
| Training data | 30,000 Latin hypercube samples |
| Parameter ranges | Δ ∈ [0.05, 0.40], δ_g ∈ [0.01, 0.08] |
| Validation split | 15% (4,500 held-out points) |
| Loss function | 0.7 × MSE + 0.3 × log-cosh + relative error |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR schedule | Cosine annealing with warm restarts (T₀=20, T_mult=2) |
| Per-observable weights | Fisher-information-based sensitivity |
| Epochs per member | 200 |
| Best model | Lowest validation loss |
| Training time | ~10 minutes per member on GPU |

### Why Hybrid Loss?

The standard MSE loss penalizes large errors quadratically, which works well for observables with Gaussian-distributed residuals. But the log-cosh term adds robustness:

$$\mathcal{L}_{\text{log-cosh}}(y, \hat{y}) = \frac{1}{N}\sum_i \log \cosh(y_i - \hat{y}_i)$$

This behaves like L2 for small residuals and L1 for large residuals, preventing outlier observables from dominating the gradient.

The relative error term ensures good fractional accuracy even for small observables:

$$\mathcal{L}_{\text{rel}} = \frac{1}{N}\sum_i \left(\frac{y_i - \hat{y}_i}{\max(|y_i|, \epsilon)}\right)^2$$

## Performance

| Metric | Value |
|--------|-------|
| Validation loss | < 10⁻⁴ |
| R² (round-trip) | 0.9995 |
| Mean absolute error | < 0.1% per observable |
| Inference time (single point, GPU) | ~50 μs |
| Inference time (single point, CPU) | ~2 ms |
| Batch throughput (GPU) | ~200K points/sec |

## Using the Predictor

```python
from sdgft_ml.inference import SDGFTPredictor

# Load ensemble (auto-detects GPU)
predictor = SDGFTPredictor()

# Single-point prediction
result = predictor.predict(delta=5/24, delta_g=1/24)
print(f"Higgs: {result['higgs_mass']:.2f} GeV")

# With uncertainty
detailed = predictor.predict_with_uncertainty(0.210, 0.042)
for name in ['higgs_mass', 'n_s', 'omega_de']:
    d = detailed[name]
    print(f"  {name}: {d['mean']:.4f} ± {d['std']:.4f}")

# Batch prediction (vectorized)
import numpy as np
params = np.random.uniform([0.19, 0.038], [0.22, 0.044], size=(10000, 2))
df = predictor.predict_batch(params)
```

## Checkpoint Layout

```
checkpoints/ensemble/
├── member_0/
│   ├── best_model.pt   # PyTorch state dict (~3.3 MB)
│   └── norms.npz       # mean + std arrays (37 × 2 floats)
├── member_1/
│   ├── best_model.pt
│   └── norms.npz
├── member_2/ ...
├── member_3/ ...
└── member_4/ ...
```

---

**Previous:** [← Experimental Validation](05_experimental_validation.md) | **Next:** [How the CVAE Inverter Works →](07_cvae_inverter.md)
