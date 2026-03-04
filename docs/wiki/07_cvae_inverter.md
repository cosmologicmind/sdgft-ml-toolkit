# How the CVAE Inverter Works

## The Inverse Problem

The **forward problem** is straightforward: given parameters (Δ, δ_g, φ), compute the 37 observables. But the **inverse problem** — given (some) observables, what parameters produced them? — is ill-posed:

1. **Many-to-one degeneracy**: Different parameter combinations may produce similar observables
2. **Noise and uncertainty**: Experimental measurements have finite precision
3. **Partial information**: Not all 37 observables are measured experimentally

The **Conditional Variational Autoencoder (CVAE)** solves this by learning a probabilistic inverse mapping.

## Architecture

```
┌─────────────────────────────────────────────┐
│         INPUT: 36 observable values          │
│  (all except eta_b, which is excluded)       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│              ENCODER                         │
│  Linear(36, 128) → SiLU                     │
│  Linear(128, 128) → SiLU                    │
│  Linear(128, 128) → SiLU                    │
│       ┌───────┴───────┐                     │
│  fc_mu(128, 16)   fc_logvar(128, 16)        │
│       μ ∈ ℝ¹⁶         log σ² ∈ ℝ¹⁶        │
└───────┬───────────────┬─────────────────────┘
        │               │
        ▼               ▼
┌─────────────────────────────────────────────┐
│      REPARAMETERIZATION TRICK                │
│                                              │
│  z = μ + σ ⊙ ε,     ε ~ N(0, I)            │
│                                              │
│  (Differentiable sampling from q(z|x))       │
└──────────────────┬──────────────────────────┘
                   │  z ∈ ℝ¹⁶
                   ▼
┌─────────────────────────────────────────────┐
│              DECODER                         │
│  Linear(16, 128) → SiLU                     │
│  Linear(128, 128) → SiLU                    │
│  Linear(128, 128) → SiLU                    │
│  Linear(128, 3)                              │
│       │                                      │
│       ▼                                      │
│  sigmoid(·) → scale to physical ranges       │
│  Δ ∈ [0.01, 0.45]                           │
│  δ_g ∈ [0.001, 0.20]                        │
│  φ ∈ [1.0, 2.0]                             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│      OUTPUT: (Δ̂, δ̂_g, φ̂)                   │
│      + uncertainty from latent sampling      │
└─────────────────────────────────────────────┘
```

### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Input dimension | 36 (not 37) | η_B excluded (unit mismatch in surrogate) |
| Latent dimension | 16 | 3 params but with multi-modal landscape |
| Hidden layers | 3 per encoder/decoder | Sufficient expressive capacity |
| Hidden dimension | 128 | Matches GNN hidden size |
| Activation | SiLU (swish) | Smooth, non-zero gradient everywhere |
| Output activation | Sigmoid → scale | Hard bounds on parameter ranges |
| Total parameters | ~100K | Lightweight; trains in minutes |

## The Variational Autoencoder Framework

### Why Not Just an MLP?

A deterministic MLP f: observables → parameters would give a single point estimate. But the inverse mapping is **not unique** — slightly different parameters can produce indistinguishable observables within measurement errors. We need a **distribution** over possible parameters.

The CVAE provides this by learning a latent distribution q(z|x) conditioned on the observables x, from which we can draw multiple samples, each decoded into a parameter estimate.

### The ELBO Loss

Training minimizes the negative Evidence Lower BOund:

$$\mathcal{L} = \underbrace{\mathbb{E}_{q(z|x)}\left[\|p_\text{true} - p_\text{pred}\|^2\right]}_{\text{Reconstruction loss}} + \beta \cdot \underbrace{D_\text{KL}\left(q(z|x) \| \mathcal{N}(0, I)\right)}_{\text{KL regularization}}$$

Where:
- **Reconstruction loss**: How well do the decoded parameters match the true inputs?
- **KL divergence**: How close is the learned latent distribution to a standard Gaussian?
- **β**: Weight controlling the trade-off (β-VAE)

The per-dimension KL divergence has a closed form for Gaussian distributions:

$$D_\text{KL} = -\frac{1}{2}\sum_{j=1}^{16}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### Free Bits

To prevent **KL collapse** (where the encoder ignores the input and collapses to the prior), we use the "free bits" trick:

$$\text{KL}_j^\text{clamped} = \max\left(\text{KL}_j, \; \lambda_\text{free}\right)$$

Each latent dimension is guaranteed to carry at least λ_free nats of information. This ensures the encoder uses all 16 latent dimensions.

## The Reparameterization Trick

Sampling z ~ q(z|x) = N(μ, σ²) is not differentiable. The reparameterization trick makes it differentiable:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The gradient flows through μ and σ (deterministic), not through the sampling step. At test time (eval mode), we simply use z = μ (the mean).

## Output Scaling

The decoder's raw output is unconstrained. Sigmoid activation maps it to [0, 1], then linear scaling maps to physical ranges:

$$\hat{p}_i = p_{\min,i} + \sigma(\text{raw}_i) \cdot (p_{\max,i} - p_{\min,i})$$

| Parameter | p_min | p_max | Axiom value |
|-----------|-------|-------|-------------|
| Δ | 0.01 | 0.45 | 0.2083 |
| δ_g | 0.001 | 0.20 | 0.04167 |
| φ | 1.0 | 2.0 | 1.618 |

## Uncertainty Quantification

At inference, we draw N samples from the latent distribution and decode each:

```python
for _ in range(n_samples):
    ε ~ N(0, I)
    z = μ + σ ⊙ ε
    params = decode(z)
    → collect

mean_params = mean over samples
std_params = std over samples  ← uncertainty estimate
```

The standard deviation across samples reflects **aleatoric uncertainty** (inherent noise in the inverse mapping) convolved with the learned latent distribution width. Narrow distributions → confident inversion; wide distributions → degenerate solutions.

## Training Data

The CVAE was trained on pairs (observables, parameters) from the same 30K Latin hypercube samples used for the GNN:

1. Generate parameter point (Δ, δ_g, φ) via Latin hypercube
2. Compute 37 observables via ParametricForward
3. Drop η_B → 36 observables as input
4. Target: the original (Δ, δ_g, φ)

Training specifics:
- Epochs: 200
- Optimizer: AdamW (lr=1e-3)
- β schedule: linear warmup 0 → 1 over first 20 epochs
- Free bits: λ = 0.1
- Parameter weights: [1, 5, 0.5] — upweighting δ_g recovery (hardest to invert)

## Using the Inverter

```python
import torch
from sdgft_ml.models import InverterCVAE
from sdgft_ml.data import ParametricForward, observable_names

# Load the trained inverter
inverter = InverterCVAE()
checkpoint = torch.load("checkpoints/inverter/best_inverter.pt", weights_only=True)
inverter.load_state_dict(checkpoint)
inverter.eval()

# Compute observables at the axiom point
fwd = ParametricForward()
obs = fwd.compute_all()
obs_names = observable_names()  # 37 names
obs_vector = torch.tensor([obs[name] for name in obs_names], dtype=torch.float32)

# Remove eta_b (index 19) for 36-dim input
obs_36 = torch.cat([obs_vector[:19], obs_vector[20:]])

# Invert: observables → parameters
mean_params, std_params = inverter.invert(obs_36, n_samples=200)
print(f"Δ = {mean_params[0]:.4f} ± {std_params[0]:.4f}")
print(f"δ_g = {mean_params[1]:.5f} ± {std_params[1]:.5f}")
print(f"φ = {mean_params[2]:.4f} ± {std_params[2]:.4f}")

# Expected output (approximately):
# Δ = 0.2083 ± 0.001
# δ_g = 0.04167 ± 0.002
# φ = 1.618 ± 0.01
```

## Round-Trip Validation

The quality of the inverter is measured by the **round-trip error**:

1. Start with true parameters (Δ, δ_g, φ)
2. Forward: compute observables via ParametricForward
3. Inverse: recover parameters via CVAE
4. Compare recovered to original

On the validation set:
- **Δ recovery**: < 1% relative error
- **δ_g recovery**: < 3% relative error (hardest — most observables are weakly sensitive to δ_g)
- **φ recovery**: < 0.5% relative error

## Checkpoint Layout

```
checkpoints/inverter/
└── best_inverter.pt   # PyTorch state dict (~1.7 MB)
```

---

**Previous:** [← How the GNN Surrogate Works](06_gnn_surrogate.md) | **Next:** [The Oracle Database →](08_oracle_database.md)
