# API Reference

Complete reference for the four main modules of the SDGFT-ML Toolkit.

---

## `sdgft_ml.inference.SDGFTPredictor`

High-level interface for GNN ensemble predictions.

### Constructor

```python
SDGFTPredictor(
    checkpoint_dir: str | Path = None,   # Default: <project>/checkpoints/ensemble/
    device: str = "auto",                # "auto", "cpu", or "cuda"
    n_members: int = 5,                  # Ensemble size
    hidden_dim: int = 128,               # GATv2 hidden dimension
    n_heads: int = 8,                    # Attention heads
    n_layers: int = 6,                   # Message-passing layers
)
```

### Methods

#### `predict(delta, delta_g, phi) → dict[str, float]`

Predict 37 observables at a single parameter point (ensemble mean).

```python
result = predictor.predict(delta=5/24, delta_g=1/24)
# Returns: {"d_star_tree": 2.79, "higgs_mass": 124.94, ...}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta` | 5/24 | Fibonacci conflict parameter |
| `delta_g` | 1/24 | Lattice tension |
| `phi` | (1+√5)/2 | Golden ratio |

#### `predict_with_uncertainty(delta, delta_g, phi) → dict[str, dict]`

Returns mean and ensemble standard deviation for each observable.

```python
result = predictor.predict_with_uncertainty(0.210, 0.042)
# Returns: {"higgs_mass": {"mean": 124.3, "std": 0.15}, ...}
```

#### `predict_batch(params, batch_size=5000) → pd.DataFrame`

Vectorized prediction for many parameter points. Uses first ensemble member for speed.

```python
import numpy as np
params = np.random.uniform([0.19, 0.04], [0.22, 0.043], size=(10000, 2))
df = predictor.predict_batch(params)
# Returns DataFrame with columns: delta, delta_g, phi, + 37 observables
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | (N, 2) or (N, 3) array | If 2 columns, φ defaults to golden ratio |
| `batch_size` | int | GPU batch size |

#### `info` (property) → dict

Model metadata: ensemble size, architecture, device, checkpoint path.

---

## `sdgft_ml.inference.OracleDB`

Query interface for the pre-computed parameter-space database.

### Constructor

```python
OracleDB(
    parquet_path: str | Path = None,      # Default: <project>/data/oracle_db.parquet
    gold_path: str | Path = None,         # Default: <project>/data/oracle_gold.parquet
    lazy: bool = True,                    # Load on first query (saves memory)
)
```

### Methods

#### `best_fit(n=10) → pd.DataFrame`

Return the n points with lowest χ²/dof.

```python
top = db.best_fit(n=5)
```

#### `gold_standard() → pd.DataFrame`

Return all Gold Standard points (χ²/dof < 1.2, ~35M rows).

```python
gold = db.gold_standard()
```

#### `filter_observable(name, min_val, max_val) → pd.DataFrame`

Filter by observable range.

```python
higgs = db.filter_observable("higgs_mass", 125.0, 125.5)
```

#### `query(expr) → pd.DataFrame`

Pandas-style query string.

```python
result = db.query("higgs_mass > 125 and n_s < 0.97 and gold_standard == True")
```

#### `parameter_range() → dict`

Statistics on Δ and δ_g ranges in the database.

#### `chi2_heatmap(bins=100, ranges=None) → tuple[ndarray, ndarray, ndarray]`

Compute 2D histogram of minimum χ²/dof per (Δ, δ_g) bin.

```python
chi2_map, delta_edges, deltag_edges = db.chi2_heatmap(bins=200)
```

#### `summary() → str`

Human-readable summary of the database.

---

## `sdgft_ml.data.ParametricForward`

Self-contained analytical theory engine. Computes all 37 SDGFT observables.

### Constructor

```python
ParametricForward(
    delta: float = 5/24,              # Fibonacci conflict
    delta_g: float = 1/24,            # Lattice tension
    phi: float = (1+√5)/2,            # Golden ratio
    gamma_ew: float = 0.12011,        # Electroweak RG correction
    v_higgs: float = 246.22,          # Higgs VEV (GeV)
)
```

### Key Methods

#### `compute_all() → dict[str, float]`

Compute all observables. Returns a dict with ~50 entries (37 OBSERVABLE_KEYS + input parameters + auxiliary quantities).

```python
fwd = ParametricForward(delta=5/24, delta_g=1/24)
obs = fwd.compute_all()
print(obs['higgs_mass'])  # 125.30
```

#### `feature_vector() → np.ndarray`

Extract the 37 ML-compatible observables as a float array, ordered by `OBSERVABLE_KEYS`.

#### `param_vector() → np.ndarray`

Return [delta, delta_g, phi] as a numpy array.

### Observable Properties & Methods

Each observable is available as a property or method:

| Category | Attributes |
|----------|-----------|
| Dimension | `d_star_tree`, `d_star_fp`, `n_tree`, `n_fp` |
| Gravity | `alpha_m(n)`, `alpha_b(n)`, `grav_slip(n, k_over_aH)` |
| Inflation | `e_folds(d_star)`, `spectral_index(n, n_e)`, `tensor_to_scalar(n, n_e)`, `slow_roll_epsilon(n, n_e)`, `slow_roll_eta(n, n_e)`, `beta_iso` |
| Cosmology | `omega_b`, `omega_c`, `omega_de`, `omega_m`, `w_de(d_star)`, `eta_b`, `s_8` |
| Particle physics | `alpha_em_inv(d_star)`, `alpha_s`, `sin2_theta_w`, `mu_e_ratio(d_star)`, `tau_mu_ratio(d_star)`, `lambda_geo`, `higgs_mass`, `n_generations` |
| Neutrinos | `theta_12()`, `theta_23()`, `theta_13()` |
| CKM | `v_us()`, `v_ub()`, `quark_hierarchy` |

### Class Variables

```python
ParametricForward.OBSERVABLE_KEYS  # List[str], 37 observable names (ML order)
ParametricForward.PARAM_KEYS       # List[str], ["param_delta", "param_delta_g", "param_phi"]
```

### Utility Functions

```python
from sdgft_ml.data import sweep_grid, sweep_to_dataframe

# Generate a grid of parameter points
samples = sweep_grid(n_delta=100, n_delta_g=100)  # returns list of ParametricForward
df = sweep_to_dataframe(samples)                    # returns pandas DataFrame
```

---

## `sdgft_ml.validation`

Experimental validation against 22 precision measurements.

### Constants

```python
from sdgft_ml.validation import EXPERIMENTAL_DATA

# Dict of 22 ExperimentalValue namedtuples:
# ExperimentalValue(name, symbol, value, sigma, source, sigma_theory)
for name, ev in EXPERIMENTAL_DATA.items():
    print(f"{name}: {ev.value} ± {ev.sigma} ({ev.source})")
```

### Functions

#### `validate_at_axiom() → list[dict]`

Validate at the axiom point (Δ = 5/24, δ_g = 1/24). Returns a list of dicts, one per observable:

```python
results = validate_at_axiom()
# Each entry: {"name": str, "predicted": float, "experimental": float,
#              "sigma": float, "pull": float, "within_1sigma": bool, ...}
```

#### `validate_at_point(delta, delta_g, phi=PHI) → list[dict]`

Validate at an arbitrary parameter point.

```python
results = validate_at_point(delta=0.210, delta_g=0.042)
```

#### `chi_squared(results) → dict`

Compute χ² statistics from validation results.

```python
chi2 = chi_squared(results)
# Returns: {"chi2_total": float, "chi2_per_dof": float,
#           "n_dof": int, "p_value": float}
```

#### `scorecard(results) → list[dict]`

Formatted scorecard with pulls, status flags, and per-category breakdown.

#### `validate_surrogate_vs_real(predictor, delta, delta_g) → pd.DataFrame`

Compare GNN predictions against analytical ParametricForward at the same point.

---

## `sdgft_ml.models`

Low-level model classes (usually accessed indirectly through SDGFTPredictor).

### `SurrogateGNN`

```python
from sdgft_ml.models import SurrogateGNN

model = SurrogateGNN(
    n_params=3,        # Input: (Δ, δ_g, φ)
    n_nodes=37,        # Output: 37 observables
    hidden_dim=128,
    n_heads=8,
    n_layers=6,
    dropout=0.1,
)

# Forward: (B, 3) params + (2, E) edge_index → (B*37,) predictions
output = model(params_tensor, edge_index)
```

### `InverterCVAE`

```python
from sdgft_ml.models import InverterCVAE

inverter = InverterCVAE(
    n_observables=36,  # Input: observables (excluding η_B)
    n_params=3,        # Output: (Δ, δ_g, φ)
    hidden_dim=128,
    latent_dim=16,
    n_hidden=3,
)

# Forward: (B, 36) → params_pred, mu, logvar
params, mu, logvar = inverter(obs_tensor)

# Inference with uncertainty:
mean_params, std_params = inverter.invert(obs_tensor, n_samples=100)
```

---

## `sdgft_ml.data.dag_builder`

DAG construction utilities.

```python
from sdgft_ml.data import build_dag, dag_to_pyg, observable_names

# Observable name list (37 entries, matching GNN node order)
names = observable_names()

# Build adjacency dict and name list
adj, names = build_dag()

# Build PyTorch Geometric edge index
edge_index = dag_to_pyg()  # returns (2, E) numpy array
```

---

**Previous:** [← Installation & Setup](09_installation.md) | **Back to → [Wiki Home](README.md)**
