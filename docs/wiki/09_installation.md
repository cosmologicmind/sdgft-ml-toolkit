# Installation & Setup

## Requirements

| Dependency | Minimum Version | Purpose |
|-----------|----------------|---------|
| Python | 3.10 | f-string syntax, type hints |
| PyTorch | 2.1 | GNN & CVAE inference |
| PyTorch Geometric | 2.4 | GATv2Conv graph layers |
| NumPy | 1.24 | Array operations |
| Pandas | 2.0 | DataFrames, Parquet I/O |
| PyArrow | 12.0 | Parquet file reader |
| SciPy | 1.10 | χ² p-value computation |
| Matplotlib | 3.7 | Plotting (notebooks) |

### Optional

| Dependency | Purpose |
|-----------|---------|
| DuckDB | SQL queries on Parquet (notebooks 01, 02) |
| JupyterLab | Running notebooks |
| Seaborn | Enhanced visualizations |

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/cosmologicmind/sdgft-ml-toolkit.git
cd sdgft-ml-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with all optional dependencies
pip install -e ".[jupyter,duckdb,dev]"
```

### Minimal Install (Inference Only)

```bash
pip install -e .
```

### Optional Dependency Groups

```bash
pip install -e ".[jupyter]"   # + jupyterlab, ipywidgets
pip install -e ".[duckdb]"    # + duckdb (SQL on Parquet)
pip install -e ".[dev]"       # + pytest, ruff, mypy
```

## Data Files

The Parquet data files (5.1 GB total) are **not included in git** — they must be downloaded or generated separately.

### Option 1: Copy from Source Project

If you have the original `sdgft_ml` project:

```bash
cp /path/to/sdgft_ml/oracle_db.parquet data/
cp /path/to/sdgft_ml/oracle_gold.parquet data/
```

### Option 2: Use Without Parquet

The toolkit works without the Parquet files for:
- ✅ GNN predictions (`SDGFTPredictor`)
- ✅ Analytical calculations (`ParametricForward`)
- ✅ Validation (`validate_at_axiom`, `chi_squared`)
- ✅ CVAE inversion
- ❌ Oracle queries (`OracleDB` — needs Parquet)
- ❌ Notebooks 01 (Oracle Queries) and 02 (Parameter Landscape)

### Verifying the Installation

```python
# Test 1: Package imports
import sdgft_ml
print(f"Version: {sdgft_ml.__version__}")

# Test 2: GNN prediction
from sdgft_ml.inference import SDGFTPredictor
p = SDGFTPredictor()
result = p.predict()
print(f"Higgs mass: {result['higgs_mass']:.2f} GeV")

# Test 3: Analytical theory
from sdgft_ml.data import ParametricForward
fwd = ParametricForward()
print(f"D* = {fwd.d_star_tree:.4f}")

# Test 4: Validation
from sdgft_ml.validation import validate_at_axiom, chi_squared
chi2 = chi_squared(validate_at_axiom())
print(f"χ²/dof = {chi2['chi2_per_dof']:.3f}")

# Test 5: Oracle (requires Parquet files)
from sdgft_ml.inference import OracleDB
db = OracleDB()
print(db.summary())
```

## GPU Support

The toolkit auto-detects CUDA:

```python
from sdgft_ml.inference import SDGFTPredictor
p = SDGFTPredictor(device="auto")  # CUDA if available, else CPU
print(p.info['device'])
```

For CPU-only systems, all features work identically — just slower for batch predictions.

## Project Directory Structure

After setup, the project should look like:

```
sdgft-ml-toolkit/
├── .venv/                  # Virtual environment (not in git)
├── checkpoints/
│   ├── ensemble/           # 5 GNN members (included in git)
│   │   ├── member_0/
│   │   │   ├── best_model.pt
│   │   │   └── norms.npz
│   │   └── ... (member_1 through member_4)
│   └── inverter/
│       └── best_inverter.pt
├── data/
│   ├── oracle_db.parquet   # 3.2 GB (NOT in git — download separately)
│   ├── oracle_gold.parquet # 1.9 GB (NOT in git — download separately)
│   └── oracle_landscape.png
├── docs/
│   ├── wiki/               # This wiki
│   ├── architecture.md
│   ├── experimental_data.md
│   └── oracle_schema.md
├── notebooks/              # 6 Jupyter notebooks
├── src/sdgft_ml/           # Python package
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

## Troubleshooting

### "No module named 'torch_geometric'"

PyTorch Geometric requires specific PyTorch/CUDA combinations:

```bash
# For PyTorch 2.x + CUDA 12.1:
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# For CPU only:
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### "FileNotFoundError: Checkpoint not found"

The GNN checkpoints should be at `checkpoints/ensemble/member_*/best_model.pt`. Verify:

```bash
ls -la checkpoints/ensemble/member_0/
# Should show: best_model.pt (~3.3 MB) and norms.npz (~4 KB)
```

### Memory Issues with Parquet Files

The full oracle_db.parquet (3.2 GB) loads entirely into RAM. For memory-constrained systems:

```python
# Load only needed columns
import pandas as pd
cols = ["delta", "delta_g", "higgs_mass", "chi2_per_dof"]
db = pd.read_parquet("data/oracle_db.parquet", columns=cols)

# Or use DuckDB for out-of-core queries
import duckdb
conn = duckdb.connect()
result = conn.sql("SELECT * FROM 'data/oracle_db.parquet' WHERE chi2_per_dof < 1.0 LIMIT 1000").df()
```

---

**Previous:** [← The Oracle Database](08_oracle_database.md) | **Next:** [API Reference →](10_api_reference.md)
