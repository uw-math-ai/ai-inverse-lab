# Model card — heat_v2_no_forward_s0 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['custom-solver', 'torch_mlp/raw', 'custom-solver', 'torch_mlp/raw'] (weights [5.0, 1.0, 1.0, 1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.022098
- Test: {"rel_l2": 0.02103913426567809, "r2": 0.9960355067605914, "rmse": 0.027683377000858725}; bootstrap: {"mean": 0.020977394973850053, "std": 0.0007353010805166448, "ci95": [0.01962891549872201, 0.02241209308421346], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.09131566848779842, "r2": 0.9440617581766898, "rmse": 0.11372411295717313}, "noise_x3": {"rel_l2": 0.0540439209594569, "r2": 0.974552357456252, "rmse": 0.07200739034335601}}
- Latency: 36.5 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s0")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.