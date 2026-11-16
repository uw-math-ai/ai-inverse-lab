# Model card — heat_v2_no_forward_s1 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['ridge/raw+logy'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.079154
- Test: {"rel_l2": 0.06349515282505278, "r2": 0.9639573519040592, "rmse": 0.0835471760000398}; bootstrap: {"mean": 0.0636154640480328, "std": 0.0016785196824505519, "ci95": [0.06073710454738062, 0.06680507397460277], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.09425907812503369, "r2": 0.9397039081312206, "rmse": 0.1173898217627644}, "noise_x3": {"rel_l2": 0.13481124556246632, "r2": 0.8421671734094673, "rmse": 0.17962068276232054}}
- Latency: 0.00319 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s1")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.