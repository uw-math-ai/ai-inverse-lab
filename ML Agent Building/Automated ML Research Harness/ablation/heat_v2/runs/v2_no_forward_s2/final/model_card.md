# Model card — heat_v2_no_forward_s2 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['custom-solver'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.021924
- Test: {"rel_l2": 0.01952101117065167, "r2": 0.996586186504634, "rmse": 0.025685824561551055}; bootstrap: {"mean": 0.019449870381729396, "std": 0.0006232747987049667, "ci95": [0.018220667552578517, 0.020523021038429776], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.09906285040507064, "r2": 0.9322965430904729, "rmse": 0.12337241763532766}, "noise_x3": {"rel_l2": 0.05269257336340505, "r2": 0.9758423387223101, "rmse": 0.07020687305832299}}
- Latency: 2.66 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s2")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.