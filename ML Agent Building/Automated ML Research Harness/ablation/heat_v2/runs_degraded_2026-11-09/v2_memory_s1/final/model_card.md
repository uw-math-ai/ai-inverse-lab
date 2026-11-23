# Model card — heat_v2_memory_s1 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['physics-MMSE[log,ms=1,it=30]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.019056
- Test: {"rel_l2": 0.018391908240856132, "r2": 0.9969709656692846, "rmse": 0.02420014641131959}; bootstrap: {"mean": 0.018453100513727408, "std": 0.0005222732078840981, "ci95": [0.017413866980179375, 0.019592755621551095], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.11550563495713094, "r2": 0.9081974287672315, "rmse": 0.14385018578503825}, "noise_x3": {"rel_l2": 0.05118918644197349, "r2": 0.9771859484622198, "rmse": 0.06820378062966982}}
- Latency: 8.5 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_memory_s1")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.