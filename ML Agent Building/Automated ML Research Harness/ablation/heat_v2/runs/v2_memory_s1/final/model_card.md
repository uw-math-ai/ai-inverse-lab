# Model card — heat_v2_memory_s1 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['physics-MMSE[log,ms=3,it=25]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.018966
- Test: {"rel_l2": 0.018271809945783972, "r2": 0.9970099322515243, "rmse": 0.024042120594398652}; bootstrap: {"mean": 0.018335301130214622, "std": 0.0005126757787993747, "ci95": [0.017287123229063592, 0.019462837180168974], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.11125213605473684, "r2": 0.9159508189924297, "rmse": 0.13855289784256744}, "noise_x3": {"rel_l2": 0.05115883620497609, "r2": 0.9772120879683803, "rmse": 0.06816334238381928}}
- Latency: 73.4 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_memory_s1")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.