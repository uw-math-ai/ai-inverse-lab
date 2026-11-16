# Model card — heat_v2_memory_s2 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['physics-MLE[log,ms=1,it=30]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.019335
- Test: {"rel_l2": 0.01841886806236584, "r2": 0.9969618386959722, "rmse": 0.024235620252272574}; bootstrap: {"mean": 0.018351365102252413, "std": 0.0005877143444078948, "ci95": [0.0172478546722959, 0.019624648573606792], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.12494414929093481, "r2": 0.8926730245849349, "rmse": 0.15560486806487978}, "noise_x3": {"rel_l2": 0.0520294521303912, "r2": 0.97642188876497, "rmse": 0.0693233392838876}}
- Latency: 7.93 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_memory_s2")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.