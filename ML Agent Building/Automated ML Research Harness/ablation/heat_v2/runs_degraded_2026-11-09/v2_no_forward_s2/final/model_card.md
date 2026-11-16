# Model card — heat_v2_no_forward_s2 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['torch_mlp/sqrt+logy+aug1', 'torch_mlp/log+logy', 'torch_mlp/log+logy+aug1'] (weights [1.0, 1.0, 1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.026202
- Test: {"rel_l2": 0.022976914082214635, "r2": 0.9952515163655046, "rmse": 0.03023311543250835}; bootstrap: {"mean": 0.022977905813577025, "std": 0.0008009249208860409, "ci95": [0.021464809553705236, 0.024540498340059926], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.059507006415739604, "r2": 0.9763092692903781, "rmse": 0.0741097517155126}, "noise_x3": {"rel_l2": 0.06225945515385517, "r2": 0.9661645471837936, "rmse": 0.08295365714104118}}
- Latency: 0.012 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s2")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.