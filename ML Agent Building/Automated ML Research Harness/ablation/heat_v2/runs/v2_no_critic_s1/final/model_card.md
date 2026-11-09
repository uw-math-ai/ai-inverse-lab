# Model card — heat_v2_no_critic_s1 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.019049
- Test: {"rel_l2": 0.018305222848774328, "r2": 0.9969991552253469, "rmse": 0.02408608542576931}; bootstrap: {"mean": 0.018370207982770666, "std": 0.0005175430339594094, "ci95": [0.01733166588587445, 0.019459493157681102], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.09566890303964426, "r2": 0.937362474565109, "rmse": 0.11914561121811329}, "noise_x3": {"rel_l2": 0.05110898943340724, "r2": 0.9772512940556659, "rmse": 0.06809692721863525}}
- Latency: 20.5 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_critic_s1")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.