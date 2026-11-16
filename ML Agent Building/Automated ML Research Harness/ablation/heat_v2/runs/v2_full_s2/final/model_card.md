# Model card — heat_v2_full_s2 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['hybrid[extra_trees/raw+logy+aug2->LM15]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.019329
- Test: {"rel_l2": 0.018418286260390347, "r2": 0.9969620469419356, "rmse": 0.02423485471490663}; bootstrap: {"mean": 0.018351099537385944, "std": 0.0005878088206469042, "ci95": [0.017246939282427484, 0.0196238926653677], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.0755688526413049, "r2": 0.9610077807071215, "rmse": 0.09411310102791462}, "noise_x3": {"rel_l2": 0.05205167669367503, "r2": 0.9764013339060579, "rmse": 0.06935295099183142}}
- Latency: 5.6 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_full_s2")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.