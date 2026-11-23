# Model card — heat_v2_no_literature_s0 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['physics-MMSE[log,ms=4,it=40]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.018951
- Test: {"rel_l2": 0.01825284197540761, "r2": 0.9970160967001367, "rmse": 0.024017162463125803}; bootstrap: {"mean": 0.01821839102698879, "std": 0.0005465343476124356, "ci95": [0.017214736057109613, 0.019396969224526953], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.11964958655633214, "r2": 0.9034176097056934, "rmse": 0.1490110440206606}, "noise_x3": {"rel_l2": 0.051123674511120376, "r2": 0.9772427032579035, "rmse": 0.06811649341783657}}
- Latency: 57.9 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s0")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.