# Model card — heat_v2_no_forward_s1 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['custom-solver'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.019969
- Test: {"rel_l2": 0.018497468982486998, "r2": 0.9969338677337175, "rmse": 0.024339043657287767}; bootstrap: {"mean": 0.018534965403056557, "std": 0.000534813127886063, "ci95": [0.017509473077083646, 0.01961326234541964], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.08759929971743409, "r2": 0.948095993656343, "rmse": 0.10909576440724278}, "noise_x3": {"rel_l2": 0.05275100750288267, "r2": 0.9757932214173554, "rmse": 0.07028472991652354}}
- Latency: 4.66 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s1")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.