# Model card — heat_v2_no_critic_s2 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['physics-MMSE[log,ms=3,it=25]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.018999
- Test: {"rel_l2": 0.018405391274667574, "r2": 0.9969662832930887, "rmse": 0.024217887441126424}; bootstrap: {"mean": 0.018348627744774863, "std": 0.0005580304386891936, "ci95": [0.01732436179372998, 0.01953172188473244], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.11549716213168877, "r2": 0.9096030991284598, "rmse": 0.14383963376725634}, "noise_x3": {"rel_l2": 0.051226932972407185, "r2": 0.9771480954254794, "rmse": 0.06825407359699699}}
- Latency: 36.1 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_critic_s2")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.