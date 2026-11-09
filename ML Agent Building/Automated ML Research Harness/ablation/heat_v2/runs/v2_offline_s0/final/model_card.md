# Model card — heat_v2_offline_s0 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['physics-MMSE[log,ms=1,it=30]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.019047
- Test: {"rel_l2": 0.018334245073669358, "r2": 0.9969895615121332, "rmse": 0.02412427298534411}; bootstrap: {"mean": 0.018289602146463742, "std": 0.0005738691247762629, "ci95": [0.017215213470461656, 0.019422793014215966], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.11230532290441524, "r2": 0.9135417580957382, "rmse": 0.13986453189443707}, "noise_x3": {"rel_l2": 0.051202152102978994, "r2": 0.9771742182490442, "rmse": 0.06822105590127313}}
- Latency: 9.03 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_offline_s0")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.