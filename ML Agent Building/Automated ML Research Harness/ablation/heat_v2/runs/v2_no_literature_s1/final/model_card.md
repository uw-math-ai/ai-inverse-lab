# Model card — heat_v2_no_literature_s1 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['hybrid[kernel_ridge/raw+logy+aug1->LM9]', 'physics-MLE[log,ms=1,it=30]', 'hybrid[kernel_ridge/raw+logy+aug1->LM10]'] (weights [1.0, 1.0, 1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.019019
- Test: {"rel_l2": 0.018339493334415383, "r2": 0.9969879422681689, "rmse": 0.024131178667821166}; bootstrap: {"mean": 0.018400537002405924, "std": 0.000535445579984816, "ci95": [0.017286424611054233, 0.019530964906253695], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.08692245438424415, "r2": 0.9481864877882431, "rmse": 0.10825282434667131}, "noise_x3": {"rel_l2": 0.05156004885592256, "r2": 0.9768515721864355, "rmse": 0.06869791270097056}}
- Latency: 33.1 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s1")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.