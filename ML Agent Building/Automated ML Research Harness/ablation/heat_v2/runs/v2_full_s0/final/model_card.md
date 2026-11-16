# Model card — heat_v2_full_s0 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['hybrid[ridge/log+logy->LM20]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.019325
- Test: {"rel_l2": 0.018411824973412937, "r2": 0.9969641878964159, "rmse": 0.0242263529276635}; bootstrap: {"mean": 0.018358835436976473, "std": 0.0005723635841994635, "ci95": [0.017279670581836108, 0.019538156016157195], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.09361248373043199, "r2": 0.9416160409632601, "rmse": 0.11658455608178235}, "noise_x3": {"rel_l2": 0.05203098724910365, "r2": 0.9764213410237892, "rmse": 0.06932538465532602}}
- Latency: 3.48 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_full_s0")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.