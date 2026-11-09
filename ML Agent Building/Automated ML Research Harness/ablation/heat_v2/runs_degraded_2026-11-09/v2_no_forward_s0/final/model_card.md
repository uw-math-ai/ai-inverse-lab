# Model card — heat_v2_no_forward_s0 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['torch_mlp/raw+logy+aug2', 'torch_mlp/log+logy+aug1', 'torch_mlp/raw+logy+aug2', 'torch_mlp/raw+logy+aug1'] (weights [2.0, 1.0, 1.0, 1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.02373
- Test: {"rel_l2": 0.022481336162390383, "r2": 0.9954624121119879, "rmse": 0.029581032023820907}; bootstrap: {"mean": 0.022428385004913672, "std": 0.0006994131501172437, "ci95": [0.021289708020109346, 0.02367556957987291], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.07958486722231536, "r2": 0.9567513214424404, "rmse": 0.0991146270903287}, "noise_x3": {"rel_l2": 0.05911059738393571, "r2": 0.9694545715691755, "rmse": 0.0787581615783784}}
- Latency: 0.0147 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s0")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.