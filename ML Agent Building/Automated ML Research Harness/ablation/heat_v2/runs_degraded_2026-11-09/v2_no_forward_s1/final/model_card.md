# Model card — heat_v2_no_forward_s1 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['torch_mlp/sqrt+logy+aug1', 'torch_mlp/raw+logy+aug1'] (weights [2.0, 1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.023096
- Test: {"rel_l2": 0.021810296918971496, "r2": 0.995722844725607, "rmse": 0.028698075903889644}; bootstrap: {"mean": 0.02183330806920436, "std": 0.0006223356927227014, "ci95": [0.020671910286943814, 0.022949806374573393], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.07728263072779142, "r2": 0.9589972528245243, "rmse": 0.09624743236358413}, "noise_x3": {"rel_l2": 0.05769699694049492, "r2": 0.9709348547590331, "rmse": 0.07687469944030105}}
- Latency: 0.00908 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s1")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.