# Model card — heat_v2_no_literature_s2 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['hybrid[ridge/raw+pca40+logy+aug1->LM15]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.018992
- Test: {"rel_l2": 0.018326342299300843, "r2": 0.9969921783674607, "rmse": 0.024113874483227354}; bootstrap: {"mean": 0.01826522391946012, "std": 0.0005530328703519321, "ci95": [0.017245714686597864, 0.019441432223436673], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.09720298036108582, "r2": 0.9351916938381711, "rmse": 0.12105614404865342}, "noise_x3": {"rel_l2": 0.05120563786214321, "r2": 0.9771619271562322, "rmse": 0.0682257002797032}}
- Latency: 34.4 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s2")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.