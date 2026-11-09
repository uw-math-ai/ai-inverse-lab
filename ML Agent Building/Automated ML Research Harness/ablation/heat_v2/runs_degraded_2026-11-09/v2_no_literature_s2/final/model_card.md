# Model card — heat_v2_no_literature_s2 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['physics-MMSE[log,ms=1,it=30]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.018997
- Test: {"rel_l2": 0.01840212060099675, "r2": 0.996967376315528, "rmse": 0.02421358387563115}; bootstrap: {"mean": 0.018344959189627464, "std": 0.0005583028229042422, "ci95": [0.017316404220854877, 0.01952870483527669], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.10832844590958716, "r2": 0.9193666893308248, "rmse": 0.13491174760159644}, "noise_x3": {"rel_l2": 0.05129644059318463, "r2": 0.9770838745847514, "rmse": 0.0683466846121177}}
- Latency: 8.58 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s2")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.