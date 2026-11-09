# Model card — heat_v2_full_s1 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['custom-solver(fwd)'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.018944
- Test: {"rel_l2": 0.018301032233173253, "r2": 0.9970005033359678, "rmse": 0.024080571397003432}; bootstrap: {"mean": 0.018358803269374894, "std": 0.0005106671289319484, "ci95": [0.01731084274303978, 0.019445403834472824], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.12274967786941679, "r2": 0.8977478558025946, "rmse": 0.1528718834637174}, "noise_x3": {"rel_l2": 0.051360931742778435, "r2": 0.9770325101386395, "rmse": 0.06843261174878819}}
- Latency: 22.6 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_full_s1")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.