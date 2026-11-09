# Model card — heat_v2_no_literature_s2 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['physics-MMSE[log,ms=4,it=40]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.018998
- Test: {"rel_l2": 0.018404560569991804, "r2": 0.9969665705150171, "rmse": 0.024216794396592113}; bootstrap: {"mean": 0.018347580293457736, "std": 0.0005579425782012675, "ci95": [0.017321170915590704, 0.019529214845567324], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.12165947515994481, "r2": 0.8999766191239105, "rmse": 0.15151415003054683}, "noise_x3": {"rel_l2": 0.05118136345065137, "r2": 0.9771921611076425, "rmse": 0.06819335738169328}}
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