# Model card — heat_v2_no_critic_s0 (V2)

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['hybrid[torch_mlp/raw+logy+aug1->LM12]'] (weights [1.0])
- Fitted on train+val (3750 samples); physics members query the simulator's forward operator at prediction time (the simulator file is copied next to the model)
- Validation estimate (rel_l2): 0.018947
- Test: {"rel_l2": 0.018250642518496597, "r2": 0.997016861950297, "rmse": 0.024014268408926807}; bootstrap: {"mean": 0.018210350928347475, "std": 0.00056163964690095, "ci95": [0.017132591313902573, 0.019279776960215735], "n_boot": 200}
- Shift sets: {"k_low_0.3": {"rel_l2": 0.09754813916009965, "r2": 0.9349779371749278, "rmse": 0.12148600322722863}, "noise_x3": {"rel_l2": 0.051177902448321655, "r2": 0.977197260539765, "rmse": 0.06818874598893511}}
- Latency: 13.9 ms/sample

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_critic_s0")
theta_hat = model.predict(X)   # raw simulator observations
```

Reproducibility metadata and research lineage: `manifest.json`.