# Model card — heat_llm_refine_s1

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['idea-024'] (weights [1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.026501
- Test: {"rel_l2": 0.025784743849971415, "r2": 0.9940511622891111, "rmse": 0.033927669069244666}

## Members
- idea-024: `{"method": "kernel_ridge", "params": {"alpha": 0.0001, "gamma_mult": 0.08}, "features": "log", "scaler": "standard", "pca": 0}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_refine_s1")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```