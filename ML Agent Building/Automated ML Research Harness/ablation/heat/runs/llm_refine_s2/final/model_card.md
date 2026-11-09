# Model card — heat_llm_refine_s2

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['idea-013'] (weights [1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.026217
- Test: {"rel_l2": 0.023995853932514414, "r2": 0.9948254672673267, "rmse": 0.03157384056220455}

## Members
- idea-013: `{"method": "custom", "params": {"n_models": 3, "hidden": 256, "epochs": 150, "lr": 0.002, "wd": 0.0001, "noise": 0.05}, "features": "raw", "scaler": "standard", "pca": 0, "code_path": "C:\\Users\\kusha\\Research Code\\UW Math AI Lab\\ai-inverse-lab\\ML Agent Building\\Automated ML Research Harness V1\\ablation\\heat\\runs\\llm_refine_s2\\workspace\\ideas\\idea-013\\model_v1.py"}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_refine_s2")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```