# Model card — heat_llm_refine_s0

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['idea-011'] (weights [1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.027731
- Test: {"rel_l2": 0.026989630280724562, "r2": 0.9934141987228233, "rmse": 0.03551306345308925}

## Members
- idea-011: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.1, "lr": 0.001, "weight_decay": 0.001, "epochs": 250, "batch_size": 64, "activation": "tanh"}, "features": "log", "scaler": "standard", "pca": 30}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_refine_s0")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```