# Model card — heat_llm_ideation_s1

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: single of ['idea-016'] (weights [1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.02839
- Test: {"rel_l2": 0.02862720192074091, "r2": 0.9925853132643121, "rmse": 0.037667786765561345}

## Members
- idea-016: `{"method": "torch_mlp", "params": {"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001, "weight_decay": 0.0001, "epochs": 200, "batch_size": 128, "activation": "gelu"}, "features": "log", "scaler": "standard", "pca": 0}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_ideation_s1")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```