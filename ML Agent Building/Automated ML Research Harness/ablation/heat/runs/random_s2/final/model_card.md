# Model card — heat_random_s2

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['idea-010', 'idea-025'] (weights [1.0, 1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.024604
- Test: {"rel_l2": 0.024142615404035676, "r2": 0.9947605647729811, "rmse": 0.031766949893321475}

## Members
- idea-010: `{"method": "kernel_ridge", "params": {"alpha": 8.97887840661626e-05, "gamma_mult": 0.022183576308829955}, "features": "raw", "scaler": "none", "pca": 64}`
- idea-025: `{"method": "torch_mlp", "params": {"hidden": [512, 256, 128], "dropout": 0.10915612691301553, "lr": 0.0020211962125925662, "weight_decay": 9.12620611000693e-06, "epochs": 265, "batch_size": 64, "activation": "gelu"}, "features": "log", "scaler": "robust", "pca": 32}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\random_s2")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```