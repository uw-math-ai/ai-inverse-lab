# Model card — heat_full_s0

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['idea-021', 'idea-018', 'idea-017'] (weights [1.0, 1.0, 1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.024889
- Test: {"rel_l2": 0.023161062842123627, "r2": 0.9951803044190694, "rmse": 0.030475419107195072}

## Members
- idea-021: `{"method": "torch_mlp", "params": {"hidden": [512, 256, 128], "dropout": 0.0, "lr": 0.001, "weight_decay": 0.0001, "epochs": 200, "batch_size": 128, "activation": "silu"}, "features": "sqrt", "scaler": "standard", "pca": 32}`
- idea-018: `{"method": "kernel_ridge", "params": {"alpha": 0.0003, "gamma_mult": 0.3}, "features": "sqrt", "scaler": "standard", "pca": 0}`
- idea-017: `{"method": "torch_mlp", "params": {"hidden": [512, 256, 128], "dropout": 0.0, "lr": 0.001, "weight_decay": 0.0001, "epochs": 150, "batch_size": 128, "activation": "silu"}, "features": "sqrt", "scaler": "standard", "pca": 0}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\full_s0")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```