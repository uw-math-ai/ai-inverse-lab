# Model card — heat_offline_s2

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['idea-027', 'idea-024', 'idea-029', 'idea-025'] (weights [3.0, 1.0, 1.0, 1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.025795
- Test: {"rel_l2": 0.023377062447665832, "r2": 0.9950807585911889, "rmse": 0.030759632251935367}

## Members
- idea-027: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.0, "lr": 0.0013598589398165094, "weight_decay": 0.0001, "epochs": 150, "batch_size": 64, "activation": "silu"}, "features": "log", "scaler": "standard", "pca": 0}`
- idea-024: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.006462707030864809, "lr": 0.0008725358113262284, "weight_decay": 0.0001, "epochs": 150, "batch_size": 128, "activation": "silu"}, "features": "raw", "scaler": "standard", "pca": 0}`
- idea-029: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.019379905337065118, "lr": 0.001, "weight_decay": 0.0001, "epochs": 156, "batch_size": 128, "activation": "gelu"}, "features": "raw", "scaler": "standard", "pca": 0}`
- idea-025: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.0, "lr": 0.0008725358113262284, "weight_decay": 0.0001, "epochs": 150, "batch_size": 128, "activation": "silu"}, "features": "sqrt", "scaler": "standard", "pca": 0}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\offline_s2")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```