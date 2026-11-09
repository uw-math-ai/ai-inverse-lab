# Model card — heat_full_s1

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['idea-026', 'idea-018', 'idea-020', 'idea-019'] (weights [2.0, 1.0, 1.0, 1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.025653
- Test: {"rel_l2": 0.024599963654579036, "r2": 0.9945409884040435, "rmse": 0.03236873055029124}

## Members
- idea-026: `{"method": "torch_mlp", "params": {"hidden": [512, 256, 128], "dropout": 0.15, "lr": 0.001, "weight_decay": 0.001, "epochs": 300, "batch_size": 128, "activation": "silu"}, "features": "log", "scaler": "standard", "pca": 30}`
- idea-018: `{"method": "torch_mlp", "params": {"hidden": [1024, 512], "dropout": 0.05, "lr": 0.0008, "weight_decay": 0.0003, "epochs": 300, "batch_size": 128, "activation": "silu"}, "features": "log", "scaler": "standard", "pca": 0}`
- idea-020: `{"method": "torch_mlp", "params": {"hidden": [512, 256, 128], "dropout": 0.1, "lr": 0.001, "weight_decay": 0.0003, "epochs": 250, "batch_size": 128, "activation": "silu"}, "features": "log", "scaler": "standard", "pca": 30}`
- idea-019: `{"method": "torch_mlp", "params": {"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001, "weight_decay": 0.0001, "epochs": 200, "batch_size": 128, "activation": "silu"}, "features": "log", "scaler": "standard", "pca": 40}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\full_s1")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```