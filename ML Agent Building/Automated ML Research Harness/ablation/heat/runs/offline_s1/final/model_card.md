# Model card — heat_offline_s1

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['idea-024', 'idea-026', 'idea-025'] (weights [2.0, 2.0, 1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.025946
- Test: {"rel_l2": 0.02437556163434319, "r2": 0.994643931440994, "rmse": 0.032073461474696356}

## Members
- idea-024: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.09927449747328324, "lr": 0.0006836807757312639, "weight_decay": 3.613829647016976e-05, "epochs": 318, "batch_size": 256, "activation": "silu"}, "features": "log", "scaler": "standard", "pca": 32}`
- idea-026: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weight_decay": 0.0001, "epochs": 147, "batch_size": 128, "activation": "silu"}, "features": "sqrt", "scaler": "standard", "pca": 32}`
- idea-025: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.10715680130128409, "lr": 0.0005437287693891764, "weight_decay": 3.613829647016976e-05, "epochs": 318, "batch_size": 256, "activation": "relu"}, "features": "log", "scaler": "standard", "pca": 32}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\offline_s1")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```