# Model card — heat_offline_s0

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['idea-030', 'idea-018', 'idea-021'] (weights [2.0, 1.0, 1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.025099
- Test: {"rel_l2": 0.023445571386356633, "r2": 0.9950461314781531, "rmse": 0.03084977658742735}

## Members
- idea-030: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.08954734302423699, "lr": 0.0015948143970774696, "weight_decay": 2.370688972464005e-06, "epochs": 157, "batch_size": 128, "activation": "silu"}, "features": "sqrt", "scaler": "standard", "pca": 0}`
- idea-018: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.08954734302423699, "lr": 0.0015948143970774696, "weight_decay": 2.331652194583013e-06, "epochs": 157, "batch_size": 128, "activation": "tanh"}, "features": "sqrt", "scaler": "standard", "pca": 0}`
- idea-021: `{"method": "torch_mlp", "params": {"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weight_decay": 0.00012028897186395869, "epochs": 175, "batch_size": 128, "activation": "gelu"}, "features": "raw", "scaler": "standard", "pca": 0}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\offline_s0")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```