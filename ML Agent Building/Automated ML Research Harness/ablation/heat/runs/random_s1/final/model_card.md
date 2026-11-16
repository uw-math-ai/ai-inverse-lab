# Model card — heat_random_s1

- Task: regression; data: 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.
- Type: ensemble of ['idea-016', 'idea-013', 'idea-023', 'idea-011'] (weights [3.0, 2.0, 2.0, 1.0])
- Trained on train+val (3750 samples)
- Validation estimate (rel_l2): 0.024059
- Test: {"rel_l2": 0.0227921198126852, "r2": 0.9953380034456011, "rmse": 0.029989962393681693}

## Members
- idea-016: `{"method": "kernel_ridge", "params": {"alpha": 0.0004422906167031346, "gamma_mult": 0.19546364342321004}, "features": "log", "scaler": "standard", "pca": 0}`
- idea-013: `{"method": "torch_mlp", "params": {"hidden": [512, 256, 128], "dropout": 0.29652243702248987, "lr": 0.002079447741866517, "weight_decay": 0.00011983043170155368, "epochs": 177, "batch_size": 64, "activation": "relu"}, "features": "raw", "scaler": "standard", "pca": 0}`
- idea-023: `{"method": "torch_mlp", "params": {"hidden": [512, 512], "dropout": 0.04579938791735105, "lr": 0.0009345938431869064, "weight_decay": 1.0111965972213143e-05, "epochs": 167, "batch_size": 256, "activation": "tanh"}, "features": "raw", "scaler": "standard", "pca": 0}`
- idea-011: `{"method": "kernel_ridge", "params": {"alpha": 2.102783947671146e-05, "gamma_mult": 1.8649835924329932}, "features": "raw", "scaler": "none", "pca": 64}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\random_s1")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```