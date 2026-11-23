# Model card — quick_friedman

- Task: regression; data: friedman simulator with params {"n_features": 10, "noise": 1.0}
- Type: single of ['idea-004'] (weights [1.0])
- Trained on train+val (2000 samples)
- Validation estimate (rmse): 1.7704
- Test: {"rmse": 1.7338516639722603, "r2": 0.8792252907663063}

## Members
- idea-004: `{"method": "extra_trees", "params": {"n_estimators": 300, "max_depth": null, "max_features": 0.5, "min_samples_leaf": 1}, "features": "raw", "scaler": "standard", "pca": 0}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"runs\quick_friedman")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```