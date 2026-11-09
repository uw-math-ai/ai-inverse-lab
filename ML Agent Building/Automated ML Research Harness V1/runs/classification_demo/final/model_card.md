# Model card — classification_demo

- Task: classification; data: classification simulator with params {"n_features": 20, "n_informative": 8, "n_classes": 3, "class_sep": 1.0, "flip_y": 0.02}
- Type: single of ['idea-014'] (weights [1.0])
- Trained on train+val (2500 samples)
- Validation estimate (accuracy): 0.88
- Test: {"accuracy": 0.872, "f1_macro": 0.8712482413532676}

## Members
- idea-014: `{"method": "extra_trees", "params": {"n_estimators": 284, "max_depth": null, "max_features": 0.5, "min_samples_leaf": 2}, "features": "raw", "scaler": "standard", "pca": 0}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"runs\classification_demo")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```