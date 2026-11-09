# Model card — graph_fpt_weights

- Task: regression; data: Random walk on a fixed {'family': 'grid', 'rows': 3, 'cols': 3} graph with 9 vertices and 12 edges; edge weights ~ U[0.2, 1.0], mu = degree, potential {'mode': 'none'}. Observed set B = [0, 1, 2, 3, 5, 6, 7, 8] (|B|=8). Features: empirical first-passage probabilities r'(t, x, y) for t = 1..9 and (x, y) in B x B from N = 10000 walks per pair, flattened in (t, x, y) order -> 576 features. Target: weights (12 values). Inverse problem: recover graph parameters from boundary first-passage data.
- Type: ensemble of ['idea-031', 'idea-032'] (weights [2.0, 1.0])
- Trained on train+val (5000 samples)
- Validation estimate (rel_l2): 0.012299
- Test: {"rel_l2": 0.011943656404762963, "r2": 0.9988793620402528, "rmse": 0.0076401162583702225}

## Members
- idea-031: `{"method": "kernel_ridge", "params": {"alpha": 0.004, "gamma_mult": 0.09}, "features": "log+fpt_summary", "scaler": "standard", "pca": 0}`
- idea-032: `{"method": "kernel_ridge", "params": {"alpha": 0.003, "gamma_mult": 0.1}, "features": "fpt_early", "scaler": "standard", "pca": 0}`

## Usage
```python
import sys
sys.path.insert(0, r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"runs\graph_fpt_weights_claude")
y_hat = model.predict(X)   # X: raw simulator features, same layout as training
```