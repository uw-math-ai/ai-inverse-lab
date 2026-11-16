"""Generic tabular simulators (useful for quick tests and as templates)."""
from __future__ import annotations

import numpy as np

from . import Simulator


class FriedmanSimulator(Simulator):
    name = "friedman"
    task = "regression"

    def __init__(self, n_features: int = 10, noise: float = 1.0):
        super().__init__(n_features=n_features, noise=noise)

    def generate(self, n, seed):
        from sklearn.datasets import make_friedman1
        X, y = make_friedman1(n_samples=n, n_features=self.params["n_features"],
                              noise=self.params["noise"], random_state=seed)
        return X, y


class LinearRegressionSimulator(Simulator):
    name = "regression"
    task = "regression"

    def __init__(self, n_features: int = 50, n_informative: int = 10, n_targets: int = 1,
                 noise: float = 5.0):
        super().__init__(n_features=n_features, n_informative=n_informative, n_targets=n_targets,
                         noise=noise)

    def generate(self, n, seed):
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=n, random_state=seed, **self.params)
        return X, y


class ClassificationSimulator(Simulator):
    name = "classification"
    task = "classification"

    def __init__(self, n_features: int = 20, n_informative: int = 8, n_classes: int = 3,
                 class_sep: float = 1.0, flip_y: float = 0.02):
        super().__init__(n_features=n_features, n_informative=n_informative, n_classes=n_classes,
                         class_sep=class_sep, flip_y=flip_y)

    def generate(self, n, seed):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n, random_state=seed, n_redundant=2, **self.params)
        return X, y.astype(int)
