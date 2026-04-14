"""Weighted ensemble regressor for combining multiple tree models."""

from __future__ import annotations

from typing import Any

import numpy as np


class WeightedEnsemble:
    """Combines multiple regressors with fixed weights."""

    def __init__(self, models: dict[str, Any], weights: tuple[float, ...]) -> None:
        self.models = models
        self.weights = weights
        self.names = list(models.keys())

    def predict(self, X: np.ndarray) -> np.ndarray:
        return sum(w * self.models[n].predict(X) for w, n in zip(self.weights, self.names))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"weights": self.weights, "models": self.names}
