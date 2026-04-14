"""SHAP-based model explainability for EC price predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import shap
import structlog
from src.model.train import FEATURE_COLUMNS

logger = structlog.get_logger()


class ModelExplainer:
    """Wraps a trained pipeline to produce SHAP explanations for predictions."""

    def __init__(self, model: Any) -> None:
        self.model = model
        # Extract regressor and scaler from either Pipeline or dict format
        if isinstance(model, dict):
            regressor = model["model"]
            self._scaler = model["scaler"]
        else:
            regressor = model.named_steps["regressor"]
            self._scaler = model.named_steps["scaler"]
        try:
            self._explainer = shap.TreeExplainer(regressor)
            self._is_tree = True
        except Exception:
            self._explainer = None
            self._is_tree = False
        logger.info("explainer_initialised", tree_explainer=self._is_tree)

    def explain(self, features: np.ndarray) -> dict[str, Any]:
        """Generate SHAP values for a single prediction.

        Args:
            features: Raw feature array (1, n_features) — before scaling.

        Returns:
            Dict with feature contributions and base value.
        """
        if self._explainer is None:
            return {"error": "Explainer not available for this model type"}

        scaled = self._scaler.transform(features)

        shap_values = self._explainer.shap_values(scaled)

        # shap_values shape: (1, n_features) for a single sample
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        contributions: dict[str, float] = {}
        for i, col in enumerate(FEATURE_COLUMNS):
            contributions[col] = round(float(shap_values[0][i]), 2)

        base_value = float(self._explainer.expected_value)
        if isinstance(self._explainer.expected_value, np.ndarray):
            base_value = float(self._explainer.expected_value[0])

        # Sort by absolute contribution
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "base_value": round(base_value, 2),
            "feature_contributions": sorted_contributions,
            "top_driver": max(contributions, key=lambda k: abs(contributions[k])),
        }
