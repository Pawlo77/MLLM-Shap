"""Internal helpers for runner logic (linear scaling, in-place updates)."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _LinearSampleScaler:
    """Scale mask count quadratically and force an even result."""

    factor: float
    """Quadratic scaling coefficient applied as factor * n^2."""

    def scale(self, n_pre: int) -> int:
        """Scale as factor * n_pre^2 and round odd outputs up to even."""
        scaled = int(self.factor * n_pre * n_pre)
        return scaled if scaled % 2 == 0 else scaled + 1


def _try_set_num_samples(explainer: Any, num_samples: int) -> bool:
    """Try to update num_samples on approximation explainers in place."""
    shap_explainer = getattr(explainer, "shap_explainer", None)
    if shap_explainer is not None and hasattr(shap_explainer, "num_samples"):
        shap_explainer.num_samples = int(num_samples)
        return True
    return False
