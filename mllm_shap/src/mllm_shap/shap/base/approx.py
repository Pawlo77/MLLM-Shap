"""Base class for SHAP explainers using approximation methods."""

from abc import ABC

from typing import Any


from .explainer import BaseShapExplainer


# pylint: disable=too-few-public-methods
class BaseShapApproximation(BaseShapExplainer, ABC):
    """
    Base class for SHAP explainers using approximation methods.
    """

    num_samples: int | None
    """
    Number of random masks to generate. If None, uses fraction.
    -1 stands for minimal number of samples (only single-feature masks and empty mask).
    """

    fraction: float | None
    """Fraction of total possible masks to generate if num_samples is None."""

    def __init__(self, *args: Any, num_samples: int | None = None, fraction: float = 0.6, **kwargs: Any) -> None:
        """
        Args:
            num_samples: Number of random masks to generate. If None, uses fraction.
            fraction: Fraction of total possible masks to generate if num_samples is None.
        Raises:
            ValueError: If both num_samples and fraction are None.
        """
        super().__init__(*args, **kwargs)

        if num_samples is None and fraction is None:
            raise ValueError("Either num_samples or fraction must be provided.")
        if fraction is not None and (not isinstance(fraction, float) or not 0 < fraction <= 1):
            raise ValueError("fraction must be a float in the range (0, 1].")
        if num_samples is not None and (not isinstance(num_samples, int) or (num_samples <= 0 and num_samples != -1)):
            raise ValueError("num_samples must be a positive integer.")

        self.num_samples = num_samples
        self.fraction = fraction
