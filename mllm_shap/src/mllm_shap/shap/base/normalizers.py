"""Base class for SHAP value normalizers."""

from abc import ABC, abstractmethod

from torch import Tensor


# pylint: disable=too-few-public-methods
class BaseNormalizer(ABC):
    """Base class for SHAP value normalizers."""

    @abstractmethod
    def __call__(self, shap_values: Tensor) -> Tensor:
        """
        Normalize the SHAP values.

        Args:
            shap_values: The input SHAP values to be normalized.
        Returns:
            The normalized SHAP values.
        """
