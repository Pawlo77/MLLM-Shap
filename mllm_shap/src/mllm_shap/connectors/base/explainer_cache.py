"""Cache for explainer computations."""

from typing import TYPE_CHECKING, Any, cast

import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr
from torch import Tensor

if TYPE_CHECKING:
    from .chat import BaseMllmChat


class ExplainerCache(BaseModel):
    """
    Cache for explainer computations associated with a chat.
    Saves and validates calculated SHAP values, masks, and reduced embeddings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chat: "BaseMllmChat"
    """The chat instance the cache is for."""
    calculated_by: int
    """Hash of the explainer that calculated the SHAP values."""
    n: int | None = None
    """Index of last token used for SHAP calculation."""
    reduced_embeddings: Tensor | None = None
    """The reduced embeddings used during SHAP calculation."""

    _values: Tensor | None = PrivateAttr(default=None)
    _normalized_values: Tensor | None = PrivateAttr(default=None)
    _masks: Tensor | None = PrivateAttr(default=None)

    @property
    def normalized_values(self) -> Tensor:
        """
        Normalized SHAP values.

        Raises:
            ValueError: If SHAP values are no longer valid or have not been computed yet.
        """
        if self._normalized_values is None:
            raise ValueError("Normalized SHAP values have not been computed yet.")
        self.__validate_values_getter("_normalized_values")
        return self._normalized_values

    @normalized_values.setter
    def normalized_values(self, values: Tensor) -> None:
        """
        Set the normalized SHAP values.

        Args:
            values: The normalized SHAP values to set.
        Raises:
            ValueError: If normalized SHAP values are not valid.
        """
        self.__values_setter("_normalized_values", values)

    @property
    def values(self) -> Tensor:
        """
        SHAP values.

        Raises:
            ValueError: If SHAP values are no longer valid or have not been computed yet.
        """
        if self._values is None:
            raise ValueError("SHAP values have not been computed yet.")
        self.__validate_values_getter("_values")
        return self._values

    @values.setter
    def values(self, values: Tensor) -> None:
        """
        Set the SHAP values.

        Args:
            values: The SHAP values to set.
        Raises:
            ValueError: If SHAP values are not valid.
        """
        self.__values_setter("_values", values)

    @property
    def masks(self) -> Tensor:
        """
        Generated masks.

        Raises:
            ValueError: If masks have not been generated yet.
        """
        if self._masks is None:
            raise ValueError("Masks have not been set yet.")
        return self._masks

    @masks.setter
    def masks(self, values: Tensor) -> None:
        """
        Set the generated masks.

        Args:
            values: The masks to set.
        Raises:
            ValueError: If masks size does not match the number of reduced_embeddings
                or the number
        """
        # force to save reduced_embeddings first
        if self.reduced_embeddings is None or values.shape[0] != self.reduced_embeddings.shape[0]:
            raise ValueError("Masks size does not match the number of reduced_embeddings in the chat.")

        values = self.extend_values(
            values, shape=(values.shape[0], self.chat.input_tokens_num - values.shape[1]), dim=1, fill_value=False
        )

        if values.shape[1] != self.chat.input_tokens_num:
            raise ValueError("Masks size does not match the number of tokens in the chat.")
        self._masks = values

    def extend_values(self, values: Tensor, shape: tuple[int, ...], dim: int, fill_value: Any) -> Tensor:
        """
        Extend SHAP values to match the chat length.

        Args:
            values: The SHAP values to extend.
            shape: The target shape for extension.
            dim: The dimension along which to extend.
            fill_value: The value to use for extension.
        Returns:
            The extended SHAP values.
        """
        return torch.cat(
            [
                values,
                torch.full(
                    shape,
                    fill_value,
                    dtype=values.dtype,
                    device=self.chat.torch_device,
                ),
            ],
            dim=dim,
        )

    def __values_setter(self, name: str, values: Tensor) -> None:
        """
        Set SHAP values.

        Args:
            name: The name of the SHAP values attribute to set.
            values: The SHAP values to set.
        Raises:
            ValueError: If SHAP values size is larger than the number of tokens in the chat
        """
        if self.chat.input_tokens_num < values.shape[0]:
            raise ValueError("Values size is larger than the number of tokens in the chat.")

        values = self.extend_values(
            values,
            shape=torch.Size((self.chat.input_tokens_num - values.shape[0],)),
            dim=0,
            fill_value=float("nan"),
        )
        self.__validate_values_setter(values)
        setattr(self, name, values)

    def __validate_values_getter(self, name: str) -> None:
        """
        Validate SHAP values when getting them.

        Args:
            values: The SHAP values to validate.
        Raises:
            ValueError: If SHAP values size does not match the number of tokens in the chat.
        """
        if getattr(self, name) is None:
            raise ValueError("SHAP values have not been computed yet.")
        if cast(Tensor, getattr(self, name)).shape[0] != self.chat.input_tokens_num:
            raise ValueError(
                "SHAP values size does not match the number of tokens in the chat. Recalculate SHAP values to update."
            )

    def __validate_values_setter(self, values: Tensor) -> None:
        """
        Validate SHAP values before setting them.

        Args:
            values: The SHAP values to validate.
        Raises:
            ValueError: If SHAP values size does not match the number of tokens in the chat,
                or if they contain NaN values for user text tokens,
                or if they contain non-NaN values for non-user text tokens.
        """
        if values.shape[0] != self.chat.input_tokens_num:
            raise ValueError("SHAP values size does not match the number of tokens in the chat.")

        mask = self.chat.shap_values_mask.clone()
        # only validate up to n
        mask[self.n :] = False  # noqa: E203

        if values[mask].isnan().any():
            raise ValueError("SHAP values contain NaN values for text tokens they should explain.")
        if not values[~mask].isnan().all():
            raise ValueError("SHAP values contain non-NaN values for text tokens they should not explain.")

    def __del__(self) -> None:
        """
        Cleanup on deletion.

        Disconnect the chat to avoid circular references.
        """
        self.chat = None  # type: ignore[assignment]
        self.reduced_embeddings = None
        self._masks = None
        self._values = None
        self._normalized_values = None
