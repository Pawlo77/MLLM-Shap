"""Cache for explainer computations."""

from typing import TYPE_CHECKING, Any, cast

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from .model_response import ModelResponse

if TYPE_CHECKING:
    from .chat import BaseMllmChat


# pylint: disable=too-many-instance-attributes
class ExplainerCache(BaseModel):
    """
    Cache for explainer computations associated with a chat.
    Saves and validates calculated SHAP values, masks, and reduced embeddings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calculated_by: int
    """Hash of the explainer that calculated the SHAP values."""

    chat: "BaseMllmChat"
    """The chat instance the cache is for."""

    n: int
    """Index of last token used for SHAP calculations."""

    responses: list[ModelResponse]
    """The model responses used for SHAP calculations."""

    masks: Tensor
    """The masks used for SHAP calculations."""

    shap_values_mask: Tensor
    """The mask indicating which SHAP values are relevant."""

    had_different_masks: bool = False
    """Whether the masks used for SHAP calculations differed from chat's masks."""

    _values: Tensor | None = None
    """The SHAP values calculated."""

    _normalized_values: Tensor | None = None
    """The normalized SHAP values calculated."""

    def __init__(
        self,
        chat: "BaseMllmChat",
        responses: list[ModelResponse],
        masks: Tensor,
        shap_values_mask: Tensor,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ExplainerCache instance.

        Args:
            data: The data to initialize the instance with.
        """
        super().__init__(chat=chat, masks=masks, responses=responses, shap_values_mask=shap_values_mask, **kwargs)

        if masks.shape[0] != len(responses):
            raise ValueError("Masks size does not match the number of responses in the chat.")

        self.shap_values_mask = shap_values_mask

        if chat is not None:
            if chat.input_tokens_num < masks.shape[1]:
                raise ValueError("Masks size is larger than the number of tokens in the chat.")

            # Extend masks to match chat length
            masks = torch.cat(
                [
                    masks,
                    torch.full(
                        (masks.shape[0], chat.input_tokens_num - masks.shape[1]),
                        False,
                        dtype=masks.dtype,
                        device=chat.torch_device,
                    ),
                ],
                dim=1,
            )

            if masks.shape[1] != chat.input_tokens_num:
                raise ValueError("Masks size does not match the number of tokens in the chat.")
            self.masks = masks

            if torch.any(chat.shap_values_mask != shap_values_mask):
                self.had_different_masks = True
        else:
            self.had_different_masks = False

    # pylint: disable=too-many-positional-arguments,too-many-arguments
    @classmethod
    def create(
        cls,
        chat: "BaseMllmChat",
        explainer_hash: int,
        responses: list[ModelResponse],
        masks: Tensor,
        normalized_values: Tensor,
        shap_values_mask: Tensor,
        values: Tensor | None = None,
    ) -> "ExplainerCache":
        """
        Create a new ExplainerCache instance.

        Args:
            chat: The chat instance the cache is for.
            explainer_hash: Hash of the explainer that calculated the SHAP values.
            responses: The model responses used for SHAP calculations.
            masks: The masks used for SHAP calculations.
            values: The SHAP values calculated.
            normalized_values: The normalized SHAP values calculated.
            shap_values_mask: The mask indicating which SHAP values are relevant.
        Returns:
            A new ExplainerCache instance.
        """
        instance = cls(
            calculated_by=explainer_hash,
            chat=chat,
            n=masks.shape[1],
            responses=responses,
            masks=masks,
            shap_values_mask=shap_values_mask,
        )
        instance.normalized_values = normalized_values
        if values is not None:
            instance.values = values
        return instance

    @property
    def normalized_values(self) -> Tensor:
        """
        Normalized SHAP values.

        Raises:
            ValueError: If SHAP values are no longer valid or have not been computed yet.
        """

        self.__validate_values_getter("_normalized_values")
        return cast(Tensor, self._normalized_values)

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
    def values(self) -> Tensor | None:
        """
        SHAP values. Can be none if :class:`HierarchicalExplainer` is used.

        Raises:
            ValueError: If SHAP values are no longer valid or have not been computed yet.
        """
        self.__validate_values_getter("_values")
        return cast(Tensor, self._values)

    @values.setter
    def values(self, values: Tensor | None) -> None:
        """
        Set the SHAP values.

        Args:
            values: The SHAP values to set.
        Raises:
            ValueError: If SHAP values are not valid.
        """
        if values is None:
            self._values = None
            return
        self.__values_setter("_values", values)

    def extend_masks(self) -> None:
        """Extend masks to match the chat length."""
        self.masks = ExplainerCache.extend_values(
            values=self.masks,
            shape=(
                self.masks.shape[0],
                self.chat.input_tokens_num - self.masks.shape[1],
            ),
            dim=1,
            fill_value=False,
            device=self.chat.torch_device,
        )

    def __values_setter(self, name: str, values: Tensor) -> None:
        """
        Set SHAP values.

        Args:
            name: The name of the SHAP values attribute to set.
            values: The SHAP values to set.
        Raises:
            ValueError: If SHAP values size is larger than the number of tokens in the chat
                or if they contain NaN values for user text tokens,
                or if they contain non-NaN values for non-user text tokens.
        """
        if self.chat.input_tokens_num < values.shape[0]:
            raise ValueError("Values size is larger than the number of tokens in the chat.")

        values = ExplainerCache.extend_values(
            values,
            shape=torch.Size((self.chat.input_tokens_num - values.shape[0],)),
            dim=0,
            fill_value=float("nan"),
            device=self.chat.torch_device,
        )

        if values.shape[0] != self.chat.input_tokens_num:
            raise ValueError("SHAP values size does not match the number of tokens in the chat.")

        mask = self.shap_values_mask.clone()
        # only validate up to n
        mask[self.n :] = False  # noqa: E203

        if values[mask].isnan().any():
            raise ValueError("SHAP values contain NaN values for text tokens they should explain.")
        if not values[~mask].isnan().all():
            raise ValueError("SHAP values contain non-NaN values for text tokens they should not explain.")

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

    @staticmethod
    def extend_values(
        values: Tensor,
        shape: tuple[int, ...],
        dim: int,
        fill_value: Any,
        device: torch.device,
    ) -> Tensor:
        """
        Extend SHAP values to match the chat length.

        Args:
            values: The SHAP values to extend.
            shape: The target shape for extension.
            dim: The dimension along which to extend.
            fill_value: The value to use for extension.
            device: The device to create the extended tensor on.
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
                    device=device,
                ),
            ],
            dim=dim,
        )

    def __del__(self) -> None:
        """
        Cleanup on deletion.

        Disconnect the chat to avoid circular references.
        """
        # needs explicit None
        self.chat = None  # type: ignore[assignment]

        # clear other references
        self.calculated_by = None  # type: ignore[assignment]
        self.n = None  # type: ignore[assignment]
        self.responses = None  # type: ignore[assignment]
        self.masks = None  # type: ignore[assignment]
        self._values = None
        self._normalized_values = None
