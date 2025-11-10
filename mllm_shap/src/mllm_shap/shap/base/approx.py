"""Base class for SHAP explainers using approximation methods."""

from abc import ABC
from functools import lru_cache
from logging import Logger
from typing import Any, Generator

import torch
from torch import Tensor

from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model_response import ModelResponse
from ...utils.logger import get_logger
from ._masks_manager import MaskGenerator, MasksManager
from .explainer import BaseShapExplainer

logger: Logger = get_logger(__name__)


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

    include_minimal_masks: bool = True
    """Whether to include minimal masks (single-feature and empty masks) in the sampling."""

    _first_call: bool
    """Indicates if it's the first call to generate masks."""
    _zero_mask_skipped: bool
    """Indicates if the zero mask was skipped."""
    _base_masks: Tensor | None
    """Holds the base masks if :attr:`include_minimal_masks` is True."""
    _base_calls_num: int
    """Number of base masks already generated."""

    def __init__(
        self,
        *args: Any,
        num_samples: int | None = None,
        fraction: float = 0.6,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            num_samples: Number of random masks to generate. If None, uses fraction.
            fraction: Fraction of total possible masks to generate if num_samples is None.
        """
        super().__init__(*args, **kwargs)
        BaseShapApproximation._validate_sampling_params(
            num_samples=num_samples,
            fraction=fraction,
        )

        self.num_samples = num_samples
        self.fraction = fraction

    def _get_next_split_base(
        self,
        target_length: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,  # pylint: disable=unused-argument
    ) -> Tensor | None:
        """
        Get the next mask split for SHAP value calculation
        from the base minimal masks, if applicable.

        Args:
            target_length: Length of the masks
            device: Torch device to create the tensor on
            generated_masks_num: Number of masks already generated
        Returns:
            Next mask tensor or None if no more masks can be generated.
        Raises:
            RuntimeError: If there are inconsistencies in mask generation logic.
        """
        if self.include_minimal_masks:
            if generated_masks_num == 0:
                if self._first_call:
                    self._base_masks = self._generate_minimal_splits(
                        target_length=target_length,
                        device=device,
                    )
                    if self._base_masks is None:
                        return None
                    self._first_call = False
                elif (
                    not self._zero_mask_skipped
                ):  # 0 mask was rejected, so start from 1
                    # base masks here cannot be None
                    self._base_masks = self._base_masks[1:]  # type: ignore[index]
                    self._zero_mask_skipped = True
                else:  # another mask was rejected, raise
                    raise RuntimeError("Multiple base masks were rejected.")

            if self._base_masks is None:
                raise RuntimeError("Base masks are not present.")
            num_splits = self._get_num_splits(target_length)
            if num_splits is not None and num_splits < self._base_masks.shape[0]:
                raise RuntimeError(
                    f"Not enough sampling budget, up to {num_splits} "
                    f"calls allowed with required {self._base_masks.shape[0]} for minimal masks."
                )

            if generated_masks_num < self._base_masks.shape[0]:
                if (
                    self._base_calls_num
                    != generated_masks_num + self._zero_mask_skipped
                ):
                    raise RuntimeError("Multiple base masks were rejected.")

                self._base_calls_num += 1
                return self._base_masks[generated_masks_num, ...].squeeze(0)
        return None

    def _generate_minimal_splits(
        self, target_length: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate a minimal set of boolean masks as a batched tensor.
        Shape: (2 * target_length + 1, target_length).

        It ensures that masks are in following order:
        - empty mask
        - single-feature masks and their negations interleaved

        Therefore, _calculate_shap_values can expect masks
        to be interleaved when computing SHAP values.

        Args:
            target_length: Length of the masks
            device: Torch device to create the tensor on
        Returns:
            Batched tensor of minimal masks
        """
        return BaseShapApproximation._generate_minimal_splits(target_length, device)

    def _initialize_state(self) -> None:
        """
        Initialize internal state before starting mask generation.
        """
        self._first_call = True
        self._zero_mask_skipped = False
        self._base_masks = None
        self._base_calls_num = 0

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None:
        self._initialize_state()
        return super().__call__(*args, **kwargs)

    @staticmethod
    def _generate_minimal_splits(
        target_length: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate a minimal set of boolean masks as a batched tensor.
        Shape: (target_length + 1, target_length)
        """
        masks = torch.ones(
            (target_length + 1, target_length), dtype=torch.bool, device=device
        )
        masks[0, :] = False
        masks[torch.arange(1, target_length + 1), torch.arange(target_length)] = False
        return masks

    @staticmethod
    def _get_random_split(
        target_length: int,
        device: torch.device,
        true_values_num: int | None = None,
    ) -> Tensor:
        """
        Generate a random split mask of shape [1, target_length].

        Args:
            target_length: Length of the mask
            device: The device to create the mask on
            true_values_num: Optional number of True values in the mask
        Returns:
            Tensor of shape [1, target_length], dtype=torch.bool, representing the random split mask.
        """
        if true_values_num is None:
            return torch.randint(
                0, 2, (1, target_length), dtype=torch.bool, device=device
            )

        mask = torch.zeros((1, target_length), dtype=torch.bool, device=device)
        true_indices = torch.randperm(target_length, device=device)[:true_values_num]
        mask[0, true_indices] = True
        return mask

    @staticmethod
    def _validate_sampling_params(
        num_samples: int | None,
        fraction: float | None,
    ) -> None:
        """
        Validate sampling parameters for SHAP approximation.

        Args:
            num_samples: Number of samples to generate.
            fraction: Fraction of total possible samples to generate.
        Raises:
            ValueError: If both parameters are None or invalid.
        """
        if num_samples is None and fraction is None:
            raise ValueError("Either num_samples or fraction must be provided.")
        if fraction is not None and (
            not isinstance(fraction, float) or not 0 < fraction <= 1
        ):
            raise ValueError("fraction must be a float in the range (0, 1].")
        if num_samples is not None and (
            not isinstance(num_samples, int) or (num_samples <= 0 and num_samples != -1)
        ):
            raise ValueError("num_samples must be a positive integer.")


class BaseComplementaryShapApproximation(BaseShapApproximation, ABC):
    """Complementary SHAP implementation class."""

    _next_mask: Tensor | None = None
    """Holds the next mask to be returned (the complement of the last generated mask)."""

    _M: Tensor
    """Matrix M used in Complementary calculations."""

    @lru_cache(maxsize=1)
    def _get_num_splits(self, target_length: int) -> int:
        return BaseComplementaryShapApproximation._get_num_splits_static(
            target_length=target_length,
            num_samples=self.num_samples,
            fraction=self.fraction,
        )

    def _initialize_state(self) -> None:
        super()._initialize_state()
        self._zero_mask_skipped = True  # this algorithm cannot use zero mask

    def _get_masks_generator(
        self,
        mask_manager: MasksManager,
        device: torch.device,
        masks: list[Tensor],
    ) -> MaskGenerator:
        n = mask_manager.n
        num_splits = self._get_num_splits(mask_manager.n)
        get_next_split = self._get_next_split
        mode = 1  # 1 - S, -1 - N \ S = ~S

        self._M = torch.zeros((n, n), dtype=torch.int16, device=device)
        M = self._M

        # We can generate only pairs --> no space for zero mask
        # that will be a pair to existing all-ones mask
        if self._get_num_splits(mask_manager.target_length) % 2 == 0:
            self._zero_mask_skipped = True
        else:
            mode = -1  # first mask will be zero mask

        class _MasksGenerator(MaskGenerator):
            """Generator class for masks."""

            def __init__(self, mode: int) -> None:
                """Initialize the MaskGenerator."""
                super().__init__()
                self._iter = self.__mask_iter()
                self._mode = mode

            def send(self, *args: Any, **kwargs: Any) -> tuple[Tensor | None, int]:
                return self._iter.send(*args, **kwargs)

            def throw(self, *args: Any, **kwargs: Any) -> tuple[Tensor | None, int]:
                return self._iter.throw(*args, **kwargs)

            def __mask_iter(self) -> Generator[tuple[Tensor | None, int], None, None]:
                while True:
                    new_split = get_next_split(
                        target_length=mask_manager.n,
                        device=device,
                        generated_masks_num=self.generated_masks,
                        existing_masks=masks,
                    )
                    if new_split is None:
                        break
                    if not new_split.any():
                        logger.debug("Generated zero mask, skipping.")
                        continue

                    new_mask = mask_manager.prepare_mask(split=new_split, device=device)
                    if new_mask is None:
                        logger.debug("Generated mask has no True values, skipping.")
                        continue

                    new_mask_hash = mask_manager.get_hash(new_mask)
                    if mask_manager.seen(mask_hash=new_mask_hash):
                        logger.debug("Generated duplicate mask, skipping.")
                        continue

                    mask_manager.mark_seen(mask_hash=new_mask_hash)
                    self.generated_masks += 1

                    coalition_size = int(new_mask.sum().item())
                    if self._mode == -1:
                        coalition_size = n - coalition_size
                    M[new_mask, coalition_size - 1] += self._mode
                    self._mode *= -1  # switch mode for complementary mask

                    yield new_mask, new_mask_hash

            def __iter__(self) -> MaskGenerator:
                return self

            def __next__(self) -> tuple[Tensor | None, int]:
                return next(self._iter)

            def __len__(self) -> int | None:
                return num_splits

        return _MasksGenerator(mode=mode)

    @staticmethod
    def _get_num_splits_static(
        target_length: int,
        num_samples: int | None = None,
        fraction: float | None = None,
        force_minimal: bool = True,
    ) -> int:
        if num_samples is not None:
            if force_minimal and num_samples < 2 * target_length:
                raise ValueError(
                    "num_samples must be at least equal to the number of features times two."
                )
            if num_samples > (2**target_length - 2):
                return int(
                    2**target_length - 2
                )  # maximum possible masks excluding all-ones and all-zeros mask
            if num_samples % 2 == 1:
                raise ValueError(
                    "num_samples must not be odd to account for complementary masks (in pairs)."
                )
            return num_samples

        # use fraction
        total_masks = int(2**target_length - 2)  # exclude all-ones and all-zeros mask
        r = int(total_masks * fraction)  # type: ignore[operator]
        if r % 2 == 0:
            return r
        return r - 1  # ensure even number of samples
