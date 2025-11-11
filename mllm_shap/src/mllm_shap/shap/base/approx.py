# pylint: disable=invalid-name
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
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,  # pylint: disable=unused-argument
    ) -> Tensor | None:
        """
        Get the next mask split for SHAP value calculation
        from the base minimal masks, if applicable.

        Args:
            n: Length of the masks
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
                    self._base_masks = BaseShapApproximation._generate_minimal_splits(
                        n=n,
                        device=device,
                    )
                    if self._base_masks is None:
                        return None
                    self._first_call = False
                elif not self._zero_mask_skipped:  # 0 mask was rejected, so start from 1
                    # base masks here cannot be None
                    self._base_masks = self._base_masks[1:]  # type: ignore[index]
                    self._zero_mask_skipped = True
                else:  # another mask was rejected, raise
                    raise RuntimeError("Multiple base masks were rejected.")

            if self._base_masks is None:
                raise RuntimeError("Base masks are not present.")
            num_splits = self._get_num_splits(n)
            if num_splits is not None and num_splits < self._base_masks.shape[0]:
                raise RuntimeError(
                    f"Not enough sampling budget, up to {num_splits} "
                    f"calls allowed with required {self._base_masks.shape[0]} for minimal masks."
                )

            if generated_masks_num < self._base_masks.shape[0]:
                if self._base_calls_num != generated_masks_num + int(self._zero_mask_skipped):
                    raise RuntimeError("Multiple base masks were rejected.")

                self._base_calls_num += 1
                return self._base_masks[generated_masks_num, ...].squeeze(0)
        return None

    def _get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        r = self._get_next_split_base(
            n=n,
            device=device,
            generated_masks_num=generated_masks_num,
            existing_masks=existing_masks,
        )
        self._first_call = False
        if r is not None:
            return r

        if generated_masks_num < self._get_num_splits(n=n):
            return self._get_random_split(n=n, device=device)
        return None

    def _initialize_state(self) -> None:
        """
        Initialize internal state before starting mask generation.
        """
        super()._initialize_state()

        self._zero_mask_skipped = False
        self._base_masks = None
        self._base_calls_num = 0

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None:
        self._initialize_state()
        return super().__call__(*args, **kwargs)

    @staticmethod
    def _generate_minimal_splits(n: int, device: torch.device) -> torch.Tensor:
        """
        Generate a minimal set of boolean masks as a batched tensor.
        Shape: (n + 1, n)
        """
        masks = torch.ones((n + 1, n), dtype=torch.bool, device=device)
        masks[0, :] = False
        masks[torch.arange(1, n + 1), torch.arange(n)] = False
        return masks

    @staticmethod
    def _get_random_split(
        n: int,
        device: torch.device,
        true_values_num: int | None = None,
        include_token: int | None = None,
    ) -> Tensor:
        """
        Generate a random split mask of shape [1, n].

        Args:
            n: Length of the mask
            device: The device to create the mask on
            true_values_num: Optional number of True values in the mask
            include_token: Optional index of a token that must be included in the mask
        Returns:
            Tensor of shape [1, n], dtype=torch.bool, representing the random split mask.
        """
        if true_values_num is None:
            return torch.randint(0, 2, (1, n), dtype=torch.bool, device=device)

        # one token is already included
        if include_token is not None:
            n -= 1
            true_values_num -= 1

        mask = torch.zeros((1, n), dtype=torch.bool, device=device)
        true_indices = torch.randperm(n, device=device)[:true_values_num]
        mask[0, true_indices] = True

        if include_token is not None:
            new_mask = torch.zeros((1, n + 1), dtype=torch.bool, device=device)
            new_mask[..., include_token] = True
            new_mask[~new_mask] = mask
            mask = new_mask
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
        if fraction is not None and (not isinstance(fraction, float) or not 0 < fraction <= 1):
            raise ValueError("fraction must be a float in the range (0, 1].")
        if num_samples is not None and (not isinstance(num_samples, int) or (num_samples <= 0 and num_samples != -1)):
            raise ValueError("num_samples must be a positive integer.")


class BaseComplementaryShapApproximation(BaseShapApproximation, ABC):
    """Complementary SHAP implementation class."""

    _M: Tensor | None
    """
    Matrix M used in Complementary calculations -
    number of times feature i appears in coalitions of size j+1.
    """

    _C: Tensor | None
    """
    Matrix C used in Complementary calculations -
    C[i, j] = sum of complementary contributions for feature i in coalitions of size j+1.
    """

    @lru_cache(maxsize=1)
    def _get_num_splits(self, n: int) -> int:
        return BaseComplementaryShapApproximation._get_num_splits_static(
            n=n,
            num_samples=self.num_samples,
            fraction=self.fraction,
        )

    def _initialize_state(self) -> None:
        super()._initialize_state()
        self._get_num_splits.cache_clear()
        self._zero_mask_skipped = True  # this algorithm cannot use zero mask
        self._M = None
        self._C = None

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _get_masks_generator(
        self,
        mask_manager: MasksManager,
        device: torch.device,
        masks: list[Tensor],
        include_mode: bool = True,
        only_unique: bool = True,
        allow_full_or_empty: bool = False,
    ) -> MaskGenerator:
        n = mask_manager.n
        num_splits = self._get_num_splits(mask_manager.n)
        get_next_split = self._get_next_split
        mode = 1  # 1 - S, -1 - N \ S = ~S

        # Initialize M matrix
        if self._M is None:
            self._M = torch.zeros((n, n + 1), dtype=torch.int16, device=device)
        M = self._M

        # We can generate only pairs --> no space for zero mask
        # that will be a pair to existing all-ones mask
        if self._get_num_splits(mask_manager.n) % 2 == 0:
            self._zero_mask_skipped = True
        else:
            mode = -1  # first mask will be zero mask

        class _MasksGenerator(MaskGenerator):
            """Generator class for masks."""

            def __init__(self, mode: int) -> None:
                """Initialize the MaskGenerator."""
                super().__init__()
                self._mode = mode
                self._next_result: tuple[Tensor | None, int] | None = None

            def _mask_iter(self) -> Generator[tuple[Tensor | None, int], None, None]:
                while True:
                    if self._next_result is not None:
                        yield self._next_result
                        self._next_result = None
                        continue

                    new_split = get_next_split(
                        n=mask_manager.n,
                        device=device,
                        generated_masks_num=self.generated_masks,
                        existing_masks=masks,
                    )
                    if new_split is None:
                        break

                    coalition_size = int(new_split.sum().item())
                    if not allow_full_or_empty and (not new_split.any() or new_split.all()):
                        logger.debug("Generated zero or all-ones mask of size %d, skipping.", coalition_size)
                        continue

                    new_split_neg = ~new_split
                    new_mask = mask_manager.prepare_mask(split=new_split, device=device)
                    new_mask_neg = mask_manager.prepare_mask(split=new_split_neg, device=device)
                    if new_mask is None or new_mask_neg is None:
                        logger.info(
                            "Generated mask of size %d (or its negation) has no True values, skipping.", coalition_size
                        )
                        continue

                    new_mask_hash = mask_manager.get_hash(new_mask)
                    new_mask_neg_hash = mask_manager.get_hash(new_mask_neg)
                    if only_unique:
                        if only_unique and (
                            mask_manager.seen(mask_hash=new_mask_hash) or mask_manager.seen(mask_hash=new_mask_neg_hash)
                        ):
                            logger.debug("Generated duplicate mask of size %d, skipping.", coalition_size)
                            continue
                        mask_manager.mark_seen(mask_hash=new_mask_hash)
                        mask_manager.mark_seen(mask_hash=new_mask_neg_hash)

                    self.generated_masks += 2

                    for split, mode in (
                        (new_split, self._mode),
                        (new_split_neg, -self._mode if include_mode else self._mode),
                    ):
                        coalition_size = int(split.sum().item())
                        logger.debug(
                            "new_split: %s, coalition_size: %d, mode: %d",
                            split.squeeze(0),
                            coalition_size,
                            self._mode,
                        )

                        BaseComplementaryShapApproximation._increment_coalition_val(  # pylint: disable=protected-access
                            M, split.squeeze(0), coalition_size, mode
                        )

                    self._next_result = (new_mask_neg, new_mask_neg_hash)
                    yield new_mask, new_mask_hash

            def __len__(self) -> int | None:
                return num_splits

        return _MasksGenerator(mode=mode)

    def _calculate_C_matrix(self, masks: Tensor, similarities: Tensor, device: torch.device) -> None:
        """
        Calculate the C matrix used in Complementary SHAP calculations.

        Args:
            masks: Tensor of shape [m, n] representing the generated masks.
            similarities: Tensor of shape [m, ] representing the similarities for each mask.
            device: The device to perform calculations on.
        Raises:
            ValueError: If masks are not in complementary pairs.
            RuntimeError: If M matrix is not initialized.
        """
        if self._M is None:
            raise RuntimeError("M matrix must be initialized before calculating C matrix.")
        if self._C is None:
            self._C = torch.zeros_like(self._M, dtype=similarities.dtype, device=device)

        m = masks.shape[0] // 2
        if 2 * m != masks.shape[0]:
            raise ValueError("Masks should be in complementary pairs.")

        for i in range(m):
            if not torch.all(masks[2 * i] == ~masks[2 * i + 1]):
                raise ValueError("Masks are not complementary pairs.")

            S = masks[2 * i]
            NS = masks[2 * i + 1]
            s_size = int(S.sum().item())
            ns_size = masks.shape[1] - s_size

            u = similarities[2 * i] - similarities[2 * i + 1]

            BaseComplementaryShapApproximation._increment_coalition_val(self._C, S, s_size, u)
            BaseComplementaryShapApproximation._increment_coalition_val(self._C, NS, ns_size, -u)

    @staticmethod
    def _get_num_splits_static(
        n: int,
        num_samples: int | None = None,
        fraction: float | None = None,
        force_minimal: bool = True,
    ) -> int:
        if num_samples is not None:
            if force_minimal and num_samples < 2 * n:
                raise ValueError("num_samples must be at least equal to the number of features times two.")
            if num_samples > (2**n - 2):
                return int(2**n - 2)  # maximum possible masks excluding all-ones and all-zeros mask
            if num_samples % 2 == 1:
                raise ValueError("num_samples must not be odd to account for complementary masks (in pairs).")
            return num_samples

        # use fraction
        total_masks = int(2**n - 2)  # exclude all-ones and all-zeros mask
        r = int(total_masks * fraction)  # type: ignore[operator]
        if r % 2 == 0:
            return r
        return r - 1  # ensure even number of samples

    @staticmethod
    def _increment_coalition_val(tensor: Tensor, indices: Tensor, coalition_size: int, value: Any) -> None:
        """
        Increment the value in the tensor for the given coalition.
        If coalition_size is 0, update the first column.

        Args:
            tensor: The tensor to update.
            indices: The indices of the features in the coalition.
            coalition_size: The size of the coalition.
            value: The value to add.
        """
        if coalition_size == 0:
            tensor[:, 0] += value
        else:
            tensor[indices, coalition_size] += value
