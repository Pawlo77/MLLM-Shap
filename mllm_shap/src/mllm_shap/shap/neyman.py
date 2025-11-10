# Similar to :class:`BaseShapExplainer`
# pylint: disable=duplicated-code
"""Complementary Neyman SHAP explainer implementation."""

import gc
from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Generator
from functools import lru_cache

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from ..connectors.base.chat import BaseMllmChat
from ..connectors.base.explainer_cache import ExplainerCache
from ..connectors.base.model import BaseMllmModel
from ..connectors.base.model_response import ModelResponse
from ..utils.logger import get_logger
from ..utils.other import extend_tensor
from .base._cache_manager import CacheManager
from .base._generate_responses import generate_responses
from .base._masks_manager import MaskGenerator, MasksManager
from .base._validators import BaseShapCallConfig, BaseShapConfig
from .base.embeddings import BaseEmbeddingReducer, BaseExternalEmbedding
from .base.normalizers import BaseNormalizer
from .base.similarity import BaseEmbeddingSimilarity
from .embeddings import MeanReducer
from .enums import Mode
from .explainer_result import ExplainerResult
from .normalizers import PowerShiftNormalizer
from .similarity import CosineSimilarity
from .base.approx import BaseComplementaryShapApproximation

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods,invalid-name
class ComplementaryNeymanShapExplainer(BaseComplementaryShapApproximation):
    """Base Complementary Neyman SHAP implementation class"""

    initial_num_samples: int | None
    """Initial number of samples to draw in the first step."""

    initial_fraction: float | None
    """Initial fraction of samples to draw in the first step."""

    _C: Tensor | None
    """C matrix for Neyman allocation."""

    _step: int
    """Current step in the Neyman allocation process."""

    _i: int
    _j: int
    """Indices for tracking position in the _M matrix."""

    def __init__(
        self,
        *args,
        initial_num_samples: int | None = None,
        initial_fraction: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._validate_sampling_params(
            initial_num_samples=initial_num_samples,
            initial_fraction=initial_fraction,
        )

        self.initial_num_samples = initial_num_samples
        self.initial_fraction = initial_fraction

    @lru_cache(maxsize=1)
    def _get_num_splits(self, target_length: int) -> int:
        """"
        Get total number of splits to generate, as well
        as number of initial splits for each entry of matrix M.
        It determines initial step duration.

        Args:
            target_length: Number of features / tokens.
        Returns:
            Number of masks to generate.
        """

        try:
            m = super()._get_num_splits(target_length=target_length)
        except ValueError as e:
            raise ValueError("Total number of splits could not be determined.") from e

        try:
            m_initial = BaseComplementaryShapApproximation._get_num_splits_static(
                target_length=target_length,
                num_samples=self.initial_num_samples,
                fraction=self.initial_fraction,
                force_minimal=False,
            )
        except ValueError as e:
            raise ValueError("Initial number of splits could not be determined.") from e

        if m_initial < m:
            raise ValueError(
                f"Initial number of splits {m_initial} is larger than total number of splits {m}."
            )

    def _get_next_split(
        self,
        target_length: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        self._first_call = False

        if self._step == 1:  # initial sampling
            if generated_masks_num < self._get_num_splits(target_length):
                if self._next_mask is not None:
                    r = self._next_mask
                    self._next_mask = None
                    return r

                new_mask = self._get_random_split(
                    target_length=target_length, device=device
                )
                self._next_mask = ~new_mask
                return new_mask
            return None

        # remaining budget allocation steps

    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        pass

    def _initialize_state(self) -> None:
        """
        Initialize internal state before starting mask generation.
        """
        super()._initialize_state()

        self._step = 1
        self._i = 0
        self._j = 0
        self._C = None

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    def __call__(
        self,
        model: BaseMllmModel,
        source_chat: BaseMllmChat,
        response: ModelResponse,
        progress_bar: bool = True,
        verbose: bool = False,
        **generate_kwargs: Any,
    ) -> list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None:
        self._initialize_state()

        __config = BaseShapCallConfig(
            model=model,
            source_chat=source_chat,
            response=response,
            progress_bar=progress_bar,
            verbose=verbose,
        )
        # validated within BaseShapCallConfig
        response_chat: BaseMllmChat = __config.response.chat  # type: ignore[assignment]
        source_chat = __config.source_chat
        device = source_chat.torch_device
        self.total_n_calls = 0

        mask_manager = MasksManager(chat=source_chat)
        cache_manager = CacheManager(
            chat=response_chat,
            explainer_hash=hash(self),
        )

        logger.info(
            "Number of tokens for explainability: %d (up to %d additional calls)",
            mask_manager.n,
            mask_manager.max_masks_number,
        )

        masks = [mask_manager.get_initial_mask(device=device)]
        responses = [__config.response]

        gen = self._get_masks_generator(
            mask_manager=mask_manager, device=device, masks=masks
        )
        chats_skipped, history = generate_responses(
            masks=masks,
            responses=responses,
            gen=gen,
            source_chat=source_chat,
            model=__config.model,
            cache_manager=cache_manager,
            n_generator_jobs=1,  # this is not parallelizable
            progress_bar=__config.progress_bar,
            verbose=__config.verbose,
            **generate_kwargs,
        )

        # retrieve generated masks from the generator
        self.total_n_calls = gen.generated_masks
