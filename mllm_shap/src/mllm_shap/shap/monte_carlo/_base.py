"""Base Monte Carlo approximation SHAP explainer implementation."""

from abc import ABC
from functools import lru_cache
from logging import Logger
from typing import Any

import gc
import torch
from torch import Tensor

from ...utils.logger import get_logger
from ..base.approx import BaseShapApproximation
from ..base.shap_explainer import minmax_normalize
from ..base._validators import BaseShapCallConfig
from ..base._masks_manager import MasksManager
from ..base._cache_manager import CacheManager
from ...connectors.base.model import BaseMllmModel
from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model_response import ModelResponse

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class BaseMcShapExplainer(BaseShapApproximation, ABC):
    """Base Monte Carlo SHAP implementation class"""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def _build_trajectory(
        self,
        *,
        masks_tensor: Tensor,
        similarities: Tensor,
        source_chat: BaseMllmChat,
        device: torch.device,
        initial_len_with_base: int,
    ) -> list[dict[str, Any]]:
        """
        Recompute SHAP after every added mask (prefixes of masks).
        Returns list of dicts with same structure as Neyman trajectory.
        """
        total = int(masks_tensor.shape[0])
        traj: list[dict[str, Any]] = []

        expl_mask = source_chat.shap_values_mask
        for k in range(2, total + 1):
            # take prefix including base mask at index 0
            masks_pref = masks_tensor[:k, ...][..., expl_mask]
            sims_pref = similarities[:k]

            # compute raw SHAP for prefix using the same calculation used during run
            raw_shap = self._calculate_shap_values(masks=masks_pref, similarities=sims_pref, device=device)

            # MinMax normalize over explainable tokens
            norm = minmax_normalize(raw_shap)

            traj.append(
                {
                    "num_masks": k - 1,  # excludes the initial all-ones mask
                    "num_pairs": k - 1,
                    "stage": "initial" if k <= initial_len_with_base else "mc",
                    "shap": raw_shap.detach().cpu(),
                    "normalized_shap": norm.detach().cpu(),
                }
            )
        return traj

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,duplicate-code
    def __call__(
        self,
        model: BaseMllmModel,
        source_chat: BaseMllmChat,
        response: ModelResponse,
        progress_bar: bool = True,
        verbose: bool = False,
        *,
        return_trajectory: bool = False,
        n_generator_jobs: int = 1,
        **generate_kwargs: Any,
    ) -> list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None:
        """
        Same behaviour as BaseShapExplainer.__call__ but optionally returns trajectory.
        """
        __config = BaseShapCallConfig(
            model=model,
            source_chat=source_chat,
            response=response,
            progress_bar=progress_bar,
            verbose=verbose,
        )
        self._initialize_state()

        response_chat: BaseMllmChat = __config.response.chat  # type: ignore[assignment]
        source_chat = __config.source_chat
        device = source_chat.torch_device

        mask_manager = MasksManager(chat=source_chat, log_stats=True)
        cache_manager = CacheManager(
            chat=response_chat,
            explainer_hash=hash(self),
        )

        masks = [mask_manager.get_initial_mask(device=device)]
        responses = [__config.response]

        chats_skipped, history = self._generate_step(
            mask_manager=mask_manager,
            masks=masks,
            device=device,
            responses=responses,
            source_chat=source_chat,
            model=__config.model,
            cache_manager=cache_manager,
            n_generator_jobs=n_generator_jobs,
            progress_bar=__config.progress_bar,
            verbose=__config.verbose,
            **generate_kwargs,
        )

        if cache_manager.extracted_num > 0:
            logger.info(
                "Deduplicated %d/%d masks using existing cache.",
                cache_manager.extracted_num,
                len(masks) - 1,  # exclude base mask
            )

        if len(masks) - 1 <= chats_skipped:
            raise RuntimeError(
                "Not enough tokens to explain after filtering out empty chats. "
                "Ensure that shap_values_mask has at least two True values."
            )

        masks_tensor = torch.stack(masks, dim=0)

        # clean up
        initial_len_with_base = len(masks)
        del mask_manager
        del cache_manager
        del masks
        gc.collect()

        similarities = self._get_similarities(responses=responses, model=model)

        shap_values, normalized_shap_values = self._get_shap_values(
            model=__config.model,
            masks=masks_tensor,
            responses=responses,
            source_chat=source_chat,
            device=device,
            similarities=similarities,
        )

        self._save_to_cache(
            chat=response_chat,
            source_chat=source_chat,
            responses=responses,
            masks=masks_tensor,
            shap_values=shap_values,
            normalized_shap_values=normalized_shap_values,
        )

        result = history
        if return_trajectory:
            trajectory = self._build_trajectory(
                masks_tensor=masks_tensor,
                similarities=similarities,
                source_chat=source_chat,
                device=device,
                initial_len_with_base=initial_len_with_base,
            )
            return {"history": history, "trajectory": trajectory}  # type: ignore[return-value]

        return result

    @lru_cache(maxsize=1)
    def _get_num_splits(self, n: int) -> int:
        if self.num_samples is not None:
            if self.num_samples == -1:
                if self.include_minimal_masks:
                    # Minimal: only single-feature masks and empty mask
                    return n + 1
                raise ValueError("num_samples cannot be -1 when include_minimal_masks is False.")
            if self.num_samples < n + 1:
                logger.warning(
                    (
                        "Number of samples (%d) is less than number of features (%d)."
                        " Using number of features as number of samples."
                    ),
                    self.num_samples,
                    n,
                )
                return n + 1
            if self.num_samples > (2**n - 1):
                return int(2**n - 1)  # maximum possible masks excluding all-ones mask
            return self.num_samples

        total_masks = 2**n - 1  # exclude all-ones mask
        r = int(total_masks * self.fraction)
        if r < n + 1:
            r = n + 1  # minimal: single-feature masks and empty mask
            logger.warning(
                (
                    "Calculated number of samples (%d) is less than minimal"
                    " required (%d). Using minimal number of samples."
                ),
                r,
                n + 1,
            )
        return r

    # pylint: disable=unused-argument
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        included_mean = (masks * similarities[:, None]).sum(dim=0) / masks.sum(dim=0)
        excluded_mean = ((~masks) * similarities[:, None]).sum(dim=0) / (~masks).sum(dim=0)
        return included_mean - excluded_mean
