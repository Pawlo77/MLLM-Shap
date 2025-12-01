"""Complementary SHAP explainer implementation."""

from logging import Logger
import math
from typing import Any, cast

import gc
import torch
from torch import Tensor

from ...utils.logger import get_logger
from ..base.complementary import BaseComplementaryShapApproximation
from ..base.shap_explainer import minmax_normalize
from ..base._validators import BaseShapCallConfig
from ..base._masks_manager import MasksManager
from ..base._cache_manager import CacheManager
from ...connectors.base.model import BaseMllmModel
from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model_response import ModelResponse

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class BaseComplementaryShapExplainer(BaseComplementaryShapApproximation):
    """Complementary SHAP implementation class."""

    # pylint: disable=unused-argument,invalid-name
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        if not self._zero_mask_skipped:
            raise RuntimeError("Zero mask was not skipped during mask generation.")
        if self._M is None:
            raise RuntimeError("M matrix must be initialized before calculating SHAP values.")

        # Adjust masks and similarities to account for skipped zero mask
        # that is remove full ones mask
        masks = masks[1:]
        similarities = similarities[1:]
        if self._C is None:
            self._calculate_C_matrix(masks=masks, similarities=similarities, device=device)

        # exclude zero-mask column
        M = self._M[:, 1:]
        C = cast(Tensor, self._C)[:, 1:]

        # it is not guaranteed especially with small budget
        non_zero_mask = M > 0
        ratio = torch.zeros_like(C)
        ratio[non_zero_mask] = C[non_zero_mask] / M[non_zero_mask]
        return torch.sum(ratio, dim=1) / M.shape[0]

    def _get_next_split_base(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,  # pylint: disable=unused-argument
    ) -> Tensor | None:
        """
        Get the base mask for the next split.
        K-th base masks are mask with close to half of features included,
        with k-th feature definitely included. There are exactly n such masks.

        Args:
            n: Number of features.
            device: Device on which the mask should be allocated.
            generated_masks_num: Number of masks already generated.
            existing_masks: List of already generated masks (not used here).
        Returns:
            The next base mask tensor or None if all base masks have been generated.
        """
        if self.include_minimal_masks and generated_masks_num < n:
            return self._get_random_split(
                n=n,
                device=device,
                true_values_num=math.ceil(n / 3),
                include_token=generated_masks_num,
            )
        return None

    def _compute_prefix_shap(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        masks_tensor: Tensor,
        similarities: Tensor,
        source_chat: BaseMllmChat,
        device: torch.device,
        k_masks: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute raw and normalized SHAP for the first `k_masks` entries
        (including the initial all-ones mask at index 0). Uses a local
        M/C accumulation so results depend only on the prefix.
        """
        COMP_PAIRS_MIN = 3
        if k_masks < COMP_PAIRS_MIN or ((k_masks - 1) % 2) != 0:
            raise ValueError("k_masks must be >= 3 and (k_masks-1) must be even (full complementary pairs).")

        expl_mask = source_chat.shap_values_mask
        n = int(expl_mask.sum().item())

        # local accumulators: (n, n+1), coalition sizes are 0..n
        dtype = similarities.dtype if similarities.numel() > 0 else torch.float32
        M_local = torch.zeros((n, n + 1), dtype=dtype, device=device)
        C_local = torch.zeros((n, n + 1), dtype=dtype, device=device)

        # take prefix excluding the initial all-ones mask
        masks_pref = masks_tensor[1:k_masks, expl_mask]
        sims_pref = similarities[: (k_masks - 1)]
        m = (k_masks - 1) // 2

        # accumulate complementary-pair contributions
        for i in range(m):
            S = masks_pref[2 * i]
            NS = masks_pref[2 * i + 1]
            if not torch.all(NS == ~S):
                raise RuntimeError("Prefix masks are not complementary pairs.")

            s_size = int(S.sum().item())
            ns_size = n - s_size

            u = sims_pref[2 * i] - sims_pref[2 * i + 1]

            # counts per coalition size
            M_local[:, s_size] += S.to(M_local.dtype)
            M_local[:, ns_size] += NS.to(M_local.dtype)

            # sum of complementary contributions
            C_local[:, s_size] += S.to(C_local.dtype) * u
            C_local[:, ns_size] += NS.to(C_local.dtype) * (-u)

        # compute raw SHAP, exclude zero-mask column
        M = M_local[:, 1:]
        C = C_local[:, 1:]

        positive = M > 0
        ratio = torch.zeros_like(C)
        ratio[positive] = C[positive] / M[positive]
        raw_shap = torch.sum(ratio, dim=1) / M.shape[0]

        # MinMax normalize over explainable tokens
        norm = minmax_normalize(raw_shap)

        return raw_shap.detach().cpu(), norm.detach().cpu()

    def _build_trajectory(  # pylint: disable=too-many-arguments
        self,
        *,
        masks_tensor: Tensor,
        similarities: Tensor,
        source_chat: BaseMllmChat,
        device: torch.device,
        initial_len_with_base: int,
    ) -> list[dict[str, Any]]:
        """
        Recompute SHAP after every complementary pair (prefix of masks).
        """
        total = int(masks_tensor.shape[0])
        traj: list[dict[str, Any]] = []

        for k in range(3, total + 1, 2):  # initial (all-ones) + pairs
            raw, norm = self._compute_prefix_shap(
                masks_tensor=masks_tensor,
                similarities=similarities,
                source_chat=source_chat,
                device=device,
                k_masks=k,
            )
            num_pairs = (k - 1) // 2
            traj.append(
                {
                    "num_masks": k - 1,  # excludes the initial all-ones mask
                    "num_pairs": num_pairs,
                    "stage": "initial" if k <= initial_len_with_base else "complementary",
                    "shap": raw,  # raw SHAP over explainable tokens
                    "normalized_shap": norm,  # min-max normalized over explainable tokens
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
