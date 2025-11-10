# As high-level API has some duplicate code with :class:`Explainer`,
# pylint: disable=duplicate-code
"""Hierarchical SHAP explainer module."""

import math
from copy import deepcopy
from logging import Logger
from time import time
from typing import Any, cast

import torch
from torch import Tensor
from tqdm.auto import tqdm

from ...connectors.base.explainer_cache import ExplainerCache
from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model_response import ModelResponse
from ...utils.logger import get_logger
from ...utils.other import extend_tensor
from ..base.explainer import BaseExplainer, BaseShapExplainer
from ..explainer_result import ExplainerResult
from ..precise import PreciseShapExplainer
from ..normalizers import MinMaxNormalizer
from .enums import Mode

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class HierarchicalExplainer(BaseExplainer):
    """
    SHAP explainer implementing hierarchical approach for speed-up.

    Groups are divided into subgroups recursively until the final group size.
    Groups cannot share different modalities (e.g., text and audio tokens).
    Uses an underlying SHAP explainer for group explanations.

    It has no history nor non-normalized shap values available. Refer to
    :class:`Mode` for details on how groups are formed at the first level.
    """

    k: int
    """Maximum final group size at each level."""

    n_calls: int = 0
    """Number of internal SHAP explainer calls made for last explanation."""

    mode: Mode
    """The mode of the hierarchical explainer."""
    
    _progress_bar: tqdm | None = None

    def __init__(
        self,
        shap_explainer: BaseShapExplainer | None = None,
        mode: Mode = Mode.TEXT,
        k: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the explainer.

        Args:
            k: Maximum final group size at each level.
            shap_explainer: The SHAP explainer instance. 
                Should use :class:`MinMaxNormalizer` for normalization. This is not
                validated internally, but strongly recommended for correct results.
            mode: The mode of the hierarchical explainer.
            kwargs: Additional keyword arguments.
        Raises:
            ValueError: If k is less than 1 or not an integer.
        """
        super().__init__(
            shap_explainer=shap_explainer or PreciseShapExplainer(normalizer=MinMaxNormalizer()), **kwargs
        )

        if k < 1 or int(k) != k:
            raise ValueError("k must be an integer, at least 1.")
        self.k = k

        self.mode = mode

    def __get_subgroups_num(self, n: int) -> int:
        """
        Get the number of subgroups for a given group size.

        Args:
            n: The size of the group.
        Returns:
            The number of subgroups.
        """
        return math.ceil(math.log(n, self.k))

    # pylint: disable=too-many-positional-arguments,too-many-arguments
    def __calculate_group_normalized_shap_values(
        self,
        chat: BaseMllmChat,
        response: ModelResponse,
        group_ids: Tensor | None = None,
        shap_values_mask: Tensor | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **explanation_kwargs: Any,
    ) -> Tensor:
        """
        Get SHAP values for a given group.

        Args:
            chat: The chat instance.
            response: The model response.
            group_ids: A tensor indicating group IDs for explainable tokens.
                Tokens with the same ID belong to the same group and will
                be treated together in SHAP calculations.
            shap_values_mask: A boolean tensor indicating which tokens
                should be considered for SHAP value calculations.
                Takes precedence over group_ids if both are provided.
            generation_kwargs: Additional generation arguments.
            explanation_kwargs: Additional explanation arguments.
        Returns:
            A tensor containing the SHAP values for the group.
        Raises:
            ValueError: If neither shap_values_mask nor group_ids is provided.
        """
        if shap_values_mask is None and group_ids is None:
            raise ValueError("Either shap_values_mask or group_ids must be provided.")
        # avoid warnings about invalid cache
        del response.chat.cache  # type: ignore[union-attr]

        if shap_values_mask is not None:
            # no need to explain a single token
            if shap_values_mask.sum().item() == 1:
                r = torch.zeros_like(shap_values_mask)
                r[shap_values_mask] = 1.0
                return r
            chat.external_shap_values_mask = shap_values_mask
            logger.debug(
                "Calculating SHAP values for %d tokens.",
                shap_values_mask.sum().item(),
            )
        else:
            group_ids = cast(Tensor, group_ids)
            chat.external_group_ids = group_ids
            n_groups = group_ids.max().item()
            # no need to explain a single group of one token
            if n_groups == 1:
                r = torch.zeros_like(group_ids, dtype=torch.float)
                r[group_ids == 1] = 1.0
                return r

            logger.debug(
                "Calculating SHAP values for %d groups of %d tokens.",
                n_groups,
                (group_ids > 0).sum().item(),
            )

        _ = self.shap_explainer(
            model=self.model,
            source_chat=chat,
            response=response,
            **explanation_kwargs,
            **(generation_kwargs or {}),
        )

        if shap_values_mask is not None:
            del chat.external_shap_values_mask
        else:
            del chat.external_group_ids

        self.n_calls += 1
        # correct because no cache hits in hierarchical explainer internal calls are possible
        self.total_n_calls += self.shap_explainer.total_n_calls
        if self._progress_bar is not None:
            self._progress_bar.update(self.shap_explainer.total_n_calls)

        cache = cast(ExplainerCache, response.chat.cache)  # type: ignore[union-attr]
        return cache.normalized_values[: cache.n]

    def __compute(
        self,
        chat: BaseMllmChat,
        response: ModelResponse,
        group_mask: Tensor,
        generation_kwargs: dict[str, Any] | None = None,
        **explanation_kwargs: Any,
    ) -> Tensor:
        """
        Recursively compute hierarchical SHAP values for a given group.

        Args:
            chat: The chat instance.
            response: The model response.
            group_mask: A boolean tensor indicating the group.
            generation_kwargs: Additional generation arguments.
            explanation_kwargs: Additional explanation arguments.
        Returns:
            A tensor containing the hierarchical SHAP values for the group.
        """

        start_idx, end_idx, n = HierarchicalExplainer.__get_group_props(group_mask)
        subgroups_num = self.__get_subgroups_num(n=n)
        subgroup_size = math.ceil(n / subgroups_num)

        logger.debug(
            "Computing SHAP values for group [%d:%d] of size %d with %d subgroups.",
            start_idx,
            end_idx,
            n,
            subgroups_num,
        )

        if subgroups_num == 1:  # base case - group size <= k
            r = self.__calculate_group_normalized_shap_values(
                chat=chat,
                response=response,
                shap_values_mask=group_mask,
                generation_kwargs=generation_kwargs,
                **explanation_kwargs,
            )
            return r

        group_ids = torch.zeros_like(group_mask, dtype=torch.long)
        group_ids[start_idx : end_idx + 1] = HierarchicalExplainer.__repeated_buckets(
            n=n, k=subgroup_size
        )  # noqa: E203

        # calculate SHAP values for this level
        normalized_shap_values = self.__calculate_group_normalized_shap_values(
            chat=chat,
            response=response,
            group_ids=group_ids,
            generation_kwargs=generation_kwargs,
            **explanation_kwargs,
        )

        # calculate SHAP values for next levels
        for subgroup_id in range(1, subgroups_num + 1):
            subgroup_mask = group_mask & (group_ids == subgroup_id)
            subgroup_shap_values = self.__compute(
                chat=chat,
                response=response,
                group_mask=subgroup_mask,
                generation_kwargs=generation_kwargs,
                **explanation_kwargs,
            )
            normalized_shap_values[subgroup_mask] *= subgroup_shap_values[subgroup_mask]

        return normalized_shap_values

    def __save_to_cache(
        self,
        chat: BaseMllmChat,
        source_chat: BaseMllmChat,
        normalized_shap_values: Tensor,
    ) -> None:
        """
        Save the explanation results to the chat cache.

        Args:
            chat: The chat instance.
            source_chat: The source chat instance.
            normalized_shap_values: The computed normalized SHAP values.
        """

        # extend normalized shap values to match response length
        normalized_shap_values = extend_tensor(
            normalized_shap_values,
            target_length=chat.input_tokens_num,
            fill_value=float("nan"),
        )
        shap_values_mask = extend_tensor(
            source_chat.shap_values_mask,
            target_length=chat.input_tokens_num,
            fill_value=False,
        )

        chat.cache = ExplainerCache.create(
            chat=chat,
            explainer_hash=hash(self.shap_explainer),
            responses=[],
            masks=torch.empty((0, chat.input_tokens_num), dtype=torch.bool),
            normalized_values=normalized_shap_values,
            shap_values_mask=shap_values_mask,
        )

    def __call__(
        self,
        *_: Any,
        chat: BaseMllmChat,
        generation_kwargs: dict[str, Any] | None = None,
        progress_bar: bool = True,
        **explanation_kwargs: Any,
    ) -> ExplainerResult:
        generation_kwargs = generation_kwargs or {}
        # disable verbose logging in internal calls
        explanation_kwargs["verbose"] = False
        explanation_kwargs["progress_bar"] = False

        # validation
        super().__call__(
            chat=chat,
            generation_kwargs=generation_kwargs,
            **explanation_kwargs,
        )

        # reset call counters
        self.n_calls = 0
        self.total_n_calls = 0

        t0 = time()
        logger.info("Generating full response from the model...")
        # keep_history=True ==> chat is set in response object
        response = self.model.generate(
            chat=chat,
            keep_history=True,
            **generation_kwargs,
        )
        logger.debug("Generation took %.2f seconds.", time() - t0)

        if progress_bar:
            self._progress_bar = tqdm(
                desc="Calculating SHAP values",
            )
        t0 = time()

        if self.mode == Mode.TEXT:
            normalized_shap_values = self.__compute(
                chat=chat,
                response=response,
                group_mask=chat.shap_values_mask,
                generation_kwargs=generation_kwargs,
                **explanation_kwargs,
            )
        else:
            # compute initial groups. This differs from :method:`__compute` as
            # at this point we cannot assume that groups are contiguous
            # First level groups are for logical purposes cannot be joined together,
            # therefore they do not get batched.
            group_ids = HierarchicalExplainer.__get_group_ids(
                chat=chat, include_role=(self.mode == Mode.MULTI_MODAL_MULTI_USER)
            )
            n_groups = int(group_ids.max().item()) + 1
            logger.info("Total number of groups at first level: %d", n_groups)

            # calculate fist level SHAP values
            response_with_cache = deepcopy(response)
            normalized_shap_values = self.__calculate_group_normalized_shap_values(
                chat=chat,
                response=response_with_cache,
                group_ids=group_ids,
                generation_kwargs=generation_kwargs,
                **explanation_kwargs,
            )

            # call for each group recursively
            for group_id in range(1, n_groups):
                group_mask = chat.shap_values_mask & (group_ids == group_id)
                group_shap_values = self.__compute(
                    chat=chat,
                    response=response,
                    group_mask=group_mask,
                    generation_kwargs=generation_kwargs,
                    **explanation_kwargs,
                )
                normalized_shap_values[group_mask] *= group_shap_values[group_mask]

        logger.debug("Explanation took %.2f seconds.", time() - t0)

        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None

        full_chat = cast(BaseMllmChat, response.chat)
        self.__save_to_cache(
            chat=full_chat,
            source_chat=chat,
            normalized_shap_values=normalized_shap_values,
        )

        return ExplainerResult(
            source_chat=chat,
            full_chat=full_chat,
            history=None,
            total_n_calls=self.total_n_calls,
        )

    @staticmethod
    def __get_group_props(mask: Tensor) -> tuple[int, int, int]:
        """
        Get the start and end indices of the True values in the mask.
        Assumes that the mask contains at least one True value and
        that True values are contiguous and appear only within one segment.

        Args:
            mask: A boolean tensor indicating explainable tokens.
        Returns:
            A tuple containing the start and end indices and the size of the group.
        """
        start_idx, end_idx = mask.nonzero(as_tuple=True)[0][[0, -1]].tolist()
        n = end_idx - start_idx + 1
        return start_idx, end_idx, n

    @staticmethod
    def __get_group_ids(chat: "BaseMllmChat", include_role: bool = True) -> Tensor:
        """
        Get initial group IDs for explainable tokens in the chat, splitting by
        contiguity, modality, and token role changes.

        Args:
            chat: The chat instance containing `shap_values_mask`,
                `tokens_modality_flag`, and `token_roles`.
            include_role: Whether to consider token roles when determining groups.
        Returns:
            Tensor: Group IDs for explainable tokens. Tokens with different modalities
            or roles will be assigned separate groups even if contiguous.
        Example:
            For `include_role=True` and the following token properties:

                mask:     tensor([T, T, F, T, T, T, F, F, T, T])
                modality: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
                roles:    tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

            The output will be:

                tensor([0, 0, 0, 1, 1, 2, 0, 0, 3, 3])
        """
        mask = chat.shap_values_mask
        modality_flag = chat.tokens_modality_flag
        token_roles = chat.token_roles
        device = mask.device

        group_ids = torch.zeros_like(mask, dtype=torch.long, device=device)

        # Previous token info
        prev_mask = torch.cat([torch.tensor([False], device=device), mask[:-1]])
        prev_modality = torch.cat(
            [torch.tensor([modality_flag[0]], device=device), modality_flag[:-1]]
        )

        # Start new group if:
        # - token is explainable
        # - AND (previous not explainable OR modality changed OR role changed (if `include_role`))
        group_mask = ~prev_mask | (modality_flag != prev_modality)
        if include_role:
            prev_role = torch.cat(
                [torch.tensor([token_roles[0]], device=device), token_roles[:-1]]
            )
            group_mask |= token_roles != prev_role
        group_start = mask & group_mask

        # Assign cumulative group IDs
        group_ids[mask] = torch.cumsum(group_start[mask].int(), dim=0)

        return group_ids

    @staticmethod
    def __repeated_buckets(n: int, k: int) -> torch.Tensor:
        """
        Create a tensor of repeated integers from 1 upwards,
        each repeated k times, total length n.

        Args:
            n: Total length of the output tensor.
            k: Number of repetitions for each integer.
        Returns:
            A tensor of shape [n] with the repeated integers.
        Example:
            For n=10 and k=3, the output will be:
                tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        """
        # Number of full repetitions needed
        reps = (n + k - 1) // k  # ceiling division
        # Create the repeated sequence
        x = torch.arange(1, reps + 1).repeat_interleave(k)
        # Trim to exact length n
        return x[:n]
