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
from ..base.explainer import BaseExplainer
from ..base.shap_explainer import BaseShapExplainer
from ..base.approx import BaseShapApproximation
from ..explainer_result import ExplainerResult
from ..precise import PreciseShapExplainer
from ..normalizers import MinMaxNormalizer
from .enums import Mode
from .graph import GraphNode

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods,too-many-instance-attributes
class HierarchicalExplainer(BaseExplainer):
    """
    SHAP explainer implementing hierarchical approach for speed-up.

    Groups are divided into subgroups recursively until the final group size.
    Groups cannot share different modalities (e.g., text and audio tokens).
    Uses an underlying SHAP explainer for group explanations.

    Should be used with SHAP explainers that normalize using :class:`MinMaxNormalizer`.

    It has no history nor non-normalized shap values available. Refer to
    :class:`Mode` for details on how groups are formed at the first level.
    """

    k: int
    """Maximum final group size at each level."""

    n_calls: int
    """Number of internal SHAP explainer calls made for last explanation."""

    mode: Mode
    """The mode of the hierarchical explainer."""

    use_importance_sampling: bool
    """Whether to use importance for setting sampling budget (for each group)."""

    importance_sampling_min_fraction: float
    """Minimum fraction for importance sampling."""

    computation_graph: GraphNode | None
    """Computation graph for the last explanation."""

    first_layer_explainer: BaseShapExplainer | None
    """
    The SHAP explainer instance for the first layer.
    If provided, first layer explanation will be done using this explainer
    - it will be fitted with all explainable tokens. Calculated SV
    will be summed per group and used as first layer SHAP values.
    """

    _progress_bar: tqdm | None = None
    """Progress bar for explanation process."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        shap_explainer: BaseShapExplainer | None = None,
        first_layer_explainer: BaseShapExplainer | None = None,
        mode: Mode = Mode.TEXT,
        k: int = 10,
        use_importance_sampling: bool = False,
        importance_sampling_min_fraction: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the explainer.

        Args:
            shap_explainer: The SHAP explainer instance.
                Should use :class:`MinMaxNormalizer` for normalization. This is not
                validated internally, but strongly recommended for correct results.
            first_layer_explainer: The SHAP explainer instance for the first layer.
                Should use :class:`MinMaxNormalizer` for normalization. This is not
                validated internally, but strongly recommended for correct results.
                If provided, instead of using standard first layer explanation, this
                instance will be fitted with all explainable tokens and used to calculate
                first layer SHAP values.
            k: Maximum final group size at each level.
            mode: The mode of the hierarchical explainer.
            use_importance_sampling: Whether to use importance for setting sampling budget (for each group).
                Applicable only if `shap_explainer` supports fraction-based sampling.
            importance_sampling_min_fraction: Minimum fraction for importance sampling.
            kwargs: Additional keyword arguments.
        Raises:
            ValueError: If k is less than 1 or not an integer.
        """
        super().__init__(
            shap_explainer=shap_explainer or PreciseShapExplainer(normalizer=MinMaxNormalizer()),
            **kwargs,
        )

        if not isinstance(self.shap_explainer.normalizer, MinMaxNormalizer):
            logger.warning(
                "It is strongly recommended to use MinMaxNormalizer with HierarchicalExplainer for correct results."
            )

        if k < 2 or int(k) != k:  # pylint: disable=magic-value-comparison
            raise ValueError("k must be an integer, at least 2.")
        self.k = k

        self.mode = mode

        if use_importance_sampling and (
            not isinstance(self.shap_explainer, BaseShapApproximation) or self.shap_explainer.fraction is None
        ):
            raise ValueError(
                "use_importance_sampling is True, but shap_explainer does not support fraction-based approximation."
            )
        self.use_importance_sampling = use_importance_sampling

        if not isinstance(importance_sampling_min_fraction, float) or not (
            0.0 < importance_sampling_min_fraction <= 1.0
        ):
            raise ValueError("importance_sampling_min_fraction must be in (0.0, 1.0].")
        self.importance_sampling_min_fraction = importance_sampling_min_fraction

        if first_layer_explainer is not None:
            if not isinstance(first_layer_explainer, BaseShapExplainer):
                raise ValueError("first_layer_explainer must be an instance of BaseShapExplainer.")
            if (
                not shap_explainer.normalizer.__class__  # type: ignore[union-attr]
                == first_layer_explainer.normalizer.__class__
            ):
                logger.warning(
                    (
                        "It is strongly recommended that first_layer_explainer "
                        "uses the same normalizer as shap_explainer for correct results."
                    )
                )
        self.first_layer_explainer = first_layer_explainer

    def __get_subgroups_num(self, n: int) -> int:
        """
        Get the number of subgroups for a given group size.

        Args:
            n: The size of the group.
        Returns:
            The number of subgroups.
        """
        return math.ceil(math.log(n, self.k))

    def __update_progress(self, explainer: BaseShapExplainer) -> None:
        """
        Update progress based on the given explainer's call count.

        Args:
            explainer: The explainer instance whose call count is used to update progress.
        """
        self.n_calls += 1
        self.total_n_calls += explainer.total_n_calls
        if self._progress_bar is not None:
            self._progress_bar.update(explainer.total_n_calls)

    # pylint: disable=too-many-positional-arguments,too-many-arguments
    def __calculate_group_normalized_shap_values(
        self,
        chat: BaseMllmChat,
        response: ModelResponse,
        group_ids: Tensor | None = None,
        shap_values_mask: Tensor | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        importance: float = 1.0,
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
            importance: Importance value for the group, used for sampling budget if
                `use_importance_sampling` is True.
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
                r = torch.zeros_like(shap_values_mask, dtype=torch.float)
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
                r = torch.full_like(group_ids, fill_value=float("nan"), dtype=torch.float)
                r[group_ids == 1] = 1.0
                return r
            logger.debug(
                "Calculating SHAP values for %d groups of %d tokens.",
                n_groups,
                (group_ids > 0).sum().item(),
            )

        if self.use_importance_sampling:
            # set fraction based on importance
            base_fraction = cast(BaseShapApproximation, self.shap_explainer).fraction
            if base_fraction is None:
                raise RuntimeError("shap_explainer fraction is None, cannot use importance sampling.")

            new_fraction = max(
                self.importance_sampling_min_fraction,
                min(1.0, base_fraction * importance),
            )
            cast(BaseShapApproximation, self.shap_explainer).fraction = new_fraction
            logger.debug(
                "Setting SHAP explainer fraction to %.4f based on importance %.4f.",
                new_fraction,
                importance,
            )

        _ = self.shap_explainer(
            model=self.model,
            source_chat=chat,
            response=response,
            **explanation_kwargs,
            **(generation_kwargs or {}),
        )

        # clean up
        if self.use_importance_sampling:
            # restore original fraction
            cast(BaseShapApproximation, self.shap_explainer).fraction = base_fraction
        if shap_values_mask is not None:
            del chat.external_shap_values_mask
        else:
            del chat.external_group_ids

        self.__update_progress(explainer=self.shap_explainer)
        return HierarchicalExplainer.__extract_normalized_shap_values(response=response)

    # pylint: disable=too-many-locals
    def __compute(
        self,
        chat: BaseMllmChat,
        response: ModelResponse,
        group_mask: Tensor,
        generation_kwargs: dict[str, Any] | None = None,
        importance: float = 1.0,
        _verbose: bool = False,
        **explanation_kwargs: Any,
    ) -> tuple[Tensor, GraphNode | None]:
        """
        Recursively compute hierarchical SHAP values for a given group.

        Args:
            chat: The chat instance.
            response: The model response.
            group_mask: A boolean tensor indicating the group.
            generation_kwargs: Additional generation arguments.
            importance: Importance value for the group, used for sampling budget if
                `use_importance_sampling` is True.
            verbose: Whether to save computation graph details.
            explanation_kwargs: Additional explanation arguments.
        Returns:
            A tuple of tensor containing the hierarchical SHAP values for the group,
            computation graph root or None if `verbose` is False.
        """
        start_idx, end_idx, n = HierarchicalExplainer.__get_group_props(group_mask)
        subgroups_num = self.__get_subgroups_num(n=n)
        computation_graph = None

        logger.debug(
            "Computing SHAP values for group [%d:%d] of size %d with %d subgroups.",
            start_idx,
            end_idx,
            n,
            subgroups_num,
        )

        if subgroups_num <= 1:  # base case - group size <= k
            r = self.__calculate_group_normalized_shap_values(
                chat=chat,
                response=response,
                shap_values_mask=group_mask,
                generation_kwargs=generation_kwargs,
                importance=importance,
                **explanation_kwargs,
            )
            if _verbose:
                computation_graph = GraphNode(shap_values=r.clone(), children=[], group_mask=group_mask.clone())
            return r, computation_graph

        subgroup_size = math.ceil(n / subgroups_num)
        group_ids = torch.zeros_like(group_mask, dtype=torch.long)
        group_ids[start_idx : end_idx + 1] = HierarchicalExplainer.__repeated_buckets(  # noqa: E203
            n=n, k=subgroup_size
        )  # noqa: E203

        # calculate SHAP values for this level
        normalized_shap_values = self.__calculate_group_normalized_shap_values(
            chat=chat,
            response=response,
            group_ids=group_ids,
            generation_kwargs=generation_kwargs,
            importance=importance,
            **explanation_kwargs,
        )

        if _verbose:
            computation_graph = GraphNode(
                shap_values=normalized_shap_values.clone(), children=[], group_ids=group_ids.clone()
            )

        # calculate SHAP values for next levels
        for subgroup_id in range(1, subgroups_num + 1):
            subgroup_mask = group_mask & (group_ids == subgroup_id)
            sv = normalized_shap_values[subgroup_mask][0].item()
            if sv == 0:
                logger.debug(
                    "Skipping group %d explanation as its SHAP value is zero.",
                    subgroup_id,
                )
                if _verbose:
                    computation_graph.children.append(GraphNode())  # type: ignore[union-attr]
                continue

            subgroup_shap_values, subgroup_computation_graph = self.__compute(
                chat=chat,
                response=response,
                group_mask=subgroup_mask,
                generation_kwargs=generation_kwargs,
                importance=float(normalized_shap_values[subgroup_mask][0].item()),
                **explanation_kwargs,
            )
            normalized_shap_values[subgroup_mask] *= subgroup_shap_values[subgroup_mask]
            if _verbose:
                computation_graph.children.append(subgroup_computation_graph)  # type: ignore

        return normalized_shap_values, computation_graph

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
            masks=torch.empty((0, chat.input_tokens_num), dtype=torch.bool, device=chat.torch_device),
            normalized_values=normalized_shap_values,
            shap_values_mask=shap_values_mask,
        )

    def __handle_with_first_level(
        self,
        chat: BaseMllmChat,
        response: ModelResponse,
        generation_kwargs: dict[str, Any] | None = None,
        first_layer_explanation_kwargs: dict[str, Any] | None = None,
        _verbose: bool = False,
        **explanation_kwargs: Any,
    ) -> Tensor:
        """
        Handle explanation mode with business-aware first level grouping.

        Args:
            chat: The chat instance.
            response: The model response.
            generation_kwargs: Additional generation arguments.
            first_layer_explanation_kwargs: Additional explanation arguments for the first layer.
            verbose: Whether to save computation graph details.
            explanation_kwargs: Additional explanation arguments.
        Returns:
            A tensor containing the hierarchical SHAP values for the group.
        """
        # compute initial groups. This differs from :method:`__compute` as
        # at this point we cannot assume that groups are contiguous
        # First level groups are for logical purposes cannot be joined together,
        # therefore they do not get batched.
        group_ids = HierarchicalExplainer.__get_group_ids(
            chat=chat, include_role=(self.mode == Mode.MULTI_MODAL_MULTI_USER)
        )
        n_groups = int(group_ids.max().item()) + 1
        logger.debug("Initial number of groups at first level: %d", n_groups - 1)

        # further split large groups to smaller ones
        global_offset = 0
        group_ids_split = torch.zeros_like(group_ids, dtype=torch.long)
        for group_id in range(1, n_groups):
            group_mask = group_ids == group_id
            start_idx, end_idx, n = HierarchicalExplainer.__get_group_props(group_mask)
            subgroup_size = math.ceil(n / self.__get_subgroups_num(n=n))
            group_ids_split[start_idx : end_idx + 1] = (  # noqa: E203
                HierarchicalExplainer.__repeated_buckets(n=n, k=subgroup_size) + global_offset
            )
            global_offset = int(group_ids_split[start_idx : end_idx + 1].max().item())  # noqa: E203
        n_groups = int(group_ids_split.max().item()) + 1

        logger.info("Total number of groups at first level: %d", n_groups)

        # calculate fist level SHAP values
        response_with_cache = deepcopy(response)
        if self.first_layer_explainer is None:  # business-aware first layer explainer
            normalized_shap_values = self.__calculate_group_normalized_shap_values(
                chat=chat,
                response=response_with_cache,
                group_ids=group_ids_split,
                generation_kwargs=generation_kwargs,
                **explanation_kwargs,
            )
        else:  # separate first-layer explainer
            logger.debug("Calculating first layer explanation using separate explainer.")
            self.first_layer_explainer(
                model=self.model,
                source_chat=chat,
                response=response_with_cache,
                **(first_layer_explanation_kwargs or {}),
                **(generation_kwargs or {}),
            )

            self.__update_progress(explainer=self.first_layer_explainer)
            response_normalized_values = HierarchicalExplainer.__extract_normalized_shap_values(
                response=response_with_cache
            )

            normalized_shap_values = torch.full_like(response_normalized_values, fill_value=float("nan"))
            # set SHAP values per group as sum of all tokens in the group
            for group_id in range(1, n_groups):
                group_mask = group_ids_split == group_id
                sv = response_normalized_values[group_mask].sum().item()
                normalized_shap_values[group_mask] = sv

        if _verbose:
            self.computation_graph = GraphNode(
                shap_values=normalized_shap_values.clone(),
                children=[],
                group_ids=group_ids_split.clone(),
            )

        # call for each group recursively
        for group_id in range(1, n_groups):
            group_mask = group_ids_split == group_id
            sv = normalized_shap_values[group_mask][0].item()
            if sv == 0:
                logger.debug(
                    "Skipping group %d explanation as its SHAP value is zero.",
                    group_id,
                )
                if _verbose:
                    self.computation_graph.children.append(GraphNode())  # type: ignore[union-attr]
                continue

            group_shap_values, subgroup_computation_graph = self.__compute(
                chat=chat,
                response=response,
                group_mask=group_mask,
                generation_kwargs=generation_kwargs,
                importance=sv,
                **explanation_kwargs,
            )
            normalized_shap_values[group_mask] *= group_shap_values[group_mask]
            if _verbose:
                self.computation_graph.children.append(subgroup_computation_graph)  # type: ignore

        return normalized_shap_values

    def __call__(
        self,
        *_: Any,
        chat: BaseMllmChat,
        generation_kwargs: dict[str, Any] | None = None,
        progress_bar: bool = True,
        first_layer_explanation_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
        **explanation_kwargs: Any,
    ) -> ExplainerResult:
        """
        Generate explanation for the given chat using hierarchical SHAP approach.

        Args:
            chat: The chat instance to explain.
            generation_kwargs: Additional generation arguments.
            progress_bar: Whether to show a progress bar during explanation.
            first_layer_explanation_kwargs: Additional explanation
                arguments for the first layer explainer, if used.
            verbose: Whether to save computation graph details.
            explanation_kwargs: Additional explanation arguments to main SHAP explainer.
        Returns:
            An ExplainerResult containing the explanation results.
        """

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
        self.n_calls = 0
        self.computation_graph = None

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
            normalized_shap_values, self.computation_graph = self.__compute(
                chat=chat,
                response=response,
                group_mask=chat.shap_values_mask,
                generation_kwargs=generation_kwargs,
                _verbose=verbose,
                **explanation_kwargs,
            )
        else:
            normalized_shap_values = self.__handle_with_first_level(
                chat=chat,
                response=response,
                generation_kwargs=generation_kwargs,
                first_layer_explanation_kwargs=first_layer_explanation_kwargs,
                _verbose=verbose,
                **explanation_kwargs,
            )

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
            or roles will be assigned separate groups even if contiguous. Tokens
            where mask is False will have group ID 0.
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
        prev_modality = torch.cat([torch.tensor([modality_flag[0]], device=device), modality_flag[:-1]])

        # Start new group if:
        # - token is explainable
        # - AND (previous not explainable OR modality changed OR role changed (if `include_role`))
        group_mask = ~prev_mask | (modality_flag != prev_modality)
        if include_role:
            prev_role = torch.cat([torch.tensor([token_roles[0]], device=device), token_roles[:-1]])
            group_mask |= token_roles != prev_role
        group_start = mask & group_mask

        explainable_starts = group_start[mask]
        cumulative = torch.cumsum(explainable_starts.int(), dim=0)

        # Map back to full-length tensor
        group_ids[mask] = cumulative
        group_ids[~mask] = 0  # keep masked-out tokens at 0

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

    @staticmethod
    def __extract_normalized_shap_values(
        response: ModelResponse,
    ) -> Tensor:
        """
        Extract normalized SHAP values from the response cache.

        Args:
            response: The model response containing the cache.
        Returns:
            A tensor containing the normalized SHAP values for explainable tokens.
        """
        cache = cast(ExplainerCache, response.chat.cache)  # type: ignore[union-attr]
        return cache.normalized_values[: cache.n]  # do not return for response tokens
