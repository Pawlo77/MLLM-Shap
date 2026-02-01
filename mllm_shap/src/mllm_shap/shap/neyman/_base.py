# Similar to :class:`BaseShapExplainer`
# pylint: disable=duplicate-code
"""Base Complementary Neyman SHAP explainer implementation."""

from enum import Enum
import gc
from logging import Logger
import math
from typing import Any, cast
from functools import lru_cache

import torch
from torch import Tensor

from ...connectors.base.chat import BaseMllmChat
from ...connectors.enums import SystemRolesSetup, Role
from ...connectors.base.model import BaseMllmModel
from ...connectors.base.model_response import ModelResponse
from ...utils.logger import get_logger
from ..base._cache_manager import CacheManager
from ..base._masks_manager import MasksManager
from ..base._validators import BaseShapCallConfig
from ..base.complementary import BaseComplementaryShapApproximation

logger: Logger = get_logger(__name__)


class _Step(int, Enum):
    """Steps in the Neyman allocation process."""

    INITIAL_SAMPLING = 1
    """Initial sampling step."""

    NEYMAN_ALLOCATION = 2
    """Neyman allocation step."""


# pylint: disable=too-few-public-methods,invalid-name,too-many-instance-attributes
class BaseComplementaryNeymanShapExplainer(BaseComplementaryShapApproximation):
    """
    Base Complementary Neyman SHAP implementation class

    Budget should be doubled in comparison to paper to account for symmetric sampling.

    Info:
        Initial step in this explainer due to performance and stability issues is not
        purely random sampling - random samples are drawn with pre-defined member
        within them. This guarantees that initialization step will cover all entries
        within `(number of players) * (number of players + 1) * initial_num_samples` calls.
        If no `:attr:initial_num_samples` or `:attr:initial_fraction` is provided,
        a default formula is used - `max(2, ceil(total_num_splits / (2 * n * n)))`.

    Warning:
        This explainer requires the source chat to have at least one non-user turn
        (:class:`Role.ASSISTANT` or :class:`Role.SYSTEM`).
        It is due to its requirement to evaluate all-zeros and all-ones splits, which
        otherwise would be rejected by connectors as invalid. It requires at least
        one turn for assistant and some system messages to be present.
    """

    initial_num_samples: int | None
    """Initial number of samples to draw in the first step."""

    initial_fraction: float | None
    """Initial fraction of samples to draw in the first step."""

    initial_steps: int | None
    """Number of initial steps performed in last call."""

    use_standard_method: bool = False
    """
    Whether to use the standard method for initial sampling.
    Default is False, which uses the modified method with pre-defined members.
    """

    __use_default_initial_sampling_formula: bool = False
    """
    Whether to use default formula for initial sampling
    Set when both `initial_num_samples` and `initial_fraction` are None.
    """

    __initial_num_splits: int
    """Number of initial splits for each entry of matrix M."""

    __C_squared: Tensor | None
    """
    C squared matrix for Neyman allocation -
    C[i, j] =sum of squared complementary contributions
        for feature i in coalitions of size j+1.
    """

    __M_hat: Tensor | None
    """
    M_hat matrix for Neyman allocation,
    holding number of samples to be allocated
    of size i + 1 at index i.
    """

    __step: int
    """Steps in the Neyman allocation process."""

    __i: int
    __j: int
    """Indices for tracking position in the _M matrix."""

    def __init__(
        self,
        *args: Any,
        initial_num_samples: int | None = None,
        initial_fraction: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the BaseComplementaryNeymanShapExplainer.

        Args:
            args: Positional arguments for the base class.
            initial_num_samples: Initial number of samples to draw in the first step.
            initial_fraction: Initial fraction of samples to draw in the first step.
            kwargs: Keyword arguments for the base class.
        Raises:
            ValueError: If sampling parameters are invalid.
        """
        kwargs["allow_mask_duplicates"] = True
        super().__init__(*args, **kwargs)

        if initial_num_samples is None and initial_fraction is None:
            logger.debug("Using default initial sampling formula.")
            self.__use_default_initial_sampling_formula = True
        else:
            self._validate_sampling_params(
                num_samples=initial_num_samples,
                fraction=initial_fraction,
            )

        self.initial_num_samples = initial_num_samples
        self.initial_fraction = initial_fraction

    def _initialize_state(self) -> None:
        """
        Initialize internal state before starting mask generation.
        """
        super()._initialize_state()
        self.__get_start.cache_clear()

        self.__step = _Step.INITIAL_SAMPLING
        self.__i = 0
        self.__j = 0
        self.__C_squared = None
        self.initial_steps = 0

    @lru_cache(maxsize=1)
    def _get_num_splits(self, n: int) -> int:
        """
        Get total number of splits to generate, as well
        as number of initial splits for each entry of matrix M.
        It determines initial step duration.

        Args:
            n: Length of the splits
        Returns:
            Number of masks to generate.
        """

        # determine total number of splits
        try:
            num_splits = super()._get_num_splits(n=n)
        except ValueError as e:
            raise ValueError("Total number of splits could not be determined.") from e

        # determine initial number of splits per entry in M
        try:
            if self.__use_default_initial_sampling_formula:
                initial_num_splits = max(2, math.ceil(num_splits / (2 * n * n)))
            else:
                initial_num_splits = BaseComplementaryShapApproximation._get_num_splits_static(
                    n=n,
                    num_samples=self.initial_num_samples,
                    fraction=self.initial_fraction,
                    force_minimal=False,
                )
        except ValueError as e:
            raise ValueError("Initial number of splits could not be determined.") from e

        # validate initial number of splits
        if initial_num_splits < 2:  # pylint: disable=magic-value-comparison
            logger.warning(
                "Initial number of splits %d is less than 2. Setting it to 2.",
                initial_num_splits,
            )
            initial_num_splits = 2

        expected_initial_budget = initial_num_splits * n * (n + 1)
        if expected_initial_budget > num_splits:
            logger.warning(
                "Estimated initial budget %d is larger than total number of splits %d. "
                "This may lead to suboptimal performance.",
                expected_initial_budget,
                num_splits,
            )
        if initial_num_splits > math.ceil(num_splits * 0.2):
            logger.warning(
                "Initial number of splits %d is more than 20%% of total number of splits %d. "
                "This may lead to suboptimal performance.",
                initial_num_splits,
                num_splits,
            )
        self.__initial_num_splits = initial_num_splits

        return num_splits

    # pylint: disable=too-many-return-statements
    def _get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        if self._M is None:
            raise RuntimeError("M matrix must be initialized before sampling.")

        if self.__step == _Step.INITIAL_SAMPLING:  # initial sampling
            logger.debug(
                "Min %f, Sum %f, Zero count %d",
                self._M.min().item(),
                self._M.sum().item(),
                (self._M == 0).sum().item(),
            )

            if generated_masks_num >= self._get_num_splits(n=n):
                logger.warning("Initial sampling exceeded total number of splits.")
                return None

            if not self._first_call and self.__update_M_position():  # stopping condition
                logger.debug("Moving to Neyman allocation step.")
                self.__step = _Step.NEYMAN_ALLOCATION
                return None
            self._first_call = False

            if self.use_standard_method:
                return self._get_random_split(
                    n=n,
                    device=device,
                    true_values_num=self.__j,
                )

            # our modified method with pre-defined members
            if not self._M[self.__i, self.__j] < self.__initial_num_splits:
                raise RuntimeError("__update_M_position did not update position correctly.")

            # `self.__j == 0` --> include no tokens
            # generate split of size `self.__j` with required token `self.__i`
            new_mask = self._get_random_split(
                n=n,
                device=device,
                true_values_num=self.__j,
                include_token=self.__i if self.__j > 0 else None,
            )
            if self.__j > 0 and not new_mask.squeeze()[self.__i]:
                raise RuntimeError("Generated mask does not include the required token.")

            return new_mask

        if self.__M_hat is None:
            raise RuntimeError("M_hat matrix must be initialized before sampling.")
        # dont end on total budget exceeded here, as it might slightly differ
        # from one estimated with `self.__M_hat`. This difference should be minimal.
        if self.__j == self.__M_hat.shape[0]:  # end of sampling
            return None

        # generate split of size `self.__j`
        if self.__M_hat[self.__j] > 0:
            new_mask = self._get_random_split(
                n=n,
                device=device,
                true_values_num=self.__j,
            )
            self.__M_hat[self.__j] -= 1
            return new_mask

        self.__j += 1
        return self._get_next_split(
            n=n,
            device=device,
            generated_masks_num=generated_masks_num,
            existing_masks=existing_masks,
        )

    def _get_masks_generator(self, *args: Any, **kwargs: Any) -> Any:
        # won't cause issues as we force `Role.ASSISTANT` presence in chat
        kwargs["allow_full_or_empty"] = True
        return super()._get_masks_generator(*args, **kwargs)

    def _calculate_C_matrix(self, masks: Tensor, similarities: Tensor, device: torch.device) -> None:
        """Overload to also calculate C squared matrix."""
        if self._M is None:
            raise RuntimeError("M matrix must be initialized before calculating C matrix.")
        if self._C is None or self.__C_squared is None:
            self._C = torch.zeros_like(self._M, dtype=similarities.dtype, device=device)
            self.__C_squared = torch.zeros_like(self._M, dtype=similarities.dtype, device=device)

        m = masks.shape[0] // 2
        if 2 * m != masks.shape[0]:
            raise ValueError("Masks should be in complementary pairs.")

        # change: include C squared calculation within same loop
        for i in range(m):
            if not torch.all(masks[2 * i] == ~masks[2 * i + 1]):
                raise ValueError("Masks are not complementary pairs.")

            S = masks[2 * i]
            NS = masks[2 * i + 1]  # complement of S
            s_size = int(S.sum().item())
            ns_size = masks.shape[1] - s_size

            u = similarities[2 * i] - similarities[2 * i + 1]
            u_squared = u * u

            BaseComplementaryShapApproximation._increment_coalition_val(self._C, S, s_size, u)
            BaseComplementaryShapApproximation._increment_coalition_val(self._C, NS, ns_size, -u)
            BaseComplementaryShapApproximation._increment_coalition_val(self.__C_squared, S, s_size, u_squared)
            BaseComplementaryShapApproximation._increment_coalition_val(self.__C_squared, NS, ns_size, u_squared)

    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        if not self._zero_mask_skipped:
            raise RuntimeError("Zero mask was not skipped during mask generation.")
        if self._M is None or self._C is None:
            raise RuntimeError("M and C matrices must be initialized before calculating SHAP values.")

        # exclude zero-mask column
        M = self._M[:, 1:]
        C = self._C[:, 1:]

        positive_mask = M > 0
        if self.__step == _Step.NEYMAN_ALLOCATION:
            if not torch.all(positive_mask):
                raise RuntimeError(
                    "Some entries in M matrix are zero. They are all expected to be >= `initial_num_splits`."
                )
            return torch.sum(C / M, dim=1) / M.shape[0]

        # it is not guaranteed especially with small budget
        non_zero_mask = M > 0
        ratio = torch.zeros_like(C)
        ratio[non_zero_mask] = C[non_zero_mask] / M[non_zero_mask]
        return torch.sum(ratio, dim=1) / M.shape[0]

    @lru_cache(maxsize=1)
    def __get_start(self) -> int:
        """
        Returns:
            Starting index in matrix M for Neyman allocation (step _Step.NEYMAN_ALLOCATION).
        Raises:
            ValueError: If matrix M is not initialized.
        """
        if self._M is None:
            raise ValueError("Matrix M is not initialized.")
        return math.ceil(self._M.shape[0] / 2)

    def __update_M_position(self) -> bool:
        """
        Update position in matrix M during `_Step.INITIAL_SAMPLING` step.
        If no more positions are left, returns True to indicate moving to next step.

        Returns:
            Boolean indicating if a full pass on M was completed.
        Raises:
            RuntimeError: If matrix M is not initialized.
        """
        if self._M is None:
            raise RuntimeError("M matrix must be initialized before updating position.")

        not_completed = self._M < self.__initial_num_splits
        if not torch.any(not_completed):  # stop initial and move to next step
            return True

        # Given current (i, j) and a boolean M[n, n], find the next (i, j)
        # position where mask is True, scanning row by row.
        flat = not_completed.flatten()
        start_idx = self.__i * self._M.shape[1] + self.__j + 1
        next_idxs = torch.nonzero(flat[start_idx:], as_tuple=False)
        if next_idxs.numel() == 0:  # next pass
            next_idxs = torch.nonzero(flat, as_tuple=False)
            start_idx = 0

        next_idx = next_idxs[0, 0].item() + start_idx
        self.__i, self.__j = cast(tuple[int, int], divmod(next_idx, self._M.shape[1]))
        return False

    def __estimate_sigma_squared(self) -> Tensor:
        """
        Estimate variance (sigma squared) for each entry in matrix M.

        Returns:
            Tensor of same shape and properties as self._C with estimated variances.
        Raises:
            RuntimeError: If M or C matrices are not initialized.
        """
        if self._M is None or self._C is None or self.__C_squared is None:
            raise RuntimeError("M, C and C_squared matrices must be initialized before estimating sigma squared.")

        M = self._M.to(self._C.dtype)
        M_small = M - 1
        sigma = (self.__C_squared - torch.pow(self._C, 2) / M) / M_small
        if torch.any(sigma < 0):  # should never happen, but numerical issues might cause it
            logger.warning("Negative variance estimates found; setting them to zero.")
            sigma = torch.clamp(sigma, min=0.0)

        return sigma

    def __estimate_M_hat(self, n: int) -> None:
        """
        Estimate M_hat matrix for Neyman allocation.

        Args:
            n: Number of features / tokens.
        Raises:
            RuntimeError: If M or C matrices are not initialized.
        """
        if self._M is None or self._C is None:
            raise RuntimeError("M and C matrices must be initialized before estimating M_hat.")

        # / 2 as each sample covers two entries symmetrically
        m = (self._get_num_splits(n=n) - self.total_n_calls) / 2
        if int(m) != m:
            raise RuntimeError("Remaining samples for Neyman allocation must be odd.")
        m = int(m)

        sigma_squared_hat = self.__estimate_sigma_squared()
        logger.debug("Sigma squared estimates: %s", sigma_squared_hat)

        left = self.__get_start()
        right = sigma_squared_hat.shape[0]

        # Indices for the right half
        k_vec = torch.arange(left, right, device=sigma_squared_hat.device)  # indexes
        k_left = k_vec
        k_right = right - k_vec - 1
        sigma_left = sigma_squared_hat[:, k_left]  # shape (n, r - l)
        sigma_right = sigma_squared_hat[:, k_right]  # shape (n, r - l)
        inner = torch.sqrt(
            torch.sum(sigma_left / (k_left + 1).to(sigma_left.dtype), dim=0)
            + torch.sum(sigma_right / (k_right + 1).to(sigma_right.dtype), dim=0)
        )

        self.__M_hat = torch.zeros(self._M.shape[0], dtype=inner.dtype, device=inner.device)
        self.__M_hat[left:right] = torch.ceil((m / inner.sum()) * inner)
        logger.debug("M hat %s", self.__M_hat)

    # pylint: disable=signature-differs
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    def __call__(  # type: ignore[override]
        self,
        model: BaseMllmModel,
        source_chat: BaseMllmChat,
        response: ModelResponse,
        progress_bar: bool = True,
        verbose: bool = False,
        **generate_kwargs: Any,
    ) -> list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None:
        generate_kwargs.pop("n_generator_jobs", None)  # not parallelizable
        __config = BaseShapCallConfig(
            model=model,
            source_chat=source_chat,
            response=response,
            progress_bar=progress_bar,
            verbose=verbose,
        )
        self._initialize_state()

        # validated within BaseShapCallConfig
        response_chat: BaseMllmChat = __config.response.chat  # type: ignore[assignment]
        source_chat = __config.source_chat
        device = source_chat.torch_device

        # additional validations guaranteeing no masks will be rejected by chat/model
        if source_chat.system_roles_setup != SystemRolesSetup.SYSTEM_ASSISTANT:
            raise ValueError("Source chat must have SYSTEM_ASSISTANT roles setup for Neyman SHAP.")
        # cant check for SYSTEM role alone, as some models might use it for steering tokens
        if Role.ASSISTANT not in source_chat.token_roles:
            logger.warning(
                "Source chat must have at least one non-user message for Neyman SHAP."
                "No assistant role found, make sure that existing messages cover it."
            )

        mask_manager = MasksManager(chat=source_chat, log_stats=True)
        cache_manager = CacheManager(
            chat=response_chat,
            explainer_hash=hash(self),
        )

        masks = [mask_manager.get_initial_mask(device=device)]
        responses = [__config.response]

        # fist step - initial sampling
        _ = self._get_num_splits(n=mask_manager.n)  # calculate _initial_num_splits
        logger.info(
            "Starting initial sampling step with %d samples per entry in M",
            self.__initial_num_splits,
        )
        chats_skipped, history = self._generate_step(
            mask_manager=mask_manager,
            masks=masks,
            device=device,
            responses=responses,
            source_chat=source_chat,
            model=__config.model,
            cache_manager=cache_manager,
            n_generator_jobs=1,
            progress_bar=__config.progress_bar,
            verbose=__config.verbose,
            **generate_kwargs,
        )

        masks_tensor = torch.stack(masks, dim=0)
        similarities = self._get_similarities(responses=responses, model=model)
        self._calculate_C_matrix(
            masks=masks_tensor[1:, source_chat.shap_values_mask],  # exclude initial all-ones mask
            similarities=similarities[1:],
            device=device,
        )

        self.initial_steps = self.total_n_calls
        logger.debug("Initial sampling step completed with %d calls.", self.initial_steps)

        # otherwise initial sampling exceeded entire budget
        if self.__step == _Step.NEYMAN_ALLOCATION:
            # prepare for step 2
            self.__estimate_M_hat(n=mask_manager.n)
            self.__j = self.__get_start()
            self.__i = 0
            existing_masks_num = len(masks)

            # second step - Neyman allocation
            logger.info(
                "Starting Neyman allocation step with %d remaining samples",
                cast(Tensor, self.__M_hat).sum().item(),
            )
            new_chats_skipped, new_history = self._generate_step(
                mask_manager=mask_manager,
                masks=masks,
                device=device,
                responses=responses,
                source_chat=source_chat,
                model=__config.model,
                cache_manager=cache_manager,
                n_generator_jobs=1,
                progress_bar=__config.progress_bar,
                verbose=__config.verbose,
                **generate_kwargs,
            )

            new_masks_tensor = torch.stack(masks[existing_masks_num:], dim=0)
            # pass initial mask response as well, but don't store its similarity
            new_similarities = self._get_similarities(
                responses=[responses[0]] + responses[existing_masks_num:], model=model
            )[1:]
            # update C matrix with new masks
            self._calculate_C_matrix(
                masks=new_masks_tensor[..., source_chat.shap_values_mask],
                similarities=new_similarities,
                device=device,
            )

            # edge case from :class:`BaseShapExplainer` does not apply here
            # as we have :attr:`_initial_num_splits` >= 1

            # merge results
            chats_skipped += new_chats_skipped
            if history is not None and new_history is not None:
                history += new_history
            masks_tensor = torch.cat((masks_tensor, new_masks_tensor), dim=0)
            similarities = torch.cat((similarities, new_similarities), dim=0)

            # clean up
            del new_chats_skipped
            del new_history
            del new_masks_tensor
            del new_similarities

        if cache_manager.extracted_num > 0:
            logger.info(
                "Deduplicated %d/%d masks using existing cache.",
                cache_manager.extracted_num,
                len(masks) - 1,  # exclude base mask
            )

        del mask_manager
        del cache_manager
        del masks
        gc.collect()

        # calculate SHAP values (relative to source_chat)
        shap_values, normalized_shap_values = self._get_shap_values(
            model=__config.model,
            masks=masks_tensor,
            responses=responses,
            source_chat=source_chat,
            device=device,
            similarities=similarities,
        )

        # cache results
        self._save_to_cache(
            chat=response_chat,
            source_chat=source_chat,
            responses=responses,
            masks=masks_tensor,
            shap_values=shap_values,
            normalized_shap_values=normalized_shap_values,
        )

        return history
