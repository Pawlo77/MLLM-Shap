"""Neyman state-machine sampling strategy for Neyman complementary sampling."""

from collections.abc import Callable
from logging import Logger

import torch
from torch import Tensor

from ...utils.logger import get_logger
from ..core.contracts import SamplingStrategy

logger: Logger = get_logger(__name__)


class NeymanStateMachineSamplingStrategy(SamplingStrategy):
    """State-machine strategy used by Neyman complementary sampling."""

    def __init__(
        self,
        get_num_splits: Callable[[int], int],
        get_random_split: Callable[..., Tensor],
        get_m_matrix: Callable[[], Tensor | None],
        get_m_hat: Callable[[], Tensor | None],
        update_m_position: Callable[[], bool],
        get_initial_num_splits: Callable[[], int],
        get_use_standard_method: Callable[[], bool],
        get_first_call: Callable[[], bool],
        set_first_call: Callable[[bool], None],
        get_step: Callable[[], int],
        set_step: Callable[[int], None],
        get_i: Callable[[], int],
        set_i: Callable[[int], None],
        get_j: Callable[[], int],
        set_j: Callable[[int], None],
        initial_step: int,
        allocation_step: int,
    ) -> None:
        """Initialize the Neyman state-machine strategy.

        Args:
            get_num_splits: Callable returning the total number of splits.
            get_random_split: Callable generating a random split for a coalition size.
            get_m_matrix: Callable returning the current initial-phase count matrix.
            get_m_hat: Callable returning the current allocation vector.
            update_m_position: Callable advancing the initial-phase matrix cursor.
            get_initial_num_splits: Callable returning per-cell initial allocation.
            get_use_standard_method: Callable indicating whether standard sampling is active.
            get_first_call: Callable returning whether the state machine is on its first call.
            set_first_call: Callable updating the first-call flag.
            get_step: Callable returning the current state-machine step.
            set_step: Callable updating the current state-machine step.
            get_i: Callable returning the current feature index.
            set_i: Callable updating the current feature index.
            get_j: Callable returning the current coalition-size index.
            set_j: Callable updating the current coalition-size index.
            initial_step: Value representing the initial sampling phase.
            allocation_step: Value representing the Neyman allocation phase.
        """
        self._get_num_splits = get_num_splits
        self._get_random_split = get_random_split
        self._get_m_matrix = get_m_matrix
        self._get_m_hat = get_m_hat
        self._update_m_position = update_m_position
        self._get_initial_num_splits = get_initial_num_splits
        self._get_use_standard_method = get_use_standard_method
        self._get_first_call = get_first_call
        self._set_first_call = set_first_call
        self._get_step = get_step
        self._set_step = set_step
        self._get_i = get_i
        self._set_i = set_i
        self._get_j = get_j
        self._set_j = set_j
        self._initial_step = initial_step
        self._allocation_step = allocation_step

    def get_num_splits(self, n: int) -> int | None:
        """Return the expected total number of splits for ``n`` features."""
        return self._get_num_splits(n)

    def get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        """Advance the Neyman state machine and return the next split mask.

        Args:
            n: Number of explainable features.
            device: Device on which the split should be created.
            generated_masks_num: Number of masks generated so far.
            existing_masks: Existing masks, unused by this strategy.

        Returns:
            The next split mask, or ``None`` when the current stage is exhausted.
        """
        del existing_masks
        m = self._get_m_matrix()
        if m is None:
            raise RuntimeError("M matrix must be initialized before sampling.")

        if self._get_step() == self._initial_step:
            logger.debug(
                "Min %f, Sum %f, Zero count %d",
                m.min().item(),
                m.sum().item(),
                (m == 0).sum().item(),
            )

            if generated_masks_num >= self._get_num_splits(n=n):
                logger.warning("Initial sampling exceeded total number of splits.")
                return None

            if (not self._get_first_call()) and self._update_m_position():
                logger.debug("Moving to Neyman allocation step.")
                self._set_step(self._allocation_step)
                return None
            self._set_first_call(False)

            j = self._get_j()
            i = self._get_i()
            if self._get_use_standard_method():
                return self._get_random_split(
                    n=n,
                    device=device,
                    true_values_num=j,
                )

            if not m[i, j] < self._get_initial_num_splits():
                raise RuntimeError(
                    "__update_M_position did not update position correctly."
                )

            new_mask = self._get_random_split(
                n=n,
                device=device,
                true_values_num=j,
                include_token=i if j > 0 else None,
            )
            if j > 0 and not new_mask.squeeze()[i]:
                raise RuntimeError(
                    "Generated mask does not include the required token."
                )
            return new_mask

        m_hat = self._get_m_hat()
        if m_hat is None:
            raise RuntimeError("M_hat matrix must be initialized before sampling.")

        j = self._get_j()
        if j == m_hat.shape[0]:
            return None

        while j < m_hat.shape[0]:
            if m_hat[j] > 0:
                new_mask = self._get_random_split(
                    n=n,
                    device=device,
                    true_values_num=j,
                )
                m_hat[j] -= 1
                return new_mask
            j += 1
            self._set_j(j)

        return None
