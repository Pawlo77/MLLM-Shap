"""Precise SHAP explainer implementation."""

from itertools import product
from typing import Generator

import torch
from torch import Tensor

from .base.shap_explainer import BaseShapExplainer


# pylint: disable=too-few-public-methods
class PreciseShapExplainer(BaseShapExplainer):
    """Precise SHAP implementation generating all possible masks."""

    __splits_generator: Generator[Tensor, None, None] | None = None

    def _get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        if self._first_call:
            self._first_call = False
            self.__splits_generator = PreciseShapExplainer.__get_splits_generator(
                n=n,
                device=device,
            )
        if self.__splits_generator is None:
            raise RuntimeError("Splits generator is not present.")

        try:
            return next(self.__splits_generator)
        except StopIteration:
            return None

    def _get_num_splits(self, n: int) -> int:
        return int(2**n - 1)  # exclude all-true mask

    # pylint: disable=too-many-locals
    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        num_features = masks.shape[1]
        shap_values = torch.zeros(num_features, dtype=similarities.dtype, device=device)

        # Precompute factorial terms for efficiency
        # using formula a! = (a - 1)! * a
        indices = torch.arange(num_features + 1, dtype=torch.float32, device=device)
        indices[0] = 1.0
        factorials = torch.cumprod(indices, dim=0)

        # Precompute hash values for all subsets
        subset_hashes = (masks * (2 ** torch.arange(num_features, device=device))).sum(dim=1)
        sorted_hashes, sort_idx = subset_hashes.sort()
        sorted_outputs = similarities[sort_idx]

        # Precompute subset sizes
        subset_sizes = masks.sum(dim=1)

        # formula: \phi_i = \sum_{S ⊆ N \ {i}} [ |S|! * (|N| - |S| - 1)! / |N|! * (f(S ∪ {i}) - f(S)) ]
        for i in range(num_features):
            # Select subsets that include feature i
            include_mask = masks[:, i]

            # All subsets that include i - IN = {S : i ∈ S}
            included_subsets = masks[include_mask]
            included_outputs = similarities[include_mask]  # f(IN)

            # Corresponding subsets with i removed - OUT = {S \ {i} : S ∈ IN}
            excluded_subsets = included_subsets.clone()
            excluded_subsets[:, i] = False
            excluded_hash = (excluded_subsets * (2 ** torch.arange(num_features, device=masks.device))).sum(dim=1)
            excluded_outputs = sorted_outputs[torch.searchsorted(sorted_hashes, excluded_hash)]  # f(OUT)

            # Corresponding subset sizes - |S| for S ∈ OUT
            excluded_subset_sizes = subset_sizes[include_mask] - 1

            weights = (
                factorials[excluded_subset_sizes]
                * factorials[num_features - excluded_subset_sizes - 1]
                / factorials[num_features]
            )
            shap_values[i] = torch.sum(weights * (included_outputs - excluded_outputs))

        return shap_values

    @staticmethod
    def __get_splits_generator(n: int, device: torch.device) -> Generator[Tensor, None, None]:
        """
        Generates all possible binary masks of a given length, excluding the all-ones mask.

        Args:
            n (int): The length of the binary masks to generate.
            device (torch.device): The device on which to create the tensors.
        Yields:
            Tensor: A binary mask tensor of shape (1, n).
        """
        for split in product([0, 1], repeat=n):
            split_tensor = torch.tensor(split, dtype=torch.bool, device=device)
            if split_tensor.sum() == n:
                continue
            yield split_tensor
