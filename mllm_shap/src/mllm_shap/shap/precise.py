"""Precise SHAP explainer implementation."""

from itertools import product

import torch
from torch import Tensor

from ._base.explainer import BaseSHAPExplainer


def generate_all_masks(n: int, device: torch.device) -> Tensor:
    """
    Generate all possible masks for n features, excluding all-ones mask.

    Args:
        n: Number of features.
        device: The device to create the masks on.
    Returns:
        A tensor of shape (2^n - 1, n) containing all possible masks
    """
    masks = list(product([0, 1], repeat=n))
    masks_tensor = torch.tensor(masks, dtype=torch.bool, device=device)

    # Drop all-ones masks
    keep_mask = masks_tensor.sum(dim=1) != n
    masks_tensor = masks_tensor[keep_mask]

    return masks_tensor


# pylint: disable=too-few-public-methods
class PreciseSHAPExplainer(BaseSHAPExplainer):
    """Precise SHAP implementation generating all possible masks."""

    def _generate_masks(self, n: int, device: torch.device, existing_masks: Tensor | None = None) -> Tensor:
        all_masks = generate_all_masks(n, device)

        if existing_masks is not None:
            # Convert to sets of tuples for fast comparison
            seen = {tuple(row.tolist()) for row in existing_masks}
            unseen_masks = [mask for mask in all_masks if tuple(mask.tolist()) not in seen]

            if unseen_masks:
                return torch.stack(unseen_masks, dim=0)
            return torch.empty((0, n), dtype=torch.bool, device=device)
        return all_masks

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
