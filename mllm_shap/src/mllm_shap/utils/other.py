"""General utility functions."""

from typing import Any, Callable

import torch
from torch import Tensor


def raise_connector_error(callable_: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Wrapper to raise connector errors with more context.

    Args:
        callable_: The callable to wrap.
        *args: Positional arguments for the callable.
        **kwargs: Keyword arguments for the callable.
    Returns:
        The result of the callable.
    Raises:
        RuntimeError: If an error occurs in the callable.
    """
    try:
        return callable_(*args, **kwargs)
    except Exception as e:
        raise RuntimeError("Error occurred in connector implementation.") from e


def safe_mask(tensor: Tensor, mask: Tensor) -> Tensor:
    """
    Mask the tensor with the given mask. If mask is
    empty, return empty Tensor while maintaining the original
    tensor properties.

    Args:
        tensor: The input tensor to be masked.
        mask: The boolean mask tensor.
    Returns:
        The masked tensor.
    """
    masked = tensor[..., mask]
    if masked.numel() == 0:
        target_shape = (tensor.shape[0], 0) if len(tensor.shape) > 1 else (tensor.shape[0],)
        masked = torch.empty(target_shape, device=tensor.device, dtype=tensor.dtype)
    return masked


def safe_mask_unsqueeze(tensor: Tensor, mask: Tensor) -> Tensor:
    """
    Mask the tensor with the given mask. If mask is
    empty, return empty Tensor while maintaining the original
    tensor properties, and unsqueeze to maintain batch dimension.

    Args:
        tensor: The input tensor to be masked.
        mask: The boolean mask tensor.
    Returns:
        The masked tensor with batch dimension.
    """
    masked = tensor[0][mask]
    if masked.numel() == 0:
        target_shape = (tensor.shape[0], 0) if len(tensor.shape) > 1 else (tensor.shape[0],)
        masked = torch.empty(target_shape, device=tensor.device, dtype=tensor.dtype)
    else:
        masked = masked.unsqueeze(0)
    return masked


def make_consecutive_ids_ignore_zero(t: torch.Tensor) -> torch.Tensor:
    """
    Renumber non-zero tensor values to consecutive integers starting from 1,
    preserving the order of first appearance. Zeros remain unchanged.

    Args:
        t: Input tensor with integer IDs.
    Returns:
        Tensor with renumbered IDs.
    """
    # Get unique consecutive non-zero values in order of appearance
    nonzero_mask = t != 0
    unique_vals = torch.unique_consecutive(t[nonzero_mask])

    # Map original IDs to new consecutive ones
    mapping = {v.item(): i + 1 for i, v in enumerate(unique_vals)}

    # Apply mapping (zeros unchanged)
    out = t.clone()
    for old_id, new_id in mapping.items():
        out[t == old_id] = new_id

    return out


def extend_tensor(t: Tensor, target_length: int, fill_value: Any) -> Tensor:
    """
    Extend a tensor to the target length by appending the fill value.

    Args:
        t: The input tensor to be extended.
        target_length: The desired length of the output tensor.
        fill_value: The value to use for extension.
    Returns:
        The extended tensor.
    """
    current_length = t.shape[0]
    if current_length >= target_length:
        return t

    extension_size = target_length - current_length
    extension = torch.full(
        (extension_size,),
        fill_value,
        dtype=t.dtype,
        device=t.device,
    )
    extended_tensor = torch.cat([t, extension], dim=0)
    return extended_tensor
