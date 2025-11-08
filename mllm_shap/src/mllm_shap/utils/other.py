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
