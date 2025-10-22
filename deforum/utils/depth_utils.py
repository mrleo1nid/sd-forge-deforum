"""Pure functions for depth tensor processing and normalization.

This module contains depth-related pure functions extracted from
scripts/deforum_helpers/animation.py, following functional programming principles
with no side effects.
"""

import numpy as np
import torch
from typing import Tuple


def normalize_depth_tensor(depth_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize depth tensor to 0-1 range.

    Args:
        depth_tensor: Raw depth tensor (H, W)

    Returns:
        Normalized depth tensor with values in [0, 1]

    Raises:
        ValueError: If depth tensor has no range (min == max)

    Examples:
        >>> t = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> result = normalize_depth_tensor(t)
        >>> result.min().item()
        0.0
        >>> result.max().item()
        1.0
    """
    if depth_tensor is None:
        raise ValueError("Depth tensor cannot be None")

    depth_min = depth_tensor.min()
    depth_max = depth_tensor.max()
    depth_range = depth_max - depth_min

    if depth_range == 0:
        raise ValueError(
            f"Depth tensor has no range (min={depth_min}, max={depth_max}). "
            "Cannot normalize."
        )

    return (depth_tensor - depth_min) / depth_range


def equalize_depth_histogram(depth_tensor: torch.Tensor, bins: int = 1024) -> torch.Tensor:
    """Perform histogram equalization on depth tensor.

    Applies histogram equalization to enhance depth contrast. This is useful
    for improving visual quality of depth maps by redistributing depth values
    to use the full [0, 1] range more effectively.

    Args:
        depth_tensor: Normalized depth tensor (H, W) with values in [0, 1]
        bins: Number of histogram bins for equalization (default: 1024)

    Returns:
        Equalized depth tensor (H, W) with improved contrast

    Examples:
        >>> t = torch.tensor([[0.0, 0.1], [0.2, 1.0]])
        >>> result = equalize_depth_histogram(t)
        >>> result.shape == t.shape
        True
        >>> 0.0 <= result.min() <= result.max() <= 1.0
        True
    """
    if depth_tensor is None:
        raise ValueError("Depth tensor cannot be None")

    if bins < 2:
        raise ValueError(f"Number of bins must be at least 2, got {bins}")

    # Convert to numpy for histogram operations
    depth_array = depth_tensor.cpu().numpy()

    # Calculate histogram
    hist, bin_edges = np.histogram(depth_array, bins=bins, range=(0, 1))

    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Normalize CDF to [0, 1]
    cdf_max = cdf[-1]
    if cdf_max == 0:
        # All values are the same, return as-is
        return depth_tensor

    cdf_normalized = cdf / float(cdf_max)

    # Map original values to equalized values using CDF
    equalized_array = np.interp(depth_array, bin_edges[:-1], cdf_normalized)

    # Convert back to torch tensor on original device with original dtype
    return torch.from_numpy(equalized_array).to(
        device=depth_tensor.device, dtype=depth_tensor.dtype
    )


def prepare_depth_tensor(depth_tensor: torch.Tensor, bins: int = 1024) -> torch.Tensor:
    """Prepare depth tensor with normalization and equalization.

    Combines normalization (to [0, 1] range) and histogram equalization
    for optimal depth representation in 3D warping operations.

    Args:
        depth_tensor: Raw depth tensor (H, W)
        bins: Number of histogram bins for equalization (default: 1024)

    Returns:
        Processed depth tensor (H, W) ready for 3D transformations

    Raises:
        ValueError: If depth tensor is None or has no range

    Examples:
        >>> t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> result = prepare_depth_tensor(t)
        >>> result.shape == t.shape
        True
        >>> 0.0 <= result.min() <= result.max() <= 1.0
        True
    """
    normalized = normalize_depth_tensor(depth_tensor)
    return equalize_depth_histogram(normalized, bins=bins)


def get_depth_min_max_formatted(
    depth_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """Calculate depth tensor min/max values with formatted strings.

    Args:
        depth_tensor: Depth tensor (H, W)

    Returns:
        Tuple of (min_value, max_value, formatted_min, formatted_max)
        where formatted values are strings with 2 decimal places

    Examples:
        >>> t = torch.tensor([[1.234, 5.678]])
        >>> min_val, max_val, min_str, max_str = get_depth_min_max_formatted(t)
        >>> float(min_val)
        1.234000...
        >>> float(max_val)
        5.678000...
        >>> min_str
        ' 1.23'
        >>> max_str
        ' 5.68'
    """
    if depth_tensor is None:
        raise ValueError("Depth tensor cannot be None")

    depth_min = depth_tensor.min()
    depth_max = depth_tensor.max()

    # Format as strings with width 5 and 2 decimal places
    formatted_min = "{:5.2f}".format(float(depth_min))
    formatted_max = "{:5.2f}".format(float(depth_max))

    return depth_min, depth_max, formatted_min, formatted_max
