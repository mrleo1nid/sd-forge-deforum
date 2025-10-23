"""Pure functions for image geometric transformations.

This module contains functions for cropping, padding, and other geometric
operations on images, following functional programming principles with no
side effects.
"""

import numpy as np
from typing import Tuple


def center_crop(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Crop image to specified dimensions from center.

    Args:
        image: Input image array (H x W x C)
        target_width: Desired output width
        target_height: Desired output height

    Returns:
        Center-cropped image array

    Raises:
        ValueError: If target dimensions are larger than image dimensions

    Examples:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100, 3)
        >>> cropped = center_crop(img, 50, 50)
        >>> cropped.shape
        (50, 50, 3)
        >>> img = np.random.rand(200, 300, 3)
        >>> cropped = center_crop(img, 100, 150)
        >>> cropped.shape
        (150, 100, 3)
    """
    height, width = image.shape[:2]

    if target_width > width or target_height > height:
        raise ValueError(
            f"Target dimensions ({target_width}x{target_height}) "
            f"exceed image dimensions ({width}x{height})"
        )

    # Calculate indents from center
    width_indent = int((width - target_width) / 2)
    height_indent = int((height - target_height) / 2)

    # Crop from center
    cropped = image[
        height_indent : height - height_indent,
        width_indent : width - width_indent
    ]

    return cropped


def calculate_center_crop_bounds(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int
) -> Tuple[int, int, int, int]:
    """Calculate bounding box for center crop.

    Args:
        image_width: Original image width
        image_height: Original image height
        target_width: Desired crop width
        target_height: Desired crop height

    Returns:
        Tuple of (left, top, right, bottom) coordinates

    Raises:
        ValueError: If target dimensions are larger than image dimensions

    Examples:
        >>> calculate_center_crop_bounds(100, 100, 50, 50)
        (25, 25, 75, 75)
        >>> calculate_center_crop_bounds(200, 100, 100, 50)
        (50, 25, 150, 75)
    """
    if target_width > image_width or target_height > image_height:
        raise ValueError(
            f"Target dimensions ({target_width}x{target_height}) "
            f"exceed image dimensions ({image_width}x{image_height})"
        )

    left = int((image_width - target_width) / 2)
    top = int((image_height - target_height) / 2)
    right = left + target_width
    bottom = top + target_height

    return (left, top, right, bottom)


def extend_with_grid(
    array: np.ndarray,
    target_width: int,
    target_height: int,
    offset_x: int = None,
    offset_y: int = None
) -> np.ndarray:
    """Extend 2D array to target dimensions with coordinate grid.

    Creates a new array of target size filled with coordinate meshgrid,
    then copies the original array into it at the specified offset.

    Args:
        array: Input array (H x W x 2) where last dim is (x, y)
        target_width: Desired output width
        target_height: Desired output height
        offset_x: X offset for placing original array (default: centered)
        offset_y: Y offset for placing original array (default: centered)

    Returns:
        Extended array with coordinate grid

    Examples:
        >>> import numpy as np
        >>> arr = np.ones((10, 10, 2))
        >>> extended = extend_with_grid(arr, 20, 20)
        >>> extended.shape
        (20, 20, 2)
    """
    array_h, array_w = array.shape[:2]

    # Calculate centered offsets if not provided
    if offset_x is None:
        offset_x = int((target_width - array_w) / 2)
    if offset_y is None:
        offset_y = int((target_height - array_h) / 2)

    # Create coordinate meshgrid
    x_grid, y_grid = np.meshgrid(np.arange(target_width), np.arange(target_height))

    # Create new array with grid
    extended = np.dstack((x_grid, y_grid)).astype(np.float32)

    # Shift original array values by offset
    array_shifted = array.copy()
    array_shifted[:, :, 0] += offset_x
    array_shifted[:, :, 1] += offset_y

    # Insert shifted array into extended array
    extended[offset_y:offset_y+array_h, offset_x:offset_x+array_w] = array_shifted

    return extended


def calculate_padding(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int
) -> Tuple[int, int, int, int]:
    """Calculate padding amounts for centering image in target dimensions.

    Args:
        image_width: Original image width
        image_height: Original image height
        target_width: Target width
        target_height: Target height

    Returns:
        Tuple of (top, bottom, left, right) padding amounts

    Examples:
        >>> calculate_padding(50, 50, 100, 100)
        (25, 25, 25, 25)
        >>> calculate_padding(100, 50, 200, 100)
        (25, 25, 50, 50)
    """
    pad_left = int((target_width - image_width) / 2)
    pad_right = target_width - image_width - pad_left
    pad_top = int((target_height - image_height) / 2)
    pad_bottom = target_height - image_height - pad_top

    return (pad_top, pad_bottom, pad_left, pad_right)


def get_crop_or_pad_bounds(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int
) -> dict:
    """Determine whether to crop or pad and calculate bounds.

    Args:
        image_width: Original image width
        image_height: Original image height
        target_width: Target width
        target_height: Target height

    Returns:
        Dictionary with keys:
            - 'operation': 'crop', 'pad', or 'mixed'
            - 'width_op': 'crop', 'pad', or 'none'
            - 'height_op': 'crop', 'pad', or 'none'
            - 'crop_bounds': (left, top, right, bottom) or None
            - 'padding': (top, bottom, left, right) or None

    Examples:
        >>> result = get_crop_or_pad_bounds(100, 100, 50, 50)
        >>> result['operation']
        'crop'
        >>> result = get_crop_or_pad_bounds(50, 50, 100, 100)
        >>> result['operation']
        'pad'
    """
    width_op = 'none'
    height_op = 'none'

    if image_width > target_width:
        width_op = 'crop'
    elif image_width < target_width:
        width_op = 'pad'

    if image_height > target_height:
        height_op = 'crop'
    elif image_height < target_height:
        height_op = 'pad'

    # Determine overall operation
    if width_op == height_op:
        operation = width_op if width_op != 'none' else 'none'
    else:
        operation = 'mixed'

    # Calculate bounds
    crop_bounds = None
    padding = None

    if width_op == 'crop' and height_op == 'crop':
        crop_bounds = calculate_center_crop_bounds(
            image_width, image_height, target_width, target_height
        )
    elif width_op == 'pad' and height_op == 'pad':
        padding = calculate_padding(
            image_width, image_height, target_width, target_height
        )

    return {
        'operation': operation,
        'width_op': width_op,
        'height_op': height_op,
        'crop_bounds': crop_bounds,
        'padding': padding,
    }
