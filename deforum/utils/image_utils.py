"""Pure functions for image processing operations.

This module contains image-related pure functions extracted from
scripts/deforum_helpers/image_sharpening.py and scripts/deforum_helpers/colors.py,
following functional programming principles with no side effects.
"""

import cv2
import numpy as np
from typing import Tuple

# ============================================================================
# IMAGE SHARPENING FUNCTIONS
# ============================================================================


def clamp_to_uint8(array: np.ndarray) -> np.ndarray:
    """Clamp array values to uint8 range [0, 255].

    Args:
        array: Input array with any numeric values

    Returns:
        Array with values clamped to [0, 255] and converted to uint8
    """
    return np.clip(array, 0, 255).astype(np.uint8)


def calculate_sharpened_image(
    image: np.ndarray, blurred: np.ndarray, amount: float
) -> np.ndarray:
    """Calculate sharpened image using unsharp mask formula.

    Formula: sharpened = image + amount * (image - blurred)

    Args:
        image: Original image
        blurred: Blurred version of image
        amount: Sharpening strength

    Returns:
        Sharpened image (may exceed uint8 range, needs clamping)
    """
    return image + amount * (image - blurred)


def apply_threshold_mask(
    sharpened: np.ndarray, image: np.ndarray, threshold: int
) -> np.ndarray:
    """Apply threshold mask to preserve low-contrast areas.

    Args:
        sharpened: Sharpened image
        image: Original image
        threshold: Contrast threshold (0-255)

    Returns:
        Image with sharpening applied only to high-contrast areas
    """
    if threshold <= 0:
        return sharpened

    lowcontrast = np.absolute(image - cv2.GaussianBlur(image, (5, 5), 0)) < threshold
    return np.where(lowcontrast, image, sharpened)


def apply_spatial_mask(sharpened: np.ndarray, image: np.ndarray, mask) -> np.ndarray:
    """Apply spatial mask to selectively sharpen regions.

    Args:
        sharpened: Sharpened image
        image: Original image
        mask: Binary mask (PIL Image or None)

    Returns:
        Image with sharpening applied only in masked regions
    """
    if mask is None:
        return sharpened

    mask_array = np.array(mask.convert("L")) / 255.0
    mask_3channel = np.stack([mask_array] * 3, axis=-1)
    return (sharpened * mask_3channel + image * (1 - mask_3channel)).astype(np.uint8)


def unsharp_mask(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.0,
    threshold: int = 0,
    mask=None,
) -> np.ndarray:
    """Apply unsharp mask sharpening to image.

    Args:
        image: Input image (HWC format, uint8)
        kernel_size: Gaussian blur kernel size
        sigma: Gaussian blur sigma
        amount: Sharpening strength (0.0 = no sharpening)
        threshold: Contrast threshold for selective sharpening
        mask: Optional spatial mask (PIL Image)

    Returns:
        Sharpened image (uint8)
    """
    if amount == 0.0:
        return image

    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = calculate_sharpened_image(image, blurred, amount)
    sharpened = apply_threshold_mask(sharpened, image, threshold)
    sharpened = clamp_to_uint8(sharpened)
    sharpened = apply_spatial_mask(sharpened, image, mask)

    return sharpened


# ============================================================================
# COLOR MATCHING FUNCTIONS
# ============================================================================


def match_in_rgb(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match image color statistics to reference in RGB color space.

    Args:
        image: Input image to adjust
        reference: Reference image with target colors

    Returns:
        Color-matched image
    """
    image_mean = image.mean(axis=(0, 1), keepdims=True)
    image_std = image.std(axis=(0, 1), keepdims=True)
    reference_mean = reference.mean(axis=(0, 1), keepdims=True)
    reference_std = reference.std(axis=(0, 1), keepdims=True)

    normalized = (image - image_mean) / (image_std + 1e-6)
    matched = (normalized * reference_std) + reference_mean

    return np.clip(matched, 0, 255).astype(np.uint8)


def match_in_hsv(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match image color statistics to reference in HSV color space.

    Args:
        image: Input image (BGR format)
        reference: Reference image (BGR format)

    Returns:
        Color-matched image (BGR format)
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    reference_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV).astype(np.float32)

    image_mean = image_hsv.mean(axis=(0, 1), keepdims=True)
    image_std = image_hsv.std(axis=(0, 1), keepdims=True)
    reference_mean = reference_hsv.mean(axis=(0, 1), keepdims=True)
    reference_std = reference_hsv.std(axis=(0, 1), keepdims=True)

    normalized = (image_hsv - image_mean) / (image_std + 1e-6)
    matched_hsv = (normalized * reference_std) + reference_mean

    matched_hsv = np.clip(matched_hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2BGR)


def match_in_lab(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match image color statistics to reference in LAB color space.

    Args:
        image: Input image (BGR format)
        reference: Reference image (BGR format)

    Returns:
        Color-matched image (BGR format)
    """
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    image_mean = image_lab.mean(axis=(0, 1), keepdims=True)
    image_std = image_lab.std(axis=(0, 1), keepdims=True)
    reference_mean = reference_lab.mean(axis=(0, 1), keepdims=True)
    reference_std = reference_lab.std(axis=(0, 1), keepdims=True)

    normalized = (image_lab - image_mean) / (image_std + 1e-6)
    matched_lab = (normalized * reference_std) + reference_mean

    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)


def maintain_colors(image: np.ndarray, reference: np.ndarray, mode: str) -> np.ndarray:
    """Maintain colors from reference image using specified color space.

    Args:
        image: Input image to adjust (BGR format)
        reference: Reference image with target colors (BGR format)
        mode: Color space ('rgb', 'hsv', 'lab')

    Returns:
        Color-matched image (BGR format)
    """
    if mode == "rgb":
        return match_in_rgb(image, reference)
    elif mode == "hsv":
        return match_in_hsv(image, reference)
    elif mode == "lab":
        return match_in_lab(image, reference)
    else:
        return image
