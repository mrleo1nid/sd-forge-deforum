"""Pure functions for image processing operations.

This module contains image-related pure functions extracted from
scripts/deforum_helpers/image_sharpening.py and scripts/deforum_helpers/colors.py,
following functional programming principles with no side effects.
"""

import cv2
import numpy as np

# ============================================================================
# IMAGE SHARPENING FUNCTIONS
# ============================================================================


def clamp_to_uint8(image: np.ndarray) -> np.ndarray:
    """Clamp image values to valid uint8 range [0, 255].

    Args:
        image: Input array with any numeric values

    Returns:
        Array with values clamped to [0, 255], rounded, and converted to uint8
    """
    clamped = np.clip(image, 0, 255)
    return clamped.round().astype(np.uint8)


def calculate_sharpened_image(
    img: np.ndarray,
    blurred: np.ndarray,
    amount: float
) -> np.ndarray:
    """Apply unsharp mask formula: (1+amount)*img - amount*blurred.

    Args:
        img: Original image
        blurred: Blurred version of image
        amount: Sharpening strength

    Returns:
        Sharpened image clamped to uint8 range
    """
    sharpened = float(amount + 1) * img - float(amount) * blurred
    return clamp_to_uint8(sharpened)


def apply_threshold_mask(
    sharpened: np.ndarray,
    original: np.ndarray,
    blurred: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Preserve original pixels where contrast is below threshold.

    Args:
        sharpened: Sharpened image
        original: Original image
        blurred: Blurred version of original
        threshold: Contrast threshold

    Returns:
        Image with sharpening applied only to high-contrast areas
    """
    if threshold <= 0:
        return sharpened

    low_contrast_mask = np.absolute(original - blurred) < threshold
    result = sharpened.copy()
    np.copyto(result, original, where=low_contrast_mask)
    return result


def apply_spatial_mask(
    sharpened: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray | None
) -> np.ndarray:
    """Apply sharpening only to masked regions.

    Args:
        sharpened: Sharpened image
        original: Original image
        mask: Optional binary mask array

    Returns:
        Image with sharpening applied only in masked regions
    """
    if mask is None:
        return sharpened

    mask_array = np.array(mask)
    masked_sharpened = cv2.bitwise_and(sharpened, sharpened, mask=mask_array)
    masked_original = cv2.bitwise_and(original, original, mask=255 - mask_array)
    return cv2.add(masked_original, masked_sharpened)


def unsharp_mask(
    img: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.0,
    threshold: float = 0,
    mask: np.ndarray | None = None
) -> np.ndarray:
    """Apply unsharp mask sharpening to image.

    Args:
        img: Input image (uint8)
        kernel_size: Gaussian blur kernel size
        sigma: Gaussian blur sigma
        amount: Sharpening strength (0 = no sharpening)
        threshold: Low-contrast threshold (preserve pixels below this)
        mask: Optional spatial mask (sharpen only masked areas)

    Returns:
        Sharpened image (uint8)
    """
    if amount == 0:
        return img

    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = calculate_sharpened_image(img, blurred, amount)
    sharpened = apply_threshold_mask(sharpened, img, blurred, threshold)
    sharpened = apply_spatial_mask(sharpened, img, mask)

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
