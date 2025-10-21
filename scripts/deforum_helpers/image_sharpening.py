import cv2
import numpy as np

# ============================================================================
# PURE FUNCTIONS
# ============================================================================

def clamp_to_uint8(image: np.ndarray) -> np.ndarray:
    """Clamp image values to valid uint8 range [0, 255]."""
    clamped = np.clip(image, 0, 255)
    return clamped.round().astype(np.uint8)

def calculate_sharpened_image(
    img: np.ndarray,
    blurred: np.ndarray,
    amount: float
) -> np.ndarray:
    """Apply unsharp mask formula: (1+amount)*img - amount*blurred."""
    sharpened = float(amount + 1) * img - float(amount) * blurred
    return clamp_to_uint8(sharpened)

def apply_threshold_mask(
    sharpened: np.ndarray,
    original: np.ndarray,
    blurred: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Preserve original pixels where contrast is below threshold."""
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
    """Apply sharpening only to masked regions."""
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
