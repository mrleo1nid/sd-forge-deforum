import cv2
import numpy as np
from skimage.exposure import match_histograms
from typing import Literal

# ============================================================================
# PURE FUNCTIONS
# ============================================================================

ColorMode = Literal['RGB', 'HSV', 'LAB']

def match_in_rgb(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram in RGB color space."""
    return match_histograms(img, reference, channel_axis=-1)

def match_in_hsv(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram in HSV color space."""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)
    matched_hsv = match_histograms(img_hsv, ref_hsv, channel_axis=-1)
    return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)

def match_in_lab(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram in LAB color space."""
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
    matched_lab = match_histograms(img_lab, ref_lab, channel_axis=-1)
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

def maintain_colors(prev_img: np.ndarray, color_match_sample: np.ndarray, mode: ColorMode) -> np.ndarray:
    """Match color histogram between images using specified color space."""
    if mode == 'RGB':
        return match_in_rgb(prev_img, color_match_sample)
    elif mode == 'HSV':
        return match_in_hsv(prev_img, color_match_sample)
    else:  # LAB
        return match_in_lab(prev_img, color_match_sample)
