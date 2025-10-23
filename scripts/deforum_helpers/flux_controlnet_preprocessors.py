"""Flux ControlNet preprocessors for Deforum.

Provides preprocessors for:
- Canny edge detection
- Depth map formatting (reuses existing Depth-Anything V2 output)
"""

import cv2
import numpy as np
from PIL import Image


def canny_edge_detection(
    image: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
    aperture_size: int = 3
) -> np.ndarray:
    """Extract Canny edges from an image.

    Args:
        image: Input image (BGR format from OpenCV or RGB numpy array)
        low_threshold: Lower threshold for edge detection (default: 100)
        high_threshold: Upper threshold for edge detection (default: 200)
        aperture_size: Aperture size for Sobel operator (default: 3)

    Returns:
        Canny edge map as uint8 numpy array (H, W, 3) - grayscale edges in RGB format
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)

    # Convert single channel to 3-channel RGB (edges are white on black background)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return edges_rgb


def depth_map_to_controlnet_format(depth_map: np.ndarray) -> np.ndarray:
    """Convert depth map to format expected by Flux Depth ControlNet.

    Args:
        depth_map: Depth map from Depth-Anything V2 (can be various formats)

    Returns:
        Formatted depth map as uint8 numpy array (H, W, 3) - RGB format
    """
    # Ensure depth map is 2D
    if len(depth_map.shape) == 3:
        if depth_map.shape[2] == 3:
            # If already RGB, convert to grayscale
            depth_gray = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
        else:
            # Take first channel
            depth_gray = depth_map[:, :, 0]
    else:
        depth_gray = depth_map

    # Normalize to 0-255 range if needed
    if depth_gray.dtype == np.float32 or depth_gray.dtype == np.float64:
        depth_normalized = ((depth_gray - depth_gray.min()) /
                           (depth_gray.max() - depth_gray.min()) * 255.0)
        depth_gray = depth_normalized.astype(np.uint8)
    elif depth_gray.dtype != np.uint8:
        depth_gray = depth_gray.astype(np.uint8)

    # Convert to RGB (depth is grayscale but needs 3 channels)
    depth_rgb = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2RGB)

    return depth_rgb


def preprocess_image_for_controlnet(
    image: np.ndarray,
    control_type: str,
    canny_low: int = 100,
    canny_high: int = 200
) -> np.ndarray:
    """Main preprocessing function that dispatches to specific preprocessors.

    Args:
        image: Input image (numpy array)
        control_type: Type of control ("canny" or "depth")
        canny_low: Low threshold for Canny (if control_type is "canny")
        canny_high: High threshold for Canny (if control_type is "canny")

    Returns:
        Preprocessed control image ready for FluxControlNetPipeline

    Raises:
        ValueError: If control_type is not supported
    """
    if control_type == "canny":
        return canny_edge_detection(image, canny_low, canny_high)
    elif control_type == "depth":
        return depth_map_to_controlnet_format(image)
    else:
        raise ValueError(f"Unsupported control type: {control_type}. Use 'canny' or 'depth'.")


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image for use with diffusers."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return Image.fromarray(image)
