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
    # Normalize dtype to uint8 if needed (OpenCV requires uint8 or float32)
    if image.dtype == np.float64 or image.dtype == np.float32:
        # Check if values are in 0-1 range or 0-255 range
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

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


def overlay_canny_edges(
    base_image: np.ndarray,
    canny_edges: np.ndarray,
    edge_color: tuple = (255, 0, 0),  # Red by default
    alpha: float = 0.8
) -> np.ndarray:
    """Overlay canny edges in color on top of base image for visualization.

    Args:
        base_image: Original image to overlay edges on (BGR or RGB format)
        canny_edges: Canny edge map (grayscale or RGB, with white edges)
        edge_color: RGB color for edges (default: red)
        alpha: Opacity of edges (0.0 = transparent, 1.0 = opaque)

    Returns:
        Image with colored edge overlay as uint8 numpy array
    """
    # Ensure base image is uint8
    if base_image.dtype == np.float64 or base_image.dtype == np.float32:
        if base_image.max() <= 1.0:
            base_image = (base_image * 255.0).astype(np.uint8)
        else:
            base_image = base_image.astype(np.uint8)

    # Ensure base image is RGB
    if len(base_image.shape) == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    elif base_image.shape[2] == 4:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGRA2RGB)

    # Extract edge mask (white pixels in canny output)
    if len(canny_edges.shape) == 3:
        # If RGB, take first channel
        edge_mask = canny_edges[:, :, 0]
    else:
        edge_mask = canny_edges

    # Create colored edge image
    overlay = base_image.copy()

    # Apply edge color where edges are detected
    for c in range(3):
        overlay[:, :, c] = np.where(
            edge_mask > 127,  # Edge pixels (white)
            int(edge_color[c] * alpha + base_image[:, :, c] * (1 - alpha)),
            base_image[:, :, c]
        )

    return overlay
