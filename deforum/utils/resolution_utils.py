"""Pure functions for resolution and dimension calculations.

This module contains pure functions for calculating video/image dimensions,
upscaling factors, and resolution strings, extracted from gradio_funcs.py.

All functions are pure (no side effects) and fully type-annotated.
"""

from typing import Tuple


def calculate_upscaled_resolution(
    input_resolution: str,
    scale_factor: int,
) -> str:
    """Calculate output resolution after upscaling.

    Takes a resolution string in format "width*height" and applies a scale factor
    to calculate the output resolution.

    Args:
        input_resolution: Input resolution in "W*H" format (e.g., "1920*1080")
        scale_factor: Integer scale factor to apply (e.g., 2, 3, 4)

    Returns:
        Output resolution string in "W*H" format after scaling

    Raises:
        ValueError: If input_resolution format is invalid or scale_factor <= 0

    Examples:
        >>> calculate_upscaled_resolution("1920*1080", 2)
        '3840*2160'
        >>> calculate_upscaled_resolution("640*480", 4)
        '2560*1920'
        >>> calculate_upscaled_resolution("---", 2)
        '---'
    """
    if not input_resolution or input_resolution == '---':
        return '---'

    if not isinstance(scale_factor, int) or scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive integer, got {scale_factor}")

    try:
        w, h = [int(x) * scale_factor for x in input_resolution.split('*')]
        return f"{w}*{h}"
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid input_resolution format: '{input_resolution}'. "
            f"Expected format: 'width*height' (e.g., '1920*1080')"
        ) from e


def get_scale_factor_for_model(model_name: str) -> int:
    """Get default scale factor for a given upscaling model.

    Different upscaling models have different native scale factors.
    This function returns the appropriate factor for each model.

    Args:
        model_name: Name of the upscaling model

    Returns:
        Scale factor (2 or 4) for the model

    Examples:
        >>> get_scale_factor_for_model("realesr-animevideov3")
        2
        >>> get_scale_factor_for_model("realesrgan-x4plus")
        4
        >>> get_scale_factor_for_model("anything_else")
        4
    """
    if not model_name:
        return 4

    # realesr-animevideov3 uses 2x scaling, all others default to 4x
    return 2 if model_name == 'realesr-animevideov3' else 4


def calculate_upscaled_resolution_by_model(
    input_resolution: str,
    model_name: str,
) -> str:
    """Calculate output resolution based on upscaling model's native scale factor.

    Combines model scale factor lookup with resolution calculation.

    Args:
        input_resolution: Input resolution in "W*H" format
        model_name: Name of the upscaling model

    Returns:
        Output resolution string in "W*H" format

    Examples:
        >>> calculate_upscaled_resolution_by_model("1920*1080", "realesr-animevideov3")
        '3840*2160'
        >>> calculate_upscaled_resolution_by_model("640*480", "realesrgan-x4plus")
        '2560*1920'
        >>> calculate_upscaled_resolution_by_model("---", "any_model")
        '---'
    """
    if not model_name or input_resolution == '---':
        return '---'

    scale_factor = get_scale_factor_for_model(model_name)
    return calculate_upscaled_resolution(input_resolution, scale_factor)


def parse_resolution_string(resolution: str) -> Tuple[int, int]:
    """Parse resolution string to width and height integers.

    Args:
        resolution: Resolution in "W*H" format (e.g., "1920*1080")

    Returns:
        Tuple of (width, height) as integers

    Raises:
        ValueError: If resolution format is invalid

    Examples:
        >>> parse_resolution_string("1920*1080")
        (1920, 1080)
        >>> parse_resolution_string("640*480")
        (640, 480)
    """
    try:
        w, h = resolution.split('*')
        return int(w), int(h)
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Invalid resolution format: '{resolution}'. "
            f"Expected format: 'width*height' (e.g., '1920*1080')"
        ) from e


def format_resolution_string(width: int, height: int) -> str:
    """Format width and height as resolution string.

    Args:
        width: Width in pixels
        height: Height in pixels

    Returns:
        Resolution string in "W*H" format

    Raises:
        ValueError: If width or height are not positive integers

    Examples:
        >>> format_resolution_string(1920, 1080)
        '1920*1080'
        >>> format_resolution_string(640, 480)
        '640*480'
    """
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"width and height must be integers, got {type(width).__name__}, {type(height).__name__}")

    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    return f"{width}*{height}"


def calculate_aspect_ratio(width: int, height: int) -> float:
    """Calculate aspect ratio from width and height.

    Args:
        width: Width in pixels
        height: Height in pixels

    Returns:
        Aspect ratio as float (width / height)

    Raises:
        ValueError: If height is zero or dimensions are invalid

    Examples:
        >>> calculate_aspect_ratio(1920, 1080)
        1.7777777777777777
        >>> calculate_aspect_ratio(1280, 720)
        1.7777777777777777
        >>> round(calculate_aspect_ratio(4, 3), 2)
        1.33
    """
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"width and height must be integers")

    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    return width / height
