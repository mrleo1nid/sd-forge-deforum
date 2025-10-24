"""Pure functions for validation and image checking.

This module contains validation-related pure functions extracted from
scripts/deforum_helpers/load_images.py and scripts/deforum_helpers/generate.py,
following functional programming principles with no side effects.
"""

import json
from PIL import Image


def blank_if_none(mask: Image.Image | None, w: int, h: int, mode: str) -> Image.Image:
    """Create blank image if mask is None.

    Args:
        mask: Input mask image or None
        w: Width for blank image
        h: Height for blank image
        mode: PIL image mode (e.g., 'L', 'RGB', 'RGBA')

    Returns:
        Original mask if not None, otherwise new blank image

    Examples:
        >>> from PIL import Image
        >>> result = blank_if_none(None, 100, 100, 'L')
        >>> result.size
        (100, 100)
        >>> result.mode
        'L'
    """
    return Image.new(mode, (w, h), (0)) if mask is None else mask


def none_if_blank(mask: Image.Image) -> Image.Image | None:
    """Return None if mask is completely black (blank).

    Args:
        mask: Input mask image

    Returns:
        None if mask is blank (all pixels are 0), otherwise returns mask

    Examples:
        >>> from PIL import Image
        >>> blank = Image.new('L', (10, 10), 0)
        >>> none_if_blank(blank) is None
        True
        >>> white = Image.new('L', (10, 10), 255)
        >>> none_if_blank(white) is not None
        True
    """
    return None if mask.getextrema() == (0, 0) else mask


def is_valid_json(json_string: str) -> bool:
    """Check if string is valid JSON.

    Args:
        json_string: String to validate as JSON

    Returns:
        True if string is valid JSON, False otherwise

    Examples:
        >>> is_valid_json('{"key": "value"}')
        True
        >>> is_valid_json('[1, 2, 3]')
        True
        >>> is_valid_json('not json')
        False
        >>> is_valid_json('')
        False
        >>> is_valid_json('null')
        True
        >>> is_valid_json('123')
        True
    """
    try:
        json.loads(json_string)
        return True
    except (ValueError, TypeError):
        return False
