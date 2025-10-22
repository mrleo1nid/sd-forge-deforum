"""Color matching functions - Legacy wrapper.

This module now imports from deforum.utils.image_utils for backward compatibility.
All pure color matching functions have been refactored to the utils module.
"""

# Import all color functions from refactored utils module
from deforum.utils.image_utils import (
    match_in_rgb,
    match_in_hsv,
    match_in_lab,
    maintain_colors,
)

# Re-export for backward compatibility
__all__ = [
    'match_in_rgb',
    'match_in_hsv',
    'match_in_lab',
    'maintain_colors',
]
