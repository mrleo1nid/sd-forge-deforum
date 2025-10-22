"""Image sharpening functions - Legacy wrapper.

This module now imports from deforum.utils.image_utils for backward compatibility.
All pure sharpening functions have been refactored to the utils module.
"""

# Import all sharpening functions from refactored utils module
from deforum.utils.image_utils import (
    clamp_to_uint8,
    calculate_sharpened_image,
    apply_threshold_mask,
    apply_spatial_mask,
    unsharp_mask,
)

# Re-export for backward compatibility
__all__ = [
    'clamp_to_uint8',
    'calculate_sharpened_image',
    'apply_threshold_mask',
    'apply_spatial_mask',
    'unsharp_mask',
]
