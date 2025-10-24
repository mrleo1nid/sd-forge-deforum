"""Masking module for Deforum.

Provides masking capabilities for selective image generation including
composable masks, human detection masks, and word-based masks.
"""

from .masks import (
    do_overlay_mask,
    # Add other mask functions as needed
)
from .composable import (
    compose_mask_with_check,
    # Add other composable functions as needed
)
from .human import (
    # Add human masking functions as needed
)
from .word import (
    # Add word masking functions as needed
)

__all__ = [
    "do_overlay_mask",
    "compose_mask_with_check",
]
