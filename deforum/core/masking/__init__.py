"""Masking module for Deforum.

Provides masking capabilities for selective image generation including
composable masks, human detection masks, and word-based masks.
"""

from .masks import (
    do_overlay_mask,
)
from .composable import (
    compose_mask_with_check,
)

# human and word modules are available but not exported at package level
# Import them directly if needed:
#   from deforum.core.masking.human import video2humanmasks
#   from deforum.core.masking.word import get_word_mask

__all__ = [
    "do_overlay_mask",
    "compose_mask_with_check",
]
