"""Legacy wrapper for keyframe classes - imports from deforum.core.keyframes.

This module now imports all keyframe classes from the refactored core module.
All core keyframe logic has been extracted to deforum/core/keyframes.py.
"""

# Import all classes from refactored core module
from deforum.core.keyframes import (
    FrameInterpolater,
    DeformAnimKeys,
    ControlNetKeys,
    LooperAnimKeys,
)

# Re-export for backward compatibility
__all__ = [
    'FrameInterpolater',
    'DeformAnimKeys',
    'ControlNetKeys',
    'LooperAnimKeys',
]