"""Frame interpolation for Deforum.

Provides frame interpolation capabilities for smooth transitions between
keyframes using various algorithms (RIFE, FILM, etc.).
"""

from .frame_interpolation import (
    interpolate_frames,
    get_interpolation_engine,
)

__all__ = [
    "interpolate_frames",
    "get_interpolation_engine",
]
