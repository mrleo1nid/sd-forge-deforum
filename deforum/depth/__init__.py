"""Depth estimation module for Deforum.

Provides depth estimation capabilities using Depth-Anything V2 for 3D
animation mode and depth-based transformations.
"""

from .depth import DepthModel
from .depth_anything_v2 import DepthAnything
from .vid2depth import (
    vid2depth_main,
    add_soundtrack,
    # Add other vid2depth functions as needed
)

__all__ = [
    "DepthModel",
    "DepthAnything",
    "vid2depth_main",
    "add_soundtrack",
]
