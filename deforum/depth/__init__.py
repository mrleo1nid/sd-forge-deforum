"""Depth estimation module for Deforum.

Provides depth estimation capabilities using Depth-Anything V2 for 3D
animation mode and depth-based transformations.
"""

from .depth import DepthModel
from .depth_anything_v2 import DepthAnything

__all__ = [
    "DepthModel",
    "DepthAnything",
]
