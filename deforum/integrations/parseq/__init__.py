"""Parseq integration for Deforum.

Parseq is a parameter sequencer for Stable Diffusion that provides
a GUI-based keyframe editor with advanced scheduling capabilities.
"""

from .adapter import ParseqAdapter

__all__ = ["ParseqAdapter"]
