"""RAFT optical flow integration for Deforum.

RAFT (Recurrent All-Pairs Field Transforms) is used for enhanced depth estimation
and optical flow analysis in video generation.
"""

from .raft_flow import RAFT

__all__ = ["RAFT"]
