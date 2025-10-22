"""Filename utilities - Mixed pure and impure functions.

This module contains:
- Pure formatting functions imported from deforum.utils.filename_utils
- Impure path construction functions (kept here due to dependencies)
"""

from pathlib import Path

from ...video_audio_utilities import get_frame_name

# Import pure functions from refactored utils module
from deforum.utils.filename_utils import (
    FileFormat,
    format_frame_index as _frame_filename_index,
    format_frame_filename,
    format_depth_filename,
)


def frame_filename(data, i: int, is_depth=False, file_format=None) -> str:
    """Legacy wrapper for format_frame_filename.

    Args:
        data: RenderData object (unused, kept for backward compatibility)
        i: Frame index
        is_depth: Whether this is a depth map
        file_format: File format (defaults to PNG)

    Returns:
        Formatted filename path
    """
    if file_format is None:
        file_format = FileFormat.frame_format()
    return format_frame_filename(i, is_depth=is_depth, file_format=file_format)


def depth_frame(data, i) -> str:
    """Legacy wrapper for format_depth_filename.

    Args:
        data: RenderData object (unused, kept for backward compatibility)
        i: Frame index

    Returns:
        Depth map filename
    """
    return format_depth_filename(i)


def preview_video_image_path(data, i) -> Path:
    frame_name = get_frame_name(data.args.anim_args.video_init_path)
    index = _frame_filename_index(i, FileFormat.video_frame_format())
    return Path(data.output_directory) / "inputframes" / (frame_name + index)
