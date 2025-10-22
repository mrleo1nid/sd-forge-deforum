"""Pure functions for filename formatting.

This module contains filename formatting utilities extracted from
scripts/deforum_helpers/rendering/util/filename_utils.py, following
functional programming principles with no side effects.
"""

from enum import Enum


class FileFormat(Enum):
    """File format enumeration for frame output."""

    JPG = "jpg"
    PNG = "png"

    @staticmethod
    def frame_format() -> "FileFormat":
        """Get default format for regular frames.

        Returns:
            PNG format for high-quality frame storage
        """
        return FileFormat.PNG

    @staticmethod
    def video_frame_format() -> "FileFormat":
        """Get default format for video frames.

        Returns:
            JPG format for video frame extraction
        """
        return FileFormat.JPG


def format_frame_index(frame_index: int, file_format: FileFormat) -> str:
    """Format frame index as zero-padded filename.

    Args:
        frame_index: Frame number to format
        file_format: File extension format

    Returns:
        Formatted filename (e.g., "000000042.png")
    """
    return f"{frame_index:09}.{file_format.value}"


def format_frame_filename(
    frame_index: int, is_depth: bool = False, file_format: FileFormat | None = None
) -> str:
    """Format frame filename with optional depth map path.

    Args:
        frame_index: Frame number to format
        is_depth: Whether this is a depth map (adds depth-maps/ subdirectory)
        file_format: File format (defaults to PNG for regular frames)

    Returns:
        Formatted filename path:
        - Regular: "000000042.png"
        - Depth: "depth-maps/000000042_depth.png"
    """
    fmt = file_format if file_format is not None else FileFormat.frame_format()

    if is_depth:
        return f"depth-maps/{frame_index:09}_depth.{fmt.value}"
    else:
        return f"{frame_index:09}.{fmt.value}"


def format_depth_filename(frame_index: int, file_format: FileFormat | None = None) -> str:
    """Format depth map filename.

    Convenience wrapper around format_frame_filename for depth maps.

    Args:
        frame_index: Frame number to format
        file_format: File format (defaults to PNG)

    Returns:
        Depth map filename: "depth-maps/000000042_depth.png"
    """
    return format_frame_filename(frame_index, is_depth=True, file_format=file_format)
