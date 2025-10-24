"""Pure functions for path manipulation and parsing.

This module contains path-related pure functions extracted from various
deforum_helpers modules, following functional programming principles with
no side effects.
"""

import os


def extract_number_from_string(string: str) -> int:
    """Extract integer from string with format like 'x2', 'x10', etc.

    Commonly used to parse multiplier strings like 'x2' to get the number 2.

    Args:
        string: Input string starting with a letter followed by digits

    Returns:
        Extracted number, or -1 if format is invalid

    Examples:
        >>> extract_number_from_string('x2')
        2
        >>> extract_number_from_string('x10')
        10
        >>> extract_number_from_string('x')
        -1
        >>> extract_number_from_string('abc')
        -1
    """
    return int(string[1:]) if len(string) > 1 and string[1:].isdigit() else -1


def get_frame_name(path: str) -> str:
    """Extract frame name from file path (basename without extension).

    Args:
        path: File path (can be absolute or relative)

    Returns:
        Filename without extension

    Examples:
        >>> get_frame_name('/path/to/frame001.png')
        'frame001'
        >>> get_frame_name('video.mp4')
        'video'
        >>> get_frame_name('/path/to/file.tar.gz')
        'file.tar'
    """
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    return name
