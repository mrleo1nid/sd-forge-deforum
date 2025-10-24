"""Pure functions for file and path operations.

This module contains file/path-related pure functions extracted from
scripts/deforum_helpers/general_utils.py, following functional programming
principles with no side effects (read-only operations).
"""

import glob
import os


def get_max_path_length(base_folder_path: str, os_name: str, supports_long_paths: bool) -> int:
    """Calculate maximum path length for given OS and configuration.

    Args:
        base_folder_path: Base folder path to subtract from total
        os_name: Operating system name ('Windows', 'Linux', 'Mac', etc.)
        supports_long_paths: Whether OS supports long paths (Windows long path support)

    Returns:
        Maximum remaining path length after accounting for base folder

    Examples:
        >>> get_max_path_length('/home/user', 'Linux', False)
        4077
        >>> get_max_path_length('C:\\\\Users\\\\user', 'Windows', True)
        32750
        >>> get_max_path_length('C:\\\\Users\\\\user', 'Windows', False)
        243
    """
    if os_name == "Windows":
        max_length = 32767 if supports_long_paths else 260
    else:
        max_length = 4096

    return max_length - len(base_folder_path) - 1


def count_files_in_folder(folder_path: str) -> int:
    """Count total number of files in a folder (non-recursive).

    Args:
        folder_path: Path to folder to count files in

    Returns:
        Number of files (not directories) in the folder

    Examples:
        >>> # Assuming folder has 5 files
        >>> count_files_in_folder('/path/to/folder')
        5
    """
    file_pattern = os.path.join(folder_path, "*")
    file_count = len(glob.glob(file_pattern))
    return file_count
