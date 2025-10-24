"""Pure functions for URL and file extension parsing.

This module contains URL/path-related pure functions extracted from
scripts/deforum_helpers/video_audio_utilities.py, following functional
programming principles with no side effects.
"""

import os
import re
from typing import Tuple


def extract_extension_from_url(url: str) -> str:
    """Extract file extension from URL, removing query parameters.

    Args:
        url: URL string, may include query parameters

    Returns:
        File extension (without dot) in lowercase

    Examples:
        >>> extract_extension_from_url('http://example.com/file.mp4')
        'mp4'
        >>> extract_extension_from_url('https://example.com/video.avi?token=abc123')
        'avi'
        >>> extract_extension_from_url('http://example.com/image.PNG')
        'png'
        >>> extract_extension_from_url('http://example.com/noext')
        ''
    """
    # Remove query string first
    url_without_query = url.rsplit('?', 1)[0]
    # Get the path component (everything after the domain)
    # Split by '/' and get the last part (filename)
    path_parts = url_without_query.split('/')
    if path_parts:
        filename = path_parts[-1]
        # Extract extension from filename
        ext_parts = filename.rsplit('.', 1)
        if len(ext_parts) == 2 and ext_parts[0]:  # Ensure there's a name before the dot
            return ext_parts[-1].lower()
    return ''


def extract_extension_from_path(path: str) -> str:
    """Extract file extension from file path.

    Args:
        path: File path string

    Returns:
        File extension (without dot) in lowercase

    Examples:
        >>> extract_extension_from_path('/path/to/file.mp4')
        'mp4'
        >>> extract_extension_from_path('video.AVI')
        'avi'
        >>> extract_extension_from_path('file')
        ''
        >>> extract_extension_from_path('/path/to/archive.tar.gz')
        'gz'
    """
    parts = path.rsplit('.', 1)
    if len(parts) == 2:
        return parts[-1].lower()
    return ''


def parse_content_disposition_filename(content_disposition: str) -> str | None:
    """Parse filename from Content-Disposition header.

    Args:
        content_disposition: Content-Disposition header value

    Returns:
        Filename if found, None otherwise

    Examples:
        >>> parse_content_disposition_filename('attachment; filename="video.mp4"')
        'video.mp4'
        >>> parse_content_disposition_filename('attachment; filename=document.pdf')
        'document.pdf'
        >>> parse_content_disposition_filename('inline; filename="my file.txt"')
        'my file.txt'
        >>> parse_content_disposition_filename('attachment') is None
        True
        >>> parse_content_disposition_filename('')
is None
        True
    """
    if not content_disposition:
        return None

    # Pattern matches: filename="value" or filename=value
    match = re.search(r'filename="?(?P<filename>[^"]+)"?', content_disposition)
    if match:
        return match.group('filename')
    return None


def extract_extension_from_content_disposition(content_disposition: str) -> str:
    """Extract file extension from Content-Disposition header.

    Args:
        content_disposition: Content-Disposition header value

    Returns:
        File extension (without dot) in lowercase, empty string if not found

    Examples:
        >>> extract_extension_from_content_disposition('attachment; filename="video.mp4"')
        'mp4'
        >>> extract_extension_from_content_disposition('attachment; filename=document.PDF')
        'pdf'
        >>> extract_extension_from_content_disposition('inline; filename="archive.tar.gz"')
        'gz'
        >>> extract_extension_from_content_disposition('attachment')
        ''
    """
    filename = parse_content_disposition_filename(content_disposition)
    if filename:
        return extract_extension_from_path(filename)
    return ''


def is_url(path: str) -> bool:
    """Check if string is a URL (starts with http:// or https://).

    Args:
        path: String to check

    Returns:
        True if string starts with http:// or https://

    Examples:
        >>> is_url('http://example.com/file.mp4')
        True
        >>> is_url('https://example.com/video.avi')
        True
        >>> is_url('/local/path/to/file.mp4')
        False
        >>> is_url('file.mp4')
        False
        >>> is_url('ftp://example.com/file')
        False
    """
    return path.startswith('http://') or path.startswith('https://')


def validate_extension(
    extension: str, acceptable_extensions: list[str]
) -> Tuple[bool, str]:
    """Validate file extension against acceptable list.

    Args:
        extension: File extension (without dot)
        acceptable_extensions: List of acceptable extensions (without dots)

    Returns:
        Tuple of (is_valid, extension_lowercase)

    Examples:
        >>> validate_extension('mp4', ['mp4', 'avi', 'mov'])
        (True, 'mp4')
        >>> validate_extension('MP4', ['mp4', 'avi'])
        (True, 'mp4')
        >>> validate_extension('txt', ['mp4', 'avi'])
        (False, 'txt')
        >>> validate_extension('', ['mp4'])
        (False, '')
    """
    ext_lower = extension.lower()
    return (ext_lower in acceptable_extensions, ext_lower)


def get_file_extension_or_default(url: str, default_extension: str = '') -> str:
    """Extract file extension from URL, returning default if not found.

    Args:
        url: URL or file path
        default_extension: Extension to return if none found (without dot)

    Returns:
        File extension (without dot) in lowercase, or default

    Examples:
        >>> get_file_extension_or_default('http://example.com/file.mp4', 'mp3')
        'mp4'
        >>> get_file_extension_or_default('http://example.com/noext', 'mp3')
        'mp3'
        >>> get_file_extension_or_default('/path/to/video.AVI', 'mp3')
        'avi'
        >>> get_file_extension_or_default('noextension', 'wav')
        'wav'
    """
    if is_url(url):
        ext = extract_extension_from_url(url)
    else:
        ext = extract_extension_from_path(url)

    return ext if ext else default_extension
