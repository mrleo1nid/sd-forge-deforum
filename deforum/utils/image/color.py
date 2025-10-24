"""Pure functions for color conversion and ANSI formatting.

This module contains functions for converting between color formats and
generating ANSI escape codes for terminal coloring, following functional
programming principles with no side effects.
"""

from typing import Tuple


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (with or without leading '#')

    Returns:
        Tuple of (red, green, blue) values (0-255)

    Raises:
        ValueError: If hex string is invalid format

    Examples:
        >>> hex_to_rgb('#FF0000')
        (255, 0, 0)
        >>> hex_to_rgb('00FF00')
        (0, 255, 0)
        >>> hex_to_rgb('#0000FF')
        (0, 0, 255)
        >>> hex_to_rgb('#FE797B')
        (254, 121, 123)
    """
    # Remove '#' prefix if present
    hex_color = hex_color.lstrip('#')

    # Validate hex string
    if len(hex_color) != 6:
        raise ValueError(
            f"Hex color must be 6 characters (got {len(hex_color)}): '{hex_color}'"
        )

    try:
        # Convert hex pairs to integers
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError as e:
        raise ValueError(
            f"Invalid hex color string '{hex_color}': {str(e)}"
        ) from e


def rgb_to_hex(r: int, g: int, b: int, include_hash: bool = True) -> str:
    """Convert RGB values to hex color string.

    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)
        include_hash: Whether to include '#' prefix

    Returns:
        Hex color string

    Raises:
        ValueError: If RGB values are out of range

    Examples:
        >>> rgb_to_hex(255, 0, 0)
        '#FF0000'
        >>> rgb_to_hex(0, 255, 0, include_hash=False)
        '00FF00'
        >>> rgb_to_hex(0, 0, 255)
        '#0000FF'
        >>> rgb_to_hex(254, 121, 123)
        '#FE797B'
    """
    # Validate RGB values
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError(
            f"RGB values must be 0-255 (got r={r}, g={g}, b={b})"
        )

    # Convert to hex
    hex_str = f"{r:02X}{g:02X}{b:02X}"
    return f"#{hex_str}" if include_hash else hex_str


def hex_to_ansi_foreground(hex_color: str) -> str:
    """Convert hex color to ANSI foreground color escape code.

    Args:
        hex_color: Hex color string (with or without '#')

    Returns:
        ANSI escape code string for foreground color

    Examples:
        >>> hex_to_ansi_foreground('#FF0000')
        '\\x1b[38;2;255;0;0m'
        >>> hex_to_ansi_foreground('#00FF00')
        '\\x1b[38;2;0;255;0m'
        >>> hex_to_ansi_foreground('0000FF')
        '\\x1b[38;2;0;0;255m'
    """
    r, g, b = hex_to_rgb(hex_color)
    return f"\x1b[38;2;{r};{g};{b}m"


def hex_to_ansi_background(hex_color: str) -> str:
    """Convert hex color to ANSI background color escape code.

    Args:
        hex_color: Hex color string (with or without '#')

    Returns:
        ANSI escape code string for background color

    Examples:
        >>> hex_to_ansi_background('#FF0000')
        '\\x1b[48;2;255;0;0m'
        >>> hex_to_ansi_background('#00FF00')
        '\\x1b[48;2;0;255;0m'
        >>> hex_to_ansi_background('0000FF')
        '\\x1b[48;2;0;0;255m'
    """
    r, g, b = hex_to_rgb(hex_color)
    return f"\x1b[48;2;{r};{g};{b}m"


def rgb_to_ansi_foreground(r: int, g: int, b: int) -> str:
    """Convert RGB values to ANSI foreground color escape code.

    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)

    Returns:
        ANSI escape code string for foreground color

    Examples:
        >>> rgb_to_ansi_foreground(255, 0, 0)
        '\\x1b[38;2;255;0;0m'
        >>> rgb_to_ansi_foreground(0, 255, 0)
        '\\x1b[38;2;0;255;0m'
    """
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError(
            f"RGB values must be 0-255 (got r={r}, g={g}, b={b})"
        )
    return f"\x1b[38;2;{r};{g};{b}m"


def rgb_to_ansi_background(r: int, g: int, b: int) -> str:
    """Convert RGB values to ANSI background color escape code.

    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)

    Returns:
        ANSI escape code string for background color

    Examples:
        >>> rgb_to_ansi_background(255, 0, 0)
        '\\x1b[48;2;255;0;0m'
        >>> rgb_to_ansi_background(0, 255, 0)
        '\\x1b[48;2;0;255;0m'
    """
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError(
            f"RGB values must be 0-255 (got r={r}, g={g}, b={b})"
        )
    return f"\x1b[48;2;{r};{g};{b}m"


def normalize_rgb_values(r: float, g: float, b: float) -> Tuple[int, int, int]:
    """Normalize floating point RGB values (0.0-1.0) to integers (0-255).

    Args:
        r: Red value (0.0-1.0)
        g: Green value (0.0-1.0)
        b: Blue value (0.0-1.0)

    Returns:
        Tuple of integer RGB values (0-255)

    Raises:
        ValueError: If values are out of 0.0-1.0 range

    Examples:
        >>> normalize_rgb_values(1.0, 0.0, 0.0)
        (255, 0, 0)
        >>> normalize_rgb_values(0.5, 0.5, 0.5)
        (127, 127, 127)
        >>> normalize_rgb_values(0.0, 1.0, 0.5)
        (0, 255, 127)
    """
    if not (0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError(
            f"RGB float values must be 0.0-1.0 (got r={r}, g={g}, b={b})"
        )

    return (
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255))
    )
