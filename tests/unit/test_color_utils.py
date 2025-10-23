"""Unit tests for deforum.utils.color_utils module."""

import pytest

from deforum.utils.color_utils import (
    hex_to_rgb,
    rgb_to_hex,
    hex_to_ansi_foreground,
    hex_to_ansi_background,
    rgb_to_ansi_foreground,
    rgb_to_ansi_background,
    normalize_rgb_values,
)


class TestHexToRgb:
    """Tests for hex_to_rgb function."""

    def test_red_with_hash(self):
        """Should convert red hex with hash."""
        assert hex_to_rgb('#FF0000') == (255, 0, 0)

    def test_green_without_hash(self):
        """Should convert green hex without hash."""
        assert hex_to_rgb('00FF00') == (0, 255, 0)

    def test_blue(self):
        """Should convert blue hex."""
        assert hex_to_rgb('#0000FF') == (0, 0, 255)

    def test_custom_color(self):
        """Should convert custom hex color."""
        assert hex_to_rgb('#FE797B') == (254, 121, 123)

    def test_black(self):
        """Should convert black."""
        assert hex_to_rgb('#000000') == (0, 0, 0)

    def test_white(self):
        """Should convert white."""
        assert hex_to_rgb('#FFFFFF') == (255, 255, 255)

    def test_lowercase_hex(self):
        """Should handle lowercase hex."""
        assert hex_to_rgb('#ff0000') == (255, 0, 0)

    def test_mixed_case_hex(self):
        """Should handle mixed case hex."""
        assert hex_to_rgb('#Ff00Ff') == (255, 0, 255)

    def test_invalid_length_raises_error(self):
        """Should raise ValueError for invalid length."""
        with pytest.raises(ValueError, match="must be 6 characters"):
            hex_to_rgb('#FFF')

    def test_invalid_characters_raises_error(self):
        """Should raise ValueError for invalid characters."""
        with pytest.raises(ValueError, match="Invalid hex color"):
            hex_to_rgb('#GGGGGG')


class TestRgbToHex:
    """Tests for rgb_to_hex function."""

    def test_red_with_hash(self):
        """Should convert red to hex with hash."""
        assert rgb_to_hex(255, 0, 0) == '#FF0000'

    def test_green_without_hash(self):
        """Should convert green to hex without hash."""
        assert rgb_to_hex(0, 255, 0, include_hash=False) == '00FF00'

    def test_blue(self):
        """Should convert blue to hex."""
        assert rgb_to_hex(0, 0, 255) == '#0000FF'

    def test_custom_color(self):
        """Should convert custom RGB to hex."""
        assert rgb_to_hex(254, 121, 123) == '#FE797B'

    def test_black(self):
        """Should convert black."""
        assert rgb_to_hex(0, 0, 0) == '#000000'

    def test_white(self):
        """Should convert white."""
        assert rgb_to_hex(255, 255, 255) == '#FFFFFF'

    def test_negative_values_raise_error(self):
        """Should raise ValueError for negative values."""
        with pytest.raises(ValueError, match="must be 0-255"):
            rgb_to_hex(-1, 0, 0)

    def test_values_above_255_raise_error(self):
        """Should raise ValueError for values > 255."""
        with pytest.raises(ValueError, match="must be 0-255"):
            rgb_to_hex(256, 0, 0)


class TestHexToAnsiForeground:
    """Tests for hex_to_ansi_foreground function."""

    def test_red_foreground(self):
        """Should convert red to ANSI foreground."""
        result = hex_to_ansi_foreground('#FF0000')
        assert result == '\x1b[38;2;255;0;0m'

    def test_green_foreground(self):
        """Should convert green to ANSI foreground."""
        result = hex_to_ansi_foreground('#00FF00')
        assert result == '\x1b[38;2;0;255;0m'

    def test_blue_foreground(self):
        """Should convert blue to ANSI foreground."""
        result = hex_to_ansi_foreground('0000FF')
        assert result == '\x1b[38;2;0;0;255m'

    def test_custom_color_foreground(self):
        """Should convert custom color to ANSI foreground."""
        result = hex_to_ansi_foreground('#FE797B')
        assert result == '\x1b[38;2;254;121;123m'


class TestHexToAnsiBackground:
    """Tests for hex_to_ansi_background function."""

    def test_red_background(self):
        """Should convert red to ANSI background."""
        result = hex_to_ansi_background('#FF0000')
        assert result == '\x1b[48;2;255;0;0m'

    def test_green_background(self):
        """Should convert green to ANSI background."""
        result = hex_to_ansi_background('#00FF00')
        assert result == '\x1b[48;2;0;255;0m'

    def test_blue_background(self):
        """Should convert blue to ANSI background."""
        result = hex_to_ansi_background('0000FF')
        assert result == '\x1b[48;2;0;0;255m'


class TestRgbToAnsiForeground:
    """Tests for rgb_to_ansi_foreground function."""

    def test_red_foreground(self):
        """Should convert RGB red to ANSI foreground."""
        result = rgb_to_ansi_foreground(255, 0, 0)
        assert result == '\x1b[38;2;255;0;0m'

    def test_green_foreground(self):
        """Should convert RGB green to ANSI foreground."""
        result = rgb_to_ansi_foreground(0, 255, 0)
        assert result == '\x1b[38;2;0;255;0m'

    def test_blue_foreground(self):
        """Should convert RGB blue to ANSI foreground."""
        result = rgb_to_ansi_foreground(0, 0, 255)
        assert result == '\x1b[38;2;0;0;255m'

    def test_invalid_values_raise_error(self):
        """Should raise ValueError for invalid RGB values."""
        with pytest.raises(ValueError, match="must be 0-255"):
            rgb_to_ansi_foreground(256, 0, 0)


class TestRgbToAnsiBackground:
    """Tests for rgb_to_ansi_background function."""

    def test_red_background(self):
        """Should convert RGB red to ANSI background."""
        result = rgb_to_ansi_background(255, 0, 0)
        assert result == '\x1b[48;2;255;0;0m'

    def test_green_background(self):
        """Should convert RGB green to ANSI background."""
        result = rgb_to_ansi_background(0, 255, 0)
        assert result == '\x1b[48;2;0;255;0m'

    def test_blue_background(self):
        """Should convert RGB blue to ANSI background."""
        result = rgb_to_ansi_background(0, 0, 255)
        assert result == '\x1b[48;2;0;0;255m'

    def test_invalid_values_raise_error(self):
        """Should raise ValueError for invalid RGB values."""
        with pytest.raises(ValueError, match="must be 0-255"):
            rgb_to_ansi_background(-1, 0, 0)


class TestNormalizeRgbValues:
    """Tests for normalize_rgb_values function."""

    def test_pure_red(self):
        """Should normalize pure red."""
        assert normalize_rgb_values(1.0, 0.0, 0.0) == (255, 0, 0)

    def test_pure_green(self):
        """Should normalize pure green."""
        assert normalize_rgb_values(0.0, 1.0, 0.0) == (0, 255, 0)

    def test_pure_blue(self):
        """Should normalize pure blue."""
        assert normalize_rgb_values(0.0, 0.0, 1.0) == (0, 0, 255)

    def test_gray(self):
        """Should normalize gray."""
        # 0.5 * 255 = 127.5, which rounds to 128 (banker's rounding)
        assert normalize_rgb_values(0.5, 0.5, 0.5) == (128, 128, 128)

    def test_custom_values(self):
        """Should normalize custom float values."""
        # 0.5 * 255 = 127.5, which rounds to 128
        assert normalize_rgb_values(0.0, 1.0, 0.5) == (0, 255, 128)

    def test_black(self):
        """Should normalize black."""
        assert normalize_rgb_values(0.0, 0.0, 0.0) == (0, 0, 0)

    def test_white(self):
        """Should normalize white."""
        assert normalize_rgb_values(1.0, 1.0, 1.0) == (255, 255, 255)

    def test_values_above_1_raise_error(self):
        """Should raise ValueError for values > 1.0."""
        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            normalize_rgb_values(1.1, 0.0, 0.0)

    def test_negative_values_raise_error(self):
        """Should raise ValueError for negative values."""
        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            normalize_rgb_values(-0.1, 0.0, 0.0)


class TestColorUtilsIntegration:
    """Integration tests combining multiple color utilities."""

    def test_hex_to_rgb_to_hex_roundtrip(self):
        """Should roundtrip from hex to RGB and back."""
        original = '#FE797B'
        r, g, b = hex_to_rgb(original)
        result = rgb_to_hex(r, g, b)
        assert result == original

    def test_rgb_to_hex_to_rgb_roundtrip(self):
        """Should roundtrip from RGB to hex and back."""
        original = (254, 121, 123)
        hex_color = rgb_to_hex(*original)
        result = hex_to_rgb(hex_color)
        assert result == original

    def test_hex_to_ansi_equivalence(self):
        """Should produce equivalent ANSI codes via different paths."""
        hex_color = '#FF0000'
        r, g, b = hex_to_rgb(hex_color)

        ansi1 = hex_to_ansi_foreground(hex_color)
        ansi2 = rgb_to_ansi_foreground(r, g, b)
        assert ansi1 == ansi2

    def test_normalize_to_hex_workflow(self):
        """Should convert normalized floats to hex."""
        float_rgb = (1.0, 0.5, 0.25)
        int_rgb = normalize_rgb_values(*float_rgb)
        hex_color = rgb_to_hex(*int_rgb)
        # 0.5*255=127.5→128 (0x80), 0.25*255=63.75→64 (0x40)
        assert hex_color == '#FF8040'

    def test_complete_color_pipeline(self):
        """Should handle complete color conversion pipeline."""
        # Start with hex
        hex_color = '#36CEDC'

        # Convert to RGB
        r, g, b = hex_to_rgb(hex_color)
        assert (r, g, b) == (54, 206, 220)

        # Convert to ANSI foreground
        ansi_fg = rgb_to_ansi_foreground(r, g, b)
        assert ansi_fg == '\x1b[38;2;54;206;220m'

        # Convert to ANSI background
        ansi_bg = rgb_to_ansi_background(r, g, b)
        assert ansi_bg == '\x1b[48;2;54;206;220m'

        # Roundtrip back to hex
        result = rgb_to_hex(r, g, b)
        assert result == hex_color


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
