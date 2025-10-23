"""Unit tests for image validation utilities."""

import pytest
from PIL import Image

from deforum.utils.validation_utils import (
    blank_if_none,
    none_if_blank,
    is_valid_json,
)


class TestBlankIfNone:
    """Test blank_if_none function."""

    def test_creates_blank_when_none(self):
        result = blank_if_none(None, 100, 50, 'L')
        assert isinstance(result, Image.Image)
        assert result.size == (100, 50)
        assert result.mode == 'L'

    def test_returns_original_when_not_none(self):
        original = Image.new('L', (200, 100), 128)
        result = blank_if_none(original, 100, 50, 'L')
        assert result is original
        assert result.size == (200, 100)

    def test_blank_image_is_black(self):
        result = blank_if_none(None, 10, 10, 'L')
        pixels = list(result.getdata())
        assert all(p == 0 for p in pixels)

    def test_supports_rgb_mode(self):
        result = blank_if_none(None, 50, 50, 'RGB')
        assert result.mode == 'RGB'
        assert result.size == (50, 50)

    def test_supports_rgba_mode(self):
        result = blank_if_none(None, 30, 30, 'RGBA')
        assert result.mode == 'RGBA'
        assert result.size == (30, 30)

    def test_different_dimensions(self):
        result = blank_if_none(None, 1920, 1080, 'L')
        assert result.size == (1920, 1080)


class TestNoneIfBlank:
    """Test none_if_blank function."""

    def test_returns_none_for_blank_image(self):
        blank = Image.new('L', (100, 100), 0)
        result = none_if_blank(blank)
        assert result is None

    def test_returns_image_for_non_blank(self):
        image = Image.new('L', (100, 100), 128)
        result = none_if_blank(image)
        assert result is image

    def test_returns_image_for_white(self):
        white = Image.new('L', (100, 100), 255)
        result = none_if_blank(white)
        assert result is white

    def test_returns_image_for_mixed(self):
        mixed = Image.new('L', (10, 10), 0)
        mixed.putpixel((5, 5), 255)
        result = none_if_blank(mixed)
        assert result is mixed

    def test_edge_case_single_white_pixel(self):
        mostly_black = Image.new('L', (10, 10), 0)
        mostly_black.putpixel((0, 0), 1)
        result = none_if_blank(mostly_black)
        assert result is mostly_black

    def test_blank_check_with_rgb(self):
        blank_rgb = Image.new('RGB', (50, 50), (0, 0, 0))
        # Convert to L for checking
        blank_l = blank_rgb.convert('L')
        result = none_if_blank(blank_l)
        assert result is None


class TestIsValidJson:
    """Test is_valid_json function."""

    def test_valid_object(self):
        """Should validate JSON objects."""
        assert is_valid_json('{"key": "value"}')
        assert is_valid_json('{"a": 1, "b": 2}')
        assert is_valid_json('{}')

    def test_valid_array(self):
        """Should validate JSON arrays."""
        assert is_valid_json('[1, 2, 3]')
        assert is_valid_json('["a", "b", "c"]')
        assert is_valid_json('[]')

    def test_valid_primitives(self):
        """Should validate JSON primitives."""
        assert is_valid_json('null')
        assert is_valid_json('true')
        assert is_valid_json('false')
        assert is_valid_json('123')
        assert is_valid_json('123.456')
        assert is_valid_json('"string"')

    def test_invalid_json(self):
        """Should reject invalid JSON."""
        assert not is_valid_json('not json')
        assert not is_valid_json('{invalid}')
        assert not is_valid_json('[1, 2,]')  # Trailing comma
        assert not is_valid_json("{'key': 'value'}")  # Single quotes

    def test_empty_string(self):
        """Should reject empty string."""
        assert not is_valid_json('')

    def test_nested_structures(self):
        """Should validate nested JSON structures."""
        assert is_valid_json('{"a": {"b": {"c": 1}}}')
        assert is_valid_json('[[1, 2], [3, 4]]')
        assert is_valid_json('{"arr": [1, 2, 3], "obj": {"key": "val"}}')

    def test_special_characters(self):
        """Should handle special characters in strings."""
        assert is_valid_json('{"key": "value\\nwith\\nnewlines"}')
        assert is_valid_json('{"unicode": "\\u0041"}')
        assert is_valid_json('{"quote": "\\""}')

    def test_numbers(self):
        """Should validate various number formats."""
        assert is_valid_json('0')
        assert is_valid_json('-1')
        assert is_valid_json('1.5')
        assert is_valid_json('-3.14')
        assert is_valid_json('1e10')
        assert is_valid_json('1.5e-10')

    def test_whitespace(self):
        """Should handle whitespace."""
        assert is_valid_json('  {"key": "value"}  ')
        assert is_valid_json('\n[\n  1,\n  2\n]\n')
        assert is_valid_json('\t\t{"a": 1}\t\t')


class TestIntegration:
    """Integration tests combining validation functions."""

    def test_blank_if_none_then_none_if_blank(self):
        # Create blank image
        blank = blank_if_none(None, 100, 100, 'L')
        # Should be blank
        assert none_if_blank(blank) is None

    def test_chain_with_existing_image(self):
        original = Image.new('L', (50, 50), 128)
        result = blank_if_none(original, 100, 100, 'L')
        # Should not be blank
        assert none_if_blank(result) is not None

    def test_create_or_validate_pattern(self):
        """Test common pattern: ensure image exists and is not blank."""
        mask = None
        # Create blank if None
        mask = blank_if_none(mask, 100, 100, 'L')
        # Check if blank
        mask = none_if_blank(mask)
        # Should be None because it's blank
        assert mask is None

    def test_preserve_non_blank_image(self):
        """Test that non-blank images are preserved."""
        mask = Image.new('L', (100, 100), 128)
        mask = blank_if_none(mask, 200, 200, 'L')  # Should return original
        mask = none_if_blank(mask)  # Should return image (not blank)
        assert mask is not None
        assert mask.size == (100, 100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
