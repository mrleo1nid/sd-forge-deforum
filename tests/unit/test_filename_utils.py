"""Unit tests for filename formatting functions."""

import pytest

from deforum.utils.filename_utils import (
    FileFormat,
    format_frame_index,
    format_frame_filename,
    format_depth_filename,
)


class TestFileFormat:
    """Test FileFormat enum."""

    def test_jpg_value(self):
        assert FileFormat.JPG.value == "jpg"

    def test_png_value(self):
        assert FileFormat.PNG.value == "png"

    def test_frame_format_returns_png(self):
        assert FileFormat.frame_format() == FileFormat.PNG

    def test_video_frame_format_returns_jpg(self):
        assert FileFormat.video_frame_format() == FileFormat.JPG


class TestFormatFrameIndex:
    """Test format_frame_index function."""

    def test_formats_single_digit(self):
        result = format_frame_index(1, FileFormat.PNG)
        assert result == "000000001.png"

    def test_formats_double_digit(self):
        result = format_frame_index(42, FileFormat.PNG)
        assert result == "000000042.png"

    def test_formats_large_number(self):
        result = format_frame_index(123456789, FileFormat.PNG)
        assert result == "123456789.png"

    def test_formats_zero(self):
        result = format_frame_index(0, FileFormat.PNG)
        assert result == "000000000.png"

    def test_formats_with_jpg(self):
        result = format_frame_index(100, FileFormat.JPG)
        assert result == "000000100.jpg"

    def test_zero_padding_length(self):
        """Verify padding is exactly 9 digits."""
        result = format_frame_index(1, FileFormat.PNG)
        # Split on '.' to get just the number part
        number_part = result.split('.')[0]
        assert len(number_part) == 9


class TestFormatFrameFilename:
    """Test format_frame_filename function."""

    def test_regular_frame_default_format(self):
        result = format_frame_filename(42)
        assert result == "000000042.png"

    def test_regular_frame_explicit_png(self):
        result = format_frame_filename(42, file_format=FileFormat.PNG)
        assert result == "000000042.png"

    def test_regular_frame_with_jpg(self):
        result = format_frame_filename(42, file_format=FileFormat.JPG)
        assert result == "000000042.jpg"

    def test_depth_frame_default_format(self):
        result = format_frame_filename(42, is_depth=True)
        assert result == "depth-maps/000000042_depth.png"

    def test_depth_frame_explicit_png(self):
        result = format_frame_filename(42, is_depth=True, file_format=FileFormat.PNG)
        assert result == "depth-maps/000000042_depth.png"

    def test_depth_frame_with_jpg(self):
        result = format_frame_filename(42, is_depth=True, file_format=FileFormat.JPG)
        assert result == "depth-maps/000000042_depth.jpg"

    def test_depth_frame_has_subdirectory(self):
        result = format_frame_filename(1, is_depth=True)
        assert result.startswith("depth-maps/")

    def test_depth_frame_has_suffix(self):
        result = format_frame_filename(1, is_depth=True)
        assert "_depth." in result

    def test_regular_frame_no_subdirectory(self):
        result = format_frame_filename(1, is_depth=False)
        assert "/" not in result

    def test_regular_frame_no_suffix(self):
        result = format_frame_filename(1, is_depth=False)
        assert "_depth" not in result


class TestFormatDepthFilename:
    """Test format_depth_filename convenience function."""

    def test_creates_depth_filename(self):
        result = format_depth_filename(42)
        assert result == "depth-maps/000000042_depth.png"

    def test_with_explicit_format(self):
        result = format_depth_filename(42, file_format=FileFormat.JPG)
        assert result == "depth-maps/000000042_depth.jpg"

    def test_matches_format_frame_filename_with_depth_true(self):
        """Verify it's equivalent to format_frame_filename with is_depth=True."""
        frame_idx = 123
        result1 = format_depth_filename(frame_idx)
        result2 = format_frame_filename(frame_idx, is_depth=True)
        assert result1 == result2

    def test_zero_frame(self):
        result = format_depth_filename(0)
        assert result == "depth-maps/000000000_depth.png"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
