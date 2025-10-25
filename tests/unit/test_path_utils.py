"""Unit tests for path manipulation and parsing functions."""

import pytest

from deforum.utils.filesystem.paths import (
    extract_number_from_string,
    get_frame_name,
)


class TestExtractNumberFromString:
    """Test extract_number_from_string function."""

    def test_extracts_single_digit(self):
        assert extract_number_from_string('x2') == 2

    def test_extracts_double_digit(self):
        assert extract_number_from_string('x10') == 10

    def test_extracts_large_number(self):
        assert extract_number_from_string('x999') == 999

    def test_returns_minus_one_for_no_digit(self):
        assert extract_number_from_string('x') == -1

    def test_returns_minus_one_for_non_digit(self):
        assert extract_number_from_string('abc') == -1
        assert extract_number_from_string('xyz') == -1

    def test_returns_minus_one_for_empty_string(self):
        assert extract_number_from_string('') == -1

    def test_single_character_with_digit(self):
        # Single char has no second character
        assert extract_number_from_string('2') == -1

    def test_different_prefix_letters(self):
        assert extract_number_from_string('a5') == 5
        assert extract_number_from_string('y100') == 100


class TestGetFrameName:
    """Test get_frame_name function."""

    def test_extracts_basename_without_extension(self):
        assert get_frame_name('/path/to/frame001.png') == 'frame001'

    def test_simple_filename(self):
        assert get_frame_name('video.mp4') == 'video'

    def test_double_extension(self):
        # splitext only removes last extension
        assert get_frame_name('/path/to/file.tar.gz') == 'file.tar'

    def test_no_extension(self):
        assert get_frame_name('/path/to/filename') == 'filename'

    def test_relative_path(self):
        assert get_frame_name('folder/subfolder/image.jpg') == 'image'

    def test_just_filename_png(self):
        assert get_frame_name('test.png') == 'test'

    def test_complex_path(self):
        assert get_frame_name('/home/user/videos/animation/frame_0042.png') == 'frame_0042'

    def test_dot_in_filename(self):
        assert get_frame_name('my.video.file.mp4') == 'my.video.file'

    def test_hidden_file(self):
        # On Unix, .hidden is the full basename, splitext gives ('', '.hidden')
        # So get_frame_name('.hidden') == '' after splitext[0]
        # But splitext returns ('', '.hidden'), so [0] is empty string
        # Actually, os.path.splitext('.hidden') returns ('.hidden', '')
        # So basename is '.hidden' and splitext[0] is '.hidden'
        assert get_frame_name('.hidden') == '.hidden'
        assert get_frame_name('/path/.config') == '.config'


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_extract_and_format_frame_name(self):
        """Test extracting number and getting frame name together."""
        multiplier_str = 'x2'
        multiplier = extract_number_from_string(multiplier_str)
        assert multiplier == 2

        path = '/path/to/frame001.png'
        name = get_frame_name(path)
        assert name == 'frame001'

        # Could be used to create scaled frame name
        scaled_name = f"{name}_x{multiplier}"
        assert scaled_name == 'frame001_x2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
