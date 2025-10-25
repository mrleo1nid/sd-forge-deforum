"""Unit tests for conversion and random name generation utilities."""

import pytest
import re

from deforum.utils.conversion.types import (
    generate_random_name,
    string_to_boolean,
)


class TestGenerateRandomName:
    """Test generate_random_name function."""

    def test_default_length(self):
        name = generate_random_name()
        # Default length=8 bytes -> 16 hex characters
        assert len(name) == 16

    def test_custom_length(self):
        name = generate_random_name(length=4)
        assert len(name) == 8  # 4 bytes -> 8 hex chars

        name = generate_random_name(length=16)
        assert len(name) == 32  # 16 bytes -> 32 hex chars

    def test_with_suffix_with_dot(self):
        name = generate_random_name(suffix=".mp4")
        assert name.endswith(".mp4")
        assert len(name) == 16 + 4  # 16 hex + 4 chars for ".mp4"

    def test_with_suffix_without_dot(self):
        name = generate_random_name(suffix="mp4")
        assert name.endswith(".mp4")  # Auto-prepends dot
        assert len(name) == 16 + 4

    def test_various_suffixes(self):
        suffixes = [".txt", "txt", ".png", "png", ".json", "json"]
        for suffix in suffixes:
            name = generate_random_name(suffix=suffix)
            expected_suffix = suffix if suffix.startswith(".") else f".{suffix}"
            assert name.endswith(expected_suffix)

    def test_hex_format(self):
        # All characters should be valid hex
        name = generate_random_name()
        hex_pattern = re.compile(r"^[0-9a-f]+$")
        assert hex_pattern.match(name)

    def test_randomness(self):
        # Generate multiple names - should be different
        names = [generate_random_name() for _ in range(100)]
        # All names should be unique
        assert len(set(names)) == 100

    def test_empty_suffix(self):
        name = generate_random_name(suffix="")
        assert len(name) == 16
        assert "." not in name

    def test_zero_length(self):
        # Edge case: zero-length random bytes
        name = generate_random_name(length=0)
        assert len(name) == 0

    def test_zero_length_with_suffix(self):
        name = generate_random_name(length=0, suffix=".txt")
        assert name == ".txt"

    def test_large_length(self):
        name = generate_random_name(length=128)
        assert len(name) == 256  # 128 bytes -> 256 hex chars


class TestStringToBoolean:
    """Test string_to_boolean function."""

    def test_true_values(self):
        true_strings = ["yes", "YES", "Yes", "true", "TRUE", "True", "t", "T", "y", "Y", "1"]
        for s in true_strings:
            assert string_to_boolean(s) is True

    def test_false_values(self):
        false_strings = ["no", "NO", "No", "false", "FALSE", "False", "f", "F", "n", "N", "0"]
        for s in false_strings:
            assert string_to_boolean(s) is False

    def test_boolean_input_true(self):
        assert string_to_boolean(True) is True

    def test_boolean_input_false(self):
        assert string_to_boolean(False) is False

    def test_mixed_case(self):
        assert string_to_boolean("YeS") is True
        assert string_to_boolean("nO") is False
        assert string_to_boolean("TrUe") is True
        assert string_to_boolean("FaLsE") is False

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="Boolean value expected"):
            string_to_boolean("maybe")

    def test_invalid_number_string(self):
        with pytest.raises(ValueError, match="Boolean value expected"):
            string_to_boolean("2")

    def test_invalid_empty_string(self):
        with pytest.raises(ValueError, match="Boolean value expected"):
            string_to_boolean("")

    def test_whitespace_not_stripped(self):
        # Whitespace is not stripped - should fail
        with pytest.raises(ValueError):
            string_to_boolean(" yes ")

    def test_partial_match_fails(self):
        # Partial matches don't work
        with pytest.raises(ValueError):
            string_to_boolean("yes!")

        with pytest.raises(ValueError):
            string_to_boolean("true1")

    def test_case_sensitivity(self):
        # Verify case insensitivity
        assert string_to_boolean("YES") is True
        assert string_to_boolean("yes") is True
        assert string_to_boolean("YeS") is True


class TestIntegration:
    """Integration tests combining conversion utilities."""

    def test_generate_names_for_video_frames(self):
        """Test generating unique frame filenames."""
        frame_names = [
            generate_random_name(length=4, suffix=".png") for _ in range(100)
        ]

        # All names should be unique
        assert len(set(frame_names)) == 100

        # All should have .png extension
        assert all(name.endswith(".png") for name in frame_names)

    def test_boolean_configuration_parsing(self):
        """Test parsing boolean configuration values."""
        config_values = {
            "enabled": "yes",
            "debug": "true",
            "verbose": "1",
            "quiet": "no",
            "strict": "false",
        }

        parsed = {k: string_to_boolean(v) for k, v in config_values.items()}

        assert parsed == {
            "enabled": True,
            "debug": True,
            "verbose": True,
            "quiet": False,
            "strict": False,
        }

    def test_filename_generation_consistency(self):
        """Test that filenames always have valid format."""
        for length in [1, 4, 8, 16, 32]:
            for suffix in ["", ".txt", ".mp4", ".png", "json"]:
                name = generate_random_name(length=length, suffix=suffix)

                # Check hex portion (before suffix)
                if suffix:
                    expected_suffix = suffix if suffix.startswith(".") else f".{suffix}"
                    hex_part = name[: -len(expected_suffix)]
                    assert len(hex_part) == length * 2
                    assert re.match(r"^[0-9a-f]+$", hex_part)
                else:
                    assert len(name) == length * 2
                    if length > 0:
                        assert re.match(r"^[0-9a-f]+$", name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
