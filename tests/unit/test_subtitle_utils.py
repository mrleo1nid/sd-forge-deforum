"""Unit tests for subtitle formatting functions."""

from decimal import Decimal
import pytest

from deforum.utils.subtitle_utils import (
    time_to_srt_format,
    calculate_frame_duration,
    frame_time,
    format_subtitle_value,
    get_user_param_names,
    SUBTITLE_PARAM_NAMES,
)


class TestTimeToSrtFormat:
    """Test time_to_srt_format function."""

    def test_zero_seconds(self):
        assert time_to_srt_format(0) == "00:00:00,000"

    def test_one_second(self):
        assert time_to_srt_format(1.0) == "00:00:01,000"

    def test_one_minute(self):
        assert time_to_srt_format(60.0) == "00:01:00,000"

    def test_one_hour(self):
        assert time_to_srt_format(3600.0) == "01:00:00,000"

    def test_with_milliseconds(self):
        assert time_to_srt_format(1.5) == "00:00:01,500"
        assert time_to_srt_format(2.123) == "00:00:02,123"
        assert time_to_srt_format(3.999) == "00:00:03,999"

    def test_complex_time(self):
        # 1 hour, 23 minutes, 45 seconds, 678 milliseconds
        # Note: Floating point precision may cause slight variations
        result = time_to_srt_format(5025.678)
        # Accept either 677 or 678 milliseconds due to floating point precision
        assert result in ["01:23:45,677", "01:23:45,678"]

    def test_with_decimal_input(self):
        decimal_time = Decimal("83.5")
        result = time_to_srt_format(decimal_time)
        assert result == "00:01:23,500"

    def test_rounds_milliseconds(self):
        # Should round down milliseconds to int
        result = time_to_srt_format(1.1234)
        assert result == "00:00:01,123"


class TestCalculateFrameDuration:
    """Test calculate_frame_duration function."""

    def test_30fps(self):
        duration = calculate_frame_duration(30)
        assert isinstance(duration, Decimal)
        # Should be 1/30 = 0.0333...
        assert abs(float(duration) - 0.0333333) < 0.0001

    def test_60fps(self):
        duration = calculate_frame_duration(60)
        assert isinstance(duration, Decimal)
        # Should be 1/60 = 0.01666...
        assert abs(float(duration) - 0.0166667) < 0.0001

    def test_24fps(self):
        duration = calculate_frame_duration(24)
        assert isinstance(duration, Decimal)
        # Should be 1/24 = 0.04166...
        assert abs(float(duration) - 0.0416667) < 0.0001

    def test_custom_precision(self):
        duration1 = calculate_frame_duration(30, precision=10)
        duration2 = calculate_frame_duration(30, precision=20)
        # Both should be valid Decimals
        assert isinstance(duration1, Decimal)
        assert isinstance(duration2, Decimal)
        # Values should be similar but precision might differ
        assert abs(float(duration1) - float(duration2)) < 0.0000001

    def test_high_fps(self):
        duration = calculate_frame_duration(120)
        assert isinstance(duration, Decimal)
        # Should be 1/120 = 0.00833...
        assert abs(float(duration) - 0.0083333) < 0.0001


class TestFrameTime:
    """Test frame_time function."""

    def test_frame_zero(self):
        duration = calculate_frame_duration(30)
        time = frame_time(0, duration)
        assert isinstance(time, Decimal)
        assert float(time) == 0.0

    def test_frame_one(self):
        duration = calculate_frame_duration(30)
        time = frame_time(1, duration)
        assert isinstance(time, Decimal)
        assert abs(float(time) - 0.0333333) < 0.0001

    def test_one_second_at_30fps(self):
        duration = calculate_frame_duration(30)
        time = frame_time(30, duration)  # Frame 30 at 30fps = 1 second
        assert isinstance(time, Decimal)
        assert abs(float(time) - 1.0) < 0.0001

    def test_one_second_at_60fps(self):
        duration = calculate_frame_duration(60)
        time = frame_time(60, duration)  # Frame 60 at 60fps = 1 second
        assert isinstance(time, Decimal)
        assert abs(float(time) - 1.0) < 0.0001

    def test_large_frame_number(self):
        duration = calculate_frame_duration(30)
        time = frame_time(300, duration)  # Frame 300 at 30fps = 10 seconds
        assert isinstance(time, Decimal)
        assert abs(float(time) - 10.0) < 0.0001


class TestFormatSubtitleValue:
    """Test format_subtitle_value function."""

    def test_float_as_integer(self):
        assert format_subtitle_value(3.0) == "3"
        assert format_subtitle_value(42.0) == "42"
        assert format_subtitle_value(0.0) == "0"

    def test_float_with_decimals(self):
        assert format_subtitle_value(3.14159) == "3.142"
        assert format_subtitle_value(2.5) == "2.500"
        assert format_subtitle_value(1.234567) == "1.235"

    def test_integer(self):
        assert format_subtitle_value(42) == "42"
        assert format_subtitle_value(0) == "0"
        assert format_subtitle_value(-5) == "-5"

    def test_string(self):
        assert format_subtitle_value("test") == "test"
        assert format_subtitle_value("euler_a") == "euler_a"

    def test_negative_float(self):
        assert format_subtitle_value(-3.14159) == "-3.142"
        assert format_subtitle_value(-1.0) == "-1"

    def test_rounding(self):
        # Test rounding to 3 decimal places
        # Check that function rounds appropriately
        assert format_subtitle_value(1.1234) == "1.123"
        assert format_subtitle_value(1.1666) == "1.167"
        assert format_subtitle_value(1.1999) == "1.200"  # Should preserve trailing zeros
        assert format_subtitle_value(2.9995) == "2.999"  # Doesn't round to 3.000 (not integer)


class TestSubtitleParamNames:
    """Test SUBTITLE_PARAM_NAMES constant."""

    def test_param_names_structure(self):
        assert isinstance(SUBTITLE_PARAM_NAMES, dict)
        assert len(SUBTITLE_PARAM_NAMES) > 30  # Should have many parameters

    def test_param_names_keys(self):
        # Check that expected keys exist
        assert "angle" in SUBTITLE_PARAM_NAMES
        assert "zoom" in SUBTITLE_PARAM_NAMES
        assert "translation_x" in SUBTITLE_PARAM_NAMES

    def test_param_names_values_structure(self):
        # Each value should have backend, user, and print keys
        for param_name, param_info in SUBTITLE_PARAM_NAMES.items():
            assert "backend" in param_info
            assert "user" in param_info
            assert "print" in param_info
            assert isinstance(param_info["backend"], str)
            assert isinstance(param_info["user"], str)
            assert isinstance(param_info["print"], str)

    def test_specific_param_mapping(self):
        angle_info = SUBTITLE_PARAM_NAMES["angle"]
        assert angle_info["backend"] == "angle_series"
        assert angle_info["user"] == "Angle"
        assert angle_info["print"] == "Angle"


class TestGetUserParamNames:
    """Test get_user_param_names function."""

    def test_returns_list(self):
        result = get_user_param_names()
        assert isinstance(result, list)

    def test_contains_prompt(self):
        result = get_user_param_names()
        assert "Prompt" in result

    def test_contains_angle(self):
        result = get_user_param_names()
        assert "Angle" in result

    def test_contains_zoom(self):
        result = get_user_param_names()
        assert "Zoom" in result

    def test_length(self):
        result = get_user_param_names()
        # Should have all params from SUBTITLE_PARAM_NAMES plus "Prompt"
        assert len(result) == len(SUBTITLE_PARAM_NAMES) + 1

    def test_all_are_strings(self):
        result = get_user_param_names()
        assert all(isinstance(name, str) for name in result)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_subtitle_workflow_30fps(self):
        """Test complete workflow for 30fps video."""
        fps = 30
        frame_num = 45  # 1.5 seconds at 30fps

        duration = calculate_frame_duration(fps)
        time = frame_time(frame_num, duration)
        time_str = time_to_srt_format(time)

        # 45 frames at 30fps = 1.5 seconds
        assert time_str == "00:00:01,500"

    def test_full_subtitle_workflow_60fps(self):
        """Test complete workflow for 60fps video."""
        fps = 60
        frame_num = 180  # 3 seconds at 60fps

        duration = calculate_frame_duration(fps)
        time = frame_time(frame_num, duration)
        time_str = time_to_srt_format(time)

        # 180 frames at 60fps = 3 seconds
        assert time_str == "00:00:03,000"

    def test_format_various_param_values(self):
        """Test formatting different types of parameter values."""
        test_cases = [
            (3.0, "3"),
            (3.14159, "3.142"),
            (42, "42"),
            ("euler_a", "euler_a"),
            (-1.5, "-1.500"),
        ]

        for value, expected in test_cases:
            assert format_subtitle_value(value) == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
