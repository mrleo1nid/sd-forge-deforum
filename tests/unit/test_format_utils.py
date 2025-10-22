"""Unit tests for formatting and conversion utilities."""

import pytest

from deforum.utils.format_utils import (
    format_value_to_schedule,
    format_frame_time,
    format_bytes,
    format_percentage,
    format_resolution,
)


class TestFormatValueToSchedule:
    """Test format_value_to_schedule function."""

    def test_integer_value(self):
        result = format_value_to_schedule(1)
        assert result == "0:(1)"

    def test_float_value(self):
        result = format_value_to_schedule(0.5)
        assert result == "0:(0.5)"

    def test_negative_value(self):
        result = format_value_to_schedule(-2.5)
        assert result == "0:(-2.5)"

    def test_zero_value(self):
        result = format_value_to_schedule(0)
        assert result == "0:(0)"

    def test_large_value(self):
        result = format_value_to_schedule(1000)
        assert result == "0:(1000)"

    def test_precision_preserved(self):
        result = format_value_to_schedule(3.14159)
        assert result == "0:(3.14159)"


class TestFormatFrameTime:
    """Test format_frame_time function."""

    def test_zero_frame(self):
        result = format_frame_time(0, 30)
        assert result == "00:00.000"

    def test_one_second(self):
        result = format_frame_time(30, 30)
        assert result == "00:01.000"

    def test_one_minute(self):
        result = format_frame_time(1800, 30)  # 60 seconds * 30 fps
        assert result == "01:00.000"

    def test_fractional_second(self):
        result = format_frame_time(15, 30)  # 0.5 seconds
        assert result == "00:00.500"

    def test_complex_time(self):
        result = format_frame_time(150, 30)  # 5 seconds
        assert result == "00:05.000"

    def test_different_fps(self):
        result = format_frame_time(60, 60)  # 1 second at 60fps
        assert result == "00:01.000"

    def test_milliseconds(self):
        result = format_frame_time(1, 30)  # 1/30 = 0.0333... seconds
        # 33 milliseconds (rounded)
        assert result.startswith("00:00.0")

    def test_long_duration(self):
        result = format_frame_time(18000, 30)  # 10 minutes
        assert result == "10:00.000"


class TestFormatBytes:
    """Test format_bytes function."""

    def test_bytes(self):
        result = format_bytes(512)
        assert result == "512.0 B"

    def test_kilobytes(self):
        result = format_bytes(1024)
        assert result == "1.0 KB"

    def test_megabytes(self):
        result = format_bytes(1048576)
        assert result == "1.0 MB"

    def test_gigabytes(self):
        result = format_bytes(1073741824)
        assert result == "1.0 GB"

    def test_terabytes(self):
        result = format_bytes(1099511627776)
        assert result == "1.0 TB"

    def test_fractional_mb(self):
        result = format_bytes(1572864)  # 1.5 MB
        assert result == "1.5 MB"

    def test_zero_bytes(self):
        result = format_bytes(0)
        assert result == "0.0 B"

    def test_large_value(self):
        result = format_bytes(int(1e15))
        # Should be in TB or PB
        assert "TB" in result or "PB" in result

    def test_boundary_values(self):
        # Just under 1 KB
        result = format_bytes(1023)
        assert result == "1023.0 B"

        # Just over 1 KB
        result = format_bytes(1025)
        assert result == "1.0 KB"


class TestFormatPercentage:
    """Test format_percentage function."""

    def test_half(self):
        result = format_percentage(0.5)
        assert result == "50.0%"

    def test_zero(self):
        result = format_percentage(0.0)
        assert result == "0.0%"

    def test_one(self):
        result = format_percentage(1.0)
        assert result == "100.0%"

    def test_decimal_places(self):
        result = format_percentage(0.755, 2)
        assert result == "75.50%"

    def test_one_decimal_place(self):
        result = format_percentage(0.755, 1)
        assert result == "75.5%"

    def test_no_decimal_places(self):
        result = format_percentage(0.755, 0)
        assert result == "76%"  # Rounded

    def test_small_percentage(self):
        result = format_percentage(0.001, 3)
        assert result == "0.100%"

    def test_over_one(self):
        # Handle values over 1.0 (over 100%)
        result = format_percentage(1.5)
        assert result == "150.0%"

    def test_negative(self):
        result = format_percentage(-0.1)
        assert result == "-10.0%"


class TestFormatResolution:
    """Test format_resolution function."""

    def test_hd(self):
        result = format_resolution(1920, 1080)
        assert result == "1920x1080"

    def test_square(self):
        result = format_resolution(512, 512)
        assert result == "512x512"

    def test_4k(self):
        result = format_resolution(3840, 2160)
        assert result == "3840x2160"

    def test_small_resolution(self):
        result = format_resolution(64, 64)
        assert result == "64x64"

    def test_portrait(self):
        result = format_resolution(1080, 1920)
        assert result == "1080x1920"

    def test_ultra_wide(self):
        result = format_resolution(3440, 1440)
        assert result == "3440x1440"


class TestIntegration:
    """Integration tests combining format utilities."""

    def test_frame_time_and_percentage(self):
        """Test combining frame time with progress percentage."""
        total_frames = 300
        current_frame = 150
        fps = 30

        time_str = format_frame_time(current_frame, fps)
        progress_str = format_percentage(current_frame / total_frames)

        assert time_str == "00:05.000"
        assert progress_str == "50.0%"

    def test_resolution_and_bytes(self):
        """Test formatting resolution with file size."""
        width, height = 1920, 1080
        file_size = 5242880  # 5 MB

        res_str = format_resolution(width, height)
        size_str = format_bytes(file_size)

        assert res_str == "1920x1080"
        assert size_str == "5.0 MB"

    def test_schedule_formatting_pipeline(self):
        """Test schedule formatting for multiple values."""
        values = [0.5, 1.0, 1.5]
        schedules = [format_value_to_schedule(v) for v in values]

        assert schedules == ["0:(0.5)", "0:(1.0)", "0:(1.5)"]

    def test_complete_video_info_format(self):
        """Test formatting complete video information."""
        frame_count = 900
        fps = 30
        width, height = 1920, 1080
        file_size = 104857600  # 100 MB

        duration = format_frame_time(frame_count, fps)
        resolution = format_resolution(width, height)
        size = format_bytes(file_size)

        assert duration == "00:30.000"
        assert resolution == "1920x1080"
        assert size == "100.0 MB"

    def test_progress_display(self):
        """Test formatting progress information."""
        current = 75
        total = 300
        elapsed_frames = 75
        fps = 30

        progress = format_percentage(current / total, 1)
        time_elapsed = format_frame_time(elapsed_frames, fps)

        assert progress == "25.0%"
        assert time_elapsed == "00:02.500"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
