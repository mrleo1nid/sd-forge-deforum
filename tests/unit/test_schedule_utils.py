"""Unit tests for schedule string parsing and interpolation utilities."""

import pytest

from deforum.utils.parsing.schedules import (
    parse_schedule_string,
    interpolate_schedule_values,
)


class TestParseScheduleString:
    """Test parse_schedule_string function."""

    def test_basic_schedule(self):
        result = parse_schedule_string("0:(1.0), 30:(2.0), 60:(1.5)")
        assert result == [(0, 1.0), (30, 2.0), (60, 1.5)]

    def test_single_keyframe(self):
        result = parse_schedule_string("0:(5.0)")
        assert result == [(0, 5.0)]

    def test_single_value_no_frame(self):
        # Single numeric value -> treated as frame 0
        result = parse_schedule_string("10")
        assert result == [(0, 10.0)]

        result = parse_schedule_string("3.14159")
        assert result == [(0, 3.14159)]

    def test_empty_string(self):
        result = parse_schedule_string("")
        assert result == [(0, 0.0)]

    def test_whitespace_only(self):
        result = parse_schedule_string("   ")
        assert result == [(0, 0.0)]

    def test_invalid_string(self):
        result = parse_schedule_string("invalid")
        assert result == [(0, 0.0)]

    def test_out_of_order_frames(self):
        # Frames should be sorted
        result = parse_schedule_string("30:(2.0), 0:(1.0), 60:(3.0)")
        assert result == [(0, 1.0), (30, 2.0), (60, 3.0)]

    def test_whitespace_handling(self):
        # Extra whitespace should be handled
        result = parse_schedule_string("0: (1.0),  30 : ( 2.0 ), 60 :(1.5)")
        assert result == [(0, 1.0), (30, 2.0), (60, 1.5)]

    def test_negative_values(self):
        result = parse_schedule_string("0:(-5.0), 10:(2.5)")
        assert result == [(0, -5.0), (10, 2.5)]

    def test_integer_values(self):
        result = parse_schedule_string("0:(1), 10:(2), 20:(3)")
        assert result == [(0, 1.0), (10, 2.0), (20, 3.0)]

    def test_large_frame_numbers(self):
        result = parse_schedule_string("0:(1.0), 1000:(2.0), 10000:(3.0)")
        assert result == [(0, 1.0), (1000, 2.0), (10000, 3.0)]

    def test_duplicate_frames(self):
        # Same frame number multiple times - all should be kept
        result = parse_schedule_string("0:(1.0), 10:(2.0), 10:(3.0)")
        # Both frame 10 entries should be present
        assert len(result) == 3
        assert result[0] == (0, 1.0)
        # Order of duplicates may vary, so check both are present
        assert (10, 2.0) in result
        assert (10, 3.0) in result

    def test_partial_invalid_entries(self):
        # Some valid, some invalid entries
        result = parse_schedule_string("0:(1.0), invalid, 20:(2.0)")
        assert result == [(0, 1.0), (20, 2.0)]

    def test_scientific_notation(self):
        result = parse_schedule_string("0:(1e-3), 10:(2.5e2)")
        assert result == [(0, 0.001), (10, 250.0)]

    def test_zero_value(self):
        result = parse_schedule_string("0:(0.0), 10:(0.0)")
        assert result == [(0, 0.0), (10, 0.0)]


class TestInterpolateScheduleValues:
    """Test interpolate_schedule_values function."""

    def test_linear_interpolation_simple(self):
        keyframes = [(0, 0.0), (10, 10.0)]
        result = interpolate_schedule_values(keyframes, 11)
        expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert result == expected

    def test_single_keyframe_constant(self):
        keyframes = [(0, 5.0)]
        result = interpolate_schedule_values(keyframes, 5)
        assert result == [5.0, 5.0, 5.0, 5.0, 5.0]

    def test_empty_keyframes(self):
        result = interpolate_schedule_values([], 5)
        assert result == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_multiple_segments(self):
        keyframes = [(0, 0.0), (5, 10.0), (10, 5.0)]
        result = interpolate_schedule_values(keyframes, 11)
        # Frame 0-5: 0 to 10 (slope +2)
        # Frame 5-10: 10 to 5 (slope -1)
        expected = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0]
        assert result == expected

    def test_frames_before_first_keyframe(self):
        # First keyframe at frame 10
        keyframes = [(10, 5.0), (20, 10.0)]
        result = interpolate_schedule_values(keyframes, 15)
        # Frames 0-9 should use first keyframe value
        for i in range(10):
            assert result[i] == 5.0
        # Frame 10 exactly on keyframe
        assert result[10] == 5.0

    def test_frames_after_last_keyframe(self):
        keyframes = [(0, 1.0), (10, 5.0)]
        result = interpolate_schedule_values(keyframes, 20)
        # Frames 11-19 should use last keyframe value
        for i in range(11, 20):
            assert result[i] == 5.0

    def test_negative_values(self):
        keyframes = [(0, -10.0), (10, 10.0)]
        result = interpolate_schedule_values(keyframes, 11)
        # Should interpolate from -10 to 10
        assert result[0] == -10.0
        assert result[5] == 0.0
        assert result[10] == 10.0

    def test_exact_keyframe_values(self):
        keyframes = [(0, 1.0), (5, 3.0), (10, 7.0)]
        result = interpolate_schedule_values(keyframes, 11)
        # Exact keyframe positions should match
        assert result[0] == 1.0
        assert result[5] == 3.0
        assert result[10] == 7.0

    def test_fractional_interpolation(self):
        keyframes = [(0, 0.0), (3, 1.0)]
        result = interpolate_schedule_values(keyframes, 4)
        # 3 steps from 0 to 1: 0.0, 0.333..., 0.666..., 1.0
        assert result[0] == 0.0
        assert abs(result[1] - 1.0 / 3.0) < 0.0001
        assert abs(result[2] - 2.0 / 3.0) < 0.0001
        assert result[3] == 1.0

    def test_single_frame(self):
        keyframes = [(0, 5.0)]
        result = interpolate_schedule_values(keyframes, 1)
        assert result == [5.0]

    def test_zero_frames(self):
        keyframes = [(0, 5.0)]
        result = interpolate_schedule_values(keyframes, 0)
        assert result == []

    def test_keyframes_beyond_max_frames(self):
        # Keyframes at frames 0 and 100, but only interpolate to frame 50
        keyframes = [(0, 0.0), (100, 100.0)]
        result = interpolate_schedule_values(keyframes, 50)
        # Should interpolate linearly up to frame 49
        assert result[0] == 0.0
        assert result[25] == 25.0
        assert result[49] == 49.0

    def test_very_large_keyframes(self):
        keyframes = [(0, 1.0), (1000000, 2.0)]
        result = interpolate_schedule_values(keyframes, 10)
        # All values should be very close to 1.0 since slope is tiny
        for val in result:
            assert 1.0 <= val < 1.00001


class TestIntegration:
    """Integration tests combining parsing and interpolation."""

    def test_parse_and_interpolate_simple(self):
        schedule_str = "0:(0.0), 10:(10.0)"
        keyframes = parse_schedule_string(schedule_str)
        result = interpolate_schedule_values(keyframes, 11)
        expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert result == expected

    def test_parse_and_interpolate_complex(self):
        schedule_str = "0:(1.0), 30:(2.0), 60:(1.5)"
        keyframes = parse_schedule_string(schedule_str)
        result = interpolate_schedule_values(keyframes, 61)

        # Check key points
        assert result[0] == 1.0
        assert result[30] == 2.0
        assert result[60] == 1.5

        # Check interpolation in first segment
        assert result[15] == 1.5  # Halfway between 1.0 and 2.0

        # Check interpolation in second segment
        assert result[45] == 1.75  # Halfway between 2.0 and 1.5

    def test_parse_empty_and_interpolate(self):
        keyframes = parse_schedule_string("")
        result = interpolate_schedule_values(keyframes, 10)
        assert result == [0.0] * 10

    def test_parse_single_value_and_interpolate(self):
        keyframes = parse_schedule_string("5.0")
        result = interpolate_schedule_values(keyframes, 10)
        assert result == [5.0] * 10

    def test_realistic_animation_schedule(self):
        # Realistic zoom animation schedule
        schedule_str = "0:(1.0), 120:(1.05), 240:(1.0)"
        keyframes = parse_schedule_string(schedule_str)
        result = interpolate_schedule_values(keyframes, 241)

        # Check start, peak, and end
        assert result[0] == 1.0
        assert result[120] == 1.05
        assert result[240] == 1.0

        # Check smooth interpolation
        assert result[60] == 1.025  # Halfway to peak
        assert result[180] == 1.025  # Halfway back down

    def test_multi_segment_camera_movement(self):
        # Pan left, hold, pan right
        schedule_str = "0:(0), 30:(-10), 60:(-10), 90:(0)"
        keyframes = parse_schedule_string(schedule_str)
        result = interpolate_schedule_values(keyframes, 91)

        assert result[0] == 0.0
        assert result[30] == -10.0
        assert result[60] == -10.0  # Hold position
        assert result[90] == 0.0

        # Check hold segment (frames 30-60)
        for i in range(30, 61):
            assert result[i] == -10.0

    def test_out_of_order_parsing_and_interpolation(self):
        # Parser should sort these
        schedule_str = "60:(3.0), 0:(1.0), 30:(2.0)"
        keyframes = parse_schedule_string(schedule_str)
        result = interpolate_schedule_values(keyframes, 61)

        assert result[0] == 1.0
        assert result[30] == 2.0
        assert result[60] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
