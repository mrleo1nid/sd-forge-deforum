"""Unit tests for deforum.utils.camera_analysis_utils module."""

import pytest

from deforum.utils.camera.analysis import (
    analyze_frame_ranges,
    generate_segment_description,
    group_similar_segments,
    generate_group_description,
    generate_rotation_group_description,
)


class TestAnalyzeFrameRanges:
    """Tests for analyze_frame_ranges function."""

    def test_detects_increasing_movement(self):
        """Should detect continuous increasing movement."""
        values = [0.0, 0.1, 0.2, 0.3, 0.4]
        segments = analyze_frame_ranges(values, "translation_x")

        assert len(segments) == 1
        assert segments[0]["direction"] == "increasing"
        assert segments[0]["movement_type"] == "translation_x"

    def test_detects_decreasing_movement(self):
        """Should detect continuous decreasing movement."""
        values = [1.0, 0.8, 0.6, 0.4, 0.2]
        segments = analyze_frame_ranges(values, "translation_y")

        assert len(segments) == 1
        assert segments[0]["direction"] == "decreasing"

    def test_detects_direction_changes(self):
        """Should detect when movement direction changes."""
        values = [0.0, 0.2, 0.4, 0.3, 0.1]  # Up then down
        segments = analyze_frame_ranges(values, "translation_x")

        assert len(segments) == 2
        assert segments[0]["direction"] == "increasing"
        assert segments[1]["direction"] == "decreasing"

    def test_ignores_small_changes(self):
        """Should ignore changes below threshold."""
        values = [0.0, 0.0001, 0.0002, 0.0003]  # Very small changes
        segments = analyze_frame_ranges(values, "translation_x")

        # Default threshold is 0.001, so these should be ignored
        assert len(segments) == 0

    def test_sensitivity_parameter(self):
        """Should respect sensitivity parameter."""
        values = [0.0, 0.0005, 0.001, 0.0015]  # Small changes

        # High sensitivity (lower threshold)
        segments_sensitive = analyze_frame_ranges(values, "translation_x", sensitivity=0.5)

        # Normal sensitivity
        segments_normal = analyze_frame_ranges(values, "translation_x", sensitivity=1.0)

        assert len(segments_sensitive) >= len(segments_normal)

    def test_different_thresholds_for_movement_types(self):
        """Should use different thresholds for different movement types."""
        values = [0.0, 0.0005, 0.001]

        # Zoom has lower threshold (0.0001) than translation (0.001)
        zoom_segments = analyze_frame_ranges(values, "zoom")
        translation_segments = analyze_frame_ranges(values, "translation_x")

        # Zoom should detect more segments due to lower threshold
        assert len(zoom_segments) >= len(translation_segments)

    def test_empty_values_list(self):
        """Should handle empty values list."""
        segments = analyze_frame_ranges([], "translation_x")
        assert segments == []

    def test_single_value(self):
        """Should handle single value (no movement possible)."""
        segments = analyze_frame_ranges([1.0], "translation_x")
        assert segments == []

    def test_segment_contains_required_keys(self):
        """Segments should contain all required keys."""
        values = [0.0, 1.0, 2.0]
        segments = analyze_frame_ranges(values, "translation_x")

        assert len(segments) > 0
        segment = segments[0]

        required_keys = {"start_frame", "end_frame", "direction",
                        "movement_type", "max_change", "total_range"}
        assert required_keys.issubset(segment.keys())

    def test_calculates_total_range(self):
        """Should calculate total range of movement."""
        values = [0.0, 1.0, 2.0, 3.0]
        segments = analyze_frame_ranges(values, "translation_x")

        assert segments[0]["total_range"] == pytest.approx(3.0)


class TestGenerateSegmentDescription:
    """Tests for generate_segment_description function."""

    def test_describes_translation_x(self):
        """Should generate correct description for X translation."""
        segment = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
            "total_range": 5.0,
        }
        desc = generate_segment_description(segment, 100)

        assert "panning right" in desc

    def test_describes_translation_y(self):
        """Should generate correct description for Y translation."""
        segment = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_y",
            "direction": "decreasing",
            "total_range": 3.0,
        }
        desc = generate_segment_description(segment, 100)

        assert "moving down" in desc

    def test_describes_rotation(self):
        """Should generate correct description for rotation."""
        segment = {
            "start_frame": 0,
            "end_frame": 20,
            "movement_type": "rotation_x",
            "direction": "increasing",
            "total_range": 10.0,
        }
        desc = generate_segment_description(segment, 100)

        assert "tilting up" in desc

    def test_describes_zoom(self):
        """Should generate correct description for zoom."""
        segment = {
            "start_frame": 0,
            "end_frame": 15,
            "movement_type": "zoom",
            "direction": "increasing",
            "total_range": 2.0,
        }
        desc = generate_segment_description(segment, 100)

        assert "zooming in" in desc

    def test_intensity_subtle(self):
        """Should describe subtle movement."""
        segment = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
            "total_range": 0.5,  # < 1.0 = subtle
        }
        desc = generate_segment_description(segment, 100)

        assert "subtle" in desc

    def test_intensity_gentle(self):
        """Should describe gentle movement."""
        segment = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
            "total_range": 3.0,  # 1-5 = gentle
        }
        desc = generate_segment_description(segment, 100)

        assert "gentle" in desc

    def test_intensity_moderate(self):
        """Should describe moderate movement."""
        segment = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
            "total_range": 10.0,  # 5-20 = moderate
        }
        desc = generate_segment_description(segment, 100)

        assert "moderate" in desc

    def test_intensity_strong(self):
        """Should describe strong movement."""
        segment = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
            "total_range": 50.0,  # > 20 = strong
        }
        desc = generate_segment_description(segment, 100)

        assert "strong" in desc

    def test_frame_range_brief(self):
        """Should indicate brief duration."""
        segment = {
            "start_frame": 0,
            "end_frame": 15,  # 15% of 100 frames = brief
            "movement_type": "translation_x",
            "direction": "increasing",
            "total_range": 5.0,
        }
        desc = generate_segment_description(segment, 100)

        assert "brief" in desc

    def test_frame_range_extended(self):
        """Should indicate extended duration."""
        segment = {
            "start_frame": 0,
            "end_frame": 50,  # 50% of 100 frames = moderate duration
            "movement_type": "translation_x",
            "direction": "increasing",
            "total_range": 5.0,
        }
        desc = generate_segment_description(segment, 100)

        assert "moderate" in desc or "extended" in desc


class TestGroupSimilarSegments:
    """Tests for group_similar_segments function."""

    def test_groups_same_type_and_direction(self):
        """Should group segments with same type and direction."""
        seg1 = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
        }
        seg2 = {
            "start_frame": 15,
            "end_frame": 25,
            "movement_type": "translation_x",
            "direction": "increasing",
        }

        groups = group_similar_segments([seg1, seg2], 100)

        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_separates_different_directions(self):
        """Should separate segments with different directions."""
        seg1 = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
        }
        seg2 = {
            "start_frame": 15,
            "end_frame": 25,
            "movement_type": "translation_x",
            "direction": "decreasing",
        }

        groups = group_similar_segments([seg1, seg2], 100)

        assert len(groups) == 2

    def test_separates_different_movement_types(self):
        """Should separate segments with different movement types."""
        seg1 = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
        }
        seg2 = {
            "start_frame": 15,
            "end_frame": 25,
            "movement_type": "translation_y",
            "direction": "increasing",
        }

        groups = group_similar_segments([seg1, seg2], 100)

        assert len(groups) == 2

    def test_separates_distant_segments(self):
        """Should separate segments that are far apart."""
        seg1 = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
        }
        seg2 = {
            "start_frame": 50,  # Far from seg1 (> 10% of 100 frames)
            "end_frame": 60,
            "movement_type": "translation_x",
            "direction": "increasing",
        }

        groups = group_similar_segments([seg1, seg2], 100)

        assert len(groups) == 2

    def test_custom_proximity_threshold(self):
        """Should respect custom proximity threshold."""
        seg1 = {
            "start_frame": 0,
            "end_frame": 10,
            "movement_type": "translation_x",
            "direction": "increasing",
        }
        seg2 = {
            "start_frame": 25,  # 15 frames apart
            "end_frame": 35,
            "movement_type": "translation_x",
            "direction": "increasing",
        }

        # With tight threshold, should be separate
        groups_tight = group_similar_segments([seg1, seg2], 100, proximity_threshold=0.05)

        # With loose threshold, should be grouped
        groups_loose = group_similar_segments([seg1, seg2], 100, proximity_threshold=0.2)

        assert len(groups_tight) == 2
        assert len(groups_loose) == 1

    def test_empty_segments_list(self):
        """Should handle empty segments list."""
        groups = group_similar_segments([], 100)
        assert groups == []


class TestGenerateGroupDescription:
    """Tests for generate_group_description function."""

    def test_combines_segment_ranges(self):
        """Should combine ranges from multiple segments."""
        segments = [
            {
                "start_frame": 0,
                "end_frame": 10,
                "movement_type": "translation_x",
                "direction": "increasing",
                "total_range": 2.0,
            },
            {
                "start_frame": 15,
                "end_frame": 25,
                "movement_type": "translation_x",
                "direction": "increasing",
                "total_range": 3.0,
            },
        ]

        desc, strength = generate_group_description(segments, 100)

        assert "panning right" in desc
        assert strength > 0.0

    def test_calculates_strength(self):
        """Should calculate movement strength."""
        segments = [
            {
                "start_frame": 0,
                "end_frame": 50,
                "movement_type": "translation_x",
                "direction": "increasing",
                "total_range": 10.0,
            }
        ]

        _, strength = generate_group_description(segments, 100)

        # Strength = (total_range * duration) / (max_frames * 10)
        # = (10.0 * 51) / (100 * 10) = 0.51
        assert strength == pytest.approx(0.51)

    def test_empty_group(self):
        """Should handle empty group."""
        desc, strength = generate_group_description([], 100)

        assert desc == ""
        assert strength == 0.0

    def test_brief_duration(self):
        """Should describe brief duration."""
        segments = [
            {
                "start_frame": 0,
                "end_frame": 10,  # < 20% of 100
                "movement_type": "translation_x",
                "direction": "increasing",
                "total_range": 5.0,
            }
        ]

        desc, _ = generate_group_description(segments, 100)

        assert "brief" in desc


class TestGenerateRotationGroupDescription:
    """Tests for generate_rotation_group_description function."""

    def test_describes_tilt(self):
        """Should describe tilt movement."""
        segments = [
            {
                "start_frame": 0,
                "end_frame": 20,
                "movement_type": "rotation_x",
                "direction": "increasing",
                "total_range": 5.0,
            }
        ]

        desc, strength = generate_rotation_group_description(segments, 100)

        assert "tilting up" in desc
        assert strength > 0.0

    def test_describes_rotate(self):
        """Should describe rotate movement."""
        segments = [
            {
                "start_frame": 0,
                "end_frame": 20,
                "movement_type": "rotation_y",
                "direction": "decreasing",
                "total_range": 8.0,
            }
        ]

        desc, _ = generate_rotation_group_description(segments, 100)

        assert "rotating left" in desc

    def test_describes_roll(self):
        """Should describe roll movement."""
        segments = [
            {
                "start_frame": 0,
                "end_frame": 20,
                "movement_type": "rotation_z",
                "direction": "increasing",
                "total_range": 3.0,
            }
        ]

        desc, _ = generate_rotation_group_description(segments, 100)

        assert "rolling clockwise" in desc

    def test_empty_group(self):
        """Should handle empty group."""
        desc, strength = generate_rotation_group_description([], 100)

        assert desc == ""
        assert strength == 0.0


class TestIntegration:
    """Integration tests for camera analysis pipeline."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        # Create movement data
        values = [0.0, 1.0, 2.0, 3.0, 2.5, 2.0, 1.5, 1.0]

        # Step 1: Analyze frame ranges
        segments = analyze_frame_ranges(values, "translation_x")

        assert len(segments) == 2  # One increasing, one decreasing

        # Step 2: Generate descriptions
        desc1 = generate_segment_description(segments[0], len(values))
        desc2 = generate_segment_description(segments[1], len(values))

        assert "panning right" in desc1
        assert "panning left" in desc2

        # Step 3: Group segments
        groups = group_similar_segments(segments, len(values))

        assert len(groups) == 2  # Different directions

        # Step 4: Generate group descriptions
        for group in groups:
            desc, strength = generate_group_description(group, len(values))
            assert len(desc) > 0
            assert strength >= 0.0

    def test_complex_movement_pattern(self):
        """Test complex multi-axis movement."""
        # X translation: steady right pan
        x_values = [i * 0.5 for i in range(20)]

        # Y translation: up then down
        y_values = [i * 0.3 for i in range(10)] + [10 - i * 0.3 for i in range(10)]

        x_segments = analyze_frame_ranges(x_values, "translation_x")
        y_segments = analyze_frame_ranges(y_values, "translation_y")

        assert len(x_segments) == 1  # Continuous right pan
        assert len(y_segments) == 2  # Up then down

        all_segments = x_segments + y_segments
        groups = group_similar_segments(all_segments, 20)

        assert len(groups) == 3  # One X group, two Y groups
