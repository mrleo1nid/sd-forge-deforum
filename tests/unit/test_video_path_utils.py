"""Unit tests for video and image path generation utilities."""

import os
import tempfile
import pytest

from deforum.utils.video_path_utils import (
    get_next_frame_path,
    get_output_video_path,
)


class TestGetNextFramePath:
    """Test get_next_frame_path function."""

    def test_input_frame_basic(self):
        result = get_next_frame_path("/output", "video", 0, False)
        assert result == os.path.join("/output", "inputframes", "video000000000.jpg")

    def test_mask_frame_basic(self):
        result = get_next_frame_path("/output", "video", 0, True)
        assert result == os.path.join("/output", "maskframes", "video000000000.jpg")

    def test_frame_index_padding(self):
        # Frame index should be zero-padded to 9 digits
        result = get_next_frame_path("/out", "vid", 42, False)
        assert result == os.path.join("/out", "inputframes", "vid000000042.jpg")

    def test_large_frame_index(self):
        result = get_next_frame_path("/data", "test", 123456789, False)
        assert result == os.path.join("/data", "inputframes", "test123456789.jpg")

    def test_zero_frame_index(self):
        result = get_next_frame_path("/dir", "name", 0, False)
        assert "name000000000.jpg" in result

    def test_different_video_names(self):
        names = ["myvideo", "test_clip", "animation-final", "vid.backup"]
        for name in names:
            result = get_next_frame_path("/out", name, 10, False)
            assert name in result
            assert "000000010.jpg" in result

    def test_different_outdirs(self):
        dirs = ["/output", "/data/frames", "/tmp/work", "relative/path"]
        for dir_path in dirs:
            result = get_next_frame_path(dir_path, "vid", 5, False)
            assert result.startswith(dir_path)

    def test_mask_vs_input_subdirectory(self):
        input_result = get_next_frame_path("/out", "vid", 10, False)
        mask_result = get_next_frame_path("/out", "vid", 10, True)

        assert "inputframes" in input_result
        assert "maskframes" in mask_result
        assert "inputframes" not in mask_result
        assert "maskframes" not in input_result

    def test_path_separator_consistency(self):
        # Should use os.path.join for platform-independent paths
        result = get_next_frame_path("/base", "video", 0, False)
        expected = os.path.join("/base", "inputframes", "video000000000.jpg")
        assert result == expected

    def test_filename_format(self):
        result = get_next_frame_path("/out", "myvid", 123, False)
        # Should end with videoname + 9-digit number + .jpg
        assert result.endswith("myvid000000123.jpg")

    def test_frame_index_range(self):
        # Test various frame indices
        indices = [0, 1, 10, 100, 1000, 10000, 100000, 1000000]
        for idx in indices:
            result = get_next_frame_path("/out", "v", idx, False)
            # Check that index is zero-padded to at least 9 digits
            filename = os.path.basename(result)
            number_part = filename.replace("v", "").replace(".jpg", "")
            assert len(number_part) >= 9


class TestGetOutputVideoPath:
    """Test get_output_video_path function."""

    def test_basic_path_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "image.png")
            result = get_output_video_path(input_path)

            # Should use parent folder name as video name
            folder_name = os.path.basename(tmpdir)
            expected = os.path.join(tmpdir, f"{folder_name}.mp4")
            assert result == expected

    def test_path_with_existing_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing video file
            folder_name = os.path.basename(tmpdir)
            existing_video = os.path.join(tmpdir, f"{folder_name}.mp4")
            open(existing_video, 'a').close()

            input_path = os.path.join(tmpdir, "image.png")
            result = get_output_video_path(input_path)

            # Should append _1 suffix to avoid collision
            expected = os.path.join(tmpdir, f"{folder_name}_1.mp4")
            assert result == expected

    def test_multiple_existing_videos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder_name = os.path.basename(tmpdir)

            # Create multiple existing video files
            for i in range(3):
                suffix = f"_{i}" if i > 0 else ""
                video_path = os.path.join(tmpdir, f"{folder_name}{suffix}.mp4")
                open(video_path, 'a').close()

            input_path = os.path.join(tmpdir, "image.png")
            result = get_output_video_path(input_path)

            # Should use _3 since _0, _1, _2 exist (_0 is the base name)
            expected = os.path.join(tmpdir, f"{folder_name}_3.mp4")
            assert result == expected

    def test_pattern_path_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Input path with printf-style pattern
            input_path = os.path.join(tmpdir, "frame%05d.png")
            result = get_output_video_path(input_path)

            folder_name = os.path.basename(tmpdir)
            expected = os.path.join(tmpdir, f"{folder_name}.mp4")
            assert result == expected

    def test_deep_nested_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c", "frames")
            os.makedirs(nested, exist_ok=True)

            input_path = os.path.join(nested, "img001.jpg")
            result = get_output_video_path(input_path)

            # Should use immediate parent folder name ("frames")
            expected = os.path.join(nested, "frames.mp4")
            assert result == expected

    def test_always_returns_mp4_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
            for ext in extensions:
                input_path = os.path.join(tmpdir, f"image{ext}")
                result = get_output_video_path(input_path)
                assert result.endswith(".mp4")

    def test_unique_suffix_increment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder_name = os.path.basename(tmpdir)

            # Create gaps in numbering (e.g., have _1 and _3 but not _2)
            # Function should still find next available number
            open(os.path.join(tmpdir, f"{folder_name}.mp4"), 'a').close()
            open(os.path.join(tmpdir, f"{folder_name}_1.mp4"), 'a').close()

            input_path = os.path.join(tmpdir, "image.png")
            result = get_output_video_path(input_path)

            # Should use _2 (next available)
            expected = os.path.join(tmpdir, f"{folder_name}_2.mp4")
            assert result == expected

    def test_preserves_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "myproject", "renders")
            os.makedirs(subdir, exist_ok=True)

            input_path = os.path.join(subdir, "frame_000.png")
            result = get_output_video_path(input_path)

            # Result should be in same directory as input
            assert os.path.dirname(result) == subdir


class TestIntegration:
    """Integration tests combining video path utilities."""

    def test_frame_and_video_path_workflow(self):
        """Test generating frame paths and then video output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate frame paths for a sequence
            video_name = "animation"
            frame_count = 100

            frame_paths = [
                get_next_frame_path(tmpdir, video_name, i, False)
                for i in range(frame_count)
            ]

            # Verify all paths are unique
            assert len(frame_paths) == len(set(frame_paths))

            # Verify paths are sequential
            for i, path in enumerate(frame_paths):
                assert f"{video_name}{i:09d}.jpg" in path

            # Generate output video path from first frame
            # Note: frame path includes /inputframes/ subdirectory, so video will be there too
            video_path = get_output_video_path(frame_paths[0])
            assert video_path.endswith(".mp4")
            # Video should be in same directory as frames (inputframes subdirectory)
            expected_dir = os.path.join(tmpdir, "inputframes")
            assert os.path.dirname(video_path) == expected_dir

    def test_mask_and_input_frames_separation(self):
        """Test that mask and input frames use different subdirectories."""
        outdir = "/output"
        video_name = "test"
        frame_idx = 50

        input_path = get_next_frame_path(outdir, video_name, frame_idx, False)
        mask_path = get_next_frame_path(outdir, video_name, frame_idx, True)

        # Paths should differ only in subdirectory
        assert "inputframes" in input_path
        assert "maskframes" in mask_path
        assert os.path.basename(input_path) == os.path.basename(mask_path)

    def test_complete_video_generation_workflow(self):
        """Test complete workflow from frames to video output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup directories
            input_dir = os.path.join(tmpdir, "inputframes")
            os.makedirs(input_dir, exist_ok=True)

            video_name = "final_render"

            # Generate frame paths
            frames = []
            for i in range(10):
                frame_path = get_next_frame_path(tmpdir, video_name, i, False)
                frames.append(frame_path)

                # Simulate creating the frame file
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                open(frame_path, 'a').close()

            # Get output video path
            video_path = get_output_video_path(frames[0])

            # Verify video path is valid
            # Video will be in inputframes subdirectory since frames are there
            expected_dir = os.path.join(tmpdir, "inputframes")
            assert os.path.dirname(video_path) == expected_dir
            assert video_path.endswith(".mp4")

            # Create video file
            open(video_path, 'a').close()

            # Try to get another video path - should increment
            video_path_2 = get_output_video_path(frames[0])
            assert video_path != video_path_2
            assert "_1.mp4" in video_path_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
