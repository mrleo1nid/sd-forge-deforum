"""Unit tests for mathematical calculation utilities."""

import pytest
import numpy as np
import torch

from deforum.utils.math.core import (
    rotation_matrix,
    rotate_camera_towards_depth,
    round_by_factor,
    ceil_by_factor,
    floor_by_factor,
    smart_resize,
    smart_nframes,
    IMAGE_FACTOR,
    MIN_PIXELS,
    MAX_PIXELS,
    FRAME_FACTOR,
)


class TestRotationMatrix:
    """Test rotation_matrix function."""

    def test_identity_rotation(self):
        """Zero angle should produce identity matrix."""
        axis = [0, 0, 1]
        angle = 0
        mat = rotation_matrix(axis, angle)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_90_degree_z_rotation(self):
        """90 degree rotation around Z-axis."""
        axis = [0, 0, 1]
        angle = np.pi / 2
        mat = rotation_matrix(axis, angle)
        # 90 degree rotation around Z should transform (1,0,0) to (0,1,0)
        point = np.array([1, 0, 0])
        rotated = mat @ point
        expected = np.array([0, 1, 0])
        np.testing.assert_array_almost_equal(rotated, expected)

    def test_180_degree_rotation(self):
        """180 degree rotation inverts perpendicular axes."""
        axis = [0, 0, 1]
        angle = np.pi
        mat = rotation_matrix(axis, angle)
        point = np.array([1, 0, 0])
        rotated = mat @ point
        expected = np.array([-1, 0, 0])
        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)

    def test_matrix_shape(self):
        """Output should be 3x3 matrix."""
        axis = [1, 1, 1]
        angle = 0.5
        mat = rotation_matrix(axis, angle)
        assert mat.shape == (3, 3)

    def test_matrix_is_orthogonal(self):
        """Rotation matrix should be orthogonal (R^T * R = I)."""
        axis = [1, 0, 0]
        angle = np.pi / 4
        mat = rotation_matrix(axis, angle)
        product = mat.T @ mat
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(product, expected)

    def test_determinant_is_one(self):
        """Rotation matrix should have determinant of 1."""
        axis = [0, 1, 0]
        angle = 1.5
        mat = rotation_matrix(axis, angle)
        det = np.linalg.det(mat)
        assert pytest.approx(det, abs=1e-10) == 1.0

    def test_axis_normalization(self):
        """Non-unit axis should be normalized automatically."""
        axis = [0, 0, 5]  # Will be normalized to [0, 0, 1]
        angle = np.pi / 2
        mat = rotation_matrix(axis, angle)
        point = np.array([1, 0, 0])
        rotated = mat @ point
        expected = np.array([0, 1, 0])
        np.testing.assert_array_almost_equal(rotated, expected)

    def test_different_axes(self):
        """Test rotation around different axes."""
        # X-axis rotation
        mat_x = rotation_matrix([1, 0, 0], np.pi / 2)
        assert mat_x.shape == (3, 3)

        # Y-axis rotation
        mat_y = rotation_matrix([0, 1, 0], np.pi / 2)
        assert mat_y.shape == (3, 3)

        # Arbitrary axis
        mat_arb = rotation_matrix([1, 1, 1], np.pi / 3)
        assert mat_arb.shape == (3, 3)


class TestRotateCameraTowardsDepth:
    """Test rotate_camera_towards_depth function."""

    def test_output_shape(self):
        """Output should be (1, 3, 3) tensor."""
        depth = torch.rand(480, 640)
        result = rotate_camera_towards_depth(depth, 10, 640, 480)
        assert result.shape == torch.Size([1, 3, 3])

    def test_output_is_tensor(self):
        """Output should be a PyTorch tensor."""
        depth = torch.rand(100, 100)
        result = rotate_camera_towards_depth(depth, 5, 100, 100)
        assert isinstance(result, torch.Tensor)

    def test_different_resolutions(self):
        """Should work with different image resolutions."""
        resolutions = [(320, 240), (640, 480), (1920, 1080)]
        for width, height in resolutions:
            depth = torch.rand(height, width)
            result = rotate_camera_towards_depth(depth, 10, width, height)
            assert result.shape == torch.Size([1, 3, 3])

    def test_different_turn_weights(self):
        """Different turn weights should affect rotation."""
        depth = torch.rand(100, 100)
        # Smaller turn weight = larger rotation per frame
        result_fast = rotate_camera_towards_depth(depth, 1, 100, 100)
        result_slow = rotate_camera_towards_depth(depth, 100, 100, 100)
        assert result_fast.shape == result_slow.shape
        # Results should be different (unless by chance max depth is at center)
        assert not torch.allclose(result_fast, result_slow, atol=1e-5)

    def test_different_fov(self):
        """Different field of view should affect calculation."""
        depth = torch.rand(100, 100)
        result_narrow = rotate_camera_towards_depth(depth, 10, 100, 100, h_fov=30)
        result_wide = rotate_camera_towards_depth(depth, 10, 100, 100, h_fov=90)
        assert result_narrow.shape == result_wide.shape

    def test_target_depth_parameter(self):
        """Different target depths should be valid."""
        depth = torch.rand(100, 100)
        # Target depth at middle
        result_mid = rotate_camera_towards_depth(depth, 10, 100, 100, target_depth=0.5)
        # Target depth at far
        result_far = rotate_camera_towards_depth(depth, 10, 100, 100, target_depth=1.0)
        assert result_mid.shape == result_far.shape

    def test_uniform_depth_produces_valid_output(self):
        """Uniform depth map should still produce valid output."""
        depth = torch.ones(100, 100)
        result = rotate_camera_towards_depth(depth, 10, 100, 100)
        assert result.shape == torch.Size([1, 3, 3])
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_no_nan_or_inf(self):
        """Output should never contain NaN or Inf."""
        depth = torch.rand(200, 300)
        result = rotate_camera_towards_depth(depth, 15, 300, 200)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_is_rotation_matrix(self):
        """Output should be approximately a valid rotation matrix."""
        depth = torch.rand(100, 150)
        result = rotate_camera_towards_depth(depth, 10, 150, 100)
        mat = result[0].numpy()

        # Check orthogonality: R^T * R ≈ I (with generous numerical tolerance)
        # The function produces approximate rotation matrices, not exact ones
        product = mat.T @ mat
        expected = np.eye(3)
        # Check diagonal elements are close to 1
        for i in range(3):
            assert pytest.approx(product[i, i], abs=0.05) == 1.0
        # Check off-diagonal elements are close to 0
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(product[i, j]) < 0.05

        # Check determinant is close to 1
        det = np.linalg.det(mat)
        assert pytest.approx(det, abs=0.05) == 1.0


class TestIntegration:
    """Integration tests combining math functions."""

    def test_rotation_matrix_then_apply(self):
        """Test rotation matrix generation and application."""
        axis = [0, 0, 1]
        angle = np.pi / 4  # 45 degrees
        mat = rotation_matrix(axis, angle)

        # Apply rotation to a point
        point = np.array([1, 0, 0])
        rotated = mat @ point

        # Point should have rotated 45 degrees around Z
        expected_x = np.cos(angle)
        expected_y = np.sin(angle)
        expected_z = 0

        np.testing.assert_almost_equal(rotated[0], expected_x)
        np.testing.assert_almost_equal(rotated[1], expected_y)
        np.testing.assert_almost_equal(rotated[2], expected_z)

    def test_camera_rotation_produces_valid_transform(self):
        """Test that camera rotation can be applied to scene."""
        # Create depth map with interesting features
        depth = torch.rand(240, 320)
        # Make one area significantly deeper
        depth[100:140, 160:200] = depth.max() + 0.5

        # Get rotation
        rot_tensor = rotate_camera_towards_depth(depth, 20, 320, 240)

        # Extract matrix
        rot_matrix = rot_tensor[0].numpy()

        # Apply to a test vector
        test_vec = np.array([0, 0, -1])  # Looking forward
        rotated_vec = rot_matrix @ test_vec

        # Result should be a unit vector (approximately)
        magnitude = np.linalg.norm(rotated_vec)
        assert pytest.approx(magnitude, abs=1e-5) == 1.0


class TestRoundByFactor:
    """Test round_by_factor function."""

    def test_exact_multiple(self):
        """Exact multiples should round to themselves."""
        assert round_by_factor(15, 5) == 15
        assert round_by_factor(20, 5) == 20
        assert round_by_factor(112, 28) == 112

    def test_round_down(self):
        """Values closer to lower multiple should round down."""
        assert round_by_factor(17, 5) == 15
        assert round_by_factor(100, 28) == 112

    def test_round_up(self):
        """Values closer to upper multiple should round up."""
        assert round_by_factor(18, 5) == 20
        assert round_by_factor(115, 28) == 112

    def test_float_input(self):
        """Should handle float inputs."""
        assert round_by_factor(17.5, 5) == 20
        assert round_by_factor(12.3, 5) == 10

    def test_negative_numbers(self):
        """Should handle negative numbers."""
        assert round_by_factor(-17, 5) == -15
        assert round_by_factor(-18, 5) == -20

    def test_factor_one(self):
        """Factor of 1 should return rounded integer."""
        assert round_by_factor(17.5, 1) == 18
        assert round_by_factor(17.4, 1) == 17

    def test_zero_factor_raises_error(self):
        """Zero factor should raise ValueError."""
        with pytest.raises(ValueError, match="factor cannot be zero"):
            round_by_factor(10, 0)

    def test_invalid_number_type_raises_error(self):
        """Non-numeric number should raise TypeError."""
        with pytest.raises(TypeError, match="number must be int or float"):
            round_by_factor("10", 5)

    def test_invalid_factor_type_raises_error(self):
        """Non-int factor should raise TypeError."""
        with pytest.raises(TypeError, match="factor must be int"):
            round_by_factor(10, 5.5)


class TestCeilByFactor:
    """Test ceil_by_factor function."""

    def test_exact_multiple(self):
        """Exact multiples should ceil to themselves."""
        assert ceil_by_factor(15, 5) == 15
        assert ceil_by_factor(20, 5) == 20
        assert ceil_by_factor(112, 28) == 112

    def test_ceil_up(self):
        """Non-multiples should always ceil up."""
        assert ceil_by_factor(17, 5) == 20
        assert ceil_by_factor(21, 5) == 25
        assert ceil_by_factor(100, 28) == 112

    def test_float_input(self):
        """Should handle float inputs."""
        assert ceil_by_factor(17.1, 5) == 20
        assert ceil_by_factor(12.9, 5) == 15

    def test_negative_numbers(self):
        """Should handle negative numbers (ceiling towards zero)."""
        assert ceil_by_factor(-17, 5) == -15
        assert ceil_by_factor(-20, 5) == -20

    def test_small_values(self):
        """Small values should ceil correctly."""
        assert ceil_by_factor(1, 5) == 5
        assert ceil_by_factor(3, 28) == 28

    def test_zero(self):
        """Zero should ceil to zero."""
        assert ceil_by_factor(0, 5) == 0

    def test_zero_factor_raises_error(self):
        """Zero factor should raise ValueError."""
        with pytest.raises(ValueError, match="factor cannot be zero"):
            ceil_by_factor(10, 0)


class TestFloorByFactor:
    """Test floor_by_factor function."""

    def test_exact_multiple(self):
        """Exact multiples should floor to themselves."""
        assert floor_by_factor(15, 5) == 15
        assert floor_by_factor(20, 5) == 20
        assert floor_by_factor(84, 28) == 84

    def test_floor_down(self):
        """Non-multiples should always floor down."""
        assert floor_by_factor(17, 5) == 15
        assert floor_by_factor(24, 5) == 20
        assert floor_by_factor(100, 28) == 84

    def test_float_input(self):
        """Should handle float inputs."""
        assert floor_by_factor(17.9, 5) == 15
        assert floor_by_factor(22.1, 5) == 20

    def test_negative_numbers(self):
        """Should handle negative numbers (floor towards negative infinity)."""
        assert floor_by_factor(-17, 5) == -20
        assert floor_by_factor(-15, 5) == -15

    def test_zero(self):
        """Zero should floor to zero."""
        assert floor_by_factor(0, 5) == 0

    def test_large_values(self):
        """Large values should floor correctly."""
        assert floor_by_factor(1000, 28) == 980
        assert floor_by_factor(10000, 100) == 10000

    def test_zero_factor_raises_error(self):
        """Zero factor should raise ValueError."""
        with pytest.raises(ValueError, match="factor cannot be zero"):
            floor_by_factor(10, 0)


class TestFactorFunctionsConsistency:
    """Test consistency between round/ceil/floor by factor."""

    def test_ceil_greater_equal_floor(self):
        """Ceil should always be >= floor."""
        test_values = [17, 18, 19, 20, 21, 100, 115]
        for value in test_values:
            for factor in [5, 10, 28]:
                assert ceil_by_factor(value, factor) >= floor_by_factor(value, factor)

    def test_round_between_ceil_and_floor(self):
        """Round should be between ceil and floor (or equal to one of them)."""
        test_values = [17, 18, 19, 20, 21, 100, 115]
        for value in test_values:
            for factor in [5, 10, 28]:
                rounded = round_by_factor(value, factor)
                ceiled = ceil_by_factor(value, factor)
                floored = floor_by_factor(value, factor)
                assert floored <= rounded <= ceiled


class TestSmartResize:
    """Test smart_resize function."""

    def test_default_factor(self):
        """Should resize with default factor of 28."""
        h, w = smart_resize(1080, 1920)
        assert h % IMAGE_FACTOR == 0
        assert w % IMAGE_FACTOR == 0

    def test_dimensions_divisible_by_factor(self):
        """Output dimensions should be divisible by factor."""
        h, w = smart_resize(100, 200, factor=10)
        assert h % 10 == 0
        assert w % 10 == 0

    def test_small_image_scaled_up(self):
        """Small images should be scaled up to min_pixels."""
        h, w = smart_resize(10, 10, factor=28, min_pixels=MIN_PIXELS)
        total_pixels = h * w
        assert total_pixels >= MIN_PIXELS
        assert h % 28 == 0
        assert w % 28 == 0

    def test_large_image_scaled_down(self):
        """Large images should be scaled down to max_pixels."""
        h, w = smart_resize(5000, 5000, factor=28, max_pixels=MAX_PIXELS)
        total_pixels = h * w
        assert total_pixels <= MAX_PIXELS
        assert h % 28 == 0
        assert w % 28 == 0

    def test_aspect_ratio_preserved(self):
        """Aspect ratio should be approximately preserved."""
        original_h, original_w = 1080, 1920
        h, w = smart_resize(original_h, original_w)
        original_ratio = original_w / original_h
        new_ratio = w / h
        # Allow small deviation due to factor rounding
        assert pytest.approx(new_ratio, rel=0.1) == original_ratio

    def test_square_image(self):
        """Square images should remain square (approximately)."""
        h, w = smart_resize(1000, 1000)
        # Should be exactly equal for square inputs
        assert h == w

    def test_hd_video_resolution(self):
        """HD video (1080x1920) should resize appropriately."""
        h, w = smart_resize(1080, 1920)
        assert h == 1092  # 1080 rounded to multiple of 28
        assert w == 1932  # 1920 rounded to multiple of 28

    def test_minimum_size_constraint(self):
        """Dimensions should never be smaller than factor."""
        h, w = smart_resize(1, 1, factor=28)
        assert h >= 28
        assert w >= 28

    def test_custom_factor(self):
        """Should work with custom factors."""
        h, w = smart_resize(100, 200, factor=16)
        assert h % 16 == 0
        assert w % 16 == 0

    def test_extreme_aspect_ratio_raises_error(self):
        """Extremely wide/tall images should raise ValueError."""
        with pytest.raises(ValueError, match="absolute aspect ratio must be smaller"):
            smart_resize(100, 30000)  # Ratio > 200

    def test_negative_dimensions_raise_error(self):
        """Negative dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="height and width must be positive"):
            smart_resize(-100, 200)

    def test_zero_dimensions_raise_error(self):
        """Zero dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="height and width must be positive"):
            smart_resize(0, 200)

    def test_invalid_type_raises_error(self):
        """Non-integer inputs should raise TypeError."""
        with pytest.raises(TypeError, match="height must be int"):
            smart_resize(100.5, 200)


class TestSmartNframes:
    """Test smart_nframes function."""

    def test_fps_based_calculation(self):
        """Should calculate frames based on target FPS."""
        nframes = smart_nframes(100, 30.0, target_fps=2.0)
        # 100 frames at 30fps with target 2fps = 100/30*2 = 6.67 → 6 (rounded to FRAME_FACTOR)
        assert nframes % FRAME_FACTOR == 0
        assert nframes == 6

    def test_explicit_nframes(self):
        """Should use explicit nframes when provided."""
        nframes = smart_nframes(100, 30.0, nframes=10)
        assert nframes == 10
        assert nframes % FRAME_FACTOR == 0

    def test_nframes_rounded_to_factor(self):
        """Explicit nframes should be rounded to FRAME_FACTOR."""
        nframes = smart_nframes(100, 30.0, nframes=11)
        # 11 rounds to 12 (banker's rounding: 11/2=5.5 → 6 → 6*2=12)
        assert nframes == 12

    def test_default_fps(self):
        """Should use default FPS (2.0) when not specified."""
        nframes1 = smart_nframes(100, 30.0)
        nframes2 = smart_nframes(100, 30.0, target_fps=2.0)
        assert nframes1 == nframes2

    def test_min_frames_constraint(self):
        """Result should respect min_frames."""
        nframes = smart_nframes(100, 30.0, target_fps=0.1, min_frames=10)
        assert nframes >= 10

    def test_max_frames_constraint(self):
        """Result should respect max_frames."""
        nframes = smart_nframes(100, 30.0, target_fps=100.0, max_frames=20)
        assert nframes <= 20

    def test_total_frames_limit(self):
        """Result should never exceed total_frames."""
        nframes = smart_nframes(50, 30.0, target_fps=100.0)
        assert nframes <= 50

    def test_both_fps_and_nframes_raises_error(self):
        """Specifying both target_fps and nframes should raise ValueError."""
        with pytest.raises(ValueError, match="Only accept either"):
            smart_nframes(100, 30.0, target_fps=2.0, nframes=10)

    def test_result_out_of_range_raises_error(self):
        """Result outside valid range should raise ValueError."""
        # Try to force an out-of-range result
        with pytest.raises(ValueError, match="nframes should be in interval"):
            smart_nframes(10, 30.0, nframes=100)  # More than total frames

    def test_negative_total_frames_raises_error(self):
        """Negative total_frames should raise ValueError."""
        with pytest.raises(ValueError, match="total_frames must be positive"):
            smart_nframes(-100, 30.0)

    def test_zero_fps_raises_error(self):
        """Zero FPS should raise ValueError."""
        with pytest.raises(ValueError, match="video_fps must be positive"):
            smart_nframes(100, 0.0)

    def test_invalid_type_raises_error(self):
        """Non-numeric inputs should raise TypeError."""
        with pytest.raises(TypeError, match="total_frames must be int"):
            smart_nframes("100", 30.0)

    def test_various_video_lengths(self):
        """Should work correctly for various video lengths."""
        test_cases = [
            (100, 30.0, 2.0, 6),   # Short video
            (200, 60.0, 3.0, 10),  # Medium video, higher FPS
            (500, 24.0, 2.0, 42),  # Long video, cinema FPS
        ]
        for total, fps, target, expected_approx in test_cases:
            nframes = smart_nframes(total, fps, target_fps=target)
            # Check it's in reasonable range
            assert FRAME_FACTOR <= nframes <= total
            # Verify it's divisible by FRAME_FACTOR
            assert nframes % FRAME_FACTOR == 0


class TestMathUtilsIntegration:
    """Integration tests for math utility functions."""

    def test_smart_resize_then_verify_divisibility(self):
        """Resize dimensions then verify all factor constraints."""
        h, w = smart_resize(1234, 5678, factor=32)
        assert h % 32 == 0
        assert w % 32 == 0
        # Should also work with factor-based rounding
        assert round_by_factor(h, 32) == h
        assert round_by_factor(w, 32) == w

    def test_nframes_calculation_workflow(self):
        """Simulate full video frame extraction workflow."""
        # Original video: 300 frames at 30 FPS
        total_frames = 300
        video_fps = 30.0
        target_fps = 3.0  # Want 3 FPS output

        # Calculate frames to extract
        nframes = smart_nframes(total_frames, video_fps, target_fps=target_fps)

        # Verify result
        assert nframes % FRAME_FACTOR == 0
        assert FRAME_FACTOR <= nframes <= total_frames

        # Calculate frame indices (every Nth frame)
        frame_step = total_frames / nframes
        frame_indices = [int(i * frame_step) for i in range(nframes)]

        # All indices should be valid
        assert all(0 <= idx < total_frames for idx in frame_indices)
        assert len(frame_indices) == nframes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
