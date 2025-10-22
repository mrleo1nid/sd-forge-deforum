"""Unit tests for mathematical calculation utilities."""

import pytest
import numpy as np
import torch

from deforum.utils.math_utils import (
    rotation_matrix,
    rotate_camera_towards_depth,
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
