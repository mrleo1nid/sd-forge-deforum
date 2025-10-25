"""Unit tests for animation transformation functions."""

import pytest
import numpy as np
import torch
import cv2

from deforum.utils.image.transforms import (
    normalize_image_to_tensor_range,
    denormalize_tensor_to_image_range,
    sample_from_cv2,
    sample_to_cv2,
    construct_rotation_matrix_rodrigues,
    create_rotation_matrix_x,
    create_rotation_matrix_y,
    create_rotation_matrix_z,
    get_rotation_matrix_manual,
    extract_2d_points,
    get_perspective_transform_points,
    calculate_fov_parameters,
    create_translation_matrix_z,
    create_projection_matrix,
    create_corner_points,
    warpMatrix,
)


class TestImageTensorConversions:
    """Test image/tensor conversion functions."""

    def test_normalize_image_to_tensor_range(self):
        # Test with simple values
        img = np.array([0, 127, 255], dtype=np.uint8)
        result = normalize_image_to_tensor_range(img)
        assert result[0] == pytest.approx(-1.0)
        assert result[1] == pytest.approx(-0.00392, abs=0.01)  # close to 0
        assert result[2] == pytest.approx(1.0)

    def test_denormalize_tensor_to_image_range(self):
        # Test with tensor range values
        tensor = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = denormalize_tensor_to_image_range(tensor)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_normalize_denormalize_roundtrip(self):
        # Test roundtrip conversion
        original = np.array([0, 128, 255], dtype=np.uint8)
        normalized = normalize_image_to_tensor_range(original)
        denormalized = denormalize_tensor_to_image_range(normalized)
        back_to_uint8 = (denormalized * 255).astype(np.uint8)
        np.testing.assert_array_almost_equal(original, back_to_uint8, decimal=0)

    def test_sample_from_cv2_shape(self):
        # Test conversion from CV2 to PyTorch tensor
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor = sample_from_cv2(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 480, 640)  # NCHW format
        assert tensor.dtype == torch.float16

    def test_sample_to_cv2_shape(self):
        # Test conversion from PyTorch tensor to CV2
        tensor = torch.randn(1, 3, 480, 640)
        img = sample_to_cv2(tensor)
        assert isinstance(img, np.ndarray)
        assert img.shape == (480, 640, 3)  # HWC format
        assert img.dtype == np.uint8

    def test_sample_conversion_roundtrip(self):
        # Test roundtrip conversion (won't be exact due to float precision)
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        tensor = sample_from_cv2(original)
        back = sample_to_cv2(tensor)
        # Should be close but not exact due to precision loss
        assert back.shape == original.shape
        assert back.dtype == original.dtype


class TestRotationMatrices:
    """Test rotation matrix creation functions."""

    def test_create_rotation_matrix_x_identity(self):
        # Zero rotation should be identity-like
        mat = create_rotation_matrix_x(0.0)
        expected = np.eye(4, 4)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_create_rotation_matrix_x_90deg(self):
        # 90 degree rotation around X
        mat = create_rotation_matrix_x(np.pi / 2)
        # Check structure (not exact values due to numerical precision)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(0.0, abs=1e-7)
        assert mat[2, 2] == pytest.approx(0.0, abs=1e-7)

    def test_create_rotation_matrix_y_identity(self):
        mat = create_rotation_matrix_y(0.0)
        expected = np.eye(4, 4)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_create_rotation_matrix_z_identity(self):
        mat = create_rotation_matrix_z(0.0)
        expected = np.eye(4, 4)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_rotation_matrices_are_4x4(self):
        assert create_rotation_matrix_x(0.5).shape == (4, 4)
        assert create_rotation_matrix_y(0.5).shape == (4, 4)
        assert create_rotation_matrix_z(0.5).shape == (4, 4)

    def test_construct_rotation_matrix_rodrigues(self):
        # Test with zero rotation
        angles = [0.0, 0.0, 0.0]
        mat = construct_rotation_matrix_rodrigues(angles)
        assert mat.shape == (4, 4)
        # Top-left 3x3 should be close to identity
        np.testing.assert_array_almost_equal(mat[:3, :3], np.eye(3, 3), decimal=5)

    def test_construct_rotation_matrix_rodrigues_invalid(self):
        # Test with invalid input
        with pytest.raises(ValueError):
            construct_rotation_matrix_rodrigues([0.0, 0.0])  # Need 3 angles
        with pytest.raises(ValueError):
            construct_rotation_matrix_rodrigues("not a list")

    def test_get_rotation_matrix_manual_zero(self):
        # Zero rotation should give identity matrix
        mat = get_rotation_matrix_manual([0.0, 0.0, 0.0])
        expected = np.eye(4, 4)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_get_rotation_matrix_manual_shape(self):
        mat = get_rotation_matrix_manual([10.0, 20.0, 30.0])
        assert mat.shape == (4, 4)


class TestPerspectiveTransforms:
    """Test perspective transformation helper functions."""

    def test_extract_2d_points_shape(self):
        # Create mock 3D points output
        pts_3d = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        result = extract_2d_points(pts_3d)
        assert result.shape == (4, 2)
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[3], [7.0, 8.0])

    def test_get_perspective_transform_points_types(self):
        # Create mock input/output points
        pts_in = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
        pts_out = np.array([[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]])
        pin, pout = get_perspective_transform_points(pts_in, pts_out, 640, 480, 800.0)

        # Verify float32 output
        assert pin.dtype == np.float32
        assert pout.dtype == np.float32
        assert pin.shape == (4, 2)
        assert pout.shape == (4, 2)


class TestFOVAndProjection:
    """Test field of view and projection matrix functions."""

    def test_calculate_fov_parameters_basic(self):
        width, height, fov_deg, scale = 640, 480, 60.0, 1.0
        side_length, focal_dist, near, far = calculate_fov_parameters(width, height, fov_deg, scale)

        # Verify types
        assert isinstance(side_length, (float, np.floating))
        assert isinstance(focal_dist, (float, np.floating))
        assert isinstance(near, (float, np.floating))
        assert isinstance(far, (float, np.floating))

        # Verify relationships
        assert far > near
        assert focal_dist > 0
        assert side_length > 0

    def test_calculate_fov_parameters_scale(self):
        # Larger scale should give larger side_length
        _, _, _, _ = calculate_fov_parameters(640, 480, 60.0, 1.0)
        sl2, _, _, _ = calculate_fov_parameters(640, 480, 60.0, 2.0)
        sl1, _, _, _ = calculate_fov_parameters(640, 480, 60.0, 1.0)
        assert sl2 > sl1

    def test_create_translation_matrix_z_zero(self):
        mat = create_translation_matrix_z(0.0)
        expected = np.eye(4, 4)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_create_translation_matrix_z_value(self):
        mat = create_translation_matrix_z(-10.0)
        assert mat[2, 3] == -10.0
        # Other elements should be identity
        assert mat[0, 0] == 1.0
        assert mat[1, 1] == 1.0
        assert mat[3, 3] == 1.0

    def test_create_projection_matrix_shape(self):
        mat = create_projection_matrix(np.deg2rad(30.0), 0.1, 100.0)
        assert mat.shape == (4, 4)
        assert mat[3, 2] == -1.0  # Perspective projection marker

    def test_create_projection_matrix_fov(self):
        # Test different FOV values
        mat1 = create_projection_matrix(np.deg2rad(30.0), 0.1, 100.0)
        mat2 = create_projection_matrix(np.deg2rad(60.0), 0.1, 100.0)

        # Larger FOV should give smaller scale factors
        assert mat2[0, 0] < mat1[0, 0]


class TestCornerPoints:
    """Test corner points generation."""

    def test_create_corner_points_shape(self):
        points = create_corner_points(640, 480)
        assert points.shape == (1, 4, 3)

    def test_create_corner_points_values(self):
        points = create_corner_points(640, 480)
        pts = points[0]

        # Check corners are in expected positions
        # Top-left
        assert pts[0][0] == -320.0
        assert pts[0][1] == 240.0
        assert pts[0][2] == 0.0

        # Top-right
        assert pts[1][0] == 320.0
        assert pts[1][1] == 240.0

        # Bottom-right
        assert pts[2][0] == 320.0
        assert pts[2][1] == -240.0

        # Bottom-left
        assert pts[3][0] == -320.0
        assert pts[3][1] == -240.0


class TestWarpMatrix:
    """Test complete warp matrix calculation."""

    def test_warp_matrix_basic(self):
        # Basic test with zero rotation
        M33, side_length = warpMatrix(640, 480, 0.0, 0.0, 0.0, 1.0, 60.0)

        # Should return 3x3 matrix
        assert M33.shape == (3, 3)
        assert isinstance(side_length, (float, np.floating))
        assert side_length > 0

    def test_warp_matrix_with_rotation(self):
        # Test with rotation angles
        M33, side_length = warpMatrix(640, 480, 10.0, 20.0, 30.0, 1.0, 60.0)

        assert M33.shape == (3, 3)
        assert side_length > 0

    def test_warp_matrix_scale_effect(self):
        # Different scales should give different side lengths
        _, sl1 = warpMatrix(640, 480, 0.0, 0.0, 0.0, 1.0, 60.0)
        _, sl2 = warpMatrix(640, 480, 0.0, 0.0, 0.0, 2.0, 60.0)

        assert sl2 > sl1

    def test_warp_matrix_fov_effect(self):
        # Different FOV should affect output
        M1, sl1 = warpMatrix(640, 480, 0.0, 0.0, 0.0, 1.0, 30.0)
        M2, sl2 = warpMatrix(640, 480, 0.0, 0.0, 0.0, 1.0, 60.0)

        # Different FOV should give different side lengths
        assert sl1 != sl2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
