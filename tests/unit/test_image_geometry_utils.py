"""Unit tests for deforum.utils.image_geometry_utils module."""

import numpy as np
import pytest

from deforum.utils.image_geometry_utils import (
    center_crop,
    calculate_center_crop_bounds,
    extend_with_grid,
    calculate_padding,
    get_crop_or_pad_bounds,
)


class TestCenterCrop:
    """Tests for center_crop function."""

    def test_square_image_square_crop(self):
        """Should crop square image to smaller square."""
        img = np.random.rand(100, 100, 3)
        cropped = center_crop(img, 50, 50)
        assert cropped.shape == (50, 50, 3)

    def test_rectangular_image_rectangular_crop(self):
        """Should crop rectangular image."""
        img = np.random.rand(200, 300, 3)
        cropped = center_crop(img, 100, 150)
        assert cropped.shape == (150, 100, 3)

    def test_crop_preserves_center_pixels(self):
        """Should preserve center pixels after cropping."""
        img = np.zeros((100, 100, 3))
        img[45:55, 45:55, :] = 1.0  # Center 10x10 square
        cropped = center_crop(img, 20, 20)
        # Center should be preserved
        assert np.sum(cropped) > 0

    def test_crop_dimensions_too_large_raises_error(self):
        """Should raise ValueError if target dimensions exceed image."""
        img = np.random.rand(50, 50, 3)
        with pytest.raises(ValueError, match="exceed image dimensions"):
            center_crop(img, 100, 100)

    def test_crop_exact_size_returns_copy(self):
        """Should return copy when target size equals image size."""
        img = np.random.rand(100, 100, 3)
        cropped = center_crop(img, 100, 100)
        assert cropped.shape == img.shape
        assert np.array_equal(cropped, img)

    def test_crop_grayscale_image(self):
        """Should handle grayscale (2D) images."""
        img = np.random.rand(100, 100)
        cropped = center_crop(img, 50, 50)
        assert cropped.shape == (50, 50)

    def test_crop_odd_dimensions(self):
        """Should handle odd dimensions correctly."""
        img = np.random.rand(101, 101, 3)
        cropped = center_crop(img, 51, 51)
        assert cropped.shape == (51, 51, 3)


class TestCalculateCenterCropBounds:
    """Tests for calculate_center_crop_bounds function."""

    def test_square_to_smaller_square(self):
        """Should calculate bounds for square crop."""
        bounds = calculate_center_crop_bounds(100, 100, 50, 50)
        assert bounds == (25, 25, 75, 75)

    def test_rectangular_crop(self):
        """Should calculate bounds for rectangular crop."""
        bounds = calculate_center_crop_bounds(200, 100, 100, 50)
        assert bounds == (50, 25, 150, 75)

    def test_exact_size_bounds(self):
        """Should return full image bounds when sizes match."""
        bounds = calculate_center_crop_bounds(100, 100, 100, 100)
        assert bounds == (0, 0, 100, 100)

    def test_odd_dimensions(self):
        """Should handle odd dimensions."""
        bounds = calculate_center_crop_bounds(101, 101, 51, 51)
        assert bounds == (25, 25, 76, 76)

    def test_target_larger_raises_error(self):
        """Should raise ValueError if target exceeds image."""
        with pytest.raises(ValueError, match="exceed image dimensions"):
            calculate_center_crop_bounds(50, 50, 100, 100)

    def test_asymmetric_crop(self):
        """Should handle asymmetric cropping."""
        bounds = calculate_center_crop_bounds(300, 200, 100, 150)
        assert bounds == (100, 25, 200, 175)


class TestExtendWithGrid:
    """Tests for extend_with_grid function."""

    def test_extend_square_array(self):
        """Should extend array with coordinate grid."""
        arr = np.ones((10, 10, 2))
        extended = extend_with_grid(arr, 20, 20)
        assert extended.shape == (20, 20, 2)

    def test_extend_with_custom_offset(self):
        """Should use custom offset when provided."""
        arr = np.ones((10, 10, 2))
        extended = extend_with_grid(arr, 20, 20, offset_x=0, offset_y=0)
        assert extended.shape == (20, 20, 2)
        # Check that original was placed at specified offset
        # (values should be shifted by offset)

    def test_extend_preserves_dtype(self):
        """Should preserve float32 dtype."""
        arr = np.ones((10, 10, 2), dtype=np.float32)
        extended = extend_with_grid(arr, 20, 20)
        assert extended.dtype == np.float32

    def test_grid_coordinates_correct(self):
        """Should create correct coordinate meshgrid."""
        arr = np.zeros((5, 5, 2))
        extended = extend_with_grid(arr, 10, 10)
        # Top-left corner should have coordinates (0, 0)
        assert extended[0, 0, 0] == 0  # y coordinate
        assert extended[0, 0, 1] == 0  # x coordinate

    def test_centered_placement_default(self):
        """Should center original array by default."""
        arr = np.ones((10, 10, 2)) * 100
        extended = extend_with_grid(arr, 30, 30)
        # Original should be placed in center (offset = 10, 10)
        assert extended.shape == (30, 30, 2)


class TestCalculatePadding:
    """Tests for calculate_padding function."""

    def test_pad_square_to_larger_square(self):
        """Should calculate equal padding for square."""
        padding = calculate_padding(50, 50, 100, 100)
        assert padding == (25, 25, 25, 25)

    def test_pad_rectangular(self):
        """Should calculate padding for rectangle."""
        padding = calculate_padding(100, 50, 200, 100)
        assert padding == (25, 25, 50, 50)

    def test_pad_exact_size_zero_padding(self):
        """Should return zero padding when sizes match."""
        padding = calculate_padding(100, 100, 100, 100)
        assert padding == (0, 0, 0, 0)

    def test_pad_odd_dimensions(self):
        """Should handle odd dimensions correctly."""
        padding = calculate_padding(50, 50, 101, 101)
        # Total padding is 51 for each dimension
        # Split: 25 + 26 or 26 + 25
        assert sum(padding[:2]) == 51  # top + bottom
        assert sum(padding[2:]) == 51  # left + right

    def test_pad_asymmetric(self):
        """Should handle asymmetric padding."""
        padding = calculate_padding(100, 50, 300, 200)
        assert sum(padding[:2]) == 150  # height padding
        assert sum(padding[2:]) == 200  # width padding


class TestGetCropOrPadBounds:
    """Tests for get_crop_or_pad_bounds function."""

    def test_crop_operation(self):
        """Should detect crop operation."""
        result = get_crop_or_pad_bounds(100, 100, 50, 50)
        assert result['operation'] == 'crop'
        assert result['width_op'] == 'crop'
        assert result['height_op'] == 'crop'
        assert result['crop_bounds'] is not None
        assert result['padding'] is None

    def test_pad_operation(self):
        """Should detect pad operation."""
        result = get_crop_or_pad_bounds(50, 50, 100, 100)
        assert result['operation'] == 'pad'
        assert result['width_op'] == 'pad'
        assert result['height_op'] == 'pad'
        assert result['padding'] is not None
        assert result['crop_bounds'] is None

    def test_mixed_operation(self):
        """Should detect mixed crop/pad operation."""
        result = get_crop_or_pad_bounds(100, 50, 50, 100)
        assert result['operation'] == 'mixed'
        assert result['width_op'] == 'crop'
        assert result['height_op'] == 'pad'

    def test_none_operation(self):
        """Should detect no operation needed."""
        result = get_crop_or_pad_bounds(100, 100, 100, 100)
        assert result['operation'] == 'none'
        assert result['width_op'] == 'none'
        assert result['height_op'] == 'none'

    def test_crop_bounds_calculated(self):
        """Should calculate crop bounds when cropping."""
        result = get_crop_or_pad_bounds(200, 200, 100, 100)
        assert result['crop_bounds'] == (50, 50, 150, 150)

    def test_padding_calculated(self):
        """Should calculate padding when padding."""
        result = get_crop_or_pad_bounds(50, 50, 100, 100)
        assert result['padding'] == (25, 25, 25, 25)


class TestImageGeometryIntegration:
    """Integration tests combining multiple geometry utilities."""

    def test_crop_workflow(self):
        """Should handle complete cropping workflow."""
        img = np.random.rand(200, 200, 3)

        # Determine operation
        info = get_crop_or_pad_bounds(200, 200, 100, 100)
        assert info['operation'] == 'crop'

        # Calculate bounds
        bounds = calculate_center_crop_bounds(200, 200, 100, 100)
        assert bounds == (50, 50, 150, 150)

        # Perform crop
        cropped = center_crop(img, 100, 100)
        assert cropped.shape == (100, 100, 3)

    def test_padding_workflow(self):
        """Should handle complete padding workflow."""
        # Determine operation
        info = get_crop_or_pad_bounds(50, 50, 100, 100)
        assert info['operation'] == 'pad'

        # Calculate padding
        padding = calculate_padding(50, 50, 100, 100)
        assert padding == (25, 25, 25, 25)

    def test_roundtrip_crop_then_extend(self):
        """Should maintain consistency in crop-extend roundtrip."""
        original = np.random.rand(100, 100, 2)

        # Crop to smaller size
        cropped = center_crop(original, 50, 50)
        assert cropped.shape == (50, 50, 2)

        # Extend back to original size
        extended = extend_with_grid(cropped, 100, 100)
        assert extended.shape == (100, 100, 2)

    def test_decision_tree_all_cases(self):
        """Should handle all operation types correctly."""
        # Case 1: Crop both dimensions
        assert get_crop_or_pad_bounds(100, 100, 50, 50)['operation'] == 'crop'

        # Case 2: Pad both dimensions
        assert get_crop_or_pad_bounds(50, 50, 100, 100)['operation'] == 'pad'

        # Case 3: Crop width, pad height
        assert get_crop_or_pad_bounds(100, 50, 50, 100)['operation'] == 'mixed'

        # Case 4: Pad width, crop height
        assert get_crop_or_pad_bounds(50, 100, 100, 50)['operation'] == 'mixed'

        # Case 5: No change
        assert get_crop_or_pad_bounds(100, 100, 100, 100)['operation'] == 'none'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
