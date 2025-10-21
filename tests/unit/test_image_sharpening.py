"""Unit tests for image sharpening functions."""

import numpy as np
import pytest

from deforum.utils.image_utils import (
    clamp_to_uint8,
    calculate_sharpened_image,
    apply_threshold_mask,
    apply_spatial_mask,
    unsharp_mask,
)


class TestClampToUint8:
    """Test uint8 clamping function."""

    def test_clamp_negative_values(self):
        img = np.array([-10, -1, 0], dtype=np.float32)
        result = clamp_to_uint8(img)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_clamp_overflow_values(self):
        img = np.array([255, 256, 300], dtype=np.float32)
        result = clamp_to_uint8(img)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [255, 255, 255])

    def test_clamp_normal_range(self):
        img = np.array([0, 127.5, 255], dtype=np.float32)
        result = clamp_to_uint8(img)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 128, 255])

    def test_clamp_multidimensional(self):
        img = np.array([[100.3, 200.7], [50.1, 150.9]], dtype=np.float32)
        result = clamp_to_uint8(img)
        assert result.shape == (2, 2)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [[100, 201], [50, 151]])


class TestCalculateSharpenedImage:
    """Test sharpening calculation."""

    def test_no_sharpening_amount_zero(self):
        img = np.array([100, 150, 200], dtype=np.uint8)
        blurred = np.array([90, 140, 190], dtype=np.uint8)
        result = calculate_sharpened_image(img, blurred, amount=0.0)
        # amount=0: (1+0)*img - 0*blurred = img
        np.testing.assert_array_equal(result, img)

    def test_sharpening_increases_contrast(self):
        # Bright pixel next to dark blur -> brighter
        # Dark pixel next to bright blur -> darker
        img = np.array([200, 50], dtype=np.uint8)
        blurred = np.array([100, 150], dtype=np.uint8)
        result = calculate_sharpened_image(img, blurred, amount=1.0)

        # First pixel: 2*200 - 1*100 = 300 -> clamped to 255
        # Second pixel: 2*50 - 1*150 = -50 -> clamped to 0
        np.testing.assert_array_equal(result, [255, 0])

    def test_sharpening_formula(self):
        img = np.array([100], dtype=np.uint8)
        blurred = np.array([80], dtype=np.uint8)
        result = calculate_sharpened_image(img, blurred, amount=0.5)

        # (1 + 0.5) * 100 - 0.5 * 80 = 150 - 40 = 110
        assert result[0] == 110


class TestApplyThresholdMask:
    """Test threshold masking."""

    def test_no_threshold_returns_sharpened(self):
        sharpened = np.array([100, 150, 200], dtype=np.uint8)
        original = np.array([90, 140, 190], dtype=np.uint8)
        blurred = np.array([85, 135, 185], dtype=np.uint8)

        result = apply_threshold_mask(sharpened, original, blurred, threshold=0)
        np.testing.assert_array_equal(result, sharpened)

    def test_threshold_preserves_low_contrast(self):
        sharpened = np.array([100, 150, 200], dtype=np.uint8)
        original = np.array([98, 140, 190], dtype=np.uint8)
        blurred = np.array([97, 135, 185], dtype=np.uint8)

        # Contrast: |98-97|=1, |140-135|=5, |190-185|=5
        # With threshold=3: preserve first (contrast=1), sharpen others (contrast=5)
        result = apply_threshold_mask(sharpened, original, blurred, threshold=3)

        assert result[0] == 98   # Low contrast -> original preserved
        assert result[1] == 150  # High contrast -> sharpened
        assert result[2] == 200  # High contrast -> sharpened


class TestApplySpatialMask:
    """Test spatial masking."""

    def test_no_mask_returns_sharpened(self):
        sharpened = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        original = np.array([[90, 140], [190, 240]], dtype=np.uint8)

        result = apply_spatial_mask(sharpened, original, mask=None)
        np.testing.assert_array_equal(result, sharpened)

    def test_mask_applies_selectively(self):
        sharpened = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        original = np.array([[90, 140], [190, 240]], dtype=np.uint8)

        # Mask: sharpen top-left and bottom-right only
        mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)

        result = apply_spatial_mask(sharpened, original, mask)

        # Top-left: sharpened (100)
        # Top-right: original (140)
        # Bottom-left: original (190)
        # Bottom-right: sharpened (250)
        assert result[0, 0] == 100
        assert result[0, 1] == 140
        assert result[1, 0] == 190
        assert result[1, 1] == 250


class TestUnsharpMask:
    """Test complete unsharp mask function."""

    def test_amount_zero_returns_original(self):
        img = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        result = unsharp_mask(img, amount=0.0)
        np.testing.assert_array_equal(result, img)

    def test_basic_sharpening(self):
        # Create a simple gradient that should sharpen
        img = np.array([[100, 150, 200]], dtype=np.uint8)
        result = unsharp_mask(img, amount=1.0, sigma=0.5)

        assert result.dtype == np.uint8
        assert result.shape == img.shape
        # Middle value should be enhanced (more contrast)
        # This is a sanity check - exact values depend on blur kernel

    def test_with_all_parameters(self):
        img = np.random.randint(50, 200, (10, 10), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8) * 255

        result = unsharp_mask(
            img,
            kernel_size=(3, 3),
            sigma=1.0,
            amount=1.5,
            threshold=10,
            mask=mask
        )

        assert result.dtype == np.uint8
        assert result.shape == img.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
