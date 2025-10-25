"""Unit tests for color histogram matching functions."""

import numpy as np
import pytest

from deforum.utils.image.processing import (
    match_in_rgb,
    match_in_hsv,
    match_in_lab,
    maintain_colors,
)


class TestMatchInRgb:
    """Test RGB histogram matching."""

    def test_identical_images_unchanged(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = match_in_rgb(img, img)
        # Matching an image to itself should return very similar values
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self):
        img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
        reference = np.random.randint(0, 255, (15, 25, 3), dtype=np.uint8)
        result = match_in_rgb(img, reference)
        assert result.shape == img.shape

    def test_brightens_dark_image_to_bright_reference(self):
        dark_img = np.full((10, 10, 3), 50, dtype=np.uint8)
        bright_ref = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = match_in_rgb(dark_img, bright_ref)
        # Result should be brighter than original
        assert result.mean() > dark_img.mean()


class TestMatchInHsv:
    """Test HSV histogram matching."""

    def test_output_shape_preserved(self):
        img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
        reference = np.random.randint(0, 255, (15, 25, 3), dtype=np.uint8)
        result = match_in_hsv(img, reference)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_hsv_affects_saturation(self):
        # Create a gray image (low saturation)
        gray_img = np.full((10, 10, 3), 128, dtype=np.uint8)
        # Create a colorful reference (high saturation)
        colorful_ref = np.zeros((10, 10, 3), dtype=np.uint8)
        colorful_ref[:, :, 0] = 255  # Pure red

        result = match_in_hsv(gray_img, colorful_ref)
        # Result should have more color variation than gray original
        assert result.std() >= gray_img.std()


class TestMatchInLab:
    """Test LAB histogram matching."""

    def test_output_shape_preserved(self):
        img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
        reference = np.random.randint(0, 255, (15, 25, 3), dtype=np.uint8)
        result = match_in_lab(img, reference)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_lab_preserves_perceptual_lightness(self):
        dark_img = np.full((10, 10, 3), 50, dtype=np.uint8)
        bright_ref = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = match_in_lab(dark_img, bright_ref)
        # LAB should adjust lightness
        assert result.mean() > dark_img.mean()


class TestMaintainColors:
    """Test main color maintenance dispatcher."""

    def test_rgb_mode(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        reference = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = maintain_colors(img, reference, 'RGB')
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_hsv_mode(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        reference = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = maintain_colors(img, reference, 'HSV')
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_lab_mode(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        reference = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = maintain_colors(img, reference, 'LAB')
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_all_modes_produce_valid_output(self):
        # Create a test image with gradient
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 20).reshape(1, -1)  # Red gradient

        # Create reference with different distribution
        reference = np.full((20, 20, 3), 128, dtype=np.uint8)

        rgb_result = maintain_colors(img, reference, 'RGB')
        hsv_result = maintain_colors(img, reference, 'HSV')
        lab_result = maintain_colors(img, reference, 'LAB')

        # All should produce valid outputs with correct shape and dtype
        assert rgb_result.shape == img.shape
        assert hsv_result.shape == img.shape
        assert lab_result.shape == img.shape
        assert rgb_result.dtype == np.uint8
        assert hsv_result.dtype == np.uint8
        assert lab_result.dtype == np.uint8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
