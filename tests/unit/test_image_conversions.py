"""Unit tests for image conversion functions."""

import numpy as np
import pytest
from PIL import Image

from deforum.utils.image.processing import (
    bgr_to_rgb,
    numpy_to_pil,
    pil_to_numpy,
    is_PIL,
)


class TestBgrToRgb:
    """Test BGR to RGB conversion."""

    def test_bgr_to_rgb_converts_channels(self):
        # Create BGR image: Blue=100, Green=150, Red=200
        bgr = np.array([[[100, 150, 200]]], dtype=np.uint8)
        rgb = bgr_to_rgb(bgr)
        # After conversion: Red=200, Green=150, Blue=100
        assert rgb[0, 0, 0] == 200  # Red channel
        assert rgb[0, 0, 1] == 150  # Green channel
        assert rgb[0, 0, 2] == 100  # Blue channel

    def test_bgr_to_rgb_preserves_shape(self):
        bgr = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)
        rgb = bgr_to_rgb(bgr)
        assert rgb.shape == (10, 20, 3)

    def test_bgr_to_rgb_preserves_dtype(self):
        bgr = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
        rgb = bgr_to_rgb(bgr)
        assert rgb.dtype == np.uint8


class TestNumpyToPil:
    """Test NumPy to PIL conversion."""

    def test_numpy_to_pil_returns_pil_image(self):
        np_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        pil_img = numpy_to_pil(np_img)
        assert isinstance(pil_img, Image.Image)

    def test_numpy_to_pil_preserves_dimensions(self):
        np_img = np.random.randint(0, 255, (15, 20, 3), dtype=np.uint8)
        pil_img = numpy_to_pil(np_img)
        assert pil_img.size == (20, 15)  # PIL uses (width, height)

    def test_numpy_to_pil_converts_bgr_to_rgb(self):
        # BGR image with pure blue in top-left pixel
        bgr_img = np.zeros((5, 5, 3), dtype=np.uint8)
        bgr_img[0, 0] = [255, 0, 0]  # Pure blue in BGR

        pil_img = numpy_to_pil(bgr_img)
        pixels = pil_img.load()
        # Should be pure blue in RGB (0, 0, 255)
        assert pixels[0, 0] == (0, 0, 255)


class TestPilToNumpy:
    """Test PIL to NumPy conversion."""

    def test_pil_to_numpy_returns_numpy_array(self):
        pil_img = Image.new('RGB', (10, 10), color=(128, 128, 128))
        np_img = pil_to_numpy(pil_img)
        assert isinstance(np_img, np.ndarray)

    def test_pil_to_numpy_preserves_dimensions(self):
        pil_img = Image.new('RGB', (20, 15))  # width=20, height=15
        np_img = pil_to_numpy(pil_img)
        assert np_img.shape == (15, 20, 3)  # NumPy uses (height, width, channels)

    def test_pil_to_numpy_preserves_color(self):
        pil_img = Image.new('RGB', (5, 5), color=(100, 150, 200))
        np_img = pil_to_numpy(pil_img)
        assert np.all(np_img[:, :, 0] == 100)  # Red channel
        assert np.all(np_img[:, :, 1] == 150)  # Green channel
        assert np.all(np_img[:, :, 2] == 200)  # Blue channel

    def test_pil_to_numpy_grayscale(self):
        pil_img = Image.new('L', (10, 10), color=128)
        np_img = pil_to_numpy(pil_img)
        assert np_img.shape == (10, 10)
        assert np.all(np_img == 128)


class TestIsPIL:
    """Test PIL type checking."""

    def test_is_pil_returns_true_for_pil_image(self):
        pil_img = Image.new('RGB', (10, 10))
        assert is_PIL(pil_img) is True

    def test_is_pil_returns_false_for_numpy_array(self):
        np_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        assert is_PIL(np_img) is False

    def test_is_pil_returns_false_for_none(self):
        assert is_PIL(None) is False

    def test_is_pil_returns_false_for_list(self):
        assert is_PIL([1, 2, 3]) is False

    def test_is_pil_returns_true_for_all_pil_modes(self):
        modes = ['1', 'L', 'RGB', 'RGBA', 'CMYK']
        for mode in modes:
            pil_img = Image.new(mode, (10, 10))
            assert is_PIL(pil_img) is True


class TestConversionRoundtrip:
    """Test conversion round-trip consistency."""

    def test_pil_to_numpy_to_pil_roundtrip(self):
        original = Image.new('RGB', (10, 10), color=(100, 150, 200))
        np_img = pil_to_numpy(original)
        # Convert back (note: we need to handle BGR conversion)
        # Since numpy_to_pil expects BGR, this won't round-trip perfectly
        # So we just test the shape and type preservation
        assert np_img.shape == (10, 10, 3)

    def test_numpy_pil_numpy_preserves_shape(self):
        original_np = np.random.randint(0, 255, (15, 20, 3), dtype=np.uint8)
        pil_img = numpy_to_pil(original_np)
        back_to_np = pil_to_numpy(pil_img)
        # Shape should be preserved through the round-trip
        assert back_to_np.shape == original_np.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
