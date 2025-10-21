"""Unit tests for noise generation functions."""

import numpy as np
import torch
import pytest
from PIL import Image

import sys
sys.path.insert(0, 'scripts/deforum_helpers')

from noise import (
    perlin_fade,
    round_to_multiple,
    normalize_perlin,
    condition_noise_mask,
    rand_perlin_2d,
    rand_perlin_2d_octaves
)


class TestPerlinFade:
    """Test Perlin fade function."""

    def test_fade_at_zero(self):
        t = torch.tensor(0.0)
        result = perlin_fade(t)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_fade_at_one(self):
        t = torch.tensor(1.0)
        result = perlin_fade(t)
        assert torch.isclose(result, torch.tensor(1.0))

    def test_fade_at_half(self):
        t = torch.tensor(0.5)
        result = perlin_fade(t)
        # 6*(0.5)^5 - 15*(0.5)^4 + 10*(0.5)^3 = 0.1875 - 0.9375 + 1.25 = 0.5
        assert torch.isclose(result, torch.tensor(0.5))

    def test_fade_vectorized(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        result = perlin_fade(t)
        assert result.shape == (3,)
        assert torch.isclose(result[0], torch.tensor(0.0))
        assert torch.isclose(result[2], torch.tensor(1.0))


class TestRoundToMultiple:
    """Test rounding to multiples."""

    def test_already_multiple(self):
        assert round_to_multiple(64, 64) == 64
        assert round_to_multiple(128, 64) == 128

    def test_round_down(self):
        assert round_to_multiple(100, 64) == 64
        assert round_to_multiple(127, 64) == 64
        assert round_to_multiple(65, 64) == 64

    def test_zero(self):
        assert round_to_multiple(0, 64) == 0

    def test_less_than_multiple(self):
        assert round_to_multiple(30, 64) == 0
        assert round_to_multiple(63, 64) == 0


class TestNormalizePerlin:
    """Test Perlin noise normalization."""

    def test_normalize_negative_one(self):
        noise = torch.tensor([[-1.0, -1.0]])
        result = normalize_perlin(noise)
        expected = torch.tensor([[0.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_normalize_one(self):
        noise = torch.tensor([[1.0, 1.0]])
        result = normalize_perlin(noise)
        expected = torch.tensor([[1.0, 1.0]])
        assert torch.allclose(result, expected)

    def test_normalize_zero(self):
        noise = torch.tensor([[0.0, 0.0]])
        result = normalize_perlin(noise)
        expected = torch.tensor([[0.5, 0.5]])
        assert torch.allclose(result, expected)

    def test_normalize_preserves_shape(self):
        noise = torch.randn(10, 20)
        result = normalize_perlin(noise)
        assert result.shape == (10, 20)


class TestConditionNoiseMask:
    """Test noise mask conditioning."""

    def test_white_mask_returns_ones(self):
        mask = Image.new('L', (10, 10), 255)
        result = condition_noise_mask(mask)
        assert result.shape == (10, 10)
        assert torch.all(result == 1.0)

    def test_black_mask_returns_zeros(self):
        mask = Image.new('L', (10, 10), 0)
        result = condition_noise_mask(mask)
        assert result.shape == (10, 10)
        assert torch.all(result == 0.0)

    def test_invert_mask(self):
        mask = Image.new('L', (10, 10), 0)
        result = condition_noise_mask(mask, invert_mask=True)
        # Black inverted to white
        assert torch.all(result == 1.0)

    def test_converts_rgb_to_grayscale(self):
        mask = Image.new('RGB', (10, 10), (255, 255, 255))
        result = condition_noise_mask(mask)
        assert result.shape == (10, 10)
        assert torch.all(result == 1.0)


class TestRandPerlin2D:
    """Test 2D Perlin noise generation."""

    def test_output_shape(self):
        shape = (64, 64)
        res = (4, 4)
        noise = rand_perlin_2d(shape, res)
        assert noise.shape == shape

    def test_different_resolutions(self):
        shape = (128, 128)
        noise_low = rand_perlin_2d(shape, (2, 2))
        noise_high = rand_perlin_2d(shape, (8, 8))
        # Higher resolution should have different pattern
        assert not torch.allclose(noise_low, noise_high)

    def test_output_range(self):
        # Perlin noise should be roughly in [-1, 1] range
        shape = (64, 64)
        res = (4, 4)
        noise = rand_perlin_2d(shape, res)
        # Allow small margin for numeric precision
        assert noise.min() >= -2.0
        assert noise.max() <= 2.0

    def test_custom_fade_function(self):
        shape = (32, 32)
        res = (4, 4)
        # Linear fade instead of Perlin fade
        linear_fade = lambda t: t
        noise = rand_perlin_2d(shape, res, fade=linear_fade)
        assert noise.shape == shape


class TestRandPerlin2DOctaves:
    """Test multi-octave Perlin noise."""

    def test_single_octave(self):
        shape = (64, 64)
        res = (4, 4)
        noise = rand_perlin_2d_octaves(shape, res, octaves=1)
        assert noise.shape == shape

    def test_multiple_octaves(self):
        shape = (64, 64)
        res = (4, 4)
        noise_1 = rand_perlin_2d_octaves(shape, res, octaves=1)
        noise_3 = rand_perlin_2d_octaves(shape, res, octaves=3)
        # More octaves should create different pattern
        assert not torch.allclose(noise_1, noise_3)

    def test_persistence_effect(self):
        shape = (64, 64)
        res = (4, 4)
        noise_low = rand_perlin_2d_octaves(shape, res, octaves=3, persistence=0.1)
        noise_high = rand_perlin_2d_octaves(shape, res, octaves=3, persistence=0.9)
        # Different persistence should create different patterns
        assert not torch.allclose(noise_low, noise_high)

    def test_zero_octaves(self):
        shape = (64, 64)
        res = (4, 4)
        noise = rand_perlin_2d_octaves(shape, res, octaves=0)
        # Zero octaves should return zeros
        assert torch.allclose(noise, torch.zeros(shape))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
