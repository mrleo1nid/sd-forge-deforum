"""Unit tests for deforum.utils.depth_utils module."""

import pytest
import torch
import numpy as np

from deforum.utils.media.depth import (
    normalize_depth_tensor,
    equalize_depth_histogram,
    prepare_depth_tensor,
    get_depth_min_max_formatted,
)


class TestNormalizeDepthTensor:
    """Tests for normalize_depth_tensor function."""

    def test_normalizes_to_zero_one_range(self):
        """Tensor should be normalized to [0, 1] range."""
        tensor = torch.tensor([[0.0, 5.0], [10.0, 15.0]])
        result = normalize_depth_tensor(tensor)

        assert result.min().item() == pytest.approx(0.0)
        assert result.max().item() == pytest.approx(1.0)

    def test_preserves_relative_values(self):
        """Relative ordering of values should be preserved."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = normalize_depth_tensor(tensor)

        # Values should still be in ascending order
        assert result[0, 0] < result[0, 1] < result[1, 0] < result[1, 1]

    def test_handles_negative_values(self):
        """Should handle negative depth values."""
        tensor = torch.tensor([[-10.0, -5.0], [0.0, 5.0]])
        result = normalize_depth_tensor(tensor)

        assert result.min().item() == pytest.approx(0.0)
        assert result.max().item() == pytest.approx(1.0)

    def test_single_value_raises_error(self):
        """Should raise error if all values are the same."""
        tensor = torch.tensor([[5.0, 5.0], [5.0, 5.0]])

        with pytest.raises(ValueError, match="no range"):
            normalize_depth_tensor(tensor)

    def test_none_tensor_raises_error(self):
        """Should raise error if tensor is None."""
        with pytest.raises(ValueError, match="cannot be None"):
            normalize_depth_tensor(None)

    def test_preserves_device(self):
        """Should preserve tensor device."""
        if torch.cuda.is_available():
            tensor = torch.tensor([[0.0, 1.0]], device="cuda")
            result = normalize_depth_tensor(tensor)
            assert result.device.type == "cuda"

    def test_large_range(self):
        """Should handle very large value ranges."""
        tensor = torch.tensor([[0.0, 1e6]])
        result = normalize_depth_tensor(tensor)

        assert result.min().item() == pytest.approx(0.0)
        assert result.max().item() == pytest.approx(1.0)


class TestEqualizeDepthHistogram:
    """Tests for equalize_depth_histogram function."""

    def test_output_in_zero_one_range(self):
        """Equalized tensor should be in [0, 1] range."""
        tensor = torch.tensor([[0.0, 0.25], [0.5, 1.0]])
        result = equalize_depth_histogram(tensor)

        assert 0.0 <= result.min().item() <= 1.0
        assert 0.0 <= result.max().item() <= 1.0

    def test_preserves_shape(self):
        """Should preserve tensor shape."""
        tensor = torch.rand(10, 20)
        result = equalize_depth_histogram(tensor)

        assert result.shape == tensor.shape

    def test_preserves_device(self):
        """Should preserve tensor device."""
        if torch.cuda.is_available():
            tensor = torch.rand(5, 5, device="cuda")
            result = equalize_depth_histogram(tensor)
            assert result.device.type == "cuda"

    def test_custom_bins(self):
        """Should support custom number of bins."""
        tensor = torch.rand(10, 10)
        result = equalize_depth_histogram(tensor, bins=256)

        assert result.shape == tensor.shape

    def test_invalid_bins_raises_error(self):
        """Should raise error for invalid bin count."""
        tensor = torch.rand(5, 5)

        with pytest.raises(ValueError, match="at least 2"):
            equalize_depth_histogram(tensor, bins=1)

    def test_none_tensor_raises_error(self):
        """Should raise error if tensor is None."""
        with pytest.raises(ValueError, match="cannot be None"):
            equalize_depth_histogram(None)

    def test_uniform_distribution(self):
        """Should handle uniform distribution without errors."""
        tensor = torch.full((5, 5), 0.5)
        result = equalize_depth_histogram(tensor)

        # When all values are identical, they map to a single output value
        # (histogram equalization can't spread a spike, so all values stay equal)
        assert torch.allclose(result, result[0, 0])  # All values equal
        assert result.shape == tensor.shape

    def test_improves_contrast(self):
        """Equalization should improve contrast in typical case."""
        # Create tensor with values clustered in middle
        tensor = torch.tensor([[0.4, 0.45], [0.5, 0.55]])
        result = equalize_depth_histogram(tensor)

        # After equalization, range should be expanded
        original_range = tensor.max() - tensor.min()
        result_range = result.max() - result.min()
        assert result_range >= original_range


class TestPrepareDepthTensor:
    """Tests for prepare_depth_tensor function."""

    def test_normalizes_and_equalizes(self):
        """Should apply both normalization and equalization."""
        tensor = torch.tensor([[5.0, 10.0], [15.0, 20.0]])
        result = prepare_depth_tensor(tensor)

        # Should be in [0, 1] range
        assert 0.0 <= result.min().item() <= 1.0
        assert 0.0 <= result.max().item() <= 1.0

    def test_preserves_shape(self):
        """Should preserve tensor shape."""
        tensor = torch.rand(15, 20)
        result = prepare_depth_tensor(tensor)

        assert result.shape == tensor.shape

    def test_custom_bins(self):
        """Should support custom bin count."""
        tensor = torch.rand(10, 10)
        result = prepare_depth_tensor(tensor, bins=512)

        assert result.shape == tensor.shape

    def test_none_tensor_raises_error(self):
        """Should raise error if tensor is None."""
        with pytest.raises(ValueError, match="cannot be None"):
            prepare_depth_tensor(None)

    def test_handles_raw_depth_data(self):
        """Should handle typical raw depth data."""
        # Simulate MiDaS-style depth output
        tensor = torch.rand(256, 256) * 100  # Values 0-100
        result = prepare_depth_tensor(tensor)

        assert result.shape == (256, 256)
        assert 0.0 <= result.min().item() <= 1.0
        assert 0.0 <= result.max().item() <= 1.0


class TestGetDepthMinMaxFormatted:
    """Tests for get_depth_min_max_formatted function."""

    def test_returns_correct_values(self):
        """Should return correct min and max values."""
        tensor = torch.tensor([[1.234, 5.678]])
        min_val, max_val, min_str, max_str = get_depth_min_max_formatted(tensor)

        assert float(min_val) == pytest.approx(1.234)
        assert float(max_val) == pytest.approx(5.678)

    def test_formats_strings_correctly(self):
        """Should format strings with 2 decimal places and width 5."""
        tensor = torch.tensor([[1.234, 5.678]])
        _, _, min_str, max_str = get_depth_min_max_formatted(tensor)

        # Should be formatted as " 1.23" and " 5.68" (width 5, 2 decimals)
        assert min_str == " 1.23"
        assert max_str == " 5.68"

    def test_handles_large_values(self):
        """Should handle large values correctly."""
        tensor = torch.tensor([[123.456, 999.999]])
        min_val, max_val, min_str, max_str = get_depth_min_max_formatted(tensor)

        assert float(min_val) == pytest.approx(123.456)
        assert float(max_val) == pytest.approx(999.999)
        assert "123.46" in min_str
        assert "1000.00" in max_str

    def test_handles_negative_values(self):
        """Should handle negative values correctly."""
        tensor = torch.tensor([[-10.5, 20.7]])
        min_val, max_val, min_str, max_str = get_depth_min_max_formatted(tensor)

        assert float(min_val) == pytest.approx(-10.5)
        assert float(max_val) == pytest.approx(20.7)

    def test_none_tensor_raises_error(self):
        """Should raise error if tensor is None."""
        with pytest.raises(ValueError, match="cannot be None"):
            get_depth_min_max_formatted(None)

    def test_single_value(self):
        """Should handle tensor with single unique value."""
        tensor = torch.tensor([[5.5, 5.5]])
        min_val, max_val, min_str, max_str = get_depth_min_max_formatted(tensor)

        assert float(min_val) == pytest.approx(5.5)
        assert float(max_val) == pytest.approx(5.5)
        assert min_str == max_str


class TestIntegration:
    """Integration tests for depth processing pipeline."""

    def test_full_pipeline(self):
        """Test complete depth processing pipeline."""
        # Simulate raw depth map from MiDaS
        raw_depth = torch.rand(128, 128) * 50 + 10  # Values 10-60

        # Process through pipeline
        prepared = prepare_depth_tensor(raw_depth)

        # Verify output characteristics
        assert prepared.shape == raw_depth.shape
        assert 0.0 <= prepared.min().item() <= 1.0
        assert 0.0 <= prepared.max().item() <= 1.0

        # Get statistics
        min_val, max_val, _, _ = get_depth_min_max_formatted(prepared)
        assert min_val < max_val

    def test_pipeline_preserves_relative_ordering(self):
        """Pipeline should preserve relative depth ordering."""
        # Create depth map with known ordering
        depth = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        prepared = prepare_depth_tensor(depth)

        # All values should still be in ascending order (within rows)
        assert prepared[0, 0] < prepared[0, 1] < prepared[0, 2]
        assert prepared[1, 0] < prepared[1, 1] < prepared[1, 2]
