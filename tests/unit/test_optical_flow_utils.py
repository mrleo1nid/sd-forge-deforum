"""Unit tests for optical flow consistency checking utilities."""

import pytest
import numpy as np

from deforum.utils.media.optical_flow import make_consistency


class TestMakeConsistency:
    """Test make_consistency function."""

    def test_output_shape(self):
        """Output should have shape (H, W, 3)."""
        h, w = 50, 60
        flow1 = np.random.randn(h, w, 2)
        flow2 = np.random.randn(h, w, 2)
        result = make_consistency(flow1, flow2)
        assert result.shape == (h, w, 3)

    def test_different_resolutions(self):
        """Should work with different image resolutions."""
        resolutions = [(10, 10), (100, 80), (200, 150)]
        for h, w in resolutions:
            flow1 = np.random.randn(h, w, 2)
            flow2 = np.random.randn(h, w, 2)
            result = make_consistency(flow1, flow2)
            assert result.shape == (h, w, 3)

    def test_perfect_consistency(self):
        """Perfect inverse flows should have high consistency."""
        h, w = 50, 50
        # Create simple flow (uniform translation)
        flow1 = np.ones((h, w, 2)) * 2  # Move right and down by 2 pixels
        flow2 = -flow1  # Perfect inverse
        result = make_consistency(flow1, flow2, edges_unreliable=False)

        # Most pixels should be reliable (value close to 1.0)
        # Allow some edge effects
        center = result[10:-10, 10:-10]
        assert np.mean(center[..., 0] > 0) > 0.8

    def test_no_flow(self):
        """Zero flow should be perfectly consistent."""
        h, w = 50, 50
        flow1 = np.zeros((h, w, 2))
        flow2 = np.zeros((h, w, 2))
        result = make_consistency(flow1, flow2)

        # All pixels should be reliable
        assert np.all(result[..., 0] > 0)

    def test_edges_unreliable_flag(self):
        """edges_unreliable flag should affect channel 1."""
        h, w = 50, 50
        flow1 = np.random.randn(h, w, 2) * 10  # Large random flow
        flow2 = -flow1

        result_edges_reliable = make_consistency(flow1, flow2, edges_unreliable=False)
        result_edges_unreliable = make_consistency(flow1, flow2, edges_unreliable=True)

        # Channel 1 should be different when edges are marked unreliable
        assert not np.array_equal(
            result_edges_reliable[..., 1], result_edges_unreliable[..., 1]
        )

    def test_occlusion_detection(self):
        """Channel 0 should detect occlusions."""
        h, w = 50, 50
        flow1 = np.zeros((h, w, 2))
        flow2 = np.zeros((h, w, 2))

        # Create inconsistent region (occlusion)
        flow1[20:30, 20:30] = [10, 10]  # Large flow in one region
        # flow2 stays zero - inconsistent

        result = make_consistency(flow1, flow2)

        # Occlusion region should have negative values in channel 0
        assert np.any(result[20:30, 20:30, 0] < 0)

    def test_motion_edge_detection(self):
        """Channel 2 should detect motion edges."""
        h, w = 100, 100
        flow1 = np.zeros((h, w, 2))

        # Create sharp motion boundary
        flow1[:, :50] = [0, 0]  # No motion left half
        flow1[:, 50:] = [5, 5]  # Motion right half

        flow2 = -flow1
        result = make_consistency(flow1, flow2)

        # Motion edge around x=50 should be detected in channel 2
        # Some pixels near the edge should have lower values
        edge_region = result[:, 48:52, 2]
        assert np.mean(edge_region) < 0.9

    def test_output_range(self):
        """Output values should be in expected range."""
        h, w = 50, 50
        flow1 = np.random.randn(h, w, 2)
        flow2 = np.random.randn(h, w, 2)
        result = make_consistency(flow1, flow2)

        # Channel 0: -0.75 to 1.0
        assert np.all(result[..., 0] >= -0.76)
        assert np.all(result[..., 0] <= 1.01)

        # Channel 1: 0.0 to 1.0
        assert np.all(result[..., 1] >= -0.01)
        assert np.all(result[..., 1] <= 1.01)

        # Channel 2: -0.75 to 1.0 (can be -0.75 from channel 0 propagation)
        assert np.all(result[..., 2] >= -0.76)
        assert np.all(result[..., 2] <= 1.01)

    def test_no_nan_or_inf(self):
        """Output should never contain NaN or Inf."""
        h, w = 50, 50
        flow1 = np.random.randn(h, w, 2) * 5
        flow2 = np.random.randn(h, w, 2) * 5
        result = make_consistency(flow1, flow2)

        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_large_flow_values(self):
        """Should handle large flow magnitudes."""
        h, w = 50, 50
        flow1 = np.random.randn(h, w, 2) * 100
        flow2 = -flow1
        result = make_consistency(flow1, flow2)

        assert result.shape == (h, w, 3)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_small_resolution(self):
        """Should work with very small images."""
        h, w = 5, 5
        flow1 = np.random.randn(h, w, 2)
        flow2 = np.random.randn(h, w, 2)
        result = make_consistency(flow1, flow2)
        assert result.shape == (h, w, 3)

    def test_asymmetric_resolution(self):
        """Should work with very asymmetric resolutions."""
        # Tall narrow image
        flow1 = np.random.randn(200, 20, 2)
        flow2 = np.random.randn(200, 20, 2)
        result = make_consistency(flow1, flow2)
        assert result.shape == (200, 20, 3)

        # Wide short image
        flow1 = np.random.randn(20, 200, 2)
        flow2 = np.random.randn(20, 200, 2)
        result = make_consistency(flow1, flow2)
        assert result.shape == (20, 200, 3)


class TestIntegration:
    """Integration tests for optical flow consistency."""

    def test_realistic_optical_flow_scenario(self):
        """Test with realistic optical flow scenario."""
        h, w = 100, 100

        # Simulate camera pan: uniform horizontal motion
        flow_fwd = np.zeros((h, w, 2))
        flow_fwd[..., 0] = 5.0  # 5 pixels right
        flow_bwd = np.zeros((h, w, 2))
        flow_bwd[..., 0] = -5.0  # 5 pixels left (inverse)

        result = make_consistency(flow_fwd, flow_bwd)

        # Should be mostly consistent except at edges
        center = result[20:-20, 20:-20]
        assert np.mean(center[..., 0] > 0) > 0.9

    def test_forward_backward_consistency(self):
        """Test that swapping forward/backward gives similar results."""
        h, w = 50, 50
        flow1 = np.random.randn(h, w, 2)
        flow2 = np.random.randn(h, w, 2)

        result_fwd = make_consistency(flow1, flow2)
        result_bwd = make_consistency(flow2, flow1)

        # Results should have similar statistics (not identical due to asymmetry)
        assert abs(np.mean(result_fwd) - np.mean(result_bwd)) < 0.2

    def test_consistency_with_partial_occlusion(self):
        """Test with partially occluded scene."""
        h, w = 80, 80

        # Background: uniform motion
        flow_fwd = np.ones((h, w, 2)) * 3
        flow_bwd = -flow_fwd

        # Foreground object: different motion (occlusion)
        flow_fwd[30:50, 30:50] = [1, 1]
        # Backward flow doesn't match - occlusion

        result = make_consistency(flow_fwd, flow_bwd)

        # Occlusion region should be detected
        assert np.mean(result[30:50, 30:50, 0] < 0) > 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
