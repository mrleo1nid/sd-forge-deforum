"""Pytest configuration for local GPU integration tests.

These tests run directly on the GPU without requiring the API server.
They are faster than full API tests but still generate real outputs.
"""

import pytest
import torch
import shutil
from pathlib import Path
from types import SimpleNamespace


@pytest.fixture(scope="session")
def gpu_available():
    """Check if CUDA GPU is available for testing."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - skipping GPU integration tests")
    return True


@pytest.fixture(scope="session")
def test_output_dir():
    """Create and return the test output directory.

    Automatically cleaned up after test session.
    """
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    yield output_dir

    # Cleanup: Remove test outputs after session
    # Comment this out if you want to inspect outputs
    # shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture
def isolated_output_dir(test_output_dir, request):
    """Create isolated output directory for each test.

    Each test gets its own subdirectory to avoid conflicts.
    """
    test_name = request.node.name
    test_dir = test_output_dir / test_name
    test_dir.mkdir(exist_ok=True)

    return test_dir


@pytest.fixture
def minimal_args():
    """Minimal Deforum args for fast testing (3 frames, low resolution)."""
    from deforum.config.args import DeforumArgs

    args = SimpleNamespace(**DeforumArgs())

    # Fast generation settings
    args.W = 512
    args.H = 512
    args.steps = 4  # Very few steps for speed
    args.sampler = 'euler'
    args.seed = 42
    args.batch_name = 'test'

    return args


@pytest.fixture
def minimal_anim_args():
    """Minimal animation args for fast testing."""
    from deforum.config.args import DeforumAnimArgs

    anim_args = SimpleNamespace(**DeforumAnimArgs())

    # Very short animation for speed
    anim_args.max_frames = 3
    anim_args.animation_mode = '3D'

    # Minimal transformations
    anim_args.angle = "0:(0)"
    anim_args.zoom = "0:(1.0)"
    anim_args.translation_x = "0:(0)"
    anim_args.translation_y = "0:(0)"
    anim_args.translation_z = "0:(0)"
    anim_args.rotation_3d_x = "0:(0)"
    anim_args.rotation_3d_y = "0:(0)"
    anim_args.rotation_3d_z = "0:(0)"

    return anim_args


@pytest.fixture
def flux_controlnet_args(minimal_anim_args):
    """Animation args with Flux ControlNet V2 enabled."""
    minimal_anim_args.enable_flux_controlnet_v2 = True
    minimal_anim_args.flux_controlnet_strength = 0.8
    minimal_anim_args.flux_controlnet_model = "depth"
    minimal_anim_args.flux_guidance_scale = 3.5

    return minimal_anim_args


@pytest.fixture
def check_flux_available():
    """Verify Flux model is available before running tests."""
    try:
        from modules import shared

        if not hasattr(shared, 'opts') or not hasattr(shared.opts, 'sd_model_checkpoint'):
            pytest.skip("Forge not properly initialized")

        checkpoint = getattr(shared.opts, 'sd_model_checkpoint', '').lower()
        if 'flux' not in checkpoint:
            pytest.skip("Flux model not loaded - needed for Flux ControlNet tests")

    except ImportError:
        pytest.skip("Forge modules not available - tests must run within Forge environment")

    return True


@pytest.fixture(scope="session", autouse=True)
def start_server(request):
    """Override parent conftest's start_server fixture.

    GPU tests run directly without API server, so we don't need
    to start or wait for the server.
    """
    # Do nothing - GPU tests don't need the API server
    pass


def pytest_configure(config):
    """Register custom markers for GPU integration tests."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (skipped if no GPU available)"
    )
    config.addinivalue_line(
        "markers", "flux: mark test as requiring Flux model"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (generates real outputs)"
    )
    config.addinivalue_line(
        "markers", "flux_controlnet: mark test as testing Flux ControlNet"
    )
