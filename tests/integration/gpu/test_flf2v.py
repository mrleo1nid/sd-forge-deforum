"""Integration tests for FLF2V (First-Last-Frame-to-Video) interpolation.

Tests verify that FLF2V can correctly interpolate between two keyframes
by transforming simple shapes and colors and checking intermediate frames.

Run with:
    pytest tests/integration/gpu/test_flf2v.py -v
    pytest tests/integration/gpu/test_flf2v.py::test_flf2v_color_interpolation -v

Requirements:
    - GPU with CUDA
    - Wan FLF2V model loaded (e.g., Wan2.2-FLF2V-14B)
    - Running within Forge environment
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


@pytest.fixture(scope="session")
def check_wan_flf2v_available():
    """Verify Wan FLF2V model is available before running tests."""
    try:
        from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration

        wan = WanSimpleIntegration()
        models = wan.discover_models()

        # Check if any FLF2V model is available
        flf2v_models = [m for m in models if m.get('type') == 'FLF2V']

        if not flf2v_models:
            pytest.skip("No Wan FLF2V model found - needed for FLF2V tests")

        return flf2v_models[0]

    except ImportError as e:
        pytest.skip(f"Wan integration not available: {e}")


@pytest.mark.gpu
@pytest.mark.slow
def test_flf2v_color_interpolation(
    gpu_available,
    check_wan_flf2v_available,
    isolated_output_dir
):
    """Test FLF2V interpolates correctly between two solid colors.

    Verifies:
    - FLF2V model loads successfully
    - Interpolation generates expected number of frames
    - Intermediate frames show gradual color transition
    - No errors during generation
    """
    from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration

    # Create simple test images: red → blue
    width, height = 512, 512

    # First frame: solid red
    first_frame = Image.new('RGB', (width, height), color=(255, 0, 0))

    # Last frame: solid blue
    last_frame = Image.new('RGB', (width, height), color=(0, 0, 255))

    # Save test frames for verification
    first_frame.save(isolated_output_dir / "first_frame.png")
    last_frame.save(isolated_output_dir / "last_frame.png")

    print(f"\n[TEST] Testing FLF2V color interpolation: red → blue")
    print(f"[TEST] Output directory: {isolated_output_dir}")

    # Initialize Wan
    wan = WanSimpleIntegration()
    models = wan.discover_models()

    # Load first available FLF2V model
    flf2v_model = check_wan_flf2v_available
    model_loaded = wan.load_model(flf2v_model['path'])

    if not model_loaded:
        pytest.fail("Failed to load Wan FLF2V model")

    # Generate interpolation with minimal settings for speed
    num_frames = 9  # Wan requires 4n+1 frames (9 = 4*2 + 1)

    try:
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.max_memory_allocated() / 1024**3

        # Generate FLF2V interpolation
        result = wan.pipeline.generate_flf2v(
            first_frame=first_frame,
            last_frame=last_frame,
            prompt="",  # Empty prompt for pure visual interpolation
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=4,  # Minimal steps for speed
            guidance_scale=3.5  # Default for smooth interpolation
        )

        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        vram_used = vram_peak - vram_before

        print(f"\n[VRAM] Peak usage during FLF2V: {vram_peak:.2f} GB")
        print(f"[VRAM] Incremental usage: {vram_used:.2f} GB")

        # Extract frames from result
        frames = result.frames[0]  # diffusers returns [batch, frames, H, W, C]

        # Verify frame count
        assert len(frames) == num_frames, (
            f"Expected {num_frames} frames, got {len(frames)}"
        )

        print(f"[SUCCESS] Generated {len(frames)} interpolated frames")

        # Analyze color progression
        red_values = []
        blue_values = []

        for i, frame in enumerate(frames):
            # Convert to PIL Image if needed
            if isinstance(frame, np.ndarray):
                frame_img = Image.fromarray(frame)
            else:
                frame_img = frame

            # Save frame
            frame_img.save(isolated_output_dir / f"interpolated_frame_{i:04d}.png")

            # Calculate average red and blue values
            frame_array = np.array(frame_img)
            avg_red = frame_array[:, :, 0].mean()
            avg_blue = frame_array[:, :, 2].mean()

            red_values.append(avg_red)
            blue_values.append(avg_blue)

            print(f"[FRAME {i}] Avg R: {avg_red:.1f}, Avg B: {avg_blue:.1f}")

        # Verify color transition
        # Red should decrease from ~255 to ~0
        # Blue should increase from ~0 to ~255

        # First frame should be mostly red
        assert red_values[0] > 200, (
            f"First frame should be red (R={red_values[0]:.1f}), expected >200"
        )
        assert blue_values[0] < 50, (
            f"First frame should have minimal blue (B={blue_values[0]:.1f}), expected <50"
        )

        # Last frame should be mostly blue
        assert red_values[-1] < 50, (
            f"Last frame should have minimal red (R={red_values[-1]:.1f}), expected <50"
        )
        assert blue_values[-1] > 200, (
            f"Last frame should be blue (B={blue_values[-1]:.1f}), expected >200"
        )

        # Middle frame should show transition
        mid_idx = len(frames) // 2
        assert 50 < red_values[mid_idx] < 200, (
            f"Middle frame should show color transition (R={red_values[mid_idx]:.1f})"
        )
        assert 50 < blue_values[mid_idx] < 200, (
            f"Middle frame should show color transition (B={blue_values[mid_idx]:.1f})"
        )

        print("[SUCCESS] ✓ Color interpolation verified: smooth red → blue transition")

    except Exception as e:
        pytest.fail(f"FLF2V color interpolation failed: {str(e)}")


@pytest.mark.gpu
@pytest.mark.slow
def test_flf2v_shape_transformation(
    gpu_available,
    check_wan_flf2v_available,
    isolated_output_dir
):
    """Test FLF2V interpolates correctly between different shapes.

    Verifies:
    - FLF2V can handle shape transformations (circle → square)
    - Intermediate frames show gradual morphing
    - Output frames are valid images
    """
    from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration

    width, height = 512, 512

    # First frame: white circle on black background
    first_frame = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw1 = ImageDraw.Draw(first_frame)
    circle_bbox = [128, 128, 384, 384]  # Centered circle
    draw1.ellipse(circle_bbox, fill=(255, 255, 255))

    # Last frame: white square on black background
    last_frame = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw2 = ImageDraw.Draw(last_frame)
    square_bbox = [128, 128, 384, 384]  # Centered square
    draw2.rectangle(square_bbox, fill=(255, 255, 255))

    # Save test frames
    first_frame.save(isolated_output_dir / "shape_first_circle.png")
    last_frame.save(isolated_output_dir / "shape_last_square.png")

    print(f"\n[TEST] Testing FLF2V shape transformation: circle → square")
    print(f"[TEST] Output directory: {isolated_output_dir}")

    # Initialize Wan
    wan = WanSimpleIntegration()

    # Load FLF2V model
    flf2v_model = check_wan_flf2v_available
    model_loaded = wan.load_model(flf2v_model['path'])

    if not model_loaded:
        pytest.fail("Failed to load Wan FLF2V model")

    num_frames = 9  # 4n+1 format

    try:
        # Generate FLF2V interpolation
        result = wan.pipeline.generate_flf2v(
            first_frame=first_frame,
            last_frame=last_frame,
            prompt="",  # Pure visual interpolation
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=4,
            guidance_scale=3.5
        )

        frames = result.frames[0]

        assert len(frames) == num_frames, (
            f"Expected {num_frames} frames, got {len(frames)}"
        )

        print(f"[SUCCESS] Generated {len(frames)} shape interpolation frames")

        # Analyze shape progression by checking white pixel count
        white_pixel_counts = []

        for i, frame in enumerate(frames):
            if isinstance(frame, np.ndarray):
                frame_img = Image.fromarray(frame)
            else:
                frame_img = frame

            frame_img.save(isolated_output_dir / f"shape_frame_{i:04d}.png")

            # Count white-ish pixels (brightness > 200)
            frame_array = np.array(frame_img)
            brightness = frame_array.mean(axis=2)
            white_pixels = (brightness > 200).sum()
            white_pixel_counts.append(white_pixels)

            print(f"[FRAME {i}] White pixels: {white_pixels}")

        # Verify all frames have some white content (shape is present)
        for i, count in enumerate(white_pixel_counts):
            assert count > 1000, (
                f"Frame {i} has too few white pixels ({count}), shape may be missing"
            )

        # Verify frames are different (actual interpolation happening)
        unique_frames = len(set(white_pixel_counts))
        assert unique_frames > 3, (
            f"Only {unique_frames} unique frame patterns found, "
            "expected more variation during interpolation"
        )

        print("[SUCCESS] ✓ Shape interpolation verified: frames show morphing progression")

    except Exception as e:
        pytest.fail(f"FLF2V shape transformation failed: {str(e)}")


@pytest.mark.gpu
def test_flf2v_single_interpolation(
    gpu_available,
    check_wan_flf2v_available,
    isolated_output_dir
):
    """Fast smoke test: verify FLF2V can generate minimal interpolation.

    Uses minimal settings for fastest possible test:
    - 5 frames (minimum 4n+1 = 5)
    - 4 inference steps
    - Simple gradient transformation
    """
    from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration

    width, height = 512, 512

    # Simple gradient: dark → bright
    first_frame = Image.new('RGB', (width, height), color=(50, 50, 50))
    last_frame = Image.new('RGB', (width, height), color=(200, 200, 200))

    print(f"\n[TEST] Fast FLF2V smoke test: 5 frames, minimal settings")

    wan = WanSimpleIntegration()
    flf2v_model = check_wan_flf2v_available
    wan.load_model(flf2v_model['path'])

    try:
        result = wan.pipeline.generate_flf2v(
            first_frame=first_frame,
            last_frame=last_frame,
            prompt="",
            height=height,
            width=width,
            num_frames=5,  # Minimum: 4*1 + 1
            num_inference_steps=4,
            guidance_scale=3.5
        )

        frames = result.frames[0]
        assert len(frames) == 5, f"Expected 5 frames, got {len(frames)}"

        # Save first frame only for quick verification
        if isinstance(frames[0], np.ndarray):
            frame_img = Image.fromarray(frames[0])
        else:
            frame_img = frames[0]

        frame_img.save(isolated_output_dir / "smoke_test_frame.png")

        print(f"[SUCCESS] ✓ FLF2V smoke test passed: {len(frames)} frames generated")

    except Exception as e:
        pytest.fail(f"FLF2V smoke test failed: {str(e)}")
