"""Integration tests for Flux ControlNet V2.

These tests verify that Flux ControlNet works correctly by actually generating
frames on the GPU. They use minimal settings (3 frames, low res) for speed.

Run with:
    pytest tests/integration/gpu/test_flux_controlnet.py -v
    pytest tests/integration/gpu/test_flux_controlnet.py::test_flux_controlnet_basic -v

Requirements:
    - GPU with CUDA
    - Flux model loaded in Forge
    - Running within Forge environment (not via API)
"""

import pytest
import torch
from pathlib import Path
from types import SimpleNamespace


@pytest.mark.gpu
@pytest.mark.flux
@pytest.mark.flux_controlnet
@pytest.mark.slow
def test_flux_controlnet_basic_generation(
    gpu_available,
    check_flux_available,
    minimal_args,
    flux_controlnet_args,
    isolated_output_dir
):
    """Test basic Flux ControlNet generation with 3 frames.

    Verifies:
    - ControlNet model loads successfully
    - No double-loading of Flux models
    - Frames generate without errors
    - Output files are created
    """
    from deforum.orchestration.run_deforum import run_deforum
    from deforum.config.args import DeforumOutputArgs, ParseqArgs

    # Setup output directory
    minimal_args.outdir = str(isolated_output_dir)
    minimal_args.timestring = "test_flux_controlnet"

    # Simple prompt
    minimal_args.prompts = {
        0: "a mountain landscape",
    }

    # Video args
    video_args = SimpleNamespace(**DeforumOutputArgs())
    video_args.fps = 15

    # Parseq args
    parseq_args = SimpleNamespace(**ParseqArgs())

    # Track VRAM before generation
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.max_memory_allocated() / 1024**3  # GB

    # Run generation
    try:
        result = run_deforum(
            minimal_args,
            flux_controlnet_args,
            video_args,
            parseq_args
        )

        # Track VRAM after generation
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
        vram_used = vram_peak - vram_before

        print(f"\n[VRAM] Peak usage during generation: {vram_peak:.2f} GB")
        print(f"[VRAM] Incremental usage: {vram_used:.2f} GB")

        # Verify output directory was created
        assert isolated_output_dir.exists(), "Output directory not created"

        # Verify frames were generated (should be 3 frames)
        frame_files = list(isolated_output_dir.glob("*_0*.png"))
        assert len(frame_files) >= 3, f"Expected at least 3 frames, got {len(frame_files)}"

        print(f"\n[SUCCESS] Generated {len(frame_files)} frames")
        print(f"[OUTPUT] Files in: {isolated_output_dir}")

        # Check for suspicious VRAM usage (potential double-loading)
        # Flux ControlNet should use ~3.6GB, base Flux ~8-12GB
        # If we see >20GB usage, likely double-loaded
        if vram_used > 20:
            pytest.fail(
                f"Excessive VRAM usage ({vram_used:.2f} GB) suggests models may be "
                f"double-loaded. Expected <15GB for Flux + ControlNet."
            )

    except Exception as e:
        pytest.fail(f"Flux ControlNet generation failed: {str(e)}")


@pytest.mark.gpu
@pytest.mark.flux
@pytest.mark.flux_controlnet
@pytest.mark.slow
def test_flux_controlnet_model_loading(gpu_available, check_flux_available):
    """Test that Flux ControlNet model loads correctly without errors.

    Verifies:
    - ControlNet model can be loaded
    - Model is loaded to correct device (CUDA)
    - Model memory footprint is reasonable (~3.6GB)
    """
    from deforum.integrations.flux_controlnet.models import load_flux_controlnet_model

    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1024**3

    try:
        # Load ControlNet model
        controlnet_model, processor = load_flux_controlnet_model("depth")

        vram_after = torch.cuda.memory_allocated() / 1024**3
        vram_used = vram_after - vram_before

        print(f"\n[VRAM] ControlNet model size: {vram_used:.2f} GB")

        # Verify model loaded
        assert controlnet_model is not None, "ControlNet model failed to load"
        assert processor is not None, "ControlNet processor failed to load"

        # Verify model is on GPU
        assert next(controlnet_model.parameters()).is_cuda, "Model not on CUDA"

        # Verify reasonable memory footprint (should be ~3-4 GB)
        assert 2.0 < vram_used < 6.0, (
            f"ControlNet model uses unexpected amount of VRAM: {vram_used:.2f} GB "
            f"(expected ~3.6 GB). May indicate loading issues."
        )

        print("[SUCCESS] ControlNet model loaded successfully")

    except Exception as e:
        pytest.fail(f"ControlNet model loading failed: {str(e)}")


@pytest.mark.gpu
@pytest.mark.flux
@pytest.mark.flux_controlnet
def test_flux_controlnet_single_frame(
    gpu_available,
    check_flux_available,
    minimal_args,
    flux_controlnet_args,
    isolated_output_dir
):
    """Test single frame generation with Flux ControlNet (fastest test).

    Verifies basic functionality with minimal overhead.
    """
    from deforum.orchestration.run_deforum import run_deforum
    from deforum.config.args import DeforumOutputArgs, ParseqArgs

    # Single frame only
    flux_controlnet_args.max_frames = 1

    minimal_args.outdir = str(isolated_output_dir)
    minimal_args.timestring = "test_single_frame"
    minimal_args.prompts = {0: "a cat"}

    video_args = SimpleNamespace(**DeforumOutputArgs())
    video_args.fps = 15
    parseq_args = SimpleNamespace(**ParseqArgs())

    try:
        result = run_deforum(
            minimal_args,
            flux_controlnet_args,
            video_args,
            parseq_args
        )

        # Verify single frame was generated
        frame_files = list(isolated_output_dir.glob("*_0*.png"))
        assert len(frame_files) >= 1, f"Expected at least 1 frame, got {len(frame_files)}"

        print(f"\n[SUCCESS] Single frame generated: {frame_files[0].name}")

    except Exception as e:
        pytest.fail(f"Single frame generation failed: {str(e)}")


@pytest.mark.gpu
@pytest.mark.flux
@pytest.mark.slow
def test_flux_controlnet_disabled(
    gpu_available,
    check_flux_available,
    minimal_args,
    minimal_anim_args,
    isolated_output_dir
):
    """Test generation with ControlNet disabled (baseline comparison).

    Useful for comparing VRAM usage and verifying ControlNet doesn't interfere
    when disabled.
    """
    from deforum.orchestration.run_deforum import run_deforum
    from deforum.config.args import DeforumOutputArgs, ParseqArgs

    # Ensure ControlNet is disabled
    minimal_anim_args.enable_flux_controlnet_v2 = False
    minimal_anim_args.max_frames = 3

    minimal_args.outdir = str(isolated_output_dir)
    minimal_args.timestring = "test_no_controlnet"
    minimal_args.prompts = {0: "a landscape"}

    video_args = SimpleNamespace(**DeforumOutputArgs())
    video_args.fps = 15
    parseq_args = SimpleNamespace(**ParseqArgs())

    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.max_memory_allocated() / 1024**3

    try:
        result = run_deforum(
            minimal_args,
            minimal_anim_args,
            video_args,
            parseq_args
        )

        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        vram_used = vram_peak - vram_before

        print(f"\n[VRAM] Baseline (no ControlNet): {vram_used:.2f} GB")

        # Verify frames generated
        frame_files = list(isolated_output_dir.glob("*_0*.png"))
        assert len(frame_files) >= 3, f"Expected at least 3 frames, got {len(frame_files)}"

        print(f"[SUCCESS] Baseline test: {len(frame_files)} frames without ControlNet")

    except Exception as e:
        pytest.fail(f"Baseline generation failed: {str(e)}")
