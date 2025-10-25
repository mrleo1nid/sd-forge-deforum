"""Integration tests for Wan FLF2V tween generation.

Tests the AI-powered frame interpolation using Wan's First-Last-Frame-to-Video pipeline.
This is a key feature of the sd-forge-deforum fork that enables high-quality tween interpolation.
"""

import glob
import json
import os
from pathlib import Path

import pytest
import requests
from moviepy.editor import VideoFileClip

from .utils import (API_BASE_URL, get_test_options_overrides, gpu_disabled, wait_for_job_to_complete, get_test_batch_name)
from deforum.api.models import DeforumJobStatusCategory


# Path to testdata directory relative to this test file
TESTDATA_DIR = Path(__file__).parent / 'testdata'


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_tween_generation():
    """Test basic FLF2V tween generation between two keyframes."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_tween_generation')

    # Configure for FLF2V tween generation
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 9  # Will create 1 tween segment (4n+1 frames)
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['wan_flf2v_guidance_scale'] = 3.5  # Recommended for smooth interpolation
    deforum_settings['wan_flf2v_prompt_mode'] = "none"  # Pure visual interpolation
    deforum_settings['prompts'] = {
        "0": "a red circle",
        "8": "a blue square"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video with FLF2V tweens"

    video_file = video_files[0]
    clip = VideoFileClip(video_file)
    assert math.ceil(clip.duration * clip.fps) == deforum_settings['max_frames'], \
        "Video should have correct frame count including FLF2V tweens"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_multiple_segments():
    """Test FLF2V with multiple tween segments."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_multiple_segments')

    # Create multiple segments (need 4n+1 frames per segment)
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 17  # Frame 0-8 (segment 1), frame 8-16 (segment 2)
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['wan_flf2v_guidance_scale'] = 3.5
    deforum_settings['wan_flf2v_prompt_mode'] = "none"
    deforum_settings['prompts'] = {
        "0": "a red apple",
        "8": "a green pear",
        "16": "a yellow banana"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated with all segments
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_guidance_scale_variations():
    """Test different guidance_scale values for FLF2V."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_guidance_scale_variations')

    # Test with balanced guidance (recommended)
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 9
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['wan_flf2v_guidance_scale'] = 5.5  # Official example value
    deforum_settings['wan_flf2v_prompt_mode'] = "blend"
    deforum_settings['prompts'] = {
        "0": "morning sunrise",
        "8": "evening sunset"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_prompt_mode_blend():
    """Test FLF2V with 'blend' prompt mode."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_prompt_mode_blend')

    # Use blend mode to combine keyframe prompts
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 9
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['wan_flf2v_guidance_scale'] = 3.5
    deforum_settings['wan_flf2v_prompt_mode'] = "blend"  # Combine prompts
    deforum_settings['prompts'] = {
        "0": "a peaceful lake",
        "8": "a stormy ocean"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_with_seed_schedule():
    """Test FLF2V with seed scheduling for keyframes."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_with_seed_schedule')

    # Use seed schedule to control keyframe generation
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 9
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['wan_flf2v_guidance_scale'] = 3.5
    deforum_settings['seed_schedule'] = "0:(42), 8:(1337)"
    deforum_settings['prompts'] = {
        "0": "first scene",
        "8": "second scene"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_with_strength_schedule():
    """Test FLF2V with strength scheduling for I2V chaining."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_with_strength_schedule')

    # Strength schedule affects how much previous frame influences next
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 17
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['wan_flf2v_guidance_scale'] = 3.5
    deforum_settings['strength_schedule'] = "0:(0.6), 8:(0.7), 16:(0.5)"
    deforum_settings['prompts'] = {
        "0": "morning",
        "8": "afternoon",
        "16": "night"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_frame_count_validation():
    """Test that FLF2V correctly handles 4n+1 frame count requirements."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_frame_count_validation')

    # Frame count between keyframes should follow 4n+1 rule
    # Frames 0-5 = 6 frames (not 4n+1, but system should handle this)
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 6
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['prompts'] = {
        "0": "start",
        "5": "end"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    # Should either succeed (by adjusting frame count) or fail gracefully
    assert jobStatus.status in [
        DeforumJobStatusCategory.SUCCEEDED,
        DeforumJobStatusCategory.FAILED
    ], f"Job {job_id} should either succeed or fail gracefully"

    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        # If it failed, the error message should mention frame count requirements
        assert "frame" in jobStatus.message.lower() or "4n+1" in jobStatus.message.lower(), \
            "Error message should mention frame count requirements"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_model_selection():
    """Test that different Wan models can be selected for FLF2V."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_model_selection')

    # Specify a particular Wan model (e.g., VACE variant)
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 9
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['wan_model_name'] = "Wan2.1-VACE-1.3B"  # VACE for better continuity
    deforum_settings['wan_flf2v_guidance_scale'] = 3.5
    deforum_settings['prompts'] = {
        "0": "scene one",
        "8": "scene two"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    # Should succeed if model is installed, or fail with clear error
    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        # Error should mention model not found
        assert "model" in jobStatus.message.lower(), \
            "Error should mention missing model"
    else:
        assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_fps_inheritance():
    """Test that FLF2V tweens inherit FPS from settings."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_fps_inheritance')

    # Set specific FPS
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 9
    deforum_settings['fps'] = 24  # Specific FPS
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['prompts'] = {
        "0": "start",
        "8": "end"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video has correct FPS
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video"

    video_file = video_files[0]
    clip = VideoFileClip(video_file)
    assert clip.fps == deforum_settings['fps'], \
        f"Video FPS should be {deforum_settings['fps']}, got {clip.fps}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_wan_flf2v_redistributed_mode():
    """Test FLF2V with 'Redistributed' keyframe distribution mode."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_wan_flf2v_redistributed_mode')

    # Redistributed mode intelligently places keyframes
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 25
    deforum_settings['keyframe_distribution_mode'] = "Redistributed"
    deforum_settings['enable_wan_flf2v_for_tweens'] = True
    deforum_settings['wan_flf2v_guidance_scale'] = 3.5
    deforum_settings['prompts'] = {
        "0": "dawn",
        "12": "noon",
        "24": "dusk"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"
