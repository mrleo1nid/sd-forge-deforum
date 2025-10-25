"""
Flux + Interpolation Mode: Flux Keyframes + Choice of Interpolation

Supports three interpolation methods:
- Wan FLF2V (default): AI-generated video interpolation with semantic understanding
- RIFE v4.6: Optical flow-based interpolation for natural motion
- FILM: Google's Frame Interpolation for Large Motion model

Architecture:
  Phase 1: Generate ALL keyframes with Flux/SD
  Phase 2: Batch interpolation between each consecutive keyframe pair (Wan/RIFE/FILM)
  Phase 3: Stitch final video

This combines the best of both worlds:
- High-quality Flux-generated keyframes
- Smooth interpolation between keyframes using your choice of method
"""

import os
import json
from pathlib import Path
from typing import List
import cv2

from modules import shared  # type: ignore

from .data.render_data import RenderData
from .data.frame import KeyFrameDistribution, DiffusionFrame
from .data.taqaddumat import Taqaddumat
from deforum.utils.system.logging import log as log_utils
from deforum.rendering.helpers import webui as web_ui_utils
from deforum.utils.image import processing as image_utils
from deforum.rendering.helpers import filename as filename_utils
from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration
from deforum.media.video_audio_utilities import ffmpeg_stitch_video


def render_wan_flux(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args, root):
    """
    Flux/Wan rendering mode: Flux for keyframes + Wan for interpolation

    1. Generate all keyframes with Flux/SD
    2. Interpolate between keyframes with Wan FLF2V
    3. Stitch final video
    """
    log_utils.info("🎬 Flux/Wan Mode: Flux Keyframes + Wan FLF2V Interpolation", log_utils.BLUE)

    # Create render data
    data = RenderData.create(args, parseq_args, anim_args, video_args, loop_args, controlnet_args, root)

    # Initialize progress tracking
    web_ui_utils.init_job(data)
    shared.total_tqdm = Taqaddumat()

    # Get keyframe distribution
    keyframe_distribution = KeyFrameDistribution.from_UI_tab(data)
    all_frames = DiffusionFrame.create_all_frames(data, keyframe_distribution)

    # Initialize progress bars with all frames (keyframes + tweens)
    shared.total_tqdm.reset(data, all_frames)

    # Extract only keyframes (frames with is_keyframe=True)
    keyframes = [f for f in all_frames if f.is_keyframe]

    log_utils.info(f"📊 Flux/Wan Workflow:", log_utils.BLUE)
    log_utils.info(f"   Total frames: {anim_args.max_frames}", log_utils.BLUE)
    log_utils.info(f"   Keyframes to generate: {len(keyframes)}", log_utils.BLUE)
    log_utils.info(f"   FLF2V segments: {len(keyframes) - 1}", log_utils.BLUE)

    # DEBUG: Show resume and path info
    log_utils.info(f"\n🔍 DEBUG Resume Info:", log_utils.YELLOW)
    log_utils.info(f"   resume_from_timestring: {anim_args.resume_from_timestring}", log_utils.YELLOW)
    log_utils.info(f"   resume_timestring: {anim_args.resume_timestring if hasattr(anim_args, 'resume_timestring') else 'N/A'}", log_utils.YELLOW)
    log_utils.info(f"   root.timestring: {root.timestring}", log_utils.YELLOW)
    log_utils.info(f"   args.outdir: {args.outdir}", log_utils.YELLOW)
    log_utils.info(f"   data.output_directory: {data.output_directory}", log_utils.YELLOW)
    log_utils.info(f"   Directory exists: {os.path.exists(data.output_directory)}", log_utils.YELLOW)
    if os.path.exists(data.output_directory):
        files_in_dir = [f for f in os.listdir(data.output_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        log_utils.info(f"   Image files in directory: {len(files_in_dir)}", log_utils.YELLOW)
        if len(files_in_dir) > 0:
            log_utils.info(f"   First few files: {files_in_dir[:5]}", log_utils.YELLOW)

    # ====================
    # PHASE 1: Batch Generate All Keyframes with Flux/SD
    # ====================
    log_utils.info("\n" + "="*60, log_utils.GREEN)
    log_utils.info("PHASE 1: Batch Keyframe Generation with Flux/SD", log_utils.GREEN)
    log_utils.info("="*60, log_utils.GREEN)

    # Check for resume mode - scan for existing keyframes
    keyframe_images = {}  # {frame_index: image_path}
    is_resuming = anim_args.resume_from_timestring
    
    if is_resuming:
        log_utils.info(f"🔄 Resume mode: Scanning for existing keyframes in {data.output_directory}...", log_utils.BLUE)
        for frame in keyframes:
            # Check simple format first (matches our save format: 000000001.png)
            simple_filename = f"{frame.i:09d}.png"
            simple_path = os.path.join(data.output_directory, simple_filename)

            # Also check for filename with timestring prefix (legacy from old runs)
            timestring_filename = filename_utils.frame_filename(data, frame.i)
            timestring_path = os.path.join(data.output_directory, timestring_filename)

            if os.path.exists(simple_path):
                keyframe_images[frame.i] = simple_path
                log_utils.info(f"   ✓ Found existing keyframe: {simple_filename}", log_utils.GREEN)
            elif os.path.exists(timestring_path):
                keyframe_images[frame.i] = timestring_path
                log_utils.info(f"   ✓ Found existing keyframe (timestring format): {timestring_filename}", log_utils.GREEN)
            else:
                log_utils.info(f"   ✗ Missing keyframe at frame {frame.i} (tried: {simple_filename}, {timestring_filename})", log_utils.YELLOW)
        
        if len(keyframe_images) > 0:
            log_utils.info(f"✅ Found {len(keyframe_images)}/{len(keyframes)} existing keyframes", log_utils.GREEN)

    # Count how many keyframes need to be generated
    keyframes_to_generate = [f for f in keyframes if f.i not in keyframe_images]
    keyframes_existing = [f for f in keyframes if f.i in keyframe_images]
    
    if keyframes_existing:
        log_utils.info(f"✅ Found {len(keyframes_existing)} existing keyframes from previous run", log_utils.GREEN)
    if keyframes_to_generate:
        log_utils.info(f"📸 Need to generate {len(keyframes_to_generate)} new keyframes", log_utils.YELLOW)
    
    for idx, frame in enumerate(keyframes):
        # Skip if keyframe already exists (resume mode)
        if frame.i in keyframe_images:
            continue

        log_utils.info(f"\n📸 Generating NEW keyframe {idx + 1}/{len(keyframes)} (frame {frame.i})...", log_utils.YELLOW)

        # Set scheduled parameters for this frame (prompt, cfg_scale, distilled_cfg_scale, checkpoint, etc.)
        keys = data.animation_keys.deform_keys
        # Clamp frame index to valid range (prompt_series has max_frames entries indexed 0 to max_frames-1)
        frame_idx = min(frame.i, data.args.anim_args.max_frames - 1)
        data.args.args.prompt = data.prompt_series[frame_idx]  # Set prompt for current frame
        data.args.args.cfg_scale = keys.cfg_scale_schedule_series[frame_idx]
        data.args.args.distilled_cfg_scale = keys.distilled_cfg_scale_schedule_series[frame_idx]

        # Checkpoint scheduling (disabled for Flux/Wan mode - always use loaded Flux model)
        if data.args.anim_args.enable_checkpoint_scheduling:
            data.args.args.checkpoint = keys.checkpoint_schedule_series[frame_idx]
        else:
            data.args.args.checkpoint = None

        # Reset progress tracking for this frame
        shared.total_tqdm.reset_step_count(frame.actual_steps(data))

        # Generate keyframe image using Flux/SD
        web_ui_utils.update_job(data, frame.i)
        image = frame.generate(data, shared.total_tqdm)

        if image is None:
            raise RuntimeError(f"Failed to generate keyframe at frame {frame.i}")

        # Save keyframe
        keyframe_path = save_keyframe(data, frame, image)
        keyframe_images[frame.i] = keyframe_path

        log_utils.info(f"✅ Keyframe {idx + 1} saved: {os.path.basename(keyframe_path)}", log_utils.GREEN)

        # Set first_frame for UI display (use first generated keyframe)
        if idx == 0:
            data.args.root.first_frame = image

    newly_generated = len(keyframes_to_generate)
    from_resume = len(keyframes_existing)

    # Ensure first_frame is set for UI display (load from disk if not yet set)
    if data.args.root.first_frame is None and keyframes:
        first_keyframe_path = keyframe_images.get(keyframes[0].i)
        if first_keyframe_path and os.path.exists(first_keyframe_path):
            from PIL import Image
            data.args.root.first_frame = Image.open(first_keyframe_path)
            log_utils.info(f"✅ Loaded first frame from disk for UI display", log_utils.GREEN)
    
    log_utils.info(f"\n✅ Phase 1 Complete: {len(keyframes)} keyframes ready", log_utils.GREEN)
    if from_resume > 0:
        log_utils.info(f"   ({newly_generated} newly generated, {from_resume} from previous run)", log_utils.GREEN)
    else:
        log_utils.info(f"   (All {newly_generated} keyframes newly generated with Flux/SD)", log_utils.GREEN)

    # ====================
    # PHASE 2: Batch Frame Interpolation (Wan/RIFE/FILM)
    # ====================
    log_utils.info("\n" + "="*60, log_utils.BLUE)
    log_utils.info("PHASE 2: Batch Frame Interpolation", log_utils.BLUE)
    log_utils.info("="*60, log_utils.BLUE)

    # Get interpolation method (check once for all segments)
    interp_method = getattr(wan_args, 'flux_flf2v_interpolation_method', 'Wan')
    log_utils.info(f"📊 Interpolation method: {interp_method}", log_utils.BLUE)

    # Unload Flux model to free GPU memory
    log_utils.info("🗑️  Unloading Flux model to free GPU memory...", log_utils.BLUE)
    from backend import memory_management
    memory_management.unload_all_models()
    memory_management.soft_empty_cache()
    log_utils.info("✅ GPU memory freed", log_utils.GREEN)

    # Initialize Wan only if needed
    wan_integration = None
    if interp_method == "Wan":
        wan_integration = WanSimpleIntegration(device='cuda')

        # Discover and load Wan model
        log_utils.info("🔍 Discovering Wan FLF2V models...", log_utils.BLUE)
        discovered_models = wan_integration.discover_models()

        if not discovered_models:
            raise RuntimeError("No Wan models found. Please download a Wan model to models/wan directory first.")

        # Use best available FLF2V model
        flf2v_models = [m for m in discovered_models if m['type'] == 'FLF2V']
        if not flf2v_models:
            ti2v_models = [m['name'] for m in discovered_models if m['type'] in ('TI2V', 'T2V', 'I2V')]
            log_utils.error("❌ No FLF2V model found!", log_utils.RED)
            if ti2v_models:
                log_utils.warning(f"   Found T2V/TI2V models: {', '.join(ti2v_models)}", log_utils.YELLOW)
                log_utils.warning("   ⚠️  TI2V/T2V models CANNOT do FLF2V interpolation!", log_utils.YELLOW)
            log_utils.info("   Download FLF2V model: huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers --local-dir models/wan/Wan2.1-FLF2V-14B", log_utils.BLUE)
            raise RuntimeError("FLF2V model required but not found. TI2V models cannot do FLF2V interpolation.")

        model_info = flf2v_models[0]
        log_utils.info(f"📦 Loading Wan model: {model_info['name']}", log_utils.BLUE)

        # Load the Wan pipeline
        success = wan_integration.load_simple_wan_pipeline(model_info, wan_args)
        if not success:
            raise RuntimeError(f"Failed to load Wan model: {model_info['name']}")

    # Generate FLF2V segments
    all_segment_frames = []

    for idx in range(len(keyframes) - 1):
        first_kf = keyframes[idx]
        last_kf = keyframes[idx + 1]

        first_frame_idx = first_kf.i
        last_frame_idx = last_kf.i
        num_tween_frames = last_frame_idx - first_frame_idx - 1  # ONLY in-between frames (exclude both keyframes)

        log_utils.info(f"\n🎞️ Interpolation Segment {idx + 1}/{len(keyframes) - 1}:", log_utils.RED)
        log_utils.info(f"   From keyframe: {first_frame_idx}", log_utils.RED)
        log_utils.info(f"   To keyframe: {last_frame_idx}", log_utils.RED)
        log_utils.info(f"   In-between frames to generate: {num_tween_frames} (frames {first_frame_idx+1} to {last_frame_idx-1})", log_utils.RED)

        # Check if all frames in this segment already exist (resume mode)
        if is_resuming:
            segment_complete = True
            segment_existing_frames = []
            for frame_offset in range(num_tween_frames):
                check_frame_idx = first_frame_idx + frame_offset + 1  # +1 to skip first keyframe

                # Check simple format first (matches our save format: 000000001.png)
                simple_filename = f"{check_frame_idx:09d}.png"
                simple_path = os.path.join(data.output_directory, simple_filename)

                # Also check timestring format (legacy from old runs)
                timestring_filename = filename_utils.frame_filename(data, check_frame_idx)
                timestring_path = os.path.join(data.output_directory, timestring_filename)

                if os.path.exists(simple_path):
                    segment_existing_frames.append(simple_path)
                elif os.path.exists(timestring_path):
                    segment_existing_frames.append(timestring_path)
                else:
                    segment_complete = False
                    break
            
            if segment_complete:
                log_utils.info(f"⏭️  Skipping segment {idx + 1} - all {num_tween_frames} frames already exist", log_utils.YELLOW)
                all_segment_frames.extend(segment_existing_frames)
                continue

        # Get prompts for BOTH keyframes
        first_prompt_idx = min(first_frame_idx, len(data.prompt_series) - 1)
        last_prompt_idx = min(last_frame_idx, len(data.prompt_series) - 1)
        first_prompt_raw = data.prompt_series[first_prompt_idx]
        last_prompt_raw = data.prompt_series[last_prompt_idx]

        # Strip --neg negative prompts (Wan doesn't understand this syntax and will interpret them positively!)
        def strip_negative_prompt(prompt_text):
            """Remove --neg ... portion from Deforum prompts to avoid Wan interpreting them as positive."""
            if '--neg' in prompt_text:
                return prompt_text.split('--neg')[0].strip()
            return prompt_text.strip()

        first_prompt = strip_negative_prompt(first_prompt_raw)
        last_prompt = strip_negative_prompt(last_prompt_raw)

        # Load keyframe images (cv2 uses BGR, convert to RGB)
        first_image_bgr = cv2.imread(keyframe_images[first_frame_idx])
        last_image_bgr = cv2.imread(keyframe_images[last_frame_idx])
        first_image_cv2 = image_utils.bgr_to_rgb(first_image_bgr)
        last_image_cv2 = image_utils.bgr_to_rgb(last_image_bgr)

        # Convert to PIL for Wan FLF2V
        first_image = image_utils.numpy_to_pil(first_image_cv2)
        last_image = image_utils.numpy_to_pil(last_image_cv2)

        # Resize keyframes if resolution changed (e.g., for VRAM savings)
        target_width = data.width()
        target_height = data.height()
        if first_image.size != (target_width, target_height):
            log_utils.info(f"   Resizing keyframes from {first_image.size} to {target_width}x{target_height}", log_utils.YELLOW)
            from PIL import Image
            first_image = first_image.resize((target_width, target_height), Image.LANCZOS)
            last_image = last_image.resize((target_width, target_height), Image.LANCZOS)

        # For FLF2V interpolation, use balanced guidance for semantic interpolation
        # High guidance forces prompt adherence, low guidance allows natural interpolation
        flf2v_guidance = getattr(wan_args, 'wan_flf2v_guidance_scale', 3.5)  # Default 3.5 for smooth morphing

        # Decide how to handle prompts for FLF2V
        # Options: 'none', 'first', 'last', 'blend'
        flf2v_prompt_mode = getattr(wan_args, 'wan_flf2v_prompt_mode', 'blend')  # Default to blend for semantic guidance
        
        if flf2v_prompt_mode == 'none':
            flf2v_prompt = ""
        elif flf2v_prompt_mode == 'first':
            flf2v_prompt = first_prompt
        elif flf2v_prompt_mode == 'last':
            flf2v_prompt = last_prompt
        elif flf2v_prompt_mode == 'blend':
            # Create a blended prompt describing the transition
            flf2v_prompt = f"{first_prompt} transitioning to {last_prompt}"
        else:
            flf2v_prompt = ""  # Default to no prompt
        
        log_utils.info(f"   🎯 Interpolation Settings:", log_utils.BLUE)
        log_utils.info(f"      Method: {interp_method}", log_utils.BLUE)

        # Route to appropriate interpolation function
        if interp_method == "RIFE v4.6":
            log_utils.info(f"      Using RIFE v4.6 optical flow interpolation", log_utils.BLUE)
            segment_frames = generate_rife_segment(
                first_image=first_image,
                last_image=last_image,
                num_frames=num_tween_frames,
                height=data.height(),
                width=data.width(),
                first_frame_idx=first_frame_idx,
                output_dir=data.output_directory,
                fps=video_args.fps
            )
        elif interp_method == "FILM":
            log_utils.info(f"      Using FILM (Frame Interpolation for Large Motion)", log_utils.BLUE)
            segment_frames = generate_film_segment(
                first_image=first_image,
                last_image=last_image,
                num_frames=num_tween_frames,
                height=data.height(),
                width=data.width(),
                first_frame_idx=first_frame_idx,
                output_dir=data.output_directory,
                fps=video_args.fps
            )
        else:  # Default: Wan
            log_utils.info(f"      Guidance scale: {flf2v_guidance} {'(pure interpolation)' if flf2v_guidance == 0.0 else ''}", log_utils.BLUE)
            log_utils.info(f"      Prompt mode: {flf2v_prompt_mode}", log_utils.BLUE)
            log_utils.info(f"      First keyframe prompt: {first_prompt[:60]}...", log_utils.BLUE)
            log_utils.info(f"      Last keyframe prompt: {last_prompt[:60]}...", log_utils.BLUE)
            log_utils.info(f"      → Using: '{flf2v_prompt[:80]}...' {'(empty = pure interpolation)' if not flf2v_prompt else ''}", log_utils.BLUE)
            log_utils.info(f"      Inference steps: {wan_args.wan_inference_steps}", log_utils.BLUE)

            # Call Wan FLF2V
            segment_frames = generate_flf2v_segment(
                wan_integration=wan_integration,
                first_image=first_image,
                last_image=last_image,
                prompt=flf2v_prompt,
                num_frames=num_tween_frames,
                height=data.height(),
                width=data.width(),
                num_inference_steps=wan_args.wan_inference_steps,
                guidance_scale=flf2v_guidance,
                first_frame_idx=first_frame_idx,
                output_dir=data.output_directory
            )

        all_segment_frames.extend(segment_frames)

        log_utils.info(f"✅ Segment {idx + 1} complete: {len(segment_frames)} frames", log_utils.GREEN)

    log_utils.info(f"\n✅ Phase 2 Complete: {len(all_segment_frames)} total frames from {interp_method}", log_utils.GREEN)

    # ====================
    # PHASE 3: Stitch Final Video
    # ====================
    log_utils.info("\n" + "="*60, log_utils.BLUE)
    log_utils.info("PHASE 3: Stitching Final Video", log_utils.BLUE)
    log_utils.info("="*60, log_utils.BLUE)

    # Stitch video using existing utilities
    output_video_path = stitch_wan_flux_video(
        data=data,
        frame_paths=all_segment_frames,
        video_args=video_args,
        interp_method=interp_method
    )

    log_utils.info(f"\n🎉 Flux + Interpolation Generation Complete!", log_utils.GREEN)
    log_utils.info(f"📁 Output: {output_video_path}", log_utils.GREEN)

    # Cleanup Wan if it was loaded
    if wan_integration is not None:
        wan_integration.unload_model()


def save_keyframe(data: RenderData, frame: DiffusionFrame, image):
    """Save keyframe image to disk with simple frame number naming (no timestring prefix)"""
    # Use simple format like Flux/Wan mode: 000000001.png instead of timestring_000000001.png
    filename = f"{frame.i:09d}.png"
    filepath = os.path.join(data.output_directory, filename)

    # Convert CV2 image to PIL if needed, then save
    if image_utils.is_PIL(image):
        image.save(filepath)
    else:
        # Convert numpy/cv2 image to PIL
        pil_image = image_utils.numpy_to_pil(image)
        pil_image.save(filepath)

    return filepath


def generate_flf2v_segment(wan_integration, first_image, last_image, prompt, num_frames,
                           height, width, num_inference_steps, guidance_scale,
                           first_frame_idx, output_dir):
    """Generate frames for one FLF2V segment"""

    # Adjust frame count to Wan's 4n+1 requirement (ROUND UP to avoid gaps)
    import math
    adjusted_frames = math.ceil((num_frames - 1) / 4) * 4 + 1
    if adjusted_frames != num_frames:
        log_utils.info(f"   Wan requires 4n+1 frames: {num_frames} → {adjusted_frames} (will generate extra, use first {num_frames})", log_utils.YELLOW)

    # Generate FLF2V interpolation
    result = wan_integration.pipeline.generate_flf2v(
        first_frame=first_image,
        last_frame=last_image,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=adjusted_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

    # Extract frames from result (handle different output formats)
    frames = None
    if isinstance(result, tuple):
        frames = result[0]
    elif hasattr(result, 'frames'):
        frames = result.frames
    elif hasattr(result, 'images'):
        frames = result.images
    elif hasattr(result, 'videos'):
        frames = result.videos
    else:
        frames = result

    # Convert frames if needed
    if isinstance(frames, list) and len(frames) > 0:
        # Already a list of PIL Images
        frame_list = frames
    elif hasattr(frames, 'shape'):
        # It's a tensor or numpy array
        import torch
        import numpy as np
        from PIL import Image
        
        log_utils.info(f"   Processing FLF2V tensor/array with shape: {frames.shape}", log_utils.BLUE)
        
        if hasattr(frames, 'cpu'):
            frames_np = frames.cpu().numpy()
        else:
            frames_np = np.array(frames)
        
        # Handle different tensor formats
        if len(frames_np.shape) == 5:
            frames_np = frames_np.squeeze(0)
        if len(frames_np.shape) == 4 and frames_np.shape[0] <= 4:
            frames_np = np.transpose(frames_np, (1, 2, 3, 0))
        
        # Extract individual frames
        frame_list = []
        for i in range(frames_np.shape[0]):
            frame = frames_np[i]
            if frame.dtype in [np.float32, np.float64]:
                if frame.min() < 0:
                    frame = (frame + 1.0) / 2.0
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame_list.append(Image.fromarray(frame))
        
        log_utils.info(f"   Extracted {len(frame_list)} FLF2V frames", log_utils.BLUE)
        
    elif hasattr(frames, '__getitem__') and hasattr(frames, '__len__'):
        # It's indexable and has length (like a tensor or array)
        log_utils.info(f"   Converting indexable FLF2V output (length: {len(frames)})", log_utils.BLUE)
        frame_list = [frames[i] for i in range(len(frames))]
    else:
        log_utils.error("Unable to extract frames from FLF2V output")
        raise RuntimeError(f"Unexpected FLF2V output format: {type(result)}")

    # Save frames (only first num_frames, discard extras from 4n+1 padding)
    frame_paths = []
    frames_to_save = min(num_frames, len(frame_list))
    
    if len(frame_list) > num_frames:
        log_utils.info(f"   Generated {len(frame_list)} frames, using first {num_frames} (discarding {len(frame_list) - num_frames} padding frames)", log_utils.YELLOW)
    
    for local_idx in range(frames_to_save):
        frame = frame_list[local_idx]
        # Start at first_frame_idx + 1 to avoid overwriting the first keyframe
        # This generates frames BETWEEN the keyframes, not including them
        global_frame_idx = first_frame_idx + 1 + local_idx
        filename = f"{global_frame_idx:09d}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Ensure it's a PIL Image
        if hasattr(frame, 'save'):
            frame.save(filepath)
        else:
            from PIL import Image
            Image.fromarray(frame).save(filepath)
        
        frame_paths.append(filepath)

    return frame_paths


def stitch_wan_flux_video(data, frame_paths, video_args, interp_method="Wan"):
    """Stitch all frames into final video"""
    from deforum.media.video_audio_utilities import get_ffmpeg_params

    # Get ffmpeg parameters from settings
    ffmpeg_location, ffmpeg_crf, ffmpeg_preset = get_ffmpeg_params()

    # Output video path - use interpolation method in filename
    # Wan → flux_wan, RIFE → flux_rife, FILM → flux_film
    method_suffix = interp_method.lower()  # "wan", "rife", "film"
    output_filename = f"{data.args.root.timestring}_flux_{method_suffix}.mp4"
    output_path = os.path.join(data.output_directory, output_filename)

    # Build frame pattern (frames are named 000000001.png, 000000002.png, etc.)
    imgs_pattern = os.path.join(data.output_directory, "%09d.png")

    # Stitch with ffmpeg
    log_utils.info(f"🎬 Stitching {len(frame_paths)} frames...", log_utils.BLUE)

    ffmpeg_stitch_video(
        ffmpeg_location=ffmpeg_location,
        fps=video_args.fps,
        outmp4_path=output_path,
        stitch_from_frame=1,
        stitch_to_frame=len(frame_paths),  # Total number of frames to stitch
        imgs_path=imgs_pattern,
        add_soundtrack=video_args.add_soundtrack,
        audio_path=video_args.soundtrack_path if video_args.add_soundtrack == 'File' else None,
        crf=ffmpeg_crf,
        preset=ffmpeg_preset
    )

    return output_path


def generate_rife_segment(first_image, last_image, num_frames, height, width,
                         first_frame_idx, output_dir, fps=30):
    """
    Generate frames for one RIFE v4.6 interpolation segment.

    Uses optical flow-based frame interpolation to generate smooth transitions
    between two keyframes.

    Args:
        first_image: PIL Image of first keyframe
        last_image: PIL Image of last keyframe
        num_frames: Number of tween frames to generate (excluding keyframes)
        height: Frame height
        width: Frame width
        first_frame_idx: Global index of first keyframe
        output_dir: Directory to save generated frames
        fps: Target FPS for interpolation

    Returns:
        List of paths to generated tween frames
    """
    import shutil
    from deforum.integrations.external_repos.rife.inference_video import run_rife_new_video_infer

    log_utils.info(f"   🔧 RIFE v4.6 interpolation: {num_frames} frames", log_utils.BLUE)

    # Create working directory for RIFE in the output directory (not /tmp)
    # This avoids cleanup issues and keeps intermediate files with the project
    rife_work_dir = os.path.join(output_dir, f"_rife_segment_{first_frame_idx}_to_{first_frame_idx + num_frames + 1}")
    temp_input = os.path.join(rife_work_dir, "input")
    temp_output = os.path.join(rife_work_dir, "output")
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)

    try:
        # Save first and last keyframes to temp input directory
        # RIFE expects numbered frames: 0.png, 1.png
        first_image.save(os.path.join(temp_input, "0000000.png"))
        last_image.save(os.path.join(temp_input, "0000001.png"))

        # Calculate interpolation amount
        # RIFE will generate interp_x_amount - 1 frames between each pair
        # So interp_x_amount = num_frames + 1
        interp_amount = num_frames + 1

        log_utils.info(f"   Temp input: {temp_input}", log_utils.YELLOW)
        log_utils.info(f"   Interpolation factor: {interp_amount}x", log_utils.YELLOW)

        # Call RIFE inference
        from deforum.media.video_audio_utilities import get_ffmpeg_params
        ffmpeg_location, _, _ = get_ffmpeg_params()

        run_rife_new_video_infer(
            output=temp_output,
            model="RIFE46",
            fp16=False,
            UHD=(height >= 2048 or width >= 2048),
            scale=1.0,
            fps=fps,
            deforum_models_path=os.path.join(os.getcwd(), "models", "Deforum"),
            raw_output_imgs_path=temp_input,
            img_batch_id=None,
            ffmpeg_location=ffmpeg_location,
            audio_track=None,
            interp_x_amount=interp_amount,
            slow_mo_enabled=False,
            slow_mo_x_amount=2,
            ffmpeg_crf=17,
            ffmpeg_preset='veryslow',
            keep_imgs=True,  # Keep interpolated frames
            orig_vid_name=None,  # None = PNG input (don't cleanup), non-None = video input (cleanup after stitch)
            srt_path=None
        )

        # Find RIFE output directory
        # RIFE creates output in the input directory, not the output directory
        search_dir = temp_input
        rife_output_dirs = [d for d in os.listdir(search_dir) if d.startswith("interpolated_frames_rife")]
        if not rife_output_dirs:
            # Debug: list what's actually in the directory
            actual_contents = os.listdir(search_dir)
            raise RuntimeError(f"RIFE output directory not found in {search_dir}. Found: {actual_contents}")

        rife_output_dir = os.path.join(search_dir, rife_output_dirs[0])

        # RIFE generates: first_kf, tween1, tween2, ..., tweenN, last_kf
        # We want only the tweens (frames between keyframes)
        rife_frames = sorted([f for f in os.listdir(rife_output_dir) if f.endswith('.png')])

        # Skip first and last frame (those are the keyframes)
        tween_frames = rife_frames[1:-1] if len(rife_frames) > 2 else rife_frames

        # Move tween frames to output directory with correct naming
        frame_paths = []
        for local_idx, rife_filename in enumerate(tween_frames[:num_frames]):
            # Calculate global frame index (start after first keyframe)
            global_frame_idx = first_frame_idx + 1 + local_idx
            target_filename = f"{global_frame_idx:09d}.png"
            target_path = os.path.join(output_dir, target_filename)

            # Copy frame from RIFE output to final output directory
            rife_frame_path = os.path.join(rife_output_dir, rife_filename)
            shutil.copy2(rife_frame_path, target_path)
            frame_paths.append(target_path)

        log_utils.info(f"   ✅ RIFE generated {len(frame_paths)} tween frames", log_utils.GREEN)
        return frame_paths

    finally:
        # Cleanup RIFE working directory
        if os.path.exists(rife_work_dir):
            shutil.rmtree(rife_work_dir, ignore_errors=True)


def generate_film_segment(first_image, last_image, num_frames, height, width,
                         first_frame_idx, output_dir, fps=30):
    """
    Generate frames for one FILM (Frame Interpolation for Large Motion) segment.

    Uses Google's ML-based frame interpolation to generate smooth transitions
    between two keyframes.

    Args:
        first_image: PIL Image of first keyframe
        last_image: PIL Image of last keyframe
        num_frames: Number of tween frames to generate (excluding keyframes)
        height: Frame height
        width: Frame width
        first_frame_idx: Global index of first keyframe
        output_dir: Directory to save generated frames
        fps: Target FPS for interpolation

    Returns:
        List of paths to generated tween frames
    """
    import shutil
    from deforum.integrations.film import film_interpolate

    log_utils.info(f"   🎬 FILM interpolation: {num_frames} frames", log_utils.BLUE)

    # Create working directory for FILM in the output directory (not /tmp)
    # This avoids cleanup issues and keeps intermediate files with the project
    film_work_dir = os.path.join(output_dir, f"_film_segment_{first_frame_idx}_to_{first_frame_idx + num_frames + 1}")
    temp_input = os.path.join(film_work_dir, "input")
    temp_output = os.path.join(film_work_dir, "output")
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)

    try:
        # Save first and last keyframes to temp input directory
        # FILM expects numbered frames: 0.png, 1.png
        first_image.save(os.path.join(temp_input, "0000000.png"))
        last_image.save(os.path.join(temp_input, "0000001.png"))

        # Calculate interpolation amount (FILM uses 2^n recursion)
        # For num_frames, find nearest power of 2
        import math
        recursion_depth = max(1, math.ceil(math.log2(num_frames + 1)))
        total_frames_generated = 2 ** recursion_depth + 1  # Includes keyframes

        log_utils.info(f"   Temp input: {temp_input}", log_utils.YELLOW)
        log_utils.info(f"   Recursion depth: {recursion_depth} (generates {total_frames_generated - 2} tweens)", log_utils.YELLOW)

        # Call FILM interpolation
        film_interpolate(
            input_dir=temp_input,
            output_dir=temp_output,
            times_to_interpolate=recursion_depth,
            fps=fps
        )

        # Find FILM output frames
        film_frames = sorted([f for f in os.listdir(temp_output) if f.endswith('.png')])

        # Skip first and last frame (those are the keyframes)
        tween_frames = film_frames[1:-1] if len(film_frames) > 2 else film_frames

        # Move tween frames to output directory with correct naming
        frame_paths = []
        for local_idx, film_filename in enumerate(tween_frames[:num_frames]):
            # Calculate global frame index (start after first keyframe)
            global_frame_idx = first_frame_idx + 1 + local_idx
            target_filename = f"{global_frame_idx:09d}.png"
            target_path = os.path.join(output_dir, target_filename)

            # Copy frame from FILM output to final output directory
            film_frame_path = os.path.join(temp_output, film_filename)
            shutil.copy2(film_frame_path, target_path)
            frame_paths.append(target_path)

        log_utils.info(f"   ✅ FILM generated {len(frame_paths)} tween frames", log_utils.GREEN)
        return frame_paths

    finally:
        # Cleanup FILM working directory
        if os.path.exists(film_work_dir):
            shutil.rmtree(film_work_dir, ignore_errors=True)
