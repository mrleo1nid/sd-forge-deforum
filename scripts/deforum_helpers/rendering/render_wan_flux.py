"""
Wan Flux Mode: Flux Keyframes + Wan FLF2V Interpolation

Architecture:
  Phase 1: Generate ALL keyframes with Flux/SD
  Phase 2: Batch run Wan FLF2V between each consecutive keyframe pair
  Phase 3: Stitch final video

This combines the best of both worlds:
- High-quality Flux-generated keyframes
- Smooth Wan FLF2V interpolation between keyframes
"""

import os
import json
from pathlib import Path
from typing import List

from modules import shared  # type: ignore

from .data.render_data import RenderData
from .data.frame import KeyFrameDistribution, DiffusionFrame
from .data.taqaddumat import Taqaddumat
from .util import log_utils, web_ui_utils, image_utils, filename_utils
from ..wan.wan_simple_integration import WanSimpleIntegration
from ..video_audio_utilities import ffmpeg_stitch_video


def render_wan_flux(args, anim_args, video_args, parseq_args, loop_args, controlnet_args,
                    freeu_args, kohya_hrfix_args, wan_args, root):
    """
    Wan Flux rendering mode: Flux for keyframes + Wan for interpolation

    1. Generate all keyframes with Flux/SD
    2. Interpolate between keyframes with Wan FLF2V
    3. Stitch final video
    """
    log_utils.info("🎬 Wan Flux Mode: Flux Keyframes + Wan FLF2V Interpolation", log_utils.BLUE)

    # Create render data
    data = RenderData.create(args, parseq_args, anim_args, video_args, loop_args, controlnet_args,
                            freeu_args, kohya_hrfix_args, root)

    # Initialize progress tracking
    web_ui_utils.init_job(data)
    shared.total_tqdm = Taqaddumat()

    # Get keyframe distribution
    keyframe_distribution = KeyFrameDistribution.from_UI_tab(data)
    all_frames = DiffusionFrame.create_all_frames(data, keyframe_distribution)

    # Extract only keyframes (frames with is_keyframe=True)
    keyframes = [f for f in all_frames if f.is_keyframe]

    log_utils.info(f"📊 Wan Flux Workflow:", log_utils.BLUE)
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
            expected_filename = filename_utils.frame_filename(data, frame.i)
            expected_path = os.path.join(data.output_directory, expected_filename)
            
            # Also check for filename without timestring prefix (legacy format)
            alt_filename = f"{frame.i:09}.png"
            alt_path = os.path.join(data.output_directory, alt_filename)
            
            if os.path.exists(expected_path):
                keyframe_images[frame.i] = expected_path
                log_utils.info(f"   ✓ Found existing keyframe: {expected_filename}", log_utils.GREEN)
            elif os.path.exists(alt_path):
                keyframe_images[frame.i] = alt_path
                log_utils.info(f"   ✓ Found existing keyframe (alt format): {alt_filename}", log_utils.GREEN)
            else:
                log_utils.info(f"   ✗ Missing keyframe at frame {frame.i} (tried: {expected_filename}, {alt_filename})", log_utils.YELLOW)
        
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

        # Generate keyframe image using Flux/SD
        web_ui_utils.update_job(data, frame.i)
        image = frame.generate(data, shared.total_tqdm)

        if image is None:
            raise RuntimeError(f"Failed to generate keyframe at frame {frame.i}")

        # Save keyframe
        keyframe_path = save_keyframe(data, frame, image)
        keyframe_images[frame.i] = keyframe_path

        log_utils.info(f"✅ Keyframe {idx + 1} saved: {os.path.basename(keyframe_path)}", log_utils.GREEN)

    newly_generated = len(keyframes_to_generate)
    from_resume = len(keyframes_existing)
    
    log_utils.info(f"\n✅ Phase 1 Complete: {len(keyframes)} keyframes ready", log_utils.GREEN)
    if from_resume > 0:
        log_utils.info(f"   ({newly_generated} newly generated, {from_resume} from previous run)", log_utils.GREEN)
    else:
        log_utils.info(f"   (All {newly_generated} keyframes newly generated with Flux/SD)", log_utils.GREEN)

    # ====================
    # PHASE 2: Batch Wan FLF2V Interpolation
    # ====================
    log_utils.info("\n" + "="*60, log_utils.BLUE)
    log_utils.info("PHASE 2: Batch Wan FLF2V Interpolation", log_utils.BLUE)
    log_utils.info("="*60, log_utils.BLUE)

    # Initialize Wan
    wan_integration = WanSimpleIntegration(device='cuda')

    # Discover and load Wan model
    log_utils.info("🔍 Discovering Wan FLF2V models...", log_utils.BLUE)
    discovered_models = wan_integration.discover_models()

    if not discovered_models:
        raise RuntimeError("No Wan models found. Please download a Wan model to models/wan directory first.")

    # Use best available FLF2V-capable model
    flf2v_models = [m for m in discovered_models if m['supports_flf2v']]
    if not flf2v_models:
        raise RuntimeError("No FLF2V-capable Wan models found. Please download Wan 2.1+ model.")

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
        num_tween_frames = last_frame_idx - first_frame_idx + 1  # Total frames including both keyframes

        log_utils.info(f"\n🎞️ FLF2V Segment {idx + 1}/{len(keyframes) - 1}:", log_utils.RED)
        log_utils.info(f"   From keyframe: {first_frame_idx}", log_utils.RED)
        log_utils.info(f"   To keyframe: {last_frame_idx}", log_utils.RED)
        log_utils.info(f"   Total frames: {num_tween_frames}", log_utils.RED)

        # Check if all frames in this segment already exist (resume mode)
        if is_resuming:
            segment_complete = True
            segment_existing_frames = []
            for frame_offset in range(num_tween_frames):
                check_frame_idx = first_frame_idx + frame_offset
                check_filename = filename_utils.frame_filename(data, check_frame_idx)
                check_path = os.path.join(data.output_directory, check_filename)
                
                # Also check alternative format without timestring
                alt_filename = f"{check_frame_idx:09}.png"
                alt_path = os.path.join(data.output_directory, alt_filename)
                
                if os.path.exists(check_path):
                    segment_existing_frames.append(check_path)
                elif os.path.exists(alt_path):
                    segment_existing_frames.append(alt_path)
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
        first_prompt = data.prompt_series[first_prompt_idx]
        last_prompt = data.prompt_series[last_prompt_idx]

        # Load keyframe images
        first_image_cv2 = image_utils.load_image(keyframe_images[first_frame_idx])
        last_image_cv2 = image_utils.load_image(keyframe_images[last_frame_idx])

        # Convert to PIL for Wan FLF2V
        first_image = image_utils.numpy_to_pil(first_image_cv2)
        last_image = image_utils.numpy_to_pil(last_image_cv2)
        
        # For FLF2V interpolation, use minimal guidance to let model smoothly transition
        # High guidance forces prompt adherence, low guidance allows natural interpolation
        flf2v_guidance = getattr(wan_args, 'wan_flf2v_guidance_scale', 1.0)  # Much lower than T2V
        
        # Decide how to handle prompts for FLF2V
        # Options: 'none', 'first', 'last', 'blend'
        flf2v_prompt_mode = getattr(wan_args, 'wan_flf2v_prompt_mode', 'none')
        
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
        
        log_utils.info(f"   🎯 FLF2V Settings:", log_utils.BLUE)
        log_utils.info(f"      Prompt mode: {flf2v_prompt_mode}", log_utils.BLUE)
        log_utils.info(f"      First keyframe prompt: {first_prompt[:60]}...", log_utils.BLUE)
        log_utils.info(f"      Last keyframe prompt: {last_prompt[:60]}...", log_utils.BLUE)
        log_utils.info(f"      → Using: '{flf2v_prompt[:80]}...' {'(empty = pure interpolation)' if not flf2v_prompt else ''}", log_utils.BLUE)
        log_utils.info(f"      Guidance scale: {flf2v_guidance}", log_utils.BLUE)
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

    log_utils.info(f"\n✅ Phase 2 Complete: {len(all_segment_frames)} total frames from FLF2V", log_utils.GREEN)

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
        video_args=video_args
    )

    log_utils.info(f"\n🎉 Wan Flux Generation Complete!", log_utils.GREEN)
    log_utils.info(f"📁 Output: {output_video_path}", log_utils.GREEN)

    # Cleanup
    wan_integration.unload_model()


def save_keyframe(data: RenderData, frame: DiffusionFrame, image):
    """Save keyframe image to disk"""
    filename = filename_utils.frame_filename(data, frame.i)
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
        global_frame_idx = first_frame_idx + local_idx
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


def stitch_wan_flux_video(data, frame_paths, video_args):
    """Stitch all frames into final video"""

    # Create frame list file for ffmpeg
    frame_list_file = os.path.join(data.output_directory, "frame_list.txt")
    with open(frame_list_file, 'w') as f:
        for path in frame_paths:
            f.write(f"file '{path}'\n")

    # Output video path
    output_filename = f"{data.args.root.timestring}_wan_flux.mp4"
    output_path = os.path.join(data.output_directory, output_filename)

    # Stitch with ffmpeg
    log_utils.info(f"🎬 Stitching {len(frame_paths)} frames...", log_utils.BLUE)

    ffmpeg_stitch_video(
        frame_pattern=None,  # Use frame list instead
        fps=video_args.fps,
        output_path=output_path,
        crf=video_args.crf,
        preset=video_args.preset,
        audio_path=video_args.soundtrack_path if video_args.add_soundtrack == 'File' else None,
        frame_list_file=frame_list_file
    )

    return output_path

