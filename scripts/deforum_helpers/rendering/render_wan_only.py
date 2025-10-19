"""
Wan Only Mode: Pure Wan T2V + FLF2V Interpolation

Architecture:
  Phase 1: Batch generate ALL keyframes with Wan T2V
  Phase 2: Batch run Wan FLF2V between each consecutive keyframe pair
  Phase 3: Stitch final video

This is fundamentally different from the integrated FLF2V approach:
- Sequential (2D/3D): Keyframe → Tweens → Keyframe → Tweens
- Wan Only (this): ALL Keyframes (Wan T2V) → ALL FLF2V interpolations → Stitch

No SD model required - 100% Wan-powered generation!
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


def render_wan_only(args, anim_args, video_args, parseq_args, loop_args, controlnet_args,
                    freeu_args, kohya_hrfix_args, wan_args, root):
    """
    Wan Only rendering mode: Pure Wan AI video generation

    1. Generate all keyframes with Wan T2V
    2. Interpolate between keyframes with Wan FLF2V
    3. Stitch final video
    
    No SD model required!
    """
    log_utils.info("🎬 Wan Only Mode: Pure Wan T2V + FLF2V Workflow", log_utils.BLUE)

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

    log_utils.info(f"📊 Wan Only Workflow:", log_utils.BLUE)
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
    # PHASE 0: Initialize Wan
    # ====================
    log_utils.info("\n" + "="*60, log_utils.PURPLE)
    log_utils.info("PHASE 0: Initializing Wan Pipeline", log_utils.PURPLE)
    log_utils.info("="*60, log_utils.PURPLE)
    
    wan_integration = WanSimpleIntegration(device='cuda')
    
    # Discover and load Wan model
    log_utils.info("🔍 Discovering Wan models...", log_utils.BLUE)
    discovered_models = wan_integration.discover_models()
    
    if not discovered_models:
        raise RuntimeError("No Wan models found. Please download a Wan model to models/wan directory first.")
    
    # Use best available model (T2V capable)
    model_info = wan_integration.get_best_model()
    if not model_info:
        model_info = discovered_models[0]
    
    log_utils.info(f"📦 Loading Wan model: {model_info['name']}", log_utils.BLUE)
    success = wan_integration.load_simple_wan_pipeline(model_info, wan_args)
    if not success:
        raise RuntimeError(f"Failed to load Wan model: {model_info['name']}")
    
    log_utils.info("✅ Wan pipeline loaded successfully", log_utils.GREEN)

    # ====================
    # PHASE 1: Batch Generate All Keyframes with Wan T2V
    # ====================
    log_utils.info("\n" + "="*60, log_utils.GREEN)
    log_utils.info("PHASE 1: Batch Keyframe Generation with Wan T2V", log_utils.GREEN)
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

        # Get prompt for this keyframe (handle 0-based indexing and bounds)
        prompt_idx = min(frame.i, len(data.prompt_series) - 1)
        prompt = data.prompt_series[prompt_idx]
        if frame.i != prompt_idx:
            log_utils.info(f"   Using prompt from index {prompt_idx} (frame {frame.i} out of range)", log_utils.YELLOW)
        log_utils.info(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}", log_utils.YELLOW)

        # Generate keyframe using Wan T2V
        keyframe_path = generate_wan_t2v_keyframe(
            wan_integration=wan_integration,
            prompt=prompt,
            height=data.height(),
            width=data.width(),
            num_inference_steps=wan_args.wan_inference_steps,
            guidance_scale=wan_args.wan_guidance_scale,
            frame_idx=frame.i,
            output_dir=data.output_directory,
            timestring=data.args.root.timestring
        )

        keyframe_images[frame.i] = keyframe_path
        log_utils.info(f"✅ Keyframe {idx + 1} saved: {os.path.basename(keyframe_path)}", log_utils.GREEN)

    newly_generated = len(keyframes_to_generate)
    from_resume = len(keyframes_existing)
    
    log_utils.info(f"\n✅ Phase 1 Complete: {len(keyframes)} keyframes ready", log_utils.GREEN)
    if from_resume > 0:
        log_utils.info(f"   ({newly_generated} newly generated, {from_resume} from previous run)", log_utils.GREEN)
    else:
        log_utils.info(f"   (All {newly_generated} keyframes newly generated with Wan T2V)", log_utils.GREEN)

    # ====================
    # PHASE 2: Batch Wan FLF2V Interpolation
    # ====================
    log_utils.info("\n" + "="*60, log_utils.BLUE)
    log_utils.info("PHASE 2: Batch Wan FLF2V Interpolation", log_utils.BLUE)
    log_utils.info("="*60, log_utils.BLUE)

    # Generate FLF2V segments (using already-loaded Wan model from Phase 0)
    all_segment_frames = []

    for idx in range(len(keyframes) - 1):
        first_kf = keyframes[idx]
        last_kf = keyframes[idx + 1]

        first_frame_idx = first_kf.i
        last_frame_idx = last_kf.i
        num_tween_frames = last_frame_idx - first_frame_idx - 1  # ONLY in-between frames (exclude both keyframes)

        log_utils.info(f"\n🎞️ FLF2V Segment {idx + 1}/{len(keyframes) - 1}:", log_utils.RED)
        log_utils.info(f"   From keyframe: {first_frame_idx}", log_utils.RED)
        log_utils.info(f"   To keyframe: {last_frame_idx}", log_utils.RED)
        log_utils.info(f"   In-between frames to generate: {num_tween_frames} (frames {first_frame_idx+1} to {last_frame_idx-1})", log_utils.RED)

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
        
        # DEBUG: Check all wan_args attributes
        log_utils.info(f"   🔍 DEBUG wan_args attributes:", log_utils.YELLOW)
        for attr in sorted(dir(wan_args)):
            if not attr.startswith('_') and 'flf2v' in attr.lower():
                log_utils.info(f"      {attr} = {getattr(wan_args, attr, 'N/A')}", log_utils.YELLOW)
        
        flf2v_guidance = getattr(wan_args, 'wan_flf2v_guidance_scale', 0.0)  # Default 0.0 = pure interpolation
        log_utils.info(f"   🔍 DEBUG: flf2v_guidance final value = {flf2v_guidance}", log_utils.YELLOW)
        
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
        
        # Show what we're actually using
        log_utils.info(f"   🎯 FLF2V Settings:", log_utils.BLUE)
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

    log_utils.info(f"\n✅ Phase 2 Complete: {len(all_segment_frames)} total frames from FLF2V", log_utils.GREEN)

    # ====================
    # PHASE 3: Stitch Final Video
    # ====================
    log_utils.info("\n" + "="*60, log_utils.BLUE)
    log_utils.info("PHASE 3: Stitching Final Video", log_utils.BLUE)
    log_utils.info("="*60, log_utils.BLUE)

    # Stitch video using existing utilities
    output_video_path = stitch_wan_only_video(
        data=data,
        frame_paths=all_segment_frames,
        video_args=video_args
    )

    log_utils.info(f"\n🎉 Wan Only Generation Complete!", log_utils.GREEN)
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


def generate_wan_t2v_keyframe(wan_integration, prompt, height, width, num_inference_steps,
                               guidance_scale, frame_idx, output_dir, timestring):
    """Generate a single keyframe using Wan T2V"""
    
    # Wan generates short videos (e.g. 17 frames), we'll extract the middle frame as keyframe
    # This gives better quality than single image generation
    num_frames_for_video = 17  # Standard Wan T2V output
    
    log_utils.info(f"   Generating {num_frames_for_video}-frame video clip to extract keyframe...", log_utils.BLUE)
    
    # Generate short video with Wan T2V
    result = wan_integration.pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames_for_video,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )
    
    # Extract frames from result (handle different output formats)
    log_utils.info(f"   Output type: {type(result)}", log_utils.BLUE)
    log_utils.info(f"   Available attributes: {[a for a in dir(result) if not a.startswith('_')][:10]}", log_utils.BLUE)
    
    frames = None
    if isinstance(result, tuple):
        log_utils.info("   Extracting from tuple", log_utils.BLUE)
        frames = result[0]
    elif hasattr(result, 'frames'):
        log_utils.info("   Extracting from .frames", log_utils.BLUE)
        frames = result.frames
    elif hasattr(result, 'images'):
        log_utils.info("   Extracting from .images", log_utils.BLUE)
        frames = result.images
    elif hasattr(result, 'videos'):
        log_utils.info("   Extracting from .videos", log_utils.BLUE)
        frames = result.videos
    else:
        log_utils.info("   Using result directly", log_utils.BLUE)
        frames = result
    
    # Convert frames if needed
    log_utils.info(f"   Frames type: {type(frames)}", log_utils.BLUE)
    
    if isinstance(frames, list) and len(frames) > 0:
        # Already a list of PIL Images
        frame_list = frames
    elif hasattr(frames, 'shape'):
        # It's a tensor or numpy array
        import torch
        import numpy as np
        from PIL import Image
        
        log_utils.info(f"   Processing tensor/array with shape: {frames.shape}", log_utils.BLUE)
        
        # Convert to numpy if it's a torch tensor
        if hasattr(frames, 'cpu'):
            frames_np = frames.cpu().numpy()
        else:
            frames_np = np.array(frames)
        
        log_utils.info(f"   Numpy shape: {frames_np.shape}, dtype: {frames_np.dtype}", log_utils.BLUE)
        
        # Handle different tensor formats
        # Expected formats: (B, F, H, W, C) or (B, C, F, H, W) or (F, H, W, C)
        if len(frames_np.shape) == 5:
            # (B, F, H, W, C) or (B, C, F, H, W)
            frames_np = frames_np.squeeze(0)  # Remove batch dimension -> (F, H, W, C) or (C, F, H, W)
            
        if len(frames_np.shape) == 4:
            # Check if it's (F, H, W, C) or (C, F, H, W)
            if frames_np.shape[0] <= 4:  # Likely (C, F, H, W) where C is small
                # Transpose to (F, H, W, C)
                frames_np = np.transpose(frames_np, (1, 2, 3, 0))
        
        log_utils.info(f"   After processing shape: {frames_np.shape}", log_utils.BLUE)
        
        # Extract individual frames
        frame_list = []
        num_frames = frames_np.shape[0]
        for i in range(num_frames):
            frame = frames_np[i]  # (H, W, C)
            
            # Normalize to [0, 255] uint8
            if frame.dtype in [np.float32, np.float64]:
                if frame.min() < 0:  # [-1, 1] range
                    frame = (frame + 1.0) / 2.0
                # Now in [0, 1] range, convert to [0, 255]
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            
            frame_list.append(Image.fromarray(frame))
        
        log_utils.info(f"   Extracted {len(frame_list)} frames", log_utils.BLUE)
        
    elif hasattr(frames, '__getitem__') and hasattr(frames, '__len__'):
        # It's indexable and has length
        log_utils.info(f"   Converting indexable object (length: {len(frames)})", log_utils.BLUE)
        frame_list = [frames[i] for i in range(len(frames))]
    else:
        log_utils.error(f"Unable to extract frames from Wan T2V output")
        log_utils.error(f"Result type: {type(result)}, Frames type: {type(frames)}")
        log_utils.error(f"Has __getitem__: {hasattr(frames, '__getitem__')}, Has __len__: {hasattr(frames, '__len__')}")
        raise RuntimeError(f"Unexpected Wan output format: {type(result)}")
    
    # Extract middle frame as the keyframe (best quality/consistency)
    middle_idx = num_frames_for_video // 2
    if middle_idx >= len(frame_list):
        middle_idx = len(frame_list) - 1
    
    keyframe = frame_list[middle_idx]
    
    # Save keyframe
    filename = f"{frame_idx:09d}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Ensure it's a PIL Image
    if hasattr(keyframe, 'save'):
        keyframe.save(filepath)
    else:
        from PIL import Image
        Image.fromarray(keyframe).save(filepath)
    
    log_utils.info(f"   Extracted frame {middle_idx+1}/{len(frame_list)} as keyframe", log_utils.BLUE)
    
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
        # It's a tensor or numpy array - use same processing as T2V
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


def stitch_wan_only_video(data, frame_paths, video_args):
    """Stitch all frames into final video"""

    # Create frame list file for ffmpeg
    frame_list_file = os.path.join(data.output_directory, "frame_list.txt")
    with open(frame_list_file, 'w') as f:
        for path in frame_paths:
            f.write(f"file '{path}'\n")

    # Output video path
    output_filename = f"{data.args.root.timestring}_wan_only.mp4"
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
