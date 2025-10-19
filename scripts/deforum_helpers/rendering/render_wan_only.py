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

    keyframe_images = {}  # {frame_index: image_path}

    for idx, frame in enumerate(keyframes):
        log_utils.info(f"\n📸 Generating keyframe {idx + 1}/{len(keyframes)} (frame {frame.i})...", log_utils.YELLOW)

        # Get prompt for this keyframe
        prompt = data.prompt_series[frame.i]
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

    log_utils.info(f"\n✅ Phase 1 Complete: {len(keyframes)} keyframes generated with Wan T2V", log_utils.GREEN)

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
        num_tween_frames = last_frame_idx - first_frame_idx + 1  # Total frames including both keyframes

        log_utils.info(f"\n🎞️ FLF2V Segment {idx + 1}/{len(keyframes) - 1}:", log_utils.MAGENTA)
        log_utils.info(f"   From keyframe: {first_frame_idx}", log_utils.MAGENTA)
        log_utils.info(f"   To keyframe: {last_frame_idx}", log_utils.MAGENTA)
        log_utils.info(f"   Total frames: {num_tween_frames}", log_utils.MAGENTA)

        # Get prompt for this segment (use first keyframe's prompt)
        prompt = data.prompt_series[first_frame_idx]

        # Load keyframe images (load_image returns cv2/numpy format)
        first_image_cv2 = image_utils.load_image(keyframe_images[first_frame_idx])
        last_image_cv2 = image_utils.load_image(keyframe_images[last_frame_idx])
        
        # Convert to PIL for Wan FLF2V
        first_image = image_utils.numpy_to_pil(first_image_cv2)
        last_image = image_utils.numpy_to_pil(last_image_cv2)

        # Call Wan FLF2V
        segment_frames = generate_flf2v_segment(
            wan_integration=wan_integration,
            first_image=first_image,
            last_image=last_image,
            prompt=prompt,
            num_frames=num_tween_frames,
            height=data.height(),
            width=data.width(),
            num_inference_steps=wan_args.wan_inference_steps,
            guidance_scale=wan_args.wan_guidance_scale,
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
    else:
        log_utils.error("Unable to extract frames from Wan T2V output")
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

    # Adjust frame count to Wan's 4n+1 requirement
    adjusted_frames = ((num_frames - 1) // 4) * 4 + 1
    if adjusted_frames != num_frames:
        log_utils.info(f"   Adjusted frame count: {num_frames} → {adjusted_frames} (4n+1 requirement)", log_utils.YELLOW)

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
    else:
        log_utils.error("Unable to extract frames from FLF2V output")
        raise RuntimeError(f"Unexpected FLF2V output format: {type(result)}")

    # Save frames
    frame_paths = []
    for local_idx, frame in enumerate(frame_list):
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
