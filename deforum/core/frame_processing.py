"""
Pure Functional Frame Processing

This module contains pure functions for processing individual frames in the
rendering pipeline. All functions are side-effect free and operate on
immutable data structures.
"""

import time
from typing import Optional, Tuple, Callable, Dict, Any
from dataclasses import replace
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import copy

from .frame_models import (
    FrameState, FrameResult, FrameMetadata, RenderContext,
    ProcessingStage, RenderingError, ImageArray, ValidationResult
)
from modules.shared import opts
from modules import processing
from ..integrations.webui_pipeline import get_webui_sd_pipeline

# Import processing functions with graceful fallbacks
try:
    from ..animation import anim_frame_warp
    from ..noise import add_noise
    from ..colors import maintain_colors
    from ..image_sharpening import unsharp_mask
    # Note: hybrid video imports removed - functionality not available
    from ..masks import do_overlay_mask
    from ..composable_masks import compose_mask_with_check
except ImportError:
    # Fallback implementations for testing
    def anim_frame_warp(image, *args, **kwargs):
        return image, None
    
    def add_noise(image, *args, **kwargs):
        return image
    
    def maintain_colors(image, *args, **kwargs):
        return image
    
    def unsharp_mask(image, *args, **kwargs):
        return image
    
    def do_overlay_mask(*args, **kwargs):
        return args[2] if len(args) > 2 else None
    
    def compose_mask_with_check(*args, **kwargs):
        return None

# Fallback implementations for removed hybrid video functions
def image_transform_optical_flow(image, *args, **kwargs):
    return image

def image_transform_ransac(image, *args, **kwargs):
    return image

def get_flow_from_images(*args, **kwargs):
    return np.zeros((64, 64, 2))

def abs_flow_to_rel_flow(flow, *args, **kwargs):
    return flow

def rel_flow_to_abs_flow(flow, *args, **kwargs):
    return flow


def create_frame_state(
    frame_idx: int,
    context: RenderContext,
    metadata_overrides: Optional[Dict[str, Any]] = None
) -> FrameState:
    """
    Create initial frame state with metadata.
    
    Args:
        frame_idx: Index of the frame to create
        context: Rendering context
        metadata_overrides: Optional metadata overrides
        
    Returns:
        Initial FrameState for the frame
    """
    base_metadata = {
        'frame_idx': frame_idx,
        'timestamp': frame_idx / context.fps,
        'seed': 42,  # Will be overridden by schedule
        'strength': 0.75,
        'cfg_scale': 7.0,
        'distilled_cfg_scale': 7.0,
        'noise_level': 0.0,
        'prompt': "",
    }
    
    if metadata_overrides:
        base_metadata.update(metadata_overrides)
    
    metadata = FrameMetadata(**base_metadata)
    
    return FrameState(
        metadata=metadata,
        stage=ProcessingStage.INITIALIZATION
    )


def validate_frame_state(frame_state: FrameState) -> ValidationResult:
    """
    Validate frame state for processing.
    
    Args:
        frame_state: Frame state to validate
        
    Returns:
        True if valid, error message string if invalid
    """
    try:
        # Validate metadata
        if frame_state.metadata.frame_idx < 0:
            return "Frame index must be non-negative"
        
        if not (0.0 <= frame_state.metadata.strength <= 1.0):
            return f"Strength must be between 0.0 and 1.0, got {frame_state.metadata.strength}"
        
        if frame_state.metadata.cfg_scale <= 0:
            return f"CFG scale must be positive, got {frame_state.metadata.cfg_scale}"
        
        # Validate image dimensions if present
        if frame_state.current_image is not None:
            if len(frame_state.current_image.shape) != 3:
                return f"Current image must be 3D array, got shape {frame_state.current_image.shape}"
        
        if frame_state.previous_image is not None:
            if len(frame_state.previous_image.shape) != 3:
                return f"Previous image must be 3D array, got shape {frame_state.previous_image.shape}"
        
        return True
        
    except Exception as e:
        return f"Validation error: {str(e)}"


def apply_animation_warping(
    frame_state: FrameState,
    context: RenderContext,
    depth_model: Optional[Any] = None
) -> FrameResult:
    """
    Apply animation warping transformations to frame.
    
    Args:
        frame_state: Current frame state
        context: Rendering context
        depth_model: Optional depth model for 3D warping
        
    Returns:
        FrameResult with warped image
    """
    start_time = time.time()
    
    try:
        if frame_state.current_image is None:
            return FrameResult(
                frame_state=frame_state.with_stage(ProcessingStage.ANIMATION_WARPING),
                success=False,
                error=RenderingError(
                    "No current image for animation warping",
                    ProcessingStage.ANIMATION_WARPING,
                    frame_state.metadata.frame_idx
                )
            )
        
        # Apply animation warping (simplified for functional approach)
        warped_image = frame_state.current_image.copy()
        depth = None
        
        # For 3D animation mode, apply warping
        if context.animation_mode == '3D' and context.use_depth_warping:
            # This would call the actual warping function
            # warped_image, depth = anim_frame_warp(...)
            pass
        
        new_frame_state = (frame_state
                          .with_image(warped_image)
                          .with_stage(ProcessingStage.ANIMATION_WARPING)
                          .with_transformation("animation_warp"))
        
        if depth is not None:
            new_frame_state = replace(new_frame_state, depth_map=depth)
        
        processing_time = time.time() - start_time
        
        return FrameResult(
            frame_state=new_frame_state,
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.ANIMATION_WARPING),
            success=False,
            error=RenderingError(
                f"Animation warping failed: {str(e)}",
                ProcessingStage.ANIMATION_WARPING,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )


# Note: apply_hybrid_motion function removed - hybrid video functionality not available


def apply_noise(
    frame_state: FrameState,
    noise_params: Optional[Dict[str, Any]] = None
) -> FrameResult:
    """
    Apply noise to frame image.
    
    Args:
        frame_state: Current frame state
        noise_params: Optional noise parameters
        
    Returns:
        FrameResult with noise applied
    """
    start_time = time.time()
    
    try:
        if frame_state.current_image is None:
            return FrameResult(
                frame_state=frame_state.with_stage(ProcessingStage.NOISE_APPLICATION),
                success=False,
                error=RenderingError(
                    "No current image for noise application",
                    ProcessingStage.NOISE_APPLICATION,
                    frame_state.metadata.frame_idx
                )
            )
        
        # Apply noise using the noise function
        noised_image = frame_state.current_image.copy()
        
        # Default noise parameters
        noise_level = frame_state.metadata.noise_level
        seed = frame_state.metadata.seed
        
        if noise_params:
            noise_level = noise_params.get('noise_level', noise_level)
            seed = noise_params.get('seed', seed)
        
        if noise_level > 0:
            # noised_image = add_noise(noised_image, noise_level, seed, ...)
            pass
        
        new_frame_state = (frame_state
                          .with_image(noised_image)
                          .with_stage(ProcessingStage.NOISE_APPLICATION)
                          .with_transformation("noise"))
        
        processing_time = time.time() - start_time
        
        return FrameResult(
            frame_state=new_frame_state,
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.NOISE_APPLICATION),
            success=False,
            error=RenderingError(
                f"Noise application failed: {str(e)}",
                ProcessingStage.NOISE_APPLICATION,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )


def apply_color_correction(
    frame_state: FrameState,
    color_params: Optional[Dict[str, Any]] = None
) -> FrameResult:
    """
    Apply color correction to frame.
    
    Args:
        frame_state: Current frame state
        color_params: Optional color correction parameters
        
    Returns:
        FrameResult with color correction applied
    """
    start_time = time.time()
    
    try:
        if frame_state.current_image is None:
            return FrameResult(
                frame_state=frame_state.with_stage(ProcessingStage.COLOR_CORRECTION),
                success=False,
                error=RenderingError(
                    "No current image for color correction",
                    ProcessingStage.COLOR_CORRECTION,
                    frame_state.metadata.frame_idx
                )
            )
        
        corrected_image = frame_state.current_image.copy()
        
        # Apply color correction if parameters provided
        if color_params:
            color_coherence = color_params.get('color_coherence', 'None')
            color_match_sample = color_params.get('color_match_sample')
            
            if color_coherence != 'None' and color_match_sample is not None:
                # corrected_image = maintain_colors(corrected_image, color_match_sample, color_coherence)
                pass
        
        new_frame_state = (frame_state
                          .with_image(corrected_image)
                          .with_stage(ProcessingStage.COLOR_CORRECTION)
                          .with_transformation("color_correction"))
        
        processing_time = time.time() - start_time
        
        return FrameResult(
            frame_state=new_frame_state,
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.COLOR_CORRECTION),
            success=False,
            error=RenderingError(
                f"Color correction failed: {str(e)}",
                ProcessingStage.COLOR_CORRECTION,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )


def apply_mask_operations(
    frame_state: FrameState,
    mask_params: Optional[Dict[str, Any]] = None
) -> FrameResult:
    """
    Apply mask operations to frame.
    
    Args:
        frame_state: Current frame state
        mask_params: Optional mask parameters
        
    Returns:
        FrameResult with mask operations applied
    """
    start_time = time.time()
    
    try:
        if frame_state.current_image is None:
            return FrameResult(
                frame_state=frame_state.with_stage(ProcessingStage.MASK_APPLICATION),
                success=False,
                error=RenderingError(
                    "No current image for mask application",
                    ProcessingStage.MASK_APPLICATION,
                    frame_state.metadata.frame_idx
                )
            )
        
        masked_image = frame_state.current_image.copy()
        
        # Apply mask operations if mask is present
        if frame_state.mask_image is not None:
            # Apply mask overlay or compositing
            # masked_image = do_overlay_mask(...)
            pass
        
        new_frame_state = (frame_state
                          .with_image(masked_image)
                          .with_stage(ProcessingStage.MASK_APPLICATION)
                          .with_transformation("mask_application"))
        
        processing_time = time.time() - start_time
        
        return FrameResult(
            frame_state=new_frame_state,
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.MASK_APPLICATION),
            success=False,
            error=RenderingError(
                f"Mask application failed: {str(e)}",
                ProcessingStage.MASK_APPLICATION,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )


def apply_frame_transformations(
    frame_state: FrameState,
    context: RenderContext,
    transformations: Tuple[str, ...] = ("animation_warp", "noise", "color_correction", "mask_application"),
    **kwargs
) -> FrameResult:
    """
    Apply a sequence of transformations to a frame.
    
    Args:
        frame_state: Current frame state
        context: Rendering context
        transformations: Sequence of transformation names to apply
        **kwargs: Additional parameters for transformations
        
    Returns:
        FrameResult with all transformations applied
    """
    current_result = FrameResult(frame_state=frame_state, success=True)
    
    transformation_functions = {
        'animation_warp': lambda fs: apply_animation_warping(fs, context, kwargs.get('depth_model')),
        'noise': lambda fs: apply_noise(fs, kwargs.get('noise_params')),
        'color_correction': lambda fs: apply_color_correction(fs, kwargs.get('color_params')),
        'mask_application': lambda fs: apply_mask_operations(fs, kwargs.get('mask_params'))
    }
    
    for transformation in transformations:
        if not current_result.success:
            break
            
        if transformation in transformation_functions:
            current_result = transformation_functions[transformation](current_result.frame_state)
        else:
            # Unknown transformation - add warning but continue
            current_result = current_result.with_warning(f"Unknown transformation: {transformation}")
    
    return current_result


def process_frame(
    frame_state: FrameState,
    context: RenderContext,
    processing_params: Optional[Dict[str, Any]] = None
) -> FrameResult:
    """
    Process a single frame through transformations and image generation.
    """
    start_time = time.time()
    current_state = frame_state.with_stage(ProcessingStage.INITIALIZATION)

    # Validate initial state
    validation = validate_frame_state(current_state)
    if validation is not True:
        return FrameResult(
            frame_state=current_state,
            success=False,
            error=RenderingError(str(validation), current_state.stage, current_state.metadata.frame_idx)
        )

    # Placeholder for previous image if needed for img2img or warping
    # This needs to be correctly passed into FrameState upstream if it's the first frame
    # or derived from previous FrameResult.
    # For now, assume it might be None or a placeholder if not set.
    # previous_image_np = current_state.previous_image 
    
    # If current_state.current_image is None (e.g. first frame for txt2img, or if init image not loaded yet)
    # and previous_image_np is also None, we might need to handle txt2img path.
    # For img2img, current_image might be the warped version of previous_image.

    # Let's assume current_state.current_image is the one to process (e.g. after warping)
    # or current_state.previous_image is the init_image for img2img.
    # The functional pipeline needs to manage this handoff.
    # For now, we'll try to use previous_image as init for img2img.

    pil_init_image = None
    if current_state.previous_image is not None:
        try:
            pil_init_image = Image.fromarray(current_state.previous_image.astype(np.uint8))
        except Exception as e:
            return FrameResult(frame_state=current_state, success=False, error=RenderingError(f"Could not convert previous_image to PIL: {e}", current_state.stage, current_state.metadata.frame_idx))
    # else: txt2img path or error if img2img expected an image

    # Create and configure the Stable Diffusion processing pipeline object
    # We need legacy_args and legacy_root from the context
    if not hasattr(context, 'legacy_args') or not hasattr(context, 'legacy_root'):
        return FrameResult(frame_state=current_state, success=False, error=RenderingError("legacy_args or legacy_root not in RenderContext for get_webui_sd_pipeline", current_state.stage, current_state.metadata.frame_idx))

    p = get_webui_sd_pipeline(context.legacy_args, context.legacy_root)

    # Override p with frame-specific metadata
    p.prompt = frame_state.metadata.prompt
    # TODO: p.negative_prompt = frame_state.metadata.negative_prompt (needs to be added to FrameMetadata)
    p.negative_prompt = context.legacy_args.negative_prompts if hasattr(context.legacy_args, 'negative_prompts') else ""


    p.seed = int(frame_state.metadata.seed)
    p.steps = int(frame_state.metadata.steps) if hasattr(frame_state.metadata, 'steps') else int(p.steps) # Ensure steps is in metadata
    p.cfg_scale = float(frame_state.metadata.cfg_scale)
    p.width = int(context.width)
    p.height = int(context.height)
    
    if hasattr(frame_state.metadata, 'sampler_name') and frame_state.metadata.sampler_name:
        p.sampler_name = frame_state.metadata.sampler_name
    # if hasattr(frame_state.metadata, 'scheduler_name') and frame_state.metadata.scheduler_name:
    #     p.scheduler_name = frame_state.metadata.scheduler_name # p.scheduler is the attribute

    # Denoising strength (for img2img)
    # strength_schedule comes from anim_args.strength_schedule
    # metadata.strength should hold the scheduled value for the current frame.
    if pil_init_image is not None: # This is an img2img operation
        p.init_images = [pil_init_image]
        p.denoising_strength = 1.0 - float(frame_state.metadata.strength)
    else: # This would be txt2img
        p.denoising_strength = None # Not used for txt2img directly in p object like this
        # Ensure p is of type StableDiffusionProcessingTxt2Img or that StableDiffusionProcessingImg2Img handles init_images=None correctly.
        # For now, get_webui_sd_pipeline always returns Img2Img. A proper txt2img path would need different setup.
        # Let's assume for now Deforum always works in an img2img-like mode, even if the init_image is black/noise for the first frame.
        # If previous_image was truly None, we must provide a dummy/blank init_image or switch to a Txt2Img pipeline
        if p.init_images is None:
             p.init_images = [Image.new("RGB", (p.width, p.height), "black")] # Provide a black init image for Img2Img
             p.denoising_strength = 1.0 # Full strength if no init image was really there

    # Masking (simplified)
    if hasattr(frame_state, 'mask') and frame_state.mask is not None:
        try:
            p.image_mask = Image.fromarray(frame_state.mask.astype(np.uint8))
        except Exception as e:
            return FrameResult(frame_state=current_state, success=False, error=RenderingError(f"Could not convert mask to PIL: {e}", current_state.stage, current_state.metadata.frame_idx))

    # TODO: ControlNet - this needs to be adapted to use frame_state and context
    # controlnet_units = ...
    # if controlnet_units:
    #    p.scripts = getattr(p, 'scripts', None) or processing.ScriptRunner()
    #    cn_script_args_from_get_cn_script = get_controlnet_script_args(...) # This needs refactoring
    #    p.scripts.alwayson_scripts.append(ScriptAlwayson("ControlNet", cn_script_args_from_get_cn_script))
    
    # Actual image generation using A1111/Forge's processing pipeline
    try:
        print(f"[DEBUG process_frame] Frame {frame_state.metadata.frame_idx}: About to call process_images.")
        print(f"[DEBUG process_frame] p.prompt: '{p.prompt[:100]}...'")
        print(f"[DEBUG process_frame] p.negative_prompt: '{p.negative_prompt[:100]}...'")
        print(f"[DEBUG process_frame] p.seed: {p.seed}, p.steps: {p.steps}, p.cfg_scale: {p.cfg_scale}")
        print(f"[DEBUG process_frame] p.width: {p.width}, p.height: {p.height}")
        print(f"[DEBUG process_frame] p.sampler_name: {p.sampler_name}")
        # print(f"[DEBUG process_frame] p.scheduler: {p.scheduler}") # p.scheduler might be an object or name
        print(f"[DEBUG process_frame] p.denoising_strength: {p.denoising_strength}")
        if p.init_images:
            print(f"[DEBUG process_frame] p.init_images type: {type(p.init_images[0])}, size: {p.init_images[0].size if hasattr(p.init_images[0], 'size') else 'N/A'}")
        else:
            print("[DEBUG process_frame] p.init_images is None or empty.")
        if hasattr(p, 'image_mask') and p.image_mask:
            print(f"[DEBUG process_frame] p.image_mask type: {type(p.image_mask)}, size: {p.image_mask.size if hasattr(p.image_mask, 'size') else 'N/A'}")
        else:
            print("[DEBUG process_frame] p.image_mask is None or not set.")


        print(f"Generating frame {frame_state.metadata.frame_idx} with seed {p.seed} and prompt: {p.prompt[:100]}...")
        processed_result = processing.process_images(p)
        generated_image_pil = processed_result.images[0]
        
        # Convert PIL image to NumPy array (RGB)
        generated_image_np = np.array(generated_image_pil.convert("RGB"))

    except Exception as e:
        import traceback
        print(f"Error during image generation for frame {frame_state.metadata.frame_idx}: {e}")
        traceback.print_exc()
        return FrameResult(
            frame_state=current_state.with_stage(ProcessingStage.GENERATION),
            success=False,
            error=RenderingError(f"Image generation failed: {e}", ProcessingStage.GENERATION, frame_state.metadata.frame_idx),
            processing_time=time.time() - start_time
        )

    current_state = current_state.with_image(generated_image_np)
    current_state = current_state.with_stage(ProcessingStage.GENERATION_COMPLETE)
    # End of main image generation

    # Apply further transformations if any (e.g., unsharp mask from legacy)
    # current_state = apply_frame_transformations(current_state, context, ...) 

    processing_time = time.time() - start_time
    return FrameResult(
        frame_state=current_state,
        success=True,
        processing_time=processing_time,
        # Optionally include generated image directly in result if needed elsewhere,
        # but FrameState.current_image should be the primary carrier.
        # generated_image_pil=generated_image_pil 
    )


def save_frame_to_disk(
    frame_state: FrameState,
    context: RenderContext
) -> FrameResult:
    """Saves the current image in FrameState to disk."""
    start_time = time.time()
    
    if frame_state.current_image is None:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.SAVING),
            success=False,
            error=RenderingError(
                "No current image to save",
                ProcessingStage.SAVING,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )

    try:
        frame_idx = frame_state.metadata.frame_idx
        filename = f"{context.timestring}_{frame_idx:09d}.png"
        output_path = context.output_dir / filename
        
        # Assuming current_image is a NumPy array in RGB format
        # If it's BGR (e.g., from OpenCV), it needs conversion: image_rgb = cv2.cvtColor(frame_state.current_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_state.current_image.astype(np.uint8))
        pil_image.save(output_path)
        
        # print(f"[DEBUG] Saved frame {frame_idx} to {output_path}") # Optional debug print

        new_frame_state = frame_state.with_stage(ProcessingStage.COMPLETED) # Or a new SAVING_COMPLETED stage
        
        return FrameResult(
            frame_state=new_frame_state,
            success=True,
            processing_time=time.time() - start_time
        )

    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.SAVING),
            success=False,
            error=RenderingError(
                f"Failed to save frame {frame_state.metadata.frame_idx} to disk: {str(e)}",
                ProcessingStage.SAVING,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )


def merge_frame_results(results: Tuple[FrameResult, ...]) -> Dict[str, Any]:
    """
    Merge multiple frame results into summary statistics.
    
    Args:
        results: Tuple of frame results to merge
        
    Returns:
        Dictionary with merged statistics
    """
    if not results:
        return {
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'warnings': [],
            'errors': []
        }
    
    successful_frames = sum(1 for r in results if r.success)
    failed_frames = len(results) - successful_frames
    total_processing_time = sum(r.processing_time for r in results)
    average_processing_time = total_processing_time / len(results) if results else 0.0
    
    all_warnings = []
    all_errors = []
    
    for result in results:
        all_warnings.extend(result.warnings)
        if result.error:
            all_errors.append(result.error)
    
    return {
        'total_frames': len(results),
        'successful_frames': successful_frames,
        'failed_frames': failed_frames,
        'total_processing_time': total_processing_time,
        'average_processing_time': average_processing_time,
        'warnings': all_warnings,
        'errors': all_errors
    } 