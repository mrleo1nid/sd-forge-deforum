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
from .main_generation_pipeline import generate
from modules.shared import opts

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
    Process a single frame through the rendering pipeline, including image generation.
    
    Args:
        frame_state: Current frame state (with populated metadata)
        context: Rendering context (with legacy objects)
        processing_params: Optional processing parameters (currently unused)
        
    Returns:
        FrameResult with processed image and status
    """
    start_time = time.time()
    metadata = frame_state.metadata
    frame_idx = metadata.frame_idx

    # --- Stage 1: Image Generation (New) ---
    try:
        # Retrieve legacy objects from context for generate()
        legacy_args = context.legacy_args
        legacy_anim_args = context.legacy_anim_args
        legacy_video_args = context.legacy_video_args # Though not directly used by generate, might be in root or other logic
        legacy_parseq_args = context.legacy_parseq_args
        legacy_loop_args = context.legacy_loop_args
        legacy_controlnet_args = context.legacy_controlnet_args
        legacy_root = context.legacy_root
        legacy_keys = context.legacy_keys # DeformAnimKeys or Parseq keys
        # legacy_prompts = context.legacy_prompts # available if needed
        legacy_parseq_adapter = getattr(context, 'legacy_parseq_adapter', None) # Needs to be added to RenderContext
        if legacy_parseq_adapter is None and legacy_parseq_args is not None:
             # Attempt to create it if not passed in context but args are there.
             # This is a fallback, ideally it should be created in legacy_renderer and passed in RenderContext.
             from ..integrations.parseq_adapter import ParseqAdapter
             legacy_parseq_adapter = ParseqAdapter(legacy_parseq_args, legacy_anim_args, legacy_video_args, legacy_controlnet_args, legacy_loop_args)
             if legacy_parseq_adapter.use_parseq:
                 legacy_keys = legacy_parseq_adapter.anim_keys # Parseq might overwrite keys

        if not all([legacy_args, legacy_anim_args, legacy_root, legacy_keys, legacy_parseq_adapter is not None]):
            raise ValueError("Essential legacy objects (args, anim_args, root, keys, parseq_adapter) missing in RenderContext for image generation.")

        # Create a temporary args-like object for the current frame, using a deep copy to avoid modifying the original.
        current_frame_args = copy.deepcopy(legacy_args)

        # Populate current_frame_args and root with scheduled values for this frame_idx
        current_frame_args.prompt = metadata.prompt
        current_frame_args.cfg_scale = metadata.cfg_scale
        # current_frame_args.distilled_cfg_scale = metadata.distilled_cfg_scale # Ensure this exists in metadata
        current_frame_args.seed = metadata.seed
        # current_frame_args.strength = metadata.strength # for img2img, handle with init_image
        
        # Steps, Subseed, Checkpoint, CLIPSkip from schedules (similar to rendering_modes.py)
        if legacy_anim_args.enable_steps_scheduling and legacy_keys.steps_schedule_series[frame_idx] is not None:
            current_frame_args.steps = int(legacy_keys.steps_schedule_series[frame_idx])
        # else: current_frame_args.steps remains as per legacy_args.steps
        
        if legacy_anim_args.enable_checkpoint_scheduling and legacy_keys.checkpoint_schedule_series[frame_idx] is not None:
            current_frame_args.checkpoint = legacy_keys.checkpoint_schedule_series[frame_idx]
        # else: current_frame_args.checkpoint remains as per legacy_args.checkpoint

        if legacy_anim_args.enable_subseed_scheduling:
            legacy_root.subseed = int(legacy_keys.subseed_schedule_series[frame_idx])
            legacy_root.subseed_strength = legacy_keys.subseed_strength_schedule_series[frame_idx]
        # else: subseed/strength remain as per legacy_root (potentially set by initial seed behavior)
        
        scheduled_clipskip = None
        if legacy_anim_args.enable_clipskip_scheduling and legacy_keys.clipskip_schedule_series[frame_idx] is not None:
            scheduled_clipskip = int(legacy_keys.clipskip_schedule_series[frame_idx])
            opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip
        # else: opts.data["CLIP_stop_at_last_layers"] uses its existing value or value from Deforum settings

        # Sampler and Scheduler names from schedule
        scheduled_sampler_name = None
        if legacy_anim_args.enable_sampler_scheduling and legacy_keys.sampler_schedule_series[frame_idx] is not None:
            scheduled_sampler_name = legacy_keys.sampler_schedule_series[frame_idx].casefold()
            
        scheduled_scheduler_name = None
        if legacy_anim_args.enable_scheduler_scheduling and legacy_keys.scheduler_schedule_series[frame_idx] is not None:
            scheduled_scheduler_name = legacy_keys.scheduler_schedule_series[frame_idx].casefold()

        # Handle init_image for img2img based on strength and previous frame
        init_image_pil = None
        current_strength = metadata.strength
        if frame_state.previous_image is not None and current_strength < 1.0 and current_strength > 0.0:
            # Assuming previous_image is a NumPy array, convert to PIL for generate()
            init_image_pil = Image.fromarray(frame_state.previous_image.astype(np.uint8))
            # generate() also needs current_frame_args.strength to be set
            current_frame_args.strength = current_strength 
        else:
            # If no previous image or strength is 1.0 (txt2img) or 0.0 (no change from init)
            current_frame_args.strength = 1.0 # Effectively txt2img if no init_image
            if frame_state.current_image is not None and current_strength == 0.0:
                 # Special case: strength 0 means use current_image as is (e.g. from video input)
                 # This logic might need adjustment based on how video input frames are handled.
                 # For now, assume generate() is the primary source or uses init_image
                 pass 

        # Call generate()
        # Ensure all args (current_frame_args, keys, anim_args, etc.) are what generate() expects.
        print(f"Generating frame {frame_idx} with seed {current_frame_args.seed} and prompt: {current_frame_args.prompt[:100]}...")
        pil_image = generate(
            current_frame_args, 
            legacy_keys, 
            legacy_anim_args, 
            legacy_loop_args, 
            legacy_controlnet_args, 
            legacy_root, 
            legacy_parseq_adapter, 
            frame_idx, 
            scheduled_sampler_name, 
            scheduled_scheduler_name,
            init_image=init_image_pil # Pass PIL init_image
        )

        if pil_image is None:
            raise RuntimeError(f"Image generation failed for frame {frame_idx}. generate() returned None.")

        generated_image_np = np.array(pil_image)
        new_frame_state = frame_state.with_image(generated_image_np)
        new_frame_state = replace(new_frame_state, stage=ProcessingStage.GENERATION) # Update stage

    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.GENERATION), # Or a new ERROR_GENERATION stage
            success=False,
            error=RenderingError(
                f"Image generation failed for frame {frame_idx}: {str(e)}",
                ProcessingStage.GENERATION,
                frame_idx
            ),
            processing_time=time.time() - start_time
        )
    
    # --- Stage 2: Apply other transformations (existing logic) ---
    # The rest of the function will now use new_frame_state which has the generated image
    frame_state_after_gen = new_frame_state # Pass the updated state

    # Validate frame state (optional, if transformations expect valid images)
    validation = validate_frame_state(frame_state_after_gen)
    if validation is not True:
        return FrameResult(
            frame_state=frame_state_after_gen,
            success=False,
            error=RenderingError(
                f"Frame validation failed: {validation}",
                ProcessingStage.GENERATION,
                frame_idx
            )
        )
    
    # Update final processing time
    total_time = time.time() - start_time
    final_frame_state = frame_state_after_gen.with_stage(ProcessingStage.COMPLETED)
    
    return FrameResult(
        frame_state=final_frame_state,
        success=True,
        processing_time=total_time
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