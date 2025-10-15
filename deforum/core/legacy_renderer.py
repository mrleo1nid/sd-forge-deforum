"""
Legacy Rendering Adapter

This module provides backward compatibility with the existing render.py interface
while using the new functional rendering system under the hood. It converts
legacy arguments to the new functional format and provides drop-in replacements.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from types import SimpleNamespace

from .frame_models import (
    RenderContext, FrameMetadata, FrameState, ModelState,
    ProcessingStage, RenderingError
)
from .rendering_pipeline import (
    create_rendering_pipeline, render_animation_functional,
    create_progress_tracker, PipelineConfig
)
from .frame_processing import create_frame_state
from .keyframe_animation import DeformAnimKeys
from ..integrations.parseq_adapter import ParseqAdapter

# Global flag to enable/disable functional rendering
# NOTE: Disabled because functional rendering doesn't save frames yet
_FUNCTIONAL_RENDERING_ENABLED = False


def enable_functional_rendering(enabled: bool = True) -> None:
    """
    Enable or disable functional rendering system.
    
    Args:
        enabled: Whether to enable functional rendering
    """
    global _FUNCTIONAL_RENDERING_ENABLED
    _FUNCTIONAL_RENDERING_ENABLED = enabled
    print(f"Functional rendering {'enabled' if enabled else 'disabled'}")


def is_functional_rendering_enabled() -> bool:
    """Check if functional rendering is enabled."""
    return _FUNCTIONAL_RENDERING_ENABLED


def convert_legacy_args_to_context(
    args: Any,
    anim_args: Any,
    video_args: Any,
    parseq_args: Any,
    loop_args: Any,
    controlnet_args: Any,
    root: Any,
    keys: Any,
    animation_prompts: Dict[str, str],
    parseq_adapter: Optional[Any] = None
) -> RenderContext:
    """
    Convert legacy argument objects to functional RenderContext.
    
    Args:
        args: Legacy args object
        anim_args: Legacy animation args
        video_args: Legacy video args
        parseq_args: Legacy parseq_args object
        loop_args: Legacy loop_args object
        controlnet_args: Legacy controlnet_args object
        root: Legacy root object
        keys: DeformAnimKeys object (or from Parseq)
        animation_prompts: Dictionary of prompts
        parseq_adapter: Optional[Any] = None
        
    Returns:
        RenderContext for functional rendering
    """
    # Extract values with fallbacks for missing attributes
    output_dir = Path(getattr(args, 'outdir', '/tmp/deforum_output'))
    timestring = getattr(root, 'timestring', 'default')
    width = getattr(args, 'W', 512)
    height = getattr(args, 'H', 512)
    max_frames = getattr(anim_args, 'max_frames', 100)
    fps = getattr(video_args, 'fps', 30.0) if hasattr(video_args, 'fps') else 30.0
    
    # Animation configuration
    animation_mode = getattr(anim_args, 'animation_mode', '2D')
    use_depth_warping = getattr(anim_args, 'use_depth_warping', False)
    save_depth_maps = getattr(anim_args, 'save_depth_maps', False)
    
    # Model configuration
    depth_algorithm = getattr(anim_args, 'depth_algorithm', 'midas')
    optical_flow_cadence = getattr(anim_args, 'optical_flow_cadence', 'None')
    diffusion_cadence = getattr(anim_args, 'diffusion_cadence', 1)
    
    # Quality settings
    motion_preview_mode = getattr(args, 'motion_preview_mode', False)
    
    # Device configuration
    device = getattr(root, 'device', 'cuda')
    half_precision = getattr(root, 'half_precision', True)
    
    return RenderContext(
        output_dir=output_dir,
        timestring=timestring,
        width=width,
        height=height,
        max_frames=max_frames,
        fps=fps,
        animation_mode=str(animation_mode),
        use_depth_warping=use_depth_warping,
        save_depth_maps=save_depth_maps,
        depth_algorithm=depth_algorithm,
        optical_flow_cadence=optical_flow_cadence,
        diffusion_cadence=diffusion_cadence,
        motion_preview_mode=motion_preview_mode,
        device=device,
        half_precision=half_precision,
        legacy_args=args,
        legacy_anim_args=anim_args,
        legacy_video_args=video_args,
        legacy_parseq_args=parseq_args,
        legacy_loop_args=loop_args,
        legacy_controlnet_args=controlnet_args,
        legacy_root=root,
        legacy_keys=keys,
        legacy_prompts=animation_prompts,
        legacy_parseq_adapter=parseq_adapter
    )


def functional_render_animation(
    args: Any,
    anim_args: Any,
    video_args: Any,
    parseq_args: Any,
    loop_args: Any,
    controlnet_args: Any,
    root: Any
) -> None:
    """
    Functional replacement for the legacy render_animation function.
    
    This function provides the same interface as the original render_animation
    but uses the new functional rendering system internally.
    
    Args:
        args: Legacy args object
        anim_args: Legacy animation args
        video_args: Legacy video args
        parseq_args: Legacy parseq args
        loop_args: Legacy loop args
        controlnet_args: Legacy controlnet args
        root: Legacy root object
    """
    if not _FUNCTIONAL_RENDERING_ENABLED:
        print("Functional rendering disabled, using stable render core...")
        from .rendering_engine import render_animation as stable_render_animation
        return stable_render_animation(
            args, anim_args, video_args, parseq_args, loop_args,
            controlnet_args, root
        )

    print("Using functional rendering system...")
    
    # Instantiate keys and parseq_adapter first
    # Ensure args.seed is an int if it comes from a UI component that might be float
    current_seed = int(args.seed) if args.seed is not None else -1
    keys = DeformAnimKeys(anim_args, current_seed) 
    parseq_adapter = ParseqAdapter(parseq_args, anim_args, video_args, controlnet_args, loop_args)
    if parseq_adapter.use_parseq:
        keys = parseq_adapter.anim_keys # Overwrite keys if Parseq is used and provides its own

    try:
        # Convert legacy arguments to functional format, now including keys and prompts
        context = convert_legacy_args_to_context(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root, keys, root.animation_prompts, parseq_adapter)
        
        # Create pipeline configuration
        config = PipelineConfig(
            max_workers=1,  # Start with sequential processing
            enable_progress_tracking=True,
            enable_error_recovery=True
        )
        
        # Create rendering pipeline
        pipeline = create_rendering_pipeline(context, config=config)
        
        # Create progress tracker
        progress_callback = create_progress_tracker(verbose=True)
        
        # Execute functional rendering
        session = render_animation_functional(
            context=context,
            pipeline=pipeline,
            start_frame=0,
            progress_callback=progress_callback
        )
        
        # Report results
        print(f"\nFunctional rendering completed:")
        print(f"  Total frames: {session.context.max_frames}")
        print(f"  Successful frames: {session.completed_frames}")
        print(f"  Failed frames: {session.failed_frames}")
        print(f"  Total processing time: {session.total_processing_time:.2f}s")
        
        if session.failed_frames > 0:
            print(f"  Warning: {session.failed_frames} frames failed to render")
            
            # Show first few errors
            errors = [r.error for r in session.frame_results if r.error][:3]
            for error in errors:
                print(f"    Frame {error.frame_idx}: {error.message}")
        
    except Exception as e:
        import traceback
        print(f"Functional rendering failed. Original error:")
        traceback.print_exc() # Print the full traceback of e
        print(f"Cannot fall back to legacy ..render module as it does not exist.")
        # from ..render import render_animation as legacy_render_animation # Commented out
        # return legacy_render_animation( # Commented out
        #     args, anim_args, video_args, parseq_args, loop_args,
        #     controlnet_args, # freeu_args, # Removed
        #     # kohya_hrfix_args, # Removed
        #     root
        # )
        raise e # Re-raise the original exception


def create_legacy_compatible_pipeline():
    """
    Create a pipeline that's compatible with legacy rendering expectations.
    
    Returns:
        RenderingPipeline configured for legacy compatibility
    """
    # This would create a pipeline that mimics the exact behavior
    # of the legacy rendering system
    pass


def migrate_legacy_settings(legacy_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate legacy settings to functional rendering format.
    
    Args:
        legacy_settings: Legacy settings dictionary
        
    Returns:
        Migrated settings for functional rendering
    """
    functional_settings = {}
    
    # Map legacy setting names to functional equivalents
    setting_mappings = {
        'animation_mode': 'animation_mode',
        'max_frames': 'max_frames',
        'strength_schedule': 'strength_schedule',
        'cfg_scale_schedule': 'cfg_scale_schedule',
        'seed_schedule': 'seed_schedule',
        'noise_schedule': 'noise_schedule',
        'use_depth_warping': 'use_depth_warping',
        'save_depth_maps': 'save_depth_maps',
    }
    
    for legacy_key, functional_key in setting_mappings.items():
        if legacy_key in legacy_settings:
            functional_settings[functional_key] = legacy_settings[legacy_key]
    
    return functional_settings


def validate_legacy_compatibility(args: Any, anim_args: Any) -> Tuple[bool, str]:
    """
    Validate that legacy arguments are compatible with functional rendering.
    
    Args:
        args: Legacy args object
        anim_args: Legacy animation args
        
    Returns:
        Tuple of (is_compatible, error_message)
    """
    try:
        # Check for required attributes
        required_attrs = [
            ('args', args, ['outdir', 'W', 'H']),
            ('anim_args', anim_args, ['animation_mode', 'max_frames'])
        ]
        
        for obj_name, obj, attrs in required_attrs:
            for attr in attrs:
                if not hasattr(obj, attr):
                    return False, f"Missing required attribute: {obj_name}.{attr}"
        
        # Check for unsupported features
        if hasattr(anim_args, 'animation_mode'):
            if anim_args.animation_mode not in ['2D', '3D', 'Video Input']:
                return False, f"Unsupported animation mode: {anim_args.animation_mode}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def create_hybrid_renderer(use_functional_for_frames: Optional[set] = None):
    """
    Create a hybrid renderer that uses functional rendering for specific frames
    and legacy rendering for others.
    
    Note: Hybrid rendering functionality removed
    """
    def hybrid_render(args, anim_args, video_args, parseq_args, loop_args,
                     controlnet_args, freeu_args, kohya_hrfix_args, root):
        # Note: Hybrid rendering functionality removed
        # Fall back to legacy rendering
        return render_animation_legacy(args, anim_args, video_args, parseq_args, 
                                     loop_args, controlnet_args, freeu_args, 
                                     kohya_hrfix_args, root)
    
    return hybrid_render 