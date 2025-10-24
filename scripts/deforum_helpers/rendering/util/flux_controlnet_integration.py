"""Integration layer for Flux ControlNet with Deforum keyframe generation.

Provides utilities to generate keyframes using Flux ControlNet based on:
- Canny edges from previous frame
- Depth maps from Depth-Anything V2
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional

from ...flux_controlnet import FluxControlNetManager
from ...flux_controlnet_preprocessors import preprocess_image_for_controlnet
from ..data.render_data import RenderData
from ..data.frame.diffusion_frame import DiffusionFrame


def is_flux_controlnet_enabled(data: RenderData) -> bool:
    """Check if Flux ControlNet is enabled in animation args.

    Args:
        data: Rendering data containing animation arguments

    Returns:
        True if Flux ControlNet is enabled
    """
    anim_args = data.args.anim_args
    return getattr(anim_args, 'enable_flux_controlnet', False)


def should_use_flux_controlnet_for_frame(data: RenderData, frame: DiffusionFrame) -> bool:
    """Determine if Flux ControlNet should be used for this specific frame.

    ControlNet is only applied to keyframes (not tweens) and requires:
    - Flux ControlNet enabled in settings
    - Previous frame exists (for Canny control)
    - Not the first frame

    Args:
        data: Rendering data
        frame: Current frame being processed

    Returns:
        True if Flux ControlNet should be used
    """
    if not is_flux_controlnet_enabled(data):
        return False

    # Only apply to keyframes, not tweens
    if not frame.is_keyframe:
        return False

    # Need previous frame for control image (except first frame)
    if frame.i == 0:
        return False

    if not data.images.has_previous():
        return False

    return True


def get_control_image_for_frame(
    data: RenderData,
    frame: DiffusionFrame,
    control_type: str
) -> Optional[np.ndarray]:
    """Get control image for Flux ControlNet.

    Args:
        data: Rendering data
        frame: Current frame
        control_type: "canny" or "depth"

    Returns:
        Control image as numpy array (RGB), or None if not available
    """
    if control_type == "canny":
        # Use previous frame for Canny edge detection
        if data.images.has_previous():
            # previous is in BGR format from OpenCV
            return data.images.previous
        return None

    elif control_type == "depth":
        # Use depth map from Depth-Anything V2 (already computed in 3D mode)
        if hasattr(frame, 'depth') and frame.depth is not None:
            return frame.depth

        # If depth model is available, compute it from previous frame
        if data.depth_model is not None and data.images.has_previous():
            # Predict depth from previous frame
            depth = data.depth_model.predict(data.images.previous, data.args.anim_args)
            return depth

        return None

    else:
        raise ValueError(f"Unsupported control type: {control_type}")


def generate_with_flux_controlnet(
    data: RenderData,
    frame: DiffusionFrame
) -> Image.Image:
    """Generate keyframe using Flux ControlNet.

    Args:
        data: Rendering data containing settings and state
        frame: Current keyframe to generate

    Returns:
        Generated PIL Image
    """
    anim_args = data.args.anim_args
    args = data.args.args

    # Get ControlNet settings from animation args
    control_type = getattr(anim_args, 'flux_controlnet_type', 'canny')
    model_name = getattr(anim_args, 'flux_controlnet_model', 'instantx')
    strength = getattr(anim_args, 'flux_controlnet_strength', 0.7)
    guidance_scale = getattr(anim_args, 'flux_guidance_scale', 3.5)
    num_steps = getattr(args, 'steps', 28)

    # Canny-specific settings
    canny_low = getattr(anim_args, 'flux_controlnet_canny_low', 100)
    canny_high = getattr(anim_args, 'flux_controlnet_canny_high', 200)

    # Get control image
    control_image = get_control_image_for_frame(data, frame, control_type)
    if control_image is None:
        raise ValueError(f"Could not get {control_type} control image for frame {frame.i}")

    # Initialize ControlNet manager (lazy loads models)
    manager = FluxControlNetManager(
        control_type=control_type,
        model_name=model_name,
        base_model=getattr(anim_args, 'flux_base_model', 'black-forest-labs/FLUX.1-dev'),
        device='cuda'
    )

    # Get prompt for this frame
    prompt = args.prompt
    negative_prompt = getattr(args, 'negative_prompt', '')

    # Generate image
    result = manager.generate(
        prompt=prompt,
        control_image=control_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=strength,
        width=args.W,
        height=args.H,
        seed=frame.seed,
        preprocess_control=True,
        canny_low=canny_low,
        canny_high=canny_high
    )

    return result
