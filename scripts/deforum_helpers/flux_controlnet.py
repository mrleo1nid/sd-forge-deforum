"""Flux ControlNet integration for Deforum.

Main interface for using Flux ControlNet with Deforum keyframe generation.
Integrates preprocessors, model loading, and generation into a simple API.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Union, Dict, Any

from .flux_controlnet_models import (
    load_flux_controlnet_model,
    load_flux_controlnet_pipeline,
    get_available_models,
    get_model_info
)
from .flux_controlnet_preprocessors import (
    preprocess_image_for_controlnet,
    numpy_to_pil
)


class FluxControlNetManager:
    """Manager for Flux ControlNet operations in Deforum."""

    def __init__(
        self,
        control_type: str = "canny",
        model_name: str = "instantx",
        base_model: str = "black-forest-labs/FLUX.1-dev",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ):
        """Initialize Flux ControlNet manager.

        Args:
            control_type: "canny" or "depth"
            model_name: Model provider ("instantx", "xlabs", "bfl", "shakker")
            base_model: Base Flux model repo ID
            torch_dtype: Torch data type for weights
            device: Device to load models on
        """
        self.control_type = control_type
        self.model_name = model_name
        self.base_model = base_model
        self.torch_dtype = torch_dtype
        self.device = device

        self.controlnet = None
        self.pipeline = None
        self.is_loaded = False

        print(f"Initialized Flux {control_type.title()} ControlNet manager")
        print(f"Model: {get_model_info(control_type, model_name)}")

    def load_models(self):
        """Load ControlNet model and pipeline."""
        if self.is_loaded:
            print("Models already loaded")
            return

        print("Loading Flux ControlNet models...")

        # Load ControlNet model
        self.controlnet = load_flux_controlnet_model(
            control_type=self.control_type,
            model_name=self.model_name,
            torch_dtype=self.torch_dtype,
            device=self.device
        )

        # Load pipeline
        self.pipeline = load_flux_controlnet_pipeline(
            base_model=self.base_model,
            controlnet=self.controlnet,
            torch_dtype=self.torch_dtype,
            device=self.device
        )

        self.is_loaded = True
        print("✓ Flux ControlNet ready for generation")

    def generate(
        self,
        prompt: str,
        control_image: Union[np.ndarray, Image.Image],
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        controlnet_conditioning_scale: float = 0.7,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        preprocess_control: bool = True,
        canny_low: int = 100,
        canny_high: int = 200,
        **kwargs
    ) -> Image.Image:
        """Generate image with Flux ControlNet.

        Args:
            prompt: Text prompt
            control_image: Control image (numpy array or PIL Image)
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale (3.5 recommended for Flux)
            controlnet_conditioning_scale: ControlNet influence strength (0.0-1.0)
            control_guidance_start: When to start ControlNet influence (0.0-1.0)
            control_guidance_end: When to end ControlNet influence (0.0-1.0)
            width: Output image width
            height: Output image height
            seed: Random seed (optional)
            preprocess_control: Whether to preprocess control image
            canny_low: Canny low threshold (if control_type is "canny")
            canny_high: Canny high threshold (if control_type is "canny")
            **kwargs: Additional arguments for pipeline

        Returns:
            Generated PIL Image
        """
        if not self.is_loaded:
            self.load_models()

        # Preprocess control image if needed
        if preprocess_control:
            if isinstance(control_image, Image.Image):
                control_image = np.array(control_image)

            control_image = preprocess_image_for_controlnet(
                control_image,
                self.control_type,
                canny_low=canny_low,
                canny_high=canny_high
            )

        # Convert to PIL if numpy
        if isinstance(control_image, np.ndarray):
            control_image = numpy_to_pil(control_image)

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate image
        print(f"Generating with Flux {self.control_type.title()} ControlNet...")
        print(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, "
              f"ControlNet Strength: {controlnet_conditioning_scale}")

        result = self.pipeline(
            prompt=prompt,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            width=width,
            height=height,
            generator=generator,
            **kwargs
        )

        return result.images[0]

    def unload(self):
        """Unload models to free VRAM."""
        self.controlnet = None
        self.pipeline = None
        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Flux ControlNet models unloaded")


def create_controlnet_manager(
    control_type: str,
    model_name: str = "instantx",
    **kwargs
) -> FluxControlNetManager:
    """Factory function to create a ControlNet manager.

    Args:
        control_type: "canny" or "depth"
        model_name: Model provider name
        **kwargs: Additional arguments for FluxControlNetManager

    Returns:
        Configured FluxControlNetManager instance
    """
    return FluxControlNetManager(
        control_type=control_type,
        model_name=model_name,
        **kwargs
    )


# Convenience functions for quick usage

def generate_with_canny_controlnet(
    prompt: str,
    control_image: Union[np.ndarray, Image.Image],
    model_name: str = "instantx",
    canny_low: int = 100,
    canny_high: int = 200,
    controlnet_strength: float = 0.7,
    **kwargs
) -> Image.Image:
    """Quick helper to generate with Canny ControlNet.

    Args:
        prompt: Text prompt
        control_image: Input image for Canny edge detection
        model_name: Model provider ("instantx", "xlabs", "bfl")
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        controlnet_strength: ControlNet influence (0.0-1.0)
        **kwargs: Additional generation parameters

    Returns:
        Generated PIL Image
    """
    manager = create_controlnet_manager("canny", model_name)
    return manager.generate(
        prompt=prompt,
        control_image=control_image,
        canny_low=canny_low,
        canny_high=canny_high,
        controlnet_conditioning_scale=controlnet_strength,
        **kwargs
    )


def generate_with_depth_controlnet(
    prompt: str,
    depth_map: Union[np.ndarray, Image.Image],
    model_name: str = "shakker",
    controlnet_strength: float = 0.7,
    **kwargs
) -> Image.Image:
    """Quick helper to generate with Depth ControlNet.

    Args:
        prompt: Text prompt
        depth_map: Depth map (from Depth-Anything V2)
        model_name: Model provider ("shakker", "instantx", "xlabs", "bfl")
        controlnet_strength: ControlNet influence (0.0-1.0)
        **kwargs: Additional generation parameters

    Returns:
        Generated PIL Image
    """
    manager = create_controlnet_manager("depth", model_name)
    return manager.generate(
        prompt=prompt,
        control_image=depth_map,
        controlnet_conditioning_scale=controlnet_strength,
        **kwargs
    )
