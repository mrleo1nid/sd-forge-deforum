"""Flux ControlNet v2 - Forge-native integration.

Loads ONLY FluxControlNetModel (~3.6GB) and computes control samples directly,
then injects into Forge's already-loaded Flux transformer (no VRAM duplication).

This replaces the v1 approach which loaded full FluxControlNetPipeline (~24GB total).
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Union, Tuple
from diffusers import FluxControlNetModel

from .flux_controlnet_models import (
    load_flux_controlnet_model,
    get_model_info,
    temporarily_unpatch_hf_download
)
from .flux_controlnet_preprocessors import (
    preprocess_image_for_controlnet,
    numpy_to_pil
)


class FluxControlNetV2Manager:
    """V2 Manager for Flux ControlNet - Forge-native integration.

    This version:
    - Loads ONLY FluxControlNetModel (~3.6GB)
    - Computes control samples via forward() call
    - Returns samples to be injected into Forge's Flux transformer
    - No pipeline, no duplicate Flux model
    """

    def __init__(
        self,
        control_type: str = "canny",
        model_name: str = "instantx",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ):
        """Initialize Flux ControlNet V2 manager.

        Args:
            control_type: "canny" or "depth"
            model_name: Model provider ("instantx", "xlabs", "bfl", "shakker")
            torch_dtype: Torch data type for weights
            device: Device to load model on
        """
        self.control_type = control_type
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device = device

        self.controlnet = None
        self.is_loaded = False

        print(f"🌐 Initialized Flux {control_type.title()} ControlNet V2 (Forge-native)")
        print(f"   Model: {get_model_info(control_type, model_name)}")

    def load_model(self):
        """Load ONLY the ControlNet model (not pipeline)."""
        if self.is_loaded:
            print("   ControlNet model already loaded")
            return

        print("🌐 Loading Flux ControlNet model (v2 - model only)...")

        # Load ControlNet model (~3.6GB)
        self.controlnet = load_flux_controlnet_model(
            control_type=self.control_type,
            model_name=self.model_name,
            torch_dtype=self.torch_dtype,
            device=self.device
        )

        self.controlnet = self.controlnet.to(self.device)
        self.is_loaded = True

        print(f"✓ Flux ControlNet model loaded ({self.control_type}, ~3.6GB)")
        print("   No pipeline created - will use Forge's Flux transformer")

    def compute_control_samples(
        self,
        hidden_states: torch.Tensor,
        control_image: Union[np.ndarray, Image.Image],
        conditioning_scale: float = 0.7,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        preprocess_control: bool = True,
        canny_low: int = 100,
        canny_high: int = 200,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Compute control samples from preprocessed control image.

        This method calls FluxControlNetModel.forward() directly with parameters
        from Forge's processing pipeline.

        Args:
            hidden_states: Latent image tensor from Forge
            control_image: Control image (preprocessed or raw)
            conditioning_scale: ControlNet strength (0.0-1.0)
            encoder_hidden_states: Text embeddings from Forge's CLIP
            pooled_projections: Pooled text embeddings
            timestep: Current denoising timestep
            img_ids: Image position IDs
            txt_ids: Text position IDs
            guidance: Guidance scale tensor
            preprocess_control: Whether to preprocess control image
            canny_low: Canny low threshold (for Canny mode)
            canny_high: Canny high threshold (for Canny mode)

        Returns:
            Tuple of (controlnet_block_samples, controlnet_single_block_samples)
            to be passed to Forge's Flux transformer
        """
        if not self.is_loaded:
            self.load_model()

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

        # Convert PIL to tensor for ControlNet model
        # Expected format: (batch, channels, height, width) in range [-1, 1]
        control_tensor = torch.from_numpy(np.array(control_image)).float() / 127.5 - 1.0
        control_tensor = control_tensor.permute(2, 0, 1).unsqueeze(0)  # (B, C, H, W)
        control_tensor = control_tensor.to(device=self.device, dtype=self.torch_dtype)

        print(f"🌐 Computing ControlNet control samples...")
        print(f"   Control image shape: {control_tensor.shape}")
        print(f"   Hidden states shape: {hidden_states.shape if hidden_states is not None else 'None'}")
        print(f"   Conditioning scale: {conditioning_scale}")

        # Call FluxControlNetModel.forward() directly
        with torch.inference_mode():
            controlnet_output = self.controlnet(
                hidden_states=hidden_states,
                controlnet_cond=control_tensor,
                conditioning_scale=conditioning_scale,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=True
            )

        # Extract control samples
        controlnet_block_samples = controlnet_output.controlnet_block_samples
        controlnet_single_block_samples = controlnet_output.controlnet_single_block_samples

        print(f"✓ ControlNet samples computed:")
        print(f"   Block samples: {len(controlnet_block_samples)} tensors")
        print(f"   Single block samples: {len(controlnet_single_block_samples)} tensors")

        return controlnet_block_samples, controlnet_single_block_samples

    def unload(self):
        """Unload model to free VRAM."""
        self.controlnet = None
        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🌐 Flux ControlNet V2 model unloaded")
