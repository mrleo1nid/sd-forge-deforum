"""Flux ControlNet V2 - Forge Processing Injection

Handles injecting pre-computed Flux ControlNet control samples into
Forge's processing pipeline via the UNet patcher's model_options.
"""

import torch
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

# Global storage for control samples (accessed by both generation code and patches)
_current_controlnet_samples: Optional[Tuple] = None


def store_control_samples(
    controlnet_block_samples: Tuple[torch.Tensor, ...],
    controlnet_single_block_samples: Tuple[torch.Tensor, ...]
):
    """Store control samples globally for the current generation.

    These will be picked up by the patched KModel.apply_model during sampling.

    Args:
        controlnet_block_samples: Control samples for double_blocks
        controlnet_single_block_samples: Control samples for single_blocks
    """
    global _current_controlnet_samples
    _current_controlnet_samples = (controlnet_block_samples, controlnet_single_block_samples)
    print(f"🌐 Stored Flux ControlNet samples for generation")
    print(f"   Block samples: {len(controlnet_block_samples)} tensors")
    print(f"   Single block samples: {len(controlnet_single_block_samples)} tensors")


def get_stored_control_samples() -> Optional[Tuple]:
    """Get currently stored control samples.

    Returns:
        Tuple of (controlnet_block_samples, controlnet_single_block_samples) or None
    """
    global _current_controlnet_samples
    return _current_controlnet_samples


def clear_control_samples():
    """Clear stored control samples after generation."""
    global _current_controlnet_samples
    _current_controlnet_samples = None


def inject_control_into_processing(p: "StableDiffusionProcessing"):
    """Inject Flux ControlNet control samples into processing object.

    This modifies the UNet patcher's model_options to include the control samples,
    which will be picked up by the patched KModel.apply_model.

    Args:
        p: Stable Diffusion processing object
    """
    global _current_controlnet_samples

    if _current_controlnet_samples is None:
        return  # No control samples to inject

    controlnet_block_samples, controlnet_single_block_samples = _current_controlnet_samples

    # Access the UNet patcher from the processing object
    try:
        # p.sd_model.forge_objects.unet is the UnetPatcher
        unet_patcher = p.sd_model.forge_objects.unet

        # Inject control samples into model_options
        if not hasattr(unet_patcher, 'model_options'):
            unet_patcher.model_options = {}

        # Store control samples in model_options (will be passed to transformer)
        unet_patcher.model_options['controlnet_block_samples'] = controlnet_block_samples
        unet_patcher.model_options['controlnet_single_block_samples'] = controlnet_single_block_samples

        print(f"✅ Injected Flux ControlNet samples into UNet model_options")

    except Exception as e:
        print(f"⚠️ Failed to inject control samples into processing: {e}")
        import traceback
        traceback.print_exc()
