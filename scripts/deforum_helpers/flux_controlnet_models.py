"""Flux ControlNet model loading and management for Deforum.

Handles loading and caching of Flux ControlNet models from HuggingFace.
Supports Canny and Depth ControlNet models.
"""

import torch
from diffusers import FluxControlNetModel, FluxControlNetPipeline
from typing import Optional, Dict
import os
from contextlib import contextmanager


@contextmanager
def temporarily_unpatch_hf_download():
    """Temporarily restore original HuggingFace download to avoid etag parameter conflict.

    Forge patches huggingface_hub download functions with a signature that doesn't
    support the 'etag' parameter that newer diffusers/transformers uses.
    This context manager temporarily restores the original during model loading.
    """
    try:
        from huggingface_hub import file_download
        patched_fn = file_download._download_to_tmp_and_move

        # Try to get the original function (it's wrapped by the patch)
        # The patch calls original_download_to_tmp_and_move, but we need to access it
        if hasattr(patched_fn, '__wrapped__'):
            original_fn = patched_fn.__wrapped__
        else:
            # Import the original directly from a fresh module
            import importlib
            import sys
            # Remove cached module to force reimport
            if 'huggingface_hub.file_download' in sys.modules:
                cached_module = sys.modules['huggingface_hub.file_download']
                # Save the patched version
                patched_fn = cached_module._download_to_tmp_and_move

                # Import fresh to get original
                from huggingface_hub.file_download import _download_to_tmp_and_move as original_fn_fresh

                # Temporarily replace with original
                file_download._download_to_tmp_and_move = original_fn_fresh

                yield

                # Restore patched version
                file_download._download_to_tmp_and_move = patched_fn
                return
            else:
                # No patching detected, just yield
                yield
                return

        # Restore original
        file_download._download_to_tmp_and_move = original_fn

        yield

        # Restore patch
        file_download._download_to_tmp_and_move = patched_fn

    except Exception as e:
        print(f"Warning: Could not unpatch HF download: {e}")
        # Continue anyway
        yield


# Available Flux ControlNet models
FLUX_CONTROLNET_MODELS = {
    "canny": {
        "instantx": "InstantX/FLUX.1-dev-Controlnet-Canny",
        "xlabs": "XLabs-AI/flux-controlnet-canny-diffusers",
        "bfl": "black-forest-labs/FLUX.1-Canny-dev",
    },
    "depth": {
        "shakker": "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
        "instantx": "InstantX/FLUX.1-dev-Controlnet-Depth",
        "xlabs": "XLabs-AI/flux-controlnet-depth-diffusers",
        "bfl": "black-forest-labs/FLUX.1-Depth-dev",
    }
}

# Model cache
_model_cache: Dict[str, FluxControlNetModel] = {}
_pipeline_cache: Optional[FluxControlNetPipeline] = None


def get_available_models(control_type: str) -> Dict[str, str]:
    """Get available models for a control type.

    Args:
        control_type: "canny" or "depth"

    Returns:
        Dictionary of model name -> HuggingFace repo ID
    """
    return FLUX_CONTROLNET_MODELS.get(control_type, {})


def load_flux_controlnet_model(
    control_type: str,
    model_name: str = "instantx",
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda"
) -> FluxControlNetModel:
    """Load a Flux ControlNet model.

    Args:
        control_type: "canny" or "depth"
        model_name: Model provider name (e.g., "instantx", "xlabs", "bfl")
        torch_dtype: Torch data type for model weights
        device: Device to load model on

    Returns:
        Loaded FluxControlNetModel

    Raises:
        ValueError: If control_type or model_name is invalid
    """
    # Check if model is already cached
    cache_key = f"{control_type}_{model_name}"
    if cache_key in _model_cache:
        print(f"Using cached Flux {control_type.title()} ControlNet model: {model_name}")
        return _model_cache[cache_key]

    # Get model repo ID
    if control_type not in FLUX_CONTROLNET_MODELS:
        raise ValueError(f"Invalid control type: {control_type}. Use 'canny' or 'depth'.")

    models = FLUX_CONTROLNET_MODELS[control_type]
    if model_name not in models:
        raise ValueError(f"Invalid model name '{model_name}' for {control_type}. "
                        f"Available: {list(models.keys())}")

    model_id = models[model_name]

    print(f"Loading Flux {control_type.title()} ControlNet model: {model_id}")
    print(f"This may take a while on first load (downloading from HuggingFace)...")

    # Load model with temporarily unpatched HF download to avoid etag parameter conflict
    try:
        with temporarily_unpatch_hf_download():
            controlnet = FluxControlNetModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype
            )

        # Cache the model
        _model_cache[cache_key] = controlnet

        print(f"✓ Flux {control_type.title()} ControlNet model loaded successfully")
        return controlnet

    except Exception as e:
        print(f"Error loading Flux ControlNet model {model_id}: {e}")
        raise


def load_flux_controlnet_pipeline(
    base_model: str = "black-forest-labs/FLUX.1-dev",
    controlnet: Optional[FluxControlNetModel] = None,
    control_type: str = "canny",
    model_name: str = "instantx",
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda"
) -> FluxControlNetPipeline:
    """Load or create a Flux ControlNet pipeline.

    Args:
        base_model: Base Flux model repo ID
        controlnet: Pre-loaded ControlNet model (optional)
        control_type: "canny" or "depth" (if controlnet not provided)
        model_name: Model provider name (if controlnet not provided)
        torch_dtype: Torch data type
        device: Device to load on

    Returns:
        FluxControlNetPipeline ready for inference
    """
    global _pipeline_cache

    # Load ControlNet if not provided
    if controlnet is None:
        controlnet = load_flux_controlnet_model(control_type, model_name, torch_dtype, device)

    # Check if we can reuse cached pipeline (for now, always create new)
    # TODO: Implement pipeline caching with hot-swappable ControlNet

    print(f"Creating Flux ControlNet pipeline with base model: {base_model}")

    try:
        with temporarily_unpatch_hf_download():
            pipe = FluxControlNetPipeline.from_pretrained(
                base_model,
                controlnet=controlnet,
                torch_dtype=torch_dtype
            )
        pipe.to(device)

        print("✓ Flux ControlNet pipeline created successfully")
        return pipe

    except Exception as e:
        print(f"Error creating Flux ControlNet pipeline: {e}")
        raise


def unload_controlnet_models():
    """Clear cached models to free VRAM."""
    global _model_cache, _pipeline_cache

    _model_cache.clear()
    _pipeline_cache = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Flux ControlNet models unloaded from cache")


def get_model_info(control_type: str, model_name: str) -> str:
    """Get human-readable info about a model.

    Args:
        control_type: "canny" or "depth"
        model_name: Model provider name

    Returns:
        Model information string
    """
    models = FLUX_CONTROLNET_MODELS.get(control_type, {})
    repo_id = models.get(model_name, "Unknown")

    provider_names = {
        "instantx": "InstantX",
        "xlabs": "XLabs-AI",
        "bfl": "Black Forest Labs (Official)",
        "shakker": "Shakker Labs"
    }

    provider = provider_names.get(model_name, model_name)

    return f"{control_type.title()} ControlNet by {provider} ({repo_id})"
