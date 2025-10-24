"""Flux ControlNet V2 integration for Deforum.

Forge-native implementation that loads ONLY FluxControlNetModel (~3.6GB)
and integrates with Forge's already-loaded Flux transformer via runtime patches.

Key components:
- manager: FluxControlNetV2Manager - Main manager class
- models: Model loading utilities
- preprocessors: Canny and depth preprocessors
- forge_injection: Global control sample storage
- diffusers_compat: Runtime patches for Forge compatibility
"""

from .manager import FluxControlNetV2Manager
from .models import (
    load_flux_controlnet_model,
    get_model_info,
    get_available_models,
)
from .preprocessors import (
    canny_edge_detection,
    depth_map_to_controlnet_format,
    preprocess_image_for_controlnet,
    overlay_canny_edges,
)
from .forge_injection import (
    store_control_samples,
    get_stored_control_samples,
    clear_control_samples,
)

__all__ = [
    # Manager
    "FluxControlNetV2Manager",
    # Models
    "load_flux_controlnet_model",
    "get_model_info",
    "get_available_models",
    # Preprocessors
    "canny_edge_detection",
    "depth_map_to_controlnet_format",
    "preprocess_image_for_controlnet",
    "overlay_canny_edges",
    # Forge injection
    "store_control_samples",
    "get_stored_control_samples",
    "clear_control_samples",
]
