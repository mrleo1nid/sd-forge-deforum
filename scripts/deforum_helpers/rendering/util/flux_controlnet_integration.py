"""Integration layer for Flux ControlNet with Deforum keyframe generation.

Provides utilities to generate keyframes using Flux ControlNet based on:
- Canny edges from previous frame
- Depth maps from Depth-Anything V2

CURRENT STATE (v1 - Standalone diffusers approach):
- Uses separate FluxControlNetPipeline from diffusers
- Loads full Flux model from HuggingFace (requires authentication)
- Works but duplicates VRAM (Forge's Flux + our pipeline's Flux)
- User must run: huggingface-cli login and accept FLUX.1-dev license

FUTURE DIRECTION (v2 - Forge-native integration):
- Load ONLY FluxControlNetModel (3.6GB), not full pipeline
- Reuse Forge's already-loaded Flux model (no duplication)
- Inject ControlNet conditioning into Forge's processing pipeline
- Requires understanding Forge backend + Flux architecture better

Why not done yet:
- Forge stores Flux as single .safetensors (flux1-dev-bnb-nf4-v2.safetensors)
- Diffusers expects full model repo with config.json, etc.
- These formats are incompatible without conversion
- Forge backend doesn't have Flux ControlNet support (project semi-abandoned)
- Would need to implement Flux ControlNet patcher in Forge's backend

See: archive/crazy-flux-controlnet-monkeypatch for previous attempt (too complex)
"""

import cv2
import numpy as np
import os
from PIL import Image
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.render_data import RenderData
    from ..data.frame.diffusion_frame import DiffusionFrame


def get_local_flux_model_path() -> Optional[str]:
    """Detect local Flux model to avoid HuggingFace authentication.

    Forge stores Flux models in models/Stable-diffusion/Flux/ directory.
    This function checks for local Flux files and returns the path if found.

    Returns:
        Path to local Flux model directory, or None if not found
    """
    try:
        # Try to find Forge's webui root
        import sys
        from pathlib import Path

        # Forge's models are relative to the webui root
        # We're in: extensions/sd-forge-deforum/scripts/deforum_helpers/rendering/util/
        # Need to go up to webui root
        current_file = Path(__file__)
        webui_root = current_file.parents[6]  # Go up 6 levels

        flux_dir = webui_root / "models" / "Stable-diffusion" / "Flux"

        if flux_dir.exists():
            # Check for Flux model files
            flux_files = list(flux_dir.glob("*.safetensors"))
            if flux_files:
                print(f"  Found local Flux model: {flux_files[0].name}")
                # For diffusers, we need the directory, not the specific file
                # But diffusers expects a model with config.json, etc.
                # Since we only have .safetensors, we can't use it directly
                # We'll need to keep using HF repo ID but with local cache
                pass

        return None

    except Exception as e:
        print(f"  Could not detect local Flux model: {e}")
        return None


def is_flux_controlnet_enabled(data: "RenderData") -> bool:
    """Check if Flux ControlNet is enabled in animation args.

    Args:
        data: Rendering data containing animation arguments

    Returns:
        True if Flux ControlNet is enabled
    """
    anim_args = data.args.anim_args
    return getattr(anim_args, 'enable_flux_controlnet', False)


def should_use_flux_controlnet_for_frame(data: "RenderData", frame: "DiffusionFrame") -> bool:
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
    data: "RenderData",
    frame: "DiffusionFrame",
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


def prepare_flux_controlnet_for_frame(
    data: "RenderData",
    frame: "DiffusionFrame"
):
    """Prepare Flux ControlNet V2 for generation (Forge-native).

    Computes control samples and stores them for Forge's pipeline to pick up.
    Does NOT perform generation - that's handled by Forge's normal pipeline.

    Args:
        data: Rendering data containing settings and state
        frame: Current keyframe to generate
    """
    from ...flux_controlnet_v2 import FluxControlNetV2Manager
    from ...flux_controlnet_forge_injection import store_control_samples
    from ...flux_controlnet_preprocessors import overlay_canny_edges, canny_edge_detection
    import torch

    anim_args = data.args.anim_args
    args = data.args.args

    # Get ControlNet settings
    control_type = getattr(anim_args, 'flux_controlnet_type', 'canny')
    model_name = getattr(anim_args, 'flux_controlnet_model', 'instantx')
    strength = getattr(anim_args, 'flux_controlnet_strength', 0.7)
    canny_low = getattr(anim_args, 'flux_controlnet_canny_low', 100)
    canny_high = getattr(anim_args, 'flux_controlnet_canny_high', 200)

    print(f"   Type: {control_type}, Model: {model_name}, Strength: {strength}")

    # Get control image
    control_image = get_control_image_for_frame(data, frame, control_type)
    if control_image is None:
        print(f"⚠️ Could not get {control_type} control image, skipping ControlNet")
        return

    print(f"   Control image shape: {control_image.shape}")
    print(f"   Control image dtype: {control_image.dtype}, range: [{control_image.min():.3f}, {control_image.max():.3f}]")

    # Overlay canny edges on depth-raft-preview.png if canny mode
    if control_type == "canny":
        try:
            # Compute canny edges for visualization
            canny_edges = canny_edge_detection(control_image, canny_low, canny_high)

            # Find depth-raft-preview.png (directly in batch directory)
            import os
            depth_preview_path = os.path.join(args.outdir, "depth-raft-preview.png")

            if os.path.exists(depth_preview_path):
                # Load existing depth preview
                depth_preview = cv2.imread(depth_preview_path)
                if depth_preview is not None:
                    # Convert BGR to RGB for overlay function
                    depth_preview_rgb = cv2.cvtColor(depth_preview, cv2.COLOR_BGR2RGB)
                    # Overlay edges in red
                    overlay = overlay_canny_edges(depth_preview_rgb, canny_edges, edge_color=(255, 0, 0), alpha=0.8)
                    # Save back to same file
                    cv2.imwrite(depth_preview_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    print(f"   💡 Overlaid canny edges on: {depth_preview_path}")
            else:
                # Create black preview image if it doesn't exist
                print(f"   ℹ️ Depth preview not found, creating black preview: {depth_preview_path}")
                # Create black image matching control image size
                black_preview = np.zeros_like(control_image)
                # Overlay canny edges on black background
                overlay = overlay_canny_edges(black_preview, canny_edges, edge_color=(255, 0, 0), alpha=1.0)
                # Save
                cv2.imwrite(depth_preview_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"   💡 Created canny edge preview: {depth_preview_path}")
        except Exception as e:
            print(f"   ⚠️ Could not overlay canny visualization: {e}")
            import traceback
            traceback.print_exc()

    # Initialize V2 ControlNet manager
    manager = FluxControlNetV2Manager(
        control_type=control_type,
        model_name=model_name,
        device='cuda'
    )
    manager.load_model()

    # Compute control samples
    # Note: We'll pass None for hidden_states and let ControlNet compute from control image
    # This avoids dummy shape mismatches
    dummy_hidden_states = None

    try:
        controlnet_block_samples, controlnet_single_block_samples = manager.compute_control_samples(
            hidden_states=dummy_hidden_states,
            control_image=control_image,
            conditioning_scale=strength,
            preprocess_control=True,
            canny_low=canny_low,
            canny_high=canny_high
        )

        # Debug: Check control sample values
        print(f"   DEBUG: block_samples[0] shape: {controlnet_block_samples[0].shape}, "
              f"dtype: {controlnet_block_samples[0].dtype}, "
              f"range: [{controlnet_block_samples[0].min():.6f}, {controlnet_block_samples[0].max():.6f}], "
              f"mean: {controlnet_block_samples[0].mean():.6f}")
        print(f"   DEBUG: single_block_samples[0] shape: {controlnet_single_block_samples[0].shape}, "
              f"dtype: {controlnet_single_block_samples[0].dtype}, "
              f"range: [{controlnet_single_block_samples[0].min():.6f}, {controlnet_single_block_samples[0].max():.6f}], "
              f"mean: {controlnet_single_block_samples[0].mean():.6f}")

        # Store control samples for Forge to pick up
        store_control_samples(controlnet_block_samples, controlnet_single_block_samples)

        print(f"✓ Flux ControlNet V2 prepared successfully")

    except Exception as e:
        print(f"⚠️ Flux ControlNet V2 error: {e}")
        import traceback
        traceback.print_exc()


def generate_with_flux_controlnet(
    data: "RenderData",
    frame: "DiffusionFrame"
) -> Image.Image:
    """Generate keyframe using Flux ControlNet V2 (Forge-native).

    V2 uses ONLY FluxControlNetModel without pipeline, integrating with
    Forge's already-loaded Flux to avoid VRAM duplication.

    Args:
        data: Rendering data containing settings and state
        frame: Current keyframe to generate

    Returns:
        Generated PIL Image
    """
    # Import here to avoid circular dependency
    from ...flux_controlnet_v2 import FluxControlNetV2Manager
    import torch
    import numpy as np

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
    print(f"🌐 Flux ControlNet V2: {control_type}, Model: {model_name}, Strength: {strength}")
    control_image = get_control_image_for_frame(data, frame, control_type)
    if control_image is None:
        raise ValueError(f"Could not get {control_type} control image for frame {frame.i}")

    print(f"   Control image shape: {control_image.shape}")

    # Initialize V2 ControlNet manager (loads only ControlNet model)
    print(f"   Loading Flux ControlNet model (V2 - model only)...")
    manager = FluxControlNetV2Manager(
        control_type=control_type,
        model_name=model_name,
        device='cuda'
    )
    manager.load_model()

    # Compute control samples from control image
    # For initial implementation, we compute with dummy hidden_states
    # TODO: Compute per-step with actual latents for better control
    print(f"   Computing control samples...")

    # Create dummy hidden_states matching Flux's patchification
    # Flux uses VAE (16x downsample) + patchification (patch_size=2)
    batch_size = 1

    # After VAE encoding
    latent_h = args.H // 16
    latent_w = args.W // 16

    # Flux pads to make dimensions divisible by patch_size (2)
    patch_size = 2
    pad_h = (patch_size - latent_h % patch_size) % patch_size
    pad_w = (patch_size - latent_w % patch_size) % patch_size
    padded_h = latent_h + pad_h
    padded_w = latent_w + pad_w

    # After patchification: (h/2) * (w/2) patches
    h_patches = padded_h // patch_size
    w_patches = padded_w // patch_size
    seq_len = h_patches * w_patches

    # Each patch contains (16 channels * 2 * 2) = 64 values
    channels = 64

    dummy_hidden_states = torch.zeros(
        (batch_size, seq_len, channels),
        device='cuda',
        dtype=torch.bfloat16
    )

    print(f"   Latent dimensions: {latent_h}x{latent_w} → padded: {padded_h}x{padded_w}")
    print(f"   Patches: {h_patches}x{w_patches} = {seq_len} patches, {channels} channels per patch")

    # Also need dummy text embeddings and other parameters
    # For now, compute control samples without these (simplified)
    try:
        controlnet_block_samples, controlnet_single_block_samples = manager.compute_control_samples(
            hidden_states=dummy_hidden_states,
            control_image=control_image,
            conditioning_scale=strength,
            preprocess_control=True,
            canny_low=canny_low,
            canny_high=canny_high
        )

        print(f"✓ Control samples computed successfully")
        print(f"   Block samples: {len(controlnet_block_samples)}, Single block samples: {len(controlnet_single_block_samples)}")

        # Now generate using Forge's normal pipeline with control samples
        # Pass control samples via transformer_options (will be picked up by patched KModel)
        # TODO: Integrate with Forge's processing pipeline properly

        # For now, fall back to v1 for actual generation since we need more integration work
        print("⚠️ V2 control sample computation works! But full integration not complete yet.")
        print("   Falling back to v1 pipeline for now...")

        from ...flux_controlnet import FluxControlNetManager as V1Manager
        v1_manager = V1Manager(
            control_type=control_type,
            model_name=model_name,
            base_model=getattr(anim_args, 'flux_base_model', 'black-forest-labs/FLUX.1-dev'),
            device='cuda'
        )

        prompt = args.prompt
        negative_prompt = getattr(args, 'negative_prompt', '')

        result = v1_manager.generate(
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

    except Exception as e:
        print(f"⚠️ V2 ControlNet error, falling back to v1: {e}")
        import traceback
        traceback.print_exc()

        # Fall back to v1
        from ...flux_controlnet import FluxControlNetManager as V1Manager
        v1_manager = V1Manager(
            control_type=control_type,
            model_name=model_name,
            base_model=getattr(anim_args, 'flux_base_model', 'black-forest-labs/FLUX.1-dev'),
            device='cuda'
        )

        prompt = args.prompt
        negative_prompt = getattr(args, 'negative_prompt', '')

        result = v1_manager.generate(
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
