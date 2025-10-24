import torch
from torch.nn.functional import interpolate
import numpy as np
import cv2
from PIL import Image

# Import pure functions from refactored utils module
from deforum.utils.noise_utils import (
    deforum_noise_gen,
    perlin_fade,
    round_to_multiple,
    normalize_perlin,
    condition_noise_mask,
    rand_perlin_2d,
    rand_perlin_2d_octaves,
)

# Re-export for backward compatibility
__all__ = [
    'deforum_noise_gen',
    'perlin_fade',
    'round_to_multiple',
    'normalize_perlin',
    'condition_noise_mask',
    'rand_perlin_2d',
    'rand_perlin_2d_octaves',
    'add_noise',
]

try:
    from modules.shared import opts
    DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)
except ImportError:
    DEBUG_MODE = False

# ============================================================================
# IMPURE FUNCTIONS (side effects: generator state, external dependencies)
# ============================================================================

def add_noise(
    sample: np.ndarray,
    noise_amt: float,
    seed: int,
    noise_type: str,
    noise_args: tuple,
    noise_mask: Image.Image | None = None,
    invert_mask: bool = False
) -> np.ndarray:
    """Add noise to image sample (white or Perlin) with optional mask."""
    try:
        from deforum.animation.animation import sample_to_cv2
    except ImportError:
        from animation import sample_to_cv2

    deforum_noise_gen.manual_seed(seed)

    perlin_w = round_to_multiple(sample.shape[0], 64)
    perlin_h = round_to_multiple(sample.shape[1], 64)
    sample2dshape = (perlin_w, perlin_h)

    noise = torch.randn((sample.shape[2], perlin_w, perlin_h), generator=deforum_noise_gen)

    if noise_type == 'perlin':
        perlin = rand_perlin_2d_octaves(
            sample2dshape,
            (int(noise_args[0]), int(noise_args[1])),
            octaves=noise_args[2],
            persistence=noise_args[3]
        )
        noise = noise * normalize_perlin(perlin)
        noise = interpolate(
            noise.unsqueeze(1),
            size=(sample.shape[0], sample.shape[1])
        ).squeeze(1)

    if noise_mask is not None:
        mask = condition_noise_mask(noise_mask, invert_mask)
        noise_to_add = sample_to_cv2(noise * mask)
    else:
        noise_to_add = sample_to_cv2(noise)

    return cv2.addWeighted(sample, 1 - noise_amt, noise_to_add, noise_amt, 0)
