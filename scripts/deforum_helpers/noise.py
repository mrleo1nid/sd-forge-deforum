import torch
from torch.nn.functional import interpolate
import numpy as np
from PIL import Image, ImageOps
import math
from typing import Callable
import cv2

try:
    from modules.shared import opts
    DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)
except ImportError:
    DEBUG_MODE = False

deforum_noise_gen = torch.Generator(device='cpu')

# ============================================================================
# PURE FUNCTIONS
# ============================================================================

def perlin_fade(t: torch.Tensor) -> torch.Tensor:
    """Perlin fade function: 6t^5 - 15t^4 + 10t^3."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def round_to_multiple(value: int, multiple: int) -> int:
    """Round value down to nearest multiple."""
    return value - value % multiple

def normalize_perlin(noise: torch.Tensor) -> torch.Tensor:
    """Shift Perlin noise from [-1, 1] to [0, 1] range."""
    return (noise + torch.ones(noise.shape)) / 2

def condition_noise_mask(noise_mask: Image.Image, invert_mask: bool = False) -> torch.Tensor:
    """Convert PIL mask to normalized torch tensor."""
    if invert_mask:
        noise_mask = ImageOps.invert(noise_mask)
    mask_array = np.array(noise_mask.convert("L")).astype(np.float32) / 255.0
    mask_array = np.around(mask_array, decimals=0)
    return torch.from_numpy(mask_array)

def rand_perlin_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    fade: Callable[[torch.Tensor], torch.Tensor] = perlin_fade,
    generator: torch.Generator | None = None
) -> torch.Tensor:
    """Generate 2D Perlin noise pattern.

    Based on: https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
    """
    if generator is None:
        generator = deforum_noise_gen

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, res[0], delta[0]),
            torch.arange(0, res[1], delta[1]),
            indexing='ij'
        ),
        dim=-1
    ) % 1

    angles = 2 * math.pi * torch.rand(res[0]+1, res[1]+1, generator=generator)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1: list[int | None], slice2: list[int | None]) -> torch.Tensor:
        return gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(
            d[0], 0
        ).repeat_interleave(d[1], 1)

    def dot(grad: torch.Tensor, shift: list[int]) -> torch.Tensor:
        shifted = torch.stack((
            grid[:shape[0], :shape[1], 0] + shift[0],
            grid[:shape[0], :shape[1], 1] + shift[1]
        ), dim=-1)
        return (shifted * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]),
        torch.lerp(n01, n11, t[..., 0]),
        t[..., 1]
    )

def rand_perlin_2d_octaves(
    shape: tuple[int, int],
    res: tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
    generator: torch.Generator | None = None
) -> torch.Tensor:
    """Generate multi-octave Perlin noise by layering frequencies."""
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1.0

    for _ in range(int(octaves)):
        noise += amplitude * rand_perlin_2d(
            shape,
            (frequency * res[0], frequency * res[1]),
            generator=generator
        )
        frequency *= 2
        amplitude *= persistence

    return noise

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
        from .animation import sample_to_cv2
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
