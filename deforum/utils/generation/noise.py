"""Pure functions for Perlin noise generation.

This module contains noise-related pure functions extracted from
scripts/deforum_helpers/noise.py, following functional programming principles
with no side effects.
"""

import torch
import numpy as np
from PIL import Image, ImageOps
import math
from typing import Callable

# Global noise generator for reproducible results
deforum_noise_gen = torch.Generator(device='cpu')

# ============================================================================
# PERLIN NOISE HELPER FUNCTIONS
# ============================================================================


def perlin_fade(t: torch.Tensor) -> torch.Tensor:
    """Perlin fade function: 6t^5 - 15t^4 + 10t^3.

    Args:
        t: Input tensor values (typically 0-1)

    Returns:
        Smoothed values
    """
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def round_to_multiple(value: int, multiple: int) -> int:
    """Round value down to nearest multiple.

    Args:
        value: Value to round
        multiple: Multiple to round to

    Returns:
        Largest multiple <= value
    """
    return value - value % multiple


def normalize_perlin(noise: torch.Tensor) -> torch.Tensor:
    """Shift Perlin noise from [-1, 1] to [0, 1] range.

    Args:
        noise: Raw Perlin noise tensor

    Returns:
        Normalized values in [0, 1]
    """
    return (noise + torch.ones(noise.shape)) / 2


# ============================================================================
# NOISE MASK CONDITIONING
# ============================================================================


def condition_noise_mask(noise_mask: Image.Image, invert_mask: bool = False) -> torch.Tensor:
    """Convert PIL mask to normalized torch tensor.

    Args:
        noise_mask: PIL Image mask
        invert_mask: Whether to invert the mask

    Returns:
        Conditioned mask as torch tensor in [0, 1]
    """
    if invert_mask:
        noise_mask = ImageOps.invert(noise_mask)
    mask_array = np.array(noise_mask.convert("L")).astype(np.float32) / 255.0
    mask_array = np.around(mask_array, decimals=0)
    return torch.from_numpy(mask_array)


# ============================================================================
# PERLIN NOISE GENERATION
# ============================================================================


def rand_perlin_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    fade: Callable[[torch.Tensor], torch.Tensor] = perlin_fade,
    generator: torch.Generator | None = None
) -> torch.Tensor:
    """Generate 2D Perlin noise pattern.

    Based on: https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57

    Args:
        shape: Output shape (height, width)
        res: Resolution (frequency) of noise
        fade: Fade function for interpolation
        generator: Random generator for reproducibility

    Returns:
        2D tensor of Perlin noise values in [-1, 1]
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
    """Generate multi-octave Perlin noise by layering frequencies.

    Args:
        shape: Output shape (height, width)
        res: Base resolution (frequency) of noise
        octaves: Number of octaves to combine
        persistence: Amplitude decay per octave
        generator: Random generator for reproducibility

    Returns:
        2D tensor of multi-octave Perlin noise
    """
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
