"""Pure functions for Perlin noise generation.

This module contains noise-related pure functions extracted from
scripts/deforum_helpers/noise.py, following functional programming principles
with no side effects.
"""

import numpy as np
from typing import Callable

# ============================================================================
# PERLIN NOISE HELPER FUNCTIONS
# ============================================================================


def perlin_fade(t: np.ndarray) -> np.ndarray:
    """Perlin fade function for smooth interpolation.

    Formula: 6t^5 - 15t^4 + 10t^3

    Args:
        t: Input values (typically 0-1)

    Returns:
        Smoothed values
    """
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def round_to_multiple(x: int, multiple: int) -> int:
    """Round integer down to nearest multiple.

    Args:
        x: Value to round
        multiple: Multiple to round to

    Returns:
        Largest multiple <= x
    """
    return (x // multiple) * multiple


def normalize_perlin(noise: np.ndarray) -> np.ndarray:
    """Normalize Perlin noise from [-1, 1] to [0, 1].

    Args:
        noise: Raw Perlin noise values

    Returns:
        Normalized values in [0, 1]
    """
    return (noise + 1.0) / 2.0


# ============================================================================
# NOISE MASK CONDITIONING
# ============================================================================


def condition_noise_mask(noise_mask: np.ndarray, invert: bool = False) -> np.ndarray:
    """Condition noise mask for application.

    Converts mask to grayscale, normalizes to [0, 1], and optionally inverts.

    Args:
        noise_mask: Input mask image (HWC format, uint8)
        invert: Whether to invert the mask

    Returns:
        Conditioned mask as float32 in [0, 1]
    """
    # Convert RGB to grayscale if needed
    if noise_mask.ndim == 3 and noise_mask.shape[2] == 3:
        grayscale = np.mean(noise_mask, axis=2)
    else:
        grayscale = noise_mask.squeeze()

    # Normalize to [0, 1]
    normalized = grayscale.astype(np.float32) / 255.0

    # Invert if requested
    if invert:
        normalized = 1.0 - normalized

    return normalized


# ============================================================================
# PERLIN NOISE GENERATION
# ============================================================================


def rand_perlin_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    fade: Callable[[np.ndarray], np.ndarray] = perlin_fade,
) -> np.ndarray:
    """Generate 2D Perlin noise.

    Args:
        shape: Output shape (height, width)
        res: Resolution (frequency) of noise
        fade: Fade function for interpolation

    Returns:
        2D array of Perlin noise values in [-1, 1]
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    )

    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_grads(slice1, slice2):
        return np.repeat(
            np.repeat(gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0),
            d[1],
            axis=1,
        )

    def dot(grad, shift):
        return (
            np.stack(
                (
                    grid[: shape[0], : shape[1], 0] + shift[0],
                    grid[: shape[0], : shape[1], 1] + shift[1],
                ),
                axis=-1,
            )
            * grad[: shape[0], : shape[1]]
        ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    t = fade(grid[: shape[0], : shape[1]])
    return np.sqrt(2) * np.lerp(
        np.lerp(n00, n10, t[..., 0]), np.lerp(n01, n11, t[..., 0]), t[..., 1]
    )


def rand_perlin_2d_octaves(
    shape: tuple[int, int],
    res: tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
    fade: Callable[[np.ndarray], np.ndarray] = perlin_fade,
) -> np.ndarray:
    """Generate multi-octave 2D Perlin noise for more natural appearance.

    Args:
        shape: Output shape (height, width)
        res: Base resolution (frequency) of noise
        octaves: Number of octaves to combine
        persistence: Amplitude decay per octave
        fade: Fade function for interpolation

    Returns:
        2D array of multi-octave Perlin noise in [-1, 1]
    """
    if octaves <= 0:
        return np.zeros(shape)

    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    max_amplitude = 0

    for _ in range(octaves):
        octave_res = (res[0] * frequency, res[1] * frequency)
        noise += amplitude * rand_perlin_2d(shape, octave_res, fade)

        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= 2

    return noise / max_amplitude
