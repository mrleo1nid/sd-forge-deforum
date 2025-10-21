"""Pure functions for seed generation and iteration logic.

This module contains all seed-related pure functions extracted from
scripts/deforum_helpers/seed.py, following functional programming principles
with no side effects.
"""

import random
from typing import Literal

# Type alias for seed iteration behaviors
SeedBehavior = Literal["iter", "fixed", "random", "ladder", "alternate"]

# ============================================================================
# CONSTANTS
# ============================================================================

MAX_SEED = 2**32 - 1

# ============================================================================
# PURE FUNCTIONS
# ============================================================================


def calculate_seed_increment(
    seed_behavior: SeedBehavior, frame_idx: int, seed_iter_N: int
) -> int:
    """Calculate seed increment based on behavior and frame index.

    Args:
        seed_behavior: Type of seed iteration behavior
        frame_idx: Current frame index
        seed_iter_N: Interval for 'iter' behavior

    Returns:
        Seed increment value (0, 1, or -1)
    """
    if seed_behavior == "iter":
        return 1 if frame_idx % seed_iter_N == 0 else 0
    elif seed_behavior == "ladder":
        return 1
    elif seed_behavior == "alternate":
        return 1
    else:  # "fixed" or "random"
        return 0


def calculate_next_control(control: int, seed_behavior: SeedBehavior) -> int:
    """Calculate next control value for ladder/alternate behaviors.

    Args:
        control: Current control value
        seed_behavior: Type of seed iteration behavior

    Returns:
        Next control value
    """
    if seed_behavior in ("ladder", "alternate"):
        return -1 if control == 1 else 1
    return control


def generate_next_seed(
    seed: int, seed_behavior: SeedBehavior, control: int, seed_increment: int
) -> int:
    """Generate next seed value based on behavior.

    Args:
        seed: Current seed value
        seed_behavior: Type of seed iteration behavior
        control: Current control value (for ladder/alternate)
        seed_increment: Calculated increment value

    Returns:
        Next seed value
    """
    if seed_behavior == "random":
        return random.randint(0, MAX_SEED)
    elif seed_behavior == "ladder":
        return seed + (seed_increment * control)
    elif seed_behavior == "alternate":
        return seed + seed_increment if control == 1 else seed - seed_increment
    else:  # "iter" or "fixed"
        return seed + seed_increment
