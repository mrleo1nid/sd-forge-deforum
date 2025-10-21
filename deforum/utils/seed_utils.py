"""Pure functions for seed generation and iteration logic.

This module contains all seed-related pure functions extracted from
scripts/deforum_helpers/seed.py, following functional programming principles
with no side effects (except random.randint for 'random' behavior).
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
    behavior: str, seed_control: int, iter_n: int
) -> int:
    """Calculate seed increment based on behavior mode.

    Args:
        behavior: Seed iteration behavior ('iter', 'ladder', 'alternate', 'fixed', 'random')
        seed_control: Current control state value
        iter_n: Interval for 'iter' behavior

    Returns:
        Seed increment value (can be 0, 1, 2, -1, or random for 'random' behavior)
    """
    if behavior == "iter":
        return 1 if seed_control % iter_n == 0 else 0
    elif behavior == "ladder":
        return 2 if seed_control == 0 else -1
    elif behavior == "alternate":
        return 1 if seed_control == 0 else -1
    elif behavior == "fixed":
        return 0
    else:  # "random"
        return random.randint(0, MAX_SEED)


def calculate_next_control(behavior: str, seed_control: int) -> int:
    """Calculate next control value for stateful seed behaviors.

    Args:
        behavior: Seed iteration behavior
        seed_control: Current control state value

    Returns:
        Next control state value
    """
    if behavior in ("ladder", "alternate"):
        return 1 if seed_control == 0 else 0
    elif behavior == "iter":
        return seed_control + 1
    else:
        return seed_control


def generate_next_seed(seed: int, behavior: str, seed_control: int, iter_n: int) -> tuple[int, int]:
    """Generate next seed value and control state.

    Args:
        seed: Current seed value
        behavior: Seed iteration behavior
        seed_control: Current control state
        iter_n: Interval for 'iter' behavior

    Returns:
        Tuple of (next_seed, next_control)
    """
    if behavior == "random":
        return random.randint(0, MAX_SEED), seed_control

    increment = calculate_seed_increment(behavior, seed_control, iter_n)
    next_control = calculate_next_control(behavior, seed_control)

    return seed + increment, next_control
