import random
from typing import Any

# Constants
MAX_SEED = 2**32 - 1

# ============================================================================
# PURE FUNCTIONS (testable, no side effects except random.randint)
# ============================================================================

def calculate_seed_increment(behavior: str, seed_control: int, iter_n: int) -> int:
    """Calculate seed increment based on behavior mode."""
    if behavior == 'iter':
        return 1 if seed_control % iter_n == 0 else 0
    elif behavior == 'ladder':
        return 2 if seed_control == 0 else -1
    elif behavior == 'alternate':
        return 1 if seed_control == 0 else -1
    elif behavior == 'fixed':
        return 0
    else:  # random
        return random.randint(0, MAX_SEED)

def calculate_next_control(behavior: str, seed_control: int) -> int:
    """Calculate next control value for stateful seed behaviors."""
    if behavior in ('ladder', 'alternate'):
        return 1 if seed_control == 0 else 0
    elif behavior == 'iter':
        return seed_control + 1
    else:
        return seed_control

def generate_next_seed(args: Any, seed: int, seed_control: int = 0) -> tuple[int, int]:
    """Generate next seed value and control state (PURE except for random behavior).

    Args:
        args: Arguments object with seed_behavior and seed_iter_N
        seed: Current seed value
        seed_control: Current control state (for iter/ladder/alternate modes)

    Returns:
        Tuple of (next_seed, next_control)
    """
    behavior = args.seed_behavior

    if behavior == 'random':
        return random.randint(0, MAX_SEED), seed_control

    increment = calculate_seed_increment(behavior, seed_control, args.seed_iter_N)
    next_control = calculate_next_control(behavior, seed_control)

    return seed + increment, next_control

# ============================================================================
# IMPURE FUNCTIONS (side effects: mutates arguments)
# ============================================================================

def next_seed(args: Any, root: Any) -> int:
    """Update seed in-place based on behavior mode (IMPURE: mutates args and root)."""
    if args.seed_behavior == 'iter':
        args.seed += 1 if root.seed_internal % args.seed_iter_N == 0 else 0
        root.seed_internal += 1
    elif args.seed_behavior == 'ladder':
        args.seed += 2 if root.seed_internal == 0 else -1
        root.seed_internal = 1 if root.seed_internal == 0 else 0
    elif args.seed_behavior == 'alternate':
        args.seed += 1 if root.seed_internal == 0 else -1
        root.seed_internal = 1 if root.seed_internal == 0 else 0
    elif args.seed_behavior == 'fixed':
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, MAX_SEED)
    return args.seed
