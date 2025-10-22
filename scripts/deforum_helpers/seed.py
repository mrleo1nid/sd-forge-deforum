import random
from typing import Any

# Import pure functions from refactored utils module
from deforum.utils.seed_utils import (
    MAX_SEED,
    calculate_seed_increment,
    calculate_next_control,
    generate_next_seed,
)

# Re-export for backward compatibility
__all__ = [
    'MAX_SEED',
    'calculate_seed_increment',
    'calculate_next_control',
    'generate_next_seed',
    'next_seed',
]

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
