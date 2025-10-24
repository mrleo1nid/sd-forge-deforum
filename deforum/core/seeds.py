"""Core seed generation and iteration logic.

This module contains the seed iteration system that controls how seeds change
between frames in Deforum animations. The seed behavior determines the variety
and consistency of generated frames.

Seed Behaviors:
    iter: Increment seed every N frames (controlled by seed_iter_N)
    ladder: Alternating pattern: +2, -1, +2, -1, ...
    alternate: Alternating pattern: +1, -1, +1, -1, ...
    fixed: Keep seed constant throughout animation
    random: Generate completely random seed each frame

The module uses pure utility functions from deforum.utils.generation.seeds for
calculations, but provides stateful iteration logic that updates arguments
in-place (necessary for integration with the rendering pipeline).

Classes:
    SeedIterator: Stateful seed iteration with behavior modes

Functions:
    next_seed: Legacy function for updating seed in-place (IMPURE)
"""

import random
from typing import Any, Literal

# Import pure utilities
from deforum.utils.generation.seeds import (
    MAX_SEED,
    calculate_seed_increment,
    calculate_next_control,
    generate_next_seed,
)

# Re-export constants
__all__ = [
    'MAX_SEED',
    'SeedIterator',
    'next_seed',
]

SeedBehavior = Literal['iter', 'ladder', 'alternate', 'fixed', 'random']


class SeedIterator:
    """Stateful seed iterator with configurable behavior modes.

    Manages seed progression through animation frames based on the selected
    behavior mode. Maintains internal state for patterns like 'ladder' and
    'alternate' that require memory between iterations.

    Attributes:
        seed: Current seed value
        behavior: Seed behavior mode
        iter_n: For 'iter' mode: increment every N frames
        internal_counter: Internal state for pattern tracking

    Examples:
        >>> # Iter mode: increment every 10 frames
        >>> iterator = SeedIterator(seed=42, behavior='iter', iter_n=10)
        >>> for i in range(20):
        ...     seed = iterator.next()
        ...     # seed increments at frame 0, 10, 20, ...

        >>> # Ladder mode: +2, -1, +2, -1, ...
        >>> iterator = SeedIterator(seed=100, behavior='ladder')
        >>> seeds = [iterator.next() for _ in range(4)]
        >>> # seeds = [102, 101, 103, 102]

        >>> # Fixed mode: constant seed
        >>> iterator = SeedIterator(seed=42, behavior='fixed')
        >>> all(iterator.next() == 42 for _ in range(100))
        True
    """

    def __init__(
        self,
        seed: int,
        behavior: SeedBehavior = 'iter',
        iter_n: int = 1
    ):
        """Initialize seed iterator.

        Args:
            seed: Starting seed value
            behavior: Iteration behavior mode
            iter_n: For 'iter' mode, increment every N frames

        Raises:
            ValueError: If behavior is invalid
        """
        valid_behaviors = {'iter', 'ladder', 'alternate', 'fixed', 'random'}
        if behavior not in valid_behaviors:
            raise ValueError(
                f"Invalid seed behavior: {behavior}. "
                f"Must be one of: {valid_behaviors}"
            )

        self.seed = seed
        self.behavior = behavior
        self.iter_n = iter_n
        self.internal_counter = 0

    def next(self) -> int:
        """Get next seed value and update internal state.

        Returns:
            Next seed value based on behavior mode

        Examples:
            >>> iterator = SeedIterator(seed=42, behavior='iter', iter_n=2)
            >>> iterator.next()  # Frame 0 (increments)
            43
            >>> iterator.next()  # Frame 1
            43
            >>> iterator.next()  # Frame 2 (increments)
            44
        """
        if self.behavior == 'iter':
            # Increment BEFORE returning (matches legacy behavior)
            if self.internal_counter % self.iter_n == 0:
                self.seed += 1
            self.internal_counter += 1
            return self.seed

        elif self.behavior == 'ladder':
            # Pattern: +2, -1, +2, -1, ...
            if self.internal_counter == 0:
                self.seed += 2
            else:
                self.seed -= 1
            self.internal_counter = 1 if self.internal_counter == 0 else 0
            return self.seed

        elif self.behavior == 'alternate':
            # Pattern: +1, -1, +1, -1, ...
            if self.internal_counter == 0:
                self.seed += 1
            else:
                self.seed -= 1
            self.internal_counter = 1 if self.internal_counter == 0 else 0
            return self.seed

        elif self.behavior == 'fixed':
            # Keep seed constant
            return self.seed

        else:  # 'random'
            self.seed = random.randint(0, MAX_SEED)
            return self.seed

    def reset(self, seed: int | None = None):
        """Reset iterator to initial state.

        Args:
            seed: New seed value (optional, keeps current if not provided)
        """
        if seed is not None:
            self.seed = seed
        self.internal_counter = 0


# ============================================================================
# LEGACY FUNCTION (IMPURE: mutates arguments)
# ============================================================================

def next_seed(args: Any, root: Any) -> int:
    """Update seed in-place based on behavior mode.

    Legacy function that mutates args.seed and root.seed_internal in-place.
    Maintained for backward compatibility with existing rendering pipeline.

    Args:
        args: Arguments object with seed and seed_behavior attributes
        root: Root object with seed_internal state

    Returns:
        Updated seed value

    Side Effects:
        - Mutates args.seed
        - Mutates root.seed_internal

    Note:
        For new code, prefer using SeedIterator class for cleaner state management.

    Examples:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(seed=42, seed_behavior='iter', seed_iter_N=2)
        >>> root = SimpleNamespace(seed_internal=0)
        >>> next_seed(args, root)  # First call
        42
        >>> next_seed(args, root)  # Second call
        42
        >>> next_seed(args, root)  # Third call (increment)
        43
    """
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
