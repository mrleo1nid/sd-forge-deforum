"""Legacy wrapper for seed logic - imports from deforum.core.seeds and deforum.utils.seed_utils.

This module now imports seed iteration logic from the refactored core module
and pure utility functions from the utils module.
"""

# Import core seed iteration logic
from deforum.core.seeds import (
    MAX_SEED,
    SeedIterator,
    next_seed,
)

# Import pure utility functions
from deforum.utils.seed_utils import (
    calculate_seed_increment,
    calculate_next_control,
    generate_next_seed,
)

# Re-export for backward compatibility
__all__ = [
    'MAX_SEED',
    'SeedIterator',
    'next_seed',
    'calculate_seed_increment',
    'calculate_next_control',
    'generate_next_seed',
]
