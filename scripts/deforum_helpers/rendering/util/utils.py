"""General utilities - Legacy wrapper.

This module now imports from deforum.utils.functional_utils for backward compatibility.
All pure utility functions have been refactored to the utils module.
"""

# Import from refactored utils module
from deforum.utils.functional_utils import (
    put_all,
    put_if_present,
    call_or_use_on_cond,
    create_img,
    generate_random_seed,
)

# Re-export for backward compatibility
__all__ = [
    'put_all',
    'put_if_present',
    'call_or_use_on_cond',
    'create_img',
    'generate_random_seed',
]
