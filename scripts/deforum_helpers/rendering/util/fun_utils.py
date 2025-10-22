"""Functional programming utilities - Legacy wrapper.

This module now imports from deforum.utils.functional_utils for backward compatibility.
All pure functional helpers have been refactored to the utils module.
"""

# Import from refactored utils module
from deforum.utils.functional_utils import flat_map, tube

# Re-export for backward compatibility
__all__ = ['flat_map', 'tube']
