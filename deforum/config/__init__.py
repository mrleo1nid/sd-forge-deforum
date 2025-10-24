"""Configuration module for Deforum.

Provides argument definitions, default values, and settings persistence
for the Deforum extension.
"""

from .args import (
    DeforumArgs,
    DeforumAnimArgs,
    ParseqArgs,
    LoopArgs,
    # Add other arg classes as needed
)
from .defaults import (
    get_default_settings,
    # Add other default functions as needed
)
from .settings import (
    load_settings,
    save_settings,
    # Add other settings functions as needed
)

__all__ = [
    # Args
    "DeforumArgs",
    "DeforumAnimArgs",
    "ParseqArgs",
    "LoopArgs",
    # Defaults
    "get_default_settings",
    # Settings
    "load_settings",
    "save_settings",
]
