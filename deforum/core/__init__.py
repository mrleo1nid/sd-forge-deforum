"""Core domain logic for Deforum extension.

This package contains the core business logic extracted from deforum_helpers/,
organized by domain for better maintainability and testability.

All classes in this package represent core domain concepts and may have some
stateful behavior, but follow clean architecture principles:
- Clear separation of concerns
- Complete type hints
- Comprehensive docstrings
- Testable design

Modules:
    keyframes: Keyframe scheduling and interpolation
    seeds: Seed generation and iteration
    prompts: Prompt scheduling and composition
    schedules: Schedule parsing and evaluation (future)
    motion: Motion calculation logic (future)
"""

__all__ = [
    "keyframes",
    "seeds",
    "prompts",
]
