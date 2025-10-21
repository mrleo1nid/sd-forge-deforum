"""
Deforum - AI-powered animation extension for Stable Diffusion WebUI Forge.

This is the main Deforum library package, following modern Python packaging standards.
Legacy code resides in scripts/deforum_helpers/ and will be gradually migrated here.

Package structure:
    deforum/
        utils/      - Pure utility functions (no side effects)
        core/       - Core business logic (future)
        rendering/  - Rendering pipeline (future)

For WebUI integration points, see:
    scripts/deforum.py - Main WebUI script entry point
    scripts/deforum_api.py - REST API endpoints
"""

__version__ = "2.5.0-refactor"
__all__ = ["utils"]
