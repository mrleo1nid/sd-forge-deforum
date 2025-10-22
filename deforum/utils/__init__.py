"""
Deforum utility modules - pure functions organized by domain.

This package contains refactored pure functions extracted from various
legacy Deforum modules in scripts/deforum_helpers/, organized by functional
domain for better maintainability and testability.

All functions in this package follow functional programming principles:
- Pure functions with no side effects
- Complete type hints
- Comprehensive docstrings
- Unit tested with >70% coverage

Modules:
    seed_utils: Seed generation and iteration logic
    image_utils: Image processing (sharpening, color matching, conversions)
    noise_utils: Perlin noise generation
    prompt_utils: Prompt parsing and interpolation
    transform_utils: 3D transformations and matrix operations
    functional_utils: Functional programming helpers and utilities
    filename_utils: Filename formatting and path generation
    subtitle_utils: Subtitle time formatting and parameter display
    string_utils: String manipulation and formatting utilities
    path_utils: Path manipulation and parsing utilities
    validation_utils: Image validation and checking utilities
    math_utils: Mathematical calculations and 3D transformations
    file_utils: File and path operations
    optical_flow_utils: Optical flow consistency checking
    format_utils: Value formatting and conversion utilities
    interpolation_utils: Frame interpolation calculations
    video_path_utils: Video and image path generation
"""

__all__ = [
    "seed_utils",
    "image_utils",
    "noise_utils",
    "prompt_utils",
    "transform_utils",
    "functional_utils",
    "filename_utils",
    "subtitle_utils",
    "string_utils",
    "path_utils",
    "validation_utils",
    "math_utils",
    "file_utils",
    "optical_flow_utils",
    "format_utils",
    "interpolation_utils",
    "video_path_utils",
]
