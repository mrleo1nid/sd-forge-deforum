# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --tb=short --cov=scripts --cov-report=xml --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest tests/unit/ -m unit              # Unit tests only
pytest tests/integration/ -m integration # Integration tests only
pytest tests/ -m "not slow"             # Skip slow tests

# Run single test file
pytest tests/unit/test_config_system.py -v

# Run tests in parallel (faster)
pytest tests/ -n auto
```

### Code Quality
```bash
# Format code with black
black .

# Sort imports
isort .

# Lint with flake8
flake8 scripts --count --select=E9,F63,F7,F82 --show-source --statistics

# Type checking
mypy scripts --ignore-missing-imports --no-strict-optional

# Security audit
bandit -r scripts/
```

### Dependencies
```bash
# Install development dependencies
pip install -r requirements-dev.txt
pip install -r requirements-test.txt

# Install main dependencies 
pip install -r requirements.txt
```

## Architecture Overview

### Core Structure
- **`deforum/`** - Main package with modular functional architecture
- **`scripts/deforum.py`** - WebUI extension entry point
- **`deforum_extend_paths.py`** - Path extension for module loading

### Key Components

#### Rendering Pipeline
- **`deforum/core/run_deforum.py`** - Main execution pipeline
- **`deforum/core/rendering_engine.py`** - Core rendering logic
- **`deforum/core/frame_processing.py`** - Individual frame processing
- **`deforum/core/animation_controller.py`** - Animation orchestration

#### Data Models (Functional Programming)
- **`deforum/models/data_models.py`** - Immutable data structures using dataclasses
- **`deforum/core/data/`** - Frame, animation, and rendering data models
- All data structures are immutable with type hints

#### Configuration System
- **`deforum/config/arguments.py`** - Argument processing and validation
- **`deforum/config/settings.py`** - Settings persistence
- **`deforum/config/defaults.py`** - Default values and UI text

#### Integrations
- **`deforum/integrations/wan/`** - Wan 2.1 AI video generation (state-of-the-art)
- **`deforum/integrations/controlnet/`** - ControlNet integration (removed in this fork)
- **`deforum/integrations/external_libs/`** - Third-party libraries (MiDaS, RIFE, etc.)

#### Media Processing
- **`deforum/media/`** - Image/video processing, ffmpeg operations, interpolation
- **`deforum/media/interpolation/`** - RIFE and FILM frame interpolation

#### UI System
- **`deforum/ui/`** - Gradio interface components
- **`deforum/ui/wan_components.py`** - Wan-specific UI elements

### Removed Features (Refactoring)
During the extensive refactoring to functional programming patterns, the following features were **completely removed**:

#### Removed Video Features
- **Hybrid Video Mode** - All `hybrid_*` settings and functionality removed
- **Optical Flow** - Complex optical flow processing removed for simplicity
- **Advanced Video Compositing** - Hybrid compositing and flow consistency removed

#### Removed AI Enhancement Features  
- **FreeU** - All `freeu_*` settings and FreeU model enhancement removed
- **Kohya HRFix** - All `kohya_hrfix_*` settings and high-resolution fix removed
- **Advanced ControlNet** - Complex ControlNet scheduling removed (basic ControlNet remains)

#### Settings Migration
- Any settings files containing these removed features will be automatically migrated
- Missing fields are filled with safe defaults or removed entirely
- The extension will display warnings for outdated settings files

**Important**: Do not attempt to restore these features - they were intentionally removed to:
1. Simplify the codebase architecture
2. Focus on core animation functionality + Wan 2.1 integration
3. Reduce maintenance burden and complexity
4. Improve stability and performance

### Functional Programming Patterns
This codebase follows functional programming principles:
- **Immutable Data**: All data models use frozen dataclasses
- **Pure Functions**: Functions avoid side effects where possible
- **Type Safety**: Comprehensive type hints throughout
- **Modular Design**: Clear separation of concerns

### Testing Architecture
- **`tests/unit/`** - Fast unit tests for individual components
- **`tests/integration/`** - Integration tests for component interaction
- **`tests/functional/`** - End-to-end functional tests
- **`tests/performance/`** - Performance and benchmark tests
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.gpu`

### Special Features

#### Wan 2.1 AI Video Generation
- Uses VACE models for unified T2V/I2V generation
- Integrates with Deforum's prompt scheduling system
- Models auto-discovered in `models/wan/` directory
- AI prompt enhancement via Qwen models in `models/qwen/`

#### Camera Shakify
- Real camera motion data from Blender plugin
- Located in `deforum/core/data/shakify/`
- Adds realistic camera shake to animations

#### Keyframe Distribution
- Experimental rendering core for optimized keyframe placement
- Reduces jitter in high/no-cadence animations
- Uses `deforum/core/data/frame/key_frame_distribution.py`

## WebUI Integration

This is a **Stable Diffusion WebUI Forge extension**, not a standalone application:
- Entry point: `scripts/deforum.py`
- Depends on WebUI's `modules.shared`, `modules.processing`, etc.
- UI components register via `script_callbacks.on_ui_tabs()`
- Settings integrate with WebUI's settings system

## Development Notes

### Removed Features (vs Original Deforum)
- ControlNet integration (tab hidden)
- FreeU functionality
- Hybrid Video processing  
- Kohya HR Fix
- Legacy depth algorithms (AdaBins, ZoeDepth, LeReS)
- HTTP REST API

### Testing Strategy
- Tests are organized by speed: unit (fast) → integration → functional (slow)
- Use `pytest -m "not slow"` for quick development cycles
- GPU tests marked with `@pytest.mark.gpu`
- Coverage target: maintain >80% for core modules

### Model Requirements
- **Flux.1**: Primary model support (dev/schnell variants)
- **Wan 2.1**: Download to `models/wan/` (VACE-1.3B recommended)
- **Depth Models**: MiDaS-3-Hybrid, Depth-Anything-V2
- **Qwen**: Auto-downloaded to `models/qwen/` for prompt enhancement