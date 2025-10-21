# Refactoring Strategy

**Branch:** `refactor/functional-patterns`

## Overview

This document outlines a **gradual, careful refactoring strategy** that prioritizes safety and testability. We will **NOT** move files around initially. Instead, we'll:

1. Extract pure functions within existing files
2. Add type hints and tests
3. Gradually reorganize structure once code is well-tested

## Current Structure (150 Python files)

```
scripts/
├── deforum.py                          # Main extension entry
├── deforum_api.py                      # REST API
├── deforum_api_models.py               # API models
├── deforum_controlnet.py               # ControlNet integration
└── deforum_helpers/                    # ❌ Poor naming (50 top-level files!)
    ├── run_deforum.py                  # Main orchestrator
    ├── args.py, defaults.py            # Configuration
    ├── animation.py, prompt.py         # Core logic
    ├── depth*.py                       # Depth estimation
    ├── ui_*.py                         # Gradio UI
    ├── *_utils.py (23 files)           # Utils scattered everywhere
    ├── rendering/                      # Render pipelines
    │   ├── experimental_core.py        # Main render loop
    │   ├── render_wan_flux.py          # Wan/Flux mode
    │   ├── img_2_img_tubes.py          # Image transformations
    │   ├── data/                       # Data structures
    │   │   ├── render_data.py          # Central state
    │   │   ├── frame/                  # Frame system
    │   │   │   ├── diffusion_frame.py
    │   │   │   ├── tween_frame.py
    │   │   │   └── key_frame_distribution.py
    │   │   ├── anim/                   # Animation data
    │   │   ├── shakify/                # Camera shake
    │   │   └── subtitle/               # Subtitle generation
    │   └── util/                       # Rendering utilities
    │       ├── *_utils.py              # More scattered utils
    │       └── call/                   # Wrapper functions (impure)
    ├── wan/                            # Wan integration
    │   ├── wan_simple_integration.py
    │   ├── qwen_prompt_expander.py
    │   ├── pipelines/
    │   ├── configs/
    │   └── utils/
    └── src/                            # ⚠️ Third-party code
        ├── adabins/
        ├── clipseg/
        ├── film_interpolation/
        ├── leres/
        ├── midas/
        ├── rife/
        └── zoedepth/
```

## Problems with Current Structure

### 1. Naming Issues
- `deforum_helpers/` should be `deforum/` (the actual package)
- `src/` mixes third-party code with our code
- Many files have redundant naming (e.g., `depth.py`, `depth_anything_v2.py`, `vid2depth.py`)

### 2. Organization Issues
- **Too flat at top level**: 50 files in `deforum_helpers/`
- **No separation**: Pure functions mixed with I/O, state, UI
- **Scattered utils**: 23 `*_utils.py` files with no clear organization
- **Deep nesting in wrong places**: `rendering/data/frame/` has 3 levels, but top is flat

### 3. Functional Programming Issues
- Pure functions mixed with side effects
- No clear separation of concerns
- Hard to test (functions do too many things)
- State is not isolated

## Target Structure (End Goal)

**Note:** This is the FINAL target. We will migrate gradually, not all at once.

```
scripts/
├── deforum_extension.py                # Main extension entry (was: deforum.py)
├── deforum_api.py                      # REST API (unchanged)
└── deforum/                            # ✅ Main package (was: deforum_helpers/)
    │
    ├── core/                           # Pure domain logic (NEW)
    │   ├── __init__.py
    │   ├── keyframes.py                # Pure keyframe calculations
    │   ├── schedules.py                # Pure schedule parsing/interpolation
    │   ├── prompts.py                  # Pure prompt parsing/expansion
    │   ├── seeds.py                    # Pure seed generation logic
    │   └── motion.py                   # Pure motion calculations
    │
    ├── utils/                          # Pure utility functions (ORGANIZED)
    │   ├── __init__.py
    │   ├── math.py                     # Math operations (lerp, clamp, etc.)
    │   ├── images.py                   # Image format conversions (pure)
    │   ├── frames.py                   # Frame number calculations
    │   ├── time.py                     # Time/FPS conversions
    │   ├── text.py                     # String parsing/formatting
    │   └── validation.py               # Input validation
    │
    ├── rendering/                      # Rendering pipeline (REFACTORED)
    │   ├── __init__.py
    │   ├── pipeline.py                 # Main render loop (was: experimental_core.py)
    │   ├── wan_flux.py                 # Wan/Flux mode (was: render_wan_flux.py)
    │   ├── transformations.py          # Image transformations (was: img_2_img_tubes.py)
    │   ├── frames/                     # Frame system
    │   │   ├── __init__.py
    │   │   ├── diffusion_frame.py      # Keyframes
    │   │   ├── tween_frame.py          # Interpolated frames
    │   │   └── distribution.py         # Frame distribution (was: key_frame_distribution.py)
    │   ├── state.py                    # Central state (was: render_data.py)
    │   └── io/                         # I/O operations (side effects isolated)
    │       ├── __init__.py
    │       ├── images.py               # Image loading/saving
    │       ├── video.py                # Video encoding
    │       ├── audio.py                # Audio handling
    │       └── subtitles.py            # Subtitle generation
    │
    ├── depth/                          # Depth estimation (CONSOLIDATED)
    │   ├── __init__.py
    │   ├── depth_anything_v2.py        # Only depth model we support
    │   └── depth_pipeline.py           # Depth estimation pipeline
    │
    ├── effects/                        # Visual effects (NEW)
    │   ├── __init__.py
    │   ├── shakify.py                  # Camera shake
    │   ├── noise.py                    # Noise effects
    │   ├── color.py                    # Color coherence
    │   └── sharpening.py               # Anti-blur
    │
    ├── wan/                            # Wan video generation (UNCHANGED)
    │   ├── __init__.py
    │   ├── integration.py              # Main integration (was: wan_simple_integration.py)
    │   ├── qwen.py                     # Prompt enhancement (was: qwen_prompt_expander.py)
    │   ├── pipelines/
    │   ├── configs/
    │   └── models/
    │
    ├── ui/                             # Gradio UI (CONSOLIDATED)
    │   ├── __init__.py
    │   ├── main.py                     # Main UI construction (from ui_left.py + ui_right.py)
    │   ├── tabs/                       # Individual tabs
    │   │   ├── __init__.py
    │   │   ├── run.py
    │   │   ├── keyframes.py
    │   │   ├── distribution.py
    │   │   ├── prompts.py
    │   │   ├── depth.py
    │   │   ├── shakify.py
    │   │   ├── wan.py
    │   │   └── output.py
    │   ├── components.py               # Reusable UI components
    │   └── callbacks.py                # Event handlers
    │
    ├── config/                         # Configuration (NEW)
    │   ├── __init__.py
    │   ├── args.py                     # Argument definitions (from args.py)
    │   ├── defaults.py                 # Default values (from defaults.py)
    │   └── settings.py                 # Settings persistence (from settings.py)
    │
    ├── orchestration/                  # High-level orchestration (side effects)
    │   ├── __init__.py
    │   └── runner.py                   # Main runner (from run_deforum.py)
    │
    └── external/                       # Third-party integrations (was: src/)
        ├── __init__.py
        ├── parseq.py                   # Parseq adapter
        ├── controlnet.py               # ControlNet integration
        └── vendors/                    # Third-party code (isolated)
            ├── adabins/
            ├── clipseg/
            └── film_interpolation/

tests/
├── unit/                               # Unit tests for pure functions
│   ├── core/
│   │   ├── test_keyframes.py
│   │   ├── test_schedules.py
│   │   └── test_prompts.py
│   └── utils/
│       ├── test_math.py
│       ├── test_images.py
│       └── test_frames.py
└── integration/                        # Integration tests (existing)
    ├── api_test.py
    └── postprocess_test.py
```

## Migration Strategy (Gradual, Safe)

### Phase 0: Preparation (CURRENT)
- ✅ Create refactoring branch
- ✅ Establish refactoring rules (REFACTORING_RULES.md)
- ✅ Set up unit test infrastructure
- ✅ Update documentation

### Phase 1: Extract Pure Functions (IN-PLACE)
**Goal:** Identify and isolate pure functions WITHOUT moving files

**Steps:**
1. **Audit existing code** for pure vs impure functions
2. **Within each file**, separate:
   - Pure functions (move to top of file, mark with comment)
   - Impure functions (keep at bottom)
3. **Add type hints** to all functions
4. **Add docstrings** to all functions
5. **Write unit tests** for pure functions
6. **Refactor to reduce complexity** (complexity ≤ 10)

**Example in `animation.py`:**
```python
# ============================================================================
# PURE FUNCTIONS (no side effects, testable)
# ============================================================================

def calculate_zoom_factor(zoom_schedule: str, frame: int) -> float:
    """Parse zoom schedule and return factor for given frame."""
    # Pure calculation - type hints are the documentation
    pass

def lerp(a: float, b: float, t: float) -> float:
    # No docstring needed - name + types are clear
    return a + (b - a) * t

def parse_schedule(schedule_str: str) -> dict[int, float]:
    """Parse "0:(1.0), 30:(1.5)" format into {0: 1.0, 30: 1.5}."""
    # Add docstring only because parsing format is non-obvious
    pass

# ============================================================================
# IMPURE FUNCTIONS (side effects: I/O, state, API calls)
# ============================================================================

def apply_zoom_to_image(image: np.ndarray, zoom: float, output_path: str) -> None:
    # Side effects here - name + types make it clear
    pass
```

**Focus areas for Phase 1:**
- `animation.py` - Extract motion calculations
- `prompt.py` - Extract prompt parsing
- `seed.py` - Extract seed generation
- `animation_key_frames.py` - Extract keyframe math
- `rendering/data/frame/*.py` - Extract frame calculations
- All `*_utils.py` files - Identify pure vs impure

**Deliverable:** Same file structure, but with pure functions isolated and tested

### Phase 2: Create New Pure Modules (GRADUAL)
**Goal:** Extract pure functions into new organized modules (one domain at a time)

**Steps:**
1. **Create `deforum/utils/` directory**
2. **Extract one utility domain at a time:**
   - Start with `deforum/utils/math.py` - Extract from all files
   - Then `deforum/utils/images.py` - Pure image conversions
   - Then `deforum/utils/frames.py` - Frame calculations
   - Then `deforum/utils/time.py` - Time/FPS conversions
3. **Update imports** incrementally
4. **Run tests** after each extraction
5. **Keep old files** until all imports updated

**Example:**
```python
# deforum/utils/math.py (NEW)
"""Pure mathematical utility functions."""

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))
```

**Deliverable:** New `deforum/utils/` with pure functions, old files still work

### Phase 3: Create Core Domain Modules (CAREFUL)
**Goal:** Extract core business logic into `deforum/core/`

**Steps:**
1. **Create `deforum/core/` directory**
2. **Extract one domain at a time:**
   - `deforum/core/keyframes.py` - From `animation_key_frames.py`
   - `deforum/core/schedules.py` - From `animation.py`
   - `deforum/core/prompts.py` - From `prompt.py`
   - `deforum/core/seeds.py` - From `seed.py`
3. **Keep old files as wrappers** initially
4. **Update imports** gradually
5. **Remove old files** only when fully migrated

**Deliverable:** Core domain logic in `deforum/core/`, tested and type-hinted

### Phase 4: Reorganize Rendering Pipeline (MAJOR)
**Goal:** Refactor rendering pipeline with clear separation

**Steps:**
1. **Create `deforum/rendering/` restructure:**
   - `pipeline.py` - Main loop
   - `transformations.py` - Image transformations
   - `frames/` - Frame system
   - `io/` - I/O operations
2. **Migrate one component at a time**
3. **Keep experimental_core.py** as entry point initially
4. **Update gradually**, test continuously

**Deliverable:** Clean rendering pipeline with isolated I/O

### Phase 5: Consolidate UI (CAREFUL)
**Goal:** Organize UI into clear structure

**Steps:**
1. **Create `deforum/ui/` directory**
2. **Extract tabs one at a time** from `ui_elements.py`
3. **Keep old UI files** until migration complete
4. **Test UI** after each tab migration

**Deliverable:** Organized UI with clear tab separation

### Phase 6: Rename Package (FINAL)
**Goal:** Rename `deforum_helpers/` → `deforum/`

**Steps:**
1. **Ensure all migrations complete**
2. **Update all imports** project-wide
3. **Rename directory** `deforum_helpers` → `deforum`
4. **Update documentation**
5. **Run full test suite**

**Deliverable:** Clean package name `deforum`

### Phase 7: Clean Up (POLISH)
**Goal:** Remove deprecated code, polish structure

**Steps:**
1. **Remove old files** that have been replaced
2. **Remove `src/` directory** (move to `external/vendors/`)
3. **Update all documentation**
4. **Run full test suite**
5. **Performance profiling**

**Deliverable:** Clean, tested, documented codebase

## Refactoring Priorities (Per Module)

For each file we refactor, follow this order:

1. **Add type hints** to all functions
2. **Add docstrings** (Google-style)
3. **Identify pure vs impure** functions
4. **Extract pure functions** to top of file
5. **Break down complex functions** (complexity ≤ 10)
6. **Extract magic numbers** to constants
7. **Write unit tests** for pure functions
8. **Add error handling**
9. **Format with Black**
10. **Lint with flake8**

## Success Metrics

### Code Quality
- **Complexity:** All functions ≤ 10 (measured with radon)
- **Type Coverage:** 100% of functions have type hints
- **Documentation:** 100% of functions have docstrings
- **Test Coverage:** 70%+ for pure functions

### Structure Quality
- **Pure Functions:** Clearly separated from impure
- **Utils:** Organized by domain, not "utils"
- **Nesting:** Max 3 levels deep
- **File Size:** No file > 500 lines

### Safety
- **Tests Pass:** All tests pass after each phase
- **No Breaking Changes:** Old imports work until migration complete
- **Gradual Migration:** One domain at a time
- **Rollback Capable:** Can revert any phase independently

## Current Focus: Phase 1

We are currently in **Phase 1: Extract Pure Functions (IN-PLACE)**

**Next Steps:**
1. Audit `animation.py` for pure vs impure functions
2. Add type hints and docstrings
3. Write unit tests for pure functions
4. Reduce complexity to ≤ 10

**DO NOT:**
- Move files around yet
- Rename `deforum_helpers`
- Reorganize directory structure
- Delete any files

**DO:**
- Extract pure functions within existing files
- Add type hints and tests
- Reduce complexity
- Document everything
