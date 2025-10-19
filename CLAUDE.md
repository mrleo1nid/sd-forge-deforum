# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**sd-forge-deforum** is an experimental fork of the Deforum extension for Stable Diffusion WebUI Forge that generates frame-precise animated videos using keyframe scheduling. This fork adds:
- **Flux.1 support** for state-of-the-art image generation
- **Wan 2.1 AI Video Generation** (Alibaba's text-to-video model) with Deforum scheduling integration
- **Parseq keyframe redistribution** for intelligent frame placement
- **Camera Shakify integration** for realistic camera shake effects from Blender data
- **QwenPromptExpander** for AI-powered prompt enhancement and movement analysis

The extension operates as a Forge extension that hooks into WebUI's script system to provide animation capabilities across 5 modes: 2D, 3D, Video Input, Interpolation, and Wan Video.

## Running the Extension

### Prerequisites

Must be installed in a working Stable Diffusion WebUI Forge installation. See parent repository CLAUDE.md for Forge setup.

### Testing the Extension

**Quick test with default settings:**
```bash
# From Forge webui directory, launch normally
python launch.py
# Or if environment already prepared:
python webui.py
```

Then in the UI:
1. Navigate to the Deforum tab
2. Set "Distribution" to "Keyframes Only"
3. Set "Animation Mode" to "3D"
4. Click generate to run the default bunny test (333 frames, 19 keyframes, 720p at 60 FPS)

**Run with Deforum API enabled:**
```bash
python webui.py --deforum-api
```

**Run tests:**
```bash
# From extension directory
pytest tests/ --start-server
# Or if server already running:
pytest tests/
```

**Install/update dependencies:**
```bash
cd extensions/sd-forge-deforum
pip install -r requirements.txt
```

## Architecture

### Entry Points and Flow

1. **Extension Loading** (`preload.py:18`)
   - Registers CLI arguments: `--deforum-api`, `--deforum-simple-api`, `--deforum-run-now`, `--deforum-terminate-after-run-now`
   - Called by Forge before main initialization

2. **Extension Initialization** (`scripts/deforum.py:24`)
   - Extends Python path with `deforum_sys_extend()`
   - Applies diffusers compatibility patches for Forge integration
   - Creates `Models/Deforum/` directory for model downloads
   - Registers UI tabs via `script_callbacks.on_ui_tabs(on_ui_tabs)`
   - Registers settings via `script_callbacks.on_ui_settings(on_ui_settings)`

3. **UI Creation** (`scripts/deforum_helpers/ui_right.py:28`)
   - Builds Gradio interface with tabs for Prompts, Keyframes, Output, Wan Video, etc.
   - Returns tuple: `(deforum_interface, "Deforum", "deforum")`

4. **Main Orchestrator** (`scripts/deforum_helpers/run_deforum.py:43`)
   - `run_deforum(*args)` is the primary entry point when user clicks generate
   - Parses component arguments from UI into structured objects
   - Detects Wan Video mode to skip SD model loading if not needed
   - Processes arguments via `process_args()` into `args`, `anim_args`, `video_args`, `parseq_args`, etc.
   - Routes to appropriate rendering pipeline based on animation mode

5. **Rendering Pipelines**
   - **Standard modes (2D/3D/Video/Interpolation):** `scripts/deforum_helpers/rendering/experimental_core.py:22` - `render_animation()`
   - **Wan Video mode:** `scripts/deforum_helpers/wan/wan_simple_integration.py:29` - `generate_wan_video()`

### Core Systems

**1. Keyframe Distribution System** (`scripts/deforum_helpers/rendering/data/frame/`)
- **Purpose:** Intelligently places keyframes across animation timeline without cadence
- **Key Files:**
  - `key_frame_distribution.py` - Distribution algorithms
  - `diffusion_frame.py` - Frame metadata and state
  - `diffusion_frame_data.py` - Collection of all frames
  - `tween_frame.py` - Interpolated frames between keyframes
- **Integration:** Works with or without Parseq for precise timing

**2. Animation Pipeline** (`scripts/deforum_helpers/rendering/experimental_core.py`)
- Central render loop that:
  1. Creates `RenderData` object (central state container)
  2. Generates subtitle .srt file asynchronously
  3. Iterates through frames calling `generate_inner()`
  4. Applies transformations (2D/3D movement, depth warping)
  5. Handles hybrid video, masks, and noise schedules
  6. Stitches final video with ffmpeg

**3. Wan Video Pipeline** (`scripts/deforum_helpers/wan/`)
- **wan_simple_integration.py:29** - Main Wan video generation
  - Auto-discovers models from `models/wan/` directory
  - Handles T2V (text-to-video) and I2V (image-to-video) chaining
  - Calculates frame counts as 4n+1 per Wan requirements
  - Integrates Deforum prompt scheduling, FPS, seed, and strength
- **qwen_prompt_expander.py** - AI prompt enhancement
  - Auto-selects Qwen model (3B/7B/14B) based on VRAM
  - Analyzes Deforum movement schedules, translates to English
  - Lazy-loads models only when "Enhance Prompts" clicked
  - Auto-cleanup before video generation to free VRAM

**4. Central State** (`scripts/deforum_helpers/rendering/data/render_data.py:42`)
- `RenderData` class holds all state during rendering:
  - Frame metadata, animation keys, schedules
  - Depth models, masks, images
  - Parseq integration data
  - Camera shake patterns
  - Progress tracking

**5. Depth Estimation** (`scripts/deforum_helpers/depth*.py`)
- Multiple backends: MiDaS, Depth-Anything V2, AdaBins, LeReS, ZoeDepth
- Used for 3D mode to warp frames based on estimated depth
- Models auto-download to `models/Deforum/` on first use

**6. Camera Shakify** (`scripts/deforum_helpers/rendering/data/shakify/`)
- Pre-recorded camera shake patterns from Blender
- Patterns: EARTHQUAKE, FILM_GRAIN, GENTLE_HANDHELD, INVESTIGATION, etc.
- Applied on top of scheduled movement transforms
- Data sourced from EatTheFuture's Camera Shakify Blender plugin (CC0 license)

**7. Parseq Integration** (`scripts/deforum_helpers/parseq_adapter.py`)
- Adapter pattern to integrate Parseq keyframe data
- Translates between Parseq JSON format and Deforum's internal structures
- Enables complex scheduling with GUI-based keyframe editor

### Directory Structure

```
scripts/
├── deforum.py                          # Main extension script (init)
├── deforum_api.py                      # REST API endpoints
├── deforum_api_models.py               # API data models
├── deforum_controlnet.py               # ControlNet integration
├── default_settings.txt                # Default configuration template
└── deforum_helpers/                    # Core implementation
    ├── run_deforum.py                  # Main orchestrator
    ├── args.py                         # Argument parsing
    ├── defaults.py                     # Default values
    ├── ui_right.py                     # UI construction
    ├── ui_settings.py                  # Settings tab
    ├── prompt.py                       # Prompt scheduling
    ├── animation.py                    # Animation calculations
    ├── depth*.py                       # Depth estimation backends
    ├── parseq_adapter.py               # Parseq integration
    ├── wan/                            # Wan video generation
    │   ├── wan_simple_integration.py   # Main Wan pipeline
    │   ├── qwen_prompt_expander.py     # AI prompt enhancement
    │   └── ...
    ├── rendering/                      # Rendering pipeline
    │   ├── experimental_core.py        # Main render loop
    │   ├── data/                       # Data structures
    │   │   ├── render_data.py          # Central state
    │   │   ├── frame/                  # Frame systems
    │   │   ├── shakify/                # Camera shake data
    │   │   └── subtitle/               # Subtitle generation
    │   └── util/                       # Rendering utilities
    └── src/                            # Third-party code
        ├── adabins/                    # AdaBins depth model
        └── clipseg/                    # CLIPSeg segmentation

tests/
├── conftest.py                         # Pytest configuration
├── deforum_test.py                     # Main test suite
└── utils.py                            # Test utilities

preload.py                              # CLI argument registration
requirements.txt                        # Python dependencies
pytest.ini                              # Pytest settings
```

### Key Concepts

**Animation Modes:**
1. **2D** - Flat transformations (pan, zoom, rotate)
2. **3D** - Depth-based warping with camera movement
3. **Video Input** - Use existing video as initialization
4. **Interpolation** - Generate between two prompts
5. **Wan Video** - AI video generation with Deforum scheduling

**Keyframe Distribution:**
- Replaces traditional cadence-based rendering
- Intelligently places diffusion keyframes at prompt boundaries
- Interpolates (tweens) non-keyframes from nearest keyframes
- Reduces diffusion steps while maintaining quality

**Experimental Render Core:**
- Activated when keyframe distribution is enabled
- Incompatible with some features (Kohya HR Fix, FreeU)
- Provides better synchronization and less jitter at high/no cadence

**Wan Integration Points:**
- Prompts from "Prompts" tab with frame numbers
- FPS from "Output" tab
- Optional seed scheduling from "Keyframes → Seed" tab
- Optional strength scheduling from "Keyframes → Strength" tab for I2V chaining
- Configuration and enhancement in "Wan Video" tab

**I2V Chaining (Wan):**
- Uses last frame of previous clip as initialization for next clip
- VACE models (Video Adaptive Conditional Enhancement) recommended for best continuity
- Strength schedule controls how much previous frame influences next clip
- Automatically handles 4n+1 frame requirements per Wan spec

## Common Development Tasks

### Adding a New Animation Mode

1. Add enum to `scripts/deforum_helpers/rendering/data/anim/animation_mode.py`
2. Add UI option in `scripts/deforum_helpers/ui_right.py`
3. Implement pipeline in new file under `scripts/deforum_helpers/rendering/`
4. Route from `run_deforum.py` based on `anim_args.animation_mode`

### Adding a New Depth Model

1. Create `scripts/deforum_helpers/depth_<model>.py` following existing pattern
2. Implement `predict()` function returning depth map
3. Register in `scripts/deforum_helpers/depth.py`
4. Add UI option for model selection

### Modifying Keyframe Distribution

Edit `scripts/deforum_helpers/rendering/data/frame/key_frame_distribution.py`
- `distribute_keyframes()` - Main distribution algorithm
- `calculate_tween_weights()` - Interpolation between keyframes

### Adding Wan Features

- Model handling: `scripts/deforum_helpers/wan/wan_model_manager.py`
- Generation logic: `scripts/deforum_helpers/wan/wan_simple_integration.py`
- UI components: `scripts/deforum_helpers/wan/wan_ui_components.py`
- Qwen enhancement: `scripts/deforum_helpers/wan/qwen_prompt_expander.py`

### Extending UI

All UI code in `scripts/deforum_helpers/ui_right.py:28` in `on_ui_tabs()`
- Uses Gradio 4 components
- Returns `(interface, "Deforum", "deforum")` tuple
- Settings tab in `ui_settings.py`

## Important Patterns

**Argument Structure:**
Arguments flow as: Raw args → `process_args()` → Structured namespaces
- `args` - General settings (seed, sampler, steps, cfg_scale, etc.)
- `anim_args` - Animation settings (animation_mode, max_frames, etc.)
- `video_args` - Video output settings (fps, codec, etc.)
- `parseq_args` - Parseq integration settings
- `wan_args` - Wan video settings

**Frame Processing:**
Each frame goes through: Prompt → Seed → Denoise strength → Transformation → Depth → Output
- Keyframes: Full diffusion generation
- Tween frames: Interpolated from neighboring keyframes (experimental core only)

**Model Loading:**
- Wan mode skips SD/Flux model loading entirely (check in `run_deforum.py:52`)
- Qwen models lazy-load only when enhancement requested
- Depth models auto-download on first use to `models/Deforum/`

**Error Handling:**
- Use `JobStatusTracker().update_phase()` for progress
- Use `JobStatusTracker().fail_job()` for errors
- Rich console output with color codes from `rendering/util/log_utils.py`

## Testing

**Run all tests:**
```bash
pytest tests/ --start-server
```

**Run specific test:**
```bash
pytest tests/deforum_test.py::test_name -v
```

**Test configuration:**
- `tests/conftest.py` - Server startup fixture
- `pytest.ini` - Pytest settings (filters deprecation warnings)
- Tests use Deforum API endpoints at `http://localhost:7860/deforum_api/`

**Manual testing workflow:**
1. Launch Forge with `python webui.py`
2. Navigate to Deforum tab
3. Load `scripts/default_settings.txt`
4. Modify settings as needed
5. Click generate
6. Check console output for errors
7. Review generated video in output directory

## File References

When discussing code, use `file_path:line_number` format:
- Extension init: `scripts/deforum.py:24`
- Main orchestrator: `scripts/deforum_helpers/run_deforum.py:43`
- Render pipeline: `scripts/deforum_helpers/rendering/experimental_core.py:22`
- Central state: `scripts/deforum_helpers/rendering/data/render_data.py:42`
- Wan integration: `scripts/deforum_helpers/wan/wan_simple_integration.py:29`
- Qwen enhancement: `scripts/deforum_helpers/wan/qwen_prompt_expander.py:1`
- UI creation: `scripts/deforum_helpers/ui_right.py:28`
- Keyframe distribution: `scripts/deforum_helpers/rendering/data/frame/key_frame_distribution.py:1`

## Dependencies

Core dependencies (from `requirements.txt`):
- `numexpr`, `matplotlib`, `pandas` - Math and plotting
- `av`, `pims`, `imageio_ffmpeg` - Video processing
- `rich` - Console output formatting
- `gdown` - Google Drive downloads
- `easydict` - Dictionary utilities
- `diffusers` (git main) - Hugging Face diffusers library
- `transformers>=4.36.0,<4.46.0` - For Wan and Qwen models
- `accelerate>=0.25.0,<0.31.0` - For model acceleration

**Model Requirements:**
- **Flux:** Requires `flux1-dev-bnb-nf4-v2.safetensors` and VAE files (see README.md)
- **Wan:** Downloaded via `huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan`
- **Qwen:** Auto-downloaded to `models/qwen/` when first used (3B/7B/14B variants)
- **Depth:** Auto-downloaded to `models/Deforum/` on first use per selected model

## Known Limitations with Experimental Core

When keyframe distribution is enabled (experimental render core):
- **Kohya HR Fix** - May need to be disabled
- **FreeU** - May need to be disabled
- **ControlNet** - Currently not working
- **Hybrid Video** - Untested
- **Flux Schnell** - Limited precision with only 4 steps

## Troubleshooting

**Import errors after installation:**
Restart WebUI completely: `Ctrl+C` then `python launch.py`

**Settings not loading correctly:**
Download latest `scripts/default_settings.txt` from repo and load in UI

**Wan models not found:**
```bash
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan
```

**Qwen enhancement fails:**
- Check VRAM available
- Use "Cleanup Qwen Cache" button in UI
- Select smaller model (3B instead of 7B/14B)
- Check console for error messages

**Generation fails with experimental core:**
- Disable Kohya HR Fix
- Disable FreeU
- Ensure keyframes align with prompt frame numbers

**Out of memory:**
- Reduce resolution
- Use quantized Flux model (bnb-nf4)
- Reduce max_frames or frame count
- For Wan: Use 1.3B instead of 14B model
