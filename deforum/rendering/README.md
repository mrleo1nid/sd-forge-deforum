# Deforum Rendering Pipeline

This directory contains the core rendering engine for Deforum animations. If you're looking to customize frame generation, add new rendering modes, or modify the animation pipeline, this is where you'll spend most of your time.

---

## 🎯 Core Files for Customization

### 1. **core.py** - Main Render Loop

**Purpose:** The heart of Deforum - orchestrates frame generation, transformations, and video output.

**Location:** `deforum/rendering/core.py:40`

**Key Function:**
```python
def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
```

**What it does:**
- Iterates through frames based on keyframe distribution or cadence
- Applies 2D/3D transformations and depth warping
- Manages noise injection and color coherence
- Generates subtitle .srt files asynchronously
- Stitches final video with ffmpeg

**When to modify:**
- Custom render logic or frame processing
- New animation transformation techniques
- Different output formats beyond video
- Custom progress tracking or callbacks
- Integration with new diffusion backends

**Key sections:**
- **Frame iteration:** Lines ~60-200 - Main render loop with keyframe distribution
- **Transformations:** Calls to `anim_frame_warp()` for 2D/3D movement
- **Noise application:** `add_noise()` for noise schedules
- **Subtitle generation:** Async thread for `.srt` file creation
- **Video stitching:** ffmpeg integration for final output

---

### 2. **img_2_img_tubes.py** - Functional IMG2IMG Pipeline

**Purpose:** Pure functional img2img pipeline for frame-to-frame generation.

**Location:** `deforum/rendering/img_2_img_tubes.py:1`

**Key Functions:**
```python
def img2img_function_with_packets(data, frame, turbo_steps, turbo_prev_image):
def populate_img2img_request_dict(data):
```

**What it does:**
- Wraps Forge's img2img processing in a functional interface
- Handles turbo mode (frame-to-frame diffusion)
- Builds request dictionaries for diffusion backend
- Manages init images, masks, and sampling parameters

**When to modify:**
- Custom img2img behavior
- New sampling techniques or schedulers
- Alternative diffusion backends
- Custom conditioning or ControlNet integration
- Frame-to-frame coherence techniques

**Design pattern:**
- **Pure functions** where possible - same inputs → same outputs
- **Packet-based data flow** - `RenderData` containers passed through pipeline
- **Separation of concerns** - Building requests vs executing diffusion

---

### 3. **wan_flux.py** - Flux + Wan Hybrid Mode

**Purpose:** Flux keyframe generation + Wan FLF2V (first-last-frame-to-video) interpolation.

**Location:** `deforum/rendering/wan_flux.py:1`

**Key Function:**
```python
def render_wan_flux(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
```

**Pipeline:**
1. **Phase 1:** Generate keyframes with Flux (high-quality diffusion)
2. **Phase 2:** Interpolate tweens with Wan FLF2V (AI video interpolation)
3. **Phase 3:** Stitch final video with ffmpeg

**When to modify:**
- Different keyframe generation backends
- Alternative interpolation methods
- Custom AI video model integration
- Hybrid workflows combining multiple models

**Key settings:**
- `guidance_scale` (default 3.5) - Controls Wan interpolation strength
- `prompt_mode` - How prompts are used during interpolation
- I2V chaining with strength schedules for continuity

---

## 📁 Directory Structure

```
deforum/rendering/
├── core.py                      # Main render loop (THE render core)
├── img_2_img_tubes.py           # Functional img2img pipeline
├── wan_flux.py                  # Flux/Wan hybrid mode
├── noise.py                     # Noise application (white/Perlin)
├── options.py                   # Rendering options management
├── helpers/                     # Helper modules (organized)
│   ├── depth.py                 # Depth estimation helpers
│   ├── filename.py              # Filename generation
│   ├── flux_controlnet.py       # Flux ControlNet integration
│   ├── image.py                 # Image processing helpers
│   ├── memory.py                # GPU memory management
│   ├── subtitle.py              # Subtitle generation
│   ├── turbo.py                 # Turbo mode helpers
│   └── webui.py                 # WebUI integration (progress, status)
├── data/                        # Data structures
│   ├── render_data.py           # RenderData - central state container
│   ├── schedule.py              # Schedule management
│   ├── mask.py                  # Mask handling
│   ├── images.py                # Image data structures
│   ├── frame/                   # Frame-related data
│   │   ├── diffusion_frame.py   # DiffusionFrame - keyframe metadata
│   │   ├── tween_frame.py       # Tween - interpolated frame
│   │   ├── key_frame_distribution.py  # Distribution algorithms
│   │   └── diffusion_frame_data.py    # Collection of all frames
│   ├── subtitle/                # Subtitle generation
│   │   └── srt.py               # SRT file generation
│   ├── shakify/                 # Camera shake data
│   │   └── shaker.py            # Shake pattern application
│   ├── taqaddumat.py            # Progress bar helper
│   └── anim/                    # Animation enums
│       └── animation_mode.py    # Animation mode definitions
└── calls/                       # Functional call wrappers
    ├── anim.py                  # Animation transformation calls
    ├── gen.py                   # Generation calls
    ├── images.py                # Image processing calls
    ├── mask.py                  # Mask processing calls
    └── subtitle.py              # Subtitle generation calls
```

---

## 🔧 Common Customization Scenarios

### Adding a New Animation Mode

1. **Define the mode** in `deforum/rendering/data/anim/animation_mode.py`
2. **Add UI controls** in `deforum/ui/ui_elements.py`
3. **Implement the pipeline** in a new file (e.g., `custom_mode.py`)
4. **Route from orchestrator** in `deforum/orchestration/render.py`

### Modifying Frame Transformations

**File:** `deforum/animation/animation.py` - `anim_frame_warp()`

Transformations are applied per-frame based on schedules:
- 2D: Translation, rotation, zoom
- 3D: Translation, rotation with depth warping
- Optical flow for motion-based warping

### Custom Keyframe Distribution

**File:** `deforum/rendering/data/frame/key_frame_distribution.py`

Algorithms for intelligent keyframe placement:
- `OFF` - Traditional cadence-based
- `KEYFRAMES_ONLY` - Only diffuse keyframes, interpolate tweens
- `ADDITIVE` - Additional keyframes at prompt boundaries
- `REDISTRIBUTED` - Optimal keyframe placement

### Integrating a New Diffusion Backend

1. **Wrap in functional interface** (see `img_2_img_tubes.py` pattern)
2. **Build request dictionary** with your backend's parameters
3. **Call from render loop** in `core.py`
4. **Handle init images** and masks appropriately

---

## 💡 Design Principles

### Functional Programming Patterns

- **Pure functions** where possible (see `CODING_GUIDE.md`)
- **Immutable data structures** - Return new objects, don't modify
- **Explicit dependencies** - Function signatures show what's needed
- **Separation of concerns** - Logic vs side effects

### Central State: RenderData

All rendering state flows through `RenderData` objects:
- Frame metadata
- Animation schedules
- Depth models, masks, images
- Parseq integration data
- Camera shake patterns
- Progress tracking

See `deforum/rendering/data/render_data.py:42`

### Keyframe vs Tween Frames

- **Keyframes** (`DiffusionFrame`) - Fully diffused with img2img
- **Tweens** (`Tween`) - Interpolated from neighboring keyframes
- Distribution algorithms decide which frames are which

---

## 🐛 Debugging Tips

### Enable Verbose Logging

```python
from deforum.utils.system.logging import log as log_utils
log_utils.debug("Your debug message here")
```

### Inspect RenderData

Add breakpoints or prints to inspect the `data` object:
```python
print(f"Frame {data.frame_idx}/{data.args.anim_args.max_frames}")
print(f"Keyframe: {data.is_keyframe}")
print(f"Prompt: {data.current_prompt}")
```

### Check Frame Files

Frames are saved to `{outdir}/{timestring}_*.png`
- Inspect individual frames for issues
- Check depth maps if depth warping is enabled
- Verify masks are applied correctly

### Subtitle Debug

The `.srt` file contains frame-by-frame metadata:
- Prompt used for that frame
- Animation parameters
- Useful for tracking what happened when

---

## 📚 Related Documentation

- **CODING_GUIDE.md** - Coding standards and best practices
- **CLAUDE.md** - AI assistant instructions and architecture overview
- **README.md** - User-facing documentation and feature list

---

## 🤝 Contributing

When modifying render code:

1. **Follow functional patterns** (see `CODING_GUIDE.md`)
2. **Add type hints** to all functions
3. **Keep complexity ≤ 10** (McCabe)
4. **Test with different modes** (3D, Flux/Wan, Interpolation)
5. **Update this README** if adding new core files

**Questions?** Open an issue on GitHub or check the documentation.

---

**Last Updated:** October 2025
**Maintainers:** See CONTRIBUTING.md
