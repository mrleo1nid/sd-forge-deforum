# Testing Strategy

## Current State (As of Oct 2025)

### What We Have: API Integration Tests

The existing test suite in `tests/` consists entirely of **end-to-end API integration tests**, not unit tests:

```
tests/
├── deforum_test.py              # 5 API integration tests (2D, 3D, Parseq, cancellation)
├── deforum_postprocess_test.py  # 5 post-processing tests (FILM, RIFE, upscaling)
├── conftest.py                  # pytest fixture for server startup
├── utils.py                     # API client utilities
└── testdata/                    # Test fixtures
```

**How they work:**
1. Start full Forge WebUI server (via `conftest.py`)
2. Submit jobs via Deforum API (`/deforum_api/batches`)
3. Wait for job completion
4. Validate output video properties (FPS, dimensions, frame count)
5. Compare generated `.srt` files against snapshots (regression testing)

**What they test:**
- ✅ Full pipeline: API → Settings parsing → Rendering → Video output
- ✅ 2D and 3D animation modes
- ✅ Parseq integration
- ✅ Post-processing (FILM, RIFE, RealESRGAN upscaling)
- ✅ Job cancellation

**What they DON'T test:**
- ❌ Individual functions/classes in isolation
- ❌ Edge cases and error handling
- ❌ New features (Flux, Wan, experimental core, keyframe distribution)
- ❌ Performance characteristics

### Empty Directories

These directories exist but contain no tests:
- `tests/unit/` - Intended for unit tests (none exist)
- `tests/functional/` - Empty
- `tests/integration/` - Empty
- `tests/performance/` - Empty

## Problems with Current Tests

### 1. **Designed for A1111, Not Forge**

The CI workflow in `.github/workflows/run_tests.yaml.disabled` (now disabled) was set up to test on **AUTOMATIC1111's WebUI v1.6.0**, not Forge:

```yaml
- name: Checkout a1111
  uses: actions/checkout@v3
  with:
    repository: AUTOMATIC1111/stable-diffusion-webui
    ref: v1.6.0
```

This is a fundamental mismatch - we're a Forge extension, not an A1111 extension.

### 2. **Missing Test Coverage for New Features**

No tests exist for major new features:
- Flux/Wan video generation mode
- Experimental render core (now the only core)
- Keyframe distribution system
- Qwen prompt expansion
- Camera Shakify integration

### 3. **Heavyweight Setup Required**

Running tests requires:
- Full Forge WebUI installation
- GPU (for most tests - they're marked `@pytest.mark.skipif(gpu_disabled())`)
- Model downloads (checkpoints, VAE, etc.)
- Server startup time (~30-60 seconds)

This makes tests slow and hard to run during development.

### 4. **No Isolation**

Tests interact with the full system:
- File system (writes videos, frames, SRT files)
- GPU memory
- Global state

Failures are hard to diagnose because any part of the pipeline could be at fault.

## What We Need: Unit Tests

### Key Components That Need Unit Tests

Based on the new architecture (see `CLAUDE.md`), these are the critical systems that should have unit tests:

#### 1. **Keyframe Distribution (`scripts/deforum_helpers/rendering/data/frame/`)**
```python
# Example unit tests needed:
def test_distribute_keyframes_basic():
    """Test basic keyframe distribution with simple prompt schedule"""

def test_distribute_keyframes_with_parseq():
    """Test distribution when Parseq data is present"""

def test_calculate_tween_weights():
    """Test interpolation weight calculation between keyframes"""
```

#### 2. **Argument Parsing (`scripts/deforum_helpers/args.py`, `run_deforum.py`)**
```python
def test_process_args_basic():
    """Test basic argument processing from UI components"""

def test_process_args_flux_wan_mode():
    """Test argument processing for Flux/Wan mode"""

def test_process_args_invalid_input():
    """Test error handling for invalid inputs"""
```

#### 3. **Prompt Scheduling (`scripts/deforum_helpers/prompt.py`)**
```python
def test_prepare_prompt_basic():
    """Test prompt preparation with simple schedule"""

def test_prepare_prompt_interpolation():
    """Test prompt interpolation between keyframes"""
```

#### 4. **Wan Integration (`scripts/deforum_helpers/wan/`)**
```python
def test_calculate_wan_frame_count():
    """Test 4n+1 frame count calculation"""

def test_qwen_prompt_expansion():
    """Test Qwen prompt enhancement (mock model calls)"""

def test_wan_i2v_chaining_setup():
    """Test I2V chaining configuration"""
```

#### 5. **Animation Calculations (`scripts/deforum_helpers/animation.py`)**
```python
def test_anim_frame_warp_2d():
    """Test 2D transformation matrix calculation"""

def test_anim_frame_warp_3d():
    """Test 3D depth warping calculation"""
```

### Unit Test Best Practices

**Good unit tests should:**
- ✅ Test ONE function/class in isolation
- ✅ Run fast (milliseconds, not seconds)
- ✅ Not require GPU, network, or file I/O
- ✅ Use mocks/fixtures for dependencies
- ✅ Test edge cases and error conditions
- ✅ Be deterministic (no randomness unless testing randomness)

**Example structure:**
```python
# tests/unit/test_keyframe_distribution.py
import pytest
from scripts.deforum_helpers.rendering.data.frame.key_frame_distribution import (
    distribute_keyframes, calculate_tween_weights
)

class TestKeyframeDistribution:
    def test_distribute_keyframes_evenly_spaced(self):
        """When prompts are evenly spaced, keyframes should match exactly"""
        prompts = {0: "prompt1", 10: "prompt2", 20: "prompt3"}
        max_frames = 21

        keyframes = distribute_keyframes(prompts, max_frames)

        assert len(keyframes) == 3
        assert keyframes[0].frame_number == 0
        assert keyframes[1].frame_number == 10
        assert keyframes[2].frame_number == 20

    def test_calculate_tween_weights_midpoint(self):
        """Tween at midpoint should have 0.5 weight from each neighbor"""
        prev_keyframe_num = 0
        next_keyframe_num = 10
        tween_num = 5

        prev_weight, next_weight = calculate_tween_weights(
            prev_keyframe_num, next_keyframe_num, tween_num
        )

        assert prev_weight == pytest.approx(0.5)
        assert next_weight == pytest.approx(0.5)
```

## Integration Tests (Keep Some)

While unit tests are ideal, we should keep a **small set** of integration tests for critical workflows:

```python
# tests/integration/test_end_to_end.py
def test_simple_2d_animation():
    """Minimal 2D animation - smoke test that basic pipeline works"""

def test_flux_wan_t2v():
    """Flux/Wan text-to-video generation"""

def test_keyframe_distribution_with_experimental_core():
    """Experimental core with keyframe distribution"""
```

These should:
- Use small frame counts (5-10 frames)
- Use tiny resolutions (256x256)
- Test the most critical paths
- Run on CI only (not during development)

## Running Tests

### Current Setup (API Integration Tests)

**Requirements:**
```bash
pip install -r requirements-dev.txt
# Installs: coverage, syrupy, pytest, tenacity, pydantic_requests, moviepy
```

**Run all tests:**
```bash
# Option 1: Auto-start server
cd ~/workspace/stable-diffusion-webui-forge
pytest extensions/sd-forge-deforum/tests/ --start-server

# Option 2: Start server manually first
cd ~/workspace/stable-diffusion-webui-forge
python webui.py --deforum-api
# Then in another terminal:
cd extensions/sd-forge-deforum
pytest tests/
```

**Run specific test:**
```bash
pytest tests/deforum_test.py::test_simple_settings -v
```

### Future Setup (Unit Tests)

Once unit tests exist:
```bash
# Run unit tests only (fast - no server needed)
pytest tests/unit/

# Run all tests
pytest tests/
```

## Next Steps

### Short-term (Stabilize Current Tests)
1. ✅ Remove obsolete hybrid video test (DONE)
2. Document how to run existing tests on Forge
3. Verify which existing tests still pass

### Medium-term (Build Unit Test Foundation)
1. Create `tests/unit/test_keyframe_distribution.py`
2. Create `tests/unit/test_args.py`
3. Create `tests/unit/test_prompt.py`
4. Add unit tests for new Wan integration code
5. Set up pytest coverage reporting

### Long-term (Comprehensive Coverage)
1. Achieve 70%+ code coverage with unit tests
2. Keep 3-5 lightweight integration tests for smoke testing
3. Set up CI to run unit tests on every commit
4. Set up CI to run integration tests nightly or on PR merge

## Files Reference

- Test suite: `tests/deforum_test.py:1`
- Post-processing tests: `tests/deforum_postprocess_test.py:1`
- Pytest config: `tests/conftest.py:1`
- Disabled CI workflow: `.github/workflows/run_tests.yaml.disabled:1`
- Keyframe distribution code: `scripts/deforum_helpers/rendering/data/frame/key_frame_distribution.py:1`
- Argument processing: `scripts/deforum_helpers/run_deforum.py:43`
