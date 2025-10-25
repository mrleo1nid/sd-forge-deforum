# GPU Test Patterns - Preserved Ideas

This document preserves useful patterns from the deleted GPU integration tests for use in future API test implementation.

## Why GPU Tests Were Removed

GPU integration tests had a fundamental design flaw: they required Forge's full environment (loaded models, shared state, etc.) but couldn't initialize it properly when run standalone. Simply importing Forge modules triggered the entire initialization sequence which conflicted with pytest.

**Solution:** Use API integration tests instead, which run within a launched Forge instance.

## Reusable Patterns for API Tests

### 1. Minimal Test Settings

Fast generation settings for quick tests:
```python
minimal_settings = {
    "W": 512,
    "H": 512,
    "steps": 4,
    "sampler": "euler",
    "seed": 42,
    "max_frames": 3,
    "animation_mode": "3D",
    "batch_name": "test"
}
```

**Why:** 3 frames at 512x512 with 4 steps generates in 30-60 seconds vs minutes for full tests.

### 2. VRAM Tracking Pattern

Useful for detecting model double-loading issues:
```python
import torch

torch.cuda.reset_peak_memory_stats()
vram_before = torch.cuda.max_memory_allocated() / 1024**3  # GB

# ... run generation ...

vram_peak = torch.cuda.max_memory_allocated() / 1024**3
vram_used = vram_peak - vram_before

print(f"[VRAM] Peak usage: {vram_peak:.2f} GB")
print(f"[VRAM] Incremental: {vram_used:.2f} GB")

# Expected values for Flux + ControlNet: ~12-15GB
# If >20GB: likely double-loading issue
if vram_used > 20:
    raise AssertionError(f"Excessive VRAM usage: {vram_used:.2f} GB")
```

**Adapt for API:** Add VRAM tracking endpoint or return in job status.

### 3. Isolated Output Directories

Each test gets its own output directory:
```python
@pytest.fixture
def isolated_output_dir(test_output_dir, request):
    """Create isolated output directory for each test."""
    test_name = request.node.name
    test_dir = test_output_dir / test_name
    test_dir.mkdir(exist_ok=True)
    return test_dir
```

**API version:** Include test name in `batch_name` to isolate outputs.

### 4. FLF2V Test Ideas

#### Color Interpolation Test
```python
def test_flf2v_color_interpolation():
    """Test FLF2V interpolates correctly between two solid colors."""
    # Create red → blue transition
    first_frame = Image.new('RGB', (512, 512), color=(255, 0, 0))
    last_frame = Image.new('RGB', (512, 512), color=(0, 0, 255))

    # Generate 9 frames (4n+1 format)
    # result = generate_flf2v(first_frame, last_frame, num_frames=9)

    # Verify color progression
    # - First frame: mostly red (R>200, B<50)
    # - Last frame: mostly blue (R<50, B>200)
    # - Middle frame: transition (50<R<200, 50<B<200)
```

**API version:** POST two keyframe images, verify interpolated video shows smooth transition.

#### Shape Transformation Test
```python
def test_flf2v_shape_transformation():
    """Test FLF2V interpolates between different shapes."""
    # Create circle → square transformation
    first_frame = create_circle_image(512, 512)
    last_frame = create_square_image(512, 512)

    # Generate interpolation
    # Verify all frames have shape present (white pixel count > threshold)
    # Verify frames are different (actual morphing happening)
```

**API version:** Similar but via API endpoints.

### 5. Flux ControlNet Test Ideas

#### Basic Generation with ControlNet
```python
def test_flux_controlnet_basic():
    """Test Flux ControlNet generates frames correctly."""
    settings = {
        **minimal_settings,
        "enable_flux_controlnet_v2": True,
        "flux_controlnet_strength": 0.8,
        "flux_controlnet_model": "depth",
        "flux_guidance_scale": 3.5,
        "prompts": {0: "a mountain landscape"}
    }

    # Track VRAM to detect double-loading
    # Expected: Flux ~8-12GB + ControlNet ~3.6GB = <15GB total

    # Verify:
    # - 3 frames generated
    # - No errors
    # - VRAM usage reasonable
```

**API version:** POST settings, wait for completion, check output frames and VRAM stats.

#### Model Loading Test
```python
def test_flux_controlnet_model_loading():
    """Verify ControlNet model loads with correct memory footprint."""
    # Expected ControlNet VRAM: ~3.6GB (±1GB tolerance)
    # This helps detect if model is loading correctly vs broken
```

**API version:** Add endpoint to query loaded models and their VRAM usage.

### 6. Test Markers

Keep these markers for API tests:
```python
@pytest.mark.slow  # For tests that generate actual outputs (minutes)
@pytest.mark.flux  # Requires Flux model
@pytest.mark.wan   # Requires Wan model
@pytest.mark.controlnet  # Tests ControlNet functionality
```

### 7. Snapshot Testing Pattern

For regression testing (verify outputs don't change):
```python
def test_with_snapshots(snapshot):
    """Use pytest-syrupy for snapshot testing."""
    result = generate_video(settings)

    # Compare generated frames against stored snapshots
    assert result.frames == snapshot
```

**Already used in:** `tests/integration/api_test.py` for SRT files.

## API Test Implementation Plan

### Phase 1: Fix & Document Existing API
- [ ] Add Swagger/OpenAPI documentation
- [ ] Test all existing endpoints
- [ ] Fix any broken endpoints
- [ ] Document API in README

### Phase 2: Add New Endpoints
- [ ] `GET /deforum_api/stats/vram` - VRAM usage tracking
- [ ] `GET /deforum_api/models` - List loaded models
- [ ] `POST /deforum_api/flf2v` - Direct FLF2V generation
- [ ] Flux/Wan specific endpoints

### Phase 3: Comprehensive Tests
- [ ] Minimal settings tests (3 frames, fast)
- [ ] Flux ControlNet tests
- [ ] FLF2V interpolation tests (color, shape)
- [ ] VRAM tracking tests
- [ ] Snapshot regression tests
- [ ] Error handling tests

### Phase 4: Performance Tests
- [ ] Benchmark different settings
- [ ] Track VRAM usage across runs
- [ ] Identify performance regressions

## Key Learnings

1. **Don't try to initialize Forge outside of its natural launch sequence** - Too many side effects and global state
2. **API tests are the right approach** - They run within a launched Forge instance with everything properly initialized
3. **Minimal settings make tests fast** - 3 frames at 512x512 with 4 steps is the sweet spot
4. **VRAM tracking catches bugs** - Especially model double-loading issues
5. **Isolated outputs prevent conflicts** - Use unique batch names per test
