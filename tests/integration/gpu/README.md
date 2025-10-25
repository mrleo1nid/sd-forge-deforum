# GPU Integration Tests

Local GPU integration tests that run Deforum directly without requiring the API server.

## Overview

These tests verify Deforum functionality by actually generating frames on the GPU, but use minimal settings (3 frames, low resolution, few steps) for speed.

### Differences from API Integration Tests

| Feature | API Tests (`tests/integration/`) | GPU Tests (`tests/integration/gpu/`) |
|---------|----------------------------------|--------------------------------------|
| **Requires Server** | Yes (`--deforum-api`) | No (direct Python calls) |
| **Speed** | Slower (HTTP overhead) | Faster (direct calls) |
| **Setup** | Complex (server startup) | Simple (just run pytest) |
| **Use Case** | End-to-end API validation | Component functionality testing |

## Requirements

- **GPU with CUDA**: Tests will be skipped if no GPU available
- **Flux Model**: Must be loaded in Forge
- **Forge Environment**: Tests must run within Forge's Python environment

## Running Tests

### Run all GPU integration tests
```bash
./run-integration-tests.sh
```

### Run specific test
```bash
pytest tests/integration/gpu/test_flux_controlnet.py::test_flux_controlnet_basic_generation -v
```

### Run with output inspection (don't cleanup)
```bash
pytest tests/integration/gpu/ -v -s
# Outputs saved to tests/integration/gpu/test_outputs/
```

### Run only fast tests (skip slow generation tests)
```bash
pytest tests/integration/gpu/ -v -m "not slow"
```

## Test Organization

### `test_flux_controlnet.py`
Tests for Flux ControlNet V2 functionality:

- **`test_flux_controlnet_basic_generation`**: Full 3-frame generation with ControlNet
  - Verifies frames generate correctly
  - Checks VRAM usage for double-loading issues
  - Fast: ~30-60 seconds

- **`test_flux_controlnet_model_loading`**: Model loading verification
  - Checks ControlNet model loads correctly
  - Verifies memory footprint (~3.6GB)
  - Very fast: ~5-10 seconds

- **`test_flux_controlnet_single_frame`**: Single frame test
  - Fastest possible test with ControlNet enabled
  - Useful for quick smoke testing
  - Very fast: ~15-30 seconds

- **`test_flux_controlnet_disabled`**: Baseline comparison
  - Generates frames WITHOUT ControlNet
  - Compares VRAM usage
  - Helps diagnose ControlNet-specific issues

## Test Fixtures

### GPU Fixtures (`conftest.py`)

- **`gpu_available`**: Checks CUDA availability, skips tests if no GPU
- **`test_output_dir`**: Creates `test_outputs/` directory for all tests
- **`isolated_output_dir`**: Unique subdirectory per test (prevents conflicts)
- **`minimal_args`**: Fast generation settings (512x512, 4 steps, seed 42)
- **`minimal_anim_args`**: Short animation (3 frames, minimal transformations)
- **`flux_controlnet_args`**: Animation args with ControlNet enabled
- **`check_flux_available`**: Verifies Flux model is loaded

## Test Markers

Use markers to run specific test categories:

```bash
# GPU tests only
pytest -m gpu

# Flux ControlNet tests only
pytest -m flux_controlnet

# Fast tests only (exclude slow generation tests)
pytest -m "not slow"

# Flux model required
pytest -m flux
```

## Debugging Failed Tests

### Check test outputs
```bash
ls -la tests/integration/gpu/test_outputs/
```

Each test creates its own subdirectory with generated frames.

### Run with verbose output
```bash
pytest tests/integration/gpu/ -v -s
```

The `-s` flag shows print statements including VRAM usage.

### Check VRAM usage
Tests automatically report VRAM usage:
```
[VRAM] Peak usage during generation: 12.34 GB
[VRAM] Incremental usage: 8.50 GB
```

If incremental usage is >20GB, likely indicates double-loading issue.

## Common Issues

### "GPU not available - skipping GPU integration tests"
**Solution**: Ensure CUDA is available and working:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### "Flux model not loaded - needed for Flux ControlNet tests"
**Solution**: Load a Flux model in Forge before running tests

### "ControlNet model loading failed"
**Solution**: Check that Flux ControlNet V2 is properly installed in Forge

## Adding New Tests

Example test template:

```python
@pytest.mark.gpu
@pytest.mark.flux
@pytest.mark.slow
def test_my_feature(
    gpu_available,
    check_flux_available,
    minimal_args,
    minimal_anim_args,
    isolated_output_dir
):
    \"\"\"Test description explaining what is being verified.\"\"\"
    from deforum.orchestration.run_deforum import run_deforum
    from deforum.config.args import DeforumOutputArgs, ParseqArgs

    # Setup
    minimal_args.outdir = str(isolated_output_dir)
    minimal_args.prompts = {0: "test prompt"}

    # Configure args for your test
    minimal_anim_args.my_feature_enabled = True

    # Run
    try:
        result = run_deforum(minimal_args, minimal_anim_args, ...)

        # Verify
        assert condition, "failure message"
        print("[SUCCESS] Test passed")

    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")
```

## Performance Guidelines

- Keep tests fast: Use max 3 frames
- Use low resolution: 512x512 or less
- Use few steps: 4-8 steps
- Use consistent seed: 42 (for reproducibility)
- Cleanup outputs: Handled automatically by fixtures
