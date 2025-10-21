# Deforum Test Suite

Organized test suite following pytest best practices.

## Directory Structure

```
tests/
├── integration/         # API and system integration tests (require server + GPU)
│   ├── api_test.py              # Basic Deforum API tests (2D, 3D, Parseq, cancellation)
│   ├── postprocess_test.py      # Post-processing tests (FILM, RIFE, upscaling)
│   ├── utils.py                 # Shared utilities for integration tests
│   ├── testdata/                # Test fixtures (settings files, videos)
│   └── __snapshots__/           # Snapshot data for regression testing
│
├── unit/                # Fast unit tests (no server, no GPU, no external deps)
│   └── (empty - to be created)
│
├── functional/          # End-to-end functional tests
│   └── (empty - reserved for future use)
│
├── performance/         # Performance benchmarking tests
│   └── (empty - reserved for future use)
│
├── conftest.py          # Pytest fixtures and configuration
└── README.md            # This file
```

## Test Types

### Integration Tests (`integration/`)
**What:** Tests that verify Deforum works correctly with the Forge API and generates valid videos.

**Requirements:**
- Running Forge server with `--deforum-api` flag
- GPU (for most tests)
- Loaded models (checkpoints, VAE)
- Slow (minutes per test)

**Examples:**
- `api_test.py::test_simple_settings` - Generate 2D animation via API
- `api_test.py::test_3d_mode` - Generate 3D depth-warped animation
- `postprocess_test.py::test_post_process_FILM` - Frame interpolation with FILM

**When to run:** Before releases, after major refactoring, in CI

### Unit Tests (`unit/`) - **To Be Created**
**What:** Tests for individual functions and classes in isolation.

**Requirements:**
- No server
- No GPU
- No external dependencies (use mocks)
- Fast (milliseconds per test)

**Examples (planned):**
- `test_keyframe_distribution.py` - Test keyframe distribution algorithms
- `test_args.py` - Test argument parsing and validation
- `test_prompt.py` - Test prompt scheduling logic
- `test_wan_integration.py` - Test Wan frame count calculations (with mocked model)

**When to run:** During development, on every commit, in CI

### Functional Tests (`functional/`) - Reserved
**What:** End-to-end tests of complete workflows from user perspective.

**Future use:** When we have more complex multi-step workflows to test.

### Performance Tests (`performance/`) - Reserved
**What:** Benchmarking tests to measure speed and resource usage.

**Future use:** To track performance regressions and optimize bottlenecks.

## Running Tests

### Quick Start (All Integration Tests)
```bash
./run-tests.sh
```

### Quick Mode (Skip Slow Post-Processing Tests)
```bash
./run-tests.sh --quick
```

### Run Specific Test
```bash
./run-tests.sh tests/integration/api_test.py::test_simple_settings
```

### Run Unit Tests Only (When Created)
```bash
pytest tests/unit/ -v
```

### Run All Tests
```bash
pytest tests/ -v
```

## Test Organization Principles

### 1. **Separation of Concerns**
- **Integration tests** verify system behavior (does it work?)
- **Unit tests** verify correctness (does this function work?)
- **Performance tests** verify efficiency (is it fast enough?)

### 2. **Fast Feedback Loop**
- Unit tests should run in seconds
- Integration tests can take minutes
- Developers should run unit tests frequently
- CI runs both, but unit tests on every commit

### 3. **Test Pyramid**
```
        /\
       /  \     Few slow integration tests (comprehensive scenarios)
      /----\
     /      \   More unit tests (edge cases, error handling)
    /________\
```

Most tests should be fast unit tests. Integration tests should focus on critical paths.

### 4. **No Test Interdependence**
- Each test should be independent
- Tests should not rely on execution order
- Use fixtures for shared setup

### 5. **Clear Naming**
```python
def test_<what>_<condition>_<expected_result>():
    """Clear docstring explaining what and why"""
```

Example:
```python
def test_distribute_keyframes_evenly_spaced_matches_exactly():
    """When prompts are evenly spaced, keyframes should match frame numbers exactly"""
```

## Writing New Tests

### For Integration Tests
Add to `tests/integration/` when testing:
- Full rendering pipeline
- API endpoints
- Video output validation
- Multi-component interaction

```python
# tests/integration/test_new_feature.py
import requests
from .utils import API_BASE_URL, wait_for_job_to_complete

def test_new_feature():
    """Test that new feature generates valid output"""
    # Given: settings with new feature enabled
    settings = {...}

    # When: job is submitted
    response = requests.post(f"{API_BASE_URL}/batches", json={"deforum_settings": [settings]})
    job_id = response.json()["job_ids"][0]

    # Then: job completes successfully
    status = wait_for_job_to_complete(job_id)
    assert status.status == "SUCCEEDED"
```

### For Unit Tests
Add to `tests/unit/` when testing:
- Individual functions
- Classes and methods
- Edge cases and error handling
- Complex algorithms

```python
# tests/unit/test_keyframe_distribution.py
import pytest
from scripts.deforum_helpers.rendering.data.frame.key_frame_distribution import distribute_keyframes

def test_distribute_keyframes_empty_prompts_returns_empty():
    """Empty prompt dict should return empty keyframe list"""
    result = distribute_keyframes({}, max_frames=100)
    assert result == []

def test_distribute_keyframes_single_prompt_creates_one_keyframe():
    """Single prompt should create exactly one keyframe at frame 0"""
    result = distribute_keyframes({0: "test prompt"}, max_frames=100)
    assert len(result) == 1
    assert result[0].frame_number == 0
```

## Best Practices

### ✅ Do
- Write descriptive test names
- Add docstrings explaining what you're testing
- Use fixtures for common setup
- Mock external dependencies in unit tests
- Test edge cases and error conditions
- Keep tests focused (one concept per test)

### ❌ Don't
- Write tests that depend on other tests
- Use sleep() unless absolutely necessary
- Test implementation details (test behavior, not internals)
- Skip writing tests for "simple" code (simple code can have bugs too)
- Commit tests that are flaky or sometimes fail

## Dependencies

Install test dependencies:
```bash
pip install -r requirements-dev.txt
```

See also: `TEST_STATUS.md` for current test status and issues.
