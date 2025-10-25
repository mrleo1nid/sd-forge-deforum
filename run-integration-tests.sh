#!/bin/bash

# Run local GPU integration tests for Deforum
# These tests run directly on the GPU without requiring the API server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Deforum GPU Integration Tests ===${NC}"
echo ""

# Activate Forge's venv if not already in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "../../venv/bin/activate" ]; then
        echo -e "${GREEN}Activating Forge virtual environment...${NC}"
        source ../../venv/bin/activate
        echo ""
    else
        echo -e "${YELLOW}Warning: Forge venv not found at ../../venv${NC}"
        echo "Tests should run within Forge's Python environment"
        echo ""
    fi
fi

# Set Python command
PYTHON_CMD="python"

# Check CUDA availability
echo -e "${GREEN}Checking GPU...${NC}"
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}Using virtual environment: $VIRTUAL_ENV${NC}"
fi
$PYTHON_CMD -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || {
    echo -e "${RED}Error: Cannot detect GPU. Tests may be skipped.${NC}"
}
echo ""

# Default test path
TEST_PATH="tests/integration/gpu/"

# Parse arguments
PYTEST_ARGS=""
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            PYTEST_ARGS="$PYTEST_ARGS -m 'not slow'"
            shift
            ;;
        --verbose|-v)
            PYTEST_ARGS="$PYTEST_ARGS -v"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options] [test-path]"
            echo ""
            echo "Options:"
            echo "  --quick         Skip slow tests (generation tests)"
            echo "  --verbose, -v   Verbose output"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all GPU integration tests"
            echo "  $0 --quick                            # Run only fast tests"
            echo "  $0 tests/integration/gpu/test_flux_controlnet.py  # Run specific test file"
            echo "  $0 --verbose                          # Run with verbose output"
            exit 0
            ;;
        *)
            TEST_PATH="$1"
            shift
            ;;
    esac
done

# Show test mode
if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}Quick mode: Skipping slow tests${NC}"
else
    echo -e "${GREEN}Running all tests (including slow generation tests)${NC}"
fi
echo ""

# Show test path
echo -e "${GREEN}Test path: ${TEST_PATH}${NC}"
echo ""

# Run pytest
echo -e "${GREEN}Running tests...${NC}"
echo ""

$PYTHON_CMD -m pytest $TEST_PATH $PYTEST_ARGS || {
    EXIT_CODE=$?
    echo ""
    echo -e "${RED}Tests failed with exit code $EXIT_CODE${NC}"
    exit $EXIT_CODE
}

echo ""
echo -e "${GREEN}=== All tests passed! ===${NC}"
echo ""
echo "Test outputs saved to: tests/integration/gpu/test_outputs/"
echo ""
