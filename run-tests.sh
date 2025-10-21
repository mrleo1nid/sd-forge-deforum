#!/bin/bash
# run-tests.sh - Start Forge server with Deforum API and run tests
#
# Usage:
#   ./run-tests.sh              # Run all tests
#   ./run-tests.sh --quick      # Skip post-processing tests (faster)
#   ./run-tests.sh tests/deforum_test.py::test_simple_settings  # Run specific test

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FORGE_DIR="$(cd "../../" && pwd)"
EXTENSION_DIR="$(pwd)"
SERVER_URL="http://localhost:7860"
DEFORUM_API_URL="${SERVER_URL}/deforum_api/jobs"
MAX_WAIT=300  # Maximum seconds to wait for server (5 minutes)
SERVER_LOG="${EXTENSION_DIR}/test-server.log"
PID_FILE="${EXTENSION_DIR}/.test-server.pid"

# Parse arguments
QUICK_MODE=false
TEST_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--quick" ]; then
        QUICK_MODE=true
    else
        TEST_ARGS="$TEST_ARGS $arg"
    fi
done

# Set default test path if none specified
if [ -z "$TEST_ARGS" ]; then
    if [ "$QUICK_MODE" = true ]; then
        TEST_ARGS="tests/integration/api_test.py"
        echo -e "${YELLOW}Quick mode: Running only API tests (skipping post-processing tests)${NC}"
    else
        TEST_ARGS="tests/integration/"
        echo -e "${BLUE}Running all integration tests (API + post-processing)${NC}"
    fi
fi

# Cleanup function - called on exit
cleanup() {
    local exit_code=$?
    echo -e "\n${BLUE}Cleaning up...${NC}"

    # Kill server if PID file exists
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${BLUE}Stopping Forge server (PID: $pid)${NC}"
            kill "$pid" 2>/dev/null || true
            # Give it a moment to shut down gracefully
            sleep 2
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${YELLOW}Force killing server${NC}"
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Tests completed successfully${NC}"
    else
        echo -e "${RED}✗ Tests failed or were interrupted${NC}"
        echo -e "${YELLOW}Server log available at: ${SERVER_LOG}${NC}"
    fi

    exit $exit_code
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Check if server is already running
check_server() {
    curl -s -f "$DEFORUM_API_URL" > /dev/null 2>&1
    return $?
}

# Wait for server to be ready
wait_for_server() {
    local waited=0
    local interval=2

    echo -e "${BLUE}Waiting for server to start (max ${MAX_WAIT}s)...${NC}"

    while [ $waited -lt $MAX_WAIT ]; do
        if check_server; then
            echo -e "${GREEN}✓ Server is ready!${NC}"
            return 0
        fi

        # Show progress every 10 seconds
        if [ $((waited % 10)) -eq 0 ]; then
            echo -e "${BLUE}  Still waiting... (${waited}s)${NC}"
        fi

        sleep $interval
        waited=$((waited + interval))
    done

    echo -e "${RED}✗ Server failed to start after ${MAX_WAIT}s${NC}"
    echo -e "${YELLOW}Check server log at: ${SERVER_LOG}${NC}"
    return 1
}

# Check if we're in the extension directory
if [ ! -f "scripts/deforum.py" ]; then
    echo -e "${RED}Error: Must run from sd-forge-deforum extension directory${NC}"
    echo -e "Current directory: $(pwd)"
    exit 1
fi

# Check if venv exists
if [ ! -d "$FORGE_DIR/venv" ]; then
    echo -e "${RED}Error: Forge venv not found at $FORGE_DIR/venv${NC}"
    echo -e "Please run this from the Forge installation"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deforum Test Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Forge directory: ${FORGE_DIR}"
echo -e "Extension directory: ${EXTENSION_DIR}"
echo -e "Test arguments: ${TEST_ARGS}"
echo -e "${GREEN}========================================${NC}\n"

# Check if server is already running
if check_server; then
    echo -e "${YELLOW}⚠ Forge server is already running at ${SERVER_URL}${NC}"
    echo -e "${YELLOW}Using existing server. Press Ctrl+C within 5s to cancel...${NC}"
    sleep 5
    EXISTING_SERVER=true
else
    EXISTING_SERVER=false

    # Start Forge server in background
    echo -e "${BLUE}Starting Forge server with Deforum API...${NC}"
    echo -e "${BLUE}Server log: ${SERVER_LOG}${NC}\n"

    cd "$FORGE_DIR"

    # Start server and capture PID
    nohup python webui.py \
        --skip-prepare-environment \
        --api \
        --deforum-api \
        --skip-version-check \
        --no-gradio-queue \
        > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"

    echo -e "${BLUE}Server started (PID: $SERVER_PID)${NC}"

    cd "$EXTENSION_DIR"

    # Wait for server to be ready
    if ! wait_for_server; then
        echo -e "${RED}Failed to start server. Last 50 lines of log:${NC}"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
fi

# Check test dependencies
echo -e "\n${BLUE}Checking test dependencies...${NC}"
if ! "$FORGE_DIR/venv/bin/python" -c "import pytest, syrupy, moviepy" 2>/dev/null; then
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    "$FORGE_DIR/venv/bin/pip" install -q -r requirements-dev.txt
    echo -e "${GREEN}✓ Test dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Test dependencies OK${NC}"
fi

# Run tests
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Running Tests${NC}"
echo -e "${GREEN}========================================${NC}\n"

set +e  # Don't exit on test failure
"$FORGE_DIR/venv/bin/python" -m pytest $TEST_ARGS -v --tb=short
TEST_EXIT_CODE=$?
set -e

# If using existing server, don't clean it up
if [ "$EXISTING_SERVER" = true ]; then
    echo -e "\n${BLUE}Leaving existing server running${NC}"
    rm -f "$PID_FILE"  # Don't kill existing server
    trap - EXIT INT TERM  # Remove cleanup trap
fi

exit $TEST_EXIT_CODE
