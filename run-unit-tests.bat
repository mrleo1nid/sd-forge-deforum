@echo off
REM run-unit-tests.bat - Run unit tests (no server required)
REM
REM Usage:
REM   run-unit-tests.bat                    REM Run all unit tests
REM   run-unit-tests.bat --coverage         REM Run with coverage report
REM   run-unit-tests.bat tests\unit\test_keyframes.py  REM Run specific test file

setlocal enabledelayedexpansion

REM Configuration
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
cd ..\..
set FORGE_DIR=%CD%
cd /d "%SCRIPT_DIR%"
set UNIT_TEST_DIR=tests\unit

REM Parse arguments
set COVERAGE_MODE=false
set TEST_ARGS=

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--coverage" (
    set COVERAGE_MODE=true
) else (
    set TEST_ARGS=!TEST_ARGS! %~1
)
shift
goto :parse_args

:args_done

REM Set default test path if none specified
if "!TEST_ARGS!"=="" (
    if exist "%UNIT_TEST_DIR%" (
        set TEST_ARGS=%UNIT_TEST_DIR%\
        echo Running all unit tests from %UNIT_TEST_DIR%\
    ) else (
        echo [WARNING] %UNIT_TEST_DIR%\ does not exist yet
        echo Create unit tests in %UNIT_TEST_DIR%\ directory
        echo.
        echo Example structure:
        echo   tests\unit\
        echo     ├── test_keyframes.py
        echo     ├── test_prompts.py
        echo     ├── test_args.py
        echo     └── test_wan_integration.py
        exit /b 1
    )
)

REM Check if we're in the extension directory
if not exist "scripts\deforum.py" (
    echo [ERROR] Must run from sd-forge-deforum extension directory
    echo Current directory: %CD%
    exit /b 1
)

REM Check if venv exists
if not exist "%FORGE_DIR%\venv" (
    echo [ERROR] Forge venv not found at %FORGE_DIR%\venv
    echo Please run this from the Forge installation
    exit /b 1
)

echo ========================================
echo Deforum Unit Test Runner
echo ========================================
echo Extension directory: %SCRIPT_DIR%
echo Test arguments: !TEST_ARGS!
if "!COVERAGE_MODE!"=="true" (
    echo Coverage reporting: ENABLED
)
echo ========================================
echo.

REM Check test dependencies
echo Checking test dependencies...
"%FORGE_DIR%\venv\Scripts\python.exe" -c "import pytest" 2>nul
if errorlevel 1 (
    echo Installing test dependencies...
    "%FORGE_DIR%\venv\Scripts\pip.exe" install -q pytest pytest-cov
    echo [OK] Test dependencies installed
) else (
    echo [OK] Test dependencies OK
)

REM Run tests
echo.
echo ========================================
echo Running Unit Tests (Integration tests excluded)
echo ========================================
echo.

REM Build pytest command
set PYTEST_CMD="%FORGE_DIR%\venv\Scripts\python.exe" -m pytest !TEST_ARGS! -v --tb=short --ignore=tests\integration

if "!COVERAGE_MODE!"=="true" (
    set PYTEST_CMD=!PYTEST_CMD! --cov=deforum --cov=scripts --cov-report=term-missing --cov-report=html
)

REM Run tests (don't exit on failure)
call !PYTEST_CMD!
set TEST_EXIT_CODE=%errorlevel%

REM Print results
echo.
echo ========================================
if %TEST_EXIT_CODE% equ 0 (
    echo [OK] All unit tests passed!
    if "!COVERAGE_MODE!"=="true" (
        echo Coverage report: htmlcov\index.html
    )
) else (
    echo [FAILED] Some unit tests failed
)
echo ========================================

exit /b %TEST_EXIT_CODE%
