#!/usr/bin/env python3
"""Manual API test script to verify Swagger documentation and endpoints.

Usage:
    python3 tests/manual_api_test.py

Requirements:
    - Server must be running with --deforum-api flag
    - Run: python3 webui.py --deforum-api
"""

import sys
import requests
import json
from typing import Dict, Any


SERVER_BASE_URL = "http://localhost:7860"
API_BASE_URL = f"{SERVER_BASE_URL}/deforum_api"


def color_print(text: str, color: str = "green") -> None:
    """Print colored text to console."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def test_server_reachable() -> bool:
    """Test if server is reachable."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/", timeout=5)
        color_print("✓ Server is reachable", "green")
        return True
    except requests.exceptions.RequestException as e:
        color_print(f"✗ Server is not reachable: {e}", "red")
        color_print("Please start the server with: python3 webui.py --deforum-api", "yellow")
        return False


def test_swagger_ui() -> bool:
    """Test Swagger UI is accessible."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/docs", timeout=5)
        if response.status_code == 200 and "swagger" in response.text.lower():
            color_print("✓ Swagger UI is accessible at /docs", "green")
            return True
        else:
            color_print(f"✗ Swagger UI returned status {response.status_code}", "red")
            return False
    except requests.exceptions.RequestException as e:
        color_print(f"✗ Swagger UI test failed: {e}", "red")
        return False


def test_openapi_schema() -> bool:
    """Test OpenAPI schema is properly generated."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/openapi.json", timeout=5)
        if response.status_code != 200:
            color_print(f"✗ OpenAPI schema returned status {response.status_code}", "red")
            return False

        schema = response.json()

        # Check basic structure
        checks = [
            ("openapi" in schema, "Has openapi version"),
            ("paths" in schema, "Has paths"),
            ("components" in schema, "Has components"),
        ]

        all_passed = True
        for check, description in checks:
            if check:
                color_print(f"  ✓ {description}", "green")
            else:
                color_print(f"  ✗ {description}", "red")
                all_passed = False

        if all_passed:
            color_print("✓ OpenAPI schema is properly structured", "green")
            return True
        return False

    except requests.exceptions.RequestException as e:
        color_print(f"✗ OpenAPI schema test failed: {e}", "red")
        return False
    except json.JSONDecodeError as e:
        color_print(f"✗ OpenAPI schema is not valid JSON: {e}", "red")
        return False


def test_endpoint_documentation() -> bool:
    """Test that all endpoints are documented."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/openapi.json", timeout=5)
        schema = response.json()
        paths = schema.get("paths", {})

        expected_endpoints = {
            "/deforum_api/batches": ["post", "get"],
            "/deforum_api/batches/{id}": ["get", "delete"],
            "/deforum_api/jobs": ["get"],
            "/deforum_api/jobs/{id}": ["get", "delete"],
            "/deforum/api_version": ["get"],
            "/deforum/version": ["get"],
            "/deforum/run": ["post"],
        }

        all_passed = True
        for endpoint, methods in expected_endpoints.items():
            if endpoint not in paths:
                color_print(f"  ✗ Missing endpoint: {endpoint}", "red")
                all_passed = False
                continue

            for method in methods:
                if method in paths[endpoint]:
                    color_print(f"  ✓ {method.upper()} {endpoint}", "green")
                else:
                    color_print(f"  ✗ Missing method: {method.upper()} {endpoint}", "red")
                    all_passed = False

        if all_passed:
            color_print("✓ All endpoints are documented", "green")
            return True
        return False

    except Exception as e:
        color_print(f"✗ Endpoint documentation test failed: {e}", "red")
        return False


def test_response_models() -> bool:
    """Test that response models are defined."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/openapi.json", timeout=5)
        schema = response.json()
        schemas = schema.get("components", {}).get("schemas", {})

        expected_models = [
            "DeforumJobStatus",
            "BatchSubmitResponse",
            "BatchCancelResponse",
            "JobCancelResponse",
            "ErrorResponse",
            "VersionResponse",
            "SimpleRunResponse",
        ]

        all_passed = True
        for model in expected_models:
            if model in schemas:
                color_print(f"  ✓ {model} model defined", "green")
            else:
                color_print(f"  ✗ Missing model: {model}", "red")
                all_passed = False

        if all_passed:
            color_print("✓ All response models are defined", "green")
            return True
        return False

    except Exception as e:
        color_print(f"✗ Response models test failed: {e}", "red")
        return False


def test_endpoint_tags() -> bool:
    """Test that endpoints have proper tags."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/openapi.json", timeout=5)
        schema = response.json()

        expected_tags = {"Batches", "Jobs", "Simple API"}
        found_tags = set()

        for path_data in schema.get("paths", {}).values():
            for method_data in path_data.values():
                if "tags" in method_data:
                    found_tags.update(method_data["tags"])

        all_passed = True
        for tag in expected_tags:
            if tag in found_tags:
                color_print(f"  ✓ Tag '{tag}' is used", "green")
            else:
                color_print(f"  ✗ Missing tag: '{tag}'", "red")
                all_passed = False

        if all_passed:
            color_print("✓ All expected tags are present", "green")
            return True
        return False

    except Exception as e:
        color_print(f"✗ Endpoint tags test failed: {e}", "red")
        return False


def test_list_jobs_endpoint() -> bool:
    """Test /jobs endpoint works."""
    try:
        response = requests.get(f"{API_BASE_URL}/jobs", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                color_print(f"✓ GET /jobs returns valid response (found {len(data)} jobs)", "green")
                return True
            else:
                color_print("✗ GET /jobs did not return a dict", "red")
                return False
        else:
            color_print(f"✗ GET /jobs returned status {response.status_code}", "red")
            return False
    except Exception as e:
        color_print(f"✗ GET /jobs test failed: {e}", "red")
        return False


def test_list_batches_endpoint() -> bool:
    """Test /batches endpoint works."""
    try:
        response = requests.get(f"{API_BASE_URL}/batches", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                color_print(f"✓ GET /batches returns valid response (found {len(data)} batches)", "green")
                return True
            else:
                color_print("✗ GET /batches did not return a dict", "red")
                return False
        else:
            color_print(f"✗ GET /batches returned status {response.status_code}", "red")
            return False
    except Exception as e:
        color_print(f"✗ GET /batches test failed: {e}", "red")
        return False


def test_api_version_endpoint() -> bool:
    """Test Simple API version endpoint."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/deforum/api_version", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "version" in data:
                color_print(f"✓ GET /deforum/api_version returns version: {data['version']}", "green")
                return True
            else:
                color_print("✗ API version response missing 'version' field", "red")
                return False
        else:
            color_print(f"✗ GET /deforum/api_version returned status {response.status_code}", "red")
            return False
    except Exception as e:
        color_print(f"✗ API version test failed: {e}", "red")
        return False


def test_deforum_version_endpoint() -> bool:
    """Test Deforum version endpoint."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/deforum/version", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "version" in data:
                color_print(f"✓ GET /deforum/version returns version: {data['version']}", "green")
                return True
            else:
                color_print("✗ Deforum version response missing 'version' field", "red")
                return False
        else:
            color_print(f"✗ GET /deforum/version returned status {response.status_code}", "red")
            return False
    except Exception as e:
        color_print(f"✗ Deforum version test failed: {e}", "red")
        return False


def test_404_responses() -> bool:
    """Test that non-existent resources return 404."""
    try:
        tests = [
            (f"{API_BASE_URL}/jobs/nonexistent-job-id", "job"),
            (f"{API_BASE_URL}/batches/nonexistent-batch-id", "batch"),
        ]

        all_passed = True
        for url, resource_type in tests:
            response = requests.get(url, timeout=5)
            if response.status_code == 404:
                color_print(f"  ✓ Non-existent {resource_type} returns 404", "green")
            else:
                color_print(f"  ✗ Non-existent {resource_type} returned status {response.status_code}", "red")
                all_passed = False

        if all_passed:
            color_print("✓ 404 responses work correctly", "green")
            return True
        return False

    except Exception as e:
        color_print(f"✗ 404 responses test failed: {e}", "red")
        return False


def main():
    """Run all API tests."""
    color_print("\n=== Deforum API Test Suite ===\n", "blue")

    tests = [
        ("Server Reachability", test_server_reachable),
        ("Swagger UI", test_swagger_ui),
        ("OpenAPI Schema", test_openapi_schema),
        ("Endpoint Documentation", test_endpoint_documentation),
        ("Response Models", test_response_models),
        ("Endpoint Tags", test_endpoint_tags),
        ("List Jobs Endpoint", test_list_jobs_endpoint),
        ("List Batches Endpoint", test_list_batches_endpoint),
        ("API Version Endpoint", test_api_version_endpoint),
        ("Deforum Version Endpoint", test_deforum_version_endpoint),
        ("404 Responses", test_404_responses),
    ]

    results = []
    for test_name, test_func in tests:
        color_print(f"\n--- {test_name} ---", "blue")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            color_print(f"✗ Test crashed: {e}", "red")
            results.append((test_name, False))

    # Print summary
    color_print("\n=== Test Summary ===\n", "blue")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        color = "green" if result else "red"
        color_print(f"{status:6} {test_name}", color)

    color_print(f"\nPassed: {passed}/{total}", "green" if passed == total else "yellow")

    # Return exit code
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
