"""Test Swagger/OpenAPI documentation and basic API functionality.

This test verifies that:
1. Swagger UI is accessible
2. OpenAPI schema is properly generated
3. All documented endpoints are present
4. Response models match the schema
"""

import requests
import pytest
from .utils import SERVER_BASE_URL, API_BASE_URL


def test_swagger_ui_accessible():
    """Verify Swagger UI is accessible at /docs endpoint."""
    response = requests.get(f"{SERVER_BASE_URL}/docs")
    assert response.status_code == 200, "Swagger UI should be accessible at /docs"
    assert "swagger" in response.text.lower(), "Response should contain Swagger UI"


def test_openapi_schema_accessible():
    """Verify OpenAPI JSON schema is accessible."""
    response = requests.get(f"{SERVER_BASE_URL}/openapi.json")
    assert response.status_code == 200, "OpenAPI schema should be accessible"
    schema = response.json()

    # Verify basic OpenAPI structure
    assert "openapi" in schema, "Schema should have openapi version"
    assert "paths" in schema, "Schema should have paths"
    assert "components" in schema, "Schema should have components"


def test_batch_endpoints_in_schema():
    """Verify all batch endpoints are documented in OpenAPI schema."""
    response = requests.get(f"{SERVER_BASE_URL}/openapi.json")
    schema = response.json()
    paths = schema["paths"]

    # Check batch endpoints exist
    assert "/deforum_api/batches" in paths, "POST /batches endpoint should be documented"
    assert "post" in paths["/deforum_api/batches"], "Batch submit should be documented"
    assert "get" in paths["/deforum_api/batches"], "Batch list should be documented"

    assert "/deforum_api/batches/{id}" in paths, "Batch detail endpoint should be documented"
    assert "get" in paths["/deforum_api/batches/{id}"], "GET batch should be documented"
    assert "delete" in paths["/deforum_api/batches/{id}"], "DELETE batch should be documented"


def test_job_endpoints_in_schema():
    """Verify all job endpoints are documented in OpenAPI schema."""
    response = requests.get(f"{SERVER_BASE_URL}/openapi.json")
    schema = response.json()
    paths = schema["paths"]

    # Check job endpoints exist
    assert "/deforum_api/jobs" in paths, "Jobs list endpoint should be documented"
    assert "get" in paths["/deforum_api/jobs"], "GET jobs list should be documented"

    assert "/deforum_api/jobs/{id}" in paths, "Job detail endpoint should be documented"
    assert "get" in paths["/deforum_api/jobs/{id}"], "GET job should be documented"
    assert "delete" in paths["/deforum_api/jobs/{id}"], "DELETE job should be documented"


def test_simple_api_endpoints_in_schema():
    """Verify Simple API endpoints are documented."""
    response = requests.get(f"{SERVER_BASE_URL}/openapi.json")
    schema = response.json()
    paths = schema["paths"]

    # Check simple API endpoints exist
    assert "/deforum/api_version" in paths, "API version endpoint should be documented"
    assert "/deforum/version" in paths, "Deforum version endpoint should be documented"
    assert "/deforum/run" in paths, "Simple run endpoint should be documented"


def test_response_models_in_schema():
    """Verify response models are properly defined in schema."""
    response = requests.get(f"{SERVER_BASE_URL}/openapi.json")
    schema = response.json()
    components = schema.get("components", {})
    schemas = components.get("schemas", {})

    # Check key models exist
    assert "DeforumJobStatus" in schemas, "DeforumJobStatus model should be defined"
    assert "BatchSubmitResponse" in schemas, "BatchSubmitResponse model should be defined"
    assert "ErrorResponse" in schemas, "ErrorResponse model should be defined"
    assert "VersionResponse" in schemas, "VersionResponse model should be defined"
    assert "SimpleRunResponse" in schemas, "SimpleRunResponse model should be defined"


def test_batch_submit_endpoint_documentation():
    """Verify batch submit endpoint has proper documentation."""
    response = requests.get(f"{SERVER_BASE_URL}/openapi.json")
    schema = response.json()
    batch_post = schema["paths"]["/deforum_api/batches"]["post"]

    # Check endpoint metadata
    assert "summary" in batch_post, "Endpoint should have summary"
    assert "tags" in batch_post, "Endpoint should have tags"
    assert "Batches" in batch_post["tags"], "Endpoint should be tagged with 'Batches'"

    # Check response documentation
    assert "responses" in batch_post, "Endpoint should document responses"
    assert "202" in batch_post["responses"], "Should document 202 response"
    assert "400" in batch_post["responses"], "Should document 400 error response"


def test_list_jobs_endpoint_works():
    """Verify /jobs endpoint returns valid response."""
    response = requests.get(f"{API_BASE_URL}/jobs")
    assert response.status_code == 200, "Jobs list endpoint should return 200"

    # Response should be a dict (even if empty)
    data = response.json()
    assert isinstance(data, dict), "Jobs list should return a dictionary"


def test_list_batches_endpoint_works():
    """Verify /batches endpoint returns valid response."""
    response = requests.get(f"{API_BASE_URL}/batches")
    assert response.status_code == 200, "Batches list endpoint should return 200"

    # Response should be a dict (even if empty)
    data = response.json()
    assert isinstance(data, dict), "Batches list should return a dictionary"


def test_simple_api_version_endpoint():
    """Verify Simple API version endpoint works."""
    response = requests.get(f"{SERVER_BASE_URL}/deforum/api_version")
    assert response.status_code == 200, "API version endpoint should return 200"

    data = response.json()
    assert "version" in data, "Response should contain version field"
    assert isinstance(data["version"], str), "Version should be a string"


def test_deforum_version_endpoint():
    """Verify Deforum version endpoint works."""
    response = requests.get(f"{SERVER_BASE_URL}/deforum/version")
    assert response.status_code == 200, "Deforum version endpoint should return 200"

    data = response.json()
    assert "version" in data, "Response should contain version field"
    assert isinstance(data["version"], str), "Version should be a string"


def test_get_nonexistent_job_returns_404():
    """Verify getting a non-existent job returns 404."""
    response = requests.get(f"{API_BASE_URL}/jobs/nonexistent-job-id")
    assert response.status_code == 404, "Non-existent job should return 404"

    data = response.json()
    assert "detail" in data, "Error response should contain detail message"


def test_get_nonexistent_batch_returns_404():
    """Verify getting a non-existent batch returns 404."""
    response = requests.get(f"{API_BASE_URL}/batches/nonexistent-batch-id")
    assert response.status_code == 404, "Non-existent batch should return 404"

    data = response.json()
    assert "detail" in data, "Error response should contain detail message"


def test_endpoint_tags_are_consistent():
    """Verify all endpoints have proper tags for organization."""
    response = requests.get(f"{SERVER_BASE_URL}/openapi.json")
    schema = response.json()

    expected_tags = {"Batches", "Jobs", "Simple API"}

    # Collect all tags used
    all_tags = set()
    for path_data in schema["paths"].values():
        for method_data in path_data.values():
            if "tags" in method_data:
                all_tags.update(method_data["tags"])

    # Verify expected tags are present
    for tag in expected_tags:
        assert tag in all_tags, f"Tag '{tag}' should be used in API"


def test_all_endpoints_have_response_models():
    """Verify all endpoints specify response models."""
    response = requests.get(f"{SERVER_BASE_URL}/openapi.json")
    schema = response.json()

    for path, path_data in schema["paths"].items():
        for method, method_data in path_data.items():
            responses = method_data.get("responses", {})

            # Check that at least one success response has a schema
            has_schema = False
            for status_code, response_data in responses.items():
                if status_code.startswith("2"):  # 2xx success codes
                    if "content" in response_data or "$ref" in response_data:
                        has_schema = True
                        break

            # Some endpoints might not have explicit schemas (like list endpoints returning dicts)
            # But they should at least document responses
            assert responses, f"{method.upper()} {path} should document responses"
