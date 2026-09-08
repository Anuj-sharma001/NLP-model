"""Unit and integration tests for the /health endpoint."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_health_check_status_code(client: TestClient):
    """Test that the /health endpoint returns HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
def test_health_check_payload_structure(client: TestClient):
    """Test that the /health response adheres to the expected schema."""
    response = client.get("/health")
    data = response.json()

    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
    assert "models" in data
    assert isinstance(data["models"], dict)


@pytest.mark.unit
def test_health_check_environment(client: TestClient):
    """Test that the test environment is correctly reflected."""
    response = client.get("/health")
    data = response.json()
    assert data["environment"] == "test"
    assert data["app_name"] == "SAKYTI NLP Service (Test)"


@pytest.mark.unit
def test_not_found_endpoint(client: TestClient):
    """Test standard 404 response for unregistered route."""
    response = client.get("/api/v1/non-existent-route")
    assert response.status_code == 404
