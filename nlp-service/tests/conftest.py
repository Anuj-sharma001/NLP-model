"""Pytest fixtures and configuration."""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Provide isolated configuration for test execution."""
    return Settings(
        app_name="SAKYTI NLP Service (Test)",
        app_env="test",
        debug=True,
        log_level="DEBUG",
        log_format="text",
    )


@pytest.fixture(scope="session")
def app(test_settings: Settings):
    """Provide configured FastAPI test app."""
    # Override settings dependency
    def _override_get_settings():
        return test_settings

    test_app = create_app(settings=test_settings)
    test_app.dependency_overrides[get_settings] = _override_get_settings
    return test_app


@pytest.fixture(scope="session")
def client(app) -> TestClient:
    """Provide synchronous test client for FastAPI endpoints."""
    with TestClient(app) as test_client:
        yield test_client
