"""Test configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from netpulse_ml.main import create_app


@pytest.fixture
def app():
    """Create a test app instance (without lifespan for unit tests)."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a test HTTP client."""
    return TestClient(app)
