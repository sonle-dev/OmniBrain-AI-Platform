import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Test client equipped with valid default X-API-Key header."""
    return TestClient(app, headers={"X-API-Key": "demo_key"})

@pytest.fixture
def unauth_client():
    """Test client without authentication headers."""
    return TestClient(app)
