import pytest
from fastapi import HTTPException
from app.core.security import (
    track_token_usage,
    get_current_token_usage,
    reset_token_usage,
    enforce_rate_limit,
    verify_api_key,
    _rate_limit_records
)

def test_missing_api_key(unauth_client):
    """Endpoints requiring verify_api_key return 401 when X-API-Key header is missing."""
    response = unauth_client.post("/api/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 401
    assert "Missing X-API-Key header" in response.json()["detail"]

def test_invalid_api_key(unauth_client):
    """Endpoints return 401 when X-API-Key is invalid."""
    response = unauth_client.post(
        "/api/v1/chat/completions",
        headers={"X-API-Key": "invalid_key_123"},
        json={"messages": [{"role": "user", "content": "Hello"}]}
    )
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]

def test_token_usage_tracking():
    """Test updating, retrieving, and resetting token counts in SQLite metrics."""
    test_app = "test_unit_app"
    reset_token_usage(test_app)
    
    assert get_current_token_usage(test_app) == 0
    
    used = track_token_usage(test_app, 150)
    assert used == 150
    assert get_current_token_usage(test_app) == 150
    
    used_again = track_token_usage(test_app, 50)
    assert used_again == 200
    assert get_current_token_usage(test_app) == 200
    
    reset_token_usage(test_app)
    assert get_current_token_usage(test_app) == 0

def test_rate_limit_enforcement():
    """Test sliding window rate limiting raises 429 when max RPM is exceeded."""
    app_id = "test_rate_limit_app"
    client_ip = "192.168.1.100"
    rpm = 3

    _rate_limit_records.pop((app_id, client_ip), None)

    # 3 requests allowed
    enforce_rate_limit(app_id, client_ip, rpm)
    enforce_rate_limit(app_id, client_ip, rpm)
    enforce_rate_limit(app_id, client_ip, rpm)

    # 4th request must raise HTTP 429
    with pytest.raises(HTTPException) as exc_info:
        enforce_rate_limit(app_id, client_ip, rpm)
    
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail
