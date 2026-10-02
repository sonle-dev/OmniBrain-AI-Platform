from unittest.mock import patch, AsyncMock

def test_health_check_endpoint(client):
    """Test /health endpoint returns 200 and expected status keys."""
    with patch("app.services.llm_factory._check_ollama_reachable", new_callable=AsyncMock) as mock_ollama:
        mock_ollama.return_value = (True, ["qwen2.5:7b", "deepseek-r1:8b"])
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["version"] == "2.0.0"
        assert data["ollama_reachable"] is True
        assert data["ollama_tags_count"] == 2
        assert "message" in data

def test_static_client_js_serving(client):
    """Test serving of universal JS SDK at /static/omnibrain-client.js."""
    response = client.get("/static/omnibrain-client.js")
    assert response.status_code == 200
    assert "OmniBrainClient" in response.text
