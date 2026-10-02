from unittest.mock import patch, AsyncMock

def test_vision_missing_image_input(client):
    """Endpoint must return 400 Bad Request if neither image_url nor image_b64 is supplied."""
    response = client.post("/api/v1/vision/analyze", json={
        "prompt": "Đọc chữ trong ảnh này"
    })
    assert response.status_code == 400
    assert "You must supply either 'image_url' or 'image_b64'" in response.json()["detail"]

def test_vision_analyze_success(client):
    """Test vision analysis with base64 image input."""
    mock_llm_response = {
        "text": "Bức ảnh chứa mã nguồn Python hiển thị hàm print('Hello World').",
        "model": "gemini/gemini-2.0-flash",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    }
    
    with patch("app.services.llm_factory.LLMFactory.chat_completion", new_callable=AsyncMock) as mock_vision:
        mock_vision.return_value = mock_llm_response
        
        response = client.post("/api/v1/vision/analyze", json={
            "prompt": "Phân tích đoạn code trong ảnh",
            "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "print('Hello World')" in data["text"]
        assert data["usage"]["total_tokens"] == 120
