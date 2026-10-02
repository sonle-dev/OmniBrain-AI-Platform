from unittest.mock import patch, AsyncMock
from app.api.v1.chat import _is_fast_path_prompt, _enrich_lms_context

def test_fast_path_intent_classifier():
    """Test early exit classifier for simple greetings and short general prompts."""
    assert _is_fast_path_prompt("xin chào") is True
    assert _is_fast_path_prompt("Hello") is True
    assert _is_fast_path_prompt("bạn là ai") is True
    assert _is_fast_path_prompt("cảm ơn") is True
    assert _is_fast_path_prompt("giúp tôi") is True
    assert _is_fast_path_prompt("Thời tiết hôm nay") is True
    
    # Complex or LMS domain queries should NOT be fast-path
    assert _is_fast_path_prompt("Cho tôi xem lịch thi môn Python") is False
    assert _is_fast_path_prompt("Thầy dạy môn Hướng đối tượng tên là gì?") is False

def test_lms_context_enrichment():
    """Test LMS domain queries inject grounded database facts into messages."""
    messages = [{"role": "user", "content": "Cho tôi xem thời khóa biểu môn Python"}]
    enriched = _enrich_lms_context(messages)
    assert len(enriched) > len(messages)
    assert enriched[0]["role"] == "system"
    assert "DỮ LIỆU THỰC TẾ TRÍCH XUẤT TỪ HỆ THỐNG LMS" in enriched[0]["content"]

def test_chat_completion_standard(client):
    """Test /api/v1/chat/completions standard JSON response."""
    mock_llm_response = {
        "text": "Xin chào! Tôi có thể giúp gì cho bạn?",
        "model": "ollama/qwen2.5:7b",
        "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22}
    }
    
    with patch("app.services.llm_factory.LLMFactory.chat_completion", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_llm_response
        
        response = client.post("/api/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Xin chào bạn"}],
            "bypass_cache": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["text"] == mock_llm_response["text"]
        assert data["model"] == mock_llm_response["model"]
        assert data["usage"]["total_tokens"] == 22

def test_chat_completion_streaming(client):
    """Test /api/v1/chat/completions SSE streaming response format."""
    async def mock_generator(*args, **kwargs):
        yield "data: {\"token\": \"Xin \"}\n\n"
        yield "data: {\"token\": \"chào!\"}\n\n"
        yield "data: [DONE]\n\n"

    with patch("app.services.llm_factory.LLMFactory.stream_completion", side_effect=mock_generator):
        response = client.post("/api/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Xin chào"}],
            "stream": True
        })
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "data: {\"token\": \"Xin \"}" in response.text
        assert "data: [DONE]" in response.text
