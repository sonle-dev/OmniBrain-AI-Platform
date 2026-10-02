from unittest.mock import patch, AsyncMock

def test_agent_run_endpoint(client):
    """Test AI agent multi-step execution loop with tool trace output."""
    mock_agent_result = {
        "text": "Tôi đã kiểm tra thời khóa biểu và lịch thi cho bạn.",
        "trace": [
            {
                "tool": "get_course_schedule",
                "arguments": {"subject": "python"},
                "status": "success",
                "result": "Thứ 2, Tiết 1-3, Phòng 301"
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 30, "total_tokens": 70}
    }
    
    with patch("app.services.agent_service.agent_service.execute_agent", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_agent_result
        
        response = client.post("/api/v1/agent/run", json={
            "prompt": "Kiểm tra lịch học môn Python giúp tôi"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["trace"]) == 1
        assert data["trace"][0]["tool"] == "get_course_schedule"
        assert data["usage"]["total_tokens"] == 70
