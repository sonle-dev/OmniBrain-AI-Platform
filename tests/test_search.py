from unittest.mock import patch, AsyncMock

def test_external_search_endpoint(client):
    """Test real-time external web search endpoint."""
    mock_search_result = {
        "source": "tavily",
        "results": [
            {
                "title": "Tin tức công nghệ AI 2026",
                "url": "https://news.example.com/ai-2026",
                "content": "Tổng hợp các mô hình AI mới nhất..."
            }
        ]
    }
    
    with patch("app.services.search_service.search_service.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_search_result
        
        response = client.post("/api/v1/search/query", json={
            "query": "Tin tức AI mới nhất",
            "max_results": 1
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["source"] == "tavily"
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Tin tức công nghệ AI 2026"
