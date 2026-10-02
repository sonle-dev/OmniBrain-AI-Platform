import io
from unittest.mock import patch, AsyncMock

def test_graph_rag_upload_invalid_extension(client):
    """Test file upload endpoint rejects unsupported extensions like .exe or .png."""
    file_bytes = io.BytesIO(b"dummy binary content")
    response = client.post(
        "/api/v1/graph-rag/upload",
        files={"file": ("unsupported_file.png", file_bytes, "image/png")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_graph_rag_upload_success(client):
    """Test uploading valid document (.txt) to GraphRAG ingestion pipeline."""
    file_bytes = io.BytesIO(b"Giao trinh Mon Hoc Python ICTU")
    
    with patch("app.services.graph_rag_service.graph_rag_service.insert_document", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = True
        
        response = client.post(
            "/api/v1/graph-rag/upload",
            files={"file": ("curriculum.txt", file_bytes, "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "curriculum.txt" in data["message"]
        mock_ingest.assert_called_once()

def test_graph_rag_query(client):
    """Test GraphRAG query endpoint over knowledge graph."""
    with patch("app.services.graph_rag_service.graph_rag_service.query_graph", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = "Theo tài liệu ICTU, Python là môn học bắt buộc của ngành CNTT."
        
        response = client.post("/api/v1/graph-rag/query", json={
            "query": "Môn Python có bắt buộc không?",
            "mode": "hybrid"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "môn học bắt buộc" in data["text"]
