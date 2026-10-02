import io
from unittest.mock import patch, AsyncMock

def test_audio_transcribe_endpoint(client):
    """Test Speech-To-Text transcription endpoint with uploaded audio file."""
    audio_bytes = io.BytesIO(b"fake audio data stream")
    
    with patch("litellm.atranscription", new_callable=AsyncMock) as mock_stt:
        mock_stt.return_value = {"text": "Chào mừng bạn đến với OmniBrain AI."}
        
        response = client.post(
            "/api/v1/audio/transcribe",
            files={"file": ("sample.mp3", audio_bytes, "audio/mpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "OmniBrain AI" in data["text"]

def test_audio_synthesize_endpoint(client):
    """Test Text-To-Speech synthesis streaming response."""
    class MockResponse:
        content = b"ID3\x03\x00\x00\x00\x00\x00%TSSE\x00\x00\x00\x0f\x00\x00\x00"
        
    with patch("litellm.aspeech", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = MockResponse()
        
        response = client.post("/api/v1/audio/synthesize", json={
            "text": "Xin chào thế giới!",
            "voice": "alloy",
            "speed": 1.0
        })
        
        assert response.status_code == 200
        assert "audio/mpeg" in response.headers["content-type"]
        assert len(response.content) > 0
