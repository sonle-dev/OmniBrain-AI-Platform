from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Message(BaseModel):
    role: str = Field(..., description="Role of the sender, e.g., 'user', 'assistant', 'system'")
    content: str = Field(..., description="Content text of the message")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Chronological conversation list")
    model: Optional[str] = Field(None, description="Model to override default")
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    bypass_cache: bool = Field(False, description="Set to True to force fresh LLM run bypassing semantic cache")

class GraphRAGQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question targeting uploaded documents")
    mode: str = Field("local", description="Query mode: 'local' (detailed/entity) or 'global' (summarized/thematic)")

class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    max_results: int = Field(5, ge=1, le=20)

class AgentRunRequest(BaseModel):
    prompt: str = Field(..., description="The user instructions for the agent loop")
    history: Optional[List[Message]] = Field(None, description="Previous steps in chat conversation")
    model: Optional[str] = Field(None, description="Model to override default")

class VisionAnalyzeRequest(BaseModel):
    prompt: str = Field("Describe this image and perform OCR extraction.", description="OCR or analysis instructions")
    image_b64: Optional[str] = Field(None, description="Base64 encoded string of image (without header)")
    image_url: Optional[str] = Field(None, description="Direct URL of the image")

class AudioSynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text content to synthesize into speech")
    voice: str = Field("alloy", description="Speech voice identifier (alloy, echo, fable, onyx, nova, shimmer)")
    speed: float = Field(1.0, ge=0.25, le=4.0)
