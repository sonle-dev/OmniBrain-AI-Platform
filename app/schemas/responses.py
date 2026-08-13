from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class BaseAPIResponse(BaseModel):
    status: str = Field("success", description="'success' or 'error'")
    cached: bool = Field(False, description="True if served from cache")
    error: Optional[str] = Field(None, description="Detailed error message if failed")

class ChatResponse(BaseAPIResponse):
    text: str = Field(..., description="Generated text response from the model")
    model: str = Field(..., description="Name of the model that generated response")
    usage: TokenUsage = Field(default_factory=TokenUsage)

class GraphRAGQueryResponse(BaseAPIResponse):
    text: str = Field(..., description="Ground truth answer derived strictly from internal Knowledge Graph")

class SearchItem(BaseModel):
    title: str
    url: str
    content: str

class SearchResponse(BaseAPIResponse):
    source: str = Field(..., description="Engine source, e.g. 'tavily', 'duckduckgo'")
    results: List[SearchItem] = Field(default_factory=list)

class AgentTraceItem(BaseModel):
    tool: str
    arguments: Dict[str, Any]
    status: str
    result: str

class AgentResponse(BaseAPIResponse):
    text: str = Field(..., description="Synthesized response from the agent")
    trace: List[AgentTraceItem] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)

class VisionResponse(BaseAPIResponse):
    text: str = Field(..., description="OCR text or analysis results")
    model: str = Field(..., description="Model used for vision task")
    usage: TokenUsage = Field(default_factory=TokenUsage)

class AudioTranscribeResponse(BaseAPIResponse):
    text: str = Field(..., description="Transcribed speech-to-text string")
