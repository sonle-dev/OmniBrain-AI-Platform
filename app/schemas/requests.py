from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class TaskType(str, Enum):
    chat = "chat"
    reasoning = "reasoning"
    graph_rag = "graph_rag"
    vision = "vision"


class Message(BaseModel):
    role: str = Field(..., description="Role of the sender: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Content text of the message")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Chronological conversation list")
    model: Optional[str] = Field(None, description="Override default model")
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: bool = Field(False, description="Enable SSE token streaming")
    task: TaskType = Field(TaskType.chat, description="Task type for specialist routing")
    bypass_cache: bool = Field(False, description="Force fresh LLM call, skip cache")


class QuizGenerateRequest(BaseModel):
    topic: str = Field(..., description="Subject topic for quiz generation (e.g. 'Lập trình Python')")
    difficulty: str = Field("medium", description="Difficulty level: 'easy', 'medium', 'hard'")
    count: int = Field(5, ge=1, le=20, description="Number of questions to generate")
    language: str = Field("vi", description="Language code: 'vi' for Vietnamese, 'en' for English")
    question_type: str = Field("multiple_choice", description="'multiple_choice' or 'true_false'")
    context: Optional[str] = Field(None, description="Optional document context to base questions on")


class GraphRAGQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question targeting uploaded documents")
    mode: str = Field("local", description="Query mode: 'local' (entity/detail) or 'global' (summary/thematic)")
    namespace: str = Field("lms", description="Target knowledge base namespace partition (e.g. 'lms', 'ecommerce', 'travel')")


class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    max_results: int = Field(5, ge=1, le=20)


class AgentRunRequest(BaseModel):
    prompt: str = Field(..., description="The user instructions for the agent loop")
    history: Optional[List[Message]] = Field(None, description="Previous conversation steps")
    model: Optional[str] = Field(None, description="Override model")


class VisionAnalyzeRequest(BaseModel):
    prompt: str = Field("Describe this image and perform OCR extraction.", description="Analysis instructions")
    image_b64: Optional[str] = Field(None, description="Base64-encoded image (without data URI header)")
    image_url: Optional[str] = Field(None, description="Direct URL of the image")


class AudioSynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech")
    voice: str = Field("alloy", description="Voice identifier (alloy, echo, fable, onyx, nova, shimmer)")
    speed: float = Field(1.0, ge=0.25, le=4.0)
