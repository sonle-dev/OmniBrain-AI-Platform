from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class BaseAPIResponse(BaseModel):
    status: str = Field("success", description="'success' or 'error'")
    cached: bool = Field(False, description="True if served from cache")
    error: Optional[str] = Field(None, description="Detailed error message if status is 'error'")


class ChatResponse(BaseAPIResponse):
    text: str = Field(..., description="Generated text response")
    model: str = Field(..., description="Model that generated the response")
    usage: TokenUsage = Field(default_factory=TokenUsage)


class QuizOption(BaseModel):
    label: str = Field(..., description="Option label: A, B, C, D")
    text: str = Field(..., description="Option text")


class QuizQuestion(BaseModel):
    id: int = Field(..., description="Question number (1-indexed)")
    question: str = Field(..., description="Question text")
    options: List[QuizOption] = Field(default_factory=list)
    correct_answer: str = Field(..., description="Correct option label (A/B/C/D or True/False)")
    explanation: str = Field(default="", description="Explanation of the correct answer")
    difficulty: str = Field(default="medium")


class QuizResponse(BaseAPIResponse):
    topic: str = Field(..., description="Quiz topic")
    language: str = Field("vi")
    count: int = Field(..., description="Number of generated questions")
    questions: List[QuizQuestion] = Field(default_factory=list)
    model: str = Field(default="")
    usage: TokenUsage = Field(default_factory=TokenUsage)


class GraphRAGQueryResponse(BaseAPIResponse):
    text: str = Field(..., description="Answer grounded in the Knowledge Graph")


class SearchItem(BaseModel):
    title: str
    url: str
    content: str


class SearchResponse(BaseAPIResponse):
    source: str = Field(..., description="Search engine used")
    results: List[SearchItem] = Field(default_factory=list)


class AgentTraceItem(BaseModel):
    tool: str
    arguments: Dict[str, Any]
    status: str
    result: str


class AgentResponse(BaseAPIResponse):
    text: str = Field(..., description="Synthesized agent response")
    trace: List[AgentTraceItem] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class VisionResponse(BaseAPIResponse):
    text: str = Field(..., description="OCR text or analysis output")
    model: str = Field(..., description="Vision model used")
    usage: TokenUsage = Field(default_factory=TokenUsage)


class AudioTranscribeResponse(BaseAPIResponse):
    text: str = Field(..., description="Transcribed speech-to-text string")
