from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.chat import router as chat_router
from app.api.v1.graph_rag import router as graph_rag_router
from app.api.v1.external_search import router as search_router
from app.api.v1.agent import router as agent_router
from app.api.v1.vision import router as vision_router
from app.api.v1.audio import router as audio_router

app = FastAPI(
    title="OmniBrain AI Platform Core API",
    description="Centralized, modular AI microservice supplying Chat, GraphRAG, Search, Agents, Vision, and Audio endpoints.",
    version="1.0.0"
)

# Enable CORS for cross-origin client apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register endpoints
app.include_router(chat_router, prefix="/api/v1")
app.include_router(graph_rag_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")
app.include_router(vision_router, prefix="/api/v1")
app.include_router(audio_router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
def health_check():
    """Service health status endpoint."""
    return {"status": "success", "message": "OmniBrain AI core is healthy and running."}

# Global exception handler for unified JSON error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "cached": False,
            "error": f"Internal Server Error: {str(exc)}"
        }
    )
