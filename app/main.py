from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.v1.chat import router as chat_router
from app.api.v1.graph_rag import router as graph_rag_router
from app.api.v1.external_search import router as search_router
from app.api.v1.agent import router as agent_router
from app.api.v1.vision import router as vision_router
from app.api.v1.audio import router as audio_router
from app.api.v1.quiz import router as quiz_router

app = FastAPI(
    title="OmniBrain AI Platform — Master Hybrid Engine",
    description=(
        "Production-grade centralized AI Gateway microservice. "
        "Supports intelligent task-based routing (chat, reasoning, vision, GraphRAG), "
        "LoRA adapter integration, LMS Database live tool extraction, SSE streaming, and "
        "universal JS SDK integration."
    ),
    version="2.0.0"
)

# ── CORS: Allow all origins for universal cross-app integration ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Serve Universal JS SDK at /static/omnibrain-client.js ──
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Register API Routers ──
app.include_router(chat_router,       prefix="/api/v1")
app.include_router(graph_rag_router,  prefix="/api/v1")
app.include_router(search_router,     prefix="/api/v1")
app.include_router(agent_router,      prefix="/api/v1")
app.include_router(vision_router,     prefix="/api/v1")
app.include_router(audio_router,      prefix="/api/v1")
app.include_router(quiz_router,       prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Service liveness probe.
    Returns Ollama reachability, installed model tags, and active cloud key count.
    """
    from app.services.llm_factory import _check_ollama_reachable
    from app.core.config import settings

    ollama_ok, installed_tags = await _check_ollama_reachable()
    active_keys = sum(1 for k in [
        settings.GEMINI_API_KEY,
        settings.OPENROUTER_API_KEY,
        settings.OPENAI_API_KEY,
        settings.DEEPSEEK_API_KEY,
        settings.ANTHROPIC_API_KEY,
    ] if k and k.strip())

    return {
        "status": "success",
        "version": "2.0.0",
        "ollama_reachable": ollama_ok,
        "ollama_tags_count": len(installed_tags),
        "cloud_keys_active": active_keys,
        "message": "OmniBrain Master Hybrid Engine is running."
    }


# ── Global exception handler: always return clean JSON ──
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
