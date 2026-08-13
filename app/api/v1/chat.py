import json
from fastapi import APIRouter, Depends, Request, HTTPException, status
from app.core.security import verify_api_key, enforce_rate_limit, track_token_usage
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse, TokenUsage
from app.core.config import settings

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/completions", response_model=ChatResponse)
async def chat_completion_endpoint(
    payload: ChatRequest,
    request: Request,
    app_id: str = Depends(verify_api_key)
):
    """
    Unified Multi-LLM Chat Endpoint with Semantic Cache and Multi-Tenant Budgets.
    """
    # 1. Enforce rate limits per App-ID
    client_ip = request.client.host if request.client else "127.0.0.1"
    rpm_limit = settings.api_keys_map.get(request.headers.get("X-API-Key"), {}).get("rate_limit_rpm", 60)
    enforce_rate_limit(app_id, client_ip, rpm_limit)

    # 2. Serialize messages for cache signature
    serialized_messages = json.dumps([m.model_dump() for m in payload.messages], sort_keys=True)
    
    # 3. Cache lookup
    if not payload.bypass_cache:
        # Lazy import of cache to maintain lazy-loaded architecture
        from app.core.cache import cache_engine
        cached_result = cache_engine.lookup(serialized_messages)
        if cached_result:
            try:
                cached_data = json.loads(cached_result)
                return ChatResponse(
                    status="success",
                    cached=True,
                    text=cached_data["text"],
                    model=cached_data["model"],
                    usage=TokenUsage(**cached_data["usage"])
                )
            except Exception:
                # Fallback to string if JSON parsing of cache failed
                return ChatResponse(
                    status="success",
                    cached=True,
                    text=cached_result,
                    model="cached-gateway"
                )

    # 4. LLM Completion call (Lazy loaded LLMFactory)
    from app.services.llm_factory import LLMFactory
    
    # Convert request schemas to list of dicts for litellm
    input_messages = [{"role": m.role, "content": m.content} for m in payload.messages]

    try:
        completion_res = await LLMFactory.chat_completion(
            messages=input_messages,
            model=payload.model,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens
        )
        
        text = completion_res["text"]
        model_used = completion_res["model"]
        usage = completion_res["usage"]
        
        # 5. Track tenant token usage
        total_tokens = usage.get("total_tokens", 0)
        track_token_usage(app_id, total_tokens)

        # 6. Store in Semantic Cache
        from app.core.cache import cache_engine
        cache_data_str = json.dumps({
            "text": text,
            "model": model_used,
            "usage": usage
        })
        cache_engine.store(serialized_messages, cache_data_str)

        return ChatResponse(
            status="success",
            cached=False,
            text=text,
            model=model_used,
            usage=TokenUsage(**usage)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(e)}"
        )
