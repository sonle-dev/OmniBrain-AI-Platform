from fastapi import APIRouter, Depends, Request, HTTPException, status
from app.core.security import verify_api_key, enforce_rate_limit, track_token_usage
from app.schemas.requests import VisionAnalyzeRequest
from app.schemas.responses import VisionResponse, TokenUsage
from app.core.config import settings

router = APIRouter(prefix="/vision", tags=["Vision"])

@router.post("/analyze", response_model=VisionResponse)
async def analyze_image_endpoint(
    payload: VisionAnalyzeRequest,
    request: Request,
    app_id: str = Depends(verify_api_key)
):
    """
    Analyzes images (accepts URLs or base64) to perform OCR extraction, label detection, or semantic answering.
    """
    # 1. Rate Limit Check
    client_ip = request.client.host if request.client else "127.0.0.1"
    rpm_limit = settings.api_keys_map.get(request.headers.get("X-API-Key"), {}).get("rate_limit_rpm", 60)
    enforce_rate_limit(app_id, client_ip, rpm_limit)

    # Validate image input presence
    if not payload.image_b64 and not payload.image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must supply either 'image_url' or 'image_b64' parameters."
        )

    # 2. Format the payload for multi-modal completion
    target_image_url = payload.image_url
    if not target_image_url and payload.image_b64:
        # standard base64 format header prefix
        target_image_url = f"data:image/jpeg;base64,{payload.image_b64}"

    prompt_content = [
        {"type": "text", "text": payload.prompt},
        {
            "type": "image_url",
            "image_url": {"url": target_image_url}
        }
    ]

    messages = [
        {
            "role": "user",
            "content": prompt_content
        }
    ]

    # 3. Call LLM factory (which uses LiteLLM)
    from app.services.llm_factory import LLMFactory
    
    # Use standard default vision-capable models (e.g. gemini-2.0-flash or gpt-4o-mini)
    vision_model = settings.DEFAULT_CHAT_MODEL
    # Quick model mapping check for safety
    if "ollama" in vision_model:
        # Fallback to standard multimodal cloud model if ollama is default chat
        vision_model = "gemini/gemini-2.0-flash"

    try:
        completion_res = await LLMFactory.chat_completion(
            messages=messages,
            model=vision_model,
            temperature=0.2
        )
        
        text = completion_res["text"]
        model_used = completion_res["model"]
        usage = completion_res["usage"]

        # Track usage token
        total_tokens = usage.get("total_tokens", 0)
        track_token_usage(app_id, total_tokens)

        return VisionResponse(
            status="success",
            cached=False,
            text=text,
            model=model_used,
            usage=TokenUsage(**usage)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision process failed: {str(e)}"
        )
