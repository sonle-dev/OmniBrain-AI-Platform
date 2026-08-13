from fastapi import APIRouter, Depends, Request, HTTPException, status
from app.core.security import verify_api_key, enforce_rate_limit, track_token_usage
from app.schemas.requests import AgentRunRequest
from app.schemas.responses import AgentResponse, AgentTraceItem, TokenUsage
from app.core.config import settings

router = APIRouter(prefix="/agent", tags=["Agent"])

@router.post("/run", response_model=AgentResponse)
async def run_agent_endpoint(
    payload: AgentRunRequest,
    request: Request,
    app_id: str = Depends(verify_api_key)
):
    """
    Spawns an AI agent context allowing loop execution of tools and action synthesis.
    """
    # 1. Rate Limit Check
    client_ip = request.client.host if request.client else "127.0.0.1"
    rpm_limit = settings.api_keys_map.get(request.headers.get("X-API-Key"), {}).get("rate_limit_rpm", 60)
    enforce_rate_limit(app_id, client_ip, rpm_limit)

    # 2. Convert schemas message histories to raw list dictionaries
    history_list = None
    if payload.history:
        history_list = [{"role": m.role, "content": m.content} for m in payload.history]

    # 3. Call Agent Service
    from app.services.agent_service import agent_service
    try:
        agent_res = await agent_service.execute_agent(
            prompt=payload.prompt,
            history=history_list,
            model=payload.model
        )
        
        # Track usage tokens
        usage = agent_res.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        track_token_usage(app_id, total_tokens)

        # Parse trace list
        parsed_trace = [
            AgentTraceItem(
                tool=item.get("tool", ""),
                arguments=item.get("arguments", {}),
                status=item.get("status", ""),
                result=item.get("result", "")
            )
            for item in agent_res.get("trace", [])
        ]

        return AgentResponse(
            status="success",
            cached=False,
            text=agent_res.get("text", ""),
            trace=parsed_trace,
            usage=TokenUsage(**usage)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent runtime loop failed: {str(e)}"
        )
