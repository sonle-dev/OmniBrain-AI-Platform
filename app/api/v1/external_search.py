from fastapi import APIRouter, Depends, Request, HTTPException, status
from app.core.security import verify_api_key, enforce_rate_limit
from app.schemas.requests import SearchQueryRequest
from app.schemas.responses import SearchResponse, SearchItem
from app.core.config import settings

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("/query", response_model=SearchResponse)
async def query_external_search(
    payload: SearchQueryRequest,
    request: Request,
    app_id: str = Depends(verify_api_key)
):
    """
    Retrieves live web data for queries regarding news or current affairs outside books.
    """
    # 1. Rate Limit Check
    client_ip = request.client.host if request.client else "127.0.0.1"
    rpm_limit = settings.api_keys_map.get(request.headers.get("X-API-Key"), {}).get("rate_limit_rpm", 60)
    enforce_rate_limit(app_id, client_ip, rpm_limit)

    # 2. Lazy load the search service and run query
    from app.services.search_service import search_service
    try:
        search_res = await search_service.search_web(payload.query, payload.max_results)
        
        parsed_items = [
            SearchItem(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", "")
            )
            for item in search_res.get("results", [])
        ]
        
        return SearchResponse(
            status="success",
            cached=False,
            source=search_res.get("source", "unknown"),
            results=parsed_items
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Live search failed: {str(e)}"
        )
