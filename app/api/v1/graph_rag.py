import os
import shutil
from fastapi import APIRouter, Depends, Request, UploadFile, File, HTTPException, status
from app.core.security import verify_api_key, enforce_rate_limit
from app.schemas.requests import GraphRAGQueryRequest
from app.schemas.responses import GraphRAGQueryResponse
from app.core.config import settings

router = APIRouter(prefix="/graph-rag", tags=["GraphRAG"])

@router.post("/upload")
async def upload_document_endpoint(
    request: Request,
    file: UploadFile = File(...),
    app_id: str = Depends(verify_api_key)
):
    """
    Uploads a document (PDF, PowerPoint Slides, Text) to construct or update the GraphRAG index.
    """
    # 1. Rate Limit Check
    client_ip = request.client.host if request.client else "127.0.0.1"
    rpm_limit = settings.api_keys_map.get(request.headers.get("X-API-Key"), {}).get("rate_limit_rpm", 60)
    enforce_rate_limit(app_id, client_ip, rpm_limit)

    # Validate file extension
    filename = file.filename or "uploaded_document"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".pptx", ".ppt", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload PDF, PPTX slides, or plain text (.txt) files."
        )

    # 2. Save file temporarily
    temp_dir = os.path.join(settings.GRAPHRAG_WORKING_DIR, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, filename)

    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Ingest via GraphRAG Service (lazy-loaded inside endpoint or service method)
        from app.services.graph_rag_service import graph_rag_service
        await graph_rag_service.insert_document(temp_filepath)
        
        return {
            "status": "success",
            "message": f"Document '{filename}' successfully parsed and ingested into GraphRAG."
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document into GraphRAG: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass

@router.post("/query", response_model=GraphRAGQueryResponse)
async def query_graph_endpoint(
    payload: GraphRAGQueryRequest,
    request: Request,
    app_id: str = Depends(verify_api_key)
):
    """
    Grounded Query over the GraphRAG Knowledge Graph. Strictly respects local context.
    """
    # 1. Rate Limit Check
    client_ip = request.client.host if request.client else "127.0.0.1"
    rpm_limit = settings.api_keys_map.get(request.headers.get("X-API-Key"), {}).get("rate_limit_rpm", 60)
    enforce_rate_limit(app_id, client_ip, rpm_limit)

    # 2. Invoke GraphRAG Service query
    from app.services.graph_rag_service import graph_rag_service
    try:
        answer = await graph_rag_service.query_graph(payload.query, mode=payload.mode)
        return GraphRAGQueryResponse(
            status="success",
            cached=False,
            text=answer
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GraphRAG query execution failed: {str(e)}"
        )
