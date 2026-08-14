import json
import logging
import re
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from app.core.security import verify_api_key, enforce_rate_limit, track_token_usage
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse, TokenUsage
from app.core.config import settings
from app.services.tools.lms_tools import get_lecturer_info, get_course_schedule, get_exam_schedule, get_transcripts

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

# Fast-path intent regex for simple greetings and general prompts
_GREETING_REGEX = re.compile(
    r"^(xin\s+chào|chào\s+bạn|hello|hi|hey|bạn\s+là\s+ai|tên\s+là\s+gì|cảm\s+ơn|thanks|cám\s+ơn|good\s+morning|good\s+evening|giúp\s+tôi|bạn\s+có\s+thể\s+làm\s+gì)[\s!\?\.]*$",
    re.IGNORECASE
)

# LMS domain keywords requiring database tool lookup
_LMS_KEYWORDS = [
    "giảng viên", "dạy môn", "thầy", "cô", "phòng làm việc", "sdt", "số điện thoại",
    "thời khóa biểu", "lịch học", "học thứ mấy", "tiết mấy", "phòng học", "mã lớp",
    "lịch thi", "ngày thi", "ca thi", "phòng thi",
    "bảng điểm", "điểm thi", "điểm chuyên cần", "tx1", "tx2", "tx3", "tx4", "điểm tbc", "kthp"
]


def _is_fast_path_prompt(prompt: str) -> bool:
    """
    Early exit classifier. Returns True if prompt is a simple greeting or general question
    that does NOT require database tool query or GraphRAG document extraction.
    """
    clean_p = prompt.strip()

    # 1. Match greeting patterns
    if _GREETING_REGEX.match(clean_p):
        return True

    # 2. Check if prompt lacks LMS domain keywords and is short
    lower_p = clean_p.lower()
    has_lms_keyword = any(kw in lower_p for kw in _LMS_KEYWORDS)

    if not has_lms_keyword and len(clean_p) < 25:
        return True

    return False


def _enrich_lms_context(input_messages: list) -> list:
    """
    Detects if user prompt asks for live LMS data.
    If detected, queries lms_tools and prepends grounded LMS facts to messages.
    """
    if not input_messages:
        return input_messages

    last_user_msg = next((m["content"] for m in reversed(input_messages) if m.get("role") == "user"), "")
    if not last_user_msg:
        return input_messages

    # Fast-Path Check: Early exit if prompt is general/greeting
    if _is_fast_path_prompt(last_user_msg):
        logger.info("Fast-Path triggered: Bypassing LMS database enrichment for prompt: '%s'", last_user_msg[:30])
        return input_messages

    lower_msg = last_user_msg.lower()
    lms_facts = []

    # Check lecturer queries
    if any(k in lower_msg for k in ["giảng viên", "dạy môn", "thầy", "cô", "phòng làm việc", "sdt"]):
        for subj in ["marketing", "python", "hướng đối tượng", "oop", "quản lý dự án", "quản trị hệ thống"]:
            if subj in lower_msg:
                res = get_lecturer_info(subj)
                if res.get("found"):
                    lms_facts.append(f"Thông tin giảng viên môn {res['subject_name']}: {res['lecturer_name']} (SĐT: {res.get('phone', 'N/A')}, Email: {res['email']}), Khoa: {res['department']}, Phòng làm việc: {res['office_room']}")

    # Check schedule queries
    if any(k in lower_msg for k in ["thời khóa biểu", "lịch học", "học thứ mấy", "tiết mấy", "phòng học"]):
        for subj in ["marketing", "python", "hướng đối tượng", "oop", "quản lý dự án", "quản trị hệ thống"]:
            if subj in lower_msg:
                res = get_course_schedule(subj)
                if res.get("found"):
                    for s in res.get("schedules", []):
                        lms_facts.append(f"Lịch học môn {s['subject_name']}: {s['day_of_week']}, {s['period']}, Phòng: {s['room']}, Mã lớp: {s['class_id']}")

    # Check exam schedule queries
    if any(k in lower_msg for k in ["lịch thi", "ngày thi", "ca thi", "phòng thi"]):
        res = get_exam_schedule(student_id="DTC235200647")
        if res.get("found"):
            for e in res.get("exams", []):
                lms_facts.append(f"Lịch thi môn {e['subject_name']} ({e['subject_code']}): Ngày {e['exam_date']}, Ca thi {e['exam_time']}, {e['exam_room']}")

    # Check transcripts queries
    if any(k in lower_msg for k in ["bảng điểm", "điểm thi", "điểm chuyên cần", "tx1", "tx2", "tx3", " điểm tbc"]):
        res = get_transcripts()
        if res.get("found"):
            for t in res.get("transcripts", []):
                lms_facts.append(f"Điểm môn {t['subject_name']} ({t['credits']} TC): C.Cần {t['attendance_score']}, TBCTN {t['process_score']}, TX1 {t['tx1']}, TX2 {t['tx2']}, TX3 {t['tx3']}, TBC {t['avg_score']}, Thi KTHP: {t['exam_eligibility']}")

    if lms_facts:
        logger.info("Enriched chat context with %d live LMS DB facts.", len(lms_facts))
        system_context = "DỮ LIỆU THỰC TẾ TRÍCH XUẤT TỪ HỆ THỐNG LMS:\n" + "\n".join(f"- {f}" for f in lms_facts) + "\n\nHãy sử dụng dữ liệu LMS thực tế ở trên để trả lời người dùng một cách chính xác và ngắn gọn."
        
        enriched = list(input_messages)
        if enriched and enriched[0]["role"] == "system":
            enriched[0]["content"] += f"\n\n{system_context}"
        else:
            enriched.insert(0, {"role": "system", "content": system_context})
        return enriched

    return input_messages


@router.post("/completions")
async def chat_completion_endpoint(
    payload: ChatRequest,
    request: Request,
    app_id: str = Depends(verify_api_key)
):
    """
    Unified Multi-LLM Chat Endpoint with Fast-Path Intent Classifier & Sub-Second TTFT.
    
    - Fast-Path: Early exit for greetings & general questions (< 300ms)
    - Standard: returns JSON ChatResponse
    - Streaming (stream=true): returns Server-Sent Events stream (Content-Type: text/event-stream)
    """
    # 1. Rate limiting
    client_ip = request.client.host if request.client else "127.0.0.1"
    rpm_limit = settings.api_keys_map.get(request.headers.get("X-API-Key"), {}).get("rate_limit_rpm", 60)
    enforce_rate_limit(app_id, client_ip, rpm_limit)

    # 2. Prepare messages and apply Fast-Path intent classification
    input_messages = [{"role": m.role, "content": m.content} for m in payload.messages]
    enriched_messages = _enrich_lms_context(input_messages)
    serialized = json.dumps(input_messages, sort_keys=True, ensure_ascii=False)

    # ── STREAMING PATH (text/event-stream) ──
    if payload.stream:
        from app.services.llm_factory import LLMFactory

        async def event_generator():
            async for chunk in LLMFactory.stream_completion(
                messages=enriched_messages,
                model=payload.model,
                task=payload.task.value,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            ):
                yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    # ── STANDARD PATH ──

    # 3. Cache lookup
    if not payload.bypass_cache:
        from app.core.cache import cache_engine
        cached_result = cache_engine.lookup(serialized)
        if cached_result:
            try:
                cached_data = json.loads(cached_result)
                return ChatResponse(
                    status="success",
                    cached=True,
                    text=cached_data["text"],
                    model=cached_data.get("model", "cached"),
                    usage=TokenUsage(**cached_data.get("usage", {}))
                )
            except Exception:
                pass

    # 4. LLM Completion
    from app.services.llm_factory import LLMFactory

    completion_res = await LLMFactory.chat_completion(
        messages=enriched_messages,
        model=payload.model,
        task=payload.task.value,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
    )

    if completion_res.get("error"):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "cached": False,
                "message": completion_res["text"],
            }
        )

    text = completion_res["text"]
    model_used = completion_res["model"]
    usage = completion_res["usage"]

    # 5. Track token usage
    track_token_usage(app_id, usage.get("total_tokens", 0))

    # 6. Store in cache
    from app.core.cache import cache_engine
    cache_engine.store(serialized, json.dumps({
        "text": text,
        "model": model_used,
        "usage": usage
    }, ensure_ascii=False))

    return ChatResponse(
        status="success",
        cached=False,
        text=text,
        model=model_used,
        usage=TokenUsage(**usage)
    )
