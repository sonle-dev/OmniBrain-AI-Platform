import json
import logging
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from app.core.security import verify_api_key, enforce_rate_limit, track_token_usage
from app.schemas.requests import QuizGenerateRequest
from app.schemas.responses import QuizResponse, QuizQuestion, QuizOption, TokenUsage
from app.core.config import settings

router = APIRouter(prefix="/quiz", tags=["Quiz"])
logger = logging.getLogger(__name__)


def _build_quiz_prompt(req: QuizGenerateRequest) -> str:
    lang_instruction = "Respond entirely in Vietnamese." if req.language == "vi" else "Respond in English."
    q_type_desc = (
        "multiple-choice with 4 options (A, B, C, D)"
        if req.question_type == "multiple_choice"
        else "True/False"
    )
    context_section = f"\n\nBase questions on this content:\n{req.context}" if req.context else ""
    return f"""
You are an expert educational quiz generator. {lang_instruction}

Generate exactly {req.count} {req.difficulty} difficulty {q_type_desc} questions about: "{req.topic}".{context_section}

Return ONLY a valid JSON object (no markdown, no explanations outside JSON) in this exact format:
{{
  "questions": [
    {{
      "id": 1,
      "question": "Question text here?",
      "options": [
        {{"label": "A", "text": "Option A text"}},
        {{"label": "B", "text": "Option B text"}},
        {{"label": "C", "text": "Option C text"}},
        {{"label": "D", "text": "Option D text"}}
      ],
      "correct_answer": "A",
      "explanation": "Brief explanation why A is correct.",
      "difficulty": "{req.difficulty}"
    }}
  ]
}}

Ensure all {req.count} questions are distinct, accurate, and pedagogically sound.
"""


def _parse_quiz_json(raw: str, req: QuizGenerateRequest) -> list:
    """Extract and validate quiz JSON from raw LLM output."""
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)
    questions_raw = data.get("questions", [])
    questions = []
    for q in questions_raw:
        options = [QuizOption(label=o["label"], text=o["text"]) for o in q.get("options", [])]
        questions.append(QuizQuestion(
            id=q.get("id", len(questions) + 1),
            question=q["question"],
            options=options,
            correct_answer=q["correct_answer"],
            explanation=q.get("explanation", ""),
            difficulty=q.get("difficulty", req.difficulty)
        ))
    if not questions:
        raise ValueError("No questions found in LLM output")
    return questions


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(
    payload: QuizGenerateRequest,
    request: Request,
    app_id: str = Depends(verify_api_key)
):
    """
    Generate a structured multiple-choice quiz for a given topic.
    Uses the 'reasoning' task specialist model (deepseek-r1 locally, cloud fallback).
    Designed for Spring Boot LMS backend integration.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    rpm_limit = settings.api_keys_map.get(request.headers.get("X-API-Key"), {}).get("rate_limit_rpm", 60)
    enforce_rate_limit(app_id, client_ip, rpm_limit)

    prompt = _build_quiz_prompt(payload)
    messages = [
        {"role": "system", "content": "You are an expert educational quiz generator. Always respond with pure JSON only."},
        {"role": "user", "content": prompt}
    ]

    from app.services.llm_factory import LLMFactory

    try:
        questions, completion_res = await LLMFactory.chat_completion_with_self_correction(
            messages=messages,
            parser_fn=lambda raw: _parse_quiz_json(raw, payload),
            task="reasoning",
            temperature=0.4,
            max_tokens=4096,
        )

        usage = completion_res.get("usage", {})
        track_token_usage(app_id, usage.get("total_tokens", 0))

        return QuizResponse(
            status="success",
            topic=payload.topic,
            language=payload.language,
            count=len(questions),
            questions=questions,
            model=completion_res.get("model", ""),
            usage=TokenUsage(**usage)
        )

    except Exception as e:
        logger.error("Quiz generation failed: %s", e)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "cached": False,
                "message": f"Quiz generation failed: {str(e)}"
            }
        )
