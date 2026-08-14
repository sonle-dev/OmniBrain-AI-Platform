import logging
import os
import asyncio
import httpx
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable, Tuple, Set
from app.core.config import settings

logger = logging.getLogger(__name__)

# Persistent Async HTTP Connection Pool with Keep-Alive Expiry 600s
_shared_async_client: Optional[httpx.AsyncClient] = None


def get_shared_async_client() -> httpx.AsyncClient:
    """Returns global singleton httpx.AsyncClient with persistent connection pooling."""
    global _shared_async_client
    if _shared_async_client is None or _shared_async_client.is_closed:
        _shared_async_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=200,
                keepalive_expiry=600.0
            ),
            timeout=httpx.Timeout(10.0, connect=3.0)
        )
        logger.info("Initialized persistent AsyncClient pool (keepalive_expiry=600s).")
    return _shared_async_client


def _inject_dynamic_system_clock(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Injects the current dynamic date, day of week, and time (timezone: Asia/Ho_Chi_Minh, GMT+7)
    into the system prompt so the LLM always has accurate time awareness without hallucinating cutoff dates.
    """
    tz_vn = timezone(timedelta(hours=7))
    now = datetime.now(tz_vn)

    weekdays_vi = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    weekday_str = weekdays_vi[now.weekday()]

    clock_info = (
        f"THỜI GIAN HỆ THỐNG THỰC TẾ HÔM NAY: {weekday_str}, Ngày {now.strftime('%d/%m/%Y')} | Giờ: {now.strftime('%H:%M:%S')} (Múi giờ Asia/Ho_Chi_Minh, GMT+7).\n"
        f"LƯU Ý: Năm hiện tại là {now.year}. Bạn luôn nhận biết chính xác thời gian thực tế này để trả lời các câu hỏi về lịch trình, môn học, ngày tháng."
    )

    formatted = list(messages) if messages else []
    if formatted and formatted[0].get("role") == "system":
        existing_sys = formatted[0]["content"]
        formatted[0] = {
            "role": "system",
            "content": f"{clock_info}\n\n{existing_sys}"
        }
    else:
        formatted.insert(0, {
            "role": "system",
            "content": clock_info
        })
    return formatted


# Ollama health check result & installed models cache (re-checked every 60s)
_ollama_reachable: Optional[bool] = None
_ollama_installed_models: Set[str] = set()
_ollama_last_check: float = 0.0
_OLLAMA_CHECK_INTERVAL = 60.0


async def _check_ollama_reachable() -> Tuple[bool, Set[str]]:
    """Ping Ollama once per minute using persistent connection pool."""
    global _ollama_reachable, _ollama_installed_models, _ollama_last_check
    import time
    now = time.time()
    if _ollama_reachable is not None and (now - _ollama_last_check) < _OLLAMA_CHECK_INTERVAL:
        return _ollama_reachable, _ollama_installed_models

    try:
        client = get_shared_async_client()
        r = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
        if r.status_code == 200:
            data = r.json()
            models_data = data.get("models", [])
            tags = set()
            for m in models_data:
                name = m.get("name", "")
                if name:
                    tags.add(name)
                    if ":latest" in name:
                        tags.add(name.replace(":latest", ""))
                    else:
                        tags.add(f"{name}:latest")
            _ollama_reachable = True
            _ollama_installed_models = tags
        else:
            _ollama_reachable = False
            _ollama_installed_models = set()
    except Exception:
        _ollama_reachable = False
        _ollama_installed_models = set()

    _ollama_last_check = now
    logger.info("Ollama reachable: %s (installed tags: %s)", _ollama_reachable, len(_ollama_installed_models))
    return _ollama_reachable, _ollama_installed_models


def _get_api_key_for_model(model: str) -> Optional[str]:
    """Return the correct API key for a given model prefix."""
    if model.startswith("openrouter/"):
        return settings.OPENROUTER_API_KEY
    if model.startswith("gemini/"):
        return settings.GEMINI_API_KEY
    if model.startswith("deepseek/"):
        return settings.DEEPSEEK_API_KEY
    if model.startswith("openai/") or model.startswith("gpt"):
        return settings.OPENAI_API_KEY
    if model.startswith("anthropic/") or model.startswith("claude"):
        return settings.ANTHROPIC_API_KEY
    return None


async def _build_model_chain(task: str, override_model: Optional[str] = None) -> List[str]:
    """
    Build the ordered list of models to try for a given task.
    - Checks if Ollama is reachable and whether the specific LoRA/base tag is installed locally.
    - Only includes cloud models with valid non-empty API keys.
    """
    specialist_map = settings.SPECIALIST_MODELS
    base_chain: List[str] = specialist_map.get(task, specialist_map["chat"])

    ollama_ok, installed_tags = await _check_ollama_reachable()

    filtered: List[str] = []
    for model in base_chain:
        if model.startswith("ollama/"):
            if ollama_ok:
                clean_tag = model.replace("ollama/", "")
                if not installed_tags or clean_tag in installed_tags or any(clean_tag in tag for tag in installed_tags):
                    filtered.append(model)
                else:
                    logger.info("LoRA / Ollama model '%s' not found locally in tags. Falling back to next candidate.", model)
        else:
            key = _get_api_key_for_model(model)
            if key and key.strip():
                filtered.append(model)

    if override_model and override_model not in filtered:
        filtered.insert(0, override_model)

    if not filtered:
        logger.warning("No available models for task '%s'. All local+cloud options exhausted.", task)

    return filtered


def _lazy_litellm():
    """Lazy-load LiteLLM to avoid startup overhead."""
    import litellm
    litellm.telemetry = False
    return litellm


class LLMFactory:
    """
    Specialist Dispatcher + Dynamic Real-Time Clock Injection + Fallback Chain AI Gateway.
    Injects real-time Asia/Ho_Chi_Minh timestamp into system prompt for 100% accurate temporal awareness.
    """

    _GRACEFUL_ERROR = {
        "text": "Hệ thống AI hiện đang bận hoặc chưa được cấu hình API Key / Ollama. Vui lòng kiểm tra lại cấu hình.",
        "model": "none",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error": True
    }

    @classmethod
    async def chat_completion(
        cls,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        task: str = "chat",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run chat completion with dynamic system clock injection & VRAM-pinned execution.
        """
        litellm = _lazy_litellm()
        models_to_try = await _build_model_chain(task, override_model=model)

        if not models_to_try:
            return cls._GRACEFUL_ERROR

        # Inject real-time system clock into messages
        clock_messages = _inject_dynamic_system_clock(messages)

        last_error = None
        for current_model in models_to_try:
            try:
                logger.info("[%s] Trying model: %s", task.upper(), current_model)
                call_kwargs: Dict[str, Any] = dict(kwargs)

                api_key = _get_api_key_for_model(current_model)
                if api_key:
                    call_kwargs["api_key"] = api_key

                if response_format and not current_model.startswith("openrouter/"):
                    call_kwargs["response_format"] = response_format

                if current_model.startswith("ollama/"):
                    call_kwargs["api_base"] = settings.OLLAMA_BASE_URL
                    call_kwargs["timeout"] = 5.0
                    call_kwargs["extra_body"] = {"keep_alive": -1}
                else:
                    call_kwargs["timeout"] = timeout

                response = await litellm.acompletion(
                    model=current_model,
                    messages=clock_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **call_kwargs
                )

                text = response.choices[0].message.content or ""
                usage = response.usage
                return {
                    "text": text,
                    "model": current_model,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                    },
                    "error": False
                }

            except Exception as e:
                logger.warning("Model %s failed (%s: %s). Trying next.", current_model, type(e).__name__, str(e)[:120])
                last_error = e
                continue

        logger.error("All models exhausted for task '%s'. Last error: %s", task, last_error)
        return cls._GRACEFUL_ERROR

    @classmethod
    async def stream_completion(
        cls,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        task: str = "chat",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion as an async generator of SSE-formatted text chunks.
        Injects dynamic system clock into system prompt.
        """
        litellm = _lazy_litellm()
        models_to_try = await _build_model_chain(task, override_model=model)

        if not models_to_try:
            yield 'data: {"chunk": "Hệ thống AI chưa sẵn sàng. Vui lòng kiểm tra cấu hình."}\n\n'
            yield 'data: [DONE]\n\n'
            return

        clock_messages = _inject_dynamic_system_clock(messages)

        for current_model in models_to_try:
            try:
                logger.info("[STREAM/%s] Trying model: %s", task.upper(), current_model)
                call_kwargs: Dict[str, Any] = dict(kwargs)

                api_key = _get_api_key_for_model(current_model)
                if api_key:
                    call_kwargs["api_key"] = api_key

                if current_model.startswith("ollama/"):
                    call_kwargs["api_base"] = settings.OLLAMA_BASE_URL
                    call_kwargs["timeout"] = 5.0
                    call_kwargs["extra_body"] = {"keep_alive": -1}
                else:
                    call_kwargs["timeout"] = 10.0

                response = await litellm.acompletion(
                    model=current_model,
                    messages=clock_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **call_kwargs
                )

                async for chunk in response:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        import json
                        yield f'data: {json.dumps({"chunk": content})}\n\n'

                yield 'data: [DONE]\n\n'
                return

            except Exception as e:
                logger.warning("Stream model %s failed: %s. Trying next.", current_model, str(e)[:120])
                continue

        yield 'data: {"chunk": "Xin lỗi, hệ thống AI hiện không phản hồi."}\n\n'
        yield 'data: [DONE]\n\n'

    @classmethod
    async def chat_completion_with_self_correction(
        cls,
        messages: List[Dict[str, str]],
        parser_fn: Callable[[str], Any],
        model: Optional[str] = None,
        task: str = "reasoning",
        max_retries: int = 3,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute completion and validate output with parser_fn.
        """
        current_messages = list(messages)
        retries = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while retries < max_retries:
            res = await cls.chat_completion(
                messages=current_messages,
                model=model,
                task=task,
                **kwargs
            )
            for k in total_usage:
                total_usage[k] += res["usage"].get(k, 0)

            raw_text = res.get("text", "")
            try:
                parsed = parser_fn(raw_text)
                res["usage"] = total_usage
                return parsed, res
            except Exception as e:
                retries += 1
                logger.warning("Self-correction tier %d/%d failed: %s", retries, max_retries, e)
                current_messages.append({"role": "assistant", "content": raw_text})
                current_messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response failed validation. Error: {str(e)}. "
                        "Output ONLY a valid JSON object matching the requested schema. "
                        "No explanations, no markdown backticks."
                    )
                })

        raise ValueError(f"Failed to generate valid structured output after {max_retries} attempts.")
