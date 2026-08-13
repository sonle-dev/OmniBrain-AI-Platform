import json
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Unified LLM Gateway providing automatic fallbacks and output self-correction.
    """
    
    @staticmethod
    def _lazy_import_litellm():
        """Lazy loads litellm to avoid startup overhead."""
        import litellm
        # Disable litellm telemetries or verbose logs if necessary
        litellm.telemetry = False
        return litellm

    @classmethod
    async def chat_completion(
        cls,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Runs chat completion with fallback models if the primary model fails.
        """
        litellm = cls._lazy_import_litellm()
        
        # Decide which models to try
        primary_model = model or settings.DEFAULT_CHAT_MODEL
        models_to_try = [primary_model] + [m for m in settings.FALLBACK_CHAT_MODELS if m != primary_model]
        
        last_error = None
        for current_model in models_to_try:
            try:
                logger.info(f"Attempting LLM call using model: {current_model}")
                
                # Make the API call using LiteLLM (which supports async)
                response = await litellm.acompletion(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs
                )
                
                # Extract text and token usage
                choice = response.choices[0]
                text = choice.message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                return {
                    "text": text,
                    "model": current_model,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
            except Exception as e:
                logger.warning(f"Model {current_model} failed. Error: {e}")
                last_error = e
                continue
                
        # If all models fail
        raise RuntimeError(f"All LLM models in fallback chain failed. Last error: {last_error}")

    @classmethod
    async def chat_completion_with_self_correction(
        cls,
        messages: List[Dict[str, str]],
        parser_fn: Callable[[str], Any],
        model: Optional[str] = None,
        max_retries: int = 3,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes a completion and runs it through a parser_fn. If parser_fn raises an exception,
        it feeds the error back to the LLM for self-correction.
        
        Returns:
            Tuple of (parsed_output, completion_result_dict)
        """
        # Ensure json response format if we are parsing structured data
        response_format = {"type": "json_object"}
        
        # Copy history so we don't pollute the caller's messages list
        current_messages = list(messages)
        
        retries = 0
        total_tokens_used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        while retries < max_retries:
            res = await cls.chat_completion(
                messages=current_messages,
                model=model,
                response_format=response_format,
                **kwargs
            )
            
            # Update token usage metrics
            for k in total_tokens_used:
                total_tokens_used[k] += res["usage"].get(k, 0)
                
            raw_text = res["text"]
            
            try:
                # Attempt to parse
                parsed_data = parser_fn(raw_text)
                res["usage"] = total_tokens_used
                return parsed_data, res
            except Exception as e:
                retries += 1
                logger.warning(f"Self-correction loop tier {retries} failed to parse JSON. Error: {e}")
                
                # Append the failed output and the error message to instructions
                current_messages.append({"role": "assistant", "content": raw_text})
                correction_instruction = (
                    f"Your previous response failed validation/parsing. "
                    f"Error detail: {str(e)}. "
                    f"Please output a corrected, valid JSON object that strictly adheres to the requested schema. "
                    f"Ensure you do not return any conversational text, backticks, or wrapping, just a clean JSON string."
                )
                current_messages.append({"role": "user", "content": correction_instruction})
                
        raise ValueError(f"Failed to generate correct structured output after {max_retries} attempts.")
