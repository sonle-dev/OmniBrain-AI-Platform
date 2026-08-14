import os
from typing import Optional, Dict, List, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Force load all environment variables from .env into system os.environ
load_dotenv(override=True)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ── General API keys for OmniBrain Core client authentication ──
    API_KEYS_RAW: str = Field(
        default="demo_key:demo_app:1000000:60",
        alias="OMNIBRAIN_API_KEYS"
    )

    # ── Database Config for LMS Live Data Extraction ──
    LMS_DB_URL: Optional[str] = Field(default=None, alias="LMS_DB_URL")

    # ── Redis Config ──
    REDIS_URL: str = Field(default="", alias="OMNIBRAIN_REDIS_URL")

    # ── SQLite backup file paths ──
    SQLITE_CACHE_DB: str = "semantic_cache.db"
    SQLITE_METRICS_DB: str = "metrics.db"

    # ── Ollama (Local) & Custom Fine-Tuned LoRA Adapters ──
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    LORA_CHAT_ADAPTER: str = Field(default="ollama/qwen2.5:7b-academic", alias="LORA_CHAT_ADAPTER")
    LORA_REASONING_ADAPTER: str = Field(default="ollama/deepseek-r1:8b-pedagogical", alias="LORA_REASONING_ADAPTER")

    # ── Cloud AI Keys (all Optional — system boots cleanly if missing) ──
    GEMINI_API_KEY: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # ── External APIs ──
    TAVILY_API_KEY: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    NARA_API_KEY: Optional[str] = Field(default=None, alias="NARA_API_KEY")

    # ── Semantic Cache ──
    SEMANTIC_CACHE_ENABLED: bool = Field(default=True, alias="SEMANTIC_CACHE_ENABLED")
    SEMANTIC_CACHE_MAX_MEMORY: int = Field(default=128, alias="SEMANTIC_CACHE_MAX_MEMORY")

    # ── GraphRAG ──
    GRAPHRAG_WORKING_DIR: str = "graphrag_data"

    # ── Specialist Model Routing Table ──
    # Format: task_name -> [primary_lora_local, base_local, ...cloud_fallbacks]
    @property
    def SPECIALIST_MODELS(self) -> Dict[str, List[str]]:
        """Returns ordered model list per task. Ollama LoRA adapters & local models first, cloud fallbacks after."""
        cloud_chat = self._build_cloud_chain([
            ("openrouter/google/gemma-4-26b-a4b-it:free", self.OPENROUTER_API_KEY),
            ("openrouter/openai/gpt-oss-20b:free", self.OPENROUTER_API_KEY),
            ("gemini/gemini-2.0-flash", self.GEMINI_API_KEY),
            ("deepseek/deepseek-chat", self.DEEPSEEK_API_KEY),
            ("openai/gpt-4o-mini", self.OPENAI_API_KEY),
        ])
        cloud_reasoning = self._build_cloud_chain([
            ("openrouter/openai/gpt-oss-20b:free", self.OPENROUTER_API_KEY),
            ("gemini/gemini-2.0-flash", self.GEMINI_API_KEY),
            ("deepseek/deepseek-reasoner", self.DEEPSEEK_API_KEY),
            ("openai/gpt-4o", self.OPENAI_API_KEY),
        ])
        cloud_vision = self._build_cloud_chain([
            ("gemini/gemini-2.0-flash", self.GEMINI_API_KEY),
            ("openai/gpt-4o", self.OPENAI_API_KEY),
        ])
        return {
            "chat":      [self.LORA_CHAT_ADAPTER, "ollama/qwen2.5:7b", "ollama/llama3"] + cloud_chat,
            "reasoning": [self.LORA_REASONING_ADAPTER, "ollama/deepseek-r1:8b", "ollama/llama3"] + cloud_reasoning,
            "graph_rag": ["ollama/nomic-embed-text", "ollama/llama3"] + cloud_chat,
            "vision":    ["ollama/llama3.2-vision", "ollama/llava", "ollama/llama3"] + cloud_vision,
            "embedding": ["ollama/nomic-embed-text"],
        }

    def _build_cloud_chain(self, model_key_pairs: List[tuple]) -> List[str]:
        """Filters cloud models to only those with a valid non-empty API key."""
        return [model for model, key in model_key_pairs if key and key.strip()]

    @property
    def api_keys_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Parses OMNIBRAIN_API_KEYS into a clean dictionary.
        Returns:
            {
                "api_key": {
                    "app_id": "app_name",
                    "token_limit": 1000000,
                    "rate_limit_rpm": 60
                }
            }
        """
        keys_map = {}
        for item in self.API_KEYS_RAW.split(","):
            if not item.strip():
                continue
            parts = item.split(":")
            if len(parts) >= 2:
                key = parts[0].strip()
                app_id = parts[1].strip()
                token_limit = int(parts[2].strip()) if len(parts) > 2 else 1000000
                rpm = int(parts[3].strip()) if len(parts) > 3 else 60
                keys_map[key] = {
                    "app_id": app_id,
                    "token_limit": token_limit,
                    "rate_limit_rpm": rpm
                }
        return keys_map

settings = Settings()

# Inject cloud keys into os.environ so LiteLLM can pick them up
_key_env_map = {
    "GEMINI_API_KEY": settings.GEMINI_API_KEY,
    "OPENAI_API_KEY": settings.OPENAI_API_KEY,
    "OPENROUTER_API_KEY": settings.OPENROUTER_API_KEY,
    "DEEPSEEK_API_KEY": settings.DEEPSEEK_API_KEY,
    "ANTHROPIC_API_KEY": settings.ANTHROPIC_API_KEY,
}
for env_key, value in _key_env_map.items():
    if value and value.strip():
        os.environ[env_key] = value
