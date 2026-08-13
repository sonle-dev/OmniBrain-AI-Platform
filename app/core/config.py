import os
from typing import Dict, List, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General API keys for OmniBrain Core client authentication
    # Configured as a comma-separated key:app_id:token_limit:rpm string
    # e.g., "key1:lms_app:1000000:60,key2:ecommerce_app:5000000:120"
    API_KEYS_RAW: str = Field(
        default="demo_key:demo_app:1000000:60",
        alias="OMNIBRAIN_API_KEYS"
    )

    # Redis Config
    REDIS_URL: str = Field(default="", alias="OMNIBRAIN_REDIS_URL")

    # SQLite backup file paths
    SQLITE_CACHE_DB: str = "semantic_cache.db"
    SQLITE_METRICS_DB: str = "metrics.db"

    # External APIs
    TAVILY_API_KEY: str = Field(default="", alias="TAVILY_API_KEY")
    NARA_API_KEY: str = Field(default="", alias="NARA_API_KEY")

    # LLM Settings (LiteLLM configuration)
    DEFAULT_CHAT_MODEL: str = Field(default="gemini/gemini-2.0-flash", alias="OMNIBRAIN_DEFAULT_MODEL")
    FALLBACK_CHAT_MODELS: List[str] = Field(
        default=[
            "gemini/gemini-1.5-flash",
            "openai/gpt-4o-mini",
            "openrouter/google/gemini-2.0-flash",
            "ollama/llama3"
        ]
    )

    # Semantic Cache Settings
    SEMANTIC_CACHE_THRESHOLD: float = 0.90
    SEMANTIC_CACHE_ENABLED: bool = True

    # GraphRAG Settings
    GRAPHRAG_WORKING_DIR: str = "graphrag_data"

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
