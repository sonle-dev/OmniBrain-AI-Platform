import json
import time
import logging
from collections import OrderedDict
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class InMemoryLRUCache:
    """Thread-safe LRU in-memory cache — used when Redis is unavailable."""

    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            self._cache.move_to_end(key)  # Mark as recently used
            return self._cache[key]
        return None

    def set(self, key: str, value: str):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # Evict oldest


class SemanticCacheEngine:
    """
    Ultra-fast exact-match cache engine.
    Primary: Redis (if configured and reachable).
    Fallback: In-process LRU dict (128 entries by default).
    No embedding models required — zero external dependencies.
    """

    def __init__(self):
        self._redis_client = None
        self._memory_cache = InMemoryLRUCache(maxsize=settings.SEMANTIC_CACHE_MAX_MEMORY)
        self._redis_ok = False
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        if settings.REDIS_URL:
            try:
                import redis
                client = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
                client.ping()
                self._redis_client = client
                self._redis_ok = True
                logger.info("SemanticCache: Redis connected at %s", settings.REDIS_URL)
            except Exception as e:
                logger.warning("SemanticCache: Redis unavailable (%s). Using in-memory LRU cache.", e)
                self._redis_ok = False
        else:
            logger.info("SemanticCache: No REDIS_URL configured. Using in-memory LRU cache.")
        self._initialized = True

    def _make_key(self, raw: str) -> str:
        """Short hash key for storage efficiency."""
        import hashlib
        return "omnibrain:cache:" + hashlib.sha256(raw.encode()).hexdigest()[:32]

    def lookup(self, query: str) -> Optional[str]:
        """Returns cached response string if found, else None."""
        if not settings.SEMANTIC_CACHE_ENABLED:
            return None
        self._lazy_init()
        key = self._make_key(query)

        # 1. Try Redis first
        if self._redis_ok and self._redis_client:
            try:
                cached = self._redis_client.get(key)
                if cached:
                    logger.debug("Cache HIT (Redis): %s", key[:20])
                    return cached
            except Exception as e:
                logger.warning("Redis GET failed: %s. Falling back to memory.", e)
                self._redis_ok = False

        # 2. Try in-memory LRU
        cached = self._memory_cache.get(key)
        if cached:
            logger.debug("Cache HIT (Memory): %s", key[:20])
            return cached

        return None

    def store(self, query: str, response: str):
        """Stores response in cache (Redis + in-memory)."""
        if not settings.SEMANTIC_CACHE_ENABLED:
            return
        self._lazy_init()
        key = self._make_key(query)

        # 1. Store in Redis with 24h TTL
        if self._redis_ok and self._redis_client:
            try:
                self._redis_client.setex(key, 86400, response)
            except Exception as e:
                logger.warning("Redis SET failed: %s", e)
                self._redis_ok = False

        # 2. Always store in memory too
        self._memory_cache.set(key, response)


# Singleton
cache_engine = SemanticCacheEngine()
