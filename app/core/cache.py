import json
import time
import sqlite3
from typing import Optional, Tuple, Dict, Any
from app.core.config import settings

class SemanticCacheEngine:
    def __init__(self):
        self._redis_client = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        
        # Try to initialize Redis if URL is provided
        if settings.REDIS_URL:
            try:
                import redis
                self._redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                # Test connection
                self._redis_client.ping()
            except Exception as e:
                print(f"Warning: Failed to connect to Redis at {settings.REDIS_URL}: {e}. Falling back to SQLite.")
                self._redis_client = None

        # Setup SQLite for Semantic Caching
        try:
            conn = sqlite3.connect(settings.SQLITE_CACHE_DB)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS semantic_cache ("
                "id INTEGER PRIMARY KEY AUTO_INCREMENT,"
                "query TEXT UNIQUE,"
                "response TEXT,"
                "embedding TEXT,"
                "created_at REAL"
                ")"
            )
            # SQLite doesn't support AUTO_INCREMENT with default setup; use AUTOINCREMENT
        except sqlite3.OperationalError:
            conn = sqlite3.connect(settings.SQLITE_CACHE_DB)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS semantic_cache ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "query TEXT UNIQUE,"
                "response TEXT,"
                "embedding TEXT,"
                "created_at REAL"
                ")"
            )
        finally:
            conn.commit()
            conn.close()

        self._initialized = True

    def _get_embedding(self, text: str) -> list:
        """
        Lazy loads litellm to generate embedding for the query.
        """
        import litellm
        
        # Use gemini-2.0 or OpenAI embedding model.
        # Fall back gracefully to standard embedder.
        try:
            response = litellm.embedding(
                model="text-embedding-3-small" if "openai" in settings.DEFAULT_CHAT_MODEL else "gemini/text-embedding-004",
                input=[text]
            )
            return response.data[0]["embedding"]
        except Exception:
            try:
                # Secondary fallback embedder
                response = litellm.embedding(
                    model="openai/text-embedding-ada-002",
                    input=[text]
                )
                return response.data[0]["embedding"]
            except Exception as e:
                print(f"Error generating embedding: {e}")
                return []

    def lookup(self, query: str) -> Optional[str]:
        """
        Looks up a response in the cache. First tries Redis exact match, then falls back to SQLite semantic match.
        """
        if not settings.SEMANTIC_CACHE_ENABLED:
            return None

        self._lazy_init()

        # 1. Exact Match via Redis (faster if available)
        if self._redis_client:
            try:
                cached_res = self._redis_client.get(f"exact_cache:{query}")
                if cached_res:
                    return cached_res
            except Exception as e:
                print(f"Redis cache lookup error: {e}")

        # 2. Semantic Match via SQLite
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return None

        import numpy as np

        conn = sqlite3.connect(settings.SQLITE_CACHE_DB)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT query, response, embedding FROM semantic_cache")
            rows = cursor.fetchall()
            
            best_score = -1.0
            best_response = None
            
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            
            if q_norm == 0:
                return None

            for cached_query, cached_response, cached_embedding_str in rows:
                if not cached_embedding_str:
                    continue
                try:
                    c_embedding = json.loads(cached_embedding_str)
                    c_vec = np.array(c_embedding, dtype=np.float32)
                    c_norm = np.linalg.norm(c_vec)
                    if c_norm == 0:
                        continue
                    
                    # Cosine Similarity
                    score = np.dot(q_vec, c_vec) / (q_norm * c_norm)
                    if score > best_score:
                        best_score = score
                        best_response = cached_response
                except Exception as e:
                    print(f"Error computing similarity for cached item: {e}")
                    continue

            if best_score >= settings.SEMANTIC_CACHE_THRESHOLD:
                print(f"Semantic cache hit (score: {best_score:.4f})")
                return best_response
                
        except Exception as e:
            print(f"SQLite semantic lookup failed: {e}")
        finally:
            conn.close()

        return None

    def store(self, query: str, response: str):
        """
        Stores response in both exact Redis cache and SQLite semantic cache database.
        """
        if not settings.SEMANTIC_CACHE_ENABLED:
            return

        self._lazy_init()

        # 1. Exact Match store in Redis
        if self._redis_client:
            try:
                self._redis_client.setex(f"exact_cache:{query}", 86400, response) # cache 24h
            except Exception as e:
                print(f"Redis cache store error: {e}")

        # 2. Semantic Match store in SQLite
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return

        conn = sqlite3.connect(settings.SQLITE_CACHE_DB)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO semantic_cache (query, response, embedding, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(query) DO UPDATE SET response=excluded.response, created_at=excluded.created_at",
                (query, response, json.dumps(query_embedding), time.time())
            )
            conn.commit()
        except Exception as e:
            print(f"SQLite cache store failed: {e}")
        finally:
            conn.close()

# Singleton Instance
cache_engine = SemanticCacheEngine()
