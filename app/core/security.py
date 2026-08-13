import time
import sqlite3
from typing import Optional, Dict, Tuple
from fastapi import Header, HTTPException, status
from app.core.config import settings

# Global dictionary for sliding-window rate limiting
# Key: (app_id, ip_address), Value: List[timestamp_float]
_rate_limit_records: Dict[Tuple[str, str], list] = {}

def get_db_connection():
    """Establishes connection to the SQLite metrics database for token usage persistence."""
    conn = sqlite3.connect(settings.SQLITE_METRICS_DB)
    conn.row_factory = sqlite3.Row
    # Initialize schema
    conn.execute(
        "CREATE TABLE IF NOT EXISTS token_usage ("
        "app_id TEXT PRIMARY KEY,"
        "tokens_consumed INTEGER DEFAULT 0"
        ")"
    )
    conn.commit()
    return conn

def track_token_usage(app_id: str, tokens: int) -> int:
    """Updates the token count for a specific app_id and returns the total used tokens."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO token_usage (app_id, tokens_consumed) VALUES (?, ?) "
            "ON CONFLICT(app_id) DO UPDATE SET tokens_consumed = tokens_consumed + excluded.tokens_consumed",
            (app_id, tokens)
        )
        conn.commit()
        cursor.execute("SELECT tokens_consumed FROM token_usage WHERE app_id = ?", (app_id,))
        row = cursor.fetchone()
        return row[0] if row else 0
    finally:
        conn.close()

def get_current_token_usage(app_id: str) -> int:
    """Retrieves the current token count for a specific app_id."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT tokens_consumed FROM token_usage WHERE app_id = ?", (app_id,))
        row = cursor.fetchone()
        return row[0] if row else 0
    finally:
        conn.close()

def reset_token_usage(app_id: str):
    """Resets the token count for a specific app_id."""
    conn = get_db_connection()
    try:
        conn.execute("UPDATE token_usage SET tokens_consumed = 0 WHERE app_id = ?", (app_id,))
        conn.commit()
    finally:
        conn.close()

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """
    Dependency to validate the X-API-Key header.
    Returns:
        The validated app_id.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header"
        )
    
    keys_map = settings.api_keys_map
    if x_api_key not in keys_map:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    
    app_info = keys_map[x_api_key]
    app_id = app_info["app_id"]
    token_limit = app_info["token_limit"]
    
    # Enforce global token usage limit
    current_tokens = get_current_token_usage(app_id)
    if current_tokens >= token_limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Token limit exceeded for App ID '{app_id}'. (Limit: {token_limit}, Used: {current_tokens})"
        )
    
    return app_id

def enforce_rate_limit(app_id: str, client_ip: str, rpm: int):
    """
    Enforces a request rate limit using an in-memory sliding window.
    """
    now = time.time()
    key = (app_id, client_ip)
    
    if key not in _rate_limit_records:
        _rate_limit_records[key] = []
        
    # Filter out timestamps older than 60 seconds
    timestamps = [t for t in _rate_limit_records[key] if now - t < 60]
    
    if len(timestamps) >= rpm:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {rpm} requests per minute are allowed."
        )
        
    timestamps.append(now)
    _rate_limit_records[key] = timestamps
