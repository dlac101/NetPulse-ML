"""Redis prediction cache for fleet-level ML endpoints.

Caches expensive predictions (anomaly scores, churn, clusters) with
configurable TTLs to avoid re-scoring the entire fleet on every request.
"""

import hashlib

import orjson
import redis.asyncio as redis
import structlog

from netpulse_ml.config import settings

log = structlog.get_logger()

_client: redis.Redis | None = None


async def get_redis() -> redis.Redis | None:
    """Get or create the async Redis client. Returns None if Redis is unavailable."""
    global _client
    if _client is not None:
        return _client
    try:
        _client = redis.from_url(settings.redis_url, decode_responses=False)
        await _client.ping()
        log.info("Redis connected", url=settings.redis_url)
        return _client
    except Exception as e:
        log.warning("Redis unavailable (caching disabled)", error=str(e))
        _client = None
        return None


async def close_redis() -> None:
    """Close the Redis connection."""
    global _client
    if _client:
        await _client.aclose()
        _client = None


async def get_cached(key: str) -> dict | list | None:
    """Get a cached value by key. Returns None on miss or if Redis is down."""
    client = await get_redis()
    if client is None:
        return None
    try:
        data = await client.get(key)
        if data is None:
            return None
        return orjson.loads(data)
    except Exception as e:
        log.debug("Cache read failed", key=key, error=str(e))
        return None


async def set_cached(key: str, value: dict | list, ttl_seconds: int) -> None:
    """Set a cached value with TTL. Silently fails if Redis is down."""
    client = await get_redis()
    if client is None:
        return
    try:
        await client.setex(key, ttl_seconds, orjson.dumps(value))
    except Exception as e:
        log.debug("Cache write failed", key=key, error=str(e))


async def invalidate(pattern: str) -> int:
    """Delete all keys matching a pattern. Returns count deleted."""
    client = await get_redis()
    if client is None:
        return 0
    try:
        keys = []
        async for key in client.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            return await client.delete(*keys)
        return 0
    except Exception as e:
        log.debug("Cache invalidation failed", pattern=pattern, error=str(e))
        return 0


def make_cache_key(prefix: str, **params: str | int | float) -> str:
    """Build a deterministic cache key from prefix + params."""
    param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    h = hashlib.md5(param_str.encode()).hexdigest()[:12]
    return f"np:{prefix}:{h}"
