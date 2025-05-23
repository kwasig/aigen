import httpx
from typing import Optional

_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        _client = httpx.AsyncClient(http2=True, limits=limits)
    return _client

async def close_http_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
