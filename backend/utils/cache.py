from collections import OrderedDict
import asyncio
import time
from typing import Any

class TTLCache:
    """A simple async-safe TTL cache."""

    def __init__(self, maxsize: int = 100, ttl: int = 60):
        self.maxsize = maxsize
        self.ttl = ttl
        self._store = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any:
        async with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, ts = item
            if time.time() - ts > self.ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: Any):
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time())
            if len(self._store) > self.maxsize:
                self._store.popitem(last=False)
