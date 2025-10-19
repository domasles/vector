"""
Cache Service - Handles LRU caching for Vector Database lookups.
Follows DDD separation of concerns.
"""

from typing import Any, Optional, Dict
from collections import OrderedDict

import logging

logger = logging.getLogger(__name__)

class CacheService:
    """LRU cache service for database lookups."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache with configurable size.

        Args:
            max_size: Maximum number of items to cache
        """

        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache, moving it to end (most recently used).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        return None

    def put(self, key: str, value: Any):
        """
        Add item to LRU cache, evicting oldest if at capacity.

        Args:
            key: Cache key
            value: Value to cache
        """

        # If key exists, move to end; otherwise add new
        if key in self._cache:
            self._cache.move_to_end(key)
        
        self._cache[key] = value
        
        # Evict oldest if over capacity
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self):
        """Clear all cached items."""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def is_full(self) -> bool:
        """Check if cache is at maximum capacity."""
        return len(self._cache) >= self._max_size

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""

        return {
            "current_size": len(self._cache),
            "max_size": self._max_size,
            "utilization": len(self._cache) / self._max_size * 100
        }

    def __repr__(self) -> str:
        return f"CacheService(size={len(self._cache)}/{self._max_size}, {len(self._cache) / self._max_size * 100:.1f}% full)"
