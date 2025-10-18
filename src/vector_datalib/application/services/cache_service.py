"""
Cache Service - Handles LRU caching for Vector Database lookups.
Follows DDD separation of concerns.
"""

from typing import Dict, Any, Optional

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

        self._cache: Dict[str, Any] = {}
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
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

        return None

    def put(self, key: str, value: Any):
        """
        Add item to LRU cache, evicting oldest if at capacity.

        Args:
            key: Cache key
            value: Value to cache
        """

        if len(self._cache) >= self._max_size:
            # Remove least recently used (first item)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = value

    def invalidate(self, key: str) -> bool:
        """
        Remove specific key from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed
        """

        return self._cache.pop(key, None) is not None

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
