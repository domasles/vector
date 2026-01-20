"""
Vector Database - Main application interface.
Delegates all operations to service layer following clean architecture.
Async-first design for non-blocking I/O operations.
"""

from typing import Dict, Any, Optional, List

import asyncio
import logging

from .services import CoordinateService, CacheService

from ..domain.coordinates import CentralAxis
from ..domain.mappings import CoordinateMapping
from ..domain.spaces import DimensionalSpace

from ..infrastructure.storage import VectorFileStorage

logger = logging.getLogger(__name__)


class VectorDB:
    """
    Vector Database - Coordinate-based database system with async-first API.

    Usage:
        async with VectorDB("data.db") as db:
            db.insert(101, {"name": "Alice", "age": 28})
            name = await db.lookup(101, "name")
    """

    def __init__(self, database_path: str = "vector.db", cache_size: int = 1000):
        """
        Initialize Vector Database facade.

        Args:
            database_path: Path to the .db file (created if doesn't exist)
            cache_size: Maximum number of items to cache (default: 1000)
        """

        self.database_path = database_path
        self.__initialized = False
        self.__closed = False
        self.__lock = None  # Lazy init in __aenter__

        # Initialize infrastructure and domain objects
        self.__storage = VectorFileStorage(database_path)
        self.__cache_service = CacheService(max_size=cache_size)

        self.__central_axis = CentralAxis()
        self.__dimensional_spaces: Dict[str, DimensionalSpace] = {}
        self.__coordinate_mappings: Dict[str, CoordinateMapping] = {}

        # Initialize service layer
        self.__coordinate_service = CoordinateService(
            self.__central_axis,
            self.__dimensional_spaces,
            self.__coordinate_mappings,
            self.__storage,
            self.__cache_service,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        self.__lock = asyncio.Lock()

        async with self.__lock:
            if not self.__initialized:
                await self.__coordinate_service.load_database_structure()
                self.__initialized = True

            logger.info(f"VectorDB initialized with {self.__central_axis.size()} vector points")
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - auto-save and cleanup."""
        async with self.__lock:
            self.__closed = True

            try:
                await self.__coordinate_service.save_database()

            except Exception as e:
                logger.error(f"Error saving database on exit: {e}")
                raise

            finally:
                self.__cache_service.clear()

        return False

    def _check_closed(self):
        """Raise RuntimeError if database is closed."""

        if self.__closed:
            raise RuntimeError("Cannot operate on closed database. Use 'async with VectorDB()' context manager.")

    async def upsert(self, vector_value: Any, attributes: Dict[str, Any], position: Optional[int] = None) -> int:
        """Smart upsert: inserts if new, updates all attributes if exists."""

        if not isinstance(attributes, dict):
            raise TypeError(f"Attributes must be dict, got {type(attributes).__name__}")

        if not attributes:
            raise ValueError("Attributes dictionary cannot be empty")

        if position is not None and not isinstance(position, int):
            raise TypeError(f"Position must be int or None, got {type(position).__name__}")

        self._check_closed()
        return await self.__coordinate_service.upsert_with_attributes(vector_value, attributes, position)

    async def lookup(self, vector_value: Any, dimension_name: str) -> Optional[Any]:
        """Look up a value for a vector point in a specific dimension."""

        if not isinstance(dimension_name, str):
            raise TypeError(f"Dimension name must be str, got {type(dimension_name).__name__}")

        self._check_closed()
        return await self.__coordinate_service.lookup_by_coordinate(vector_value, dimension_name)

    async def save(self) -> bool:
        """Save the database to file."""

        if self.__lock is None:
            raise RuntimeError("save() requires using 'async with' context manager")

        async with self.__lock:
            self._check_closed()
            return await self.__coordinate_service.save_database()

    async def batch_upsert(self, records: List[tuple]) -> List[int]:
        """Batch upsert vector points concurrently (insert or update)."""
        self._check_closed()

        async def _upsert_single(record):
            if len(record) == 2:
                vector_value, attributes = record
                position = None
            elif len(record) == 3:
                vector_value, attributes, position = record
            else:
                raise ValueError(
                    "Each record must be (vector_value, attributes) or (vector_value, attributes, position)"
                )

            return await self.__coordinate_service.upsert_with_attributes(vector_value, attributes, position)

        tasks = [_upsert_single(record) for record in records]
        coordinates = await asyncio.gather(*tasks)
        return list(coordinates)

    async def batch_lookup(self, queries: List[tuple]) -> List[Optional[Any]]:
        """Perform multiple lookups concurrently."""
        self._check_closed()

        tasks = [
            self.__coordinate_service.lookup_by_coordinate(vector_value, dimension_name)
            for vector_value, dimension_name in queries
        ]
        return await asyncio.gather(*tasks)

    async def delete(self, vector_value: Any) -> bool:
        """
        Delete a vector point and all its dimensional mappings.

        Args:
            vector_value: The vector point to delete

        Returns:
            bool: True if deleted, False if not found
        """
        self._check_closed()
        return await self.__coordinate_service.delete_coordinate(vector_value)

    async def batch_delete(self, vector_values: List[Any]) -> int:
        """
        Delete multiple vector points concurrently.

        Args:
            vector_values: List of vector point values to delete

        Returns:
            int: Number of successfully deleted vector points
        """
        self._check_closed()

        async def _delete_single(vector_value):
            try:
                return await self.__coordinate_service.delete_coordinate(vector_value)
            except Exception as e:
                logger.warning(f"Failed to delete {vector_value} - {e}")
                return False

        tasks = [_delete_single(vector_value) for vector_value in vector_values]
        results = await asyncio.gather(*tasks)

        return sum(results)

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        self._check_closed()
        return self.__coordinate_service.get_database_statistics()

    def get_vector_point(self, vector_value: Any):
        """Get complete vector point with all its dimensional attributes."""

        return self.__coordinate_service.get_vector_point_complete(vector_value)

    def get_all_vector_points(self) -> List:
        """Get all vector points with their complete attribute sets."""

        return self.__coordinate_service.get_all_vector_points_complete()

    def get_dimensions(self) -> List[str]:
        """Get all dimensional space names."""

        return self.__coordinate_service.get_dimensions_list()

    @property
    def vector_count(self) -> int:
        """Get the number of vector points in the database"""

        return self.__central_axis.size()

    @property
    def dimension_count(self) -> int:
        """Get the number of dimensions in the database"""

        return len(self.__dimensional_spaces)

    def __len__(self) -> int:
        """Return number of vectors in database."""
        return self.vector_count

    def __contains__(self, vector_value: Any) -> bool:
        """Check if vector_value exists in database."""

        return self.__central_axis.get_coordinate(vector_value) is not None

    def __repr__(self) -> str:
        return f"VectorDB(path='{self.database_path}', points={self.__central_axis.size()}, dimensions={len(self.__dimensional_spaces)})"
