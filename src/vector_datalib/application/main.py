"""
Vector Database - Main application interface.
Delegates all operations to service layer following clean architecture.
"""

from typing import Dict, Any, Optional, List
import logging

from .services import CoordinateService, CacheService

from ..domain.coordinates import CentralAxis
from ..domain.mappings import CoordinateMapping
from ..domain.spaces import DimensionalSpace
from ..infrastructure.storage import VectorFileStorage

logger = logging.getLogger(__name__)

class VectorDB:
    """
    Vector Database - Pure async coordinate-based database system.

    Usage:
        async with VectorDB("data.db") as db:
            await db.insert(101, {"name": "Alice", "age": 28})
            name = await db.lookup(101, "name")
    """

    def __init__(self, database_path: str = "vector.db"):
        """
        Initialize Vector Database facade.

        Args:
            database_path: Path to the .db file (created if doesn't exist)
        """

        self.database_path = database_path
        self._initialized = False

        # Initialize infrastructure and domain objects
        self._storage = VectorFileStorage(database_path)
        self._cache_service = CacheService(max_size=1000)

        self._central_axis = CentralAxis()
        self._dimensional_spaces: Dict[str, DimensionalSpace] = {}
        self._coordinate_mappings: Dict[str, CoordinateMapping] = {}

        # Initialize service layer
        self._coordinate_service = CoordinateService(
            self._central_axis,
            self._dimensional_spaces, 
            self._coordinate_mappings,
            self._storage,
            self._cache_service
        )

    async def __aenter__(self):
        """Async context manager entry."""

        if not self._initialized:
            await self._coordinate_service.load_database_structure()
            self._initialized = True

        logger.info(f"VectorDB initialized with {self._central_axis.size()} vector points")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - auto-save and cleanup."""

        try:
            await self._coordinate_service.save_database()

        except Exception as e:
            logger.error(f"Error saving database on exit: {e}")

        finally:
            self._cache_service.clear()

        return False

    async def insert(self, vector_value: Any, attributes: Dict[str, Any], position: Optional[int] = None) -> int:
        """Smart insert with collision detection (insert or update if exists)."""
        return await self._coordinate_service.insert_with_attributes(vector_value, attributes, position)

    async def lookup(self, vector_value: Any, dimension_name: str) -> Optional[Any]:
        """Look up a value for a vector point in a specific dimension."""
        return await self._coordinate_service.lookup_by_coordinate(vector_value, dimension_name)

    async def update(self, vector_value: Any, dimension_name: str, new_value: Any) -> bool:
        """Update a specific value for a vector point in a dimension."""
        return await self._coordinate_service.update_coordinate_attribute(vector_value, dimension_name, new_value)

    async def save(self) -> bool:
        """Save the database to file."""
        return await self._coordinate_service.save_database()

    async def batch_insert(self, records: List[tuple]) -> List[int]:
        """Smart batch insert with collision detection (insert or update if exists)."""
        return await self._coordinate_service.batch_insert_with_attributes(records)

    async def batch_lookup(self, queries: List[tuple]) -> List[Optional[Any]]:
        """Perform multiple lookups concurrently."""
        return await self._coordinate_service.batch_lookup_coordinates(queries)

    async def batch_update(self, updates: List[tuple]) -> int:
        """Perform multiple updates efficiently."""
        return await self._coordinate_service.batch_update_coordinates(updates)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self._coordinate_service.get_database_statistics()

    def get_vector_point(self, vector_value: Any):
        """Get complete vector point with all its dimensional attributes."""
        return self._coordinate_service.get_vector_point_complete(vector_value)

    def get_all_vector_points(self) -> List:
        """Get all vector points with their complete attribute sets."""
        return self._coordinate_service.get_all_vector_points_complete()

    def get_dimensions(self) -> List[str]:
        """Get all dimensional space names."""
        return self._coordinate_service.get_dimensions_list()

    def __repr__(self) -> str:
        return f"VectorDB(path='{self.database_path}', points={self._central_axis.size()}, dimensions={len(self._dimensional_spaces)})"
