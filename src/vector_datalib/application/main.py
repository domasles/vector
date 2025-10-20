"""
Vector Database - Main application interface.
Delegates all operations to service layer following clean architecture.
"""

from typing import Dict, Any, Optional, List

import threading
import logging

from .services import CoordinateService, CacheService

from ..domain.coordinates import CentralAxis
from ..domain.mappings import CoordinateMapping
from ..domain.spaces import DimensionalSpace

from ..infrastructure.storage import VectorFileStorage

logger = logging.getLogger(__name__)

class VectorDB:
    """
    Vector Database - Coordinate-based database system.

    Usage:
        with VectorDB("data.db") as db:
            db.insert(101, {"name": "Alice", "age": 28})
            name = db.lookup(101, "name")
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
        self.__lock = threading.RLock()

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
            self.__cache_service
        )

    def __enter__(self):
        """Context manager entry."""

        with self.__lock:
            if not self.__initialized:
                self.__coordinate_service.load_database_structure()
                self.__initialized = True

            logger.info(f"VectorDB initialized with {self.__central_axis.size()} vector points")
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-save and cleanup."""

        with self.__lock:
            self.__closed = True

            try:
                self.__coordinate_service.save_database()

            except Exception as e:
                logger.error(f"Error saving database on exit: {e}")
                raise

            finally:
                self.__cache_service.clear()

        return False

    def _check_closed(self):
        """Raise RuntimeError if database is closed."""

        if self.__closed:
            raise RuntimeError("Cannot operate on closed database. Use 'with VectorDB()' context manager.")

    def insert(self, vector_value: Any, attributes: Dict[str, Any], position: Optional[int] = None) -> int:
        """Smart insert with collision detection (insert or update if exists)."""

        if not isinstance(attributes, dict):
            raise TypeError(f"Attributes must be dict, got {type(attributes).__name__}")

        if not attributes:
            raise ValueError("Attributes dictionary cannot be empty")

        if position is not None and not isinstance(position, int):
            raise TypeError(f"Position must be int or None, got {type(position).__name__}")

        with self.__lock:
            self._check_closed()
            return self.__coordinate_service.insert_with_attributes(vector_value, attributes, position)

    def lookup(self, vector_value: Any, dimension_name: str) -> Optional[Any]:
        """Look up a value for a vector point in a specific dimension."""

        if not isinstance(dimension_name, str):
            raise TypeError(f"Dimension name must be str, got {type(dimension_name).__name__}")

        with self.__lock:
            self._check_closed()
            return self.__coordinate_service.lookup_by_coordinate(vector_value, dimension_name)

    def update(self, vector_value: Any, dimension_name: str, new_value: Any) -> bool:
        """Update a specific value for a vector point in a dimension."""

        if not isinstance(dimension_name, str):
            raise TypeError(f"Dimension name must be str, got {type(dimension_name).__name__}")

        with self.__lock:
            self._check_closed()
            return self.__coordinate_service.update_coordinate_attribute(vector_value, dimension_name, new_value)

    def save(self) -> bool:
        """Save the database to file."""

        with self.__lock:
            self._check_closed()
            return self.__coordinate_service.save_database()

    def batch_insert(self, records: List[tuple]) -> List[int]:
        """Smart batch insert with collision detection (insert or update if exists)."""

        with self.__lock:
            self._check_closed()
            return self.__coordinate_service.batch_insert_with_attributes(records)

    def batch_lookup(self, queries: List[tuple]) -> List[Optional[Any]]:
        """Perform multiple lookups efficiently."""

        with self.__lock:
            self._check_closed()
            return self.__coordinate_service.batch_lookup_coordinates(queries)

    def batch_update(self, updates: List[tuple]) -> int:
        """Perform multiple updates efficiently."""

        with self.__lock:
            self._check_closed()
            return self.__coordinate_service.batch_update_coordinates(updates)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""

        with self.__lock:
            return self.__coordinate_service.get_database_statistics()

    def get_vector_point(self, vector_value: Any):
        """Get complete vector point with all its dimensional attributes."""

        with self.__lock:
            return self.__coordinate_service.get_vector_point_complete(vector_value)

    def get_all_vector_points(self) -> List:
        """Get all vector points with their complete attribute sets."""

        with self.__lock:
            return self.__coordinate_service.get_all_vector_points_complete()

    def get_dimensions(self) -> List[str]:
        """Get all dimensional space names."""

        with self.__lock:
            return self.__coordinate_service.get_dimensions_list()

    def verify(self) -> Dict[str, Any]:
        """Verify data integrity and return statistics."""

        with self.__lock:
            return self.__coordinate_service.verify_integrity()

    @property
    def vector_count(self) -> int:
        """Get the number of vector points in the database"""

        with self.__lock:
            return self.__central_axis.size()

    @property
    def dimension_count(self) -> int:
        """Get the number of dimensions in the database"""

        with self.__lock:
            return len(self.__dimensional_spaces)

    def __len__(self) -> int:
        """Return number of vectors in database."""
        return self.vector_count

    def __contains__(self, vector_value: Any) -> bool:
        """Check if vector_value exists in database."""

        with self.__lock:
            return self.__central_axis.get_coordinate(vector_value) is not None

    def __repr__(self) -> str:
        with self.__lock:
            return f"VectorDB(path='{self.database_path}', points={self.__central_axis.size()}, dimensions={len(self.__dimensional_spaces)})"
