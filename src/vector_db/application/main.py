"""
Vector Database - Main application interface.
This is the primary API that developers will use.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List

import logging
import asyncio

from ..domain.coordinates import CentralAxis, VectorPoint
from ..domain.mappings import CoordinateMapping
from ..domain.spaces import DimensionalSpace

from ..infrastructure.storage import VectorFileStorage

logger = logging.getLogger(__name__)

class VectorDB:
    """
    Vector Database - N-dimensional coordinate-based database system.

    Usage:
        db = VectorDB("mydata.db")
        db.insert("Name", {"surname": "Surname", "age": 28})
        age = db.lookup("Name", "age")  # Returns 28

    "Name" is stored on the central axis (X-axis). It acts as a unique identifier, the primary key.
    All dimensional attributes (surname, age, etc.) are stored in separate dimensional spaces.

    Context Manager Usage:
        with VectorDB("mydata.db") as db:
            db.insert(101, "name", "Alice")
            name = db.lookup(101, "name")

    Database automatically saved and resources cleaned up
    """

    def __init__(self, database_path: str = "vector.db"):
        """
        Initialize Vector Database.

        Args:
            database_path: Path to the .db file (created if doesn't exist)
        """

        self.database_path = database_path
        self.storage = VectorFileStorage(database_path)

        self.central_axis = CentralAxis()
        self.dimensional_spaces: Dict[str, DimensionalSpace] = {}
        self.coordinate_mappings: Dict[str, CoordinateMapping] = {}

        self._cache: Dict[str, Any] = {}
        self._cache_max_size = 1000

        self._load_database()

        logger.info(f"VectorDB initialized with {self.central_axis.size()} vector points")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-save and cleanup."""

        try:
            self.save()

        except Exception as e:
            logger.error(f"Error saving database on exit: {e}")

        finally:
            self._cache.clear()

        return False

    def insert(self, vector_value: Any, attributes: Dict[str, Any], position: Optional[int] = None) -> int:
        """
        Insert a new vector point with its dimensional attributes.

        Args:
            vector_value: The primary identifier for the vector point
            attributes: Dictionary of dimension_name -> value mappings
            position: Optional position to insert at (None = append)

        Returns:
            int: The coordinate of the inserted vector point
        """

        coordinate = self.central_axis.add_vector_point(vector_value, position)

        # Process each dimensional attribute
        for dimension_name, value in attributes.items():
            if dimension_name not in self.dimensional_spaces:
                self.add_dimension(dimension_name)

            # Add value to dimensional space (with deduplication)
            value_id = self.dimensional_spaces[dimension_name].add_value(value)

            # Create mapping f_dimension(coordinate) = value_id
            self.coordinate_mappings[dimension_name].set_mapping(coordinate, value_id)

        # Handle coordinate shifting if inserted at specific position
        if position is not None and position < coordinate:
            self._shift_mappings_after_insertion(position, 1)

        logger.debug(f"Inserted vector point '{vector_value}' at coordinate {coordinate}")
        return coordinate

    def lookup(self, vector_value: Any, dimension_name: str) -> Optional[Any]:
        """
        Look up a value for a vector point in a specific dimension.
        O(1) lookup with LRU caching: vector_value -> coordinate -> value_id -> value

        Args:
            vector_value: The vector point identifier
            dimension_name: The dimension to look up

        Returns:
            Optional[Any]: The value, or None if not found
        """

        # Check cache first
        cache_key = f"{vector_value}:{dimension_name}"

        if cache_key in self._cache:
            # Move to end (most recently used)
            value = self._cache.pop(cache_key)
            self._cache[cache_key] = value

            return value

        # Get coordinate from central axis
        coordinate = self.central_axis.get_coordinate(vector_value)
        if coordinate is None: return None

        # Get value_id from coordinate mapping
        if dimension_name not in self.coordinate_mappings: return None
        
        value_id = self.coordinate_mappings[dimension_name].get_mapping(coordinate)
        if value_id is None: return None

        # Get value from dimensional space
        if dimension_name not in self.dimensional_spaces: return None
        result = self.dimensional_spaces[dimension_name].get_value(value_id)

        if result is not None:
            self._add_to_cache(cache_key, result)

        return result

    def _add_to_cache(self, key: str, value: Any):
        """Add item to LRU cache, evicting oldest if at capacity."""

        if len(self._cache) >= self._cache_max_size:
            # Remove least recently used (first item)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = value

    def update_dimension_value(self, dimension_name: str, old_value: Any, new_value: Any) -> bool:
        """
        Update a value in a dimensional space. All vector points referencing 
        this value will automatically see the change.

        Args:
            dimension_name: The dimension to update
            old_value: The current value to replace
            new_value: The new value

        Returns:
            bool: True if update succeeded
        """

        if dimension_name not in self.dimensional_spaces: return False
        return self.dimensional_spaces[dimension_name].update_value(old_value, new_value)
    
    def update(self, vector_value: Any, dimension_name: str, new_value: Any) -> bool:
        """
        Update a specific value for a vector point in a dimension.

        Args:
            vector_value: The vector point identifier
            dimension_name: The dimension to update
            new_value: The new value to set

        Returns:
            bool: True if update succeeded
        """

        # Validate inputs
        if not self._validate_input(vector_value, "vector_value"): return False
        if not self._validate_input(dimension_name, "dimension_name"): return False
        if not self._validate_input(new_value, "new_value"): return False

        cache_key = f"{vector_value}:{dimension_name}"
        self._cache.pop(cache_key, None)
        
        # Get coordinate
        coordinate = self.central_axis.get_coordinate(vector_value)

        if coordinate is None:
            logger.warning(f"Vector point {vector_value} not found")
            return False
            
        # Ensure dimension exists
        if dimension_name not in self.dimensional_spaces:
            self.add_dimension(dimension_name)
            
        # Add new value to dimensional space
        value_id = self.dimensional_spaces[dimension_name].add_value(new_value)
        self.coordinate_mappings[dimension_name].set_mapping(coordinate, value_id)

        logger.debug(f"Updated {vector_value}:{dimension_name} = {new_value}")
        return True

    def _validate_input(self, value: Any, param_name: str) -> bool:
        """Enhanced input validation."""

        if value is None:
            logger.error(f"Parameter {param_name} cannot be None")
            return False

        if isinstance(value, str) and not value.strip():
            logger.error(f"Parameter {param_name} cannot be empty string")
            return False

        if isinstance(value, (str, bytes)) and len(value) > 10000:
            logger.error(f"Parameter {param_name} exceeds maximum size limit")
            return False

        return True
    
    def add_dimension(self, dimension_name: str):
        """
        Add a new dimensional space to the database.
        This is how the database scales to n-dimensions.

        Args:
            dimension_name: Name of the new dimension (e.g., "email", "department")
        """

        if dimension_name not in self.dimensional_spaces:
            self.dimensional_spaces[dimension_name] = DimensionalSpace(dimension_name)
            self.coordinate_mappings[dimension_name] = CoordinateMapping(dimension_name)

            logger.info(f"Added new dimension: '{dimension_name}'")

    def get_vector_point(self, vector_value: Any) -> Optional[VectorPoint]:
        """
        Get complete vector point with all its dimensional attributes.

        Args:
            vector_value: The vector point identifier

        Returns:
            Optional[VectorPoint]: Complete vector point or None if not found
        """

        coordinate = self.central_axis.get_coordinate(vector_value)
        if coordinate is None: return None

        attributes = {}

        for dimension_name in self.dimensional_spaces:
            value = self.lookup(vector_value, dimension_name)

            if value is not None:
                attributes[dimension_name] = value

        return VectorPoint(coordinate, vector_value, attributes)

    def get_all_vector_points(self) -> List[VectorPoint]:
        """Get all vector points with their complete attribute sets."""

        points = []

        for vector_value in self.central_axis.get_all_points():
            point = self.get_vector_point(vector_value)

            if point:
                points.append(point)

        return points

    def get_dimensions(self) -> List[str]:
        """Get all dimensional space names."""

        return list(self.dimensional_spaces.keys())

    def save(self) -> bool:
        """Save the database to file."""

        database_data = self._serialize_database()

        self.storage.update_metadata({
            "total_vector_points": self.central_axis.size(),
            "total_dimensions": len(self.dimensional_spaces)
        })

        return self.storage.save_database(database_data)

    def _serialize_database(self) -> Dict[str, Any]:
        """Serialize the complete database structure."""

        return {
            "central_axis": {
                "vector_points": self.central_axis.vector_points,
                "coordinate_map": self.central_axis.coordinate_map
            },
            "dimensional_spaces": {
                name: {
                    "value_domain": space.value_domain,
                    "value_to_id": space.value_to_id,
                    "next_id": space.next_id
                }
                for name, space in self.dimensional_spaces.items()
            },
            "coordinate_mappings": {
                name: mapping.coordinate_to_value_id
                for name, mapping in self.coordinate_mappings.items()
            }
        }

    def _load_database(self):
        """Load database from file if it exists."""

        database_data = self.storage.load_database()
        if database_data is None: return

        try:
            # Restore central axis
            axis_data = database_data.get("central_axis", {})

            self.central_axis.vector_points = axis_data.get("vector_points", [])
            self.central_axis.coordinate_map = axis_data.get("coordinate_map", {})

            # Restore dimensional spaces
            spaces_data = database_data.get("dimensional_spaces", {})

            for name, space_data in spaces_data.items():
                space = DimensionalSpace(name)

                # Convert string keys back to integers for value_domain
                space.value_domain = {int(k): v for k, v in space_data.get("value_domain", {}).items()}
                space.value_to_id = space_data.get("value_to_id", {})
                space.next_id = space_data.get("next_id", 1)

                self.dimensional_spaces[name] = space

            # Restore coordinate mappings
            mappings_data = database_data.get("coordinate_mappings", {})

            for name, mapping_data in mappings_data.items():
                mapping = CoordinateMapping(name)

                # Convert string keys back to integers (JSON serialization converts int keys to strings)
                mapping.coordinate_to_value_id = {int(k): v for k, v in mapping_data.items()}
                self.coordinate_mappings[name] = mapping

            logger.info("Database loaded successfully from file")

        except Exception as e:
            logger.error(f"Failed to load database: {e}")

    def _shift_mappings_after_insertion(self, from_position: int, shift_amount: int):
        """Shift all coordinate mappings after insertion."""

        for mapping in self.coordinate_mappings.values():
            mapping.shift_coordinates(from_position, shift_amount)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""

        return {
            "vector_points": self.central_axis.size(),
            "dimensions": len(self.dimensional_spaces),
            "dimension_details": {
                name: space.get_value_count()
                for name, space in self.dimensional_spaces.items()
            },
            "file_size_bytes": self.storage.get_file_size(),
            "metadata": self.storage.get_metadata()
        }

    def batch_insert(self, records: List[tuple]) -> List[int]:
        """
        Insert multiple records efficiently in a single operation.

        Args:
            records: List of tuples (vector_value, attributes_dict, optional_position)
                   
        Returns:
            List[int]: Coordinates of inserted vector points
        """

        coordinates = []
        self._cache.clear()

        for record in records:
            if len(record) == 2:
                vector_value, attributes = record
                position = None

            elif len(record) == 3:
                vector_value, attributes, position = record

            else:
                raise ValueError("Each record must be (vector_value, attributes) or (vector_value, attributes, position)")

            coordinate = self.insert(vector_value, attributes, position)
            coordinates.append(coordinate)

        logger.info(f"Batch inserted {len(records)} records")
        return coordinates

    def batch_lookup(self, queries: List[tuple]) -> List[Optional[Any]]:
        """
        Perform multiple lookups efficiently.

        Args:
            queries: List of tuples (vector_value, dimension_name)

        Returns:
            List of lookup results (maintains order)
        """

        results = []

        for vector_value, dimension_name in queries:
            result = self.lookup(vector_value, dimension_name)
            results.append(result)

        return results

    def batch_update(self, updates: List[tuple]) -> int:
        """
        Perform multiple updates efficiently.

        Args:
            updates: List of tuples (vector_value, dimension_name, new_value)

        Returns:
            int: Number of successful updates
        """

        successful_updates = 0
        self._cache.clear()

        for vector_value, dimension_name, new_value in updates:
            try:
                if self.update(vector_value, dimension_name, new_value):
                    successful_updates += 1

            except Exception as e:
                logger.warning(f"Failed to update {vector_value}:{dimension_name} - {e}")

        logger.info(f"Batch updated {successful_updates}/{len(updates)} records")
        return successful_updates

    async def async_insert(self, vector_value: Any, attributes: Dict[str, Any], position: Optional[int] = None) -> int:
        """Async version of insert."""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.insert, vector_value, attributes, position)
    
    async def async_lookup(self, vector_value: Any, dimension_name: str) -> Optional[Any]:
        """Async version of lookup."""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.lookup, vector_value, dimension_name)
    
    async def async_update(self, vector_value: Any, dimension_name: str, new_value: Any) -> bool:
        """Async version of update."""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.update, vector_value, dimension_name, new_value)
    
    async def async_save(self) -> bool:
        """Async version of save."""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.save)
    
    async def async_batch_insert(self, records: List[tuple]) -> List[int]:
        """Async version of batch_insert."""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.batch_insert, records)

    def __repr__(self) -> str:
        return f"VectorDB(path='{self.database_path}', points={self.central_axis.size()}, dimensions={len(self.dimensional_spaces)})"
