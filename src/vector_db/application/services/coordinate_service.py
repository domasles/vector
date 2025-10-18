"""
Coordinate Service - Orchestrates operations between domain objects.
Follows DDD principles by coordinating domain logic without containing business rules.
"""

from typing import Dict, Any, Optional, List

import logging
import asyncio

from ...domain.spaces import DimensionalSpace
from ...domain.mappings import CoordinateMapping

logger = logging.getLogger(__name__)

class CoordinateService:
    """Service layer that orchestrates coordinate operations between domain objects."""

    def __init__(self, central_axis, dimensional_spaces, coordinate_mappings, storage, cache_service):
        """
        Initialize with domain objects and infrastructure services.

        Args:
            central_axis: CentralAxis domain object
            dimensional_spaces: Dict of DimensionalSpace objects
            coordinate_mappings: Dict of CoordinateMapping objects  
            storage: VectorFileStorage infrastructure
            cache_service: CacheService for LRU caching
        """

        self.central_axis = central_axis
        self.dimensional_spaces = dimensional_spaces
        self.coordinate_mappings = coordinate_mappings
        self.storage = storage
        self.cache_service = cache_service

    async def insert_with_attributes(self, vector_value: Any, attributes: Dict[str, Any], position: Optional[int] = None) -> int:
        """
        Smart insert with collision detection (insert or update if exists).

        If vector_value exists: Updates all provided attributes
        If vector_value is new: Inserts as new vector point
        """

        existing_coordinate = self.central_axis.get_coordinate(vector_value)

        if existing_coordinate is not None:
            logger.debug(f"Updating existing vector point '{vector_value}' at coordinate {existing_coordinate}")

            for dimension_name, value in attributes.items():
                await self.update_coordinate_attribute(vector_value, dimension_name, value)

            return existing_coordinate

        else:
            logger.debug(f"Inserting new vector point '{vector_value}'")

            # Add to central axis first
            coordinate = self.central_axis.add_vector_point(vector_value, position)

            # Process each dimensional attribute
            for dimension_name, value in attributes.items():
                if dimension_name not in self.dimensional_spaces:
                    self._add_dimension(dimension_name)

                # Add value to dimensional space (with deduplication)
                value_id = self.dimensional_spaces[dimension_name].add_value(value)
                self.coordinate_mappings[dimension_name].set_mapping(coordinate, value_id)

            # Handle coordinate shifting if inserted at specific position
            if position is not None and position < coordinate:
                self.central_axis.shift_coordinates_after_insertion(self.coordinate_mappings, position, 1)

            return coordinate

    async def lookup_by_coordinate(self, vector_value: Any, dimension_name: str) -> Optional[Any]:
        """
        Look up a value for a vector point in a specific dimension.
        O(1) lookup with LRU caching coordination.
        """

        cache_key = f"{vector_value}:{dimension_name}"
        cached_result = self.cache_service.get(cache_key)

        if cached_result is not None: return cached_result

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

        # Cache the result
        if result is not None:
            self.cache_service.put(cache_key, result)

        return result

    async def update_coordinate_attribute(self, vector_value: Any, dimension_name: str, new_value: Any) -> bool:
        """
        Update a specific value for a vector point in a dimension.
        Uses reference counting to safely manage dimensional values.
        """

        cache_key = f"{vector_value}:{dimension_name}"
        self.cache_service.invalidate(cache_key)
        
        # Get coordinate
        coordinate = self.central_axis.get_coordinate(vector_value)

        if coordinate is None:
            logger.warning(f"Vector point {vector_value} not found")
            return False

        # Ensure dimension exists
        if dimension_name not in self.dimensional_spaces:
            self._add_dimension(dimension_name)

        # Get current value_id for this coordinate (if any)
        old_value_id = self.coordinate_mappings[dimension_name].get_mapping(coordinate)
        new_value_id = self.dimensional_spaces[dimension_name].get_value_id(new_value)

        if new_value_id is not None:
            self.coordinate_mappings[dimension_name].set_mapping(coordinate, new_value_id)

        else:
            new_value_id = self.dimensional_spaces[dimension_name].add_value(new_value)
            self.coordinate_mappings[dimension_name].set_mapping(coordinate, new_value_id)

        # Clean up old value if no other coordinates reference it
        if old_value_id is not None and old_value_id != new_value_id:
            ref_count = self.coordinate_mappings[dimension_name].count_references_to_value(old_value_id)

            if ref_count == 0:
                self.dimensional_spaces[dimension_name].remove_value_if_unused(old_value_id)
                logger.debug(f"Cleaned up unused value (ID {old_value_id}) from dimension '{dimension_name}'")

        logger.debug(f"Updated {vector_value}:{dimension_name} = {new_value}")
        return True

    async def batch_insert_with_attributes(self, records: List[tuple]) -> List[int]:
        """
        Insert multiple records concurrently.
        Orchestrates concurrent operations with proper cache management.
        """

        self.cache_service.clear()
        insert_tasks = []

        for record in records:
            if len(record) == 2:
                vector_value, attributes = record
                position = None

            elif len(record) == 3:
                vector_value, attributes, position = record

            else:
                raise ValueError("Each record must be (vector_value, attributes) or (vector_value, attributes, position)")

            insert_tasks.append(self.insert_with_attributes(vector_value, attributes, position))

        coordinates = await asyncio.gather(*insert_tasks)
        logger.info(f"Batch inserted {len(records)} records concurrently")

        return coordinates

    async def batch_lookup_coordinates(self, queries: List[tuple]) -> List[Optional[Any]]:
        """
        Perform multiple lookups concurrently.
        Orchestrates concurrent cache and domain lookups.
        """

        lookup_tasks = [
            self.lookup_by_coordinate(vector_value, dimension_name)
            for vector_value, dimension_name in queries
        ]

        results = await asyncio.gather(*lookup_tasks)
        return results

    async def batch_update_coordinates(self, updates: List[tuple]) -> int:
        """
        Perform multiple updates efficiently.
        Coordinates batch updates with proper error handling.
        """

        self.cache_service.clear()

        update_tasks = [
            self.update_coordinate_attribute(vector_value, dimension_name, new_value)
            for vector_value, dimension_name, new_value in updates
        ]

        try:
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            successful_updates = sum(1 for result in results if result is True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    vector_value, dimension_name, new_value = updates[i]
                    logger.warning(f"Failed to update {vector_value}:{dimension_name} - {result}")

        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            return 0

        logger.info(f"Batch updated {successful_updates}/{len(updates)} records concurrently")
        return successful_updates

    async def save_database(self) -> bool:
        """
        Save database using enhanced storage service.
        Coordinates between domain state and storage infrastructure.
        """

        database_data = self.storage.serialize_database_structure(
            self.central_axis, self.dimensional_spaces, self.coordinate_mappings
        )

        return self.storage.save_with_auto_metadata(
            database_data, self.central_axis, self.dimensional_spaces
        )

    async def load_database_structure(self):
        """
        Load database structure using enhanced storage service.
        Coordinates restoration of domain objects from storage.
        """

        database_data = self.storage.load_database_structure()
        if database_data is None: return

        try:
            # Restore central axis
            axis_data = database_data.get("central_axis", {})
            self.central_axis.vector_points = axis_data.get("vector_points", [])

            coordinate_map_data = axis_data.get("coordinate_map", {})
            self.central_axis.coordinate_map = {}

            for key, value in coordinate_map_data.items():
                try:
                    int_key = int(key)
                    self.central_axis.coordinate_map[int_key] = value

                except (ValueError, TypeError):
                    try:
                        float_key = float(key)
                        self.central_axis.coordinate_map[float_key] = value

                    except (ValueError, TypeError):
                        self.central_axis.coordinate_map[key] = value

            # Restore dimensional spaces
            spaces_data = database_data.get("dimensional_spaces", {})
            
            for name, space_data in spaces_data.items():
                space = DimensionalSpace(name)

                # Convert string keys back to integers for value_domain
                space.value_domain = {int(k): v for k, v in space_data.get("value_domain", {}).items()}

                # Rebuild value_to_id from value_domain (ensures consistency)
                space.value_to_id = {v: int(k) for k, v in space_data.get("value_domain", {}).items()}

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

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        Delegates to enhanced storage service.
        """

        return self.storage.get_database_stats(self.central_axis, self.dimensional_spaces)

    def get_vector_point_complete(self, vector_value: Any):
        """
        Get complete vector point with all attributes.
        Delegates to enhanced CentralAxis.
        """

        return self.central_axis.get_vector_point_with_attributes(vector_value, self.dimensional_spaces, self.coordinate_mappings)

    def get_all_vector_points_complete(self) -> List:
        """Get all vector points with their complete attribute sets."""

        points = []

        for vector_value in self.central_axis.get_all_points():
            point = self.get_vector_point_complete(vector_value)

            if point:
                points.append(point)

        return points

    def get_dimensions_list(self) -> List[str]:
        """Get all dimensional space names."""
        return list(self.dimensional_spaces.keys())

    def _add_dimension(self, dimension_name: str):
        """
        Add a new dimensional space to the database.
        Internal method for dimension management.
        """

        if dimension_name not in self.dimensional_spaces:
            self.dimensional_spaces[dimension_name] = DimensionalSpace(dimension_name)
            self.coordinate_mappings[dimension_name] = CoordinateMapping(dimension_name)

            logger.info(f"Added new dimension: '{dimension_name}'")
