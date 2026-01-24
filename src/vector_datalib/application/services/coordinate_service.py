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

    async def upsert_with_attributes(
        self, vector_value: Any, attributes: Dict[str, Any], position: Optional[int] = None
    ) -> int:
        """
        Upsert (insert or update) with attributes.

        If vector_value exists: Updates all provided attributes
        If vector_value is new: Inserts as new vector point

        Args:
            vector_value: The vector point value
            attributes: Dict of dimension -> value mappings
            position: Optional insertion position (only used for new inserts)

        Returns:
            int: The coordinate of the vector point
        """

        existing_coordinate = self.central_axis.get_coordinate(vector_value)

        if existing_coordinate is not None:
            logger.debug(f"Updating existing vector point '{vector_value}' at coordinate {existing_coordinate}")

            # Update all provided attributes
            for dimension_name, value in attributes.items():
                # Ensure dimension exists
                if dimension_name not in self.dimensional_spaces:
                    self._add_dimension(dimension_name)

                # Get old value_id for this coordinate
                old_value_id = self.coordinate_mappings[dimension_name].get_mapping(existing_coordinate)

                # Get or add new value
                new_value_id = self.dimensional_spaces[dimension_name].get_value_id(value)
                if new_value_id is None:
                    new_value_id = self.dimensional_spaces[dimension_name].add_value(value)

                # Update mapping
                self.coordinate_mappings[dimension_name].set_mapping(existing_coordinate, new_value_id)

                # Clean up old value if unreferenced
                if old_value_id is not None and old_value_id != new_value_id:
                    ref_count = self.coordinate_mappings[dimension_name].count_references_to_value(old_value_id)
                    if ref_count == 0:
                        self.dimensional_spaces[dimension_name].remove_value_if_unused(old_value_id)

                # Invalidate cache
                cache_key = f"{vector_value}:{dimension_name}"
                await self.cache_service.invalidate(cache_key)

            return existing_coordinate

        else:
            # New insert
            logger.debug(f"Inserting new vector point '{vector_value}'")

            # Add to central axis
            coordinate = self.central_axis.add_vector_point(vector_value, position)

            # Add dimensional attributes
            for dimension_name, value in attributes.items():
                if dimension_name not in self.dimensional_spaces:
                    self._add_dimension(dimension_name)

                value_id = self.dimensional_spaces[dimension_name].add_value(value)
                self.coordinate_mappings[dimension_name].set_mapping(coordinate, value_id)

            # Handle position-based shifting
            if position is not None and position < coordinate:
                self.central_axis.shift_coordinates_after_insertion(self.coordinate_mappings, position, 1)

            return coordinate

    async def lookup_by_coordinate(self, vector_value: Any, dimension_name: str) -> Optional[Any]:
        """
        Look up a value for a vector point in a specific dimension.
        O(1) lookup with async LRU caching coordination.
        """

        cache_key = f"{vector_value}:{dimension_name}"
        cached_result = await self.cache_service.get(cache_key)

        if cached_result is not None:
            return cached_result

        # Get coordinate from central axis
        coordinate = self.central_axis.get_coordinate(vector_value)
        if coordinate is None:
            return None

        # Get value_id from coordinate mapping
        if dimension_name not in self.coordinate_mappings:
            return None

        value_id = self.coordinate_mappings[dimension_name].get_mapping(coordinate)
        if value_id is None:
            return None

        # Get value from dimensional space
        if dimension_name not in self.dimensional_spaces:
            return None
        result = self.dimensional_spaces[dimension_name].get_value(value_id)

        # Cache the result
        if result is not None:
            await self.cache_service.put(cache_key, result)

        return result

    async def delete_coordinate(self, vector_value: Any) -> bool:
        """
        Delete a vector point and clean up all dimensional mappings.
        Uses reference counting to safely remove unused values.
        Uses tombstoning (O(1)) - no coordinate shifting needed.

        Args:
            vector_value: The vector point to delete

        Returns:
            bool: True if deleted, False if not found
        """
        # Get coordinate
        coordinate = self.central_axis.get_coordinate(vector_value)
        if coordinate is None:
            logger.warning(f"Vector point {vector_value} not found for deletion")
            return False

        # Invalidate all cache entries for this vector point
        for dimension_name in self.dimensional_spaces.keys():
            cache_key = f"{vector_value}:{dimension_name}"
            await self.cache_service.invalidate(cache_key)

        # Clean up dimensional mappings and values
        for dimension_name, mapping in self.coordinate_mappings.items():
            # Get the value_id that this coordinate references
            value_id = mapping.get_mapping(coordinate)

            if value_id is not None:
                # Remove the mapping
                mapping.remove_mapping(coordinate)

                # Check if any other coordinates reference this value
                ref_count = mapping.count_references_to_value(value_id)

                # If no more references, remove the value from dimensional space
                if ref_count == 0:
                    self.dimensional_spaces[dimension_name].remove_value_if_unused(value_id)
                    logger.debug(f"Cleaned up unused value (ID {value_id}) from dimension '{dimension_name}'")

        # Tombstone in central axis (O(1) - no shifting)
        self.central_axis.remove_vector_point(vector_value)

        logger.info(f"Deleted vector point '{vector_value}' at coordinate {coordinate}")
        return True

    async def save_database(self) -> bool:
        """
        Save database using async storage service.
        Coordinates between domain state and storage infrastructure.
        """

        database_data = self.storage.serialize_database_structure(
            self.central_axis, self.dimensional_spaces, self.coordinate_mappings
        )

        return await self.storage.save_with_auto_metadata(database_data, self.central_axis, self.dimensional_spaces)

    async def load_database_structure(self):
        """
        Load database structure using async storage service.
        Coordinates restoration of domain objects from storage.
        """

        database_data = await self.storage.load_database_structure()
        if database_data is None:
            return

        try:
            # Restore central axis
            axis_data = database_data.get("central_axis", {})

            self.central_axis.vector_points = axis_data.get("vector_points", [])
            self.central_axis.coordinate_map = axis_data.get("coordinate_map", {})
            self.central_axis.free_slots = axis_data.get("free_slots", [])

            # Restore dimensional spaces
            spaces_data = database_data.get("dimensional_spaces", {})

            for name, space_data in spaces_data.items():
                space = DimensionalSpace(name)

                # Convert string keys back to integers for value_domain
                space.value_domain = {int(k): v for k, v in space_data.get("value_domain", {}).items()}

                space.next_id = space_data.get("next_id", 1)
                self.dimensional_spaces[name] = space

            # Restore coordinate mappings
            mappings_data = database_data.get("coordinate_mappings", {})

            for name, mapping_data in mappings_data.items():
                mapping = CoordinateMapping(name)

                # Handle both old format (direct dict) and new format (nested with ref_counts)
                if isinstance(mapping_data, dict) and "coordinate_to_value_id" in mapping_data:
                    # New format with ref_counts
                    mapping.coordinate_to_value_id = {
                        int(k): v for k, v in mapping_data["coordinate_to_value_id"].items()
                    }
                    mapping.ref_counts = {int(k): v for k, v in mapping_data.get("ref_counts", {}).items()}

                else:
                    # Old format - rebuild ref_counts from mappings
                    mapping.coordinate_to_value_id = {int(k): v for k, v in mapping_data.items()}

                    # Rebuild ref_counts
                    for value_id in mapping.coordinate_to_value_id.values():
                        mapping.ref_counts[value_id] = mapping.ref_counts.get(value_id, 0) + 1

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

        return self.central_axis.get_vector_point_with_attributes(
            vector_value, self.dimensional_spaces, self.coordinate_mappings
        )

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
