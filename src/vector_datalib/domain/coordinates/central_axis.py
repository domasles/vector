"""
Central Axis - The X coordinate system that serves as the primary reference point.
All other dimensional spaces radiate from this central axis like propeller blades.
"""

from typing import Dict, Any, Optional, List

import logging

from .vector_point import VectorPoint

logger = logging.getLogger(__name__)


class CentralAxis:
    """
    Central coordinate axis representing the primary objects (X-axis).
    Acts as the hub from which all dimensional spaces extend.
    """

    def __init__(self):
        self.vector_points: List[Any] = []
        self.coordinate_map: Dict[Any, int] = {}  # value -> coordinate lookup

    def add_vector_point(self, value: Any, position: Optional[int] = None) -> int:
        """
        Add a new vector point to the central axis.

        Args:
            value: The vector point value to add
            position: Optional position to insert at (None = append)

        Returns:
            int: The coordinate index of the added point
        """

        if value in self.coordinate_map:
            return self.coordinate_map[value]

        if position is None:
            coordinate = len(self.vector_points)

            self.vector_points.append(value)
            self.coordinate_map[value] = coordinate

            return coordinate

        else:
            # Insert at specific position - requires shifting indices
            self.vector_points.insert(position, value)
            self.coordinate_map.clear()

            for idx, point in enumerate(self.vector_points):
                self.coordinate_map[point] = idx

            return position

    def get_coordinate(self, value: Any) -> Optional[int]:
        """Get the coordinate index for a given vector point value."""
        return self.coordinate_map.get(value)

    def get_vector_point(self, coordinate: int) -> Optional[Any]:
        """Get the vector point value at a given coordinate."""

        if 0 <= coordinate < len(self.vector_points):
            return self.vector_points[coordinate]

        return None

    def get_all_points(self) -> List[Any]:
        """Get all vector points in coordinate order (excluding deleted/None)."""
        return [vp for vp in self.vector_points if vp is not None]

    def size(self) -> int:
        """Get the number of vector points in the central axis (excluding deleted/None)."""
        return sum(1 for vp in self.vector_points if vp is not None)

    def remove_vector_point(self, value: Any) -> bool:
        """
        Remove a vector point using tombstoning (mark as None).
        No coordinate shifting - O(1) operation.

        Args:
            value: The vector point value to remove

        Returns:
            bool: True if removed, False if not found
        """
        if value not in self.coordinate_map:
            return False

        coordinate = self.coordinate_map[value]

        # Tombstone: mark as None (don't shift)
        self.vector_points[coordinate] = None

        # Remove from lookup map
        del self.coordinate_map[value]

        logger.debug(f"Tombstoned vector point '{value}' at coordinate {coordinate}")
        return True

    def shift_coordinates_after_insertion(self, coordinate_mappings, from_position: int, shift_amount: int):
        """
        Shift all coordinate mappings after insertion.
        Moved from main.py to follow DDD principles.
        """

        for mapping in coordinate_mappings.values():
            mapping.shift_coordinates(from_position, shift_amount)

    def get_vector_point_with_attributes(self, value: Any, dimensional_spaces, coordinate_mappings):
        """
        Get complete vector point with all its dimensional attributes.
        Moved from main.py to follow DDD principles.
        """

        coordinate = self.get_coordinate(value)
        if coordinate is None:
            return None

        attributes = {}

        for dimension_name in dimensional_spaces:
            if dimension_name not in coordinate_mappings:
                continue

            value_id = coordinate_mappings[dimension_name].get_mapping(coordinate)
            if value_id is None:
                continue

            result = dimensional_spaces[dimension_name].get_value(value_id)

            if result is not None:
                attributes[dimension_name] = result

        return VectorPoint(coordinate, value, attributes)

    def __repr__(self) -> str:
        return f"CentralAxis(points={len(self.vector_points)})"
