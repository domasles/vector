"""
Central Axis - The X coordinate system that serves as the primary reference point.
All other dimensional spaces radiate from this central axis like propeller blades.
"""

from typing import Dict, Any, Optional, List
from array import array

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
        self._free_slots: array = array("q")  # Compact int64 array for memory efficiency

    @property
    def free_slots(self) -> List[int]:
        """Get free_slots as list (for serialization compatibility)."""
        return self._free_slots.tolist()

    @free_slots.setter
    def free_slots(self, value: List[int]):
        """Set free_slots from list (for deserialization)."""
        self._free_slots = array("q", value)

    def add_vector_point(self, value: Any, position: Optional[int] = None) -> int:
        """
        Add a new vector point to the central axis.
        Reuses tombstoned slots when available (LIFO order).

        Args:
            value: The vector point value to add
            position: Optional position to insert at (None = append or reuse tombstone)

        Returns:
            int: The coordinate index of the added point
        """

        if value in self.coordinate_map:
            return self.coordinate_map[value]

        if position is None:
            # Check for reusable tombstoned slots first
            if self._free_slots:
                # Pop the highest free slot (LIFO - more cache friendly)
                coordinate = self._free_slots.pop()

                self.vector_points[coordinate] = value
                self.coordinate_map[value] = coordinate

                logger.debug(f"Reused tombstoned slot {coordinate} for '{value}'")
                return coordinate

            # No free slots - append to end
            coordinate = len(self.vector_points)

            self.vector_points.append(value)
            self.coordinate_map[value] = coordinate

            return coordinate

        else:
            # Insert at specific position - requires shifting indices
            self.vector_points.insert(position, value)
            self.coordinate_map.clear()

            for idx, point in enumerate(self.vector_points):
                if point is not None:
                    self.coordinate_map[point] = idx

            # Shift all free_slots that are >= position
            self._free_slots = array("q", [slot + 1 if slot >= position else slot for slot in self._free_slots])

            return position

    def get_coordinate(self, value: Any) -> Optional[int]:
        """Get the coordinate index for a given vector point value."""
        return self.coordinate_map.get(value)

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

        # Clean up trailing tombstones and update free_slots
        self._cleanup_trailing_tombstones()

        # If coordinate still exists after cleanup (wasn't a trailing tombstone), track it for reuse
        if coordinate < len(self.vector_points):
            # Insert into free_slots
            self._add_free_slot(coordinate)

        return True

    def _cleanup_trailing_tombstones(self):
        """
        Remove consecutive None values from the end of vector_points.
        Also removes those coordinates from free_slots since they no longer exist.
        """

        while self.vector_points and self.vector_points[-1] is None:
            removed_coord = len(self.vector_points) - 1
            self.vector_points.pop()

            try:
                idx = self._free_slots.index(removed_coord)
                self._free_slots.pop(idx)

            except ValueError:
                pass  # Not in array

            logger.debug(f"Cleaned up trailing tombstone at coordinate {removed_coord}")

    def _add_free_slot(self, coordinate: int):
        """Add a coordinate to _free_slots in descending order."""

        if not self._free_slots:
            self._free_slots.append(coordinate)

        else:
            inserted = False

            for i in range(len(self._free_slots)):
                if coordinate > self._free_slots[i]:
                    self._free_slots.insert(i, coordinate)
                    inserted = True
                    break

            if not inserted:
                self._free_slots.append(coordinate)

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
