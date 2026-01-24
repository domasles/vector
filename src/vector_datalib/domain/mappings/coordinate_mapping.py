"""
Coordinate Mapping - The f_axis(x_coordinate) functions that connect the central axis
to dimensional spaces. These form the "propeller blades" radiating from the central hub.
"""

from typing import Dict, Optional

import logging

logger = logging.getLogger(__name__)


class CoordinateMapping:
    """
    Represents the mapping function f_axis(x_coordinate) -> value_id.
    This is the "propeller blade" connecting central axis to a dimensional space.
    """

    def __init__(self, dimension_name: str):
        self.dimension_name = dimension_name
        self.coordinate_to_value_id: Dict[int, int] = {}
        self.ref_counts: Dict[int, int] = {}  # value_id -> reference count (O(1) lookups)

    def set_mapping(self, x_coordinate: int, value_id: int):
        """
        Set the mapping for a coordinate to a value ID.
        f_axis(x_coordinate) = value_id

        Args:
            x_coordinate: The coordinate in the central axis
            value_id: The ID of the value in the dimensional space
        """

        old_value_id = self.coordinate_to_value_id.get(x_coordinate)

        if old_value_id is not None and old_value_id != value_id:
            self.ref_counts[old_value_id] = self.ref_counts.get(old_value_id, 1) - 1

            if self.ref_counts[old_value_id] <= 0:
                del self.ref_counts[old_value_id]

        self.coordinate_to_value_id[x_coordinate] = value_id
        self.ref_counts[value_id] = self.ref_counts.get(value_id, 0) + 1

        logger.debug(f"Set mapping {self.dimension_name}[{x_coordinate}] = {value_id}")

    def get_mapping(self, x_coordinate: int) -> Optional[int]:
        """
        Get the value ID for a given coordinate.
        Returns f_axis(x_coordinate)

        Args:
            x_coordinate: The coordinate in the central axis

        Returns:
            Optional[int]: The value ID, or None if no mapping exists
        """

        return self.coordinate_to_value_id.get(x_coordinate)

    def shift_coordinates(self, from_coordinate: int, shift_amount: int):
        """
        Shift coordinate mappings after central axis insertion/deletion.
        Updates all coordinates >= from_coordinate by shift_amount.

        Args:
            from_coordinate: The coordinate position to start shifting from
            shift_amount: How much to shift (+1 for insertion, -1 for deletion)
        """

        if shift_amount == 0:
            return
        new_mapping = {}

        for coord, value_id in self.coordinate_to_value_id.items():
            if coord >= from_coordinate:
                new_coord = coord + shift_amount
                new_mapping[new_coord] = value_id

            else:
                new_mapping[coord] = value_id

        self.coordinate_to_value_id = new_mapping
        logger.debug(f"Shifted coordinates in {self.dimension_name} from {from_coordinate} by {shift_amount}")

    def count_references_to_value(self, value_id: int) -> int:
        """Count how many coordinates reference a specific value ID."""
        return self.ref_counts.get(value_id, 0)

    def remove_mapping(self, x_coordinate: int) -> bool:
        """
        Remove the mapping for a coordinate.

        Args:
            x_coordinate: The coordinate to remove the mapping for

        Returns:
            bool: True if mapping was removed, False if it didn't exist
        """

        if x_coordinate in self.coordinate_to_value_id:
            value_id = self.coordinate_to_value_id[x_coordinate]
            self.ref_counts[value_id] = self.ref_counts.get(value_id, 1) - 1

            if self.ref_counts[value_id] <= 0:
                del self.ref_counts[value_id]

            del self.coordinate_to_value_id[x_coordinate]
            logger.debug(f"Removed mapping {self.dimension_name}[{x_coordinate}]")
            return True
        return False

    def __repr__(self) -> str:
        return f"CoordinateMapping(dimension='{self.dimension_name}', mappings={len(self.coordinate_to_value_id)})"
