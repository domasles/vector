"""
Central Axis - The X coordinate system that serves as the primary reference point.
All other dimensional spaces radiate from this central axis like propeller blades.
"""

from typing import Dict, Any, Optional, List

import logging

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
        """Get all vector points in coordinate order."""

        return self.vector_points.copy()

    def size(self) -> int:
        """Get the number of vector points in the central axis."""

        return len(self.vector_points)

    def __repr__(self) -> str:
        return f"CentralAxis(points={len(self.vector_points)})"
