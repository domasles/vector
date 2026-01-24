"""
Vector Point - A point in the central coordinate system.
Represents individual objects/entities in the vector database.
Uses __slots__ for memory efficiency.
"""

from typing import Any, Dict, Optional


class VectorPoint:
    """
    A point in the vector coordinate system.
    Represents an individual entity with its position and attributes.
    """

    __slots__ = ("coordinate", "value", "attributes")

    def __init__(self, coordinate: int, value: Any, attributes: Dict[str, Any] = None):
        self.coordinate = coordinate
        self.value = value
        self.attributes = attributes if attributes is not None else {}

    def get_attribute(self, dimension_name: str) -> Optional[Any]:
        """Get the value for a specific dimensional attribute."""
        return self.attributes.get(dimension_name)

    def set_attribute(self, dimension_name: str, value: Any):
        """Set the value for a specific dimensional attribute."""
        self.attributes[dimension_name] = value

    def has_attribute(self, dimension_name: str) -> bool:
        """Check if this vector point has a value for the given dimension."""
        return dimension_name in self.attributes

    def get_all_attributes(self) -> Dict[str, Any]:
        """Get all dimensional attributes for this vector point."""
        return self.attributes.copy()

    def __repr__(self) -> str:
        return f"VectorPoint(coord={self.coordinate}, value={self.value}, attrs={len(self.attributes)})"
