"""
Vector Point - A point in the central coordinate system.
Represents individual objects/entities in the vector database.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class VectorPoint:
    """
    A point in the vector coordinate system.
    Represents an individual entity with its position and attributes.
    """

    coordinate: int
    value: Any  # The actual value/identifier 
    attributes: Dict[str, Any]

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

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
