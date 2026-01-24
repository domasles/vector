"""
Dimensional Space - Represents attribute dimensions (Y, Z, J...) with value domains.
Each space contains unique values with deduplication for memory efficiency.
"""

from typing import Any, Optional, Dict

import logging

from bidict import bidict

logger = logging.getLogger(__name__)


class DimensionalSpace:
    """
    A dimensional space representing an attribute axis (Y, Z, J, etc.).
    Maintains a value domain with deduplication for efficient storage.
    """

    def __init__(self, name: str):
        self.name = name
        self._values: bidict[int, Any] = bidict()
        self.next_id = 1  # Auto-incrementing ID counter

    @property
    def value_domain(self) -> Dict[int, Any]:
        """Forward lookup (id → value). Returns dict for serialization compatibility."""
        return dict(self._values)

    @value_domain.setter
    def value_domain(self, data: Dict[int, Any]):
        """Set values from a dict (used during deserialization)."""
        self._values = bidict(data)

    def add_value(self, value: Any) -> int:
        """
        Add a value to the dimensional space's value domain.
        Returns existing ID if value already exists (deduplication).

        Args:
            value: The value to add to the dimension

        Returns:
            int: The unique ID for this value in the domain
        """
        # Check if value already exists (O(1) reverse lookup)
        if value in self._values.inverse:
            return self._values.inverse[value]

        # Add new value to domain
        value_id = self.next_id
        self._values[value_id] = value
        self.next_id += 1

        logger.debug(f"Added value '{value}' to dimension '{self.name}' with ID {value_id}")
        return value_id

    def get_value(self, value_id: int) -> Optional[Any]:
        """Get the value for a given ID in the value domain."""
        return self._values.get(value_id)

    def get_value_id(self, value: Any) -> Optional[int]:
        """Get the ID for a given value in the value domain."""
        return self._values.inverse.get(value)

    def get_value_count(self) -> int:
        """Get the number of unique values in this dimensional space."""
        return len(self._values)

    def remove_value_if_unused(self, value_id: int) -> bool:
        """
        Safely remove a value from the domain if it's not referenced.
        WARNING: Only call this after verifying no coordinates reference this value_id!

        Args:
            value_id: The value ID to remove

        Returns:
            bool: True if value was removed, False if it didn't exist
        """
        if value_id in self._values:
            old_value = self._values[value_id]
            del self._values[value_id]  # bidict handles both directions

            logger.debug(f"Removed unused value '{old_value}' (ID {value_id}) from dimension '{self.name}'")
            return True

        return False

    def __repr__(self) -> str:
        return f"DimensionalSpace(name='{self.name}', values={self.get_value_count()})"
