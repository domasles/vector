"""
Dimensional Space - Represents attribute dimensions (Y, Z, J...) with value domains.
Each space contains unique values with deduplication for memory efficiency.
"""

from typing import Dict, Any, Optional, Set

import logging

logger = logging.getLogger(__name__)

class DimensionalSpace:
    """
    A dimensional space representing an attribute axis (Y, Z, J, etc.).
    Maintains a value domain with deduplication for efficient storage.
    """

    def __init__(self, name: str):
        self.name = name
        self.value_domain: Dict[int, Any] = {}  # id -> value mapping
        self.value_to_id: Dict[Any, int] = {}   # value -> id reverse lookup
        self.next_id = 1  # Auto-incrementing ID counter

    def add_value(self, value: Any) -> int:
        """
        Add a value to the dimensional space's value domain.
        Returns existing ID if value already exists (deduplication).

        Args:
            value: The value to add to the dimension

        Returns:
            int: The unique ID for this value in the domain
        """

        if value in self.value_to_id: return self.value_to_id[value]

        # Add new value to domain
        value_id = self.next_id

        self.value_domain[value_id] = value
        self.value_to_id[value] = value_id
        self.next_id += 1

        logger.debug(f"Added value '{value}' to dimension '{self.name}' with ID {value_id}")
        return value_id

    def get_value(self, value_id: int) -> Optional[Any]:
        """Get the value for a given ID in the value domain."""
        return self.value_domain.get(value_id)

    def get_value_id(self, value: Any) -> Optional[int]:
        """Get the ID for a given value in the value domain."""
        return self.value_to_id.get(value)

    def update_value(self, old_value: Any, new_value: Any) -> bool:
        """
        Update a value in the domain. All references automatically updated.

        Args:
            old_value: The current value to replace
            new_value: The new value to set

        Returns:
            bool: True if update succeeded, False if old_value not found
        """

        value_id = self.value_to_id.get(old_value)
        if value_id is None: return False

        # Update both mappings
        del self.value_to_id[old_value]

        self.value_domain[value_id] = new_value
        self.value_to_id[new_value] = value_id

        logger.debug(f"Updated value in dimension '{self.name}': '{old_value}' -> '{new_value}'")
        return True

    def get_value_count(self) -> int:
        """Get the number of unique values in this dimensional space."""
        return len(self.value_domain)

    def remove_value_if_unused(self, value_id: int) -> bool:
        """
        Safely remove a value from the domain if it's not referenced.
        WARNING: Only call this after verifying no coordinates reference this value_id!

        Args:
            value_id: The value ID to remove

        Returns:
            bool: True if value was removed, False if it didn't exist
        """

        if value_id in self.value_domain:
            old_value = self.value_domain[value_id]
            del self.value_domain[value_id]

            if old_value in self.value_to_id:
                del self.value_to_id[old_value]

            logger.debug(f"Removed unused value '{old_value}' (ID {value_id}) from dimension '{self.name}'")
            return True

        return False

    def __repr__(self) -> str:
        return f"DimensionalSpace(name='{self.name}', values={self.get_value_count()})"
