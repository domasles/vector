"""
Vector Database File Storage - single file storage system.
Handles persistence of the vector database to a single .db file.
Uses MessagePack for efficient binary serialization and file locking for multi-process safety.
"""

import logging
import msgpack
import gzip

from typing import Dict, Any, Optional
from datetime import datetime
from filelock import FileLock
from pathlib import Path

from ...meta import __version__

logger = logging.getLogger(__name__)

class VectorFileStorage:
    """Single-file storage system for Vector Database."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.lock_path = Path(str(file_path) + ".lock")

        self.metadata = {
            "version": __version__,
            "created_at": None,
            "last_modified": None,
            "total_vector_points": 0,
            "total_dimensions": 0
        }

    def save_database(self, database_data: Dict[str, Any]) -> bool:
        """
        Save the complete vector database to file.
        Uses MessagePack for efficient binary serialization and file locking for safety.

        Args:
            database_data: Complete serialized database structure

        Returns:
            bool: True if save succeeded
        """

        try:
            now = datetime.now().isoformat()

            if self.metadata["created_at"] is None:
                self.metadata["created_at"] = now

            self.metadata["last_modified"] = now

            complete_data = {
                "metadata": self.metadata,
                "database": database_data
            }

            # Convert coordinate_map to list of tuples to preserve mixed-type keys
            if "database" in complete_data and "central_axis" in complete_data["database"]:
                coord_map = complete_data["database"]["central_axis"].get("coordinate_map", {})
                complete_data["database"]["central_axis"]["coordinate_map"] = list(coord_map.items())

            msgpack_data = msgpack.packb(complete_data, use_bin_type=True)
            compressed_data = gzip.compress(msgpack_data)

            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            with FileLock(self.lock_path, timeout=10):
                with open(self.file_path, 'wb') as f:
                    f.write(compressed_data)

            logger.info(f"Vector database saved to {self.file_path} (MessagePack + gzip)")
            return True

        except Exception as e:
            logger.error(f"Failed to save vector database: {e}")
            return False

    def load_database(self) -> Optional[Dict[str, Any]]:
        """
        Load the vector database from file.
        Uses MessagePack for efficient binary deserialization and file locking for safety.

        Returns:
            Optional[Dict]: Database data if load succeeded, None otherwise
        """

        try:
            if not self.file_path.exists():
                logger.info(f"Database file {self.file_path} does not exist")
                return None

            with FileLock(self.lock_path, timeout=10):
                with open(self.file_path, 'rb') as f:
                    compressed_data = f.read()

            msgpack_data = gzip.decompress(compressed_data)
            complete_data = msgpack.unpackb(msgpack_data, raw=False, strict_map_key=False)

            self.metadata = complete_data.get("metadata", self.metadata)
            database_data = complete_data.get("database", {})

            # Convert coordinate_map back from list of tuples to dict
            if "central_axis" in database_data:
                coord_map_list = database_data["central_axis"].get("coordinate_map", [])

                if isinstance(coord_map_list, list):
                    database_data["central_axis"]["coordinate_map"] = dict(coord_map_list)

            logger.info(f"Vector database loaded from {self.file_path} (MessagePack + gzip)")
            return database_data

        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            return None

    def exists(self) -> bool:
        """Check if the database file exists."""

        return self.file_path.exists()

    def delete(self) -> bool:
        """Delete the database file."""

        try:
            if self.file_path.exists():
                self.file_path.unlink()
                logger.info(f"Deleted database file {self.file_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete database file: {e}")
            return False

    def get_file_size(self) -> int:
        """Get the size of the database file in bytes."""

        if self.file_path.exists():
            return self.file_path.stat().st_size

        return 0

    def get_metadata(self) -> Dict[str, Any]:
        """Get database metadata."""

        return self.metadata.copy()

    def update_metadata(self, updates: Dict[str, Any]):
        """Update database metadata."""

        self.metadata.update(updates)

    def serialize_database_structure(self, central_axis, dimensional_spaces, coordinate_mappings) -> Dict[str, Any]:
        """
        Serialize the complete database structure.
        Moved from main.py to follow DDD principles.
        """

        return {
            "central_axis": {
                "vector_points": central_axis.vector_points,
                "coordinate_map": central_axis.coordinate_map
            },
            "dimensional_spaces": {
                name: {
                    "value_domain": space.value_domain,
                    "value_to_id": space.value_to_id,
                    "next_id": space.next_id
                }
                for name, space in dimensional_spaces.items()
            },
            "coordinate_mappings": {
                name: mapping.coordinate_to_value_id
                for name, mapping in coordinate_mappings.items()
            }
        }

    def load_database_structure(self):
        """
        Load and return database structure.
        Moved from main.py to follow DDD principles.
        """

        return self.load_database()

    def get_database_stats(self, central_axis, dimensional_spaces) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        Moved from main.py to follow DDD principles.
        """

        return {
            "vector_points": central_axis.size(),
            "dimensions": len(dimensional_spaces),
            "dimension_details": {
                name: space.get_value_count()
                for name, space in dimensional_spaces.items()
            },
            "file_size_bytes": self.get_file_size(),
            "metadata": self.get_metadata()
        }

    def save_with_auto_metadata(self, database_data: Dict[str, Any], central_axis, dimensional_spaces) -> bool:
        """
        Save database with automatic metadata updates.
        Moved from main.py to follow DDD principles.
        """

        self.update_metadata({
            "total_vector_points": central_axis.size(),
            "total_dimensions": len(dimensional_spaces)
        })

        return self.save_database(database_data)

    def __repr__(self) -> str:
        size = self.get_file_size()
        exists = "exists" if self.exists() else "not found"

        return f"VectorFileStorage(path='{self.file_path}', {exists}, {size} bytes)"
